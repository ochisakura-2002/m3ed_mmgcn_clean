"""Synthetic forward/loss/backward audit for project causal GS-MCC-inspired."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any, Dict

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.baselines.gsmcc import (  # noqa: E402
    CausalGSMCCConfig,
    CausalGSMCCInspiredBaseline,
    compute_causal_gsmcc_loss,
)
from utils.io import load_yaml  # noqa: E402


def _config_from_yaml(raw: Dict[str, Any]) -> CausalGSMCCConfig:
    if "model" not in raw or "loss" not in raw:
        raise ValueError("config must contain explicit model and loss mappings")
    model = dict(raw["model"])
    expected_name = model.pop("name", None)
    if expected_name != "CausalGSMCCInspiredBaseline":
        raise ValueError("unexpected model.name")
    overlap = set(model).intersection(raw["loss"])
    if overlap:
        raise ValueError(f"duplicate model/loss keys: {sorted(overlap)}")
    return CausalGSMCCConfig(**model, **dict(raw["loss"]))


def _synthetic_batch(
    raw: Dict[str, Any],
    config: CausalGSMCCConfig,
) -> Dict[str, torch.Tensor]:
    synthetic = raw["synthetic"]
    device = torch.device(synthetic["device"])
    batch_size = int(synthetic["batch_size"])
    seq_len = int(synthetic["seq_len"])
    lengths = torch.tensor(synthetic["lengths"], dtype=torch.long, device=device)
    if batch_size != 2 or seq_len < 5 or lengths.shape != (batch_size,):
        raise ValueError("synthetic audit requires B=2, T>=5, and two lengths")
    if lengths[0] == lengths[1]:
        raise ValueError("the two synthetic dialogues must have different lengths")
    time = torch.arange(seq_len, device=device).unsqueeze(0)
    attention_mask = time < lengths.unsqueeze(1)
    speakers = time.expand(batch_size, -1) % int(synthetic["num_speakers"])
    labels = torch.randint(
        0,
        config.num_classes,
        (batch_size, seq_len),
        device=device,
    ).masked_fill(~attention_mask, -100)
    return {
        "text_features": torch.randn(batch_size, seq_len, config.text_dim, device=device),
        "audio_features": torch.randn(batch_size, seq_len, config.audio_dim, device=device),
        "visual_features": torch.randn(batch_size, seq_len, config.visual_dim, device=device),
        "attention_mask": attention_mask,
        "lengths": lengths,
        "speaker_ids_int": speakers.long(),
        "labels": labels,
    }


def _assert_finite_gradients(model: torch.nn.Module) -> None:
    missing = []
    nonfinite = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.grad is None:
            missing.append(name)
        elif not torch.isfinite(parameter.grad).all():
            nonfinite.append(name)
    if missing or nonfinite:
        raise AssertionError(f"gradient audit failed: missing={missing}, nonfinite={nonfinite}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    raw = load_yaml(args.config)
    torch.manual_seed(int(raw["seed"]))
    config = _config_from_yaml(raw)
    batch = _synthetic_batch(raw, config)
    model = CausalGSMCCInspiredBaseline(config).to(batch["text_features"].device)
    model.train()
    auxiliary = model(
        text_features=batch["text_features"],
        audio_features=batch["audio_features"],
        visual_features=batch["visual_features"],
        attention_mask=batch["attention_mask"],
        lengths=batch["lengths"],
        speaker_ids_int=batch["speaker_ids_int"],
        return_aux=True,
    )
    expected_shape = (
        int(raw["synthetic"]["batch_size"]),
        int(raw["synthetic"]["seq_len"]),
        config.num_classes,
    )
    if auxiliary["logits"].shape != expected_shape:
        raise AssertionError(f"unexpected logits shape: {auxiliary['logits'].shape}")
    losses = compute_causal_gsmcc_loss(
        auxiliary["logits"],
        batch["labels"],
        batch["attention_mask"],
        auxiliary["low_frequency_modal_repr"],
        auxiliary["high_frequency_modal_repr"],
        classification_weight=config.classification_weight,
        consistency_weight=config.consistency_weight,
        complementarity_weight=config.complementarity_weight,
    )
    changed_labels = batch["labels"].clone()
    changed_labels[~batch["attention_mask"]] = 0
    changed_losses = compute_causal_gsmcc_loss(
        auxiliary["logits"],
        changed_labels,
        batch["attention_mask"],
        auxiliary["low_frequency_modal_repr"],
        auxiliary["high_frequency_modal_repr"],
        classification_weight=config.classification_weight,
        consistency_weight=config.consistency_weight,
        complementarity_weight=config.complementarity_weight,
    )
    torch.testing.assert_close(losses["total_loss"], changed_losses["total_loss"])
    losses["total_loss"].backward()
    _assert_finite_gradients(model)

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(f"logits_shape={tuple(auxiliary['logits'].shape)}")
    print(f"low_shape={tuple(auxiliary['low_frequency_repr'].shape)}")
    print(f"high_shape={tuple(auxiliary['high_frequency_repr'].shape)}")
    print(f"parameter_count={parameter_count}")
    print(f"edge_count={int(auxiliary['adjacency'].sum().item())}")
    for name, value in losses.items():
        print(f"{name}={float(value.detach()):.8f}")
    print("gradient_audit=PASS")
    print("padding_loss_exclusion=PASS")


if __name__ == "__main__":
    main()

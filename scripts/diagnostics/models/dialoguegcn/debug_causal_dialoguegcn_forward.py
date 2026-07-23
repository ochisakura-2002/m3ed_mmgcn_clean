"""Synthetic forward/loss/backward audit for project causal DialogueGCN."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any, Dict

import torch
import torch.nn.functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.dialoguegcn.unified import (  # noqa: E402
    CausalDialogueGCNBaseline,
    CausalDialogueGCNConfig,
)
from utils.io import load_yaml  # noqa: E402


def _config_from_yaml(raw: Dict[str, Any]) -> CausalDialogueGCNConfig:
    if "model" not in raw:
        raise ValueError("config must contain an explicit model mapping")
    model = dict(raw["model"])
    expected_name = model.pop("name", None)
    if expected_name != "CausalDialogueGCNBaseline":
        raise ValueError("unexpected model.name")
    return CausalDialogueGCNConfig(**model)


def _synthetic_batch(
    raw: Dict[str, Any],
    config: CausalDialogueGCNConfig,
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
    speakers = time.expand(batch_size, -1) % config.num_speakers
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


def _masked_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    valid = attention_mask.to(dtype=torch.bool) & (labels >= 0)
    return F.cross_entropy(logits[valid], labels[valid].long())


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
    model = CausalDialogueGCNBaseline(config).to(batch["text_features"].device)
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
    loss = _masked_cross_entropy(
        auxiliary["logits"],
        batch["labels"],
        batch["attention_mask"],
    )
    changed_logits = auxiliary["logits"].clone()
    changed_logits[~batch["attention_mask"]] = 1e4
    changed_loss = _masked_cross_entropy(
        changed_logits,
        batch["labels"],
        batch["attention_mask"],
    )
    torch.testing.assert_close(loss, changed_loss)
    row_sums = auxiliary["edge_attention"].sum(dim=-1)
    torch.testing.assert_close(
        row_sums[batch["attention_mask"]],
        torch.ones_like(row_sums[batch["attention_mask"]]),
        atol=1e-6,
        rtol=0,
    )
    if torch.count_nonzero(row_sums[~batch["attention_mask"]]).item() != 0:
        raise AssertionError("padding targets received edge attention")
    loss.backward()
    _assert_finite_gradients(model)

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    active_relations = auxiliary["relation_ids"][auxiliary["adjacency"]]
    print(f"logits_shape={tuple(auxiliary['logits'].shape)}")
    print(f"context_shape={tuple(auxiliary['context_repr'].shape)}")
    print(f"graph_shape={tuple(auxiliary['graph_repr'].shape)}")
    print(f"parameter_count={parameter_count}")
    print(f"edge_count={int(auxiliary['adjacency'].sum().item())}")
    print(f"active_relation_count={int(torch.unique(active_relations).numel())}")
    print(f"relation_mapping={auxiliary['relation_mapping']}")
    print(f"classification_loss={float(loss.detach()):.8f}")
    print("edge_attention_normalization=PASS")
    print("gradient_audit=PASS")
    print("padding_loss_exclusion=PASS")


if __name__ == "__main__":
    main()

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from models.baselines.causal_graph_common import assert_no_future_edges
from models.baselines.gsmcc import (
    CausalGSMCCConfig,
    CausalGSMCCInspiredBaseline,
    compute_causal_gsmcc_loss,
)


TARGET = 2
FEATURE_KEYS = ("text_features", "audio_features", "visual_features")


def _make_batch(require_grad: bool = False) -> Dict[str, torch.Tensor]:
    torch.manual_seed(1103)
    lengths = torch.tensor([6, 5])
    mask = torch.arange(6).unsqueeze(0) < lengths.unsqueeze(1)
    batch = {
        "text_features": torch.randn(2, 6, 4),
        "audio_features": torch.randn(2, 6, 3),
        "visual_features": torch.randn(2, 6, 2),
        "attention_mask": mask,
        "lengths": lengths,
        "speaker_ids_int": (torch.arange(6).unsqueeze(0).expand(2, -1) % 2).long(),
        "labels": torch.randint(0, 6, (2, 6)).masked_fill(~mask, -100),
    }
    if require_grad:
        for key in FEATURE_KEYS:
            batch[key].requires_grad_(True)
    return batch


def _make_model() -> CausalGSMCCInspiredBaseline:
    torch.manual_seed(1201)
    return CausalGSMCCInspiredBaseline(
        CausalGSMCCConfig(
            text_dim=4,
            audio_dim=3,
            visual_dim=2,
            hidden_dim=8,
            num_classes=6,
            dropout=0.0,
            window_past=None,
            num_filter_steps=3,
            num_graph_layers=2,
            modality_encoder_type="causal_gru",
            fusion_type="concat",
            context_mode="causal",
        )
    ).eval()


def _forward(
    model: CausalGSMCCInspiredBaseline,
    batch: Dict[str, torch.Tensor],
    return_aux: bool = False,
):
    return model(
        text_features=batch["text_features"],
        audio_features=batch["audio_features"],
        visual_features=batch["visual_features"],
        attention_mask=batch["attention_mask"],
        lengths=batch["lengths"],
        speaker_ids_int=batch["speaker_ids_int"],
        return_aux=return_aux,
    )


def _history_mask(batch: Dict[str, torch.Tensor]) -> torch.Tensor:
    return batch["attention_mask"] & (torch.arange(6).unsqueeze(0) <= TARGET)


def _future_mask(batch: Dict[str, torch.Tensor]) -> torch.Tensor:
    return batch["attention_mask"] & (torch.arange(6).unsqueeze(0) > TARGET)


def _assert_causal_close(actual: torch.Tensor, expected: torch.Tensor) -> None:
    max_diff = float((actual - expected).abs().max())
    assert max_diff <= 1e-6, f"max_diff={max_diff}; pass_at_1e-5={max_diff <= 1e-5}"


def _perturb(
    batch: Dict[str, torch.Tensor],
    keys: tuple[str, ...],
    mode: str = "random",
) -> Dict[str, torch.Tensor]:
    changed = {key: value.clone() for key, value in batch.items()}
    future = _future_mask(batch)
    torch.manual_seed(1301)
    for key in keys:
        if mode == "zero":
            replacement = torch.zeros_like(changed[key])
        elif mode == "random":
            replacement = torch.randn_like(changed[key]) * 7.0
        elif mode == "shuffle":
            replacement = changed[key].flip(0)
        else:
            raise ValueError(f"unknown perturbation mode: {mode}")
        changed[key][future] = replacement[future]
    return changed


def test_future_text_audio_visual_and_joint_perturbations_preserve_history() -> None:
    model = _make_model()
    batch = _make_batch()
    with torch.no_grad():
        original = _forward(model, batch)
        for key in FEATURE_KEYS:
            for mode in ("zero", "random", "shuffle"):
                changed = _forward(model, _perturb(batch, (key,), mode))
                _assert_causal_close(changed[_history_mask(batch)], original[_history_mask(batch)])
        changed = _forward(model, _perturb(batch, FEATURE_KEYS, "random"))
        _assert_causal_close(changed[_history_mask(batch)], original[_history_mask(batch)])


def test_low_and_high_frequency_branches_are_strictly_causal() -> None:
    model = _make_model()
    batch = _make_batch()
    with torch.no_grad():
        original = _forward(model, batch, return_aux=True)
        changed = _forward(model, _perturb(batch, FEATURE_KEYS), return_aux=True)
    history = _history_mask(batch)
    _assert_causal_close(changed["low_frequency_repr"][history], original["low_frequency_repr"][history])
    _assert_causal_close(changed["high_frequency_repr"][history], original["high_frequency_repr"][history])


def test_prefix_and_full_current_outputs_are_equivalent() -> None:
    model = _make_model()
    batch = _make_batch()
    prefix = {
        key: value[:, : TARGET + 1].clone() if value.dim() >= 2 else value.clamp_max(TARGET + 1)
        for key, value in batch.items()
    }
    with torch.no_grad():
        full = _forward(model, batch, return_aux=True)
        short = _forward(model, prefix, return_aux=True)
    _assert_causal_close(full["logits"][:, TARGET], short["logits"][:, -1])
    _assert_causal_close(
        full["low_frequency_repr"][:, TARGET],
        short["low_frequency_repr"][:, -1],
    )
    _assert_causal_close(
        full["high_frequency_repr"][:, TARGET],
        short["high_frequency_repr"][:, -1],
    )


def test_current_logit_has_zero_future_text_audio_visual_gradient() -> None:
    model = _make_model()
    batch = _make_batch(require_grad=True)
    scalar = _forward(model, batch)[:, TARGET, 0].sum()
    gradients = torch.autograd.grad(scalar, [batch[key] for key in FEATURE_KEYS])
    future = _future_mask(batch)
    for gradient in gradients:
        assert torch.count_nonzero(gradient[future]).item() == 0


def test_multilayer_graph_structure_and_normalization_remain_causal() -> None:
    model = _make_model()
    batch = _make_batch()
    prefix = {
        key: value[:, : TARGET + 1].clone() if value.dim() >= 2 else value.clamp_max(TARGET + 1)
        for key, value in batch.items()
    }
    with torch.no_grad():
        full = _forward(model, batch, return_aux=True)
        short = _forward(model, prefix, return_aux=True)
    node_time = full["causal_diagnostics"]["node_time"]
    assert_no_future_edges(full["adjacency"], node_time)
    assert_no_future_edges(full["normalized_adjacency"], node_time)
    prefix_nodes = (TARGET + 1) * 3
    _assert_causal_close(
        full["normalized_adjacency"][:, :prefix_nodes, :prefix_nodes],
        short["normalized_adjacency"],
    )
    # Same-time cross-modal source is legal.
    assert bool(full["adjacency"][0, TARGET * 3, TARGET * 3 + 2])


def test_auxiliary_losses_do_not_read_future_labels() -> None:
    model = _make_model()
    batch = _make_batch()
    auxiliary = _forward(model, batch, return_aux=True)
    original = compute_causal_gsmcc_loss(
        auxiliary["logits"],
        batch["labels"],
        batch["attention_mask"],
        auxiliary["low_frequency_modal_repr"],
        auxiliary["high_frequency_modal_repr"],
    )
    changed_labels = batch["labels"].clone()
    changed_labels[_future_mask(batch)] = (changed_labels[_future_mask(batch)] + 1) % 6
    changed = compute_causal_gsmcc_loss(
        auxiliary["logits"],
        changed_labels,
        batch["attention_mask"],
        auxiliary["low_frequency_modal_repr"],
        auxiliary["high_frequency_modal_repr"],
    )
    torch.testing.assert_close(original["consistency_loss"], changed["consistency_loss"])
    torch.testing.assert_close(original["complementarity_loss"], changed["complementarity_loss"])


def test_eval_is_reproducible_and_has_no_bidirectional_encoder() -> None:
    model = _make_model()
    batch = _make_batch()
    with torch.no_grad():
        first = _forward(model, batch)
        second = _forward(model, batch)
    torch.testing.assert_close(first, second, atol=0, rtol=0)
    recurrent = [module for module in model.modules() if isinstance(module, (nn.GRU, nn.LSTM, nn.RNN))]
    assert recurrent
    assert all(not module.bidirectional for module in recurrent)


def test_causal_config_rejects_future_and_bidirectional_requests() -> None:
    base = dict(text_dim=4, audio_dim=3, visual_dim=2, hidden_dim=8, num_classes=6)
    for invalid in (
        {"context_mode": "full"},
        {"window_future": 1},
        {"bidirectional": True},
        {"nodal_attention": "full"},
    ):
        try:
            CausalGSMCCConfig(**base, **invalid)
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid config was accepted: {invalid}")

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from models.baselines.causal_graph_common import assert_no_future_edges
from models.baselines.dialoguegcn import (
    CausalDialogueGCNBaseline,
    CausalDialogueGCNConfig,
)


TARGET = 2
FEATURE_KEYS = ("text_features", "audio_features", "visual_features")


def _make_batch(require_grad: bool = False) -> Dict[str, torch.Tensor]:
    torch.manual_seed(2101)
    lengths = torch.tensor([6, 5])
    mask = torch.arange(6).unsqueeze(0) < lengths.unsqueeze(1)
    batch = {
        "text_features": torch.randn(2, 6, 4),
        "audio_features": torch.randn(2, 6, 3),
        "visual_features": torch.randn(2, 6, 2),
        "attention_mask": mask,
        "lengths": lengths,
        "speaker_ids_int": (torch.arange(6).unsqueeze(0).expand(2, -1) % 2).long(),
    }
    if require_grad:
        for key in FEATURE_KEYS:
            batch[key].requires_grad_(True)
    return batch


def _make_model() -> CausalDialogueGCNBaseline:
    torch.manual_seed(2203)
    return CausalDialogueGCNBaseline(
        CausalDialogueGCNConfig(
            text_dim=4,
            audio_dim=3,
            visual_dim=2,
            hidden_dim=8,
            context_hidden_dim=8,
            graph_hidden_dim=8,
            num_classes=6,
            dropout=0.0,
            window_past=None,
            context_encoder_type="causal_gru",
            num_graph_layers=2,
            num_speakers=2,
            context_mode="causal",
        )
    ).eval()


def _forward(
    model: CausalDialogueGCNBaseline,
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
    torch.manual_seed(2309)
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
    _assert_causal_close(full["context_repr"][:, TARGET], short["context_repr"][:, -1])
    _assert_causal_close(full["graph_repr"][:, TARGET], short["graph_repr"][:, -1])


def test_current_logit_has_zero_future_text_audio_visual_gradient() -> None:
    model = _make_model()
    batch = _make_batch(require_grad=True)
    scalar = _forward(model, batch)[:, TARGET, 0].sum()
    gradients = torch.autograd.grad(scalar, [batch[key] for key in FEATURE_KEYS])
    future = _future_mask(batch)
    for gradient in gradients:
        assert torch.count_nonzero(gradient[future]).item() == 0


def test_relation_ids_and_graph_direction_match_source_target_speakers() -> None:
    model = _make_model()
    batch = _make_batch()
    auxiliary = _forward(model, batch, return_aux=True)
    assert_no_future_edges(auxiliary["adjacency"], torch.arange(6))
    # target t=1 speaker 1, source t=0 speaker 0 -> relation 0*2+1.
    assert int(auxiliary["relation_ids"][0, 1, 0]) == 1
    assert auxiliary["relation_mapping"][1] == "source_0->target_1"
    assert torch.all(auxiliary["relation_ids"][~auxiliary["adjacency"]] == -1)


def test_edge_softmax_normalizes_only_legal_history_neighbors() -> None:
    model = _make_model()
    batch = _make_batch()
    auxiliary = _forward(model, batch, return_aux=True)
    weights = auxiliary["edge_attention"]
    adjacency = auxiliary["adjacency"]
    assert torch.count_nonzero(weights[~adjacency]).item() == 0
    row_sums = weights.sum(-1)
    torch.testing.assert_close(
        row_sums[batch["attention_mask"]],
        torch.ones_like(row_sums[batch["attention_mask"]]),
        atol=1e-6,
        rtol=0,
    )
    assert torch.count_nonzero(row_sums[~batch["attention_mask"]]).item() == 0


def test_multilayer_relational_graph_is_causal_and_gru_is_forward_only() -> None:
    model = _make_model()
    assert len(model.graph_layers) == 2
    recurrent = [module for module in model.modules() if isinstance(module, (nn.GRU, nn.LSTM, nn.RNN))]
    assert recurrent
    assert all(not module.bidirectional for module in recurrent)
    batch = _make_batch()
    with torch.no_grad():
        original = _forward(model, batch)
        changed = _forward(model, _perturb(batch, FEATURE_KEYS))
    _assert_causal_close(changed[_history_mask(batch)], original[_history_mask(batch)])


def test_eval_is_reproducible() -> None:
    model = _make_model()
    batch = _make_batch()
    with torch.no_grad():
        first = _forward(model, batch)
        second = _forward(model, batch)
    torch.testing.assert_close(first, second, atol=0, rtol=0)


def test_causal_config_rejects_future_bidirectional_and_full_attention() -> None:
    base = dict(
        text_dim=4,
        audio_dim=3,
        visual_dim=2,
        hidden_dim=8,
        context_hidden_dim=8,
        graph_hidden_dim=8,
        num_classes=6,
    )
    for invalid in (
        {"context_mode": "full"},
        {"window_future": 1},
        {"bidirectional": True},
        {"nodal_attention": "full"},
    ):
        try:
            CausalDialogueGCNConfig(**base, **invalid)
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid config was accepted: {invalid}")

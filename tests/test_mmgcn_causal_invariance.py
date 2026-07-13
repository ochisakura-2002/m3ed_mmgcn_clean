from __future__ import annotations

import torch
import torch.nn as nn

from models.baselines.mmgcn.mm_gcn import M3EDMMGCN


TARGET = 2


def make_batch() -> dict[str, torch.Tensor]:
    torch.manual_seed(17)
    batch_size, seq_len = 2, 5
    lengths = torch.tensor([5, 4])
    mask = torch.arange(seq_len).unsqueeze(0) < lengths.unsqueeze(1)
    return {
        "text_features": torch.randn(batch_size, seq_len, 4),
        "audio_features": torch.randn(batch_size, seq_len, 3),
        "visual_features": torch.randn(batch_size, seq_len, 2),
        "lengths": lengths,
        "attention_mask": mask,
        "speaker_ids_int": torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1) % 2,
    }


def make_model(context_mode: str = "causal") -> M3EDMMGCN:
    torch.manual_seed(23)
    model = M3EDMMGCN(
        text_dim=4,
        audio_dim=3,
        visual_dim=2,
        hidden_dim=8,
        num_classes=6,
        num_layers=2,
        dropout=0.0,
        context_mode=context_mode,
        window_past=None,
        window_future=0 if context_mode == "causal" else None,
    )
    return model.eval()


def forward(model: M3EDMMGCN, batch: dict[str, torch.Tensor], return_graph: bool = False):
    return model(
        text_features=batch["text_features"],
        audio_features=batch["audio_features"],
        visual_features=batch["visual_features"],
        lengths=batch["lengths"],
        attention_mask=batch["attention_mask"],
        speaker_ids_int=batch["speaker_ids_int"],
        return_graph=return_graph,
    )


def perturb_future(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    changed = {key: value.clone() for key, value in batch.items()}
    future = batch["attention_mask"] & (
        torch.arange(batch["attention_mask"].shape[1]).unsqueeze(0) > TARGET
    )
    torch.manual_seed(29)
    for key in ("text_features", "audio_features", "visual_features"):
        replacement = torch.randn_like(changed[key])
        changed[key][future] = replacement[future]
    return changed


def test_future_features_do_not_change_history_logits() -> None:
    model = make_model()
    batch = make_batch()
    changed = perturb_future(batch)
    with torch.no_grad():
        original_logits = forward(model, batch)["logits"]
        changed_logits = forward(model, changed)["logits"]
    history = batch["attention_mask"] & (
        torch.arange(batch["attention_mask"].shape[1]).unsqueeze(0) <= TARGET
    )
    torch.testing.assert_close(original_logits[history], changed_logits[history], atol=1e-6, rtol=0)


def test_prefix_and_full_current_logits_are_equivalent() -> None:
    model = make_model()
    batch = make_batch()
    prefix = {
        key: value[:, : TARGET + 1].clone() if value.dim() >= 2 else value.clone()
        for key, value in batch.items()
    }
    prefix["lengths"] = batch["lengths"].clamp_max(TARGET + 1)
    with torch.no_grad():
        full_logits = forward(model, batch)["logits"][:, TARGET]
        prefix_logits = forward(model, prefix)["logits"][:, -1]
    torch.testing.assert_close(full_logits, prefix_logits, atol=1e-6, rtol=0)


def test_current_logit_has_zero_future_feature_gradient() -> None:
    model = make_model()
    batch = make_batch()
    inputs = []
    for key in ("text_features", "audio_features", "visual_features"):
        batch[key] = batch[key].requires_grad_(True)
        inputs.append(batch[key])
    scalar = forward(model, batch)["logits"][:, TARGET, 0].sum()
    gradients = torch.autograd.grad(scalar, inputs)
    future = batch["attention_mask"] & (
        torch.arange(batch["attention_mask"].shape[1]).unsqueeze(0) > TARGET
    )
    for gradient in gradients:
        assert torch.count_nonzero(gradient[future]).item() == 0


def test_causal_adjacency_has_no_future_or_cross_dialogue_edges() -> None:
    batch = make_batch()
    adjacency = forward(make_model(), batch, return_graph=True)["adjacency"]
    positions = []
    for batch_index, length in enumerate(batch["lengths"].tolist()):
        positions.extend((batch_index, time_index) for time_index in range(length))
    node_positions = positions * 3
    for target_node, source_node in (adjacency != 0).nonzero().tolist():
        target_batch, target_time = node_positions[target_node]
        source_batch, source_time = node_positions[source_node]
        assert source_batch == target_batch
        assert source_time <= target_time


def test_causal_path_contains_no_bidirectional_recurrent_encoder() -> None:
    recurrent = tuple(
        module
        for module in make_model().modules()
        if isinstance(module, (nn.RNN, nn.GRU, nn.LSTM))
    )
    assert all(not module.bidirectional for module in recurrent)


def test_full_context_negative_control_is_not_strict_causal() -> None:
    model = make_model(context_mode="full")
    batch = make_batch()
    changed = perturb_future(batch)
    with torch.no_grad():
        original = forward(model, batch)["logits"][:, TARGET]
        perturbed = forward(model, changed)["logits"][:, TARGET]
    assert float((original - perturbed).abs().max()) > 1e-6


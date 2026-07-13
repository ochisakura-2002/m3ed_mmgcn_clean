from __future__ import annotations

import torch
import torch.nn as nn

from models.baselines.multidag_cl import MultiDAGCLBaseline


TARGET = 2


def make_batch() -> dict[str, torch.Tensor]:
    torch.manual_seed(41)
    batch_size, seq_len = 2, 5
    lengths = torch.tensor([5, 4])
    mask = torch.arange(seq_len).unsqueeze(0) < lengths.unsqueeze(1)
    return {
        "text_features": torch.randn(batch_size, seq_len, 4),
        "audio_features": torch.randn(batch_size, seq_len, 3),
        "visual_features": torch.randn(batch_size, seq_len, 2),
        "attention_mask": mask,
        "speaker_ids_int": torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1) % 2,
    }


def make_model(encoder_type: str = "causal_gru") -> MultiDAGCLBaseline:
    torch.manual_seed(43)
    return MultiDAGCLBaseline(
        text_dim=4,
        audio_dim=3,
        visual_dim=2,
        hidden_dim=8,
        num_classes=6,
        window_past=2,
        dropout=0.0,
        num_graph_layers=2,
        modality_encoder_type=encoder_type,
    ).eval()


def forward(model: MultiDAGCLBaseline, batch: dict[str, torch.Tensor]):
    return model(
        text_features=batch["text_features"],
        audio_features=batch["audio_features"],
        visual_features=batch["visual_features"],
        attention_mask=batch["attention_mask"],
        speaker_ids_int=batch["speaker_ids_int"],
    )


def perturb_future(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    changed = {key: value.clone() for key, value in batch.items()}
    future = batch["attention_mask"] & (
        torch.arange(batch["attention_mask"].shape[1]).unsqueeze(0) > TARGET
    )
    torch.manual_seed(47)
    for key in ("text_features", "audio_features", "visual_features"):
        replacement = torch.randn_like(changed[key])
        changed[key][future] = replacement[future]
    return changed


def test_future_features_do_not_change_history_logits() -> None:
    model = make_model()
    batch = make_batch()
    with torch.no_grad():
        original = forward(model, batch)["logits"]
        perturbed = forward(model, perturb_future(batch))["logits"]
    history = batch["attention_mask"] & (
        torch.arange(batch["attention_mask"].shape[1]).unsqueeze(0) <= TARGET
    )
    torch.testing.assert_close(original[history], perturbed[history], atol=1e-6, rtol=0)


def test_prefix_and_full_current_logits_are_equivalent() -> None:
    model = make_model()
    batch = make_batch()
    prefix = {key: value[:, : TARGET + 1].clone() for key, value in batch.items()}
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


def test_causal_adjacency_has_no_future_edges() -> None:
    adjacency = forward(make_model(), make_batch())["adjacency"]
    for _, target_index, source_index in (adjacency != 0).nonzero().tolist():
        assert source_index <= target_index


def test_causal_configuration_has_unidirectional_encoder() -> None:
    model = make_model()
    recurrent = [
        module
        for module in model.modules()
        if isinstance(module, (nn.RNN, nn.GRU, nn.LSTM))
    ]
    assert recurrent
    assert all(not module.bidirectional for module in recurrent)
    assert model.modality_encoder_type == "causal_gru"


def test_linear_legacy_branch_is_also_positionwise_causal() -> None:
    model = make_model(encoder_type="linear")
    batch = make_batch()
    with torch.no_grad():
        original = forward(model, batch)["logits"]
        perturbed = forward(model, perturb_future(batch))["logits"]
    history = batch["attention_mask"] & (
        torch.arange(batch["attention_mask"].shape[1]).unsqueeze(0) <= TARGET
    )
    torch.testing.assert_close(original[history], perturbed[history], atol=1e-6, rtol=0)

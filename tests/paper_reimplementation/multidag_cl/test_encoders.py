from __future__ import annotations

import torch

from models.multidag_cl.paper_reimplementation.encoders import (
    OfficialSourceInputAdapter,
    OfficialSourceSingleProjectionEncoder,
    PaperFormulaModalityEncoder,
)

from _helpers import make_batch, make_config


def test_paper_encoder_uses_named_avt_order_dimensions_and_exact_padding() -> None:
    torch.manual_seed(3)
    encoder = PaperFormulaModalityEncoder(make_config()).eval()
    batch = make_batch()
    encoded = encoder(
        text_features=batch["text_features"],
        audio_features=batch["audio_features"],
        visual_features=batch["visual_features"],
        lengths=batch["lengths"],
        attention_mask=batch["attention_mask"],
    )
    assert encoded.audio.shape == (2, 4, 100)
    assert encoded.visual.shape == (2, 4, 100)
    assert encoded.text.shape == (2, 4, 100)
    assert encoded.fused.shape == (2, 4, 300)
    assert encoded.modality_order == ("audio", "visual", "text")
    torch.testing.assert_close(encoded.fused[..., :100], encoded.audio)
    torch.testing.assert_close(encoded.fused[..., 100:200], encoded.visual)
    torch.testing.assert_close(encoded.fused[..., 200:], encoded.text)
    assert torch.count_nonzero(encoded.fused[1, 3]).item() == 0


def test_source_adapter_packs_exact_tav_order_and_projection_masks_bias() -> None:
    config = make_config(profile="official_source_behavior")
    adapter = OfficialSourceInputAdapter(config)
    encoder = OfficialSourceSingleProjectionEncoder(config).eval()
    text = torch.tensor([[[1.0, 2.0, 3.0, 4.0], [0.0, 0.0, 0.0, 0.0]]])
    audio = torch.tensor([[[5.0, 6.0, 7.0], [0.0, 0.0, 0.0]]])
    visual = torch.tensor([[[8.0, 9.0], [0.0, 0.0]]])
    mask = torch.tensor([[1, 0]])
    packed = adapter.pack(
        text_features=text,
        audio_features=audio,
        visual_features=visual,
        attention_mask=mask,
    )
    torch.testing.assert_close(
        packed[0, 0],
        torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]),
    )
    with torch.no_grad():
        encoder.projection.bias.fill_(2.0)
    projected = encoder(packed, mask)
    assert projected.shape == (1, 2, 300)
    assert torch.count_nonzero(projected[0, 1]).item() == 0


def test_primary_dialogue_axis_bilstm_exposes_future_sensitivity() -> None:
    torch.manual_seed(17)
    encoder = PaperFormulaModalityEncoder(make_config()).eval()
    text = torch.randn(1, 4, 4)
    audio = torch.zeros(1, 4, 3)
    visual = torch.zeros(1, 4, 2)
    mask = torch.ones(1, 4, dtype=torch.long)
    lengths = torch.tensor([4])
    baseline = encoder(
        text_features=text,
        audio_features=audio,
        visual_features=visual,
        lengths=lengths,
        attention_mask=mask,
    ).text
    perturbed_text = text.clone()
    perturbed_text[:, 3] += 20.0
    perturbed = encoder(
        text_features=perturbed_text,
        audio_features=audio,
        visual_features=visual,
        lengths=lengths,
        attention_mask=mask,
    ).text
    assert not torch.allclose(baseline[:, 1], perturbed[:, 1], atol=1e-7, rtol=1e-7)
    identity = encoder(
        text_features=text,
        audio_features=audio,
        visual_features=visual,
        lengths=lengths,
        attention_mask=mask,
    ).context_visibility_identity
    assert identity.dag_topology_causal
    assert not identity.end_to_end_causal
    assert identity.context_leakage_risk == ("paper_dialogue_axis_text_bilstm",)


def test_causal_text_ablation_prefix_and_future_perturbation_are_invariant() -> None:
    torch.manual_seed(19)
    config = make_config(causal_text_ablation=True)
    encoder = PaperFormulaModalityEncoder(config).eval()
    text = torch.randn(1, 4, 4)
    audio = torch.zeros(1, 4, 3)
    visual = torch.zeros(1, 4, 2)
    full = encoder(
        text_features=text,
        audio_features=audio,
        visual_features=visual,
        lengths=torch.tensor([4]),
        attention_mask=torch.ones(1, 4, dtype=torch.long),
    )
    prefix = encoder(
        text_features=text[:, :2],
        audio_features=audio[:, :2],
        visual_features=visual[:, :2],
        lengths=torch.tensor([2]),
        attention_mask=torch.ones(1, 2, dtype=torch.long),
    )
    torch.testing.assert_close(full.text[:, :2], prefix.text, rtol=0, atol=1e-6)

    perturbed = text.clone()
    perturbed[:, 2:] += 50.0
    changed = encoder(
        text_features=perturbed,
        audio_features=audio,
        visual_features=visual,
        lengths=torch.tensor([4]),
        attention_mask=torch.ones(1, 4, dtype=torch.long),
    )
    torch.testing.assert_close(full.text[:, :2], changed.text[:, :2], rtol=0, atol=1e-6)
    assert changed.model_math_deviation_list == ("causal_unidirectional_text_encoder_ablation",)
    assert changed.context_visibility_identity.end_to_end_causal_assuming_local_text_features
    assert not changed.context_visibility_identity.end_to_end_causal


def test_causal_text_ablation_has_zero_gradient_to_future_inputs() -> None:
    torch.manual_seed(23)
    encoder = PaperFormulaModalityEncoder(make_config(causal_text_ablation=True)).eval()
    text = torch.randn(1, 4, 4, requires_grad=True)
    encoded = encoder(
        text_features=text,
        audio_features=torch.zeros(1, 4, 3),
        visual_features=torch.zeros(1, 4, 2),
        lengths=torch.tensor([4]),
        attention_mask=torch.ones(1, 4, dtype=torch.long),
    )
    encoded.text[0, 1].sum().backward()
    assert text.grad is not None
    assert torch.count_nonzero(text.grad[0, 2:]).item() == 0
    assert torch.count_nonzero(text.grad[0, :2]).item() > 0

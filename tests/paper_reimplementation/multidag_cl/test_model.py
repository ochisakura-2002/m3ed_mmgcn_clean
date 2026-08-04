from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from models.multidag_cl.paper_reimplementation.contracts import SpeakerRelations
from models.multidag_cl.paper_reimplementation.dag_layer import MultiDAGLayer
from models.multidag_cl.paper_reimplementation.graph import (
    CausalPredecessorBuilder,
    SpeakerRelationBuilder,
)
from models.multidag_cl.paper_reimplementation.model import (
    MultiDAGCLPaperReimplementation,
)

from _helpers import make_batch, make_config


def _forward_batch(batch: dict) -> dict:
    return {
        name: batch[name]
        for name in (
            "text_features",
            "audio_features",
            "visual_features",
            "attention_mask",
            "lengths",
            "speaker_ids_int",
            "labels",
        )
    }


def test_one_dag_layer_uses_only_past_completed_states_and_optional_diagnostics() -> None:
    torch.manual_seed(29)
    previous = torch.randn(2, 4, 300)
    previous[1, 3] = 0
    speakers = torch.tensor([[0, 1, 0, 1], [1, 0, 1, 0]], dtype=torch.int64)
    mask = torch.tensor([[1, 1, 1, 1], [1, 1, 1, 0]], dtype=torch.int64)
    adjacency = CausalPredecessorBuilder(1).build(speakers, mask)
    relations = SpeakerRelationBuilder.build(speakers, mask)
    layer = MultiDAGLayer(300, collect_diagnostics=True)
    result = layer(previous, adjacency, relations, mask)
    assert result.state.shape == (2, 4, 300)
    assert torch.count_nonzero(result.state[1, 3]).item() == 0
    assert result.attention_weights is not None
    assert not torch.triu(result.attention_weights, diagonal=0).any()
    rows_with_edges = adjacency.any(dim=-1)
    torch.testing.assert_close(
        result.attention_weights.sum(dim=-1)[rows_with_edges],
        torch.ones_like(result.attention_weights.sum(dim=-1)[rows_with_edges]),
    )

    compact = MultiDAGLayer(300, collect_diagnostics=False)(
        previous,
        adjacency,
        relations,
        mask,
    )
    assert compact.attention_logits is None
    assert compact.attention_weights is None
    assert compact.messages is None


def test_dag_layer_rejects_self_or_future_adjacency_even_if_it_could_slice_it() -> None:
    previous = torch.zeros(1, 2, 300)
    mask = torch.ones(1, 2, dtype=torch.long)
    same = torch.ones(1, 2, 2, dtype=torch.bool)
    relations = SpeakerRelations(same_speaker=same, different_speaker=~same)
    bad = torch.zeros(1, 2, 2, dtype=torch.bool)
    bad[0, 0, 1] = True
    with pytest.raises(ValueError, match="self or future"):
        MultiDAGLayer(300)(previous, bad, relations, mask)


def test_primary_model_one_layer_shapes_padding_loss_and_context_identity() -> None:
    torch.manual_seed(31)
    model = MultiDAGCLPaperReimplementation(
        make_config(graph_layers=1),
        collect_attention_diagnostics=True,
    ).eval()
    batch = make_batch()
    output = model(**_forward_batch(batch))
    assert output.logits.shape == (2, 4, 3)
    assert output.encoded_state.shape == (2, 4, 300)
    assert len(output.layer_states) == 2
    assert output.representation.shape == (2, 4, 600)
    assert model.classifier_input_dim == 600
    assert torch.count_nonzero(output.logits[1, 3]).item() == 0
    assert torch.count_nonzero(output.representation[1, 3]).item() == 0
    assert output.profile_identity == "paper_formula_behavior"
    assert output.context_visibility_identity.dag_topology_causal
    assert not output.context_visibility_identity.end_to_end_causal
    assert output.diagnostics is not None
    assert output.diagnostics.layer_diagnostics[0].attention_weights is not None

    valid = batch["attention_mask"].bool() & (batch["labels"] != -100)
    expected_loss = F.cross_entropy(output.logits[valid], batch["labels"][valid])
    torch.testing.assert_close(output.loss, expected_loss)


def test_primary_four_layer_representation_is_exactly_1500() -> None:
    torch.manual_seed(37)
    model = MultiDAGCLPaperReimplementation(make_config(graph_layers=4)).eval()
    output = model(
        text_features=torch.randn(1, 2, 4),
        audio_features=torch.randn(1, 2, 3),
        visual_features=torch.randn(1, 2, 2),
        attention_mask=torch.ones(1, 2, dtype=torch.long),
        lengths=torch.tensor([2]),
        speaker_ids_int=torch.tensor([[0, 1]]),
        labels=torch.tensor([[0, 1]]),
    )
    assert len(output.layer_states) == 5
    assert output.representation.shape == (1, 2, 1500)
    assert model.classifier_input_dim == 1500


def test_source_profile_has_named_raw_skip_and_distinct_classifier_dimension() -> None:
    torch.manual_seed(41)
    model = MultiDAGCLPaperReimplementation(
        make_config(profile="official_source_behavior", graph_layers=1)
    ).eval()
    batch = make_batch()
    output = model(**_forward_batch(batch))
    assert output.raw_concat is not None
    assert output.raw_concat.shape[-1] == 9
    assert output.representation.shape[-1] == 609
    assert model.classifier_input_dim == 609
    torch.testing.assert_close(output.representation[..., -9:], output.raw_concat)
    assert output.encoded_modalities is None
    assert output.profile_identity == "official_source_behavior"


def test_model_forward_backward_is_finite_without_optimizer_steps() -> None:
    torch.manual_seed(43)
    model = MultiDAGCLPaperReimplementation(make_config(graph_layers=1))
    output = model(**_forward_batch(make_batch()))
    assert output.loss is not None and torch.isfinite(output.loss)
    output.loss.backward()
    gradients = {name: parameter.grad for name, parameter in model.named_parameters()}
    assert all(value is not None for value in gradients.values())
    assert all(torch.isfinite(value).all() for value in gradients.values())


def test_masked_cross_entropy_excludes_minus_100_and_rejects_no_valid_label() -> None:
    torch.manual_seed(47)
    model = MultiDAGCLPaperReimplementation(make_config()).eval()
    batch = make_batch()
    batch["labels"][0, 1] = -100
    output = model(**_forward_batch(batch))
    valid = batch["attention_mask"].bool() & (batch["labels"] != -100)
    expected = F.cross_entropy(output.logits[valid], batch["labels"][valid])
    torch.testing.assert_close(output.loss, expected)

    batch["labels"][batch["attention_mask"].bool()] = -100
    with pytest.raises(ValueError, match="at least one valid label"):
        model(**_forward_batch(batch))


def test_strict_state_dict_reload_is_exact_and_cross_profile_load_is_rejected_first() -> None:
    torch.manual_seed(53)
    config = make_config()
    source = MultiDAGCLPaperReimplementation(config).eval()
    target = MultiDAGCLPaperReimplementation(config).eval()
    state = source.state_dict()
    assert "_extra_state" in state
    target.load_state_dict(state, strict=True)
    batch = _forward_batch(make_batch())
    with torch.no_grad():
        first = source(**batch).logits
        second = target(**batch).logits
    torch.testing.assert_close(first, second, rtol=0, atol=0)

    different_profile = MultiDAGCLPaperReimplementation(
        make_config(profile="official_source_behavior")
    )
    with pytest.raises(ValueError, match="profile identity mismatch"):
        different_profile.load_state_dict(state, strict=True)


def test_forward_cannot_switch_profile_and_causal_ablation_is_future_invariant() -> None:
    torch.manual_seed(59)
    model = MultiDAGCLPaperReimplementation(
        make_config(causal_text_ablation=True)
    ).eval()
    batch = make_batch()
    batch["attention_mask"] = torch.ones(2, 4, dtype=torch.long)
    batch["lengths"] = torch.tensor([4, 4])
    batch["labels"][1, 3] = 0
    baseline = model(**_forward_batch(batch))
    perturbed = {name: value.clone() for name, value in _forward_batch(batch).items()}
    perturbed["text_features"][:, 3] += 100.0
    perturbed["audio_features"][:, 3] -= 100.0
    perturbed["visual_features"][:, 3] += 50.0
    changed = model(**perturbed)
    torch.testing.assert_close(baseline.logits[:, :3], changed.logits[:, :3], rtol=0, atol=1e-6)
    assert baseline.context_visibility_identity.end_to_end_causal_assuming_local_text_features
    assert not baseline.context_visibility_identity.end_to_end_causal

    with pytest.raises(TypeError):
        model(**_forward_batch(batch), conformance_profile="official_source_behavior")


def test_inference_without_labels_returns_none_loss() -> None:
    model = MultiDAGCLPaperReimplementation(make_config()).eval()
    batch = _forward_batch(make_batch())
    del batch["labels"]
    assert model(**batch).loss is None

from __future__ import annotations

import torch

from models.common.causal_graph import (
    assert_no_future_edges,
    build_causal_multimodal_adjacency,
    build_causal_utterance_adjacency,
    build_full_context_negative_control_adjacency,
    build_speaker_pair_relation_ids,
    masked_softmax,
    row_normalize_adjacency,
    utterance_time_to_multimodal_node_time,
)


def _mask_and_lengths() -> tuple[torch.Tensor, torch.Tensor]:
    lengths = torch.tensor([5, 3])
    mask = torch.arange(5).unsqueeze(0) < lengths.unsqueeze(1)
    return mask, lengths


def test_utterance_adjacency_has_no_future_cross_dialogue_or_padding_edges() -> None:
    mask, lengths = _mask_and_lengths()
    adjacency = build_causal_utterance_adjacency(mask, lengths, window_past=None)
    assert adjacency.shape == (2, 5, 5)
    assert_no_future_edges(adjacency, torch.arange(5))
    assert torch.count_nonzero(adjacency[1, 3:]).item() == 0
    assert torch.count_nonzero(adjacency[1, :, 3:]).item() == 0
    assert bool(adjacency[0, 4, 0])
    assert not bool(adjacency[0, 0, 4])
    # A batch dimension is a set of disjoint graphs; no tensor index can cross it.
    assert adjacency[0].data_ptr() != adjacency[1].data_ptr()


def test_window_past_none_and_zero_have_explicit_semantics() -> None:
    mask, lengths = _mask_and_lengths()
    all_history = build_causal_utterance_adjacency(mask, lengths, window_past=None)
    same_time = build_causal_utterance_adjacency(mask, lengths, window_past=0)
    assert bool(all_history[0, 4, 0])
    assert torch.equal(same_time[0], torch.eye(5, dtype=torch.bool))


def test_multimodal_adjacency_allows_same_time_cross_modal_edges_only_causally() -> None:
    mask, lengths = _mask_and_lengths()
    adjacency = build_causal_multimodal_adjacency(mask, lengths, window_past=2)
    node_time = utterance_time_to_multimodal_node_time(5, 3)
    assert adjacency.shape == (2, 15, 15)
    assert_no_future_edges(adjacency, node_time)
    target = 2 * 3 + 0
    same_time_other_modality = 2 * 3 + 2
    future = 3 * 3 + 0
    assert bool(adjacency[0, target, same_time_other_modality])
    assert not bool(adjacency[0, target, future])
    assert torch.count_nonzero(adjacency[1, 9:]).item() == 0
    assert torch.count_nonzero(adjacency[1, :, 9:]).item() == 0


def test_row_normalization_is_over_legal_sources() -> None:
    mask, lengths = _mask_and_lengths()
    adjacency = build_causal_utterance_adjacency(mask, lengths, window_past=2)
    normalized = row_normalize_adjacency(adjacency)
    row_sums = normalized.sum(dim=-1)
    torch.testing.assert_close(row_sums[mask], torch.ones_like(row_sums[mask]))
    assert torch.count_nonzero(row_sums[~mask]).item() == 0
    assert_no_future_edges(normalized, torch.arange(5))


def test_masked_softmax_masks_before_normalization() -> None:
    scores = torch.tensor([[[1.0, 2.0, 1000.0], [3.0, 4.0, -1000.0]]])
    mask = torch.tensor([[[True, True, False], [True, False, False]]])
    probabilities = masked_softmax(scores, mask)
    changed_scores = scores.clone()
    changed_scores[..., 2] = -1e8
    changed = masked_softmax(changed_scores, mask)
    torch.testing.assert_close(probabilities, changed, atol=0, rtol=0)
    torch.testing.assert_close(probabilities.sum(-1), torch.ones(1, 2))
    assert torch.count_nonzero(probabilities[~mask]).item() == 0


def test_speaker_pair_relation_ids_follow_source_to_target_mapping() -> None:
    mask, lengths = _mask_and_lengths()
    speakers = torch.tensor([[0, 1, 0, 1, 0], [1, 0, 1, 0, 0]])
    adjacency = build_causal_utterance_adjacency(mask, lengths, window_past=None)
    relation_ids, mapping = build_speaker_pair_relation_ids(speakers, adjacency, 2)
    # target t=1 speaker 1, source t=0 speaker 0 -> id 0*2+1 == 1.
    assert int(relation_ids[0, 1, 0]) == 1
    assert mapping[1] == "source_0->target_1"
    assert torch.all(relation_ids[~adjacency] == -1)


def test_full_context_negative_control_fails_causal_assertion() -> None:
    mask, lengths = _mask_and_lengths()
    adjacency = build_full_context_negative_control_adjacency(mask, lengths)
    assert bool(adjacency[0, 0, 4])
    try:
        assert_no_future_edges(adjacency, torch.arange(5))
    except AssertionError:
        pass
    else:
        raise AssertionError("full-context negative control unexpectedly passed")

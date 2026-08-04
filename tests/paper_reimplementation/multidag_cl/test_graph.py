from __future__ import annotations

import pytest
import torch
from typing import Optional

from models.multidag_cl.paper_reimplementation.graph import (
    CausalPredecessorBuilder,
    SpeakerRelationBuilder,
)


def _build(speakers, *, window: int = 1, length: Optional[int] = None) -> torch.Tensor:
    tensor = torch.tensor([speakers], dtype=torch.int64)
    valid = len(speakers) if length is None else length
    mask = torch.tensor([[1] * valid + [0] * (len(speakers) - valid)], dtype=torch.int64)
    return CausalPredecessorBuilder(window).build(tensor, mask)[0]


def test_exact_five_node_predecessor_fixture() -> None:
    adjacency = _build([0, 1, 1, 0, 1])
    expected = torch.tensor(
        [
            [0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [1, 1, 1, 0, 0],
            [0, 0, 1, 1, 0],
        ],
        dtype=torch.bool,
    )
    assert torch.equal(adjacency, expected)


def test_first_node_and_unseen_speaker_include_the_available_history() -> None:
    adjacency = _build([0, 1, 2])
    assert not adjacency[0].any()
    assert torch.equal(adjacency[1], torch.tensor([1, 0, 0], dtype=torch.bool))
    assert torch.equal(adjacency[2], torch.tensor([1, 1, 0], dtype=torch.bool))


def test_window_one_and_two_for_single_speaker_are_exact() -> None:
    window_one = _build([0, 0, 0, 0], window=1)
    window_two = _build([0, 0, 0, 0], window=2)
    expected_one = torch.tensor(
        [[0, 0, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]],
        dtype=torch.bool,
    )
    expected_two = torch.tensor(
        [[0, 0, 0, 0], [1, 0, 0, 0], [1, 1, 0, 0], [0, 1, 1, 0]],
        dtype=torch.bool,
    )
    assert torch.equal(window_one, expected_one)
    assert torch.equal(window_two, expected_two)


def test_alternating_and_consecutive_speaker_scans_include_intervening_nodes() -> None:
    alternating = _build([0, 1, 0, 1])
    assert torch.equal(
        alternating,
        torch.tensor(
            [[0, 0, 0, 0], [1, 0, 0, 0], [1, 1, 0, 0], [0, 1, 1, 0]],
            dtype=torch.bool,
        ),
    )
    consecutive = _build([0, 1, 1, 1])
    assert torch.equal(consecutive[2], torch.tensor([0, 1, 0, 0], dtype=torch.bool))
    assert torch.equal(consecutive[3], torch.tensor([0, 0, 1, 0], dtype=torch.bool))


def test_padding_rows_columns_self_and_future_are_always_false() -> None:
    speakers = torch.tensor([[0, 1, 0, 0], [1, 0, 1, 0]], dtype=torch.int64)
    mask = torch.tensor([[1, 1, 1, 0], [1, 1, 1, 1]], dtype=torch.int64)
    adjacency = CausalPredecessorBuilder(1).build(speakers, mask)
    assert not adjacency[0, 3].any()
    assert not adjacency[0, :, 3].any()
    assert not torch.diagonal(adjacency, dim1=1, dim2=2).any()
    assert not torch.triu(adjacency, diagonal=0).any()


@pytest.mark.parametrize("window", [0, -1])
def test_illegal_predecessor_window_is_rejected(window: int) -> None:
    with pytest.raises(ValueError, match="at least 1"):
        CausalPredecessorBuilder(window)


def test_exact_same_and_different_relation_fixture_and_edge_intersection() -> None:
    speakers = torch.tensor([[0, 1, 1, 0, 1]], dtype=torch.int64)
    mask = torch.ones(1, 5, dtype=torch.int64)
    relations = SpeakerRelationBuilder.build(speakers, mask)
    expected_same = torch.tensor(
        [
            [1, 0, 0, 1, 0],
            [0, 1, 1, 0, 1],
            [0, 1, 1, 0, 1],
            [1, 0, 0, 1, 0],
            [0, 1, 1, 0, 1],
        ],
        dtype=torch.bool,
    )
    expected_different = ~expected_same
    assert torch.equal(relations.same_speaker[0], expected_same)
    assert torch.equal(relations.different_speaker[0], expected_different)
    assert not (relations.same_speaker & relations.different_speaker).any()
    assert (relations.same_speaker | relations.different_speaker).all()

    adjacency = CausalPredecessorBuilder(1).build(speakers, mask)
    same_edges = adjacency & relations.same_speaker
    different_edges = adjacency & relations.different_speaker
    assert torch.equal(same_edges | different_edges, adjacency)
    assert not (same_edges & different_edges).any()


def test_invalid_relation_pairs_are_false_for_padding() -> None:
    speakers = torch.tensor([[0, 1, 0, 0]], dtype=torch.int64)
    mask = torch.tensor([[1, 1, 1, 0]], dtype=torch.int64)
    relations = SpeakerRelationBuilder.build(speakers, mask)
    assert not relations.same_speaker[0, 3].any()
    assert not relations.same_speaker[0, :, 3].any()
    assert not relations.different_speaker[0, 3].any()
    assert not relations.different_speaker[0, :, 3].any()

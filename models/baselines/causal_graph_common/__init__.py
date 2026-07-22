"""Compatibility wrapper for :mod:`models.common.causal_graph`."""

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

__all__ = [
    "assert_no_future_edges",
    "build_causal_multimodal_adjacency",
    "build_causal_utterance_adjacency",
    "build_full_context_negative_control_adjacency",
    "build_speaker_pair_relation_ids",
    "masked_softmax",
    "row_normalize_adjacency",
    "utterance_time_to_multimodal_node_time",
]

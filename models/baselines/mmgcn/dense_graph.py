"""Compatibility wrapper for :mod:`models.mmgcn.unified.dense_graph`."""

from models.mmgcn.unified.dense_graph import (
    active_modalities_to_node_mask,
    adjacency_density,
    build_official_like_multimodal_adjacency,
    build_utterance_index,
    context_allowed,
    cosine_to_mm_similarity,
    lengths_to_list,
    normalize_active_modalities,
    normalize_window,
    paired_mm_similarity,
    pairwise_mm_similarity,
)

__all__ = [
    "active_modalities_to_node_mask",
    "adjacency_density",
    "build_official_like_multimodal_adjacency",
    "build_utterance_index",
    "context_allowed",
    "cosine_to_mm_similarity",
    "lengths_to_list",
    "normalize_active_modalities",
    "normalize_window",
    "paired_mm_similarity",
    "pairwise_mm_similarity",
]

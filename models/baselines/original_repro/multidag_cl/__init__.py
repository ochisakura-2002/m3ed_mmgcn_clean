"""Compatibility wrapper for :mod:`models.multidag_cl.paper_aligned`."""

from models.multidag_cl.paper_aligned import (
    OriginalReproMultiDAGCL,
    build_get_adj_v1,
    curriculum_baby_step_indices,
    dialogue_difficulty,
    dialogue_difficulty_from_sequences,
)

__all__ = [
    "OriginalReproMultiDAGCL",
    "build_get_adj_v1",
    "curriculum_baby_step_indices",
    "dialogue_difficulty",
    "dialogue_difficulty_from_sequences",
]

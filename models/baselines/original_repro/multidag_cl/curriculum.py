"""Compatibility wrapper for the paper-aligned Curriculum Learning helpers."""

from models.multidag_cl.paper_aligned.curriculum import (
    curriculum_baby_step_indices,
    dialogue_difficulty,
    dialogue_difficulty_from_sequences,
)

__all__ = [
    "curriculum_baby_step_indices",
    "dialogue_difficulty",
    "dialogue_difficulty_from_sequences",
]

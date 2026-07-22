"""Compatibility wrapper for :mod:`models.multidag_cl.unified.multidag_cl_model`."""

from models.multidag_cl.unified.multidag_cl_model import (
    CausalModalityEncoder,
    MultiDAGCLBaseline,
    build_directed_past_adjacency,
    build_speaker_relation_masks,
)

__all__ = [
    "CausalModalityEncoder",
    "MultiDAGCLBaseline",
    "build_directed_past_adjacency",
    "build_speaker_relation_masks",
]

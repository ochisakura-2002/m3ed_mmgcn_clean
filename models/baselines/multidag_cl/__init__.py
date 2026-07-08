"""Device-neutral MultiDAG+CL baseline package."""

from .multidag_cl_model import (
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

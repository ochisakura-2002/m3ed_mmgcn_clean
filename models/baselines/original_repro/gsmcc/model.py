"""Compatibility wrapper for the full-context GS-MCC project variant."""

from models.gsmcc.project_variant.full_context.model import (
    FourierGraphOperator,
    ProjectPaperOrientedGSMCC,
    angular_similarity,
    build_sliding_multimodal_graph,
    cross_frequency_contrastive_loss,
)

__all__ = [
    "FourierGraphOperator",
    "ProjectPaperOrientedGSMCC",
    "angular_similarity",
    "build_sliding_multimodal_graph",
    "cross_frequency_contrastive_loss",
]

"""Compatibility wrapper for :mod:`models.common.paper_aligned`."""

from models.common.paper_aligned import (
    CAUSAL_GRADE,
    active_feature_concat,
    class_weight_tensor,
    flatten_valid,
    masked_cross_entropy,
    masked_softmax,
    run_packed_rnn,
    scatter_valid,
    structured_output,
    validate_dialogue_batch,
)

__all__ = [
    "CAUSAL_GRADE",
    "active_feature_concat",
    "class_weight_tensor",
    "flatten_valid",
    "masked_cross_entropy",
    "masked_softmax",
    "run_packed_rnn",
    "scatter_valid",
    "structured_output",
    "validate_dialogue_batch",
]

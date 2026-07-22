"""Compatibility wrapper for :mod:`models.registry.causal`."""

from models.registry.causal import (
    CAUSAL_DIALOGUEGCN_NAME,
    CAUSAL_GSMCC_NAME,
    build_new_causal_baseline,
    get_new_causal_model_family,
    normalize_new_causal_model_name,
    validate_new_causal_model_config,
)

__all__ = [
    "CAUSAL_DIALOGUEGCN_NAME",
    "CAUSAL_GSMCC_NAME",
    "build_new_causal_baseline",
    "get_new_causal_model_family",
    "normalize_new_causal_model_name",
    "validate_new_causal_model_config",
]

"""Compatibility wrapper for :mod:`models.registry.paper_aligned`."""

from models.registry.paper_aligned import (
    MODEL_REGISTRY,
    ORIGINAL_REPRO_MODEL_KEYS,
    SOURCE_PROVENANCE,
    build_original_repro_model,
    canonical_model_key,
    get_model_constructor_args,
    get_source_metadata,
)

__all__ = [
    "MODEL_REGISTRY",
    "ORIGINAL_REPRO_MODEL_KEYS",
    "SOURCE_PROVENANCE",
    "build_original_repro_model",
    "canonical_model_key",
    "get_model_constructor_args",
    "get_source_metadata",
]

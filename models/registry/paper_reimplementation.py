"""Canonical registry for independently authored paper reimplementations."""

from __future__ import annotations

from typing import Any, Mapping

from models.multidag_cl.paper_reimplementation import (
    MultiDAGCLConfig,
    MultiDAGCLPaperReimplementation,
)


REGISTRY_KEY = "multidag_cl_paper_reimplementation"

MODEL_REGISTRY = {
    REGISTRY_KEY: MultiDAGCLPaperReimplementation,
}

MODEL_METADATA: dict[str, dict[str, Any]] = {
    REGISTRY_KEY: {
        "canonical_name": REGISTRY_KEY,
        "implementation_identity": "paper_reimplementation",
        "conformance_profile": "paper_formula_behavior",
        "data_track": "project_fair",
        "supported_data_tracks": ["project_fair", "paper_data"],
        "paper_title": (
            "Curriculum Learning Meets Directed Acyclic Graph for "
            "Multimodal Emotion Recognition"
        ),
        "paper_venue": "LREC-COLING",
        "paper_year": 2024,
        "paper_sha256": (
            "290ecd8b29c7038e2abec1b873bcd150929d3819bb02ba1416bb9686e327593b"
        ),
        "official_repo_url": "https://github.com/vanntc711/MultiDAG-CL",
        "official_commit": "59a75877065a91bf9388fbb564607fe79717fd4f",
        "paper_data_track_status": "ASSET_GATED_STAGE_C2_INGESTION_READY",
        "dag_topology_causal": True,
        "end_to_end_causal": False,
        "context_label": "full_context",
    }
}


def canonical_model_key(value: Any) -> str:
    key = str(value).strip().lower()
    if key not in MODEL_REGISTRY:
        raise KeyError(
            f"unknown paper-reimplementation registry key {value!r}; "
            f"expected {REGISTRY_KEY!r}"
        )
    return key


def get_model_metadata(value: Any = REGISTRY_KEY) -> dict[str, Any]:
    """Return a defensive copy of the frozen registry metadata."""

    return dict(MODEL_METADATA[canonical_model_key(value)])


def build_paper_reimplementation_model(config: Mapping[str, Any]):
    """Build the registered model from a fully resolved ``model_core`` map."""

    if not isinstance(config, Mapping):
        raise TypeError("config must be a mapping")
    key = canonical_model_key(config.get("registry_key", REGISTRY_KEY))
    core_mapping = config.get("model_core")
    if not isinstance(core_mapping, Mapping):
        raise TypeError("config.model_core must be a mapping")
    model_config = MultiDAGCLConfig.from_mapping(core_mapping)
    metadata = MODEL_METADATA[key]
    identity = {
        "canonical_name": model_config.canonical_name,
        "implementation_identity": model_config.implementation_identity,
        "conformance_profile": model_config.conformance_profile.value,
    }
    expected = {name: metadata[name] for name in identity}
    if identity != expected:
        raise ValueError(
            f"model_core identity does not match registry metadata: "
            f"configured={identity}, expected={expected}"
        )
    if model_config.data_track.value not in metadata["supported_data_tracks"]:
        raise ValueError(
            f"unsupported paper-reimplementation data track: "
            f"{model_config.data_track.value!r}"
        )
    return MODEL_REGISTRY[key](model_config)


__all__ = [
    "MODEL_METADATA",
    "MODEL_REGISTRY",
    "REGISTRY_KEY",
    "build_paper_reimplementation_model",
    "canonical_model_key",
    "get_model_metadata",
]

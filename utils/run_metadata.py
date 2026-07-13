"""Build and atomically write reproducibility metadata for training runs."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple


FEATURE_CAUSALITY_STATUS = (
    "utterance_level_but_extractor_not_fully_verified"
)
DEFAULT_CAUSAL_CONTRACT_VERSION = "1.0"


def compute_file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Return a file SHA256 without loading the complete file into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as file:
        while True:
            chunk = file.read(int(chunk_size))
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _optional_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _optional_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    return int(value)


def _normalized_optional_window(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() in {"", "none", "null"}:
        return None
    return int(value)


def _model_name(config: Mapping[str, Any]) -> str:
    model = config.get("model", {})
    return str(model.get("name", "")).strip()


def infer_model_causality(
    config: Mapping[str, Any],
) -> Tuple[Optional[bool], Optional[bool]]:
    """Infer causal flags from the graph and encoder paths used by builders.

    The top-level ``causal`` label is intentionally ignored: metadata follows
    the effective model semantics rather than a filename or declarative tag.
    """

    name = _model_name(config).lower()
    graph = config.get("graph", {})
    context_mode = str(graph.get("context_mode", "full")).strip().lower()

    if name == "mmgcn":
        if context_mode == "causal":
            return True, False
        if context_mode == "full":
            window_future = _normalized_optional_window(
                graph.get("window_future")
            )
            future_allowed = window_future is None or window_future > 0
            return not future_allowed, future_allowed
        return None, None

    if name in {"multidagcl", "multidag", "multidag-inspired"}:
        # The project builder maps causal/full/past_all_causal to a directed
        # past-only window. Both supported encoders are causal: a forward GRU
        # or an utterance-local linear projection.
        graph_is_causal = context_mode in {
            "causal",
            "full",
            "past_all_causal",
        }
        encoder_type = str(
            config.get("model", {}).get(
                "modality_encoder_type", "causal_gru"
            )
        ).strip().lower()
        encoder_is_causal = encoder_type in {"causal_gru", "linear"}
        encoder_is_noncausal = encoder_type in {
            "bigru",
            "bilstm",
            "bidirectional_gru",
            "bidirectional_lstm",
        }
        if graph_is_causal and encoder_is_causal:
            return True, False
        if encoder_is_noncausal:
            return False, True
        return None, None

    return None, None


def _model_family_and_variant(
    config: Mapping[str, Any],
    causal_mode: Optional[bool],
    future_context_allowed: Optional[bool],
) -> Tuple[str, str]:
    explicit_family = _optional_text(config.get("model_family"))
    explicit_variant = _optional_text(config.get("model_variant"))
    name = _model_name(config)
    normalized_name = name.lower()

    if explicit_family is not None:
        family = explicit_family
    elif normalized_name == "mmgcn":
        family = "MMGCN"
    elif normalized_name in {"multidagcl", "multidag", "multidag-inspired"}:
        family = "MultiDAG"
    else:
        family = name or "unknown"

    if explicit_variant is not None:
        variant = explicit_variant
    elif normalized_name == "mmgcn":
        if causal_mode is True:
            variant = "causal_dense_graph"
        elif future_context_allowed is True:
            variant = "full_context_dense_graph"
        else:
            variant = "dense_graph"
    elif normalized_name in {"multidagcl", "multidag", "multidag-inspired"}:
        encoder_type = str(
            config.get("model", {}).get(
                "modality_encoder_type", "causal_gru"
            )
        ).strip().lower()
        variant = f"project_multidag_inspired_{encoder_type}"
    else:
        variant = name or "unknown"
    return family, variant


def _feature_sha256(
    dataset: Mapping[str, Any],
    project_root: Path,
) -> Optional[str]:
    configured = _optional_text(dataset.get("feature_sha256"))
    if configured is not None:
        return configured

    raw_path = _optional_text(dataset.get("feature_pkl_path"))
    if raw_path is None:
        return None
    feature_path = Path(raw_path).expanduser()
    if not feature_path.is_absolute():
        feature_path = project_root / feature_path
    if not feature_path.is_file():
        return None
    return compute_file_sha256(feature_path)


def _checkpoint_selection_metric(
    config: Mapping[str, Any], model_family: str
) -> Optional[str]:
    if model_family == "MMGCN":
        return _optional_text(
            config.get("logging", {}).get("monitor_metric", "val_uar")
        )
    if model_family == "MultiDAG":
        return _optional_text(
            config.get("training", {}).get(
                "select_best_by", "val_weighted_f1"
            )
        )
    return _optional_text(
        config.get("training", {}).get("select_best_by")
    )


def build_run_metadata(
    config: Mapping[str, Any], project_root: Path
) -> Dict[str, Any]:
    """Build the canonical, JSON-safe metadata mapping for one run."""

    dataset = config.get("dataset", {})
    model = config.get("model", {})
    causal_mode, future_context_allowed = infer_model_causality(config)
    model_family, model_variant = _model_family_and_variant(
        config,
        causal_mode=causal_mode,
        future_context_allowed=future_context_allowed,
    )

    feature_path = _optional_text(dataset.get("feature_pkl_path"))
    contract_version = _optional_text(config.get("causal_contract_version"))
    if contract_version is None:
        contract_version = DEFAULT_CAUSAL_CONTRACT_VERSION

    return {
        "model_family": model_family,
        "model_variant": model_variant,
        "causal_mode": causal_mode,
        "causal_contract_version": contract_version,
        # Preserve the configured relative/original value. Resolution is used
        # only internally when an on-disk hash must be computed.
        "feature_pkl_path": feature_path,
        "feature_sha256": _feature_sha256(dataset, Path(project_root)),
        "val_split_strategy": _optional_text(
            dataset.get("val_split_strategy")
        ),
        "val_session_id": _optional_text(dataset.get("val_session_id")),
        "future_context_allowed": future_context_allowed,
        "checkpoint_selection_metric": _checkpoint_selection_metric(
            config, model_family
        ),
        "seed": int(config.get("system", {}).get("seed", 42)),
        "text_dim": _optional_int(model.get("text_feature_dim")),
        "audio_dim": _optional_int(model.get("audio_feature_dim")),
        "visual_dim": _optional_int(model.get("visual_feature_dim")),
        "feature_causality_status": FEATURE_CAUSALITY_STATUS,
    }


def write_run_metadata(
    config: Mapping[str, Any], output_path: Path, project_root: Path
) -> Dict[str, Any]:
    """Atomically write ``run_metadata.json`` and return its content."""

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    metadata = build_run_metadata(config, project_root=Path(project_root))
    temp_path = destination.with_name(
        f".{destination.name}.{os.getpid()}.tmp"
    )
    try:
        with temp_path.open("w", encoding="utf-8", newline="\n") as file:
            json.dump(metadata, file, ensure_ascii=False, indent=2)
            file.write("\n")
            file.flush()
            os.fsync(file.fileno())
        os.replace(temp_path, destination)
    finally:
        if temp_path.exists():
            temp_path.unlink()
    return metadata

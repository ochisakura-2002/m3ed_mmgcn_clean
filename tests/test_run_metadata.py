from __future__ import annotations

import hashlib
import json
from pathlib import Path

from utils.run_metadata import (
    FEATURE_CAUSALITY_STATUS,
    build_run_metadata,
    infer_model_causality,
    write_run_metadata,
)


def _mmgcn_config(feature_path: str) -> dict:
    return {
        "causal": False,  # Deliberately conflicts; graph semantics must win.
        "causal_contract_version": "1.0",
        "system": {"seed": 17},
        "dataset": {
            "feature_pkl_path": feature_path,
            "val_split_strategy": "session_holdout",
            "val_session_id": "Ses02",
        },
        "model": {
            "name": "MMGCN",
            "text_feature_dim": 4,
            "audio_feature_dim": 3,
            "visual_feature_dim": 2,
        },
        "graph": {
            "context_mode": "causal",
            "window_past": 5,
            "window_future": 0,
        },
        "logging": {"monitor_metric": "val_weighted_f1"},
    }


def _multidag_config(feature_path: str) -> dict:
    return {
        "causal": False,  # The project builder still uses directed-past graph.
        "system": {"seed": 23},
        "dataset": {
            "feature_pkl_path": feature_path,
            "val_split_strategy": "official_prefix",
        },
        "model": {
            "name": "MultiDAGCL",
            "text_feature_dim": 5,
            "audio_feature_dim": 4,
            "visual_feature_dim": 3,
            "modality_encoder_type": "causal_gru",
        },
        # In this project, "full" is resolved to past_all_causal.
        "graph": {"context_mode": "full", "window_past": None},
        "training": {"select_best_by": "val_weighted_f1"},
    }


def test_mmgcn_metadata_uses_effective_graph_semantics_and_stream_hash(
    tmp_path: Path,
) -> None:
    feature_path = tmp_path / "features.pkl"
    feature_bytes = b"small synthetic feature payload"
    feature_path.write_bytes(feature_bytes)
    config = _mmgcn_config("features.pkl")

    metadata = build_run_metadata(config, project_root=tmp_path)

    assert metadata == {
        "model_family": "MMGCN",
        "model_variant": "causal_dense_graph",
        "causal_mode": True,
        "causal_contract_version": "1.0",
        "feature_pkl_path": "features.pkl",
        "feature_sha256": hashlib.sha256(feature_bytes).hexdigest(),
        "val_split_strategy": "session_holdout",
        "val_session_id": "Ses02",
        "future_context_allowed": False,
        "checkpoint_selection_metric": "val_weighted_f1",
        "seed": 17,
        "text_dim": 4,
        "audio_dim": 3,
        "visual_dim": 2,
        "feature_causality_status": FEATURE_CAUSALITY_STATUS,
        "test_split_used_for_selection": False,
    }


def test_configured_feature_sha256_takes_precedence(tmp_path: Path) -> None:
    (tmp_path / "features.pkl").write_bytes(b"on-disk value")
    config = _mmgcn_config("features.pkl")
    config["dataset"]["feature_sha256"] = "configured-digest"

    metadata = build_run_metadata(config, project_root=tmp_path)

    assert metadata["feature_sha256"] == "configured-digest"


def test_missing_feature_file_records_null_hash(tmp_path: Path) -> None:
    metadata = build_run_metadata(
        _mmgcn_config("missing/features.pkl"), project_root=tmp_path
    )

    assert metadata["feature_pkl_path"] == "missing/features.pkl"
    assert metadata["feature_sha256"] is None


def test_causality_inference_covers_full_context_and_future_window() -> None:
    full_config = _mmgcn_config("unused.pkl")
    full_config["graph"] = {
        "context_mode": "full",
        "window_future": None,
    }
    zero_future_config = _mmgcn_config("unused.pkl")
    zero_future_config["graph"] = {
        "context_mode": "full",
        "window_future": 0,
    }

    assert infer_model_causality(full_config) == (False, True)
    assert infer_model_causality(zero_future_config) == (True, False)


def test_multidag_metadata_reflects_project_past_only_semantics(
    tmp_path: Path,
) -> None:
    metadata = build_run_metadata(
        _multidag_config("missing.pkl"), project_root=tmp_path
    )

    assert metadata["model_family"] == "MultiDAG"
    assert metadata["model_variant"] == "project_multidag_inspired_causal_gru"
    assert metadata["causal_mode"] is True
    assert metadata["future_context_allowed"] is False
    assert metadata["checkpoint_selection_metric"] == "val_weighted_f1"


def test_write_run_metadata_is_atomic_and_json_round_trips(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "run" / "run_metadata.json"
    expected = write_run_metadata(
        _multidag_config("missing.pkl"),
        output_path=destination,
        project_root=tmp_path,
    )

    with destination.open("r", encoding="utf-8") as file:
        actual = json.load(file)
    assert actual == expected
    assert list(destination.parent.glob(".run_metadata.json.*.tmp")) == []

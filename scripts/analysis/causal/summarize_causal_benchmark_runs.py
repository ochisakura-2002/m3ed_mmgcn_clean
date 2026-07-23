"""Audit and summarize the configured eight-run causal benchmark.

The script is intentionally read-only with respect to training runs.  It never
evaluates a checkpoint and never chooses an artifact by its metric value.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from utils.output_paths import (  # noqa: E402
    configured_output_root,
    find_run_directory,
    infer_experiment_date_from_run,
    resolve_experiment_date,
    resolve_output_category,
    sanitize_run_name,
)
UNCONFIRMED = "UNCONFIRMED"
SUPPORTED_ARTIFACT_SUFFIXES = {".json", ".csv", ".yaml", ".yml"}
EXPECTED_SESSIONS = ("Ses01", "Ses02", "Ses03", "Ses04")

METRIC_ALIASES: Dict[str, Tuple[str, ...]] = {
    "epoch": ("epoch", "epoch_number", "epoch_index"),
    "train_loss": ("train_loss", "training_loss"),
    "val_loss": ("val_loss", "validation_loss"),
    "val_accuracy": (
        "val_accuracy",
        "val_acc",
        "validation_accuracy",
        "validation_acc",
    ),
    "val_weighted_f1": (
        "val_weighted_f1",
        "val_wf1",
        "validation_weighted_f1",
        "validation_wf1",
    ),
    "val_macro_f1": ("val_macro_f1", "validation_macro_f1"),
    "val_uar": (
        "val_uar",
        "validation_uar",
        "val_unweighted_recall",
        "validation_unweighted_recall",
    ),
    "learning_rate": (
        "learning_rate",
        "lr",
        "lr_after_scheduler",
    ),
}

TEST_METRIC_ALIASES: Dict[str, Tuple[str, ...]] = {
    "test_loss": ("test_loss", "loss"),
    "test_accuracy": ("test_accuracy", "test_acc", "accuracy", "acc"),
    "test_weighted_f1": (
        "test_weighted_f1",
        "test_wf1",
        "weighted_f1",
        "wf1",
    ),
    "test_macro_f1": ("test_macro_f1", "macro_f1"),
    "test_uar": (
        "test_uar",
        "test_unweighted_recall",
        "uar",
        "unweighted_recall",
    ),
}

PROTOCOL_FIELDS = (
    "model_family",
    "model_variant",
    "causal_mode",
    "future_context_allowed",
    "feature_pkl_path",
    "feature_sha256",
    "feature_causality_status",
    "val_split_strategy",
    "val_session_id",
    "checkpoint_selection_metric",
    "seed",
    "text_dim",
    "audio_dim",
    "visual_dim",
    "num_classes",
    "epochs",
    "batch_size",
    "learning_rate",
    "weight_decay",
    "dropout",
    "context_mode",
    "window_past",
    "window_future",
    "graph_layers",
    "train_batch_cap",
    "val_batch_cap",
    "test_batch_cap",
)

COMPLETENESS_COLUMNS = (
    "model",
    "val_session",
    "run_id",
    "metadata_exists",
    "config_exists",
    "epoch_metrics_exists",
    "epoch_rows",
    "configured_epochs",
    "training_complete",
    "best_checkpoint_exists",
    "last_checkpoint_exists",
    "validation_evaluation_exists",
    "test_evaluation_exists",
    "confusion_matrix_exists",
    "per_class_metrics_exists",
    "eligible_for_summary",
    "notes",
)


class ReviewError(RuntimeError):
    """Base class for expected review failures."""


class MissingRunDirectoriesError(ReviewError):
    """Raised before output creation when configured run directories are absent."""

    def __init__(self, paths: Sequence[Path]) -> None:
        self.paths = list(paths)
        lines = ["Configured run directories are missing:"]
        lines.extend(f"  - {path}" for path in self.paths)
        lines.append("No review outputs were created.")
        super().__init__("\n".join(lines))


@dataclass
class EvaluationArtifact:
    path: Path
    kind: str
    split: Optional[str]
    checkpoint: Optional[str]
    data: Any = None
    load_error: Optional[str] = None


@dataclass
class RunRecord:
    expected_model: str
    expected_session: str
    run_id: str
    run_dir: Path
    metadata: Dict[str, Any] = field(default_factory=dict)
    config: Dict[str, Any] = field(default_factory=dict)
    epoch_metrics: Optional[pd.DataFrame] = None
    artifacts: List[EvaluationArtifact] = field(default_factory=list)
    source_files: List[Path] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    best_validation: Dict[str, Any] = field(default_factory=dict)
    test_result: Dict[str, Any] = field(default_factory=dict)
    completeness: Dict[str, Any] = field(default_factory=dict)
    protocol: Dict[str, Any] = field(default_factory=dict)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit eight causal benchmark runs and one duplicate run."
    )
    parser.add_argument("--config", required=True, help="Review YAML path.")
    parser.add_argument("--runs-root", default=None, help="Override runs root.")
    parser.add_argument("--output-dir", default=None, help="Override CSV output directory.")
    parser.add_argument("--report-path", default=None, help="Override Markdown report path.")
    parser.add_argument("--experiment-date", default=None)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Return non-zero when required evidence is missing or inconsistent.",
    )
    return parser.parse_args()


def resolve_project_path(path_value: Any) -> Path:
    path = Path(str(path_value)).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def project_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def normalize_name(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).strip().lower())


def normalize_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if pd.isna(value) if not isinstance(value, (list, dict, tuple)) else False:
        return None
    return value


def parse_bool(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y", "on"}:
        return True
    if text in {"false", "0", "no", "n", "off"}:
        return False
    return None


def to_float(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def to_optional_int(value: Any) -> Optional[int]:
    numeric = to_float(value)
    if numeric is None or not float(numeric).is_integer():
        return None
    return int(numeric)


def safe_get(mapping: Mapping[str, Any], *paths: Sequence[str]) -> Any:
    for path in paths:
        current: Any = mapping
        found = True
        for key in path:
            if not isinstance(current, Mapping) or key not in current:
                found = False
                break
            current = current[key]
        if found and current is not None:
            return current
    return None


def load_review_config(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise ReviewError(f"Review config does not exist: {path}")
    try:
        with path.open("r", encoding="utf-8") as file:
            config = yaml.safe_load(file)
    except (OSError, yaml.YAMLError) as error:
        raise ReviewError(f"Failed to read review config {path}: {error}") from error
    if not isinstance(config, dict):
        raise ReviewError(f"Review config must contain a YAML mapping: {path}")

    for model_key in ("mmgcn", "multidag"):
        runs = config.get(model_key)
        if not isinstance(runs, dict):
            raise ReviewError(f"Review config key {model_key!r} must be a mapping.")
        missing_sessions = [session for session in EXPECTED_SESSIONS if not runs.get(session)]
        if missing_sessions:
            raise ReviewError(
                f"Review config {model_key!r} is missing sessions: "
                + ", ".join(missing_sessions)
            )

    duplicates = config.get("duplicate_runs", {})
    if not isinstance(duplicates, dict) or not duplicates:
        raise ReviewError("Review config duplicate_runs must be a non-empty mapping.")
    for name, item in duplicates.items():
        if not isinstance(item, dict) or not item.get("run_id") or not item.get("compare_with"):
            raise ReviewError(
                f"duplicate_runs.{name} must define run_id and compare_with."
            )
    return config


def configured_formal_runs(config: Mapping[str, Any]) -> List[Tuple[str, str, str]]:
    rows: List[Tuple[str, str, str]] = []
    for model_key, display_name in (("mmgcn", "MMGCN"), ("multidag", "MultiDAG")):
        for session in EXPECTED_SESSIONS:
            rows.append((display_name, session, str(config[model_key][session])))
    return rows


def configured_all_run_ids(config: Mapping[str, Any]) -> List[str]:
    run_ids = [run_id for _, _, run_id in configured_formal_runs(config)]
    for item in config.get("duplicate_runs", {}).values():
        run_ids.extend((str(item["run_id"]), str(item["compare_with"])))
    return list(dict.fromkeys(run_ids))


def resolve_run_directory(runs_root: Path, run_id: str) -> Path:
    direct = runs_root / run_id
    if direct.is_dir():
        return direct
    output_root = runs_root.parent if runs_root.name == "runs" else runs_root
    return find_run_directory(run_id, output_root)


def ensure_run_directories(config: Mapping[str, Any], runs_root: Path) -> None:
    missing = [
        runs_root / run_id
        for run_id in configured_all_run_ids(config)
        if not _run_directory_exists(runs_root, run_id)
    ]
    if missing:
        raise MissingRunDirectoriesError(missing)


def _run_directory_exists(runs_root: Path, run_id: str) -> bool:
    try:
        resolve_run_directory(runs_root, run_id)
    except (FileNotFoundError, RuntimeError):
        return False
    return True


def load_json_mapping(path: Path, notes: List[str]) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        with path.open("r", encoding="utf-8-sig") as file:
            value = json.load(file)
    except (OSError, json.JSONDecodeError) as error:
        notes.append(f"Failed to parse {project_relative(path)}: {error}")
        return {}
    if not isinstance(value, dict):
        notes.append(f"Expected a mapping in {project_relative(path)}.")
        return {}
    return value


def load_yaml_mapping(path: Path, notes: List[str]) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        with path.open("r", encoding="utf-8-sig") as file:
            value = yaml.safe_load(file)
    except (OSError, yaml.YAMLError) as error:
        notes.append(f"Failed to parse {project_relative(path)}: {error}")
        return {}
    if not isinstance(value, dict):
        notes.append(f"Expected a mapping in {project_relative(path)}.")
        return {}
    return value


def load_run_metadata(run_dir: Path, notes: List[str]) -> Dict[str, Any]:
    return load_json_mapping(run_dir / "run_metadata.json", notes)


def load_experiment_config(run_dir: Path, notes: List[str]) -> Dict[str, Any]:
    return load_yaml_mapping(run_dir / "logs" / "experiment_config.yaml", notes)


def load_epoch_metrics(run_dir: Path, notes: List[str]) -> Optional[pd.DataFrame]:
    path = run_dir / "logs" / "epoch_metrics.csv"
    if not path.is_file():
        return None
    try:
        frame = pd.read_csv(path, encoding="utf-8-sig")
    except (OSError, pd.errors.ParserError, UnicodeError) as error:
        notes.append(f"Failed to parse {project_relative(path)}: {error}")
        return None
    if frame.empty:
        notes.append(f"Epoch metrics file is empty: {project_relative(path)}")
    return frame


def iter_flat_values(value: Any, prefix: Tuple[str, ...] = ()) -> Iterable[Tuple[str, str, Any]]:
    if isinstance(value, pd.DataFrame):
        for record in value.to_dict(orient="records"):
            yield from iter_flat_values(record, prefix)
        return
    if isinstance(value, Mapping):
        for key, child in value.items():
            key_text = str(key)
            child_prefix = prefix + (key_text,)
            if isinstance(child, (Mapping, list, tuple, pd.DataFrame)):
                yield from iter_flat_values(child, child_prefix)
            else:
                yield (".".join(child_prefix), key_text, normalize_scalar(child))
        return
    if isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            yield from iter_flat_values(child, prefix + (str(index),))


def find_data_value(data: Any, aliases: Sequence[str]) -> Any:
    normalized_aliases = [normalize_name(alias) for alias in aliases]
    flattened = list(iter_flat_values(data))
    for alias in normalized_aliases:
        for full_key, leaf_key, value in flattened:
            if normalize_name(leaf_key) == alias or normalize_name(full_key) == alias:
                if value is not None and str(value).strip() != "":
                    return value
    return None


def infer_artifact_split(path: Path, data: Any) -> Optional[str]:
    explicit = find_data_value(data, ("split", "evaluation_split", "dataset_split"))
    if explicit is not None:
        normalized = normalize_name(explicit)
        if normalized.startswith("test"):
            return "test"
        if normalized.startswith("val") or normalized.startswith("validation"):
            return "val"
    path_text = normalize_name(path.as_posix())
    if "test" in path_text:
        return "test"
    if "validation" in path_text or re.search(r"(^|[^a-z])val([^a-z]|$)", path.as_posix().lower()):
        return "val"
    return None


def infer_artifact_checkpoint(path: Path, data: Any) -> Optional[str]:
    value = find_data_value(
        data,
        (
            "checkpoint",
            "checkpoint_path",
            "checkpoint_name",
            "selected_checkpoint",
            "best_checkpoint",
        ),
    )
    if value is not None:
        return str(value)
    normalized = normalize_name(path.as_posix())
    if "bestmodel" in normalized:
        return "best_model.pt"
    if "bestcheckpoint" in normalized:
        return "best_checkpoint"
    return None


def classify_artifact(path: Path) -> str:
    normalized = normalize_name(path.as_posix())
    if "confusionmatrix" in normalized:
        return "confusion_matrix"
    if "perclass" in normalized:
        return "per_class"
    if "prediction" in normalized:
        return "predictions"
    if "metric" in normalized or "summary" in normalized:
        return "metrics"
    return "other"


def load_evaluation_artifact(path: Path) -> Tuple[Any, Optional[str]]:
    try:
        if path.suffix.lower() == ".csv":
            return pd.read_csv(path, encoding="utf-8-sig"), None
        if path.suffix.lower() == ".json":
            with path.open("r", encoding="utf-8-sig") as file:
                return json.load(file), None
        with path.open("r", encoding="utf-8-sig") as file:
            return yaml.safe_load(file), None
    except (OSError, ValueError, yaml.YAMLError, pd.errors.ParserError, UnicodeError) as error:
        return None, str(error)


def discover_evaluation_artifacts(run_dir: Path) -> List[EvaluationArtifact]:
    evaluations_dir = run_dir / "logs" / "evaluations"
    if not evaluations_dir.is_dir():
        return []
    artifacts: List[EvaluationArtifact] = []
    for path in sorted(evaluations_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_ARTIFACT_SUFFIXES:
            continue
        kind = classify_artifact(path)
        # Predictions can be large, while confusion/per-class files are read
        # only after their sibling Test metrics source has been selected.
        if kind in {"predictions", "confusion_matrix", "per_class"}:
            data, load_error = None, None
        else:
            data, load_error = load_evaluation_artifact(path)
        artifacts.append(
            EvaluationArtifact(
                path=path,
                kind=kind,
                split=infer_artifact_split(path, data),
                checkpoint=infer_artifact_checkpoint(path, data),
                data=data,
                load_error=load_error,
            )
        )
    return artifacts


def discover_run_files(run_dir: Path) -> Dict[str, Any]:
    artifacts = discover_evaluation_artifacts(run_dir)
    return {
        "metadata": run_dir / "run_metadata.json",
        "config": run_dir / "logs" / "experiment_config.yaml",
        "epoch_metrics": run_dir / "logs" / "epoch_metrics.csv",
        "best_checkpoint": run_dir / "checkpoints" / "best_model.pt",
        "last_checkpoint": run_dir / "checkpoints" / "last_model.pt",
        "artifacts": artifacts,
    }


def build_run_record(model: str, session: str, run_id: str, runs_root: Path) -> RunRecord:
    record = RunRecord(
        expected_model=model,
        expected_session=session,
        run_id=run_id,
        run_dir=resolve_run_directory(runs_root, run_id),
    )
    files = discover_run_files(record.run_dir)
    record.metadata = load_run_metadata(record.run_dir, record.notes)
    record.config = load_experiment_config(record.run_dir, record.notes)
    record.epoch_metrics = load_epoch_metrics(record.run_dir, record.notes)
    record.artifacts = files["artifacts"]
    for key in (
        "metadata",
        "config",
        "epoch_metrics",
        "best_checkpoint",
        "last_checkpoint",
    ):
        if files[key].is_file():
            record.source_files.append(files[key])
    record.source_files.extend(artifact.path for artifact in record.artifacts)
    for artifact in record.artifacts:
        if artifact.load_error:
            record.notes.append(
                f"Failed to parse {project_relative(artifact.path)}: {artifact.load_error}"
            )
    return record


def find_metric_column(columns: Iterable[Any], metric_name: str) -> Optional[str]:
    aliases = METRIC_ALIASES.get(metric_name, (metric_name,))
    normalized_columns: Dict[str, str] = {}
    for column in columns:
        normalized_columns.setdefault(normalize_name(column), str(column))
    for alias in aliases:
        match = normalized_columns.get(normalize_name(alias))
        if match is not None:
            return match
    return None


def known_tie_policy(model: str, metric_name: str) -> Tuple[str, bool, str]:
    if normalize_name(metric_name) != normalize_name("val_weighted_f1"):
        return UNCONFIRMED, False, ""
    if model == "MMGCN":
        return (
            "earliest_tied_epoch_strict_greater_than",
            True,
            "scripts/train_mmgcn.py:is_better_monitor_value",
        )
    if model == "MultiDAG":
        return (
            "earliest_tied_epoch_strict_greater_than",
            True,
            "scripts/baselines/train_multidag_cl.py:is_better_monitor_value",
        )
    return UNCONFIRMED, False, ""


def empty_best_validation(record: RunRecord, note: str) -> Dict[str, Any]:
    return {
        "model": record.expected_model,
        "val_session": record.expected_session,
        "run_id": record.run_id,
        "epoch": None,
        "train_loss": None,
        "val_loss": None,
        "val_accuracy": None,
        "val_weighted_f1": None,
        "val_macro_f1": None,
        "val_uar": None,
        "learning_rate": None,
        "tied_epochs": "",
        "tie_policy": UNCONFIRMED,
        "tie_policy_confirmed": UNCONFIRMED,
        "notes": note,
    }


def select_best_validation_epoch(record: RunRecord, selection_metric: str) -> Dict[str, Any]:
    frame = record.epoch_metrics
    if frame is None or frame.empty:
        return empty_best_validation(record, "epoch_metrics.csv is missing or empty.")
    metric_column = find_metric_column(frame.columns, selection_metric)
    if metric_column is None:
        return empty_best_validation(
            record,
            f"No column matched selection metric {selection_metric!r}.",
        )
    numeric_metric = pd.to_numeric(frame[metric_column], errors="coerce")
    finite_mask = np.isfinite(numeric_metric.to_numpy(dtype=float, na_value=np.nan))
    valid = frame.loc[finite_mask].copy()
    if valid.empty:
        return empty_best_validation(
            record,
            f"Selection metric column {metric_column!r} has no finite values.",
        )
    valid_values = pd.to_numeric(valid[metric_column], errors="coerce")
    best_value = float(valid_values.max())
    tied = valid.loc[valid_values == best_value].copy()

    epoch_column = find_metric_column(frame.columns, "epoch")
    tied_epochs: List[int] = []
    if epoch_column is not None:
        tied_epochs = [
            int(value)
            for value in pd.to_numeric(tied[epoch_column], errors="coerce").dropna().tolist()
        ]
    tie_policy, policy_confirmed, policy_source = known_tie_policy(
        record.expected_model, selection_metric
    )
    if len(tied) > 1 and not policy_confirmed:
        chosen = tied.iloc[0]
        note = "Multiple best rows exist; tie policy could not be confirmed."
    else:
        if epoch_column is not None:
            tied = tied.assign(
                __epoch=pd.to_numeric(tied[epoch_column], errors="coerce")
            ).sort_values("__epoch", kind="stable")
        chosen = tied.iloc[0]
        note = ""

    result: Dict[str, Any] = {
        "model": record.expected_model,
        "val_session": record.expected_session,
        "run_id": record.run_id,
        "tied_epochs": ";".join(str(epoch) for epoch in tied_epochs),
        "tie_policy": tie_policy,
        "tie_policy_confirmed": policy_confirmed if policy_confirmed else UNCONFIRMED,
        "tie_policy_source": policy_source,
        "notes": note,
    }
    for canonical in (
        "epoch",
        "train_loss",
        "val_loss",
        "val_accuracy",
        "val_weighted_f1",
        "val_macro_f1",
        "val_uar",
        "learning_rate",
    ):
        column = find_metric_column(frame.columns, canonical)
        result[canonical] = to_float(chosen[column]) if column is not None else None
    result["epoch"] = to_optional_int(result["epoch"])
    return result


def select_dataframe_record(frame: pd.DataFrame) -> Tuple[Mapping[str, Any], bool]:
    if frame.empty:
        return {}, False
    working = frame.copy()
    split_column = next(
        (column for column in working.columns if normalize_name(column) in {"split", "evaluationsplit"}),
        None,
    )
    if split_column is not None:
        test_rows = working[
            working[split_column].astype(str).map(normalize_name).str.startswith("test")
        ]
        if not test_rows.empty:
            working = test_rows
    checkpoint_column = next(
        (
            column
            for column in working.columns
            if normalize_name(column)
            in {"checkpoint", "checkpointpath", "checkpointname", "selectedcheckpoint"}
        ),
        None,
    )
    if checkpoint_column is not None:
        best_rows = working[
            working[checkpoint_column].astype(str).map(normalize_name).str.contains("bestmodel")
        ]
        if not best_rows.empty:
            working = best_rows
    return working.iloc[0].to_dict(), len(working) > 1


def extract_metrics_from_artifact(
    artifact: EvaluationArtifact,
) -> Tuple[Dict[str, Optional[float]], bool]:
    data = artifact.data
    ambiguous_rows = False
    if isinstance(data, pd.DataFrame):
        data, ambiguous_rows = select_dataframe_record(data)
    values: Dict[str, Optional[float]] = {}
    for canonical, aliases in TEST_METRIC_ALIASES.items():
        values[canonical] = to_float(find_data_value(data, aliases))
    return values, ambiguous_rows


def artifact_test_score(artifact: EvaluationArtifact) -> int:
    """Rank source evidence without consulting any metric value."""
    score = 0
    normalized_path = normalize_name(artifact.path.as_posix())
    normalized_name_only = normalize_name(artifact.path.name)
    if artifact.split == "test":
        score += 60
    if "test" in normalized_path:
        score += 25
    if artifact.kind == "metrics":
        score += 35
    if "metrics" in normalized_name_only:
        score += 15
    elif "summary" in normalized_name_only:
        score += 10
    checkpoint = normalize_name(artifact.checkpoint or "")
    if "bestmodel" in checkpoint or "bestcheckpoint" in checkpoint:
        score += 70
    if "bestmodel" in normalized_path or "bestcheckpoint" in normalized_path:
        score += 40
    if "testbestmodel" in normalized_path and normalized_name_only.startswith("metrics"):
        score += 100
    if artifact.kind in {"predictions", "confusion_matrix", "per_class"}:
        score -= 100
    return score


def configured_checkpoint_metric(record: RunRecord) -> Any:
    metadata_value = record.metadata.get("checkpoint_selection_metric")
    if metadata_value is not None:
        return metadata_value
    return safe_get(
        record.config,
        ("logging", "monitor_metric"),
        ("training", "select_best_by"),
        ("train", "select_best_by"),
    )


def extract_test_metrics(record: RunRecord, expected_metric: str) -> Dict[str, Any]:
    candidates: List[Tuple[int, EvaluationArtifact, Dict[str, Optional[float]], bool]] = []
    for artifact in record.artifacts:
        if artifact.load_error or artifact.kind not in {"metrics", "other"}:
            continue
        if artifact.split != "test" and "test" not in normalize_name(artifact.path.as_posix()):
            continue
        metrics, ambiguous_rows = extract_metrics_from_artifact(artifact)
        if not any(value is not None for value in metrics.values()):
            continue
        candidates.append((artifact_test_score(artifact), artifact, metrics, ambiguous_rows))

    base: Dict[str, Any] = {
        "model": record.expected_model,
        "val_session": record.expected_session,
        "run_id": record.run_id,
        "selected_checkpoint": (
            "checkpoints/best_model.pt"
            if (record.run_dir / "checkpoints" / "best_model.pt").is_file()
            else UNCONFIRMED
        ),
        "metric_source_file": "",
        "test_loss": None,
        "test_accuracy": None,
        "test_weighted_f1": None,
        "test_macro_f1": None,
        "test_uar": None,
        "source_confirmed_as_validation_selected": UNCONFIRMED,
        "notes": "",
    }
    if not candidates:
        base["notes"] = "No readable Test metric artifact was found."
        return base

    candidates.sort(key=lambda item: (-item[0], project_relative(item[1].path)))
    top_score, artifact, metrics, ambiguous_rows = candidates[0]
    tied_sources = [item for item in candidates if item[0] == top_score]
    base.update(metrics)
    base["metric_source_file"] = project_relative(artifact.path)

    normalized_checkpoint = normalize_name(artifact.checkpoint or artifact.path.as_posix())
    checkpoint_is_best = (
        "bestmodel" in normalized_checkpoint or "bestcheckpoint" in normalized_checkpoint
    )
    split_is_test = artifact.split == "test" or "test" in normalize_name(artifact.path.as_posix())
    selection_matches = normalize_name(configured_checkpoint_metric(record)) == normalize_name(
        expected_metric
    )
    best_exists = (record.run_dir / "checkpoints" / "best_model.pt").is_file()
    ambiguous_source = len(tied_sources) > 1 or ambiguous_rows
    confirmed = (
        checkpoint_is_best
        and split_is_test
        and selection_matches
        and best_exists
        and not ambiguous_source
    )
    if confirmed:
        base["source_confirmed_as_validation_selected"] = True
        base["notes"] = "Source identifies Test evaluation of best_model.pt."
    else:
        reasons: List[str] = []
        if not checkpoint_is_best:
            reasons.append("artifact does not identify best_model.pt")
        if not split_is_test:
            reasons.append("Test split is not identifiable")
        if not selection_matches:
            reasons.append("run selection metric does not match the configured validation metric")
        if not best_exists:
            reasons.append("best_model.pt is missing")
        if len(tied_sources) > 1:
            reasons.append("multiple sources have equal evidence rank")
        if ambiguous_rows:
            reasons.append("the selected artifact contains multiple equally eligible rows")
        base["notes"] = "; ".join(reasons) or UNCONFIRMED
    return base


def configured_epochs(config: Mapping[str, Any]) -> Optional[int]:
    return to_optional_int(
        safe_get(config, ("train", "max_epochs"), ("training", "epochs"))
    )


def identity_from_record(record: RunRecord, metadata_key: str, config_paths: Sequence[Sequence[str]]) -> Any:
    if record.metadata.get(metadata_key) not in (None, ""):
        return record.metadata[metadata_key]
    value = safe_get(record.config, *config_paths)
    return value if value not in (None, "") else UNCONFIRMED


def check_run_completeness(record: RunRecord) -> Dict[str, Any]:
    metadata_path = record.run_dir / "run_metadata.json"
    config_path = record.run_dir / "logs" / "experiment_config.yaml"
    epoch_path = record.run_dir / "logs" / "epoch_metrics.csv"
    best_path = record.run_dir / "checkpoints" / "best_model.pt"
    last_path = record.run_dir / "checkpoints" / "last_model.pt"
    epoch_rows = len(record.epoch_metrics) if record.epoch_metrics is not None else 0
    expected_epochs = configured_epochs(record.config)
    epoch_column = (
        find_metric_column(record.epoch_metrics.columns, "epoch")
        if record.epoch_metrics is not None
        else None
    )
    max_epoch = None
    if record.epoch_metrics is not None and epoch_column is not None:
        epoch_values = pd.to_numeric(record.epoch_metrics[epoch_column], errors="coerce").dropna()
        if not epoch_values.empty:
            max_epoch = int(epoch_values.max())
    training_complete: Any = UNCONFIRMED
    if expected_epochs is not None:
        training_complete = bool(epoch_rows >= expected_epochs and max_epoch is not None and max_epoch >= expected_epochs)

    validation_exists = any(
        artifact.split == "val" and artifact.kind in {"metrics", "other"} and not artifact.load_error
        for artifact in record.artifacts
    )
    test_exists = any(
        artifact.split == "test" and artifact.kind in {"metrics", "other"} and not artifact.load_error
        for artifact in record.artifacts
    )
    confusion_exists = any(
        artifact.kind == "confusion_matrix" and not artifact.load_error
        for artifact in record.artifacts
    )
    per_class_exists = any(
        artifact.kind == "per_class" and not artifact.load_error
        for artifact in record.artifacts
    )
    best_identified = record.best_validation.get("val_weighted_f1") is not None
    test_confirmed = record.test_result.get("source_confirmed_as_validation_selected") is True
    core_values = [
        metadata_path.is_file(),
        config_path.is_file(),
        epoch_path.is_file(),
        training_complete is True,
        best_path.is_file(),
        last_path.is_file(),
        validation_exists,
        test_exists,
        confusion_exists,
        per_class_exists,
        best_identified,
        test_confirmed,
    ]
    notes = list(record.notes)
    if not best_identified:
        notes.append("Best validation Weighted-F1 is unconfirmed.")
    if not test_confirmed:
        notes.append("Test source is not confirmed as validation-selected.")
    return {
        "model": identity_from_record(
            record, "model_family", (("model", "name"),)
        ),
        "val_session": identity_from_record(
            record, "val_session_id", (("dataset", "val_session_id"),)
        ),
        "run_id": record.run_id,
        "metadata_exists": metadata_path.is_file(),
        "config_exists": config_path.is_file(),
        "epoch_metrics_exists": epoch_path.is_file(),
        "epoch_rows": epoch_rows,
        "configured_epochs": expected_epochs,
        "training_complete": training_complete,
        "best_checkpoint_exists": best_path.is_file(),
        "last_checkpoint_exists": last_path.is_file(),
        "validation_evaluation_exists": validation_exists,
        "test_evaluation_exists": test_exists,
        "confusion_matrix_exists": confusion_exists,
        "per_class_metrics_exists": per_class_exists,
        "eligible_for_summary": all(core_values),
        "notes": " | ".join(notes),
    }


def model_family_from_config(config: Mapping[str, Any]) -> Any:
    name = str(safe_get(config, ("model", "name")) or "").strip()
    normalized = normalize_name(name)
    if normalized == "mmgcn":
        return "MMGCN"
    if normalized in {"multidag", "multidagcl", "multidaginspired"}:
        return "MultiDAG"
    return name or UNCONFIRMED


def infer_future_context(config: Mapping[str, Any]) -> Optional[bool]:
    model_family = model_family_from_config(config)
    context_mode = str(safe_get(config, ("graph", "context_mode")) or "").lower()
    if model_family == "MMGCN" and context_mode == "causal":
        return False
    if model_family == "MultiDAG" and context_mode in {"causal", "full", "past_all_causal"}:
        return False
    window_future = safe_get(config, ("graph", "window_future"))
    if window_future is not None:
        numeric = to_float(window_future)
        return None if numeric is None else numeric > 0
    return None


def protocol_value(record: RunRecord, metadata_key: str, *config_paths: Sequence[str]) -> Any:
    if metadata_key in record.metadata and record.metadata[metadata_key] is not None:
        return record.metadata[metadata_key]
    return safe_get(record.config, *config_paths)


def extract_protocol_fields(record: RunRecord) -> Dict[str, Any]:
    config = record.config
    future_context = protocol_value(record, "future_context_allowed")
    if future_context is None:
        future_context = infer_future_context(config)
    causal_mode = protocol_value(record, "causal_mode")
    if causal_mode is None:
        causal_mode = parse_bool(config.get("causal"))
    values = {
        "model_family": protocol_value(record, "model_family") or model_family_from_config(config),
        "model_variant": protocol_value(record, "model_variant") or UNCONFIRMED,
        "causal_mode": causal_mode,
        "future_context_allowed": future_context,
        "feature_pkl_path": protocol_value(
            record, "feature_pkl_path", ("dataset", "feature_pkl_path")
        ),
        "feature_sha256": protocol_value(
            record, "feature_sha256", ("dataset", "feature_sha256")
        ),
        "feature_causality_status": protocol_value(record, "feature_causality_status"),
        "val_split_strategy": protocol_value(
            record, "val_split_strategy", ("dataset", "val_split_strategy")
        ),
        "val_session_id": protocol_value(
            record, "val_session_id", ("dataset", "val_session_id")
        ),
        "checkpoint_selection_metric": protocol_value(
            record,
            "checkpoint_selection_metric",
            ("logging", "monitor_metric"),
            ("training", "select_best_by"),
        ),
        "seed": protocol_value(record, "seed", ("system", "seed")),
        "text_dim": protocol_value(record, "text_dim", ("model", "text_feature_dim")),
        "audio_dim": protocol_value(record, "audio_dim", ("model", "audio_feature_dim")),
        "visual_dim": protocol_value(record, "visual_dim", ("model", "visual_feature_dim")),
        "num_classes": safe_get(config, ("dataset", "num_classes"), ("model", "num_classes")),
        "epochs": safe_get(config, ("train", "max_epochs"), ("training", "epochs")),
        "batch_size": safe_get(config, ("train", "batch_size"), ("training", "batch_size")),
        "learning_rate": safe_get(config, ("train", "learning_rate"), ("training", "lr")),
        "weight_decay": safe_get(config, ("train", "weight_decay"), ("training", "weight_decay")),
        "dropout": safe_get(config, ("model", "dropout")),
        "context_mode": safe_get(config, ("graph", "context_mode")),
        "window_past": safe_get(
            config,
            ("graph", "effective_window_past"),
            ("graph", "window_past"),
            ("model", "window_past"),
        ),
        "window_future": safe_get(config, ("graph", "window_future")),
        "graph_layers": safe_get(
            config, ("graph", "num_layers"), ("model", "num_graph_layers")
        ),
        "train_batch_cap": safe_get(config, ("training", "max_train_batches")),
        "val_batch_cap": safe_get(config, ("training", "max_val_batches")),
        "test_batch_cap": safe_get(
            config,
            ("training", "max_test_batches"),
            ("evaluation", "max_batches"),
        ),
    }
    return values


IGNORED_CONFIG_KEYS = {
    "runid",
    "timestamp",
    "outputdir",
    "generatedat",
    "experimentname",
    "valsessionid",
}


def normalize_path_for_comparison(value: str) -> str:
    text = value.replace("\\", "/")
    return text.rstrip("/").split("/")[-1]


def normalize_config_value(value: Any, key: str = "") -> Any:
    if isinstance(value, Mapping):
        normalized: Dict[str, Any] = {}
        for child_key, child_value in sorted(value.items(), key=lambda item: str(item[0])):
            if normalize_name(child_key) in IGNORED_CONFIG_KEYS:
                continue
            normalized[str(child_key)] = normalize_config_value(child_value, str(child_key))
        return normalized
    if isinstance(value, list):
        return [normalize_config_value(item, key) for item in value]
    if isinstance(value, str) and "path" in normalize_name(key):
        return normalize_path_for_comparison(value)
    return normalize_scalar(value)


def normalize_config_for_comparison(config: Mapping[str, Any]) -> Dict[str, Any]:
    training = config.get("training", config.get("train", {}))
    optimizer = config.get("optimizer", training.get("optimizer", {}) if isinstance(training, Mapping) else {})
    scheduler = config.get("scheduler", training.get("scheduler", {}) if isinstance(training, Mapping) else {})
    loss = config.get("loss", training.get("loss", {}) if isinstance(training, Mapping) else {})
    selected = {
        "dataset": copy.deepcopy(config.get("dataset", {})),
        "model": copy.deepcopy(config.get("model", {})),
        "graph": copy.deepcopy(config.get("graph", {})),
        "training": copy.deepcopy(training),
        "optimizer": copy.deepcopy(optimizer),
        "scheduler": copy.deepcopy(scheduler),
        "loss": copy.deepcopy(loss),
        "seed": safe_get(config, ("system", "seed")),
    }
    return normalize_config_value(selected)


def config_differences(left: Any, right: Any, prefix: str = "") -> List[str]:
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        differences: List[str] = []
        for key in sorted(set(left) | set(right)):
            path = f"{prefix}.{key}" if prefix else str(key)
            if key not in left or key not in right:
                differences.append(path)
            else:
                differences.extend(config_differences(left[key], right[key], path))
        return differences
    if left != right:
        return [prefix or "<root>"]
    return []


def config_uses_test_for_control(config: Mapping[str, Any]) -> Optional[bool]:
    relevant_values: List[Any] = []
    for full_key, _, value in iter_flat_values(config):
        normalized = normalize_name(full_key)
        if any(
            marker in normalized
            for marker in (
                "monitor",
                "selectbestby",
                "checkpointselection",
                "testsplitusedforselection",
            )
        ):
            relevant_values.append(value)
    if not relevant_values:
        return None
    for value in relevant_values:
        if isinstance(value, bool) and value:
            continue
        if "test" in normalize_name(value):
            return True
    explicit = safe_get(config, ("runtime", "test_split_used_for_selection"))
    if explicit is not None:
        parsed = parse_bool(explicit)
        if parsed is not None:
            return parsed
    return False


def status_from_checks(checks: Sequence[Any]) -> Any:
    if any(value is False for value in checks):
        return False
    if any(value == UNCONFIRMED or value is None for value in checks):
        return UNCONFIRMED
    return True


def check_protocol_consistency(
    records: Sequence[RunRecord], review_config: Mapping[str, Any]
) -> List[Dict[str, Any]]:
    extracted = {record.run_id: extract_protocol_fields(record) for record in records}
    family_consistency: Dict[str, Any] = {}
    family_differences: Dict[str, str] = {}
    for family in ("MMGCN", "MultiDAG"):
        family_records = [record for record in records if record.expected_model == family]
        if any(not record.config for record in family_records):
            family_consistency[family] = UNCONFIRMED
            family_differences[family] = "missing experiment_config.yaml"
            continue
        normalized = [normalize_config_for_comparison(record.config) for record in family_records]
        differences: List[str] = []
        for candidate in normalized[1:]:
            differences.extend(config_differences(normalized[0], candidate))
        family_consistency[family] = not differences
        family_differences[family] = ";".join(sorted(set(differences)))

    feature_values = [extracted[record.run_id].get("feature_sha256") for record in records]
    feature_same = (
        len(feature_values) == len(records)
        and all(value not in (None, "") for value in feature_values)
        and len({str(value) for value in feature_values}) == 1
    )
    seed_values = [extracted[record.run_id].get("seed") for record in records]
    seed_same = (
        len(seed_values) == len(records)
        and all(value not in (None, "") for value in seed_values)
        and len({str(value) for value in seed_values}) == 1
    )

    expected_sha = str(review_config.get("expected_feature_sha256", ""))
    expected_strategy = str(review_config.get("expected_val_split_strategy", ""))
    expected_checkpoint_metric = str(review_config.get("expected_checkpoint_metric", ""))
    rows: List[Dict[str, Any]] = []
    for record in records:
        values = extracted[record.run_id]
        test_control = config_uses_test_for_control(record.config)
        test_not_control: Any = UNCONFIRMED if test_control is None else not test_control
        caps = [values.get("train_batch_cap"), values.get("val_batch_cap"), values.get("test_batch_cap")]
        no_batch_cap = all(value in (None, "", 0, "0") for value in caps)
        checks: Dict[str, Any] = {
            "feature_sha_matches_expected": (
                str(values.get("feature_sha256")) == expected_sha
                if values.get("feature_sha256") not in (None, "")
                else UNCONFIRMED
            ),
            "feature_sha_same_across_runs": feature_same,
            "val_split_strategy_matches": (
                str(values.get("val_split_strategy")) == expected_strategy
                if values.get("val_split_strategy") not in (None, "")
                else UNCONFIRMED
            ),
            "val_session_matches_manifest": (
                str(values.get("val_session_id")) == record.expected_session
                if values.get("val_session_id") not in (None, "")
                else UNCONFIRMED
            ),
            "causal_mode_is_true": (
                parse_bool(values.get("causal_mode")) is True
                if values.get("causal_mode") is not None
                else UNCONFIRMED
            ),
            "future_context_disallowed": (
                parse_bool(values.get("future_context_allowed")) is False
                if values.get("future_context_allowed") is not None
                else UNCONFIRMED
            ),
            "checkpoint_metric_matches": (
                normalize_name(values.get("checkpoint_selection_metric"))
                == normalize_name(expected_checkpoint_metric)
                if values.get("checkpoint_selection_metric") not in (None, "")
                else UNCONFIRMED
            ),
            "test_not_used_for_training_control": test_not_control,
            "same_family_config_consistent": family_consistency[record.expected_model],
            "seed_consistent_across_runs": seed_same,
            "no_batch_cap": no_batch_cap,
        }
        row: Dict[str, Any] = {
            "model": record.expected_model,
            "val_session": record.expected_session,
            "run_id": record.run_id,
        }
        row.update({field: values.get(field) for field in PROTOCOL_FIELDS})
        row.update(checks)
        row["config_difference_fields"] = family_differences[record.expected_model]
        row["protocol_consistent"] = status_from_checks(list(checks.values()))
        rows.append(row)
        record.protocol = row
    return rows


def analyze_training_stability(record: RunRecord) -> Dict[str, Any]:
    base: Dict[str, Any] = {
        "model": record.expected_model,
        "val_session": record.expected_session,
        "run_id": record.run_id,
        "best_epoch": record.best_validation.get("epoch"),
        "total_epochs": None,
        "best_val_loss": None,
        "final_val_loss": None,
        "val_loss_rebound": None,
        "best_val_wf1": None,
        "final_val_wf1": None,
        "val_wf1_range": None,
        "contains_nan": UNCONFIRMED,
        "contains_inf": UNCONFIRMED,
        "learning_rate_changed": UNCONFIRMED,
        "notes": "",
    }
    frame = record.epoch_metrics
    if frame is None or frame.empty:
        base["notes"] = "epoch_metrics.csv is missing or empty."
        return base
    epoch_column = find_metric_column(frame.columns, "epoch")
    ordered = frame.copy()
    if epoch_column is not None:
        ordered = ordered.assign(
            __epoch=pd.to_numeric(ordered[epoch_column], errors="coerce")
        ).sort_values("__epoch", kind="stable")
    base["total_epochs"] = len(ordered)
    numeric_columns = [
        column
        for column in ordered.columns
        if any(
            marker in normalize_name(column)
            for marker in (
                "epoch",
                "loss",
                "accuracy",
                "acc",
                "weightedf1",
                "macrof1",
                "uar",
                "learningrate",
                "lr",
                "gradnorm",
            )
        )
        and "monitor" not in normalize_name(column)
        and "earlystop" not in normalize_name(column)
    ]
    numeric_frame = ordered[numeric_columns].apply(pd.to_numeric, errors="coerce")
    numeric_values = numeric_frame.to_numpy(dtype=float)
    base["contains_nan"] = bool(np.isnan(numeric_values).any())
    base["contains_inf"] = bool(np.isinf(numeric_values).any())

    val_loss_column = find_metric_column(frame.columns, "val_loss")
    wf1_column = find_metric_column(frame.columns, "val_weighted_f1")
    lr_column = find_metric_column(frame.columns, "learning_rate")
    notes: List[str] = []
    if val_loss_column is not None:
        values = pd.to_numeric(ordered[val_loss_column], errors="coerce")
        finite = values[np.isfinite(values)]
        if not finite.empty:
            base["best_val_loss"] = float(finite.min())
            base["final_val_loss"] = float(finite.iloc[-1])
            base["val_loss_rebound"] = float(finite.iloc[-1] - finite.min())
    else:
        notes.append("Validation loss column was not recognized.")
    if wf1_column is not None:
        values = pd.to_numeric(ordered[wf1_column], errors="coerce")
        finite = values[np.isfinite(values)]
        if not finite.empty:
            base["best_val_wf1"] = float(finite.max())
            base["final_val_wf1"] = float(finite.iloc[-1])
            base["val_wf1_range"] = float(finite.max() - finite.min())
    else:
        notes.append("Validation Weighted-F1 column was not recognized.")
    if lr_column is not None:
        values = pd.to_numeric(ordered[lr_column], errors="coerce")
        finite = values[np.isfinite(values)]
        base["learning_rate_changed"] = bool(finite.nunique() > 1) if not finite.empty else UNCONFIRMED
    else:
        notes.append("Learning-rate column was not recognized.")
    base["notes"] = " | ".join(notes)
    return base


def metric_vector_for_duplicate(record: RunRecord) -> Optional[np.ndarray]:
    if record.test_result.get("source_confirmed_as_validation_selected") is not True:
        return None
    values = [
        record.best_validation.get("val_weighted_f1"),
        record.test_result.get("test_loss"),
        record.test_result.get("test_accuracy"),
        record.test_result.get("test_weighted_f1"),
        record.test_result.get("test_macro_f1"),
        record.test_result.get("test_uar"),
    ]
    if any(value is None for value in values):
        return None
    return np.asarray(values, dtype=float)


def compare_duplicate_runs(
    review_config: Mapping[str, Any], records_by_id: Mapping[str, RunRecord]
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for name, specification in review_config.get("duplicate_runs", {}).items():
        earlier = records_by_id[str(specification["run_id"])]
        selected = records_by_id[str(specification["compare_with"])]
        earlier_normalized = normalize_config_for_comparison(earlier.config) if earlier.config else None
        selected_normalized = normalize_config_for_comparison(selected.config) if selected.config else None
        differences = (
            config_differences(earlier_normalized, selected_normalized)
            if earlier_normalized is not None and selected_normalized is not None
            else []
        )
        config_equivalent: Any = (
            not differences
            if earlier_normalized is not None and selected_normalized is not None
            else UNCONFIRMED
        )
        earlier_vector = metric_vector_for_duplicate(earlier)
        selected_vector = metric_vector_for_duplicate(selected)
        numerically_identical: Any = UNCONFIRMED
        if earlier_vector is not None and selected_vector is not None:
            numerically_identical = bool(
                np.allclose(earlier_vector, selected_vector, rtol=0.0, atol=1e-12)
            )
        selected_complete = selected.completeness.get("eligible_for_summary") is True
        earlier_complete = earlier.completeness.get("eligible_for_summary") is True
        earlier_protocol = extract_protocol_fields(earlier)
        selected_protocol = extract_protocol_fields(selected)
        if selected_complete:
            decision = "USE_CONFIGURED_NEWER_RUN"
            reason = (
                "The manifest designates the newer run; it is complete, and excluding the "
                "earlier duplicate prevents double-counting Ses01."
            )
        else:
            decision = "NEWER_RUN_INCOMPLETE_DO_NOT_FORCE_SELECTION"
            reason = (
                "The configured newer comparison target is incomplete; it must not be forced "
                "into formal statistics."
            )
        row: Dict[str, Any] = {
            "comparison_name": name,
            "earlier_run_id": earlier.run_id,
            "configured_formal_run_id": selected.run_id,
            "config_equivalent": config_equivalent,
            "config_difference_fields": ";".join(differences),
            "same_seed": (
                str(earlier_protocol.get("seed")) == str(selected_protocol.get("seed"))
                if earlier_protocol.get("seed") is not None and selected_protocol.get("seed") is not None
                else UNCONFIRMED
            ),
            "earlier_epoch_rows": earlier.completeness.get("epoch_rows"),
            "newer_epoch_rows": selected.completeness.get("epoch_rows"),
            "earlier_configured_epochs": earlier.completeness.get("configured_epochs"),
            "newer_configured_epochs": selected.completeness.get("configured_epochs"),
            "earlier_training_complete": earlier.completeness.get("training_complete"),
            "newer_training_complete": selected.completeness.get("training_complete"),
            "earlier_best_checkpoint_exists": earlier.completeness.get("best_checkpoint_exists"),
            "newer_best_checkpoint_exists": selected.completeness.get("best_checkpoint_exists"),
            "earlier_last_checkpoint_exists": earlier.completeness.get("last_checkpoint_exists"),
            "newer_last_checkpoint_exists": selected.completeness.get("last_checkpoint_exists"),
            "earlier_validation_evaluation_exists": earlier.completeness.get("validation_evaluation_exists"),
            "newer_validation_evaluation_exists": selected.completeness.get("validation_evaluation_exists"),
            "earlier_test_evaluation_exists": earlier.completeness.get("test_evaluation_exists"),
            "newer_test_evaluation_exists": selected.completeness.get("test_evaluation_exists"),
            "earlier_best_val_wf1": earlier.best_validation.get("val_weighted_f1"),
            "newer_best_val_wf1": selected.best_validation.get("val_weighted_f1"),
            "earlier_best_epoch": earlier.best_validation.get("epoch"),
            "newer_best_epoch": selected.best_validation.get("epoch"),
            "earlier_test_weighted_f1": earlier.test_result.get("test_weighted_f1"),
            "newer_test_weighted_f1": selected.test_result.get("test_weighted_f1"),
            "earlier_test_loss": earlier.test_result.get("test_loss"),
            "newer_test_loss": selected.test_result.get("test_loss"),
            "earlier_test_accuracy": earlier.test_result.get("test_accuracy"),
            "newer_test_accuracy": selected.test_result.get("test_accuracy"),
            "earlier_test_macro_f1": earlier.test_result.get("test_macro_f1"),
            "newer_test_macro_f1": selected.test_result.get("test_macro_f1"),
            "earlier_test_uar": earlier.test_result.get("test_uar"),
            "newer_test_uar": selected.test_result.get("test_uar"),
            "results_numerically_identical": numerically_identical,
            "earlier_run_complete": earlier_complete,
            "newer_run_complete": selected_complete,
            "selection_decision": decision,
            "selection_reason": reason,
            "notes": "",
        }
        rows.append(row)
    return rows


def compute_group_statistics(
    records: Sequence[RunRecord], std_ddof: int
) -> List[Dict[str, Any]]:
    specifications = [
        ("validation", "accuracy", "val_accuracy"),
        ("validation", "weighted_f1", "val_weighted_f1"),
        ("validation", "macro_f1", "val_macro_f1"),
        ("validation", "uar", "val_uar"),
        ("validation", "loss", "val_loss"),
        ("validation", "best_epoch", "epoch"),
        ("test", "accuracy", "test_accuracy"),
        ("test", "weighted_f1", "test_weighted_f1"),
        ("test", "macro_f1", "test_macro_f1"),
        ("test", "uar", "test_uar"),
        ("test", "loss", "test_loss"),
    ]
    rows: List[Dict[str, Any]] = []
    for model in ("MMGCN", "MultiDAG"):
        group = [record for record in records if record.expected_model == model]
        for split, metric, key in specifications:
            values: List[float] = []
            for record in group:
                if record.completeness.get("eligible_for_summary") is not True:
                    continue
                source = record.best_validation if split == "validation" else record.test_result
                value = to_float(source.get(key))
                if value is not None:
                    values.append(value)
            mean = float(np.mean(values)) if values else None
            std = (
                float(np.std(values, ddof=std_ddof))
                if len(values) > std_ddof
                else None
            )
            rows.append(
                {
                    "model": model,
                    "split": split,
                    "metric": metric,
                    "mean": mean,
                    "std": std,
                    "n": len(values),
                    "expected_n": len(group),
                    "std_ddof": std_ddof,
                    "mean_plus_minus_std": (
                        f"{mean:.6f} ± {std:.6f}"
                        if mean is not None and std is not None
                        else UNCONFIRMED
                    ),
                    "notes": (
                        "All four formal runs contributed."
                        if len(values) == len(group)
                        else "Missing or ineligible values were excluded, not replaced by zero."
                    ),
                }
            )
    return rows


def clean_confusion_label(value: Any) -> str:
    text = str(value)
    return re.sub(r"^(true|pred)_", "", text, flags=re.IGNORECASE)


def confusion_per_class(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, index_col=0, encoding="utf-8-sig")
    if frame.empty or frame.shape[0] != frame.shape[1]:
        raise ValueError("confusion matrix must be non-empty and square")
    matrix = frame.apply(pd.to_numeric, errors="raise").to_numpy(dtype=float)
    labels = [clean_confusion_label(value) for value in frame.index]
    total = float(matrix.sum())
    rows: List[Dict[str, Any]] = []
    for index, label in enumerate(labels):
        tp = matrix[index, index]
        fn = matrix[index, :].sum() - tp
        fp = matrix[:, index].sum() - tp
        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = tp / (tp + fn) if tp + fn > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
        rows.append(
            {
                "label": label,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": matrix[index, :].sum(),
                "total": total,
            }
        )
    return pd.DataFrame(rows)


def analyze_per_class(records: Sequence[RunRecord]) -> List[str]:
    frames: List[pd.DataFrame] = []
    messages: List[str] = []
    for record in records:
        source_text = str(record.test_result.get("metric_source_file", ""))
        if record.test_result.get("source_confirmed_as_validation_selected") is not True or not source_text:
            messages.append(
                f"[UNCONFIRMED] {record.expected_model} {record.expected_session}: "
                "Test source is not confirmed, so no class-level values were summarized."
            )
            continue
        source_path = resolve_project_path(source_text)
        confusion_path = source_path.parent / "confusion_matrix.csv"
        if not confusion_path.is_file():
            available = [
                project_relative(artifact.path)
                for artifact in record.artifacts
                if artifact.kind in {"confusion_matrix", "per_class"}
            ]
            messages.append(
                f"[UNCONFIRMED] {record.expected_model} {record.expected_session}: "
                "no parseable confusion matrix beside the selected Test metrics; sources="
                + (", ".join(available) if available else "none")
            )
            continue
        try:
            class_frame = confusion_per_class(confusion_path)
        except (OSError, ValueError, pd.errors.ParserError) as error:
            messages.append(
                f"[UNCONFIRMED] {record.expected_model} {record.expected_session}: "
                f"could not parse {project_relative(confusion_path)} ({error})."
            )
            continue
        class_frame.insert(0, "val_session", record.expected_session)
        class_frame.insert(0, "model", record.expected_model)
        frames.append(class_frame)

    if not frames:
        return messages or ["[UNCONFIRMED] No class-level artifact could be parsed safely."]
    combined = pd.concat(frames, ignore_index=True)
    summaries: Dict[str, pd.DataFrame] = {}
    for model in ("MMGCN", "MultiDAG"):
        model_frame = combined[combined["model"] == model]
        if model_frame.empty:
            messages.append(f"[UNCONFIRMED] No class-level rows were available for {model}.")
            continue
        summary = model_frame.groupby("label", as_index=False).agg(
            mean_recall=("recall", "mean"),
            mean_f1=("f1", "mean"),
            f1_std=("f1", lambda values: values.std(ddof=1)),
            folds=("val_session", "nunique"),
            exact_zero_recall_folds=("recall", lambda values: int((values == 0.0).sum())),
            exact_zero_f1_folds=("f1", lambda values: int((values == 0.0).sum())),
        )
        summaries[model] = summary
        weakest = summary.sort_values(["mean_f1", "label"]).iloc[0]
        strongest = summary.sort_values(["mean_f1", "label"], ascending=[False, True]).iloc[0]
        variable = summary.dropna(subset=["f1_std"]).sort_values(
            ["f1_std", "label"], ascending=[False, True]
        )
        variable_text = (
            f"{variable.iloc[0]['label']} (F1 sample std={variable.iloc[0]['f1_std']:.6f})"
            if not variable.empty
            else UNCONFIRMED
        )
        persistent_zero = summary[
            (summary["exact_zero_recall_folds"] == summary["folds"])
            | (summary["exact_zero_f1_folds"] == summary["folds"])
        ]["label"].tolist()
        messages.append(
            f"[FACT] {model}: weakest mean-F1 class={weakest['label']} "
            f"({weakest['mean_f1']:.6f}); strongest={strongest['label']} "
            f"({strongest['mean_f1']:.6f}); largest fold variation={variable_text}."
        )
        messages.append(
            f"[FACT] {model}: classes with exactly zero recall or F1 in every available fold="
            + (", ".join(persistent_zero) if persistent_zero else "none")
            + ". No additional near-zero threshold was imposed."
        )
    if set(summaries) == {"MMGCN", "MultiDAG"}:
        merged = summaries["MMGCN"].merge(
            summaries["MultiDAG"], on="label", suffixes=("_mmgcn", "_multidag")
        )
        if not merged.empty:
            merged["f1_delta_multidag_minus_mmgcn"] = (
                merged["mean_f1_multidag"] - merged["mean_f1_mmgcn"]
            )
            largest = merged.loc[
                merged["f1_delta_multidag_minus_mmgcn"].abs().idxmax()
            ]
            messages.append(
                "[INTERPRETATION] The largest class-level mean-F1 difference is for "
                f"{largest['label']}: MultiDAG - MMGCN="
                f"{largest['f1_delta_multidag_minus_mmgcn']:.6f}."
            )
    return messages


def selected_runs_rows(
    review_config: Mapping[str, Any], records_by_id: Mapping[str, RunRecord]
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for model, session, run_id in configured_formal_runs(review_config):
        record = records_by_id[run_id]
        rows.append(
            {
                "model": model,
                "val_session": session,
                "run_id": run_id,
                "role": "formal",
                "included_in_statistics": record.completeness.get("eligible_for_summary") is True,
                "selection_status": (
                    "INCLUDED"
                    if record.completeness.get("eligible_for_summary") is True
                    else "CONFIGURED_FORMAL_BUT_INELIGIBLE"
                ),
                "notes": record.completeness.get("notes", ""),
            }
        )
    for name, specification in review_config.get("duplicate_runs", {}).items():
        rows.append(
            {
                "model": "MultiDAG",
                "val_session": "Ses01",
                "run_id": str(specification["run_id"]),
                "role": "duplicate",
                "included_in_statistics": False,
                "selection_status": "EXCLUDED_DUPLICATE",
                "notes": f"Compared as {name}; formal manifest target={specification['compare_with']}.",
            }
        )
    return rows


def dataframe(rows: Sequence[Mapping[str, Any]], columns: Optional[Sequence[str]] = None) -> pd.DataFrame:
    frame = pd.DataFrame(list(rows))
    if columns is not None:
        for column in columns:
            if column not in frame.columns:
                frame[column] = None
        frame = frame[list(columns)]
    return frame


def write_dataframe(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig", float_format="%.6f")


def write_csv_outputs(
    output_dir: Path,
    frames: Mapping[str, pd.DataFrame],
    source_files: Sequence[Path],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename, frame in frames.items():
        write_dataframe(frame, output_dir / filename)
    unique_sources = sorted({project_relative(path) for path in source_files})
    source_files_path = output_dir / "source_files.txt"
    source_files_path.parent.mkdir(parents=True, exist_ok=True)
    with source_files_path.open("w", encoding="utf-8", newline="\n") as file:
        file.write("\n".join(unique_sources) + "\n")


def format_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if not math.isfinite(value):
            return ""
        return f"{value:.6f}"
    text = str(value).replace("|", "\\|").replace("\n", " ")
    return text


def markdown_table(frame: pd.DataFrame, columns: Sequence[str]) -> str:
    if frame.empty:
        return "_No rows._"
    selected = frame.copy()
    for column in columns:
        if column not in selected.columns:
            selected[column] = None
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for _, row in selected[list(columns)].iterrows():
        lines.append("| " + " | ".join(format_cell(row[column]) for column in columns) + " |")
    return "\n".join(lines)


def find_stat(
    statistics: pd.DataFrame, model: str, split: str, metric: str
) -> Optional[pd.Series]:
    rows = statistics[
        (statistics["model"] == model)
        & (statistics["split"] == split)
        & (statistics["metric"] == metric)
    ]
    return None if rows.empty else rows.iloc[0]


def format_stat(statistics: pd.DataFrame, model: str, split: str, metric: str) -> str:
    row = find_stat(statistics, model, split, metric)
    if row is None or pd.isna(row.get("mean")) or pd.isna(row.get("std")):
        n = 0 if row is None else int(row.get("n", 0))
        return f"{UNCONFIRMED} (n={n})"
    return f"{float(row['mean']):.6f} ± {float(row['std']):.6f} (n={int(row['n'])})"


def validation_stronger_model(statistics: pd.DataFrame) -> Optional[str]:
    rows = {
        model: find_stat(statistics, model, "validation", "weighted_f1")
        for model in ("MMGCN", "MultiDAG")
    }
    if any(row is None or int(row["n"]) < 4 or pd.isna(row["mean"]) for row in rows.values()):
        return None
    mmgcn_mean = float(rows["MMGCN"]["mean"])
    multidag_mean = float(rows["MultiDAG"]["mean"])
    if mmgcn_mean == multidag_mean:
        return "TIE"
    return "MMGCN" if mmgcn_mean > multidag_mean else "MultiDAG"


def stability_interpretation(stability: pd.DataFrame) -> str:
    indicators: Dict[str, Tuple[Optional[float], Optional[float]]] = {}
    for model in ("MMGCN", "MultiDAG"):
        rows = stability[stability["model"] == model]
        rebound = pd.to_numeric(rows["val_loss_rebound"], errors="coerce").dropna()
        wf1_drop = (
            pd.to_numeric(rows["best_val_wf1"], errors="coerce")
            - pd.to_numeric(rows["final_val_wf1"], errors="coerce")
        ).dropna()
        indicators[model] = (
            float(rebound.mean()) if len(rebound) == 4 else None,
            float(wf1_drop.mean()) if len(wf1_drop) == 4 else None,
        )
    if any(value[0] is None or value[1] is None for value in indicators.values()):
        return f"[UNCONFIRMED] Training stability cannot be compared across all four folds."
    mmgcn = indicators["MMGCN"]
    multidag = indicators["MultiDAG"]
    if mmgcn[0] < multidag[0] and mmgcn[1] < multidag[1]:
        favored = "MMGCN"
    elif multidag[0] < mmgcn[0] and multidag[1] < mmgcn[1]:
        favored = "MultiDAG"
    else:
        favored = "mixed"
    return (
        "[INTERPRETATION] Descriptive stability comparison uses smaller mean validation-loss "
        "rebound and smaller mean (best-final) Validation Weighted-F1 drop; "
        f"result={favored}. MMGCN=({mmgcn[0]:.6f}, {mmgcn[1]:.6f}), "
        f"MultiDAG=({multidag[0]:.6f}, {multidag[1]:.6f}). No severity threshold is imposed."
    )


def write_markdown_report(
    report_path: Path,
    review_config_path: Path,
    selected_runs: pd.DataFrame,
    completeness: pd.DataFrame,
    protocol: pd.DataFrame,
    best_validation: pd.DataFrame,
    test_results: pd.DataFrame,
    statistics: pd.DataFrame,
    stability: pd.DataFrame,
    duplicate: pd.DataFrame,
    per_class_messages: Sequence[str],
    source_files: Sequence[Path],
    generated_at: str,
) -> None:
    formal = selected_runs[selected_runs["role"] == "formal"]
    duplicate_rows = selected_runs[selected_runs["role"] == "duplicate"]
    all_complete = bool(len(completeness) == 8 and completeness["eligible_for_summary"].eq(True).all())
    all_protocol = bool(len(protocol) == 8 and protocol["protocol_consistent"].eq(True).all())
    same_family_configs = bool(
        len(protocol) == 8
        and protocol["same_family_config_consistent"].eq(True).all()
    )
    all_test_confirmed = bool(
        len(test_results) == 8
        and test_results["source_confirmed_as_validation_selected"].eq(True).all()
    )
    stronger = validation_stronger_model(statistics)
    readiness = all_complete and all_protocol and all_test_confirmed and stronger not in (None, "TIE")

    lines: List[str] = [
        "# Causal benchmark eight-run review",
        "",
        "## 1. Scope and run selection",
        "",
        "[FACT] The formal manifest contains the following eight session-holdout runs:",
        "",
        markdown_table(formal, ("model", "val_session", "run_id", "selection_status")),
        "",
        "[FACT] Excluded duplicate run(s): "
        + (", ".join(duplicate_rows["run_id"].astype(str)) if not duplicate_rows.empty else "none"),
        "",
        "## 2. Duplicate MultiDAG Ses01 comparison",
        "",
        markdown_table(
            duplicate,
            (
                "earlier_run_id",
                "configured_formal_run_id",
                "config_equivalent",
                "results_numerically_identical",
                "earlier_run_complete",
                "newer_run_complete",
                "selection_decision",
            ),
        ),
        "",
    ]
    for _, row in duplicate.iterrows():
        lines.append(f"[FACT] {row['selection_reason']}")
    lines.extend(
        [
            "",
            "## 3. Run completeness",
            "",
            markdown_table(
                completeness,
                (
                    "model",
                    "val_session",
                    "epoch_rows",
                    "configured_epochs",
                    "training_complete",
                    "best_checkpoint_exists",
                    "last_checkpoint_exists",
                    "validation_evaluation_exists",
                    "test_evaluation_exists",
                    "eligible_for_summary",
                ),
            ),
            "",
            f"[FACT] All eight formal runs are complete and eligible: {all_complete}.",
            "",
            "## 4. Protocol consistency",
            "",
            markdown_table(
                protocol,
                (
                    "model",
                    "val_session",
                    "feature_sha_matches_expected",
                    "val_session_matches_manifest",
                    "causal_mode_is_true",
                    "future_context_disallowed",
                    "checkpoint_metric_matches",
                    "same_family_config_consistent",
                    "seed_consistent_across_runs",
                    "no_batch_cap",
                    "protocol_consistent",
                ),
            ),
            "",
            f"[FACT] All formal protocol checks pass: {all_protocol}.",
            "",
            "## 5. Best validation results",
            "",
            markdown_table(
                best_validation,
                (
                    "model",
                    "val_session",
                    "epoch",
                    "val_accuracy",
                    "val_weighted_f1",
                    "val_macro_f1",
                    "val_uar",
                    "val_loss",
                    "tied_epochs",
                    "tie_policy",
                ),
            ),
            "",
            "[FACT] The best epoch is selected from Validation Weighted-F1 only. Both known "
            "training entrypoints use strict `>` updates, so exact ties retain the earliest epoch.",
            "",
            "## 6. Test results from validation-selected checkpoints",
            "",
            markdown_table(
                test_results,
                (
                    "model",
                    "val_session",
                    "selected_checkpoint",
                    "metric_source_file",
                    "test_accuracy",
                    "test_weighted_f1",
                    "test_macro_f1",
                    "test_uar",
                    "source_confirmed_as_validation_selected",
                ),
            ),
            "",
            f"[FACT] All Test sources are confirmed as validation-selected: {all_test_confirmed}.",
            "",
            "## 7. Four-session aggregate statistics",
            "",
            markdown_table(
                statistics,
                ("model", "split", "metric", "mean", "std", "n", "expected_n"),
            ),
            "",
            f"[FACT] MMGCN Val WF1: {format_stat(statistics, 'MMGCN', 'validation', 'weighted_f1')}.",
            f"[FACT] MultiDAG Val WF1: {format_stat(statistics, 'MultiDAG', 'validation', 'weighted_f1')}.",
            f"[FACT] MMGCN Test WF1: {format_stat(statistics, 'MMGCN', 'test', 'weighted_f1')}.",
            f"[FACT] MultiDAG Test WF1: {format_stat(statistics, 'MultiDAG', 'test', 'weighted_f1')}.",
            "",
            "## 8. Training stability",
            "",
            markdown_table(
                stability,
                (
                    "model",
                    "val_session",
                    "best_epoch",
                    "total_epochs",
                    "val_loss_rebound",
                    "best_val_wf1",
                    "final_val_wf1",
                    "val_wf1_range",
                    "contains_nan",
                    "contains_inf",
                    "learning_rate_changed",
                ),
            ),
            "",
            stability_interpretation(stability),
            "",
            "## 9. Per-class observations",
            "",
        ]
    )
    lines.extend(per_class_messages)
    stronger_text = stronger or UNCONFIRMED
    lines.extend(
        [
            "",
            "## 10. Factual conclusions",
            "",
            f"1. [FACT] Formal runs included by the manifest: {', '.join(formal['run_id'].astype(str))}.",
            "2. [FACT] Excluded duplicate run(s): "
            + (", ".join(duplicate_rows["run_id"].astype(str)) if not duplicate_rows.empty else "none")
            + ".",
            f"3. [FACT] All eight runs complete: {all_complete}.",
            f"4. [FACT] Same-family four-fold configurations consistent: {same_family_configs}.",
            f"5. [FACT] All Test results confirmed from validation-selected checkpoints: {all_test_confirmed}.",
            f"6. [FACT] MMGCN Val WF1 mean ± std: {format_stat(statistics, 'MMGCN', 'validation', 'weighted_f1')}.",
            f"7. [FACT] MultiDAG Val WF1 mean ± std: {format_stat(statistics, 'MultiDAG', 'validation', 'weighted_f1')}.",
            f"8. [FACT] MMGCN Test WF1 mean ± std: {format_stat(statistics, 'MMGCN', 'test', 'weighted_f1')}.",
            f"9. [FACT] MultiDAG Test WF1 mean ± std: {format_stat(statistics, 'MultiDAG', 'test', 'weighted_f1')}.",
            f"10. [INTERPRETATION] Stronger four-session validation Weighted-F1: {stronger_text}.",
            "11. " + stability_interpretation(stability),
            (
                f"12. [INTERPRETATION] The evidence is sufficient to use {stronger} as a provisional "
                "single-seed causal reference, but not as a seed-stable final baseline."
                if readiness
                else "12. [UNCONFIRMED] The current audit is not sufficient to designate a provisional main causal baseline."
            ),
            "13. [FACT] The four folds vary validation session, not random seed; a single shared seed cannot quantify optimization variance. Multi-seed paired reruns remain required.",
            (
                f"14. [INTERPRETATION] Advance both models to paired multi-seed evaluation, prioritizing {stronger} while retaining the other as the controlled comparator."
                if stronger not in (None, "TIE")
                else "14. [UNCONFIRMED] Advance both models to paired multi-seed evaluation; no priority can be assigned from confirmed validation aggregates."
            ),
            "",
            "## 11. Remaining uncertainties",
            "",
        ]
    )
    uncertainty_rows = completeness[completeness["eligible_for_summary"] != True]
    if uncertainty_rows.empty and all_protocol and all_test_confirmed:
        lines.append("[FACT] No artifact-level uncertainty remains in this single-seed audit. Seed variance remains unmeasured.")
    else:
        for _, row in uncertainty_rows.iterrows():
            lines.append(f"[UNCONFIRMED] {row['run_id']}: {row['notes'] or 'ineligible for summary'}")
        if not all_protocol:
            lines.append("[UNCONFIRMED] One or more protocol checks are false or unconfirmed.")
        if not all_test_confirmed:
            lines.append("[UNCONFIRMED] One or more Test sources cannot be tied to best_model.pt.")
    lines.extend(
        [
            "",
            "## 12. Recommended next step",
            "",
            "[INTERPRETATION] Resolve every `UNCONFIRMED` item first. Then run paired multi-seed experiments with identical session folds, feature hash, and validation-only checkpoint selection.",
            "",
            "## Appendix A. Source files",
            "",
        ]
    )
    lines.extend(f"- `{project_relative(path)}`" for path in sorted(set(source_files)))
    lines.extend(
        [
            "",
            "## Appendix B. Generation metadata",
            "",
            f"- Generated at (UTC): `{generated_at}`",
            f"- Review config: `{project_relative(review_config_path)}`",
            "- Selection metric: `val_weighted_f1`",
            "- Test artifacts were ranked without using metric values.",
            "- Missing numeric fields were excluded rather than filled with zero.",
            "",
        ]
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8", newline="\n") as file:
        file.write("\n".join(lines))


def strict_failures(
    formal_records: Sequence[RunRecord], duplicate_rows: Sequence[Mapping[str, Any]]
) -> List[str]:
    failures: List[str] = []
    for record in formal_records:
        if record.completeness.get("eligible_for_summary") is not True:
            failures.append(f"{record.run_id}: run is incomplete or ineligible")
        if record.best_validation.get("val_weighted_f1") is None:
            failures.append(f"{record.run_id}: best validation Weighted-F1 is unconfirmed")
        if record.test_result.get("source_confirmed_as_validation_selected") is not True:
            failures.append(f"{record.run_id}: Test source is not validation-selected")
        if record.protocol.get("protocol_consistent") is not True:
            failures.append(f"{record.run_id}: protocol consistency failed or is unconfirmed")
    for row in duplicate_rows:
        if row.get("newer_run_complete") is not True:
            failures.append(
                f"{row.get('configured_formal_run_id')}: configured newer duplicate target is incomplete"
            )
    return list(dict.fromkeys(failures))


def run_review(args: argparse.Namespace) -> int:
    review_config_path = resolve_project_path(args.config)
    review_config = load_review_config(review_config_path)
    output_root = resolve_project_path(configured_output_root(review_config))
    runs_root = resolve_project_path(
        args.runs_root or review_config.get("runs_root", output_root)
    )
    ensure_run_directories(review_config, runs_root)
    inferred_date = next(
        (
            value
            for value in (
                infer_experiment_date_from_run(
                    resolve_run_directory(runs_root, run_id)
                )
                for run_id in configured_all_run_ids(review_config)
            )
            if value is not None
        ),
        None,
    )
    frozen_date = resolve_experiment_date(
        cli_date=args.experiment_date,
        config=review_config,
        inferred_date=inferred_date,
    )
    output_config = review_config.get("output", {})
    review_name = sanitize_run_name(
        str(output_config.get("name", "causal_benchmark_8run_review"))
        if isinstance(output_config, Mapping)
        else "causal_benchmark_8run_review"
    )
    output_dir = (
        resolve_project_path(args.output_dir)
        if args.output_dir is not None
        else resolve_project_path(review_config["output_dir"])
        if review_config.get("output_dir") is not None
        else resolve_output_category("review", frozen_date, output_root) / review_name
    )
    report_path = (
        resolve_project_path(args.report_path)
        if args.report_path is not None
        else resolve_project_path(review_config["report_path"])
        if review_config.get("report_path") is not None
        else output_dir / "review_report.md"
    )

    formal_records = [
        build_run_record(model, session, run_id, runs_root)
        for model, session, run_id in configured_formal_runs(review_config)
    ]
    records_by_id: Dict[str, RunRecord] = {record.run_id: record for record in formal_records}
    for duplicate_spec in review_config.get("duplicate_runs", {}).values():
        duplicate_id = str(duplicate_spec["run_id"])
        if duplicate_id not in records_by_id:
            records_by_id[duplicate_id] = build_run_record(
                "MultiDAG", "Ses01", duplicate_id, runs_root
            )

    selection_metric = str(review_config.get("selection_metric", "val_weighted_f1"))
    expected_checkpoint_metric = str(
        review_config.get("expected_checkpoint_metric", selection_metric)
    )
    for record in records_by_id.values():
        record.best_validation = select_best_validation_epoch(record, selection_metric)
        record.test_result = extract_test_metrics(record, expected_checkpoint_metric)
        record.completeness = check_run_completeness(record)

    protocol_rows = check_protocol_consistency(formal_records, review_config)
    for record in formal_records:
        if record.protocol.get("protocol_consistent") is not True:
            record.completeness["eligible_for_summary"] = False
            protocol_note = "Protocol consistency is false or unconfirmed."
            existing = record.completeness.get("notes", "")
            record.completeness["notes"] = " | ".join(
                value for value in (existing, protocol_note) if value
            )
    duplicate_rows = compare_duplicate_runs(review_config, records_by_id)

    selected_rows = selected_runs_rows(review_config, records_by_id)
    completeness_rows = [record.completeness for record in formal_records]
    best_validation_rows = [record.best_validation for record in formal_records]
    test_rows = [record.test_result for record in formal_records]
    stability_rows = [analyze_training_stability(record) for record in formal_records]
    statistics_rows = compute_group_statistics(
        formal_records, int(review_config.get("std_ddof", 1))
    )
    per_class_messages = analyze_per_class(formal_records)

    frames = {
        "selected_runs.csv": dataframe(selected_rows),
        "run_completeness.csv": dataframe(completeness_rows, COMPLETENESS_COLUMNS),
        "protocol_consistency.csv": dataframe(protocol_rows),
        "best_validation_results.csv": dataframe(best_validation_rows),
        "test_results.csv": dataframe(test_rows),
        "four_session_statistics.csv": dataframe(statistics_rows),
        "training_stability.csv": dataframe(stability_rows),
        "duplicate_ses01_comparison.csv": dataframe(duplicate_rows),
    }
    source_files = [review_config_path]
    for record in records_by_id.values():
        source_files.extend(record.source_files)
    write_csv_outputs(output_dir, frames, source_files)
    generated_at = datetime.now(timezone.utc).isoformat()
    write_markdown_report(
        report_path=report_path,
        review_config_path=review_config_path,
        selected_runs=frames["selected_runs.csv"],
        completeness=frames["run_completeness.csv"],
        protocol=frames["protocol_consistency.csv"],
        best_validation=frames["best_validation_results.csv"],
        test_results=frames["test_results.csv"],
        statistics=frames["four_session_statistics.csv"],
        stability=frames["training_stability.csv"],
        duplicate=frames["duplicate_ses01_comparison.csv"],
        per_class_messages=per_class_messages,
        source_files=source_files,
        generated_at=generated_at,
    )

    failures = strict_failures(formal_records, duplicate_rows)
    print(f"Review output directory: {project_relative(output_dir)}")
    print(f"Markdown report: {project_relative(report_path)}")
    if args.strict and failures:
        print("Strict review failed:", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        return 3
    if failures:
        print("Review completed with UNCONFIRMED items:")
        for failure in failures:
            print(f"  - {failure}")
    else:
        print("Review completed with all required evidence confirmed.")
    return 0


def main() -> int:
    try:
        return run_review(parse_args())
    except ReviewError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"ERROR: causal benchmark review failed: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

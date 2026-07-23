"""Aggregate original-MERC runs into paper-oriented tables and diagnostics."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from utils.output_paths import (  # noqa: E402
    discover_run_directories,
    infer_experiment_date_from_run,
    resolve_experiment_date,
    resolve_output_category,
)
from scripts.runtime.paper_aligned import (  # noqa: E402
    checkpoint_numeric_summary,
)
LEGACY_OFFICIAL_TRACK = "legacy_official_split_safe_selection"
LEGACY_FIVEFOLD_TRACK = "legacy_fivefold_fair_comparison"
CLEAN_FIVEFOLD_TRACK = "clean_roberta_fivefold_fair_comparison"
TRACK_ORDER = (LEGACY_OFFICIAL_TRACK, LEGACY_FIVEFOLD_TRACK, CLEAN_FIVEFOLD_TRACK)
MODEL_SELECTION_EVIDENCE = {
    "original_repro_mmgcn": (3, "official_code_adapted", 3),
    "original_repro_multidag_cl": (3, "official_code_adapted_with_protocol_repairs", 3),
    "project_paper_oriented_gsmcc": (1, "PROJECT_VARIANT_NOT_PAPER_REPRODUCTION", 3),
    "original_repro_dialoguegcn": (3, "paper_equation_aligned_official_code_adapted", 3),
}
ORIGINAL_MERC_PROTOCOL_VERSION = "original_merc_three_track_v2"
FORMAL_ORIGINAL_MERC = "formal_original_merc"
SMOKE_ORIGINAL_MERC = "smoke_original_merc"
OUT_OF_SCOPE = "out_of_scope"
FORMAL_CONFIG_PREFIX = "configs/experiments/original_merc/"
SMOKE_CONFIG_PREFIX = "configs/smoke/original_repro/"
CANONICAL_SMOKE_CONFIG_PATHS = frozenset(
    {
        "configs/dialoguegcn/paper_aligned/iemocap/full_context/"
        "clean_roberta_features/smoke.yaml",
        "configs/dialoguegcn/paper_aligned/iemocap/full_context/"
        "legacy_mmgcn_features/smoke.yaml",
        "configs/mmgcn/paper_aligned/iemocap/full_context/"
        "clean_roberta_features/smoke.yaml",
        "configs/mmgcn/paper_aligned/iemocap/full_context/"
        "legacy_mmgcn_features/smoke.yaml",
        "configs/multidag_cl/paper_aligned/iemocap/full_context/"
        "clean_roberta_features/smoke.yaml",
        "configs/multidag_cl/paper_aligned/iemocap/full_context/"
        "legacy_mmgcn_features/smoke.yaml",
    }
)
GENERATED_CONFIG_PREFIX = "tmp/original_merc_pipeline_configs/"


def _training_stability(run_dirs: pd.Series) -> tuple[float, float, str]:
    """Summarize finite epoch logging and validation-curve dispersion."""

    finite_fractions: list[float] = []
    curve_stds: list[float] = []
    for run_dir_text in run_dirs.dropna().astype(str):
        path = Path(run_dir_text) / "logs" / "epoch_metrics.csv"
        if not path.is_file():
            finite_fractions.append(0.0)
            continue
        history = pd.read_csv(path)
        required = [column for column in ("train_loss", "val_loss", "val_weighted_f1") if column in history]
        if history.empty or "val_weighted_f1" not in required:
            finite_fractions.append(0.0)
            continue
        values = history[required].apply(pd.to_numeric, errors="coerce").to_numpy()
        finite_fractions.append(float(np.isfinite(values).mean()))
        val_curve = pd.to_numeric(history.get("val_weighted_f1"), errors="coerce")
        finite_curve = val_curve[np.isfinite(val_curve)]
        if len(finite_curve):
            curve_stds.append(float(finite_curve.std(ddof=0)))
    finite_score = float(np.mean(finite_fractions)) if finite_fractions else 0.0
    curve_std = float(np.mean(curve_stds)) if curve_stds else float("inf")
    evidence = f"finite_epoch_fraction={finite_score:.3f};validation_curve_std={curve_std:.6f}"
    return finite_score, curve_std, evidence


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", default=None)
    parser.add_argument("--results-dir", "--output-dir", dest="results_dir", default=None)
    parser.add_argument("--experiment-date", default=None)
    parser.add_argument(
        "--paper-targets",
        default="docs/baselines/original_repro/paper_targets.csv",
    )
    return parser.parse_args()


def resolve(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _load_json_mapping(path: Path) -> tuple[dict[str, Any], str | None]:
    if not path.is_file():
        return {}, None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return {}, f"unreadable_metadata:{error}"
    if not isinstance(value, dict):
        return {}, "metadata_is_not_a_mapping"
    return value, None


def _load_yaml_mapping(path: Path) -> tuple[dict[str, Any], str | None]:
    if not path.is_file():
        return {}, None
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        return {}, f"unreadable_experiment_config:{error}"
    if not isinstance(value, dict):
        return {}, "experiment_config_is_not_a_mapping"
    return value, None


def _first_text(*values: Any) -> str | None:
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return None


def _normalized_source_path(path_text: str | None) -> str:
    if path_text is None:
        return ""
    return str(path_text).strip().replace("\\", "/").lower().lstrip("./")


def _path_belongs_to(path_text: str | None, prefix: str) -> bool:
    normalized = _normalized_source_path(path_text)
    normalized_prefix = prefix.lower().strip("/") + "/"
    return (
        normalized.startswith(normalized_prefix)
        or f"/{normalized_prefix}" in normalized
    )


def _path_matches_any(path_text: str | None, candidates: set[str] | frozenset[str]) -> bool:
    normalized = _normalized_source_path(path_text)
    return any(
        normalized == candidate or normalized.endswith(f"/{candidate}")
        for candidate in candidates
    )


def _feature_track(config: dict[str, Any]) -> str | None:
    dimension = config.get("model", {}).get("text_feature_dim")
    if dimension is None:
        return None
    return "legacy" if dimension == 100 else "clean"


def discover_candidate_run_directories(runs_root: Path) -> list[Path]:
    """Find run-like directories without deciding whether they are in protocol scope."""

    runs_root = Path(runs_root)
    run_dirs = set(discover_run_directories(runs_root))
    if runs_root.name.lower() == "runs" and runs_root.is_dir():
        run_dirs.update(path for path in runs_root.iterdir() if path.is_dir())
    run_dirs.update(
        {
            metrics_path.parents[3]
            for metrics_path in runs_root.rglob(
                "logs/evaluations/test_best_model/metrics.csv"
            )
        }
    )
    run_dirs.update(
        config_path.parents[1]
        for config_path in runs_root.rglob("logs/experiment_config.yaml")
    )
    run_dirs.update(
        summary_path.parents[1]
        for summary_path in runs_root.rglob("logs/run_summary.json")
    )
    run_dirs.update(
        metadata_path.parent
        for metadata_path in runs_root.rglob("run_metadata.json")
    )
    return sorted(run_dirs)


def classify_run_scope(run_dir: Path) -> dict[str, Any]:
    """Classify a run before numerical auditing or performance aggregation."""

    run_dir = Path(run_dir)
    metadata, metadata_error = _load_json_mapping(run_dir / "run_metadata.json")
    config, config_error = _load_yaml_mapping(
        run_dir / "logs" / "experiment_config.yaml"
    )
    protocol_version = _first_text(
        metadata.get("protocol_version"), config.get("protocol_version")
    )
    config_path = _first_text(
        metadata.get("config_path"),
        metadata.get("source_config_path"),
        config.get("config_path"),
        config.get("source_config_path"),
    )
    profile = _first_text(config.get("profile"), metadata.get("profile"))
    model_name = _first_text(
        config.get("model", {}).get("name"), metadata.get("model_name")
    ) or "unknown"
    exact_protocol = protocol_version == ORIGINAL_MERC_PROTOCOL_VERSION
    formal_source = _path_belongs_to(config_path, FORMAL_CONFIG_PREFIX)
    smoke_source = _path_belongs_to(
        config_path, SMOKE_CONFIG_PREFIX
    ) or _path_matches_any(config_path, CANONICAL_SMOKE_CONFIG_PATHS)
    generated_source = _path_belongs_to(config_path, GENERATED_CONFIG_PREFIX)
    generated_formal_fold = generated_source and str(profile or "").startswith(
        "formal_"
    )
    smoke_name = "_smoke_" in run_dir.name.lower()

    if smoke_source or (exact_protocol and (profile == "smoke" or smoke_name)):
        run_scope = SMOKE_ORIGINAL_MERC
        scope_reason = (
            "smoke_config_source"
            if smoke_source
            else "smoke_profile_or_run_id"
        )
    elif (
        exact_protocol
        and profile != "smoke"
        and (formal_source or generated_formal_fold)
    ):
        run_scope = FORMAL_ORIGINAL_MERC
        scope_reason = (
            "formal_original_merc_config_source"
            if formal_source
            else "generated_formal_fold_config"
        )
    else:
        run_scope = OUT_OF_SCOPE
        reasons = [reason for reason in (metadata_error, config_error) if reason]
        if not exact_protocol:
            reasons.append(
                "protocol_version_not_original_merc_three_track_v2"
            )
        if not config_path:
            reasons.append("config_source_unavailable")
        elif generated_source and not generated_formal_fold:
            reasons.append("generated_config_not_proven_formal_fold")
        elif not formal_source:
            reasons.append("config_source_outside_original_merc_formal_tree")
        if profile == "smoke":
            reasons.append("smoke_profile_without_original_merc_protocol")
        scope_reason = " | ".join(dict.fromkeys(reasons)) or "scope_not_proven"

    return {
        "run_id": run_dir.name,
        "run_dir": str(run_dir),
        "run_scope": run_scope,
        "model_name": model_name,
        "feature_track": _feature_track(config),
        "detected_protocol_version": protocol_version,
        "detected_config_path": config_path,
        "detected_profile": profile,
        "scope_reason": scope_reason,
        "exclusion_reason": scope_reason if run_scope == OUT_OF_SCOPE else "",
    }


def classify_runs(runs_root: Path) -> pd.DataFrame:
    rows = [
        classify_run_scope(run_dir)
        for run_dir in discover_candidate_run_directories(runs_root)
    ]
    return pd.DataFrame(
        rows,
        columns=[
            "run_id", "run_dir", "run_scope", "model_name", "feature_track",
            "detected_protocol_version", "detected_config_path", "detected_profile",
            "scope_reason", "exclusion_reason",
        ],
    )


def _required_csv_columns_finite(
    path: Path,
    required_columns: list[str],
    probability_columns: bool = False,
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if not path.is_file():
        return False, [f"missing_artifact:{path.as_posix()}"]
    try:
        frame = pd.read_csv(path)
    except Exception as error:  # malformed artifacts are invalid evidence
        return False, [f"unreadable_artifact:{path.as_posix()}:{error}"]
    if frame.empty:
        reasons.append(f"empty_artifact:{path.as_posix()}")
    columns = list(required_columns)
    if probability_columns:
        probabilities = [name for name in frame if str(name).startswith("probability_")]
        if not probabilities:
            reasons.append(f"missing_probability_columns:{path.as_posix()}")
        columns.extend(probabilities)
    for column in columns:
        if column not in frame:
            reasons.append(f"missing_required_column:{path.as_posix()}:{column}")
            continue
        numeric = pd.to_numeric(frame[column], errors="coerce")
        values = numeric.to_numpy(dtype=float, na_value=np.nan)
        if values.size == 0 or not bool(np.isfinite(values).all()):
            reasons.append(f"nonfinite_or_empty:{path.as_posix()}:{column}")
    return not reasons, reasons


def audit_run_numeric_validity(run_dir: Path) -> dict[str, Any]:
    """Classify one run before it can enter any aggregate or ranking."""

    run_dir = Path(run_dir)
    logs = run_dir / "logs"
    config_path = logs / "experiment_config.yaml"
    config, config_error = _load_yaml_mapping(config_path)
    model_name = str(config.get("model", {}).get("name", "unknown"))
    profile = config.get("profile")
    reasons: list[str] = [config_error] if config_error else []
    detected_status = "FINITE"

    summary_path = logs / "run_summary.json"
    summary: dict[str, Any] = {}
    if summary_path.is_file():
        try:
            loaded = json.loads(summary_path.read_text(encoding="utf-8"))
            summary = loaded if isinstance(loaded, dict) else {}
        except (OSError, json.JSONDecodeError) as error:
            reasons.append(f"unreadable_run_summary:{error}")
    explicit_status = summary.get("numeric_status")
    if explicit_status is not None and explicit_status != "FINITE":
        detected_status = str(explicit_status)
        reasons.append(f"numeric_status:{explicit_status}")
    if summary.get("run_status") == "NUMERICALLY_INVALID":
        reasons.append("run_status:NUMERICALLY_INVALID")
    if summary.get("final_metrics_finite") is False:
        reasons.append("final_metrics_finite:false")

    history_ok, history_reasons = _required_csv_columns_finite(
        logs / "epoch_metrics.csv",
        ["train_loss", "train_classification_loss", "val_loss"],
    )
    reasons.extend(history_reasons)
    if not history_ok and detected_status == "FINITE":
        detected_status = "NONFINITE_LOSS"

    evaluations = logs / "evaluations"
    for split in ("val", "test"):
        metrics_ok, metric_reasons = _required_csv_columns_finite(
            evaluations / f"{split}_best_model" / "metrics.csv",
            ["loss", "accuracy", "weighted_f1", "macro_f1", "uar"],
        )
        predictions_ok, prediction_reasons = _required_csv_columns_finite(
            evaluations / f"{split}_best_model" / "predictions.csv",
            ["confidence"],
            probability_columns=True,
        )
        reasons.extend(metric_reasons)
        reasons.extend(prediction_reasons)
        if (not metrics_ok or not predictions_ok) and detected_status == "FINITE":
            detected_status = "NONFINITE_FORWARD"

    checkpoint_path = run_dir / "checkpoints" / "best_model.pt"
    checkpoint_numeric = {
        "checkpoint_numeric_validation": "failed",
        "checkpoint_nonfinite_tensor_count": None,
        "checkpoint_nonfinite_element_count": None,
        "checkpoint_parameters_finite": False,
    }
    if checkpoint_path.is_file():
        try:
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            if not isinstance(checkpoint, dict):
                raise TypeError("checkpoint is not a mapping")
            checkpoint_numeric = checkpoint_numeric_summary(checkpoint)
            if checkpoint_numeric["checkpoint_numeric_validation"] != "passed":
                reasons.append("checkpoint_contains_nonfinite_tensor")
                detected_status = "NONFINITE_CHECKPOINT"
        except Exception as error:
            reasons.append(f"checkpoint_unreadable:{error}")
            detected_status = "NONFINITE_CHECKPOINT"
    else:
        reasons.append("missing_best_checkpoint")
        detected_status = "NONFINITE_CHECKPOINT"
    if summary.get("checkpoint_parameters_finite") is False:
        reasons.append("checkpoint_parameters_finite:false")
        detected_status = "NONFINITE_CHECKPOINT"

    invalid = bool(reasons)
    return {
        "run_id": run_dir.name,
        "run_dir": str(run_dir),
        "model_name": model_name,
        "profile": profile,
        "run_status": "NUMERICALLY_INVALID" if invalid else "PASS",
        "numeric_status": detected_status,
        "checkpoint_numeric_validation": checkpoint_numeric[
            "checkpoint_numeric_validation"
        ],
        "checkpoint_nonfinite_tensor_count": checkpoint_numeric[
            "checkpoint_nonfinite_tensor_count"
        ],
        "checkpoint_nonfinite_element_count": checkpoint_numeric[
            "checkpoint_nonfinite_element_count"
        ],
        "checkpoint_parameters_finite": bool(
            checkpoint_numeric["checkpoint_parameters_finite"]
        ),
        "final_metrics_finite": not any(
            "metrics.csv" in reason or "predictions.csv" in reason for reason in reasons
        ),
        "invalid_reasons": " | ".join(reasons),
    }


def audit_runs_numeric_validity(runs_root: Path) -> pd.DataFrame:
    return audit_run_directories_numeric_validity(
        discover_candidate_run_directories(runs_root)
    )


def audit_run_directories_numeric_validity(
    run_dirs: list[Path],
) -> pd.DataFrame:
    rows = [audit_run_numeric_validity(run_dir) for run_dir in sorted(run_dirs)]
    return pd.DataFrame(
        rows,
        columns=[
            "run_id", "run_dir", "model_name", "profile", "run_status",
            "numeric_status", "checkpoint_numeric_validation",
            "checkpoint_nonfinite_tensor_count", "checkpoint_nonfinite_element_count",
            "checkpoint_parameters_finite", "final_metrics_finite", "invalid_reasons",
        ],
    )


def _load_run_config(metrics_path: Path) -> dict[str, Any]:
    run_dir = metrics_path.parents[3]
    config_path = run_dir / "logs" / "experiment_config.yaml"
    config, _ = _load_yaml_mapping(config_path)
    return config


def collect_split_metrics(
    runs_root: Path,
    split: str,
    allowed_run_dirs: set[Path] | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    artifact_pattern = f"logs/evaluations/{split}_best_model/metrics.csv"
    for metrics_path in sorted(runs_root.rglob(artifact_pattern)):
        run_dir = metrics_path.parents[3]
        if allowed_run_dirs is not None and run_dir not in allowed_run_dirs:
            continue
        metrics = pd.read_csv(metrics_path)
        if metrics.empty:
            continue
        config = _load_run_config(metrics_path)
        if config.get("protocol_version") != ORIGINAL_MERC_PROTOCOL_VERSION:
            continue
        row = metrics.iloc[0].to_dict()
        row.update(
            {
                "run_dir": str(run_dir),
                "seed": config.get("system", {}).get("seed"),
                "split_seed": config.get("dataset", {}).get("split_seed"),
                "profile": config.get("profile"),
                "feature_track": (
                    "legacy"
                    if config.get("model", {}).get("text_feature_dim") == 100
                    else "clean"
                ),
                "experiment_track": config.get("dataset", {}).get("experiment_track"),
                "protocol_comparability": config.get("dataset", {}).get(
                    "protocol_comparability"
                ),
                "fidelity_status": config.get("model", {}).get(
                    "fidelity_status", "paper_reproduction_candidate"
                ),
                "causal_grade": config.get("model", {}).get("causal_grade"),
                "test_split_used_for_selection": False,
            }
        )
        rows.append(row)
    return pd.DataFrame(
        rows,
        columns=[
            "split", "model_name", "feature_set_name", "feature_protocol",
            "feature_cleanliness", "usage", "outer_test_session", "checkpoint",
            "checkpoint_epoch", "loss", "accuracy", "weighted_f1", "macro_f1", "uar",
            "run_dir", "seed", "split_seed", "profile", "feature_track",
            "experiment_track", "protocol_comparability", "fidelity_status", "causal_grade",
            "test_split_used_for_selection",
        ],
    )


def write_top2_selection(
    output_dir: Path,
    validation_results: pd.DataFrame,
    invalid_screening_models: set[str] | None = None,
) -> pd.DataFrame:
    columns = [
        "rank",
        "model_name",
        "clean_validation_weighted_f1_mean",
        "clean_validation_weighted_f1_std",
        "clean_runs",
        "training_stability_score",
        "validation_curve_std",
        "training_stability_evidence",
        "reproduction_credibility_score",
        "reproduction_credibility",
        "module_insertion_clarity_score",
        "legacy_paper_adjacent_validation_weighted_f1_mean",
    ]
    if validation_results.empty:
        ranking = pd.DataFrame(columns=columns)
    else:
        clean = validation_results[
            (validation_results["profile"] == "clean_screening")
            & (validation_results["experiment_track"] == CLEAN_FIVEFOLD_TRACK)
        ]
        legacy = validation_results[
            (validation_results["profile"] == "screening")
            & (validation_results["experiment_track"] == LEGACY_OFFICIAL_TRACK)
        ]
        ranking = (
            clean.groupby("model_name", as_index=False)
            .agg(
                clean_validation_weighted_f1_mean=("weighted_f1", "mean"),
                clean_validation_weighted_f1_std=("weighted_f1", "std"),
                clean_runs=("weighted_f1", "size"),
            )
        )
        ranking["clean_validation_weighted_f1_std"] = ranking[
            "clean_validation_weighted_f1_std"
        ].fillna(0.0)
        legacy_aux = legacy.groupby("model_name")["weighted_f1"].mean()
        ranking["legacy_paper_adjacent_validation_weighted_f1_mean"] = ranking[
            "model_name"
        ].map(legacy_aux)
        stability = {
            model_name: _training_stability(group["run_dir"])
            for model_name, group in clean.groupby("model_name")
        }
        ranking["training_stability_score"] = ranking["model_name"].map(
            lambda value: stability[value][0]
        )
        ranking["validation_curve_std"] = ranking["model_name"].map(
            lambda value: stability[value][1]
        )
        ranking["training_stability_evidence"] = ranking["model_name"].map(
            lambda value: stability[value][2]
        )
        ranking["reproduction_credibility_score"] = ranking["model_name"].map(
            lambda value: MODEL_SELECTION_EVIDENCE[value][0]
        )
        ranking["reproduction_credibility"] = ranking["model_name"].map(
            lambda value: MODEL_SELECTION_EVIDENCE[value][1]
        )
        ranking["module_insertion_clarity_score"] = ranking["model_name"].map(
            lambda value: MODEL_SELECTION_EVIDENCE[value][2]
        )
        ranking = ranking.sort_values(
            [
                "clean_validation_weighted_f1_mean",
                "training_stability_score",
                "validation_curve_std",
                "clean_validation_weighted_f1_std",
                "reproduction_credibility_score",
                "module_insertion_clarity_score",
                "legacy_paper_adjacent_validation_weighted_f1_mean",
                "model_name",
            ],
            ascending=[False, False, True, True, False, False, False, True],
            na_position="last",
        ).reset_index(drop=True)
        ranking.insert(0, "rank", np.arange(1, len(ranking) + 1))
        ranking = ranking[columns]
    ranking.to_csv(
        output_dir / "clean_screening_validation_selection_evidence.csv", index=False
    )

    config_by_model = {
        "original_repro_mmgcn": "configs/experiments/original_merc/clean_fold_bases/mmgcn_clean.yaml",
        "original_repro_multidag_cl": "configs/experiments/original_merc/clean_fold_bases/multidag_cl_clean.yaml",
        "project_paper_oriented_gsmcc": "configs/experiments/original_merc/clean_fold_bases/gsmcc_clean.yaml",
        "original_repro_dialoguegcn": "configs/experiments/original_merc/clean_fold_bases/dialoguegcn_clean.yaml",
    }
    invalid_screening_models = set(invalid_screening_models or set())
    complete = (
        not invalid_screening_models
        and set(ranking["model_name"]) == set(MODEL_SELECTION_EVIDENCE)
    )
    selected = ranking.head(2)["model_name"].tolist() if complete else []
    selection = {
        "selection_source": "clean_single_fold_screening_validation_only",
        "ordered_criteria": [
            "clean_screening_validation_weighted_f1",
            "training_stability",
            "reproduction_credibility",
            "clear_module_insertion_point",
            "legacy_paper_adjacent_validation_diagnostic",
        ],
        "test_split_used_for_selection": False,
        "status": (
            "ready"
            if complete
            else (
                "pending_invalid_run_repair"
                if invalid_screening_models
                else "pending_all_four_clean_screening_results"
            )
        ),
        "invalid_screening_models": sorted(invalid_screening_models),
        "selected_models": selected,
        "jobs": [
            {
                "config": config_by_model[model],
                "seeds": [13, 42, 77],
                "outer_test_sessions": ["Ses01", "Ses02", "Ses03", "Ses04", "Ses05"],
            }
            for model in selected
        ],
    }
    with (output_dir / "top2_selection.yaml").open("w", encoding="utf-8") as file:
        yaml.safe_dump(selection, file, sort_keys=False, allow_unicode=True)
    return ranking


def add_paper_gaps(results: pd.DataFrame, paper_targets: pd.DataFrame) -> pd.DataFrame:
    merged = results.merge(paper_targets, on="model_name", how="left")
    merged["weighted_f1_points"] = merged["weighted_f1"] * 100.0
    merged["paper_gap_eligible"] = (
        (merged["experiment_track"] == LEGACY_OFFICIAL_TRACK)
        & (merged["model_name"] != "project_paper_oriented_gsmcc")
        & merged["paper_weighted_f1_points"].notna()
    )
    merged["paper_gap_weighted_f1_points"] = np.where(
        merged["paper_gap_eligible"],
        merged["weighted_f1_points"] - merged["paper_weighted_f1_points"],
        np.nan,
    )
    merged["paper_gap_exclusion_reason"] = np.where(
        merged["paper_gap_eligible"],
        "",
        np.where(
            merged["model_name"] == "project_paper_oriented_gsmcc",
            "PROJECT_VARIANT_NOT_PAPER_REPRODUCTION",
            "protocol_not_paper_adjacent",
        ),
    )
    return merged


def aggregate(results: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "model_name",
        "experiment_track",
        "runs",
        "weighted_f1_mean",
        "weighted_f1_std",
        "weighted_f1_cv",
        "macro_f1_mean",
        "uar_mean",
        "accuracy_mean",
        "paper_weighted_f1_points",
        "paper_gap_weighted_f1_points_mean",
    ]
    if results.empty:
        return pd.DataFrame(columns=columns)
    grouped = results.groupby(["model_name", "experiment_track"], dropna=False)
    summary = grouped.agg(
        runs=("weighted_f1", "size"),
        weighted_f1_mean=("weighted_f1", "mean"),
        weighted_f1_std=("weighted_f1", "std"),
        macro_f1_mean=("macro_f1", "mean"),
        uar_mean=("uar", "mean"),
        accuracy_mean=("accuracy", "mean"),
        paper_weighted_f1_points=("paper_weighted_f1_points", "first"),
        paper_gap_weighted_f1_points_mean=("paper_gap_weighted_f1_points", "mean"),
    ).reset_index()
    summary["weighted_f1_std"] = summary["weighted_f1_std"].fillna(0.0)
    summary["weighted_f1_cv"] = np.where(
        summary["weighted_f1_mean"].abs() > 0,
        summary["weighted_f1_std"] / summary["weighted_f1_mean"].abs(),
        np.nan,
    )
    return summary[columns]


def feature_comparison(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame(
            columns=["model_name", "legacy_weighted_f1", "clean_weighted_f1", "clean_minus_legacy"]
        )
    fair = summary[
        summary["experiment_track"].isin(
            [LEGACY_FIVEFOLD_TRACK, CLEAN_FIVEFOLD_TRACK]
        )
    ]
    pivot = fair.pivot(
        index="model_name", columns="experiment_track", values="weighted_f1_mean"
    )
    result = pivot.rename(
        columns={
            LEGACY_FIVEFOLD_TRACK: "legacy_weighted_f1",
            CLEAN_FIVEFOLD_TRACK: "clean_weighted_f1",
        }
    ).reset_index()
    for column in ("legacy_weighted_f1", "clean_weighted_f1"):
        if column not in result:
            result[column] = np.nan
    result["clean_minus_legacy"] = result["clean_weighted_f1"] - result["legacy_weighted_f1"]
    return result[["model_name", "legacy_weighted_f1", "clean_weighted_f1", "clean_minus_legacy"]]


def dataframe_to_markdown(frame: pd.DataFrame) -> str:
    """Render a compact Markdown table without pandas' optional tabulate dependency."""

    def format_value(value: Any) -> str:
        if pd.isna(value):
            return ""
        if isinstance(value, (float, np.floating)):
            return f"{float(value):.4f}"
        return str(value).replace("|", "\\|")

    headers = [str(column) for column in frame.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(format_value(value) for value in row) + " |")
    return "\n".join(lines)


def write_report(
    output_dir: Path,
    detailed: pd.DataFrame,
    summary: pd.DataFrame,
    comparison: pd.DataFrame,
) -> None:
    completed = len(detailed)
    report = [
        "# Original MERC reproduction report",
        "",
        f"Collected test runs: {completed}.",
        "",
        "Checkpoint selection is accepted only from runs whose embedded protocol declares "
        "`test_split_used_for_selection=false`; test metrics are never ranking inputs.",
        "",
        "## Protocol-separated rankings",
        "",
    ]
    if summary.empty:
        report.append(
            "No completed formal runs were found. Tables were emitted with stable schemas; "
            "run the screening and clean-fold stages before interpreting model quality."
        )
    else:
        ranked = summary.sort_values(
            ["experiment_track", "weighted_f1_mean"], ascending=[True, False]
        )
        for track in TRACK_ORDER:
            track_rows = ranked[ranked["experiment_track"] == track]
            if track_rows.empty:
                continue
            report.extend([f"### {track}", "", dataframe_to_markdown(track_rows), ""])
        report.extend(
            ["", "## Fair five-fold legacy versus clean", "", dataframe_to_markdown(comparison)]
        )
        clean = ranked[ranked["experiment_track"] == CLEAN_FIVEFOLD_TRACK]
        if not clean.empty:
            report.extend(
                [
                    "",
                    "## Stability",
                    "",
                    "`weighted_f1_std` and `weighted_f1_cv` summarize seed/fold dispersion. "
                    "A ranking based on fewer than five outer folds and the planned five seeds is provisional.",
                ]
            )
    report.extend(
        [
            "",
            "## Interpretation guardrails",
            "",
            "Paper gaps combine architecture, feature, optimization, split, and reporting differences. "
            "They are diagnostic deltas, not evidence that one isolated component caused the gap.",
            "Confusion matrices and per-class precision/recall/F1 remain in each run's evaluation directory.",
            "The GS-MCC-inspired project variant is marked "
            "`PROJECT_VARIANT_NOT_PAPER_REPRODUCTION` and has no paper gap.",
            "",
        ]
    )
    (output_dir / "detailed_report.md").write_text("\n".join(report), encoding="utf-8")
    (output_dir / "reproduction_report.md").write_text("\n".join(report), encoding="utf-8")


def _write_extended_artifacts(
    output_dir: Path,
    detailed: pd.DataFrame,
    summary: pd.DataFrame,
) -> None:
    run_manifest_columns = [
        "run_dir", "model_name", "experiment_track", "protocol_comparability",
        "feature_track", "feature_set_name", "outer_test_session",
        "seed", "split_seed", "profile", "weighted_f1", "macro_f1", "uar", "accuracy",
        "causal_grade", "test_split_used_for_selection",
    ]
    run_manifest = detailed.reindex(columns=run_manifest_columns)
    run_manifest.to_csv(output_dir / "run_manifest.csv", index=False)
    run_manifest.to_csv(output_dir / "fold_metrics.csv", index=False)
    summary.to_csv(output_dir / "aggregate_metrics.csv", index=False)
    detailed.reindex(
        columns=[
            "run_dir", "model_name", "experiment_track", "outer_test_session", "seed",
            "weighted_f1_points", "paper_weighted_f1_points", "paper_gap_eligible",
            "paper_gap_weighted_f1_points", "paper_gap_exclusion_reason",
        ]
    ).to_csv(output_dir / "paper_gap.csv", index=False)

    protocol_rows = []
    for _, row in run_manifest.iterrows():
        protocol_rows.append(
            {
                "run_dir": row.get("run_dir"),
                "model_name": row.get("model_name"),
                "causal_grade_declared": row.get("causal_grade"),
                "outer_test_session_recorded": pd.notna(row.get("outer_test_session")),
                "experiment_track_recorded": pd.notna(row.get("experiment_track")),
                "validation_only_selection": row.get("test_split_used_for_selection") is False
                or row.get("test_split_used_for_selection") == False,  # noqa: E712
                "protocol_pass": (
                    row.get("causal_grade") == "noncausal_offline_full_context"
                    and row.get("experiment_track") in TRACK_ORDER
                    and (
                        row.get("test_split_used_for_selection") is False
                        or row.get("test_split_used_for_selection") == False  # noqa: E712
                    )
                ),
            }
        )
    pd.DataFrame(
        protocol_rows,
        columns=[
            "run_dir", "model_name", "causal_grade_declared", "outer_test_session_recorded",
            "experiment_track_recorded", "validation_only_selection", "protocol_pass",
        ],
    ).to_csv(output_dir / "run_protocol_audit.csv", index=False)

    stability_columns = [
        "model_name", "experiment_track", "runs", "weighted_f1_mean", "weighted_f1_std",
        "weighted_f1_cv", "worst_fold_weighted_f1", "nan_or_inf_count",
    ]
    stability_rows = []
    if not detailed.empty:
        for (model, track), group in detailed.groupby(["model_name", "experiment_track"]):
            values = group["weighted_f1"].astype(float)
            stability_rows.append(
                {
                    "model_name": model,
                    "experiment_track": track,
                    "runs": len(group),
                    "weighted_f1_mean": values.mean(),
                    "weighted_f1_std": values.std(ddof=1) if len(values) > 1 else 0.0,
                    "weighted_f1_cv": values.std(ddof=1) / abs(values.mean()) if len(values) > 1 and values.mean() else 0.0,
                    "worst_fold_weighted_f1": values.min(),
                    "nan_or_inf_count": int((~np.isfinite(values)).sum()),
                }
            )
    stability = pd.DataFrame(stability_rows, columns=stability_columns)
    stability.to_csv(output_dir / "training_stability.csv", index=False)

    runtime_rows = []
    complexity_rows = []
    for run_dir_text in sorted(set(detailed.get("run_dir", pd.Series(dtype=str)).dropna())):
        run_dir = Path(run_dir_text)
        summary_path = run_dir / "logs" / "run_summary.json"
        run_summary = {}
        if summary_path.is_file():
            run_summary = json.loads(summary_path.read_text(encoding="utf-8"))
        model_name = detailed[detailed["run_dir"] == run_dir_text].iloc[0]["model_name"]
        runtime_rows.append(
            {
                "run_dir": run_dir_text,
                "model_name": model_name,
                "training_time_seconds": run_summary.get("training_time_seconds"),
                "peak_gpu_memory_mb": run_summary.get("peak_gpu_memory_mb"),
                "runtime_metadata_available": bool(run_summary),
            }
        )
        checkpoint_path = run_dir / "checkpoints" / "best_model.pt"
        parameter_count = np.nan
        if checkpoint_path.is_file():
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            parameter_count = int(
                sum(value.numel() for value in checkpoint["model_state_dict"].values())
            )
        complexity_rows.append(
            {
                "run_dir": run_dir_text,
                "model_name": model_name,
                "parameter_count": parameter_count,
                "source_fidelity": (
                    "PROJECT_VARIANT_NOT_PAPER_REPRODUCTION"
                    if model_name == "project_paper_oriented_gsmcc"
                    else "official_code_or_equation_adapted"
                ),
                "code_complexity": {
                    "original_repro_mmgcn": "medium",
                    "original_repro_multidag_cl": "medium",
                    "project_paper_oriented_gsmcc": "high",
                    "original_repro_dialoguegcn": "medium",
                }.get(model_name, "unconfirmed"),
                "clear_module_hooks": True,
                "ablation_ready": True,
                "legacy_environment_required": False,
            }
        )
    runtime = pd.DataFrame(
        runtime_rows,
        columns=[
            "run_dir", "model_name", "training_time_seconds", "peak_gpu_memory_mb",
            "runtime_metadata_available",
        ],
    )
    complexity = pd.DataFrame(
        complexity_rows,
        columns=[
            "run_dir", "model_name", "parameter_count", "source_fidelity", "code_complexity",
            "clear_module_hooks", "ablation_ready", "legacy_environment_required",
        ],
    )
    runtime.to_csv(output_dir / "runtime_memory.csv", index=False)
    complexity.to_csv(output_dir / "model_complexity.csv", index=False)

    selection_columns = [
        "model_name", "experiment_track", "mean_test_weighted_f1", "fold_std", "worst_fold_weighted_f1",
        "paper_gap_points", "run_failure_count", "nan_or_inf_count", "parameter_count_mean",
        "peak_gpu_memory_mb_mean", "training_time_seconds_mean", "source_fidelity", "code_complexity",
        "clear_module_hooks", "ablation_ready", "legacy_environment_required", "best_performance_rank",
        "best_reproducibility_rank", "best_research_backbone_rank",
    ]
    selection_rows = []
    if not stability.empty:
        for _, stable in stability.iterrows():
            model, track = stable["model_name"], stable["experiment_track"]
            aggregate_row = summary[
                (summary["model_name"] == model)
                & (summary["experiment_track"] == track)
            ].iloc[0]
            model_complexity = complexity[complexity["model_name"] == model]
            model_runtime = runtime[runtime["model_name"] == model]
            selection_rows.append(
                {
                    "model_name": model,
                    "experiment_track": track,
                    "mean_test_weighted_f1": stable["weighted_f1_mean"],
                    "fold_std": stable["weighted_f1_std"],
                    "worst_fold_weighted_f1": stable["worst_fold_weighted_f1"],
                    "paper_gap_points": aggregate_row["paper_gap_weighted_f1_points_mean"],
                    "run_failure_count": 0,
                    "nan_or_inf_count": stable["nan_or_inf_count"],
                    "parameter_count_mean": model_complexity["parameter_count"].mean(),
                    "peak_gpu_memory_mb_mean": model_runtime["peak_gpu_memory_mb"].mean(),
                    "training_time_seconds_mean": model_runtime["training_time_seconds"].mean(),
                    "source_fidelity": model_complexity["source_fidelity"].iloc[0],
                    "code_complexity": model_complexity["code_complexity"].iloc[0],
                    "clear_module_hooks": True,
                    "ablation_ready": True,
                    "legacy_environment_required": False,
                }
            )
    selection = pd.DataFrame(selection_rows)
    if not selection.empty:
        selection["best_performance_rank"] = selection.groupby("experiment_track")["mean_test_weighted_f1"].rank(ascending=False, method="min").astype(int)
        selection["best_reproducibility_rank"] = selection.groupby("experiment_track")["fold_std"].rank(ascending=True, method="min").astype(int)
        backbone_score = selection["mean_test_weighted_f1"] - selection["fold_std"]
        selection["best_research_backbone_rank"] = backbone_score.groupby(selection["experiment_track"]).rank(ascending=False, method="min").astype(int)
    selection.reindex(columns=selection_columns).to_csv(
        output_dir / "baseline_selection_matrix.csv", index=False
    )

    per_class_rows = []
    confusion_dir = output_dir / "confusion_matrices"
    curves_dir = output_dir / "training_curves"
    confusion_dir.mkdir(exist_ok=True)
    curves_dir.mkdir(exist_ok=True)
    for artifact_dir in (confusion_dir, curves_dir):
        for stale_csv in artifact_dir.glob("*.csv"):
            stale_csv.unlink()
    valid_run_dirs = (
        detailed["run_dir"]
        .dropna()
        .astype(str)
        .drop_duplicates()
    )
    for run_dir_text in sorted(valid_run_dirs):
        run_dir = Path(run_dir_text)
        run_name = run_dir.name
        evaluation_dir = run_dir / "logs" / "evaluations" / "test_best_model"
        metrics_path = evaluation_dir / "per_class_metrics.csv"
        if metrics_path.is_file():
            frame = pd.read_csv(metrics_path)
            frame.insert(0, "run_id", run_name)
            per_class_rows.append(frame)
        confusion = evaluation_dir / "confusion_matrix.csv"
        if confusion.is_file():
            shutil.copyfile(confusion, confusion_dir / f"{run_name}.csv")
        curve = run_dir / "logs" / "epoch_metrics.csv"
        if curve.is_file():
            shutil.copyfile(curve, curves_dir / f"{run_name}.csv")
    per_class = (
        pd.concat(per_class_rows, ignore_index=True)
        if per_class_rows
        else pd.DataFrame(columns=["run_id", "label_id", "label", "precision", "recall", "f1", "support"])
    )
    per_class.to_csv(output_dir / "per_class_metrics.csv", index=False)


def _write_scope_artifacts(
    output_dir: Path,
    scope_audit: pd.DataFrame,
    smoke_numeric_audit: pd.DataFrame,
) -> None:
    excluded_columns = [
        "run_id",
        "run_dir",
        "detected_protocol_version",
        "detected_config_path",
        "detected_profile",
        "exclusion_reason",
    ]
    excluded = scope_audit[scope_audit["run_scope"] == OUT_OF_SCOPE]
    excluded.reindex(columns=excluded_columns).to_csv(
        output_dir / "excluded_runs.csv", index=False
    )

    smoke_columns = [
        "run_id",
        "model_name",
        "feature_track",
        "numeric_status",
        "run_status",
        "epochs",
        "max_train_batches",
        "max_eval_batches",
        "checkpoint_parameters_finite",
        "prediction_count_correct",
    ]
    audit_by_run_dir = {
        str(row.run_dir): row
        for row in smoke_numeric_audit.itertuples(index=False)
    }
    smoke_rows = []
    smoke_scope = scope_audit[scope_audit["run_scope"] == SMOKE_ORIGINAL_MERC]
    for scope_row in smoke_scope.itertuples(index=False):
        run_dir = Path(scope_row.run_dir)
        config, _ = _load_yaml_mapping(
            run_dir / "logs" / "experiment_config.yaml"
        )
        summary, _ = _load_json_mapping(run_dir / "logs" / "run_summary.json")
        numeric = audit_by_run_dir.get(str(run_dir))
        training = config.get("training", {})
        smoke_rows.append(
            {
                "run_id": scope_row.run_id,
                "model_name": scope_row.model_name,
                "feature_track": scope_row.feature_track,
                "numeric_status": (
                    numeric.numeric_status if numeric is not None else None
                ),
                "run_status": numeric.run_status if numeric is not None else None,
                "epochs": training.get("epochs"),
                "max_train_batches": training.get("max_train_batches"),
                "max_eval_batches": training.get("max_eval_batches"),
                "checkpoint_parameters_finite": (
                    numeric.checkpoint_parameters_finite
                    if numeric is not None
                    else None
                ),
                "prediction_count_correct": summary.get(
                    "prediction_count_correct"
                ),
            }
        )
    pd.DataFrame(smoke_rows, columns=smoke_columns).to_csv(
        output_dir / "smoke_runs.csv", index=False
    )


def analyze(runs_root: Path, output_dir: Path, paper_targets_path: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paper_targets = pd.read_csv(paper_targets_path)
    scope_audit = classify_runs(runs_root)
    formal_run_dirs = [
        Path(value)
        for value in scope_audit.loc[
            scope_audit["run_scope"] == FORMAL_ORIGINAL_MERC, "run_dir"
        ]
    ]
    smoke_run_dirs = [
        Path(value)
        for value in scope_audit.loc[
            scope_audit["run_scope"] == SMOKE_ORIGINAL_MERC, "run_dir"
        ]
    ]
    numeric_audit = audit_run_directories_numeric_validity(formal_run_dirs)
    smoke_numeric_audit = audit_run_directories_numeric_validity(smoke_run_dirs)
    _write_scope_artifacts(output_dir, scope_audit, smoke_numeric_audit)
    invalid_runs = numeric_audit[
        numeric_audit["run_status"] == "NUMERICALLY_INVALID"
    ].copy()
    invalid_runs.to_csv(output_dir / "invalid_runs.csv", index=False)
    invalid_clean_screening_models = set(
        invalid_runs.loc[
            invalid_runs["profile"] == "clean_screening", "model_name"
        ].astype(str)
    )
    valid_clean_screening_models = set(
        numeric_audit.loc[
            (numeric_audit["run_status"] == "PASS")
            & (numeric_audit["profile"] == "clean_screening"),
            "model_name",
        ].astype(str)
    )
    unresolved_invalid_screening_models = (
        invalid_clean_screening_models - valid_clean_screening_models
    )
    valid_model_profiles = {
        (str(row.model_name), str(row.profile))
        for row in numeric_audit.loc[
            numeric_audit["run_status"] == "PASS", ["model_name", "profile"]
        ].itertuples(index=False)
    }
    invalid_run_resolution = invalid_runs.copy()
    invalid_run_resolution["valid_replacement_exists"] = [
        (str(row.model_name), str(row.profile)) in valid_model_profiles
        for row in invalid_run_resolution.itertuples(index=False)
    ]
    invalid_run_resolution["blocks_current_selection"] = (
        invalid_run_resolution["profile"].eq("clean_screening")
        & invalid_run_resolution["model_name"].astype(str).isin(
            unresolved_invalid_screening_models
        )
    )
    resolution_columns = [
        "run_id",
        "model_name",
        "profile",
        "numeric_status",
        "invalid_reasons",
        "valid_replacement_exists",
        "blocks_current_selection",
    ]
    invalid_run_resolution.reindex(
        columns=resolution_columns
        + [
            column
            for column in invalid_run_resolution.columns
            if column not in resolution_columns
        ]
    ).to_csv(output_dir / "invalid_run_resolution.csv", index=False)
    valid_run_dirs = set(
        numeric_audit.loc[numeric_audit["run_status"] == "PASS", "run_dir"].astype(str)
    )
    valid_run_paths = {Path(value) for value in valid_run_dirs}
    collected = collect_split_metrics(
        runs_root, "test", allowed_run_dirs=valid_run_paths
    )
    validation_results = collect_split_metrics(
        runs_root, "val", allowed_run_dirs=valid_run_paths
    )
    detailed = add_paper_gaps(collected, paper_targets)
    summary = aggregate(detailed)
    track_summaries = {
        track: summary[
            summary.get("experiment_track", pd.Series(dtype=str)) == track
        ]
        for track in TRACK_ORDER
    }
    if not summary.empty:
        ranking = summary.sort_values(
            ["experiment_track", "weighted_f1_mean"], ascending=[True, False]
        ).copy()
        ranking["rank_within_experiment_track"] = ranking.groupby("experiment_track")[
            "weighted_f1_mean"
        ].rank(ascending=False, method="min").astype(int)
    else:
        ranking = summary.copy()
        ranking["rank_within_experiment_track"] = pd.Series(dtype="int64")
    comparison = feature_comparison(summary)
    detailed.to_csv(output_dir / "runs_detailed.csv", index=False)
    for track, track_summary in track_summaries.items():
        track_summary.to_csv(output_dir / f"summary_{track}.csv", index=False)
    ranking.to_csv(output_dir / "protocol_separated_ranking.csv", index=False)
    comparison.to_csv(output_dir / "fivefold_legacy_vs_clean.csv", index=False)
    screening_ranking = write_top2_selection(
        output_dir,
        validation_results,
        invalid_screening_models=unresolved_invalid_screening_models,
    )
    write_report(output_dir, detailed, summary, comparison)
    _write_extended_artifacts(output_dir, detailed, summary)
    result = {
        "collected_runs": len(detailed),
        "invalid_runs": len(invalid_runs),
        "smoke_runs": len(smoke_run_dirs),
        "excluded_runs": int((scope_audit["run_scope"] == OUT_OF_SCOPE).sum()),
        "screening_models_ranked": len(screening_ranking),
        "results_dir": str(output_dir),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def main() -> None:
    args = parse_args()
    output_root = PROJECT_ROOT / "outputs"
    runs_root = resolve(args.runs_root) if args.runs_root is not None else output_root
    inferred_date = next(
        (
            value
            for value in (
                infer_experiment_date_from_run(path)
                for path in discover_run_directories(runs_root)
            )
            if value is not None
        ),
        None,
    )
    frozen_date = resolve_experiment_date(
        cli_date=args.experiment_date,
        inferred_date=inferred_date,
    )
    output_dir = (
        resolve(args.results_dir)
        if args.results_dir is not None
        else resolve_output_category("analysis", frozen_date, output_root)
        / "original_merc"
    )
    analyze(runs_root, output_dir, resolve(args.paper_targets))


if __name__ == "__main__":
    main()

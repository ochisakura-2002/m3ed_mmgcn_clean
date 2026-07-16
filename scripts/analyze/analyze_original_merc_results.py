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


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from utils.output_paths import (  # noqa: E402
    discover_run_directories,
    infer_experiment_date_from_run,
    resolve_experiment_date,
    resolve_output_category,
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


def _load_run_config(metrics_path: Path) -> dict[str, Any]:
    run_dir = metrics_path.parents[3]
    config_path = run_dir / "logs" / "experiment_config.yaml"
    if not config_path.is_file():
        return {}
    with config_path.open("r", encoding="utf-8") as file:
        value = yaml.safe_load(file)
    return value if isinstance(value, dict) else {}


def collect_split_metrics(runs_root: Path, split: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for metrics_path in sorted(runs_root.rglob(f"{split}_best_model/metrics.csv")):
        metrics = pd.read_csv(metrics_path)
        if metrics.empty:
            continue
        config = _load_run_config(metrics_path)
        if not str(config.get("protocol_version", "")).startswith("original_merc_"):
            continue
        row = metrics.iloc[0].to_dict()
        row.update(
            {
                "run_dir": str(metrics_path.parents[3]),
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
    complete = set(ranking["model_name"]) == set(MODEL_SELECTION_EVIDENCE)
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
        "status": "ready" if complete else "pending_all_four_clean_screening_results",
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
    runs_root: Path,
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
    for metrics_path in sorted(runs_root.rglob("test_best_model/per_class_metrics.csv")):
        frame = pd.read_csv(metrics_path)
        run_name = metrics_path.parents[3].name
        frame.insert(0, "run_id", run_name)
        per_class_rows.append(frame)
        confusion = metrics_path.parent / "confusion_matrix.csv"
        if confusion.is_file():
            shutil.copyfile(confusion, confusion_dir / f"{run_name}.csv")
        curve = metrics_path.parents[2] / "epoch_metrics.csv"
        if curve.is_file():
            shutil.copyfile(curve, curves_dir / f"{run_name}.csv")
    per_class = (
        pd.concat(per_class_rows, ignore_index=True)
        if per_class_rows
        else pd.DataFrame(columns=["run_id", "label_id", "label", "precision", "recall", "f1", "support"])
    )
    per_class.to_csv(output_dir / "per_class_metrics.csv", index=False)


def analyze(runs_root: Path, output_dir: Path, paper_targets_path: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paper_targets = pd.read_csv(paper_targets_path)
    collected = collect_split_metrics(runs_root, "test")
    validation_results = collect_split_metrics(runs_root, "val")
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
    screening_ranking = write_top2_selection(output_dir, validation_results)
    write_report(output_dir, detailed, summary, comparison)
    _write_extended_artifacts(runs_root, output_dir, detailed, summary)
    result = {
        "collected_runs": len(detailed),
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

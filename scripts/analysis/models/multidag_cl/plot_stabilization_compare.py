"""Compare MultiDAG+CL stabilization runs from saved CSV logs.

The script reads only lightweight logs under a discovered dated or legacy run:
experiment_config.yaml, epoch_metrics.csv, validation/test metrics.csv, and an
optional single-run diagnosis CSV. It writes summary tables, figures, and a
short Markdown report for one stabilization comparison.
"""

from __future__ import annotations

import argparse
import math
import os
import tempfile
from pathlib import Path
import sys
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[4]
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

OUTPUT_ROOT = PROJECT_ROOT / "outputs"
BEST_SELECTION_METRIC = "val_weighted_f1"

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "m3ed_mmgcn_matplotlib"),
)
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SUMMARY_COLUMNS = [
    "run_id",
    "display_name",
    "group",
    "changed_factor",
    "experiment_name",
    "context_mode",
    "window_past",
    "active_modalities",
    "encoder",
    "graph_layers",
    "dropout",
    "lr",
    "weight_decay",
    "epochs",
    "best_epoch_by_val_weighted_f1",
    "best_val_acc",
    "best_val_weighted_f1",
    "best_val_macro_f1",
    "best_val_uar",
    "final_train_loss",
    "final_val_loss",
    "test_acc",
    "test_weighted_f1",
    "test_macro_f1",
    "test_uar",
]

TEST_METRICS = [
    "test_acc",
    "test_weighted_f1",
    "test_macro_f1",
    "test_uar",
]

DELTA_METRICS = [
    "test_acc",
    "test_weighted_f1",
    "test_macro_f1",
    "test_uar",
    "best_val_weighted_f1",
    "final_val_loss",
]

HEATMAP_METRICS = [
    "test_acc",
    "test_weighted_f1",
    "test_macro_f1",
    "test_uar",
    "best_val_weighted_f1",
    "final_val_loss",
]

LOWER_IS_BETTER = {"final_val_loss"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build MultiDAG+CL stabilization multi-run analysis."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to stabilization comparison YAML config.",
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--experiment-date", default=None)
    return parser.parse_args()


def resolve_path(path_text: str) -> Path:
    path = Path(str(path_text))
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def project_relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT.resolve()))
    except ValueError:
        return str(path)


def load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"YAML config not found: {path}")
    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise TypeError(f"Expected YAML dict: {path}")
    return data


def read_csv_required(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required CSV: {path}")
    df = pd.read_csv(path)
    if len(df) == 0:
        raise RuntimeError(f"Empty CSV: {path}")
    return df


def read_csv_optional(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if len(df) == 0:
        return None
    return df


def safe_get(config: Dict[str, Any], keys: Iterable[str], default: Any = "") -> Any:
    current: Any = config
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    if current is None:
        return default
    return current


def first_config_value(
    config: Dict[str, Any],
    paths: Sequence[Sequence[str]],
    default: Any = "",
) -> Any:
    for path in paths:
        value = safe_get(config, path, None)
        if value is not None and value != "":
            return value
    return default


def to_float(value: Any) -> Any:
    if value is None or pd.isna(value):
        return pd.NA
    try:
        return float(value)
    except (TypeError, ValueError):
        return pd.NA


def metric_value(row: Optional[pd.Series], names: Sequence[str]) -> Any:
    if row is None:
        return pd.NA
    for name in names:
        if name in row.index:
            return to_float(row[name])
    return pd.NA


def get_first_row(df: Optional[pd.DataFrame]) -> Optional[pd.Series]:
    if df is None or len(df) == 0:
        return None
    return df.iloc[0]


def get_final_row(epoch_df: pd.DataFrame) -> pd.Series:
    if "epoch" in epoch_df.columns:
        return epoch_df.sort_values("epoch").iloc[-1]
    return epoch_df.iloc[-1]


def get_best_val_row(epoch_df: pd.DataFrame) -> pd.Series:
    if BEST_SELECTION_METRIC not in epoch_df.columns:
        raise ValueError(
            f"Missing required column '{BEST_SELECTION_METRIC}' in epoch metrics."
        )
    valid_df = epoch_df.dropna(subset=[BEST_SELECTION_METRIC])
    if len(valid_df) == 0:
        raise RuntimeError(f"No valid values for {BEST_SELECTION_METRIC}.")
    return valid_df.loc[valid_df[BEST_SELECTION_METRIC].idxmax()]


def normalize_modalities(value: Any) -> str:
    mapping = {
        "text": "T",
        "t": "T",
        "audio": "A",
        "a": "A",
        "visual": "V",
        "video": "V",
        "v": "V",
    }
    if isinstance(value, (list, tuple)):
        labels = []
        for item in value:
            text = str(item).strip()
            labels.append(mapping.get(text.lower(), text))
        joined = "".join(labels)
        if set(joined).issubset({"T", "A", "V"}):
            return joined
        return "+".join(labels)
    if value is None:
        return ""
    return str(value)


def normalize_optional_diagnosis(
    diagnosis_df: Optional[pd.DataFrame],
) -> Dict[str, Any]:
    result: Dict[str, Any] = {"diagnosis_available": False}
    if diagnosis_df is None or len(diagnosis_df) == 0:
        return result

    row = diagnosis_df.iloc[0]
    result["diagnosis_available"] = True
    for column in [
        "behavior_label",
        "train_loss_relative_drop",
        "val_loss_worsens_after_best",
        "val_weighted_f1_worsens_after_best",
    ]:
        if column in row.index:
            result[f"diagnosis_{column}"] = row[column]
    return result


def validate_required_run_files(run_dir: Path) -> Dict[str, Path]:
    paths = {
        "config": run_dir / "logs" / "experiment_config.yaml",
        "epoch": run_dir / "logs" / "epoch_metrics.csv",
        "val_metrics": run_dir
        / "logs"
        / "evaluations"
        / "val_best_model"
        / "metrics.csv",
        "test_metrics": run_dir
        / "logs"
        / "evaluations"
        / "test_best_model"
        / "metrics.csv",
        "diagnosis": run_dir
        / "figures"
        / "diagnostics"
        / "multidag_cl_training_diagnosis.csv",
    }
    missing = [
        path
        for key, path in paths.items()
        if key != "diagnosis" and not path.exists()
    ]
    if missing:
        missing_text = "\n".join(f"  - {project_relative(path)}" for path in missing)
        raise FileNotFoundError(
            f"Run is missing required analysis files: {run_dir.name}\n"
            f"{missing_text}"
        )
    return paths


def build_summary_row(
    run_item: Dict[str, Any],
    run_dir: Path,
    config: Dict[str, Any],
    epoch_df: pd.DataFrame,
    val_eval_df: pd.DataFrame,
    test_eval_df: pd.DataFrame,
    diagnosis_df: Optional[pd.DataFrame],
) -> Dict[str, Any]:
    best_row = get_best_val_row(epoch_df)
    final_row = get_final_row(epoch_df)
    val_eval_row = get_first_row(val_eval_df)
    test_eval_row = get_first_row(test_eval_df)

    model_config = config.get("model", {})
    if not isinstance(model_config, dict):
        model_config = {}

    epochs_from_config = first_config_value(
        config,
        [
            ["training", "epochs"],
            ["train", "max_epochs"],
            ["training", "max_epochs"],
            ["train", "epochs"],
        ],
        default="",
    )

    row: Dict[str, Any] = {
        "run_id": str(run_item["run_id"]),
        "display_name": str(run_item.get("display_name", run_item["run_id"])),
        "group": str(run_item.get("group", "")),
        "changed_factor": str(run_item.get("changed_factor", "")),
        "experiment_name": first_config_value(
            config,
            [["project", "experiment_name"], ["experiment_name"]],
        ),
        "context_mode": first_config_value(
            config,
            [["graph", "context_mode"], ["model", "context_mode"]],
        ),
        "window_past": first_config_value(
            config,
            [["graph", "window_past"], ["model", "window_past"]],
        ),
        "active_modalities": normalize_modalities(
            first_config_value(
                config,
                [["model", "active_modalities"], ["data", "active_modalities"]],
            )
        ),
        "encoder": first_config_value(
            config,
            [
                ["model", "modality_encoder_type"],
                ["model", "encoder_type"],
                ["model", "encoder"],
            ],
        ),
        "graph_layers": first_config_value(
            config,
            [["model", "num_graph_layers"], ["graph", "num_layers"]],
        ),
        "dropout": first_config_value(config, [["model", "dropout"]]),
        "lr": first_config_value(
            config,
            [["training", "lr"], ["train", "learning_rate"], ["optimizer", "lr"]],
        ),
        "weight_decay": first_config_value(
            config,
            [
                ["training", "weight_decay"],
                ["train", "weight_decay"],
                ["optimizer", "weight_decay"],
            ],
        ),
        "epochs": epochs_from_config if epochs_from_config != "" else int(len(epoch_df)),
        "best_epoch_by_val_weighted_f1": int(best_row["epoch"])
        if "epoch" in best_row.index and not pd.isna(best_row["epoch"])
        else pd.NA,
        "best_val_acc": metric_value(best_row, ["val_acc"]),
        "best_val_weighted_f1": metric_value(best_row, ["val_weighted_f1"]),
        "best_val_macro_f1": metric_value(best_row, ["val_macro_f1"]),
        "best_val_uar": metric_value(best_row, ["val_uar"]),
        "final_train_loss": metric_value(final_row, ["train_loss"]),
        "final_val_loss": metric_value(final_row, ["val_loss"]),
        "test_acc": metric_value(test_eval_row, ["acc", "accuracy"]),
        "test_weighted_f1": metric_value(test_eval_row, ["weighted_f1"]),
        "test_macro_f1": metric_value(test_eval_row, ["macro_f1"]),
        "test_uar": metric_value(test_eval_row, ["uar"]),
        "run_dir": project_relative(run_dir),
        "val_best_model_acc": metric_value(val_eval_row, ["acc", "accuracy"]),
        "val_best_model_weighted_f1": metric_value(val_eval_row, ["weighted_f1"]),
        "val_best_model_macro_f1": metric_value(val_eval_row, ["macro_f1"]),
        "val_best_model_uar": metric_value(val_eval_row, ["uar"]),
        "final_epoch": metric_value(final_row, ["epoch"]),
    }
    row.update(normalize_optional_diagnosis(diagnosis_df))
    return row


def load_run_bundle(
    run_item: Dict[str, Any],
    runs_dir: Path,
) -> Dict[str, Any]:
    run_id = str(run_item["run_id"])
    run_dir = find_run_directory(run_id, runs_dir)
    paths = validate_required_run_files(run_dir)
    config = load_yaml(paths["config"])
    epoch_df = read_csv_required(paths["epoch"])
    val_eval_df = read_csv_required(paths["val_metrics"])
    test_eval_df = read_csv_required(paths["test_metrics"])
    diagnosis_df = read_csv_optional(paths["diagnosis"])
    row = build_summary_row(
        run_item=run_item,
        run_dir=run_dir,
        config=config,
        epoch_df=epoch_df,
        val_eval_df=val_eval_df,
        test_eval_df=test_eval_df,
        diagnosis_df=diagnosis_df,
    )
    return {
        "run_id": run_id,
        "run_item": run_item,
        "run_dir": run_dir,
        "epoch_df": epoch_df,
        "summary_row": row,
    }


def get_runs(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    runs = config.get("runs", [])
    if not isinstance(runs, list) or not runs:
        raise ValueError("Config field 'runs' must be a non-empty list.")
    normalized = []
    for item in runs:
        if not isinstance(item, dict) or "run_id" not in item:
            raise ValueError("Each run item must be a dict with 'run_id'.")
        normalized.append(item)
    return normalized


def get_plot_config(config: Dict[str, Any]) -> Dict[str, Any]:
    plots = config.get("plots", {})
    if not isinstance(plots, dict):
        plots = {}
    return {
        "save_png": bool(plots.get("save_png", True)),
        "save_pdf": bool(plots.get("save_pdf", True)),
        "dpi": int(plots.get("dpi", 300)),
    }


def save_dataframe(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print("Saved:", path)


def save_figure(fig: plt.Figure, stem_path: Path, plot_config: Dict[str, Any]) -> None:
    stem_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    if plot_config["save_png"]:
        png_path = stem_path.with_suffix(".png")
        fig.savefig(png_path, dpi=plot_config["dpi"])
        print("  [PNG]", png_path)
    if plot_config["save_pdf"]:
        pdf_path = stem_path.with_suffix(".pdf")
        fig.savefig(pdf_path)
        print("  [PDF]", pdf_path)
    plt.close(fig)


def as_float_series(df: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(df[column], errors="coerce")


def build_delta_table(
    summary_df: pd.DataFrame,
    baseline_run_id: str,
) -> pd.DataFrame:
    baseline_rows = summary_df[summary_df["run_id"] == baseline_run_id]
    if len(baseline_rows) == 0:
        raise ValueError(f"baseline_run_id not found in summary: {baseline_run_id}")
    baseline = baseline_rows.iloc[0]

    columns = ["run_id", "display_name", "group", "changed_factor"]
    delta_df = summary_df[columns].copy()
    for metric in DELTA_METRICS:
        baseline_value = to_float(baseline.get(metric))
        values = pd.to_numeric(summary_df[metric], errors="coerce")
        if pd.isna(baseline_value):
            delta_df[f"delta_{metric}"] = pd.NA
        else:
            delta_df[f"delta_{metric}"] = values - float(baseline_value)
    return delta_df


def build_ranking_table(
    summary_df: pd.DataFrame,
    primary_metric: str,
) -> pd.DataFrame:
    if primary_metric not in summary_df.columns:
        raise ValueError(f"Primary metric not found in summary: {primary_metric}")
    ranking_df = summary_df.copy()
    ranking_df[primary_metric] = pd.to_numeric(
        ranking_df[primary_metric],
        errors="coerce",
    )
    ranking_df = ranking_df.sort_values(
        by=primary_metric,
        ascending=False,
        na_position="last",
    ).reset_index(drop=True)
    ranking_df.insert(0, "rank", np.arange(1, len(ranking_df) + 1))
    return ranking_df


def label_list(summary_df: pd.DataFrame) -> List[str]:
    return [str(value) for value in summary_df["display_name"].tolist()]


def plot_final_test_metrics_bar(
    summary_df: pd.DataFrame,
    figures_dir: Path,
    plot_config: Dict[str, Any],
) -> None:
    plot_df = summary_df.copy()
    labels = label_list(plot_df)
    x = np.arange(len(labels))
    width = 0.8 / len(TEST_METRICS)

    fig, ax = plt.subplots(figsize=(max(10, len(labels) * 1.35), 5.5))
    for idx, metric in enumerate(TEST_METRICS):
        values = as_float_series(plot_df, metric).to_numpy()
        offsets = x - 0.4 + width / 2 + idx * width
        ax.bar(offsets, values, width=width, label=metric.replace("test_", ""))

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    ax.set_title("Final test metrics")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    save_figure(fig, figures_dir / "final_test_metrics_bar", plot_config)


def plot_test_weighted_f1_ranking(
    ranking_df: pd.DataFrame,
    baseline_run_id: str,
    figures_dir: Path,
    plot_config: Dict[str, Any],
) -> None:
    plot_df = ranking_df.sort_values("test_weighted_f1", ascending=True).copy()
    labels = label_list(plot_df)
    values = as_float_series(plot_df, "test_weighted_f1")
    colors = [
        "#4f6bed" if run_id != baseline_run_id else "#d14f4f"
        for run_id in plot_df["run_id"].tolist()
    ]

    baseline_value = values[plot_df["run_id"] == baseline_run_id]

    fig, ax = plt.subplots(figsize=(9, max(5, len(labels) * 0.45)))
    ax.barh(labels, values, color=colors)
    if len(baseline_value) > 0 and not pd.isna(baseline_value.iloc[0]):
        ax.axvline(
            float(baseline_value.iloc[0]),
            color="#d14f4f",
            linestyle="--",
            linewidth=1.5,
            label="original_w5 baseline",
        )
        ax.legend(loc="lower right")
    ax.set_xlabel("Test Weighted-F1")
    ax.set_xlim(0, 1)
    ax.set_title("Test Weighted-F1 ranking")
    ax.grid(axis="x", alpha=0.25)
    save_figure(fig, figures_dir / "test_weighted_f1_ranking", plot_config)


def plot_delta_from_baseline(
    delta_df: pd.DataFrame,
    baseline_run_id: str,
    figures_dir: Path,
    plot_config: Dict[str, Any],
) -> None:
    metrics = [
        "delta_test_weighted_f1",
        "delta_test_acc",
        "delta_test_macro_f1",
        "delta_test_uar",
    ]
    plot_df = delta_df[delta_df["run_id"] != baseline_run_id].copy()
    labels = label_list(plot_df)
    x = np.arange(len(labels))
    width = 0.8 / len(metrics)

    fig, ax = plt.subplots(figsize=(max(10, len(labels) * 1.35), 5.5))
    for idx, metric in enumerate(metrics):
        values = pd.to_numeric(plot_df[metric], errors="coerce").to_numpy()
        offsets = x - 0.4 + width / 2 + idx * width
        ax.bar(offsets, values, width=width, label=metric.replace("delta_test_", ""))

    ax.axhline(0.0, color="black", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("Delta from original_w5")
    ax.set_title("Delta from original w5 baseline")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    save_figure(fig, figures_dir / "delta_from_baseline", plot_config)


def plot_val_weighted_f1_curves(
    bundles: Sequence[Dict[str, Any]],
    baseline_run_id: str,
    figures_dir: Path,
    plot_config: Dict[str, Any],
) -> None:
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for bundle in bundles:
        epoch_df = bundle["epoch_df"].sort_values("epoch")
        if "val_weighted_f1" not in epoch_df.columns:
            continue
        item = bundle["run_item"]
        run_id = bundle["run_id"]
        linewidth = 2.6 if run_id == baseline_run_id else 1.7
        ax.plot(
            epoch_df["epoch"],
            epoch_df["val_weighted_f1"],
            marker="o",
            markersize=3.5,
            linewidth=linewidth,
            label=str(item.get("display_name", run_id)),
        )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation Weighted-F1")
    ax.set_ylim(0, 1)
    ax.set_title("Validation Weighted-F1 curves")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.25)
    save_figure(fig, figures_dir / "val_weighted_f1_curves", plot_config)


def plot_loss_curves_overlay(
    bundles: Sequence[Dict[str, Any]],
    baseline_run_id: str,
    figures_dir: Path,
    plot_config: Dict[str, Any],
) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for idx, bundle in enumerate(bundles):
        epoch_df = bundle["epoch_df"].sort_values("epoch")
        item = bundle["run_item"]
        run_id = bundle["run_id"]
        display_name = str(item.get("display_name", run_id))
        color = color_cycle[idx % len(color_cycle)]
        linewidth = 2.4 if run_id == baseline_run_id else 1.6
        if "train_loss" in epoch_df.columns:
            ax.plot(
                epoch_df["epoch"],
                epoch_df["train_loss"],
                color=color,
                linestyle="-",
                linewidth=linewidth,
                label=f"{display_name} train",
            )
        if "val_loss" in epoch_df.columns:
            ax.plot(
                epoch_df["epoch"],
                epoch_df["val_loss"],
                color=color,
                linestyle="--",
                linewidth=linewidth,
                label=f"{display_name} val",
            )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Train and validation loss curves")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.25)
    save_figure(fig, figures_dir / "loss_curves_overlay", plot_config)


def plot_best_epoch_bar(
    summary_df: pd.DataFrame,
    figures_dir: Path,
    plot_config: Dict[str, Any],
) -> None:
    labels = label_list(summary_df)
    values = as_float_series(summary_df, "best_epoch_by_val_weighted_f1")
    fig, ax = plt.subplots(figsize=(max(9, len(labels) * 1.1), 5))
    x = np.arange(len(labels))
    ax.bar(x, values, color="#5b8c5a")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("Epoch")
    ax.set_title("Best epoch by validation Weighted-F1")
    ax.grid(axis="y", alpha=0.25)
    save_figure(fig, figures_dir / "best_epoch_bar", plot_config)


def plot_val_test_gap(
    summary_df: pd.DataFrame,
    figures_dir: Path,
    plot_config: Dict[str, Any],
) -> None:
    plot_df = summary_df.copy()
    gap = (
        pd.to_numeric(plot_df["best_val_weighted_f1"], errors="coerce")
        - pd.to_numeric(plot_df["test_weighted_f1"], errors="coerce")
    )
    fig, ax = plt.subplots(figsize=(max(9, len(plot_df) * 1.1), 5))
    labels = label_list(plot_df)
    x = np.arange(len(labels))
    ax.bar(x, gap, color="#7a6fbd")
    ax.axhline(0.0, color="black", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("Best val Weighted-F1 - test Weighted-F1")
    ax.set_title("Validation-test generalization gap")
    ax.grid(axis="y", alpha=0.25)
    save_figure(fig, figures_dir / "val_test_gap", plot_config)


def normalize_heatmap_values(values: pd.DataFrame) -> np.ndarray:
    matrix = values.astype(float).to_numpy()
    normalized = np.zeros_like(matrix, dtype=float)
    for col_idx, column in enumerate(values.columns):
        column_values = matrix[:, col_idx]
        valid = column_values[~np.isnan(column_values)]
        if len(valid) == 0:
            normalized[:, col_idx] = np.nan
            continue
        min_value = float(np.min(valid))
        max_value = float(np.max(valid))
        if math.isclose(min_value, max_value):
            normalized[:, col_idx] = 0.5
        else:
            normalized[:, col_idx] = (column_values - min_value) / (max_value - min_value)
        if column in LOWER_IS_BETTER:
            normalized[:, col_idx] = 1.0 - normalized[:, col_idx]
    return normalized


def plot_metric_heatmap(
    summary_df: pd.DataFrame,
    figures_dir: Path,
    plot_config: Dict[str, Any],
) -> None:
    labels = label_list(summary_df)
    metric_df = summary_df[HEATMAP_METRICS].apply(pd.to_numeric, errors="coerce")
    normalized = normalize_heatmap_values(metric_df)

    fig, ax = plt.subplots(figsize=(10, max(5, len(labels) * 0.5)))
    image = ax.imshow(normalized, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    ax.set_xticks(np.arange(len(HEATMAP_METRICS)))
    ax.set_xticklabels(HEATMAP_METRICS, rotation=30, ha="right")
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_title("Metric heatmap (column-normalized color; raw values shown)")

    for row_idx in range(metric_df.shape[0]):
        for col_idx in range(metric_df.shape[1]):
            value = metric_df.iloc[row_idx, col_idx]
            text = "" if pd.isna(value) else f"{value:.4f}"
            ax.text(
                col_idx,
                row_idx,
                text,
                ha="center",
                va="center",
                fontsize=8,
                color="black",
            )
    fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02)
    save_figure(fig, figures_dir / "metric_heatmap", plot_config)


def format_value(value: Any, digits: int = 6) -> str:
    if value is None or pd.isna(value):
        return ""
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.{digits}g}"
    return str(value)


def dataframe_to_markdown(df: pd.DataFrame, columns: Sequence[str]) -> str:
    selected = df[list(columns)].copy()
    lines = [
        "| " + " | ".join(str(column) for column in selected.columns) + " |",
        "| " + " | ".join("---" for _ in selected.columns) + " |",
    ]
    for _, row in selected.iterrows():
        cells = [format_value(row[column]) for column in selected.columns]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def list_early_best_runs(summary_df: pd.DataFrame) -> List[str]:
    names: List[str] = []
    for _, row in summary_df.iterrows():
        best_epoch = to_float(row.get("best_epoch_by_val_weighted_f1"))
        epochs = to_float(row.get("epochs"))
        if pd.isna(best_epoch) or pd.isna(epochs) or float(epochs) <= 0:
            continue
        if float(best_epoch) <= max(3.0, float(epochs) / 3.0):
            names.append(str(row["display_name"]))
    return names


def write_report(
    path: Path,
    summary_df: pd.DataFrame,
    ranking_df: pd.DataFrame,
    delta_df: pd.DataFrame,
    baseline_run_id: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    best_row = ranking_df.iloc[0]
    baseline_row = summary_df[summary_df["run_id"] == baseline_run_id].iloc[0]
    gap = (
        pd.to_numeric(summary_df["best_val_weighted_f1"], errors="coerce")
        - pd.to_numeric(summary_df["test_weighted_f1"], errors="coerce")
    )
    max_gap_idx = gap.abs().idxmax()
    early_best = list_early_best_runs(summary_df)

    lines = [
        "# MultiDAG+CL IEMOCAP stabilization comparison, 20260709",
        "",
        "## 1. Runs compared",
        "",
        dataframe_to_markdown(
            summary_df,
            ["display_name", "run_id", "group", "changed_factor"],
        ),
        "",
        "## 2. Best run by test Weighted-F1",
        "",
        (
            f"The best run by test Weighted-F1 is `{best_row['display_name']}` "
            f"with `test_weighted_f1={format_value(best_row['test_weighted_f1'])}`."
        ),
    ]

    if str(best_row["display_name"]) == "stable_candidate":
        lines.extend(
            [
                "",
                (
                    "stable_candidate is the current best candidate by test "
                    "Weighted-F1, but it should be confirmed with additional "
                    "seeds or at least repeated context settings before "
                    "becoming the final baseline."
                ),
            ]
        )

    lines.extend(
        [
            "",
            "## 3. Ranking table",
            "",
            dataframe_to_markdown(
                ranking_df,
                [
                    "rank",
                    "display_name",
                    "test_weighted_f1",
                    "test_acc",
                    "test_macro_f1",
                    "test_uar",
                    "best_val_weighted_f1",
                    "final_val_loss",
                ],
            ),
            "",
            "## 4. Delta from original w5 baseline",
            "",
            (
                f"Baseline: `{baseline_row['display_name']}` "
                f"(`{baseline_row['run_id']}`). Positive deltas mean higher "
                "metric values than the original w5 baseline, except "
                "`delta_final_val_loss`, where lower is usually better."
            ),
            "",
            dataframe_to_markdown(
                delta_df,
                [
                    "display_name",
                    "delta_test_weighted_f1",
                    "delta_test_acc",
                    "delta_test_macro_f1",
                    "delta_test_uar",
                    "delta_best_val_weighted_f1",
                    "delta_final_val_loss",
                ],
            ),
            "",
            "## 5. Training curve observations",
            "",
        ]
    )

    if early_best:
        lines.append(
            "The following runs reach their validation-best Weighted-F1 early: "
            + ", ".join(f"`{name}`" for name in early_best)
            + ". This can be consistent with early overfitting, but the curves "
            "and repeated seeds should be checked before making that claim."
        )
    else:
        lines.append(
            "No run clearly reaches its validation-best Weighted-F1 in the early "
            "epoch heuristic used here."
        )

    lines.extend(
        [
            "",
            "The loss overlay and validation Weighted-F1 curves should be used "
            "to separate optimization failure from overfitting-like behavior.",
            "",
            "## 6. Generalization gap observations",
            "",
            (
                f"The largest absolute best-validation/test Weighted-F1 gap is "
                f"for `{summary_df.loc[max_gap_idx, 'display_name']}`: "
                f"{format_value(gap.loc[max_gap_idx])}."
            ),
            "",
            (
                "A positive gap means validation Weighted-F1 is higher than test "
                "Weighted-F1 for the validation-selected checkpoint."
            ),
            "",
            "## 7. Recommendation for next baseline",
            "",
            (
                f"The current single-seed recommendation by test Weighted-F1 is "
                f"`{best_row['display_name']}`. Treat it as a candidate rather "
                "than a final baseline until additional seeds or repeated "
                "context settings confirm the ordering."
            ),
            "",
            "## 8. Cautions",
            "",
            "- This is a single-seed comparison.",
            "- The test split is not used for checkpoint selection.",
            "- The current baseline is MultiDAG+CL-inspired, not exact official reproduction.",
            "- The H-R route keeps causal online constraints and does not use future utterances.",
        ]
    )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("Saved:", path)


def plot_all_figures(
    bundles: Sequence[Dict[str, Any]],
    summary_df: pd.DataFrame,
    ranking_df: pd.DataFrame,
    delta_df: pd.DataFrame,
    baseline_run_id: str,
    figures_dir: Path,
    plot_config: Dict[str, Any],
) -> None:
    plot_final_test_metrics_bar(summary_df, figures_dir, plot_config)
    plot_test_weighted_f1_ranking(ranking_df, baseline_run_id, figures_dir, plot_config)
    plot_delta_from_baseline(delta_df, baseline_run_id, figures_dir, plot_config)
    plot_val_weighted_f1_curves(bundles, baseline_run_id, figures_dir, plot_config)
    plot_loss_curves_overlay(bundles, baseline_run_id, figures_dir, plot_config)
    plot_best_epoch_bar(summary_df, figures_dir, plot_config)
    plot_val_test_gap(summary_df, figures_dir, plot_config)
    plot_metric_heatmap(summary_df, figures_dir, plot_config)


def main() -> None:
    args = parse_args()
    config_path = resolve_path(args.config)
    config = load_yaml(config_path)

    runs = get_runs(config)
    baseline_run_id = str(config.get("baseline_run_id", "")).strip()
    if not baseline_run_id:
        raise ValueError("Config field 'baseline_run_id' is required.")
    baseline_run_dir = find_run_directory(baseline_run_id, OUTPUT_ROOT)
    inferred_date = infer_experiment_date_from_run(baseline_run_dir)
    frozen_date = resolve_experiment_date(
        cli_date=args.experiment_date,
        config=config,
        inferred_date=inferred_date,
    )
    configured_root = resolve_path(str(configured_output_root(config)))
    analysis_name = sanitize_run_name(
        str(config.get("analysis_name", "multidag_cl_stabilization_compare"))
    )
    output_dir = (
        resolve_path(args.output_dir)
        if args.output_dir is not None
        else resolve_path(str(config["output_dir"]))
        if config.get("output_dir") is not None
        else resolve_output_category("analysis", frozen_date, configured_root)
        / analysis_name
    )
    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"

    plot_config = get_plot_config(config)
    primary_metric = str(config.get("metrics", {}).get("primary", "test_weighted_f1"))

    print("=" * 100)
    print("MultiDAG+CL stabilization comparison")
    print("=" * 100)
    print("Project root:", PROJECT_ROOT)
    print("Config:", config_path)
    print("Runs root:", OUTPUT_ROOT)
    print("Output dir:", output_dir)
    print("Baseline run:", baseline_run_id)
    print("=" * 100)

    bundles: List[Dict[str, Any]] = []
    missing_errors: List[str] = []
    for item in runs:
        try:
            bundle = load_run_bundle(item, OUTPUT_ROOT)
            bundles.append(bundle)
            diagnosis_note = (
                "with diagnosis"
                if bundle["summary_row"].get("diagnosis_available")
                else "without diagnosis"
            )
            print(f"[OK] {item['run_id']} ({diagnosis_note})")
        except FileNotFoundError as error:
            missing_errors.append(str(error))

    if missing_errors:
        raise FileNotFoundError(
            "Some requested runs are missing required logs.\n"
            + "\n\n".join(missing_errors)
        )

    summary_df = pd.DataFrame([bundle["summary_row"] for bundle in bundles])
    leading_columns = [
        column for column in SUMMARY_COLUMNS if column in summary_df.columns
    ]
    trailing_columns = [
        column for column in summary_df.columns if column not in leading_columns
    ]
    summary_df = summary_df[leading_columns + trailing_columns]
    ranking_df = build_ranking_table(summary_df, primary_metric)
    delta_df = build_delta_table(summary_df, baseline_run_id)

    save_dataframe(summary_df, tables_dir / "run_summary.csv")
    save_dataframe(ranking_df, tables_dir / "run_ranking.csv")
    save_dataframe(delta_df, tables_dir / "delta_from_baseline.csv")

    print("\nPlot figures:")
    plot_all_figures(
        bundles=bundles,
        summary_df=summary_df,
        ranking_df=ranking_df,
        delta_df=delta_df,
        baseline_run_id=baseline_run_id,
        figures_dir=figures_dir,
        plot_config=plot_config,
    )

    write_report(
        path=output_dir / "analysis_report.md",
        summary_df=summary_df,
        ranking_df=ranking_df,
        delta_df=delta_df,
        baseline_run_id=baseline_run_id,
    )

    print("=" * 100)
    print("Finished.")
    print("Best by", primary_metric, ":", ranking_df.iloc[0]["display_name"])
    print("=" * 100)


if __name__ == "__main__":
    main()

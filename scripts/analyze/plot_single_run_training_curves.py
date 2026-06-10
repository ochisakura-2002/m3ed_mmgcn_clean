"""
Plot single-run training curves by epoch.

Purpose:
    For one experiment run, plot each recorded training/validation metric
    as a separate figure along epochs.

Input:
    outputs/runs/<run_id>/logs/epoch_metrics.csv

Output:
    outputs/runs/<run_id>/figures/training_curves/
        <metric_name>.png
        <metric_name>.pdf
        best_epoch_summary.csv

Usage:
    python scripts/analyze/plot_single_run_training_curves.py \
      --run-id 20260526_154008_mmgcn_m3ed_baseline_debug
"""

from pathlib import Path
from typing import Dict, List, Any
import argparse
import re

import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUNS_DIR = PROJECT_ROOT / "outputs" / "runs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot single-run training curves by epoch."
    )

    parser.add_argument(
        "--run-id",
        type=str,
        required=True,
        help="Run ID under outputs/runs/.",
    )

    return parser.parse_args()


def sanitize_filename(name: str) -> str:
    """
    Convert a metric name to a safe filename.
    """
    name = str(name)
    name = re.sub(r"[^a-zA-Z0-9_\\-]+", "_", name)
    name = name.strip("_")
    return name


def metric_display_name(metric: str) -> str:
    """
    Convert metric column name to a readable figure title.
    """
    mapping = {
        "train_loss": "Training Loss",
        "val_loss": "Validation Loss",
        "val_acc": "Validation Accuracy",
        "val_uar": "Validation UAR",
        "val_macro_f1": "Validation Macro-F1",
        "val_weighted_f1": "Validation Weighted-F1",
    }

    return mapping.get(metric, metric.replace("_", " ").title())


def get_metric_selection_rule(metric: str) -> str:
    """
    Decide whether a metric should be maximized or minimized.

    Loss metrics are minimized.
    Other metrics are maximized by default.
    """
    metric_lower = metric.lower()

    if "loss" in metric_lower:
        return "min"

    return "max"


def load_epoch_metrics(run_dir: Path) -> pd.DataFrame:
    metrics_path = run_dir / "logs" / "epoch_metrics.csv"

    if not metrics_path.exists():
        raise FileNotFoundError(
            f"Missing epoch metrics file: {metrics_path}"
        )

    df = pd.read_csv(metrics_path)

    if "epoch" not in df.columns:
        raise ValueError(
            f"Missing required column 'epoch' in {metrics_path}"
        )

    return df


def get_plottable_metric_columns(df: pd.DataFrame) -> List[str]:
    """
    Return numeric metric columns except epoch.
    """
    metric_columns: List[str] = []

    for column in df.columns:
        if column == "epoch":
            continue

        if pd.api.types.is_numeric_dtype(df[column]):
            metric_columns.append(column)

    return metric_columns


def find_best_epoch(
    df: pd.DataFrame,
    metric: str,
) -> Dict[str, Any]:
    """
    Find the best epoch for a metric.
    """
    rule = get_metric_selection_rule(metric)

    valid_df = df.dropna(subset=[metric]).copy()

    if len(valid_df) == 0:
        return {
            "metric": metric,
            "selection_rule": rule,
            "best_epoch": "",
            "best_value": "",
        }

    if rule == "min":
        best_row = valid_df.loc[valid_df[metric].idxmin()]
    else:
        best_row = valid_df.loc[valid_df[metric].idxmax()]

    return {
        "metric": metric,
        "selection_rule": rule,
        "best_epoch": int(best_row["epoch"]),
        "best_value": float(best_row[metric]),
    }


def plot_one_metric(
    df: pd.DataFrame,
    metric: str,
    output_dir: Path,
    run_id: str,
) -> None:
    """
    Plot one metric as one independent figure.
    """
    title = f"{run_id}: {metric_display_name(metric)}"
    filename = sanitize_filename(metric)

    best_info = find_best_epoch(df, metric)
    best_epoch = best_info["best_epoch"]
    best_value = best_info["best_value"]

    plt.figure(figsize=(8, 5))
    plt.plot(
        df["epoch"],
        df[metric],
        marker="o",
        label=metric,
    )

    if best_epoch != "":
        plt.axvline(
            x=best_epoch,
            linestyle="--",
            linewidth=1,
            label=f"best epoch = {best_epoch}",
        )

    plt.xlabel("Epoch")
    plt.ylabel(metric_display_name(metric))
    plt.title(title)
    plt.legend()
    plt.tight_layout()

    png_path = output_dir / f"{filename}.png"
    pdf_path = output_dir / f"{filename}.pdf"

    plt.savefig(png_path, dpi=300)
    plt.savefig(pdf_path)
    plt.close()

    print(f"  [Figure] {metric}: {png_path}")


def build_best_epoch_summary(
    df: pd.DataFrame,
    metric_columns: List[str],
) -> pd.DataFrame:
    rows = []

    for metric in metric_columns:
        rows.append(find_best_epoch(df, metric))

    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()

    run_id = args.run_id
    run_dir = RUNS_DIR / run_id

    if not run_dir.exists():
        raise FileNotFoundError(
            f"Run directory not found: {run_dir}"
        )

    df = load_epoch_metrics(run_dir)
    metric_columns = get_plottable_metric_columns(df)

    if len(metric_columns) == 0:
        raise RuntimeError(
            f"No numeric metric columns found in {run_dir / 'logs' / 'epoch_metrics.csv'}"
        )

    output_dir = run_dir / "figures" / "training_curves"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 100)
    print("Plot single-run training curves")
    print("=" * 100)
    print("Project root:", PROJECT_ROOT)
    print("Run ID:", run_id)
    print("Run dir:", run_dir)
    print("Epoch metrics:", run_dir / "logs" / "epoch_metrics.csv")
    print("Output dir:", output_dir)
    print("Metrics to plot:", metric_columns)
    print("=" * 100)

    for metric in metric_columns:
        plot_one_metric(
            df=df,
            metric=metric,
            output_dir=output_dir,
            run_id=run_id,
        )

    best_df = build_best_epoch_summary(
        df=df,
        metric_columns=metric_columns,
    )

    best_summary_path = output_dir / "best_epoch_summary.csv"
    best_df.to_csv(
        best_summary_path,
        index=False,
        encoding="utf-8-sig",
    )

    print("\nSaved best epoch summary:")
    print(" ", best_summary_path)

    print("\nBest epoch summary:")
    print(best_df.to_string(index=False))

    print("=" * 100)
    print("Finished.")
    print("=" * 100)


if __name__ == "__main__":
    main()
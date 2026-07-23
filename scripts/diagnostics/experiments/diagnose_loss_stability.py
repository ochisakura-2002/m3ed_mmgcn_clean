"""Diagnose loss stability from one or more epoch_metrics.csv files.

This script reads lightweight logs from discovered dated or legacy runs and writes
CSV, Markdown, and matplotlib loss-curve artifacts. It never loads checkpoints.
"""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from utils.output_paths import (  # noqa: E402
    find_run_directory,
    infer_experiment_date_from_run,
    resolve_experiment_date,
    resolve_output_category,
)

OUTPUT_ROOT = PROJECT_ROOT / "outputs"
EPSILON = 1.0e-8

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
    "num_epochs",
    "first_train_loss",
    "final_train_loss",
    "best_train_loss",
    "train_loss_drop",
    "train_loss_drop_ratio",
    "train_loss_monotonic_fraction",
    "train_loss_increase_count",
    "first_val_loss",
    "final_val_loss",
    "best_val_loss",
    "best_val_loss_epoch",
    "val_loss_drop",
    "val_loss_drop_ratio",
    "val_loss_after_best_increase",
    "best_val_weighted_f1",
    "best_val_weighted_f1_epoch",
    "final_val_weighted_f1",
    "loss_stability_label",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose train/validation loss stability for saved runs."
    )
    parser.add_argument(
        "--run-id",
        action="append",
        required=True,
        help="Dated or legacy run ID. Repeat for multiple runs.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Comparison output directory. Defaults to "
            "the dated analysis category for multiple runs."
        ),
    )
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


def run_dir(run_id: str) -> Path:
    return find_run_directory(str(run_id), OUTPUT_ROOT)


def epoch_metrics_path(run_id: str) -> Path:
    return run_dir(run_id) / "logs" / "epoch_metrics.csv"


def read_epoch_metrics(run_id: str) -> pd.DataFrame:
    path = epoch_metrics_path(run_id)
    if not path.exists():
        raise FileNotFoundError(f"Missing epoch metrics: {path}")
    df = pd.read_csv(path)
    if len(df) == 0:
        raise RuntimeError(f"Empty epoch metrics: {path}")
    if "epoch" in df.columns:
        df = df.sort_values("epoch").reset_index(drop=True)
    return df


def numeric_series(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series(dtype="float64")
    return pd.to_numeric(df[column], errors="coerce").dropna()


def first_value(series: pd.Series) -> Any:
    if len(series) == 0:
        return pd.NA
    return float(series.iloc[0])


def final_value(series: pd.Series) -> Any:
    if len(series) == 0:
        return pd.NA
    return float(series.iloc[-1])


def min_value(series: pd.Series) -> Any:
    if len(series) == 0:
        return pd.NA
    return float(series.min())


def max_value(series: pd.Series) -> Any:
    if len(series) == 0:
        return pd.NA
    return float(series.max())


def relative_drop(first: Any, final: Any) -> Any:
    if pd.isna(first) or pd.isna(final) or abs(float(first)) <= EPSILON:
        return pd.NA
    return (float(first) - float(final)) / abs(float(first))


def absolute_drop(first: Any, final: Any) -> Any:
    if pd.isna(first) or pd.isna(final):
        return pd.NA
    return float(first) - float(final)


def epoch_for_best(df: pd.DataFrame, metric: str, mode: str) -> Any:
    if "epoch" not in df.columns or metric not in df.columns:
        return pd.NA
    values = pd.to_numeric(df[metric], errors="coerce")
    valid = df.loc[values.notna()].copy()
    if len(valid) == 0:
        return pd.NA
    valid_values = pd.to_numeric(valid[metric], errors="coerce")
    index = valid_values.idxmin() if mode == "min" else valid_values.idxmax()
    return int(valid.loc[index, "epoch"])


def train_monotonic_stats(train_loss: pd.Series) -> Dict[str, Any]:
    if len(train_loss) < 2:
        return {
            "train_loss_monotonic_fraction": pd.NA,
            "train_loss_increase_count": pd.NA,
        }
    diffs = train_loss.diff().dropna()
    increases = diffs > EPSILON
    non_increases = diffs <= EPSILON
    return {
        "train_loss_monotonic_fraction": float(non_increases.mean()),
        "train_loss_increase_count": int(increases.sum()),
    }


def is_significant_val_rebound(best_val_loss: Any, final_val_loss: Any) -> bool:
    if pd.isna(best_val_loss) or pd.isna(final_val_loss):
        return False
    rebound = float(final_val_loss) - float(best_val_loss)
    threshold = max(0.05, abs(float(best_val_loss)) * 0.10)
    return rebound > threshold


def classify_loss_stability(row: Dict[str, Any]) -> str:
    num_epochs = int(row.get("num_epochs", 0) or 0)
    train_drop_ratio = row.get("train_loss_drop_ratio", pd.NA)
    monotonic_fraction = row.get("train_loss_monotonic_fraction", pd.NA)
    increase_count = row.get("train_loss_increase_count", pd.NA)
    best_val_loss = row.get("best_val_loss", pd.NA)
    first_val_loss = row.get("first_val_loss", pd.NA)
    final_val_loss = row.get("final_val_loss", pd.NA)

    if num_epochs < 2 or pd.isna(train_drop_ratio) or pd.isna(monotonic_fraction):
        return "insufficient_data"

    train_decreases = float(train_drop_ratio) >= 0.15
    train_barely_moves = float(train_drop_ratio) < 0.05
    train_fairly_smooth = float(monotonic_fraction) >= 0.60
    train_very_bumpy = (
        not pd.isna(increase_count)
        and int(increase_count) > max(2, int(0.50 * max(num_epochs - 1, 1)))
    )

    val_has_drop = False
    if not pd.isna(first_val_loss) and not pd.isna(final_val_loss):
        val_has_drop = float(final_val_loss) <= float(first_val_loss) + EPSILON
    if not pd.isna(first_val_loss) and not pd.isna(best_val_loss):
        val_has_drop = val_has_drop or (
            float(best_val_loss) <= float(first_val_loss) * 0.95
        )

    if train_barely_moves:
        return "not_decreasing"
    if train_decreases and is_significant_val_rebound(best_val_loss, final_val_loss):
        return "decreasing_but_overfit"
    if train_decreases and train_fairly_smooth and val_has_drop:
        return "stable_decreasing"
    if train_very_bumpy or float(monotonic_fraction) < 0.45:
        return "unstable"
    return "unstable"


def summarize_run(run_id: str, df: pd.DataFrame) -> Dict[str, Any]:
    train_loss = numeric_series(df, "train_loss")
    val_loss = numeric_series(df, "val_loss")
    val_weighted_f1 = numeric_series(df, "val_weighted_f1")

    first_train = first_value(train_loss)
    final_train = final_value(train_loss)
    first_val = first_value(val_loss)
    final_val = final_value(val_loss)
    best_val = min_value(val_loss)

    row: Dict[str, Any] = {
        "run_id": str(run_id),
        "num_epochs": int(len(df)),
        "first_train_loss": first_train,
        "final_train_loss": final_train,
        "best_train_loss": min_value(train_loss),
        "train_loss_drop": absolute_drop(first_train, final_train),
        "train_loss_drop_ratio": relative_drop(first_train, final_train),
        **train_monotonic_stats(train_loss),
        "first_val_loss": first_val,
        "final_val_loss": final_val,
        "best_val_loss": best_val,
        "best_val_loss_epoch": epoch_for_best(df, "val_loss", "min"),
        "val_loss_drop": absolute_drop(first_val, final_val),
        "val_loss_drop_ratio": relative_drop(first_val, final_val),
        "val_loss_after_best_increase": (
            pd.NA if pd.isna(best_val) or pd.isna(final_val) else float(final_val) - float(best_val)
        ),
        "best_val_weighted_f1": max_value(val_weighted_f1),
        "best_val_weighted_f1_epoch": epoch_for_best(df, "val_weighted_f1", "max"),
        "final_val_weighted_f1": final_value(val_weighted_f1),
    }
    row["loss_stability_label"] = classify_loss_stability(row)
    return {column: row.get(column, pd.NA) for column in SUMMARY_COLUMNS}


def format_value(value: Any) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def dataframe_to_markdown(df: pd.DataFrame, columns: List[str]) -> str:
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for _, row in df.iterrows():
        lines.append(
            "| "
            + " | ".join(format_value(row.get(column, pd.NA)) for column in columns)
            + " |"
        )
    return "\n".join(lines)


def write_report(path: Path, summary_df: pd.DataFrame, run_ids: List[str]) -> None:
    columns = [
        "run_id",
        "num_epochs",
        "train_loss_drop_ratio",
        "train_loss_monotonic_fraction",
        "val_loss_after_best_increase",
        "best_val_weighted_f1",
        "loss_stability_label",
    ]
    lines = [
        "# Loss stability report",
        "",
        "## Inputs",
        "",
        *[f"- `{run_id}`" for run_id in run_ids],
        "",
        "## Decision Logic",
        "",
        "- `stable_decreasing`: train loss drops by at least 15%, at least 60% of epoch-to-epoch train-loss steps are non-increasing, and validation loss improves overall or has a clear best-loss drop.",
        "- `decreasing_but_overfit`: train loss drops by at least 15%, but validation loss rebounds clearly after its best epoch.",
        "- `not_decreasing`: train loss drops by less than 5%.",
        "- `unstable`: train loss is bumpy or does not satisfy the stable-decreasing conditions.",
        "- `insufficient_data`: fewer than two usable epochs or missing loss columns.",
        "",
        "## Summary",
        "",
        dataframe_to_markdown(summary_df, columns),
        "",
        "## Notes",
        "",
        "- These labels are diagnostics for training behavior, not claims of test-set improvement.",
        "- Validation loss and validation Weighted-F1 can disagree; checkpoint selection should still follow the training config.",
        "- One seed is not enough to claim a stable baseline, but it is enough to reject visibly broken loss behavior.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_figure(fig: plt.Figure, stem: Path) -> None:
    fig.savefig(stem.with_suffix(".png"), dpi=300)
    fig.savefig(stem.with_suffix(".pdf"))
    plt.close(fig)


def plot_single_loss_curves(run_id: str, df: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = df["epoch"] if "epoch" in df.columns else range(1, len(df) + 1)
    if "train_loss" in df.columns:
        ax.plot(x, pd.to_numeric(df["train_loss"], errors="coerce"), label="train_loss")
    if "val_loss" in df.columns:
        ax.plot(x, pd.to_numeric(df["val_loss"], errors="coerce"), label="val_loss")
    ax.set_title(f"Loss curves: {run_id}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    save_figure(fig, output_dir / "loss_curves")


def plot_overlay(
    run_dfs: Dict[str, pd.DataFrame],
    metric: str,
    output_dir: Path,
    stem: str,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    for run_id, df in run_dfs.items():
        if metric not in df.columns:
            continue
        x = df["epoch"] if "epoch" in df.columns else range(1, len(df) + 1)
        ax.plot(x, pd.to_numeric(df[metric], errors="coerce"), label=run_id)
    ax.set_title(metric.replace("_", " ").title())
    ax.set_xlabel("Epoch")
    ax.set_ylabel(metric)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    save_figure(fig, output_dir / stem)


def run_single(run_id: str, df: pd.DataFrame, summary_df: pd.DataFrame) -> None:
    output_dir = run_dir(run_id) / "figures" / "loss_stability"
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(
        output_dir / "loss_stability_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    write_report(
        path=output_dir / "loss_stability_report.md",
        summary_df=summary_df,
        run_ids=[run_id],
    )
    plot_single_loss_curves(run_id, df, output_dir)
    print("Output dir:", project_relative(output_dir))


def run_multi(
    run_ids: List[str],
    run_dfs: Dict[str, pd.DataFrame],
    summary_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(
        output_dir / "loss_stability_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    write_report(
        path=output_dir / "loss_stability_report.md",
        summary_df=summary_df,
        run_ids=run_ids,
    )
    plot_overlay(run_dfs, "train_loss", output_dir, "train_loss_overlay")
    plot_overlay(run_dfs, "val_loss", output_dir, "val_loss_overlay")
    print("Output dir:", project_relative(output_dir))


def main() -> None:
    args = parse_args()
    run_ids = [str(run_id) for run_id in args.run_id]
    run_dfs = {run_id: read_epoch_metrics(run_id) for run_id in run_ids}
    summary_rows = [
        summarize_run(run_id=run_id, df=run_dfs[run_id])
        for run_id in run_ids
    ]
    summary_df = pd.DataFrame(summary_rows, columns=SUMMARY_COLUMNS)

    if len(run_ids) == 1 and args.output_dir is None:
        run_single(run_ids[0], run_dfs[run_ids[0]], summary_df)
        return

    inferred_date = next(
        (
            value
            for value in (
                infer_experiment_date_from_run(run_dir(run_id)) for run_id in run_ids
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
        resolve_path(args.output_dir)
        if args.output_dir is not None
        else resolve_output_category(
            "analysis", frozen_date, OUTPUT_ROOT
        )
        / "loss_stability_compare"
    )
    run_multi(run_ids, run_dfs, summary_df, output_dir)


if __name__ == "__main__":
    main()

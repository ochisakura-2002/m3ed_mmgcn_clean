"""
Plot single-run training curves by epoch.

Purpose:
    For one experiment run, plot each recorded training/validation metric
    as a separate figure along epochs.

Input:
    outputs/<YYYYMMDD>/runs/<run_id>/logs/epoch_metrics.csv

Output:
    outputs/<YYYYMMDD>/runs/<run_id>/figures/training_curves/
        <metric_name>.png
        <metric_name>.pdf
        best_epoch_summary.csv

Usage:
    python scripts/analysis/common/plot_single_run_training_curves.py \
      --run-id 20260526_154008_mmgcn_m3ed_baseline_debug
"""

import argparse
from pathlib import Path
import re
import sys
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from utils.output_paths import find_run_directory  # noqa: E402

OUTPUT_ROOT = PROJECT_ROOT / "outputs"
BEST_SELECTION_METRIC = "val_weighted_f1"

BEST_VAL_VS_TEST_COLUMNS = [
    "run_id",
    "best_epoch",
    "best_selection_metric",
    "best_val_loss",
    "best_val_acc",
    "best_val_weighted_f1",
    "best_val_macro_f1",
    "best_val_uar",
    "test_loss",
    "test_accuracy",
    "test_acc",
    "test_weighted_f1",
    "test_macro_f1",
    "test_uar",
    "val_test_acc_gap",
    "val_test_weighted_f1_gap",
    "val_test_macro_f1_gap",
    "val_test_uar_gap",
    "epoch_metrics_path",
    "test_metrics_path",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot single-run training curves by epoch."
    )

    parser.add_argument(
        "--run-id",
        type=str,
        required=True,
        help="Run ID discovered from dated outputs first, then legacy outputs.",
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


def metric_curve_title(metric: str) -> str:
    """
    Return an unambiguous process-curve title.

    The val_* curves are validation process metrics, not test metrics.
    """
    mapping = {
        "train_loss": "Training Loss over Epochs",
        "val_loss": "Validation Loss over Epochs",
        "val_acc": "Validation Accuracy over Epochs",
        "val_uar": "Validation UAR over Epochs",
        "val_macro_f1": "Validation Macro-F1 over Epochs",
        "val_weighted_f1": "Validation Weighted-F1 over Epochs",
    }

    return mapping.get(metric, f"{metric_display_name(metric)} over Epochs")


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
    title = f"{run_id}: {metric_curve_title(metric)}"
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


def read_csv_or_none(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None

    try:
        return pd.read_csv(path)
    except Exception as error:
        print(f"[WARN] Failed to read CSV: {path} | {error}")
        return None


def first_existing_metric(row: pd.Series, names: List[str]) -> Any:
    for name in names:
        if name in row.index:
            return row[name]
    return pd.NA


def to_float_or_na(value: Any) -> Any:
    if value is None or pd.isna(value):
        return pd.NA
    try:
        return float(value)
    except (TypeError, ValueError):
        return pd.NA


def project_relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT.resolve()))
    except ValueError:
        return str(path)


def choose_validation_best_epoch(
    epoch_df: pd.DataFrame,
    training_curves_dir: Path,
) -> Tuple[Any, str]:
    """
    Choose the validation-best epoch without consulting test metrics.

    Priority:
    1. figures/training_curves/best_epoch_summary.csv
    2. logs/epoch_metrics.csv max val_weighted_f1
    """
    summary_path = training_curves_dir / "best_epoch_summary.csv"
    summary_df = read_csv_or_none(summary_path)

    if summary_df is not None and len(summary_df) > 0:
        if {"metric", "best_epoch"}.issubset(summary_df.columns):
            metric_df = summary_df[summary_df["metric"] == BEST_SELECTION_METRIC]
            metric_df = metric_df.dropna(subset=["best_epoch"])
            if len(metric_df) > 0:
                return int(metric_df.iloc[0]["best_epoch"]), BEST_SELECTION_METRIC

    if BEST_SELECTION_METRIC not in epoch_df.columns:
        print(
            "[WARN] Cannot determine validation-best epoch: "
            f"missing {BEST_SELECTION_METRIC} in epoch metrics."
        )
        return pd.NA, BEST_SELECTION_METRIC

    valid_df = epoch_df.dropna(subset=[BEST_SELECTION_METRIC])
    if len(valid_df) == 0:
        print(
            "[WARN] Cannot determine validation-best epoch: "
            f"all {BEST_SELECTION_METRIC} values are NA."
        )
        return pd.NA, BEST_SELECTION_METRIC

    best_row = valid_df.loc[valid_df[BEST_SELECTION_METRIC].idxmax()]
    return int(best_row["epoch"]), BEST_SELECTION_METRIC


def get_epoch_row(epoch_df: pd.DataFrame, epoch: Any) -> Optional[pd.Series]:
    if pd.isna(epoch) or "epoch" not in epoch_df.columns:
        return None

    match_df = epoch_df[epoch_df["epoch"] == int(epoch)]
    if len(match_df) == 0:
        return None

    return match_df.iloc[0]


def metric_gap(val_value: Any, test_value: Any) -> Any:
    val_float = to_float_or_na(val_value)
    test_float = to_float_or_na(test_value)
    if pd.isna(val_float) or pd.isna(test_float):
        return pd.NA
    return float(val_float) - float(test_float)


def build_best_val_vs_test_summary(
    run_dir: Path,
    run_id: str,
    epoch_df: pd.DataFrame,
    training_curves_dir: Path,
) -> pd.DataFrame:
    best_epoch, selection_metric = choose_validation_best_epoch(
        epoch_df=epoch_df,
        training_curves_dir=training_curves_dir,
    )
    best_row = get_epoch_row(epoch_df, best_epoch)

    test_metrics_path = run_dir / "logs" / "evaluations" / "test_best_model" / "metrics.csv"
    test_df = read_csv_or_none(test_metrics_path)
    test_row: Optional[pd.Series] = None

    if test_df is None or len(test_df) == 0:
        print(
            "[WARN] Missing test metrics for validation-best checkpoint. "
            f"Leave test fields empty: {test_metrics_path}"
        )
    else:
        test_row = test_df.iloc[0]

    def best_value(metric_name: str) -> Any:
        if best_row is None or metric_name not in best_row.index:
            return pd.NA
        return to_float_or_na(best_row[metric_name])

    def test_value(names: List[str]) -> Any:
        if test_row is None:
            return pd.NA
        return to_float_or_na(first_existing_metric(test_row, names))

    best_val_loss = best_value("val_loss")
    best_val_acc = best_value("val_acc")
    best_val_weighted_f1 = best_value("val_weighted_f1")
    best_val_macro_f1 = best_value("val_macro_f1")
    best_val_uar = best_value("val_uar")

    test_loss = test_value(["loss", "test_loss"])
    test_accuracy = test_value(["accuracy", "acc", "test_accuracy", "test_acc"])
    test_acc = test_value(["acc", "accuracy", "test_acc", "test_accuracy"])
    test_weighted_f1 = test_value(["weighted_f1", "test_weighted_f1"])
    test_macro_f1 = test_value(["macro_f1", "test_macro_f1"])
    test_uar = test_value(["uar", "test_uar"])

    row = {
        "run_id": run_id,
        "best_epoch": best_epoch,
        "best_selection_metric": selection_metric,
        "best_val_loss": best_val_loss,
        "best_val_acc": best_val_acc,
        "best_val_weighted_f1": best_val_weighted_f1,
        "best_val_macro_f1": best_val_macro_f1,
        "best_val_uar": best_val_uar,
        "test_loss": test_loss,
        "test_accuracy": test_accuracy,
        "test_acc": test_acc,
        "test_weighted_f1": test_weighted_f1,
        "test_macro_f1": test_macro_f1,
        "test_uar": test_uar,
        "val_test_acc_gap": metric_gap(best_val_acc, test_acc),
        "val_test_weighted_f1_gap": metric_gap(best_val_weighted_f1, test_weighted_f1),
        "val_test_macro_f1_gap": metric_gap(best_val_macro_f1, test_macro_f1),
        "val_test_uar_gap": metric_gap(best_val_uar, test_uar),
        "epoch_metrics_path": project_relative(run_dir / "logs" / "epoch_metrics.csv"),
        "test_metrics_path": (
            project_relative(test_metrics_path) if test_metrics_path.exists() else ""
        ),
    }

    return pd.DataFrame([row], columns=BEST_VAL_VS_TEST_COLUMNS)


def summary_value(summary_row: pd.Series, name: str) -> Any:
    if name not in summary_row.index:
        return pd.NA
    return to_float_or_na(summary_row[name])


def annotate_caption() -> None:
    plt.figtext(
        0.5,
        0.01,
        "Test metrics are final metrics of the validation-best checkpoint, "
        "not per-epoch process metrics.",
        ha="center",
        fontsize=8,
    )


def plot_best_val_vs_test_classification_metrics(
    summary_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    if len(summary_df) == 0:
        return

    row = summary_df.iloc[0]
    metric_labels = ["Acc", "Weighted-F1", "Macro-F1", "UAR"]
    val_values = [
        summary_value(row, "best_val_acc"),
        summary_value(row, "best_val_weighted_f1"),
        summary_value(row, "best_val_macro_f1"),
        summary_value(row, "best_val_uar"),
    ]
    test_values = [
        summary_value(row, "test_acc"),
        summary_value(row, "test_weighted_f1"),
        summary_value(row, "test_macro_f1"),
        summary_value(row, "test_uar"),
    ]

    plot_df = pd.DataFrame(
        {
            "metric": metric_labels,
            "Validation": val_values,
            "Test": test_values,
        }
    ).set_index("metric")
    plot_df = plot_df.apply(pd.to_numeric, errors="coerce")

    ax = plot_df.plot(kind="bar", figsize=(8, 5))
    ax.set_ylabel("Metric Value")
    ax.set_ylim(0, 1)
    ax.set_title("Validation-Best Checkpoint: Validation vs Test Metrics")
    ax.legend(loc="best")
    plt.xticks(rotation=0)
    annotate_caption()
    plt.tight_layout(rect=(0, 0.06, 1, 1))

    png_path = output_dir / "best_val_vs_test_classification_metrics.png"
    pdf_path = output_dir / "best_val_vs_test_classification_metrics.pdf"
    plt.savefig(png_path, dpi=300)
    plt.savefig(pdf_path)
    plt.close()

    print(f"  [Figure] best val vs test classification metrics: {png_path}")


def plot_best_val_vs_test_loss(
    summary_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    if len(summary_df) == 0:
        return

    row = summary_df.iloc[0]
    plot_df = pd.DataFrame(
        {
            "split": ["Validation", "Test"],
            "loss": [
                summary_value(row, "best_val_loss"),
                summary_value(row, "test_loss"),
            ],
        }
    )
    plot_df["loss"] = pd.to_numeric(plot_df["loss"], errors="coerce")

    plt.figure(figsize=(6, 5))
    plt.bar(plot_df["split"], plot_df["loss"])
    plt.ylabel("Loss")
    plt.title("Validation-Best Checkpoint: Validation vs Test Loss")
    annotate_caption()
    plt.tight_layout(rect=(0, 0.06, 1, 1))

    png_path = output_dir / "best_val_vs_test_loss.png"
    pdf_path = output_dir / "best_val_vs_test_loss.pdf"
    plt.savefig(png_path, dpi=300)
    plt.savefig(pdf_path)
    plt.close()

    print(f"  [Figure] best val vs test loss: {png_path}")


def save_best_val_vs_test_outputs(
    run_dir: Path,
    run_id: str,
    epoch_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    summary_df = build_best_val_vs_test_summary(
        run_dir=run_dir,
        run_id=run_id,
        epoch_df=epoch_df,
        training_curves_dir=output_dir,
    )

    summary_path = output_dir / "best_val_vs_test_summary.csv"
    summary_df.to_csv(
        summary_path,
        index=False,
        encoding="utf-8-sig",
    )

    print("\nSaved validation-best vs final test summary:")
    print(" ", summary_path)
    print(summary_df.to_string(index=False))

    plot_best_val_vs_test_classification_metrics(
        summary_df=summary_df,
        output_dir=output_dir,
    )
    plot_best_val_vs_test_loss(
        summary_df=summary_df,
        output_dir=output_dir,
    )


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
    run_dir = find_run_directory(run_id, OUTPUT_ROOT)

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

    print("\nBuild validation-best epoch vs final test summary:")
    save_best_val_vs_test_outputs(
        run_dir=run_dir,
        run_id=run_id,
        epoch_df=df,
        output_dir=output_dir,
    )

    print("=" * 100)
    print("Finished.")
    print("=" * 100)


if __name__ == "__main__":
    main()

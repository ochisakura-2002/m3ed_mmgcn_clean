"""
Plot single-run final evaluation analysis.

Purpose:
    For one experiment run, analyze the final evaluated model on test_best_model.

Input:
    outputs/<YYYYMMDD>/runs/<run_id>/logs/evaluations/test_best_model/metrics.csv
    outputs/<YYYYMMDD>/runs/<run_id>/logs/evaluations/test_best_model/confusion_matrix.csv

Output:
    outputs/<YYYYMMDD>/runs/<run_id>/figures/final_analysis/
        overall_metrics.csv
        per_class_metrics_from_confusion_matrix.csv

        overall_metrics.png / .pdf

        per_class_precision.png / .pdf
        per_class_recall.png / .pdf
        per_class_f1.png / .pdf
        per_class_support.png / .pdf
        per_class_one_vs_rest_acc.png / .pdf

        confusion_matrix_raw.png / .pdf
        confusion_matrix_normalized.png / .pdf

Usage:
    python scripts/analysis/common/plot_single_run_final_analysis.py \
      --run-id 20260526_154008_mmgcn_m3ed_baseline_debug
"""

from pathlib import Path
from typing import Dict, List, Any
import argparse
import re
import sys

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from utils.output_paths import find_run_directory  # noqa: E402

OUTPUT_ROOT = PROJECT_ROOT / "outputs"

EVAL_NAME = "test_best_model"

OVERALL_METRICS = [
    "loss",
    "acc",
    "uar",
    "macro_f1",
    "weighted_f1",
]

PER_CLASS_METRICS = [
    "precision",
    "recall",
    "f1",
    "support",
    "one_vs_rest_acc",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot single-run final evaluation analysis."
    )

    parser.add_argument(
        "--run-id",
        type=str,
        required=True,
        help="Run ID discovered from dated outputs first, then legacy outputs.",
    )
    parser.add_argument(
        "--eval-name",
        type=str,
        default=EVAL_NAME,
        help=(
            "Evaluation folder under logs/evaluations/. "
            "Examples: test_best_model, val_best_model."
        ),
    )

    return parser.parse_args()


def sanitize_filename(name: str) -> str:
    name = str(name)
    name = re.sub(r"[^a-zA-Z0-9_\\-]+", "_", name)
    name = name.strip("_")
    return name


def metric_display_name(metric: str) -> str:
    mapping = {
        "loss": "Loss",
        "acc": "Accuracy",
        "uar": "UAR",
        "macro_f1": "Macro-F1",
        "weighted_f1": "Weighted-F1",
        "precision": "Precision",
        "recall": "Recall",
        "f1": "F1",
        "support": "Support",
        "one_vs_rest_acc": "One-vs-Rest Accuracy",
    }

    return mapping.get(metric, metric.replace("_", " ").title())


def split_display_name(split: str) -> str:
    split = str(split).strip().lower()
    mapping = {
        "val": "Validation",
        "valid": "Validation",
        "validation": "Validation",
        "test": "Test",
        "train": "Train",
    }
    return mapping.get(split, split.title())


def infer_split_from_eval_name(eval_name: str) -> str:
    eval_name = str(eval_name).strip()
    if "_" in eval_name:
        return eval_name.split("_", 1)[0]
    return eval_name


def infer_split(metrics_row: pd.Series, eval_name: str) -> str:
    if "split" in metrics_row.index and not pd.isna(metrics_row["split"]):
        return str(metrics_row["split"])
    return infer_split_from_eval_name(eval_name)


def clean_label_name(name: str) -> str:
    name = str(name)

    if name.startswith("true_"):
        return name[len("true_"):]

    if name.startswith("pred_"):
        return name[len("pred_"):]

    return name


def load_overall_metrics(eval_dir: Path) -> pd.DataFrame:
    metrics_path = eval_dir / "metrics.csv"

    if not metrics_path.exists():
        raise FileNotFoundError(
            f"Missing metrics file: {metrics_path}"
        )

    df = pd.read_csv(metrics_path)

    if len(df) == 0:
        raise RuntimeError(
            f"Empty metrics file: {metrics_path}"
        )

    return df


def load_confusion_matrix(eval_dir: Path) -> pd.DataFrame:
    matrix_path = eval_dir / "confusion_matrix.csv"

    if not matrix_path.exists():
        raise FileNotFoundError(
            f"Missing confusion matrix file: {matrix_path}"
        )

    df = pd.read_csv(matrix_path, index_col=0)

    if df.shape[0] == 0 or df.shape[1] == 0:
        raise RuntimeError(
            f"Empty confusion matrix file: {matrix_path}"
        )

    return df


def compute_per_class_metrics(confusion_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-class precision, recall, F1, support, one-vs-rest accuracy
    from a confusion matrix.

    Rows are true labels.
    Columns are predicted labels.
    """
    matrix = confusion_df.values.astype(float)
    total = matrix.sum()

    true_labels = [clean_label_name(x) for x in confusion_df.index.tolist()]
    pred_labels = [clean_label_name(x) for x in confusion_df.columns.tolist()]

    if true_labels != pred_labels:
        print("[WARN] True label order and predicted label order are not identical.")
        print("  true labels:", true_labels)
        print("  pred labels:", pred_labels)

    rows: List[Dict[str, Any]] = []

    for i, label_name in enumerate(true_labels):
        tp = matrix[i, i]
        fn = matrix[i, :].sum() - tp
        fp = matrix[:, i].sum() - tp
        tn = total - tp - fn - fp

        support = matrix[i, :].sum()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        one_vs_rest_acc = (
            (tp + tn) / total
            if total > 0
            else 0.0
        )

        rows.append(
            {
                "label_id": i,
                "label_name": label_name,
                "support": int(support),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "one_vs_rest_acc": float(one_vs_rest_acc),
                "tp": int(tp),
                "fp": int(fp),
                "fn": int(fn),
                "tn": int(tn),
            }
        )

    return pd.DataFrame(rows)


def plot_overall_metrics(
    metrics_row: pd.Series,
    output_dir: Path,
    run_id: str,
    eval_name: str,
) -> None:
    """
    Plot all overall metrics in one figure.

    This includes loss, acc, uar, macro_f1, weighted_f1 if they exist.
    Loss may be on a different scale from the 0-1 metrics, but keeping it
    in the same chart matches the requested single-figure summary.
    """
    available_metrics = [
        metric
        for metric in OVERALL_METRICS
        if metric in metrics_row.index
    ]

    if len(available_metrics) == 0:
        print("[WARN] No overall metrics found. Skip overall metric figure.")
        return

    labels = [
        metric_display_name(metric)
        for metric in available_metrics
    ]

    values = [
        float(metrics_row[metric])
        for metric in available_metrics
    ]

    plt.figure(figsize=(8, 5))
    plt.bar(labels, values)

    split_name = split_display_name(infer_split(metrics_row, eval_name))

    plt.ylabel("Metric Value")
    plt.title(
        f"{run_id}: Overall {split_name} Metrics "
        "of Validation-Best Checkpoint"
    )

    for index, value in enumerate(values):
        plt.text(
            index,
            value,
            f"{value:.4f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()

    png_path = output_dir / "overall_metrics.png"
    pdf_path = output_dir / "overall_metrics.pdf"

    plt.savefig(png_path, dpi=300)
    plt.savefig(pdf_path)
    plt.close()

    print(f"  [Figure] overall metrics: {png_path}")


def plot_per_class_metric(
    per_class_df: pd.DataFrame,
    metric: str,
    output_dir: Path,
    run_id: str,
) -> None:
    if metric not in per_class_df.columns:
        print(f"[WARN] Per-class metric not found: {metric}")
        return

    title = f"{run_id}: Per-class {metric_display_name(metric)}"
    filename = f"per_class_{sanitize_filename(metric)}"

    x_labels = per_class_df["label_name"].tolist()
    values = per_class_df[metric].values

    plt.figure(figsize=(9, 5))
    plt.bar(x_labels, values)

    plt.xlabel("Emotion Class")
    plt.ylabel(metric_display_name(metric))
    plt.title(title)

    if metric != "support":
        plt.ylim(0, 1)

    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()

    png_path = output_dir / f"{filename}.png"
    pdf_path = output_dir / f"{filename}.pdf"

    plt.savefig(png_path, dpi=300)
    plt.savefig(pdf_path)
    plt.close()

    print(f"  [Figure] per-class {metric}: {png_path}")


def normalize_confusion_matrix_by_row(confusion_df: pd.DataFrame) -> pd.DataFrame:
    matrix = confusion_df.values.astype(float)
    row_sums = matrix.sum(axis=1, keepdims=True)

    normalized = np.divide(
        matrix,
        row_sums,
        out=np.zeros_like(matrix),
        where=row_sums != 0,
    )

    return pd.DataFrame(
        normalized,
        index=confusion_df.index,
        columns=confusion_df.columns,
    )


def plot_confusion_matrix(
    confusion_df: pd.DataFrame,
    output_dir: Path,
    run_id: str,
    normalized: bool,
) -> None:
    if normalized:
        plot_df = normalize_confusion_matrix_by_row(confusion_df)
        filename = "confusion_matrix_normalized"
        title = f"{run_id}: Row-normalized Confusion Matrix"
        value_format = ".2f"
    else:
        plot_df = confusion_df.copy()
        filename = "confusion_matrix_raw"
        title = f"{run_id}: Raw Confusion Matrix"
        value_format = ".0f"

    labels = [clean_label_name(x) for x in plot_df.index.tolist()]
    matrix = plot_df.values.astype(float)

    plt.figure(figsize=(8, 7))
    image = plt.imshow(matrix, aspect="auto")
    plt.colorbar(image)

    plt.xticks(
        ticks=np.arange(len(labels)),
        labels=labels,
        rotation=45,
        ha="right",
    )
    plt.yticks(
        ticks=np.arange(len(labels)),
        labels=labels,
    )

    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(title)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            plt.text(
                j,
                i,
                format(matrix[i, j], value_format),
                ha="center",
                va="center",
                fontsize=8,
            )

    plt.tight_layout()

    png_path = output_dir / f"{filename}.png"
    pdf_path = output_dir / f"{filename}.pdf"

    plt.savefig(png_path, dpi=300)
    plt.savefig(pdf_path)
    plt.close()

    print(f"  [Figure] {filename}: {png_path}")


def main() -> None:
    args = parse_args()

    run_id = args.run_id
    run_dir = find_run_directory(run_id, OUTPUT_ROOT)

    if not run_dir.exists():
        raise FileNotFoundError(
            f"Run directory not found: {run_dir}"
        )

    eval_name = str(args.eval_name)
    eval_dir = run_dir / "logs" / "evaluations" / eval_name

    if not eval_dir.exists():
        raise FileNotFoundError(
            f"Evaluation directory not found: {eval_dir}"
        )

    output_dir = run_dir / "figures" / "final_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 100)
    print("Plot single-run final evaluation analysis")
    print("=" * 100)
    print("Project root:", PROJECT_ROOT)
    print("Run ID:", run_id)
    print("Run dir:", run_dir)
    print("Evaluation dir:", eval_dir)
    print("Output dir:", output_dir)
    print("=" * 100)

    overall_df = load_overall_metrics(eval_dir)
    overall_row = overall_df.iloc[0]

    overall_save_path = output_dir / "overall_metrics.csv"
    overall_df.to_csv(
        overall_save_path,
        index=False,
        encoding="utf-8-sig",
    )

    confusion_df = load_confusion_matrix(eval_dir)
    per_class_df = compute_per_class_metrics(confusion_df)

    per_class_save_path = output_dir / "per_class_metrics_from_confusion_matrix.csv"
    per_class_df.to_csv(
        per_class_save_path,
        index=False,
        encoding="utf-8-sig",
    )

    print("Saved overall metrics:")
    print(" ", overall_save_path)

    print("Saved per-class metrics:")
    print(" ", per_class_save_path)

    print("\nOverall metrics:")
    print(overall_df.to_string(index=False))

    print("\nPer-class metrics:")
    print(
        per_class_df[
            [
                "label_id",
                "label_name",
                "support",
                "precision",
                "recall",
                "f1",
                "one_vs_rest_acc",
            ]
        ].to_string(index=False)
    )

    print("\nPlot overall metrics:")
    plot_overall_metrics(
        metrics_row=overall_row,
        output_dir=output_dir,
        run_id=run_id,
        eval_name=eval_name,
    )

    print("\nPlot per-class metrics:")
    for metric in PER_CLASS_METRICS:
        plot_per_class_metric(
            per_class_df=per_class_df,
            metric=metric,
            output_dir=output_dir,
            run_id=run_id,
        )

    print("\nPlot confusion matrices:")
    plot_confusion_matrix(
        confusion_df=confusion_df,
        output_dir=output_dir,
        run_id=run_id,
        normalized=False,
    )

    plot_confusion_matrix(
        confusion_df=confusion_df,
        output_dir=output_dir,
        run_id=run_id,
        normalized=True,
    )

    print("=" * 100)
    print("Finished.")
    print("=" * 100)


if __name__ == "__main__":
    main()

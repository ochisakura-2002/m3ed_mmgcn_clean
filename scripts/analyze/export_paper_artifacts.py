"""
Export paper-ready single-run tables and figures.

This script reads an existing run directory and writes a compact
paper_artifacts/ folder without changing training, evaluation, checkpoint
selection, or dataset splits.

Usage:
    python scripts/analyze/export_paper_artifacts.py \
      --run-dir outputs/<YYYYMMDD>/runs/<run_id> \
      --split test \
      --also-val
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import shutil
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[2]
BEST_SELECTION_METRIC = "val_weighted_f1"

VAL_VS_TEST_COLUMNS = [
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

SINGLE_RUN_SUMMARY_COLUMNS = [
    "run_id",
    "experiment_name",
    "dataset",
    "model",
    "context_mode",
    "window_past",
    "active_modalities",
    "seed",
    "epochs",
    "batch_size",
    "lr",
    "weight_decay",
    "hidden_dim",
    "dropout",
    "num_graph_layers",
    "modality_encoder_type",
    "best_epoch",
    "val_loss",
    "val_acc",
    "val_weighted_f1",
    "val_macro_f1",
    "val_uar",
    "test_loss",
    "test_acc",
    "test_weighted_f1",
    "test_macro_f1",
    "test_uar",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export paper-ready tables and figures for one run."
    )
    parser.add_argument(
        "--run-dir",
        required=True,
        help="Explicit dated or legacy run directory.",
    )
    parser.add_argument(
        "--split",
        default="test",
        choices=("train", "val", "test"),
        help="Primary final-analysis split to export.",
    )
    parser.add_argument(
        "--also-val",
        action="store_true",
        help="Also export validation per-class metrics when available.",
    )
    parser.add_argument(
        "--output-subdir",
        default="paper_artifacts",
        help="Output subdirectory under the run directory.",
    )
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


def safe_get(config: Dict[str, Any], keys: Iterable[str], default: Any = "") -> Any:
    current: Any = config
    for key in keys:
        if not isinstance(current, dict):
            return default
        if key not in current:
            return default
        current = current[key]
    if current is None:
        return default
    return current


def load_yaml_or_empty(path: Path) -> Dict[str, Any]:
    if not path.exists():
        print(f"[WARN] YAML not found: {path}")
        return {}
    try:
        with path.open("r", encoding="utf-8") as file:
            data = yaml.safe_load(file)
    except Exception as error:
        print(f"[WARN] Failed to read YAML: {path} | {error}")
        return {}
    if data is None:
        return {}
    if not isinstance(data, dict):
        print(f"[WARN] YAML is not a dict: {path}")
        return {}
    return data


def read_csv_or_none(path: Path, **kwargs: Any) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    try:
        return pd.read_csv(path, **kwargs)
    except Exception as error:
        print(f"[WARN] Failed to read CSV: {path} | {error}")
        return None


def format_cell(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.6g}"
    return str(value)


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    columns = [str(column) for column in df.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for _, row in df.iterrows():
        values = [format_cell(row[column]) for column in df.columns]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def save_table_bundle(df: pd.DataFrame, output_dir: Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / f"{stem}.csv"
    md_path = output_dir / f"{stem}.md"
    tex_path = output_dir / f"{stem}.tex"

    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    md_path.write_text(dataframe_to_markdown(df), encoding="utf-8")

    try:
        tex_text = df.to_latex(index=False)
    except Exception as error:
        print(f"[WARN] Failed to render LaTeX for {stem}: {error}")
        tex_text = "% Failed to render LaTeX table.\n"
    tex_path.write_text(tex_text, encoding="utf-8")

    print(f"  [Table] {stem}: {csv_path}")


def to_float_or_na(value: Any) -> Any:
    if value is None or pd.isna(value):
        return pd.NA
    try:
        return float(value)
    except (TypeError, ValueError):
        return pd.NA


def first_existing_metric(row: Optional[pd.Series], names: List[str]) -> Any:
    if row is None:
        return pd.NA
    for name in names:
        if name in row.index:
            return row[name]
    return pd.NA


def metric_from_row(row: Optional[pd.Series], names: List[str]) -> Any:
    return to_float_or_na(first_existing_metric(row, names))


def metric_gap(val_value: Any, test_value: Any) -> Any:
    val_float = to_float_or_na(val_value)
    test_float = to_float_or_na(test_value)
    if pd.isna(val_float) or pd.isna(test_float):
        return pd.NA
    return float(val_float) - float(test_float)


def active_modalities_text(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        return "+".join(str(item) for item in value)
    if value is None:
        return ""
    return str(value)


def load_epoch_metrics(run_dir: Path) -> Optional[pd.DataFrame]:
    return read_csv_or_none(run_dir / "logs" / "epoch_metrics.csv")


def load_eval_metrics_row(run_dir: Path, split: str) -> Tuple[Optional[pd.Series], Path]:
    metrics_path = run_dir / "logs" / "evaluations" / f"{split}_best_model" / "metrics.csv"
    df = read_csv_or_none(metrics_path)
    if df is None or len(df) == 0:
        print(f"[WARN] Missing {split} metrics: {metrics_path}")
        return None, metrics_path
    return df.iloc[0], metrics_path


def choose_validation_best_epoch(run_dir: Path, epoch_df: Optional[pd.DataFrame]) -> Any:
    summary_path = run_dir / "figures" / "training_curves" / "best_epoch_summary.csv"
    summary_df = read_csv_or_none(summary_path)

    if summary_df is not None and len(summary_df) > 0:
        if {"metric", "best_epoch"}.issubset(summary_df.columns):
            metric_df = summary_df[summary_df["metric"] == BEST_SELECTION_METRIC]
            metric_df = metric_df.dropna(subset=["best_epoch"])
            if len(metric_df) > 0:
                return int(metric_df.iloc[0]["best_epoch"])

    if epoch_df is None or len(epoch_df) == 0:
        return pd.NA
    if "epoch" not in epoch_df.columns or BEST_SELECTION_METRIC not in epoch_df.columns:
        return pd.NA

    valid_df = epoch_df.dropna(subset=[BEST_SELECTION_METRIC])
    if len(valid_df) == 0:
        return pd.NA

    best_row = valid_df.loc[valid_df[BEST_SELECTION_METRIC].idxmax()]
    return int(best_row["epoch"])


def get_epoch_row(epoch_df: Optional[pd.DataFrame], epoch: Any) -> Optional[pd.Series]:
    if epoch_df is None or pd.isna(epoch) or "epoch" not in epoch_df.columns:
        return None
    match_df = epoch_df[epoch_df["epoch"] == int(epoch)]
    if len(match_df) == 0:
        return None
    return match_df.iloc[0]


def build_val_vs_test_summary(run_dir: Path) -> pd.DataFrame:
    run_id = run_dir.name
    epoch_df = load_epoch_metrics(run_dir)
    best_epoch = choose_validation_best_epoch(run_dir, epoch_df)
    best_row = get_epoch_row(epoch_df, best_epoch)
    test_row, test_metrics_path = load_eval_metrics_row(run_dir, "test")

    best_val_loss = metric_from_row(best_row, ["val_loss"])
    best_val_acc = metric_from_row(best_row, ["val_acc"])
    best_val_weighted_f1 = metric_from_row(best_row, ["val_weighted_f1"])
    best_val_macro_f1 = metric_from_row(best_row, ["val_macro_f1"])
    best_val_uar = metric_from_row(best_row, ["val_uar"])

    test_loss = metric_from_row(test_row, ["loss", "test_loss"])
    test_accuracy = metric_from_row(test_row, ["accuracy", "acc", "test_accuracy", "test_acc"])
    test_acc = metric_from_row(test_row, ["acc", "accuracy", "test_acc", "test_accuracy"])
    test_weighted_f1 = metric_from_row(test_row, ["weighted_f1", "test_weighted_f1"])
    test_macro_f1 = metric_from_row(test_row, ["macro_f1", "test_macro_f1"])
    test_uar = metric_from_row(test_row, ["uar", "test_uar"])

    row = {
        "run_id": run_id,
        "best_epoch": best_epoch,
        "best_selection_metric": BEST_SELECTION_METRIC,
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
    return pd.DataFrame([row], columns=VAL_VS_TEST_COLUMNS)


def fallback(summary_row: pd.Series, name: str) -> Any:
    if name not in summary_row.index:
        return pd.NA
    return summary_row[name]


def build_single_run_summary(run_dir: Path, val_vs_test_df: pd.DataFrame) -> pd.DataFrame:
    config = load_yaml_or_empty(run_dir / "logs" / "experiment_config.yaml")
    val_row, _ = load_eval_metrics_row(run_dir, "val")
    test_row, _ = load_eval_metrics_row(run_dir, "test")
    summary_row = val_vs_test_df.iloc[0] if len(val_vs_test_df) else pd.Series(dtype=object)

    training = config.get("training", {}) if isinstance(config.get("training"), dict) else {}
    legacy_train = config.get("train", {}) if isinstance(config.get("train"), dict) else {}
    model = config.get("model", {}) if isinstance(config.get("model"), dict) else {}

    row = {
        "run_id": run_dir.name,
        "experiment_name": safe_get(config, ["project", "experiment_name"]),
        "dataset": safe_get(config, ["dataset", "name"]),
        "model": safe_get(config, ["model", "name"]),
        "context_mode": safe_get(config, ["graph", "context_mode"]),
        "window_past": safe_get(config, ["graph", "window_past"]),
        "active_modalities": active_modalities_text(model.get("active_modalities", "")),
        "seed": safe_get(config, ["system", "seed"]),
        "epochs": training.get("epochs", legacy_train.get("max_epochs", "")),
        "batch_size": training.get("batch_size", legacy_train.get("batch_size", "")),
        "lr": training.get("lr", legacy_train.get("learning_rate", "")),
        "weight_decay": training.get("weight_decay", legacy_train.get("weight_decay", "")),
        "hidden_dim": safe_get(config, ["model", "hidden_dim"]),
        "dropout": safe_get(config, ["model", "dropout"]),
        "num_graph_layers": model.get("num_graph_layers", safe_get(config, ["graph", "num_layers"])),
        "modality_encoder_type": safe_get(config, ["model", "modality_encoder_type"]),
        "best_epoch": fallback(summary_row, "best_epoch"),
        "val_loss": metric_from_row(val_row, ["loss"]) if val_row is not None else fallback(summary_row, "best_val_loss"),
        "val_acc": metric_from_row(val_row, ["acc", "accuracy"]) if val_row is not None else fallback(summary_row, "best_val_acc"),
        "val_weighted_f1": metric_from_row(val_row, ["weighted_f1"]) if val_row is not None else fallback(summary_row, "best_val_weighted_f1"),
        "val_macro_f1": metric_from_row(val_row, ["macro_f1"]) if val_row is not None else fallback(summary_row, "best_val_macro_f1"),
        "val_uar": metric_from_row(val_row, ["uar"]) if val_row is not None else fallback(summary_row, "best_val_uar"),
        "test_loss": metric_from_row(test_row, ["loss"]) if test_row is not None else fallback(summary_row, "test_loss"),
        "test_acc": metric_from_row(test_row, ["acc", "accuracy"]) if test_row is not None else fallback(summary_row, "test_acc"),
        "test_weighted_f1": metric_from_row(test_row, ["weighted_f1"]) if test_row is not None else fallback(summary_row, "test_weighted_f1"),
        "test_macro_f1": metric_from_row(test_row, ["macro_f1"]) if test_row is not None else fallback(summary_row, "test_macro_f1"),
        "test_uar": metric_from_row(test_row, ["uar"]) if test_row is not None else fallback(summary_row, "test_uar"),
    }
    return pd.DataFrame([row], columns=SINGLE_RUN_SUMMARY_COLUMNS)


def clean_label_name(name: str) -> str:
    name = str(name)
    if name.startswith("true_"):
        return name[len("true_"):]
    if name.startswith("pred_"):
        return name[len("pred_"):]
    return name


def compute_per_class_metrics(confusion_df: pd.DataFrame) -> pd.DataFrame:
    matrix = confusion_df.values.astype(float)
    total = matrix.sum()
    true_labels = [clean_label_name(x) for x in confusion_df.index.tolist()]
    pred_labels = [clean_label_name(x) for x in confusion_df.columns.tolist()]

    if true_labels != pred_labels:
        print("[WARN] True/pred label orders differ in confusion matrix.")

    rows: List[Dict[str, Any]] = []
    for index, label_name in enumerate(true_labels):
        tp = matrix[index, index]
        fn = matrix[index, :].sum() - tp
        fp = matrix[:, index].sum() - tp
        tn = total - tp - fn - fp
        support = matrix[index, :].sum()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        one_vs_rest_acc = (tp + tn) / total if total > 0 else 0.0

        rows.append(
            {
                "label_id": int(index),
                "label_name": label_name,
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "support": int(support),
                "one_vs_rest_acc": float(one_vs_rest_acc),
            }
        )

    return pd.DataFrame(rows)


def load_per_class_metrics(run_dir: Path, split: str) -> Optional[pd.DataFrame]:
    eval_dir = run_dir / "logs" / "evaluations" / f"{split}_best_model"
    confusion_path = eval_dir / "confusion_matrix.csv"
    confusion_df = read_csv_or_none(confusion_path, index_col=0)
    if confusion_df is not None and len(confusion_df) > 0:
        return compute_per_class_metrics(confusion_df)

    if split == "test":
        final_path = run_dir / "figures" / "final_analysis" / "per_class_metrics_from_confusion_matrix.csv"
        final_df = read_csv_or_none(final_path)
        if final_df is not None and len(final_df) > 0:
            return final_df

    print(f"[WARN] Per-class metrics unavailable for split={split}.")
    return None


def caption_text() -> str:
    return (
        "Test metrics are final metrics of the validation-best checkpoint, "
        "not per-epoch process metrics."
    )


def save_val_vs_test_classification_figure(summary_df: pd.DataFrame, output_dir: Path) -> None:
    row = summary_df.iloc[0]
    plot_df = pd.DataFrame(
        {
            "metric": ["Acc", "Weighted-F1", "Macro-F1", "UAR"],
            "Validation": [
                to_float_or_na(row["best_val_acc"]),
                to_float_or_na(row["best_val_weighted_f1"]),
                to_float_or_na(row["best_val_macro_f1"]),
                to_float_or_na(row["best_val_uar"]),
            ],
            "Test": [
                to_float_or_na(row["test_acc"]),
                to_float_or_na(row["test_weighted_f1"]),
                to_float_or_na(row["test_macro_f1"]),
                to_float_or_na(row["test_uar"]),
            ],
        }
    ).set_index("metric")
    plot_df = plot_df.apply(pd.to_numeric, errors="coerce")

    ax = plot_df.plot(kind="bar", figsize=(8, 5))
    ax.set_ylabel("Metric Value")
    ax.set_ylim(0, 1)
    ax.set_title("Validation-Best Checkpoint: Validation vs Test Metrics")
    ax.legend(loc="best")
    plt.xticks(rotation=0)
    plt.figtext(0.5, 0.01, caption_text(), ha="center", fontsize=8)
    plt.tight_layout(rect=(0, 0.06, 1, 1))

    png_path = output_dir / "val_vs_test_classification_metrics.png"
    pdf_path = output_dir / "val_vs_test_classification_metrics.pdf"
    plt.savefig(png_path, dpi=300)
    plt.savefig(pdf_path)
    plt.close()
    print(f"  [Figure] val_vs_test_classification_metrics: {png_path}")


def save_val_vs_test_loss_figure(summary_df: pd.DataFrame, output_dir: Path) -> None:
    row = summary_df.iloc[0]
    losses = [
        to_float_or_na(row["best_val_loss"]),
        to_float_or_na(row["test_loss"]),
    ]
    plot_df = pd.DataFrame({"split": ["Validation", "Test"], "loss": losses})
    plot_df["loss"] = pd.to_numeric(plot_df["loss"], errors="coerce")

    plt.figure(figsize=(6, 5))
    plt.bar(plot_df["split"], plot_df["loss"])
    plt.ylabel("Loss")
    plt.title("Validation-Best Checkpoint: Validation vs Test Loss")
    plt.figtext(0.5, 0.01, caption_text(), ha="center", fontsize=8)
    plt.tight_layout(rect=(0, 0.06, 1, 1))

    png_path = output_dir / "val_vs_test_loss.png"
    pdf_path = output_dir / "val_vs_test_loss.pdf"
    plt.savefig(png_path, dpi=300)
    plt.savefig(pdf_path)
    plt.close()
    print(f"  [Figure] val_vs_test_loss: {png_path}")


def copy_if_exists(source: Path, target: Path) -> None:
    if not source.exists():
        print(f"[WARN] Figure source missing: {source}")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    print(f"  [Figure] copied: {target}")


def copy_paper_ready_figures(run_dir: Path, figures_dir: Path, split: str) -> None:
    final_dir = run_dir / "figures" / "final_analysis"
    curves_dir = run_dir / "figures" / "training_curves"

    mappings = [
        (final_dir / "confusion_matrix_raw.png", figures_dir / f"confusion_matrix_raw_{split}.png"),
        (final_dir / "confusion_matrix_raw.pdf", figures_dir / f"confusion_matrix_raw_{split}.pdf"),
        (final_dir / "confusion_matrix_normalized.png", figures_dir / f"confusion_matrix_normalized_{split}.png"),
        (final_dir / "confusion_matrix_normalized.pdf", figures_dir / f"confusion_matrix_normalized_{split}.pdf"),
        (final_dir / "per_class_recall.png", figures_dir / f"per_class_recall_{split}.png"),
        (final_dir / "per_class_recall.pdf", figures_dir / f"per_class_recall_{split}.pdf"),
        (final_dir / "overall_metrics.png", figures_dir / f"overall_{split}_metrics.png"),
        (final_dir / "overall_metrics.pdf", figures_dir / f"overall_{split}_metrics.pdf"),
        (curves_dir / "val_weighted_f1.png", figures_dir / "val_weighted_f1_curve.png"),
        (curves_dir / "val_weighted_f1.pdf", figures_dir / "val_weighted_f1_curve.pdf"),
        (curves_dir / "val_loss.png", figures_dir / "val_loss_curve.png"),
        (curves_dir / "val_loss.pdf", figures_dir / "val_loss_curve.pdf"),
    ]

    for source, target in mappings:
        copy_if_exists(source, target)


def split_list(primary_split: str, also_val: bool) -> List[str]:
    splits = [primary_split]
    if also_val and "val" not in splits:
        splits.append("val")
    return splits


def main() -> None:
    args = parse_args()
    run_dir = resolve_path(args.run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    output_dir = run_dir / args.output_subdir
    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"
    cases_dir = output_dir / "cases"
    for path in (tables_dir, figures_dir, cases_dir):
        path.mkdir(parents=True, exist_ok=True)

    print("=" * 100)
    print("Export paper-ready single-run artifacts")
    print("=" * 100)
    print("Project root:", PROJECT_ROOT)
    print("Run dir:", run_dir)
    print("Output dir:", output_dir)
    print("Primary split:", args.split)
    print("Also validation:", args.also_val)
    print("=" * 100)

    val_vs_test_df = build_val_vs_test_summary(run_dir)
    single_run_df = build_single_run_summary(run_dir, val_vs_test_df)

    save_table_bundle(single_run_df, tables_dir, "single_run_summary")
    save_table_bundle(val_vs_test_df, tables_dir, "val_vs_test_summary")

    for split in split_list(args.split, args.also_val):
        per_class_df = load_per_class_metrics(run_dir, split)
        if per_class_df is not None:
            save_table_bundle(per_class_df, tables_dir, f"per_class_metrics_{split}")

    save_val_vs_test_classification_figure(val_vs_test_df, figures_dir)
    save_val_vs_test_loss_figure(val_vs_test_df, figures_dir)
    copy_paper_ready_figures(run_dir, figures_dir, args.split)

    print("=" * 100)
    print("Finished paper artifact export.")
    print("=" * 100)


if __name__ == "__main__":
    main()

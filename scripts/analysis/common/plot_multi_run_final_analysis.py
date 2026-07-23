"""
Plot multi-run final evaluation analysis.

Purpose:
    Compare final evaluation results of multiple experiment runs.

Input:
    Config YAML:
        configs/analysis/multi_run_final_analysis.yaml

    Master tables:
        outputs/<YYYYMMDD>/analysis/analysis_tables/evaluation_master.csv

    Confusion matrices:
        outputs/<YYYYMMDD>/runs/<run_id>/logs/evaluations/<eval_name>/confusion_matrix.csv

Output:
    <analysis.output_dir>/final_analysis/
        selected_overall_metrics.csv
        selected_per_class_metrics_from_confusion_matrix.csv
        selected_run_ranking.csv

        overall_metrics_comparison.png / .pdf

        per_class_recall_comparison.png / .pdf
        per_class_precision_comparison.png / .pdf
        per_class_f1_comparison.png / .pdf
        per_class_support_comparison.png / .pdf
        per_class_one_vs_rest_acc_comparison.png / .pdf

Usage:
    python scripts/analysis/common/plot_multi_run_final_analysis.py \
      --config configs/analysis/multi_run_final_analysis.yaml

Important:
    If runs changed, rebuild master tables first:
        python scripts/analysis/common/build_analysis_tables.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List
import argparse
import re
import sys

import numpy as np
import pandas as pd
import yaml

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from utils.output_paths import (  # noqa: E402
    configured_output_root,
    find_analysis_artifact,
    find_run_directory,
    infer_experiment_date_from_run,
    resolve_experiment_date,
    resolve_output_category,
    sanitize_run_name,
)

OUTPUT_ROOT = PROJECT_ROOT / "outputs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot multi-run final evaluation analysis."
    )

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to multi-run final analysis YAML config.",
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--experiment-date", default=None)
    parser.add_argument("--evaluation-master", default=None)

    return parser.parse_args()


def resolve_path(path_text: str) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if data is None:
        raise RuntimeError(f"Empty config file: {path}")

    return data


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


def clean_label_name(name: str) -> str:
    name = str(name)

    if name.startswith("true_"):
        return name[len("true_"):]

    if name.startswith("pred_"):
        return name[len("pred_"):]

    return name


def get_runs_from_config(config: Dict[str, Any]) -> List[Dict[str, str]]:
    runs = config.get("runs", [])

    if not isinstance(runs, list) or len(runs) == 0:
        raise ValueError("Config field 'runs' must be a non-empty list.")

    normalized_runs: List[Dict[str, str]] = []

    for item in runs:
        if "run_id" not in item:
            raise ValueError("Each run item must contain 'run_id'.")

        run_id = str(item["run_id"])
        display_name = str(item.get("display_name", run_id))

        normalized_runs.append(
            {
                "run_id": run_id,
                "display_name": display_name,
            }
        )

    return normalized_runs


def get_split(config: Dict[str, Any]) -> str:
    return str(config.get("analysis", {}).get("split", "test"))


def get_output_dir(
    config: Dict[str, Any],
    explicit_output_dir: str | None,
    experiment_date: str,
) -> Path:
    analysis = config.get("analysis", {})
    output_root = resolve_path(str(configured_output_root(config)))
    if explicit_output_dir is not None:
        output_dir = resolve_path(explicit_output_dir)
    elif analysis.get("output_dir") is not None:
        output_dir = resolve_path(str(analysis["output_dir"])) / "final_analysis"
    else:
        name = sanitize_run_name(str(analysis.get("name", "multi_run_final_analysis")))
        output_dir = (
            resolve_output_category("analysis", experiment_date, output_root)
            / name
            / "final_analysis"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    return output_dir


def get_figure_config(config: Dict[str, Any]) -> Dict[str, Any]:
    figure_config = config.get("figure", {})

    return {
        "save_png": bool(figure_config.get("save_png", True)),
        "save_pdf": bool(figure_config.get("save_pdf", True)),
        "dpi": int(figure_config.get("dpi", 300)),
    }


def get_overall_metrics(config: Dict[str, Any]) -> List[str]:
    metrics = config.get("overall", {}).get(
        "metrics",
        ["acc", "uar", "macro_f1", "weighted_f1", "loss"],
    )

    if not isinstance(metrics, list) or len(metrics) == 0:
        raise ValueError("overall.metrics must be a non-empty list.")

    return [str(metric) for metric in metrics]


def get_per_class_metrics(config: Dict[str, Any]) -> List[str]:
    metrics = config.get("per_class", {}).get(
        "metrics",
        ["recall", "precision", "f1", "support", "one_vs_rest_acc"],
    )

    if not isinstance(metrics, list) or len(metrics) == 0:
        raise ValueError("per_class.metrics must be a non-empty list.")

    return [str(metric) for metric in metrics]


def get_per_class_title(config: Dict[str, Any], metric: str) -> str:
    titles = config.get("per_class", {}).get("titles", {})
    return str(titles.get(metric, f"Per-class {metric_display_name(metric)}"))


def get_per_class_filename(config: Dict[str, Any], metric: str) -> str:
    filenames = config.get("per_class", {}).get("filenames", {})
    return sanitize_filename(filenames.get(metric, f"per_class_{metric}_comparison"))


def load_evaluation_master(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    required_columns = ["run_id", "split"]

    for column in required_columns:
        if column not in df.columns:
            raise ValueError(
                f"Missing required column '{column}' in {path}"
            )

    return df


def select_overall_rows(
    evaluation_df: pd.DataFrame,
    runs: List[Dict[str, str]],
    split: str,
) -> pd.DataFrame:
    run_ids = [item["run_id"] for item in runs]
    display_name_map = {
        item["run_id"]: item["display_name"]
        for item in runs
    }

    selected_df = evaluation_df[
        (evaluation_df["run_id"].isin(run_ids))
        & (evaluation_df["split"] == split)
    ].copy()

    found_run_ids = set(selected_df["run_id"].unique().tolist())

    missing = [
        run_id
        for run_id in run_ids
        if run_id not in found_run_ids
    ]

    if missing:
        print("[WARN] Some run IDs were not found in evaluation_master.csv for split =", split)
        for run_id in missing:
            print("  ", run_id)

    if len(selected_df) == 0:
        raise RuntimeError(
            f"No evaluation rows found for selected runs and split={split}."
        )

    selected_df["display_name"] = selected_df["run_id"].map(display_name_map)

    ordered_rows = []

    for run_id in run_ids:
        run_df = selected_df[selected_df["run_id"] == run_id].copy()

        if len(run_df) == 0:
            continue

        if len(run_df) > 1:
            print(f"[WARN] Multiple evaluation rows found for {run_id}, split={split}. Use the first row.")
            run_df = run_df.head(1)

        ordered_rows.append(run_df.iloc[0].to_dict())

    return pd.DataFrame(ordered_rows)


def find_confusion_matrix_path(overall_row: pd.Series) -> Path:
    eval_dir = Path(str(overall_row["eval_dir"]))
    confusion_path = eval_dir / "confusion_matrix.csv"

    if not confusion_path.exists():
        raise FileNotFoundError(
            f"Missing confusion matrix: {confusion_path}"
        )

    return confusion_path


def load_confusion_matrix(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0)

    if df.shape[0] == 0 or df.shape[1] == 0:
        raise RuntimeError(f"Empty confusion matrix: {path}")

    return df


def compute_per_class_metrics(confusion_df: pd.DataFrame) -> pd.DataFrame:
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


def build_selected_per_class_metrics(overall_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[pd.DataFrame] = []

    for _, overall_row in overall_df.iterrows():
        confusion_path = find_confusion_matrix_path(overall_row)
        confusion_df = load_confusion_matrix(confusion_path)

        per_class_df = compute_per_class_metrics(confusion_df)

        per_class_df.insert(0, "display_name", overall_row["display_name"])
        per_class_df.insert(0, "run_id", overall_row["run_id"])
        per_class_df.insert(0, "split", overall_row["split"])

        rows.append(per_class_df)

    if len(rows) == 0:
        raise RuntimeError("No per-class metrics were computed.")

    return pd.concat(rows, ignore_index=True)


def save_dataframe(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print("Saved:", path)


def plot_overall_metrics(
    overall_df: pd.DataFrame,
    metrics: List[str],
    config: Dict[str, Any],
    output_dir: Path,
    figure_config: Dict[str, Any],
) -> None:
    available_metrics = [
        metric
        for metric in metrics
        if metric in overall_df.columns
    ]

    if len(available_metrics) == 0:
        print("[WARN] No requested overall metrics found. Skip overall plot.")
        return

    title = str(config.get("overall", {}).get("title", "Overall Metrics Comparison"))
    filename = sanitize_filename(
        config.get("overall", {}).get("filename", "overall_metrics_comparison")
    )

    display_names = overall_df["display_name"].tolist()

    x = np.arange(len(display_names))
    width = 0.8 / len(available_metrics)

    plt.figure(figsize=(max(8, len(display_names) * 1.5), 5))

    for idx, metric in enumerate(available_metrics):
        values = overall_df[metric].astype(float).values
        positions = x - 0.4 + width / 2 + idx * width

        plt.bar(
            positions,
            values,
            width,
            label=metric_display_name(metric),
        )

    plt.xticks(x, display_names, rotation=25, ha="right")
    plt.ylabel("Metric Value")
    plt.title(title)
    plt.legend()
    plt.tight_layout()

    if figure_config["save_png"]:
        png_path = output_dir / f"{filename}.png"
        plt.savefig(png_path, dpi=figure_config["dpi"])
        print("  [PNG] overall:", png_path)

    if figure_config["save_pdf"]:
        pdf_path = output_dir / f"{filename}.pdf"
        plt.savefig(pdf_path)
        print("  [PDF] overall:", pdf_path)

    plt.close()


def plot_per_class_metric(
    per_class_df: pd.DataFrame,
    runs: List[Dict[str, str]],
    metric: str,
    title: str,
    filename: str,
    output_dir: Path,
    figure_config: Dict[str, Any],
) -> None:
    if metric not in per_class_df.columns:
        print(f"[WARN] Per-class metric '{metric}' not found. Skip.")
        return

    metric_df = per_class_df.dropna(subset=[metric]).copy()

    if len(metric_df) == 0:
        print(f"[WARN] Per-class metric '{metric}' has no valid values. Skip.")
        return

    label_order = (
        metric_df[["label_id", "label_name"]]
        .drop_duplicates()
        .sort_values("label_id")["label_name"]
        .tolist()
    )

    display_order = [item["display_name"] for item in runs]

    pivot_df = metric_df.pivot_table(
        index="label_name",
        columns="display_name",
        values=metric,
        aggfunc="mean",
    )

    pivot_df = pivot_df.reindex(label_order)
    pivot_df = pivot_df.reindex(columns=display_order)

    x = np.arange(len(label_order))
    num_runs = len(display_order)
    width = 0.8 / max(num_runs, 1)

    plt.figure(figsize=(max(9, len(label_order) * 1.3), 5))

    for idx, display_name in enumerate(display_order):
        if display_name not in pivot_df.columns:
            continue

        values = pivot_df[display_name].values
        positions = x - 0.4 + width / 2 + idx * width

        plt.bar(
            positions,
            values,
            width,
            label=display_name,
        )

    plt.xticks(x, label_order, rotation=30, ha="right")
    plt.xlabel("Emotion Class")
    plt.ylabel(metric_display_name(metric))
    plt.title(title)

    if metric != "support":
        plt.ylim(0, 1)

    plt.legend()
    plt.tight_layout()

    if figure_config["save_png"]:
        png_path = output_dir / f"{filename}.png"
        plt.savefig(png_path, dpi=figure_config["dpi"])
        print(f"  [PNG] per-class {metric}: {png_path}")

    if figure_config["save_pdf"]:
        pdf_path = output_dir / f"{filename}.pdf"
        plt.savefig(pdf_path)
        print(f"  [PDF] per-class {metric}: {pdf_path}")

    plt.close()


def build_ranking_table(
    overall_df: pd.DataFrame,
    config: Dict[str, Any],
) -> pd.DataFrame:
    ranking_config = config.get("ranking", {})

    primary_metric = str(ranking_config.get("primary_metric", "macro_f1"))
    secondary_metric = str(ranking_config.get("secondary_metric", "uar"))

    higher_is_better = ranking_config.get(
        "higher_is_better",
        {
            "loss": False,
            "acc": True,
            "uar": True,
            "macro_f1": True,
            "weighted_f1": True,
        },
    )

    ranking_df = overall_df.copy()

    sort_columns = []
    ascending = []

    for metric in [primary_metric, secondary_metric]:
        if metric in ranking_df.columns:
            sort_columns.append(metric)
            ascending.append(not bool(higher_is_better.get(metric, True)))

    if len(sort_columns) == 0:
        print("[WARN] Ranking metrics not found. Return unsorted ranking table.")
        return ranking_df

    ranking_df = ranking_df.sort_values(
        by=sort_columns,
        ascending=ascending,
    ).reset_index(drop=True)

    ranking_df.insert(0, "rank", np.arange(1, len(ranking_df) + 1))

    return ranking_df


def main() -> None:
    args = parse_args()

    config_path = resolve_path(args.config)
    config = load_yaml(config_path)

    runs = get_runs_from_config(config)
    split = get_split(config)
    try:
        first_run = find_run_directory(str(runs[0]["run_id"]), OUTPUT_ROOT)
        inferred_date = infer_experiment_date_from_run(first_run)
    except (FileNotFoundError, RuntimeError):
        inferred_date = None
    frozen_date = resolve_experiment_date(
        cli_date=args.experiment_date,
        config=config,
        inferred_date=inferred_date,
    )
    output_dir = get_output_dir(config, args.output_dir, frozen_date)
    figure_config = get_figure_config(config)
    overall_metrics = get_overall_metrics(config)
    per_class_metrics = get_per_class_metrics(config)

    evaluation_master = (
        resolve_path(args.evaluation_master)
        if args.evaluation_master is not None
        else find_analysis_artifact(
            "analysis_tables/evaluation_master.csv", OUTPUT_ROOT
        )
    )
    evaluation_df = load_evaluation_master(evaluation_master)
    selected_overall_df = select_overall_rows(
        evaluation_df=evaluation_df,
        runs=runs,
        split=split,
    )

    selected_per_class_df = build_selected_per_class_metrics(selected_overall_df)

    ranking_df = build_ranking_table(
        overall_df=selected_overall_df,
        config=config,
    )

    print("=" * 100)
    print("Plot multi-run final evaluation analysis")
    print("=" * 100)
    print("Project root:", PROJECT_ROOT)
    print("Config:", config_path)
    print("Evaluation master:", evaluation_master)
    print("Output dir:", output_dir)
    print("Split:", split)
    print("Selected runs:")
    for item in runs:
        print(f"  {item['run_id']} -> {item['display_name']}")
    print("Overall metrics:", overall_metrics)
    print("Per-class metrics:", per_class_metrics)
    print("=" * 100)

    save_dataframe(
        selected_overall_df,
        output_dir / "selected_overall_metrics.csv",
    )

    save_dataframe(
        selected_per_class_df,
        output_dir / "selected_per_class_metrics_from_confusion_matrix.csv",
    )

    save_dataframe(
        ranking_df,
        output_dir / "selected_run_ranking.csv",
    )

    print("\nPlot overall metrics:")
    plot_overall_metrics(
        overall_df=selected_overall_df,
        metrics=overall_metrics,
        config=config,
        output_dir=output_dir,
        figure_config=figure_config,
    )

    print("\nPlot per-class metrics:")
    for metric in per_class_metrics:
        title = get_per_class_title(config, metric)
        filename = get_per_class_filename(config, metric)

        plot_per_class_metric(
            per_class_df=selected_per_class_df,
            runs=runs,
            metric=metric,
            title=title,
            filename=filename,
            output_dir=output_dir,
            figure_config=figure_config,
        )

    print("=" * 100)
    print("Finished.")
    print("=" * 100)


if __name__ == "__main__":
    main()

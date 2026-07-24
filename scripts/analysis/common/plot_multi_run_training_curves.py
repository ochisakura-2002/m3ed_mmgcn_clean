"""
Plot multi-run training curves by epoch.

Purpose:
    Compare training/validation curves of multiple experiment runs.
    Each metric is saved as an independent figure.

Input:
    Config YAML, e.g.
        configs/benchmarks/paper/analysis/multi_run_curves.yaml

    Master table:
        outputs/<YYYYMMDD>/analysis/analysis_tables/epoch_metrics_master.csv

Output:
    <analysis.output_dir>/training_curves/
        <metric_filename>.png
        <metric_filename>.pdf
        selected_epoch_metrics.csv

Usage:
    python scripts/analysis/common/plot_multi_run_training_curves.py \
      --config configs/benchmarks/paper/analysis/multi_run_curves.yaml

Important:
    Run this before plotting if runs changed:
        python scripts/analysis/common/build_analysis_tables.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List
import argparse
import re
import sys

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
        description="Plot multi-run training curves by epoch."
    )

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to multi-run training curve YAML config.",
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--experiment-date", default=None)
    parser.add_argument("--epoch-master", default=None)

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


def get_metric_list(config: Dict[str, Any]) -> List[str]:
    metrics = (
        config
        .get("training_curves", {})
        .get("metrics", [])
    )

    if not isinstance(metrics, list) or len(metrics) == 0:
        raise ValueError(
            "Config field training_curves.metrics must be a non-empty list."
        )

    return [str(metric) for metric in metrics]


def get_title(config: Dict[str, Any], metric: str) -> str:
    titles = (
        config
        .get("training_curves", {})
        .get("titles", {})
    )

    return str(titles.get(metric, metric.replace("_", " ").title()))


def get_filename(config: Dict[str, Any], metric: str) -> str:
    filenames = (
        config
        .get("training_curves", {})
        .get("filenames", {})
    )

    return sanitize_filename(filenames.get(metric, metric))


def get_output_dir(
    config: Dict[str, Any],
    explicit_output_dir: str | None,
    experiment_date: str,
) -> Path:
    analysis_config = config.get("analysis", {})
    output_root = resolve_path(str(configured_output_root(config)))
    if explicit_output_dir is not None:
        output_dir = resolve_path(explicit_output_dir)
    elif analysis_config.get("output_dir") is not None:
        output_dir = resolve_path(str(analysis_config["output_dir"])) / "training_curves"
    else:
        name = sanitize_run_name(
            str(analysis_config.get("name", "multi_run_training_curves"))
        )
        output_dir = (
            resolve_output_category("analysis", experiment_date, output_root)
            / name
            / "training_curves"
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


def load_epoch_master(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    required_columns = ["run_id", "epoch"]

    for column in required_columns:
        if column not in df.columns:
            raise ValueError(
                f"Missing required column '{column}' in {path}"
            )

    return df


def filter_selected_runs(
    epoch_df: pd.DataFrame,
    runs: List[Dict[str, str]],
) -> pd.DataFrame:
    selected_run_ids = [item["run_id"] for item in runs]

    filtered_df = epoch_df[
        epoch_df["run_id"].isin(selected_run_ids)
    ].copy()

    found_run_ids = set(filtered_df["run_id"].unique().tolist())
    missing_run_ids = [
        run_id
        for run_id in selected_run_ids
        if run_id not in found_run_ids
    ]

    if missing_run_ids:
        print("[WARN] Some run IDs were not found in epoch_metrics_master.csv:")
        for run_id in missing_run_ids:
            print("  ", run_id)

    if len(filtered_df) == 0:
        raise RuntimeError(
            "No epoch rows found for selected runs. "
            "Check run IDs or rebuild analysis tables."
        )

    display_name_map = {
        item["run_id"]: item["display_name"]
        for item in runs
    }

    filtered_df["display_name"] = filtered_df["run_id"].map(display_name_map)

    return filtered_df


def save_selected_epoch_metrics(
    selected_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    output_path = output_dir / "selected_epoch_metrics.csv"
    selected_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print("Saved selected epoch metrics:", output_path)


def plot_one_metric(
    selected_df: pd.DataFrame,
    runs: List[Dict[str, str]],
    metric: str,
    title: str,
    filename: str,
    output_dir: Path,
    figure_config: Dict[str, Any],
) -> None:
    if metric not in selected_df.columns:
        print(f"[WARN] Metric '{metric}' not found. Skip.")
        return

    metric_df = selected_df.dropna(subset=[metric]).copy()

    if len(metric_df) == 0:
        print(f"[WARN] Metric '{metric}' has no valid values. Skip.")
        return

    max_epoch = int(metric_df["epoch"].max())

    plt.figure(figsize=(9, 5))

    for item in runs:
        run_id = item["run_id"]
        display_name = item["display_name"]

        run_df = metric_df[metric_df["run_id"] == run_id].copy()

        if len(run_df) == 0:
            continue

        run_df = run_df.sort_values("epoch")

        plt.plot(
            run_df["epoch"],
            run_df[metric],
            marker="o",
            label=display_name,
        )

    plt.xlabel("Epoch")
    plt.ylabel(title)
    plt.title(title)
    plt.xlim(1, max_epoch)

    if metric != "train_loss" and metric != "val_loss" and "loss" not in metric.lower():
        plt.ylim(0, 1)

    plt.legend()
    plt.tight_layout()

    if figure_config["save_png"]:
        png_path = output_dir / f"{filename}.png"
        plt.savefig(png_path, dpi=figure_config["dpi"])
        print(f"  [PNG] {metric}: {png_path}")

    if figure_config["save_pdf"]:
        pdf_path = output_dir / f"{filename}.pdf"
        plt.savefig(pdf_path)
        print(f"  [PDF] {metric}: {pdf_path}")

    plt.close()


def main() -> None:
    args = parse_args()

    config_path = resolve_path(args.config)
    config = load_yaml(config_path)

    runs = get_runs_from_config(config)
    metrics = get_metric_list(config)
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

    epoch_master = (
        resolve_path(args.epoch_master)
        if args.epoch_master is not None
        else find_analysis_artifact(
            "analysis_tables/epoch_metrics_master.csv", OUTPUT_ROOT
        )
    )
    epoch_df = load_epoch_master(epoch_master)
    selected_df = filter_selected_runs(epoch_df, runs)

    print("=" * 100)
    print("Plot multi-run training curves")
    print("=" * 100)
    print("Project root:", PROJECT_ROOT)
    print("Config:", config_path)
    print("Epoch master:", epoch_master)
    print("Output dir:", output_dir)
    print("Selected runs:")
    for item in runs:
        print(f"  {item['run_id']} -> {item['display_name']}")
    print("Metrics:", metrics)
    print("=" * 100)

    save_selected_epoch_metrics(
        selected_df=selected_df,
        output_dir=output_dir,
    )

    for metric in metrics:
        title = get_title(config, metric)
        filename = get_filename(config, metric)

        plot_one_metric(
            selected_df=selected_df,
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

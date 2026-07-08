#!/usr/bin/env python
"""
Plot missing-modality evaluation summary.

Input:
    outputs/runs/<run_id>/logs/missing_modalities/test_best_model/summary.csv

Output:
    outputs/runs/<run_id>/figures/missing_modalities/test_best_model/
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import pandas as pd


ORDER: List[str] = [
    "TAV",
    "TA",
    "missing_visual",
    "TV",
    "missing_audio",
    "AV",
    "missing_text",
    "T",
    "A",
    "V",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot missing-modality evaluation summary."
    )
    parser.add_argument(
        "--summary",
        type=str,
        required=True,
        help="Path to missing modality summary.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help=(
            "Figure output directory. Default: "
            "<run_dir>/figures/missing_modalities/<split_checkpoint>/"
        ),
    )
    return parser.parse_args()


def infer_output_dir(summary_path: Path) -> Path:
    # summary path:
    # <run_dir>/logs/missing_modalities/test_best_model/summary.csv
    stage_name = summary_path.parent.name
    run_dir = summary_path.parents[3]
    return run_dir / "figures" / "missing_modalities" / stage_name


def sort_settings(df: pd.DataFrame) -> pd.DataFrame:
    order_map = {name: idx for idx, name in enumerate(ORDER)}

    df = df.copy()
    df["_order"] = df["setting"].map(order_map)
    df = df.sort_values("_order").drop(columns=["_order"])

    return df


def save_figure(path_without_suffix: Path, dpi: int = 300) -> None:
    plt.tight_layout()
    plt.savefig(path_without_suffix.with_suffix(".png"), dpi=dpi)
    plt.savefig(path_without_suffix.with_suffix(".pdf"))
    plt.close()


def plot_metric_bar(
    df: pd.DataFrame,
    metric: str,
    title: str,
    ylabel: str,
    output_path: Path,
) -> None:
    plt.figure(figsize=(8, 5))
    plt.bar(
        df["setting"].astype(str),
        df[metric].astype(float),
    )
    plt.xlabel("Test-time active modalities")
    plt.ylabel(ylabel)
    plt.title(title)
    save_figure(output_path)


def plot_core_metrics(df: pd.DataFrame, output_path: Path) -> None:
    metrics = ["acc", "uar", "macro_f1", "weighted_f1"]

    x = list(range(len(df)))
    width = 0.18

    plt.figure(figsize=(10, 5))

    for idx, metric in enumerate(metrics):
        offsets = [value + (idx - 1.5) * width for value in x]
        plt.bar(
            offsets,
            df[metric].astype(float),
            width=width,
            label=metric,
        )

    plt.xticks(
        x,
        df["setting"].astype(str),
    )
    plt.xlabel("Test-time active modalities")
    plt.ylabel("Score")
    plt.title("Missing-modality evaluation: core metrics")
    plt.legend()
    save_figure(output_path)


def plot_drop_from_tav(
    df: pd.DataFrame,
    metric: str,
    output_path: Path,
) -> None:
    if "TAV" not in set(df["setting"]):
        print(f"[Skip] TAV not found. Cannot plot drop for {metric}.")
        return

    tav_value = float(df.loc[df["setting"] == "TAV", metric].iloc[0])

    plot_df = df.copy()
    drop_col = f"{metric}_drop_from_TAV"
    plot_df[drop_col] = tav_value - plot_df[metric].astype(float)

    plt.figure(figsize=(8, 5))
    plt.bar(
        plot_df["setting"].astype(str),
        plot_df[drop_col].astype(float),
    )
    plt.axhline(0.0, linewidth=1)
    plt.xlabel("Test-time active modalities")
    plt.ylabel(f"{metric} drop from TAV")
    plt.title(f"Missing-modality degradation relative to TAV: {metric}")
    save_figure(output_path)


def main() -> None:
    args = parse_args()

    summary_path = Path(args.summary).resolve()

    if not summary_path.exists():
        raise FileNotFoundError(f"Summary file not found: {summary_path}")

    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir is not None
        else infer_output_dir(summary_path)
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(summary_path)

    required = {
        "setting",
        "loss",
        "acc",
        "uar",
        "macro_f1",
        "weighted_f1",
    }
    missing = required - set(df.columns)

    if missing:
        raise RuntimeError(
            f"Missing required columns in {summary_path}: {sorted(missing)}"
        )

    df = sort_settings(df)

    sorted_csv = output_dir / "summary_sorted.csv"
    df.to_csv(
        sorted_csv,
        index=False,
        encoding="utf-8-sig",
    )

    plot_metric_bar(
        df=df,
        metric="weighted_f1",
        title="Missing-modality evaluation: weighted F1",
        ylabel="Weighted F1",
        output_path=output_dir / "weighted_f1_bar",
    )

    plot_core_metrics(
        df=df,
        output_path=output_dir / "core_metrics_bar",
    )

    plot_drop_from_tav(
        df=df,
        metric="weighted_f1",
        output_path=output_dir / "weighted_f1_drop_from_TAV",
    )

    plot_drop_from_tav(
        df=df,
        metric="macro_f1",
        output_path=output_dir / "macro_f1_drop_from_TAV",
    )

    plot_metric_bar(
        df=df,
        metric="loss",
        title="Missing-modality evaluation: loss",
        ylabel="Loss",
        output_path=output_dir / "loss_bar",
    )

    print("=" * 100)
    print(f"Saved missing-modality figures to: {output_dir}")
    print("=" * 100)

    for path in sorted(output_dir.iterdir()):
        print(path)


if __name__ == "__main__":
    main()

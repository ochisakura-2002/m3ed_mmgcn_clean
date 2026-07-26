"""
Rebuild a dated experiment_summary.csv from existing run directories.

这个脚本用于修复或重建按日期归档的 experiment_summary.csv。

为什么需要它：
不同训练脚本可能写入不同数量的字段。
如果直接 append 到旧 CSV，容易出现表头和数据列数不一致的问题。

这个脚本会扫描：
    outputs/<YYYYMMDD>/runs/*/logs/experiment_config.yaml
    outputs/<YYYYMMDD>/runs/*/logs/epoch_metrics.csv

Legacy run directories are discovered as a compatibility fallback.

然后重建统一 schema 的：
    outputs/<YYYYMMDD>/analysis/experiment_summary/experiment_summary.csv

运行方式：
    python scripts/maintenance/rebuild_experiment_summary.py
"""

from pathlib import Path
import argparse
import shutil
import sys
from datetime import datetime
from typing import Dict, Any, List

import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from utils.output_paths import (  # noqa: E402
    discover_run_directories,
    infer_experiment_date_from_run,
    infer_experiment_group_from_run,
    resolve_experiment_date,
    resolve_output_category,
)

OUTPUT_ROOT = PROJECT_ROOT / "outputs"


SUMMARY_COLUMNS = [
    "run_id",
    "experiment_name",
    "dataset",
    "model",
    "feature_pkl_path",
    "batch_size",
    "learning_rate",
    "weight_decay",
    "max_epochs",

    "graph_context_mode",
    "graph_window_past",
    "graph_window_future",
    "graph_num_layers",
    "graph_lamda",
    "graph_alpha",
    "graph_use_speaker",
    "graph_use_modal",
    "graph_use_residual",

    "best_epoch",
    "best_val_loss",
    "best_val_acc",
    "best_val_uar",
    "best_val_macro_f1",
    "best_val_weighted_f1",

    "run_dir",
]


def load_yaml(path: Path) -> Dict[str, Any]:
    """读取 YAML。"""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def find_best_epoch_row(
    metrics_df: pd.DataFrame,
    monitor_metric: str,
) -> pd.Series:
    """
    根据 monitor_metric 找 best epoch。

    train 脚本里一般使用 val_uar。
    如果 monitor_metric 不存在，则退回 val_uar。
    如果 val_uar 也不存在，则退回最后一行。
    """
    if monitor_metric in metrics_df.columns:
        index = metrics_df[monitor_metric].astype(float).idxmax()
        return metrics_df.loc[index]

    if "val_uar" in metrics_df.columns:
        index = metrics_df["val_uar"].astype(float).idxmax()
        return metrics_df.loc[index]

    return metrics_df.iloc[-1]


def build_summary_row(run_dir: Path) -> Dict[str, Any]:
    """
    从一个 run_dir 构建 summary row。
    """
    config_path = run_dir / "logs" / "experiment_config.yaml"
    metrics_path = run_dir / "logs" / "epoch_metrics.csv"

    if not config_path.exists():
        raise FileNotFoundError(f"Missing config: {config_path}")

    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing metrics: {metrics_path}")

    config = load_yaml(config_path)
    metrics_df = pd.read_csv(metrics_path)

    monitor_metric = config.get("logging", {}).get("monitor_metric", "val_uar")
    best_row = find_best_epoch_row(metrics_df, monitor_metric)

    graph_config = config.get("graph", {})

    row = {
        "run_id": run_dir.name,
        "experiment_name": config.get("project", {}).get("experiment_name", ""),
        "dataset": config.get("dataset", {}).get("name", ""),
        "model": config.get("model", {}).get("name", ""),
        "feature_pkl_path": config.get("dataset", {}).get("feature_pkl_path", ""),
        "batch_size": config.get("train", {}).get("batch_size", ""),
        "learning_rate": config.get("train", {}).get("learning_rate", ""),
        "weight_decay": config.get("train", {}).get("weight_decay", ""),
        "max_epochs": config.get("train", {}).get("max_epochs", ""),

        "graph_context_mode": graph_config.get("context_mode", ""),
        "graph_window_past": graph_config.get("window_past", ""),
        "graph_window_future": graph_config.get("window_future", ""),
        "graph_num_layers": graph_config.get("num_layers", ""),
        "graph_lamda": graph_config.get("lamda", ""),
        "graph_alpha": graph_config.get("alpha", ""),
        "graph_use_speaker": graph_config.get("use_speaker", ""),
        "graph_use_modal": graph_config.get("use_modal", ""),
        "graph_use_residual": graph_config.get("use_residual", ""),

        "best_epoch": int(best_row["epoch"]),
        "best_val_loss": float(best_row["val_loss"]),
        "best_val_acc": float(best_row["val_acc"]),
        "best_val_uar": float(best_row["val_uar"]),
        "best_val_macro_f1": float(best_row["val_macro_f1"]),
        "best_val_weighted_f1": float(best_row["val_weighted_f1"]),

        "run_dir": str(run_dir),
    }

    return row


def backup_old_summary(summary_path: Path) -> None:
    """备份旧的 experiment_summary.csv。"""
    if not summary_path.exists():
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = summary_path.with_suffix(f".backup_{timestamp}.csv")

    shutil.copy2(summary_path, backup_path)

    print("Old summary backed up to:", backup_path)


def main() -> None:
    """主函数。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--experiment-date", default=None)
    args = parser.parse_args()
    run_dirs = discover_run_directories(OUTPUT_ROOT)
    inferred_date = next(
        (
            value
            for value in (infer_experiment_date_from_run(path) for path in run_dirs)
            if value is not None
        ),
        None,
    )
    frozen_date = resolve_experiment_date(
        cli_date=args.experiment_date,
        inferred_date=inferred_date,
    )
    experiment_group = next(
        (
            value
            for value in (
                infer_experiment_group_from_run(path) for path in run_dirs
            )
            if value is not None
        ),
        "cross_group_experiment_summary",
    )
    summary_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir is not None
        else resolve_output_category(
            "analysis",
            frozen_date,
            OUTPUT_ROOT,
            experiment_group=experiment_group,
        )
        / "experiment_summary"
    )
    summary_path = summary_dir / "experiment_summary.csv"
    print("=" * 80)
    print("Rebuild experiment_summary.csv")
    print("=" * 80)

    print("Project root:", PROJECT_ROOT)
    print("Runs root:", OUTPUT_ROOT)
    print("Summary path:", summary_path)

    rows: List[Dict[str, Any]] = []

    for run_dir in run_dirs:
        config_path = run_dir / "logs" / "experiment_config.yaml"
        metrics_path = run_dir / "logs" / "epoch_metrics.csv"

        if not config_path.exists() or not metrics_path.exists():
            print("[SKIP]", run_dir.name, "missing config or metrics")
            continue

        try:
            row = build_summary_row(run_dir)
            rows.append(row)
            print("[OK]", run_dir.name)
        except Exception as error:
            print("[SKIP/ERROR]", run_dir.name, repr(error))

    if len(rows) == 0:
        raise RuntimeError("No valid runs found. Summary not rebuilt.")

    backup_old_summary(summary_path)

    summary_df = pd.DataFrame(rows)

    for column in SUMMARY_COLUMNS:
        if column not in summary_df.columns:
            summary_df[column] = ""

    summary_df = summary_df[SUMMARY_COLUMNS]

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")

    print("\nSummary rebuilt successfully.")
    print("Number of runs:", len(summary_df))
    print("Saved to:", summary_path)
    print("=" * 80)


if __name__ == "__main__":
    main()

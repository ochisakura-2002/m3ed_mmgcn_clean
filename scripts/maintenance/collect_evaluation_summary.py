"""
Collect evaluation results into a dated evaluation_summary.csv.

这个脚本扫描所有 run 目录下的 evaluation 结果：

    outputs/<YYYYMMDD>/runs/*/logs/evaluations/*/metrics.csv

Legacy run directories are discovered as a compatibility fallback.

并汇总成统一表格：

    outputs/<YYYYMMDD>/analysis/evaluation_summary/evaluation_summary.csv

用途：
1. 对比 SimpleMLP / MMGCN / 后续 causal-MMGCN
2. 汇总 val / test 指标
3. 为组会和论文实验表格提供统一结果来源

运行方式：
    python scripts/maintenance/collect_evaluation_summary.py
"""

from pathlib import Path
import argparse
import sys
from typing import Dict, Any, List

import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from utils.output_paths import (  # noqa: E402
    discover_run_directories,
    infer_experiment_date_from_run,
    resolve_experiment_date,
    resolve_output_category,
)

OUTPUT_ROOT = PROJECT_ROOT / "outputs"


SUMMARY_COLUMNS = [
    "run_id",
    "experiment_name",
    "dataset",
    "model",
    "split",
    "loss",
    "acc",
    "uar",
    "macro_f1",
    "weighted_f1",

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

    "checkpoint",
    "run_dir",
    "eval_dir",
]


def load_yaml(path: Path) -> Dict[str, Any]:
    """读取 YAML。"""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_row_from_metrics(metrics_path: Path) -> Dict[str, Any]:
    """
    从一个 metrics.csv 构建 summary row。
    """
    eval_dir = metrics_path.parent

    # metrics.csv 路径：
    # <run_dir>/logs/evaluations/<eval_name>/metrics.csv
    run_dir = eval_dir.parents[2]
    run_id = run_dir.name

    config_path = run_dir / "logs" / "experiment_config.yaml"

    if not config_path.exists():
        raise FileNotFoundError(f"Missing experiment config: {config_path}")

    config = load_yaml(config_path)
    metrics_df = pd.read_csv(metrics_path)

    if len(metrics_df) != 1:
        raise ValueError(f"Expected one row in metrics.csv, got {len(metrics_df)}: {metrics_path}")

    metrics = metrics_df.iloc[0].to_dict()
    graph_config = config.get("graph", {})

    row = {
        "run_id": run_id,
        "experiment_name": config.get("project", {}).get("experiment_name", ""),
        "dataset": config.get("dataset", {}).get("name", ""),
        "model": config.get("model", {}).get("name", ""),
        "split": metrics.get("split", ""),

        "loss": metrics.get("loss", ""),
        "acc": metrics.get("acc", ""),
        "uar": metrics.get("uar", ""),
        "macro_f1": metrics.get("macro_f1", ""),
        "weighted_f1": metrics.get("weighted_f1", ""),

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

        "checkpoint": metrics.get("checkpoint", ""),
        "run_dir": str(run_dir),
        "eval_dir": str(eval_dir),
    }

    return row


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
    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir is not None
        else resolve_output_category("analysis", frozen_date, OUTPUT_ROOT)
        / "evaluation_summary"
    )
    output_path = output_dir / "evaluation_summary.csv"
    print("=" * 80)
    print("Collect evaluation summary")
    print("=" * 80)

    print("Project root:", PROJECT_ROOT)
    print("Runs root:", OUTPUT_ROOT)
    print("Output path:", output_path)

    metrics_paths = sorted(
        path
        for run_dir in run_dirs
        for path in (run_dir / "logs" / "evaluations").glob("*/metrics.csv")
    )

    if len(metrics_paths) == 0:
        raise RuntimeError("No evaluation metrics.csv files found.")

    rows: List[Dict[str, Any]] = []

    for metrics_path in metrics_paths:
        try:
            row = build_row_from_metrics(metrics_path)
            rows.append(row)
            print("[OK]", metrics_path)
        except Exception as error:
            print("[SKIP/ERROR]", metrics_path, repr(error))

    if len(rows) == 0:
        raise RuntimeError("No valid evaluation rows collected.")

    summary_df = pd.DataFrame(rows)

    for column in SUMMARY_COLUMNS:
        if column not in summary_df.columns:
            summary_df[column] = ""

    summary_df = summary_df[SUMMARY_COLUMNS]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(output_path, index=False, encoding="utf-8-sig")

    print("\nEvaluation summary saved.")
    print("Number of rows:", len(summary_df))
    print("Saved to:", output_path)

    print("\nCompact view:")
    compact_columns = [
        "run_id",
        "model",
        "split",
        "acc",
        "uar",
        "macro_f1",
        "weighted_f1",
    ]

    print(summary_df[compact_columns].to_string(index=False))
    print("=" * 80)


if __name__ == "__main__":
    main()

"""
Collect evaluation results into outputs/evaluation_summary.csv.

这个脚本扫描所有 run 目录下的 evaluation 结果：

    outputs/runs/*/logs/evaluations/*/metrics.csv

并汇总成统一表格：

    outputs/evaluation_summary.csv

用途：
1. 对比 SimpleMLP / MMGCN / 后续 causal-MMGCN
2. 汇总 val / test 指标
3. 为组会和论文实验表格提供统一结果来源

运行方式：
    python scripts/maintenance/collect_evaluation_summary.py
"""

from pathlib import Path
from typing import Dict, Any, List

import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUNS_DIR = PROJECT_ROOT / "outputs" / "runs"
OUTPUT_PATH = PROJECT_ROOT / "outputs" / "evaluation_summary.csv"


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
    # outputs/runs/<run_id>/logs/evaluations/<eval_name>/metrics.csv
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
    print("=" * 80)
    print("Collect evaluation summary")
    print("=" * 80)

    print("Project root:", PROJECT_ROOT)
    print("Runs dir:", RUNS_DIR)
    print("Output path:", OUTPUT_PATH)

    if not RUNS_DIR.exists():
        raise FileNotFoundError(f"Runs directory not found: {RUNS_DIR}")

    metrics_paths = sorted(
        RUNS_DIR.glob("*/logs/evaluations/*/metrics.csv")
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

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

    print("\nEvaluation summary saved.")
    print("Number of rows:", len(summary_df))
    print("Saved to:", OUTPUT_PATH)

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
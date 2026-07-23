"""
Build clean analysis master tables from dated and legacy run directories.

This script rebuilds analysis tables from discovered run directories directly.
It does not depend on the optional experiment/evaluation summary CSV files.

Outputs:
  outputs/<YYYYMMDD>/analysis/analysis_tables/run_file_status.csv
  outputs/<YYYYMMDD>/analysis/analysis_tables/run_summary_master.csv
  outputs/<YYYYMMDD>/analysis/analysis_tables/epoch_metrics_master.csv
  outputs/<YYYYMMDD>/analysis/analysis_tables/evaluation_master.csv
  outputs/<YYYYMMDD>/analysis/analysis_tables/per_class_master.csv
"""

from pathlib import Path
from typing import Any, Dict, List, Optional
import argparse
import sys

import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.output_paths import (  # noqa: E402
    discover_run_directories,
    infer_experiment_date_from_run,
    resolve_experiment_date,
    resolve_output_category,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build clean analysis master tables from dated and legacy runs."
    )

    parser.add_argument(
        "--runs-dir",
        type=str,
        default=None,
        help="Explicit directory containing run folders; defaults to new+legacy discovery.",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Explicit output directory; otherwise uses the dated analysis category.",
    )
    parser.add_argument("--experiment-date", default=None)

    parser.add_argument(
        "--include-run-ids",
        nargs="*",
        default=None,
        help="Only include selected run IDs.",
    )

    parser.add_argument(
        "--exclude-run-ids",
        nargs="*",
        default=None,
        help="Exclude selected run IDs.",
    )

    return parser.parse_args()


def resolve_path(path_text: str) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def safe_get(config: Dict[str, Any], keys: List[str], default: Any = "") -> Any:
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
        return {}

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if data is None:
            return {}
        return data
    except Exception as error:
        print(f"[WARN] Failed to read YAML: {path} | {error}")
        return {}


def read_csv_or_none(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None

    try:
        return pd.read_csv(path)
    except Exception as error:
        print(f"[WARN] Failed to read CSV: {path} | {error}")
        return None


def config_to_row(config: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "project_name": safe_get(config, ["project", "name"]),
        "experiment_name": safe_get(config, ["project", "experiment_name"]),
        "dataset": safe_get(config, ["dataset", "name"]),
        "model": safe_get(config, ["model", "name"]),
        "feature_pkl_path": safe_get(config, ["dataset", "feature_pkl_path"]),
        "seed": safe_get(config, ["system", "seed"]),
        "device": safe_get(config, ["system", "device"]),
        "batch_size": safe_get(config, ["train", "batch_size"]),
        "learning_rate": safe_get(config, ["train", "learning_rate"]),
        "weight_decay": safe_get(config, ["train", "weight_decay"]),
        "max_epochs": safe_get(config, ["train", "max_epochs"]),
        "hidden_dim": safe_get(config, ["model", "hidden_dim"]),
        "dropout": safe_get(config, ["model", "dropout"]),
        "graph_context_mode": safe_get(config, ["graph", "context_mode"]),
        "graph_window_past": safe_get(config, ["graph", "window_past"]),
        "graph_window_future": safe_get(config, ["graph", "window_future"]),
        "graph_num_layers": safe_get(config, ["graph", "num_layers"]),
        "graph_lamda": safe_get(config, ["graph", "lamda"]),
        "graph_alpha": safe_get(config, ["graph", "alpha"]),
        "graph_use_speaker": safe_get(config, ["graph", "use_speaker"]),
        "graph_use_modal": safe_get(config, ["graph", "use_modal"]),
        "graph_use_residual": safe_get(config, ["graph", "use_residual"]),
    }


def best_epoch_values(epoch_df: Optional[pd.DataFrame]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}

    if epoch_df is None or len(epoch_df) == 0:
        return result

    if "epoch" not in epoch_df.columns:
        return result

    maximize_metrics = [
        "val_acc",
        "val_uar",
        "val_macro_f1",
        "val_weighted_f1",
    ]

    minimize_metrics = [
        "train_loss",
        "val_loss",
    ]

    for metric in maximize_metrics:
        if metric in epoch_df.columns:
            valid_df = epoch_df.dropna(subset=[metric])
            if len(valid_df) > 0:
                row = valid_df.loc[valid_df[metric].idxmax()]
                result[f"best_epoch_by_{metric}"] = int(row["epoch"])
                result[f"best_{metric}"] = float(row[metric])

    for metric in minimize_metrics:
        if metric in epoch_df.columns:
            valid_df = epoch_df.dropna(subset=[metric])
            if len(valid_df) > 0:
                row = valid_df.loc[valid_df[metric].idxmin()]
                result[f"best_epoch_by_{metric}"] = int(row["epoch"])
                result[f"best_{metric}"] = float(row[metric])

    result["num_epochs_recorded"] = int(len(epoch_df))

    return result


def build_file_status(run_dir: Path) -> Dict[str, Any]:
    logs_dir = run_dir / "logs"

    status = {
        "run_id": run_dir.name,
        "run_dir": str(run_dir),
        "has_experiment_config": (logs_dir / "experiment_config.yaml").exists(),
        "has_epoch_metrics": (logs_dir / "epoch_metrics.csv").exists(),
        "has_best_model": (run_dir / "checkpoints" / "best_model.pt").exists(),
        "has_last_model": (run_dir / "checkpoints" / "last_model.pt").exists(),
        "has_val_metrics": (
            logs_dir / "evaluations" / "val_best_model" / "metrics.csv"
        ).exists(),
        "has_test_metrics": (
            logs_dir / "evaluations" / "test_best_model" / "metrics.csv"
        ).exists(),
        "has_val_per_class_recall": (
            logs_dir / "evaluations" / "val_best_model" / "per_class_recall.csv"
        ).exists(),
        "has_test_per_class_recall": (
            logs_dir / "evaluations" / "test_best_model" / "per_class_recall.csv"
        ).exists(),
    }

    status["is_train_run"] = bool(
        status["has_experiment_config"]
        and status["has_epoch_metrics"]
        and status["has_best_model"]
    )

    status["has_full_val_test_evaluation"] = bool(
        status["has_val_metrics"]
        and status["has_test_metrics"]
        and status["has_val_per_class_recall"]
        and status["has_test_per_class_recall"]
    )

    status["is_complete_run"] = bool(
        status["is_train_run"]
        and status["has_full_val_test_evaluation"]
    )

    return status


def build_run_summary_row(
    run_dir: Path,
    config: Dict[str, Any],
    epoch_df: Optional[pd.DataFrame],
    status_row: Dict[str, Any],
) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "run_id": run_dir.name,
        "run_dir": str(run_dir),
    }

    row.update(config_to_row(config))
    row.update(best_epoch_values(epoch_df))

    row["is_train_run"] = status_row["is_train_run"]
    row["has_full_val_test_evaluation"] = status_row["has_full_val_test_evaluation"]
    row["is_complete_run"] = status_row["is_complete_run"]

    return row


def build_epoch_rows(
    run_dir: Path,
    config: Dict[str, Any],
    epoch_df: Optional[pd.DataFrame],
) -> List[Dict[str, Any]]:
    if epoch_df is None or len(epoch_df) == 0:
        return []

    base = {
        "run_id": run_dir.name,
        "experiment_name": safe_get(config, ["project", "experiment_name"]),
        "dataset": safe_get(config, ["dataset", "name"]),
        "model": safe_get(config, ["model", "name"]),
        "run_dir": str(run_dir),
    }

    rows: List[Dict[str, Any]] = []

    for _, epoch_row in epoch_df.iterrows():
        row = dict(base)
        for column in epoch_df.columns:
            row[column] = epoch_row[column]
        rows.append(row)

    return rows


def infer_split_from_eval_dir(eval_dir: Path) -> str:
    name = eval_dir.name
    if "_" in name:
        return name.split("_", 1)[0]
    return name


def build_evaluation_and_per_class_rows(
    run_dir: Path,
    config: Dict[str, Any],
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    evaluation_rows: List[Dict[str, Any]] = []
    per_class_rows: List[Dict[str, Any]] = []

    evaluations_dir = run_dir / "logs" / "evaluations"

    if not evaluations_dir.exists():
        return evaluation_rows, per_class_rows

    eval_dirs = sorted([p for p in evaluations_dir.iterdir() if p.is_dir()])

    base = {
        "run_id": run_dir.name,
        "experiment_name": safe_get(config, ["project", "experiment_name"]),
        "dataset": safe_get(config, ["dataset", "name"]),
        "model": safe_get(config, ["model", "name"]),
        "feature_pkl_path": safe_get(config, ["dataset", "feature_pkl_path"]),
        "graph_context_mode": safe_get(config, ["graph", "context_mode"]),
        "graph_window_past": safe_get(config, ["graph", "window_past"]),
        "graph_window_future": safe_get(config, ["graph", "window_future"]),
        "graph_num_layers": safe_get(config, ["graph", "num_layers"]),
        "graph_lamda": safe_get(config, ["graph", "lamda"]),
        "graph_alpha": safe_get(config, ["graph", "alpha"]),
        "run_dir": str(run_dir),
    }

    for eval_dir in eval_dirs:
        metrics_path = eval_dir / "metrics.csv"
        metrics_df = read_csv_or_none(metrics_path)

        if metrics_df is not None and len(metrics_df) > 0:
            for _, metrics_row in metrics_df.iterrows():
                split = metrics_row.get("split", infer_split_from_eval_dir(eval_dir))

                row = dict(base)
                row["split"] = split
                row["eval_name"] = eval_dir.name
                row["eval_dir"] = str(eval_dir)

                for column in metrics_df.columns:
                    row[column] = metrics_row[column]

                evaluation_rows.append(row)

        recall_path = eval_dir / "per_class_recall.csv"
        recall_df = read_csv_or_none(recall_path)

        if recall_df is not None and len(recall_df) > 0:
            split = infer_split_from_eval_dir(eval_dir)

            if metrics_df is not None and len(metrics_df) > 0:
                if "split" in metrics_df.columns:
                    split = metrics_df.iloc[0]["split"]

            for _, recall_row in recall_df.iterrows():
                row = dict(base)
                row["split"] = split
                row["eval_name"] = eval_dir.name
                row["eval_dir"] = str(eval_dir)

                for column in recall_df.columns:
                    row[column] = recall_row[column]

                per_class_rows.append(row)

    return evaluation_rows, per_class_rows


def filter_run_dirs(
    run_dirs: List[Path],
    include_run_ids: Optional[List[str]],
    exclude_run_ids: Optional[List[str]],
) -> List[Path]:
    filtered = run_dirs

    if include_run_ids:
        include_set = set(include_run_ids)
        filtered = [p for p in filtered if p.name in include_set]

        missing = sorted(include_set - set(p.name for p in filtered))
        if missing:
            print("[WARN] include-run-ids not found:")
            for run_id in missing:
                print("  ", run_id)

    if exclude_run_ids:
        exclude_set = set(exclude_run_ids)
        filtered = [p for p in filtered if p.name not in exclude_set]

    return filtered


def save_table(rows: List[Dict[str, Any]], output_path: Path) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    return df


def main() -> None:
    args = parse_args()
    if args.runs_dir is not None:
        runs_dir = resolve_path(args.runs_dir)
        if not runs_dir.exists():
            raise FileNotFoundError(f"Runs directory not found: {runs_dir}")
        run_dirs = sorted([p for p in runs_dir.iterdir() if p.is_dir()])
    else:
        runs_dir = DEFAULT_OUTPUT_ROOT
        run_dirs = discover_run_directories(DEFAULT_OUTPUT_ROOT)
    run_dirs = filter_run_dirs(
        run_dirs=run_dirs,
        include_run_ids=args.include_run_ids,
        exclude_run_ids=args.exclude_run_ids,
    )
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
        resolve_path(args.output_dir)
        if args.output_dir is not None
        else resolve_output_category(
            "analysis", frozen_date, DEFAULT_OUTPUT_ROOT
        )
        / "analysis_tables"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 100)
    print("Build analysis master tables")
    print("=" * 100)
    print("Project root:", PROJECT_ROOT)
    print("Runs dir:", runs_dir)
    print("Output dir:", output_dir)
    print("Number of selected run dirs:", len(run_dirs))
    print("=" * 100)

    status_rows: List[Dict[str, Any]] = []
    run_summary_rows: List[Dict[str, Any]] = []
    epoch_rows: List[Dict[str, Any]] = []
    evaluation_rows: List[Dict[str, Any]] = []
    per_class_rows: List[Dict[str, Any]] = []

    for run_dir in run_dirs:
        config_path = run_dir / "logs" / "experiment_config.yaml"
        epoch_path = run_dir / "logs" / "epoch_metrics.csv"

        config = load_yaml_or_empty(config_path)
        epoch_df = read_csv_or_none(epoch_path)

        status_row = build_file_status(run_dir)
        status_rows.append(status_row)

        run_summary_rows.append(
            build_run_summary_row(
                run_dir=run_dir,
                config=config,
                epoch_df=epoch_df,
                status_row=status_row,
            )
        )

        epoch_rows.extend(
            build_epoch_rows(
                run_dir=run_dir,
                config=config,
                epoch_df=epoch_df,
            )
        )

        eval_rows, cls_rows = build_evaluation_and_per_class_rows(
            run_dir=run_dir,
            config=config,
        )

        evaluation_rows.extend(eval_rows)
        per_class_rows.extend(cls_rows)

        print(
            f"[OK] {run_dir.name} | "
            f"train={status_row['is_train_run']} | "
            f"full_eval={status_row['has_full_val_test_evaluation']} | "
            f"complete={status_row['is_complete_run']}"
        )

    status_df = save_table(
        status_rows,
        output_dir / "run_file_status.csv",
    )

    run_summary_df = save_table(
        run_summary_rows,
        output_dir / "run_summary_master.csv",
    )

    epoch_metrics_df = save_table(
        epoch_rows,
        output_dir / "epoch_metrics_master.csv",
    )

    evaluation_df = save_table(
        evaluation_rows,
        output_dir / "evaluation_master.csv",
    )

    per_class_df = save_table(
        per_class_rows,
        output_dir / "per_class_master.csv",
    )

    print("\nSaved tables:")
    print("  run_file_status:", output_dir / "run_file_status.csv", f"rows={len(status_df)}")
    print("  run_summary_master:", output_dir / "run_summary_master.csv", f"rows={len(run_summary_df)}")
    print("  epoch_metrics_master:", output_dir / "epoch_metrics_master.csv", f"rows={len(epoch_metrics_df)}")
    print("  evaluation_master:", output_dir / "evaluation_master.csv", f"rows={len(evaluation_df)}")
    print("  per_class_master:", output_dir / "per_class_master.csv", f"rows={len(per_class_df)}")

    print("\nComplete runs:")
    if len(status_df) > 0 and "is_complete_run" in status_df.columns:
        complete_df = status_df[status_df["is_complete_run"] == True]
        if len(complete_df) > 0:
            for run_id in complete_df["run_id"].tolist():
                print("  ", run_id)
        else:
            print("  No complete runs found.")

    print("=" * 100)


if __name__ == "__main__":
    main()

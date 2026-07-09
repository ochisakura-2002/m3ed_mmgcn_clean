"""
Export lightweight paper tables across multiple completed runs.

This first version writes one row per run and intentionally does not compute
mean/std statistics. It is meant to gather completed validation-best test
results into paper-facing CSV/Markdown/LaTeX tables.

Usage:
    python scripts/analyze/export_paper_multi_run_tables.py \
      --run-ids <run1> <run2> <run3> \
      --output-dir outputs/paper_artifacts/tables
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUNS_DIR = PROJECT_ROOT / "outputs" / "runs"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "paper_artifacts" / "tables"
BEST_SELECTION_METRIC = "val_weighted_f1"

RESULT_COLUMNS = [
    "run_id",
    "experiment_name",
    "dataset",
    "model",
    "context_mode",
    "window_past",
    "active_modalities",
    "seed",
    "test_acc",
    "test_weighted_f1",
    "test_macro_f1",
    "test_uar",
    "test_loss",
    "val_acc",
    "val_weighted_f1",
    "val_macro_f1",
    "val_uar",
    "best_epoch",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export paper tables for multiple runs."
    )
    parser.add_argument(
        "--run-ids",
        nargs="*",
        default=[],
        help="Run IDs under --runs-dir.",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Optional analysis YAML with a runs list.",
    )
    parser.add_argument(
        "--runs-dir",
        default=str(DEFAULT_RUNS_DIR),
        help="Directory containing run folders.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Output directory for paper tables.",
    )
    return parser.parse_args()


def resolve_path(path_text: str) -> Path:
    path = Path(str(path_text))
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


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
        return {}
    return data


def read_csv_or_none(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception as error:
        print(f"[WARN] Failed to read CSV: {path} | {error}")
        return None


def to_float_or_na(value: Any) -> Any:
    if value is None or pd.isna(value):
        return pd.NA
    try:
        return float(value)
    except (TypeError, ValueError):
        return pd.NA


def metric_from_row(row: Optional[pd.Series], names: List[str]) -> Any:
    if row is None:
        return pd.NA
    for name in names:
        if name in row.index:
            return to_float_or_na(row[name])
    return pd.NA


def first_row_or_none(path: Path) -> Optional[pd.Series]:
    df = read_csv_or_none(path)
    if df is None or len(df) == 0:
        return None
    return df.iloc[0]


def active_modalities_text(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        return "+".join(str(item) for item in value)
    if value is None:
        return ""
    return str(value)


def choose_validation_best_epoch(run_dir: Path) -> Any:
    summary_path = run_dir / "figures" / "training_curves" / "best_epoch_summary.csv"
    summary_df = read_csv_or_none(summary_path)
    if summary_df is not None and {"metric", "best_epoch"}.issubset(summary_df.columns):
        metric_df = summary_df[summary_df["metric"] == BEST_SELECTION_METRIC]
        metric_df = metric_df.dropna(subset=["best_epoch"])
        if len(metric_df) > 0:
            return int(metric_df.iloc[0]["best_epoch"])

    epoch_path = run_dir / "logs" / "epoch_metrics.csv"
    epoch_df = read_csv_or_none(epoch_path)
    if epoch_df is None or BEST_SELECTION_METRIC not in epoch_df.columns:
        return pd.NA
    valid_df = epoch_df.dropna(subset=[BEST_SELECTION_METRIC])
    if len(valid_df) == 0:
        return pd.NA
    best_row = valid_df.loc[valid_df[BEST_SELECTION_METRIC].idxmax()]
    return int(best_row["epoch"])


def build_run_result_row(run_dir: Path) -> Dict[str, Any]:
    config = load_yaml_or_empty(run_dir / "logs" / "experiment_config.yaml")
    model = config.get("model", {}) if isinstance(config.get("model"), dict) else {}
    val_row = first_row_or_none(run_dir / "logs" / "evaluations" / "val_best_model" / "metrics.csv")
    test_row = first_row_or_none(run_dir / "logs" / "evaluations" / "test_best_model" / "metrics.csv")

    return {
        "run_id": run_dir.name,
        "experiment_name": safe_get(config, ["project", "experiment_name"]),
        "dataset": safe_get(config, ["dataset", "name"]),
        "model": safe_get(config, ["model", "name"]),
        "context_mode": safe_get(config, ["graph", "context_mode"]),
        "window_past": safe_get(config, ["graph", "window_past"]),
        "active_modalities": active_modalities_text(model.get("active_modalities", "")),
        "seed": safe_get(config, ["system", "seed"]),
        "test_acc": metric_from_row(test_row, ["acc", "accuracy"]),
        "test_weighted_f1": metric_from_row(test_row, ["weighted_f1"]),
        "test_macro_f1": metric_from_row(test_row, ["macro_f1"]),
        "test_uar": metric_from_row(test_row, ["uar"]),
        "test_loss": metric_from_row(test_row, ["loss"]),
        "val_acc": metric_from_row(val_row, ["acc", "accuracy"]),
        "val_weighted_f1": metric_from_row(val_row, ["weighted_f1"]),
        "val_macro_f1": metric_from_row(val_row, ["macro_f1"]),
        "val_uar": metric_from_row(val_row, ["uar"]),
        "best_epoch": choose_validation_best_epoch(run_dir),
    }


def load_run_ids_from_config(config_path: Optional[str]) -> List[str]:
    if not config_path:
        return []

    config = load_yaml_or_empty(resolve_path(config_path))
    raw_runs = config.get("runs", [])
    if not isinstance(raw_runs, list):
        print("[WARN] config runs field is not a list.")
        return []

    run_ids: List[str] = []
    for item in raw_runs:
        if isinstance(item, dict):
            run_id = str(item.get("run_id", "")).strip()
        else:
            run_id = str(item).strip()
        if run_id:
            run_ids.append(run_id)
    return run_ids


def unique_in_order(values: Iterable[str]) -> List[str]:
    seen = set()
    result: List[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


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
        lines.append("| " + " | ".join(format_cell(row[column]) for column in df.columns) + " |")
    return "\n".join(lines) + "\n"


def save_table_bundle(df: pd.DataFrame, output_dir: Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{stem}.csv"
    md_path = output_dir / f"{stem}.md"
    tex_path = output_dir / f"{stem}.tex"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    md_path.write_text(dataframe_to_markdown(df), encoding="utf-8")
    tex_path.write_text(df.to_latex(index=False), encoding="utf-8")
    print(f"  [Table] {stem}: {csv_path}")


def has_any_nonempty(df: pd.DataFrame, columns: List[str]) -> bool:
    for column in columns:
        if column not in df.columns:
            continue
        series = df[column].fillna("").astype(str).str.strip()
        if series.ne("").any():
            return True
    return False


def main() -> None:
    args = parse_args()
    runs_dir = resolve_path(args.runs_dir)
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    run_ids = unique_in_order(list(args.run_ids) + load_run_ids_from_config(args.config))
    if not run_ids:
        raise ValueError("No run IDs provided. Use --run-ids or --config with runs.")

    print("=" * 100)
    print("Export paper multi-run tables")
    print("=" * 100)
    print("Runs dir:", runs_dir)
    print("Output dir:", output_dir)
    print("Run IDs:", run_ids)
    print("=" * 100)

    rows: List[Dict[str, Any]] = []
    for run_id in run_ids:
        run_dir = runs_dir / run_id
        if not run_dir.exists():
            print(f"[WARN] Run directory not found, skip: {run_dir}")
            continue
        rows.append(build_run_result_row(run_dir))
        print(f"[OK] {run_id}")

    if not rows:
        raise RuntimeError("No existing run directories were exported.")

    results_df = pd.DataFrame(rows, columns=RESULT_COLUMNS)
    save_table_bundle(results_df, output_dir, "main_results")

    if has_any_nonempty(results_df, ["context_mode", "window_past"]):
        context_df = results_df.sort_values(["context_mode", "window_past", "seed", "run_id"])
        save_table_bundle(context_df, output_dir, "context_window_results")

    if has_any_nonempty(results_df, ["active_modalities"]):
        modality_df = results_df.sort_values(["active_modalities", "seed", "run_id"])
        save_table_bundle(modality_df, output_dir, "modality_ablation_results")

    print("=" * 100)
    print("Finished paper multi-run table export.")
    print("=" * 100)


if __name__ == "__main__":
    main()

"""Diagnose one MultiDAG+CL training run from saved CSV logs.

The script reads a completed run under outputs/runs/<run_id>/, summarizes the
validation-best epoch and late-training behavior, and writes lightweight CSV,
Markdown, and matplotlib diagnostics. It never loads checkpoints.
"""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "m3ed_mmgcn_matplotlib"),
)
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


RUNS_DIR = PROJECT_ROOT / "outputs" / "runs"
BEST_SELECTION_METRIC = "val_weighted_f1"
EPSILON = 1e-8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose a MultiDAG+CL run from epoch/evaluation logs."
    )
    parser.add_argument(
        "--run-id",
        required=True,
        help="Run ID under outputs/runs/.",
    )
    return parser.parse_args()


def read_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing experiment config: {path}")
    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise TypeError(f"Expected YAML dict: {path}")
    return data


def read_csv_required(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required CSV: {path}")
    df = pd.read_csv(path)
    if len(df) == 0:
        raise RuntimeError(f"Empty CSV: {path}")
    return df


def read_csv_optional(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if len(df) == 0:
        return None
    return df


def project_relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT.resolve()))
    except ValueError:
        return str(path)


def to_float(value: Any) -> Optional[float]:
    if value is None or pd.isna(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def metric_value(row: Optional[pd.Series], metric: str) -> Optional[float]:
    if row is None or metric not in row.index:
        return None
    return to_float(row[metric])


def find_best_row(df: pd.DataFrame, metric: str) -> Optional[pd.Series]:
    if metric not in df.columns:
        return None
    valid = df.dropna(subset=[metric])
    if len(valid) == 0:
        return None
    return valid.loc[valid[metric].idxmax()]


def get_final_row(df: pd.DataFrame) -> pd.Series:
    if "epoch" in df.columns:
        return df.sort_values("epoch").iloc[-1]
    return df.iloc[-1]


def extract_eval_metrics(df: Optional[pd.DataFrame], prefix: str) -> Dict[str, Any]:
    if df is None or len(df) == 0:
        return {
            f"{prefix}_loss": None,
            f"{prefix}_acc": None,
            f"{prefix}_weighted_f1": None,
            f"{prefix}_macro_f1": None,
            f"{prefix}_uar": None,
        }

    row = df.iloc[0]
    return {
        f"{prefix}_loss": metric_value(row, "loss"),
        f"{prefix}_acc": metric_value(row, "acc"),
        f"{prefix}_weighted_f1": metric_value(row, "weighted_f1"),
        f"{prefix}_macro_f1": metric_value(row, "macro_f1"),
        f"{prefix}_uar": metric_value(row, "uar"),
    }


def config_value(config: Dict[str, Any], path: List[str], default: Any = None) -> Any:
    current: Any = config
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def rows_after_epoch(df: pd.DataFrame, epoch: Optional[int]) -> pd.DataFrame:
    if epoch is None or "epoch" not in df.columns:
        return df.iloc[0:0]
    return df[df["epoch"] > int(epoch)]


def best_epoch_from_row(row: Optional[pd.Series]) -> Optional[int]:
    if row is None or "epoch" not in row.index or pd.isna(row["epoch"]):
        return None
    return int(row["epoch"])


def series_min(df: pd.DataFrame, metric: str) -> Optional[float]:
    if metric not in df.columns or len(df) == 0:
        return None
    valid = df[metric].dropna()
    if len(valid) == 0:
        return None
    return float(valid.min())


def series_max(df: pd.DataFrame, metric: str) -> Optional[float]:
    if metric not in df.columns or len(df) == 0:
        return None
    valid = df[metric].dropna()
    if len(valid) == 0:
        return None
    return float(valid.max())


def detect_worsening(
    df: pd.DataFrame,
    best_epoch: Optional[int],
    metric: str,
    mode: str,
) -> Dict[str, Any]:
    if best_epoch is None or metric not in df.columns:
        return {
            f"{metric}_worsens_after_best": None,
            f"{metric}_worst_after_best": None,
            f"{metric}_final_minus_best": None,
        }

    best_df = df[df["epoch"] == int(best_epoch)]
    if len(best_df) == 0:
        return {
            f"{metric}_worsens_after_best": None,
            f"{metric}_worst_after_best": None,
            f"{metric}_final_minus_best": None,
        }

    best_value = to_float(best_df.iloc[0][metric])
    final_value = metric_value(get_final_row(df), metric)
    after = rows_after_epoch(df, best_epoch)
    if best_value is None or final_value is None or len(after) == 0:
        return {
            f"{metric}_worsens_after_best": False,
            f"{metric}_worst_after_best": None,
            f"{metric}_final_minus_best": (
                None if best_value is None or final_value is None else final_value - best_value
            ),
        }

    if mode == "min":
        worst_after = series_max(after, metric)
        worsens = bool(worst_after is not None and worst_after > best_value + EPSILON)
    elif mode == "max":
        worst_after = series_min(after, metric)
        worsens = bool(worst_after is not None and worst_after < best_value - EPSILON)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    return {
        f"{metric}_worsens_after_best": worsens,
        f"{metric}_worst_after_best": worst_after,
        f"{metric}_final_minus_best": final_value - best_value,
    }


def relative_drop(first_value: Optional[float], final_value: Optional[float]) -> Optional[float]:
    if first_value is None or final_value is None or abs(first_value) <= EPSILON:
        return None
    return (first_value - final_value) / abs(first_value)


def interpret_training_behavior(
    df: pd.DataFrame,
    best_epoch: Optional[int],
    val_loss_worsens: Optional[bool],
    val_f1_worsens: Optional[bool],
) -> str:
    first_row = df.iloc[0]
    final_row = get_final_row(df)
    first_train_loss = metric_value(first_row, "train_loss")
    final_train_loss = metric_value(final_row, "train_loss")
    train_loss_drop = relative_drop(first_train_loss, final_train_loss)

    train_improves = bool(train_loss_drop is not None and train_loss_drop >= 0.05)
    train_stalls = bool(train_loss_drop is not None and train_loss_drop < 0.02)
    validation_degrades = bool(val_loss_worsens or val_f1_worsens)

    if train_improves and validation_degrades:
        return "overfitting-like"

    if train_stalls:
        return "optimization-failure-like"

    if best_epoch is not None:
        total_epochs = int(len(df))
        if best_epoch <= max(3, total_epochs // 4) and validation_degrades:
            return "overfitting-like"

    return "mixed-or-unclear"


def build_diagnosis_row(
    run_id: str,
    run_dir: Path,
    config: Dict[str, Any],
    epoch_df: pd.DataFrame,
    val_eval_df: Optional[pd.DataFrame],
    test_eval_df: Optional[pd.DataFrame],
) -> Dict[str, Any]:
    best_f1_row = find_best_row(epoch_df, BEST_SELECTION_METRIC)
    best_acc_row = find_best_row(epoch_df, "val_acc")
    final_row = get_final_row(epoch_df)
    best_f1_epoch = best_epoch_from_row(best_f1_row)
    best_acc_epoch = best_epoch_from_row(best_acc_row)

    loss_worsening = detect_worsening(
        df=epoch_df,
        best_epoch=best_f1_epoch,
        metric="val_loss",
        mode="min",
    )
    f1_worsening = detect_worsening(
        df=epoch_df,
        best_epoch=best_f1_epoch,
        metric=BEST_SELECTION_METRIC,
        mode="max",
    )
    behavior = interpret_training_behavior(
        df=epoch_df,
        best_epoch=best_f1_epoch,
        val_loss_worsens=loss_worsening["val_loss_worsens_after_best"],
        val_f1_worsens=f1_worsening["val_weighted_f1_worsens_after_best"],
    )

    first_row = epoch_df.iloc[0]
    train_loss_drop = relative_drop(
        metric_value(first_row, "train_loss"),
        metric_value(final_row, "train_loss"),
    )

    row: Dict[str, Any] = {
        "run_id": run_id,
        "run_dir": project_relative(run_dir),
        "num_epochs_logged": int(len(epoch_df)),
        "best_epoch_by_val_weighted_f1": best_f1_epoch,
        "best_val_weighted_f1": metric_value(best_f1_row, BEST_SELECTION_METRIC),
        "best_epoch_by_val_acc": best_acc_epoch,
        "best_val_acc": metric_value(best_acc_row, "val_acc"),
        "final_epoch": metric_value(final_row, "epoch"),
        "final_train_loss": metric_value(final_row, "train_loss"),
        "final_train_acc": metric_value(final_row, "train_acc"),
        "final_train_weighted_f1": metric_value(final_row, "train_weighted_f1"),
        "final_train_macro_f1": metric_value(final_row, "train_macro_f1"),
        "final_train_uar": metric_value(final_row, "train_uar"),
        "final_val_loss": metric_value(final_row, "val_loss"),
        "final_val_acc": metric_value(final_row, "val_acc"),
        "final_val_weighted_f1": metric_value(final_row, BEST_SELECTION_METRIC),
        "final_val_macro_f1": metric_value(final_row, "val_macro_f1"),
        "final_val_uar": metric_value(final_row, "val_uar"),
        "final_lr": metric_value(final_row, "lr"),
        "final_grad_norm": metric_value(final_row, "grad_norm"),
        "train_loss_relative_drop": train_loss_drop,
        "behavior_label": behavior,
        "model_modality_encoder_type": config_value(config, ["model", "modality_encoder_type"]),
        "model_num_graph_layers": config_value(config, ["model", "num_graph_layers"]),
        "model_dropout": config_value(config, ["model", "dropout"]),
        "training_lr": config_value(config, ["training", "lr"]),
        "training_weight_decay": config_value(config, ["training", "weight_decay"]),
        "training_grad_clip": config_value(config, ["training", "grad_clip"]),
        "graph_context_mode": config_value(config, ["graph", "context_mode"]),
        "graph_window_past": config_value(config, ["graph", "window_past"]),
    }
    row.update(loss_worsening)
    row.update(f1_worsening)
    row.update(extract_eval_metrics(val_eval_df, "val_best_model"))
    row.update(extract_eval_metrics(test_eval_df, "test_best_model"))
    return row


def format_value(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def write_markdown(
    path: Path,
    diagnosis: Dict[str, Any],
    epoch_metrics_path: Path,
    val_metrics_path: Path,
    test_metrics_path: Path,
    figure_paths: List[Path],
) -> None:
    lines = [
        "# MultiDAG+CL training diagnosis",
        "",
        "## Inputs",
        "",
        f"- Run ID: `{diagnosis['run_id']}`",
        f"- Epoch metrics: `{project_relative(epoch_metrics_path)}`",
        f"- Validation metrics: `{project_relative(val_metrics_path)}`",
        f"- Test metrics: `{project_relative(test_metrics_path)}`",
        "",
        "## Summary",
        "",
        f"- Best epoch by `val_weighted_f1`: {format_value(diagnosis['best_epoch_by_val_weighted_f1'])}",
        f"- Best `val_weighted_f1`: {format_value(diagnosis['best_val_weighted_f1'])}",
        f"- Best epoch by `val_acc`: {format_value(diagnosis['best_epoch_by_val_acc'])}",
        f"- Best `val_acc`: {format_value(diagnosis['best_val_acc'])}",
        f"- Final epoch: {format_value(diagnosis['final_epoch'])}",
        f"- Final train loss: {format_value(diagnosis['final_train_loss'])}",
        f"- Final val loss: {format_value(diagnosis['final_val_loss'])}",
        f"- Final val weighted F1: {format_value(diagnosis['final_val_weighted_f1'])}",
        f"- Val loss worsens after best epoch: {format_value(diagnosis['val_loss_worsens_after_best'])}",
        f"- Val weighted F1 worsens after best epoch: {format_value(diagnosis['val_weighted_f1_worsens_after_best'])}",
        f"- Behavior label: `{diagnosis['behavior_label']}`",
        "",
        "## Final Epoch Metrics",
        "",
        "| Metric | Value |",
        "|---|---:|",
    ]
    for key in [
        "final_train_acc",
        "final_train_weighted_f1",
        "final_train_macro_f1",
        "final_train_uar",
        "final_val_acc",
        "final_val_weighted_f1",
        "final_val_macro_f1",
        "final_val_uar",
        "final_lr",
        "final_grad_norm",
    ]:
        lines.append(f"| `{key}` | {format_value(diagnosis.get(key))} |")

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            (
                "- `overfitting-like` means train loss keeps improving while validation "
                "loss or validation Weighted-F1 degrades after the validation-best epoch."
            ),
            (
                "- `optimization-failure-like` means train loss barely improves across "
                "the logged epochs."
            ),
            "- `mixed-or-unclear` means the CSV alone is not decisive.",
            "",
            "## Figures",
            "",
        ]
    )
    if figure_paths:
        for figure_path in figure_paths:
            lines.append(f"- `{project_relative(figure_path)}`")
    else:
        lines.append("- No diagnostic figure was generated.")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_diagnostics(
    epoch_df: pd.DataFrame,
    output_dir: Path,
    best_epoch: Optional[int],
) -> List[Path]:
    if "epoch" not in epoch_df.columns:
        return []

    loss_columns = [name for name in ["train_loss", "val_loss"] if name in epoch_df.columns]
    metric_columns = [
        name
        for name in ["val_acc", "val_weighted_f1", "val_macro_f1", "val_uar"]
        if name in epoch_df.columns
    ]
    if not loss_columns and not metric_columns:
        return []

    figure_paths: List[Path] = []
    fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)

    if loss_columns:
        for column in loss_columns:
            axes[0].plot(epoch_df["epoch"], epoch_df[column], marker="o", label=column)
        axes[0].set_ylabel("Loss")
        axes[0].set_title("Train/Validation Loss")
        axes[0].legend(loc="best")
    else:
        axes[0].axis("off")

    if metric_columns:
        for column in metric_columns:
            axes[1].plot(epoch_df["epoch"], epoch_df[column], marker="o", label=column)
        axes[1].set_ylabel("Metric")
        axes[1].set_ylim(0, 1)
        axes[1].set_title("Validation Metrics")
        axes[1].legend(loc="best")
    else:
        axes[1].axis("off")

    for axis in axes:
        if best_epoch is not None:
            axis.axvline(
                best_epoch,
                linestyle="--",
                linewidth=1,
                color="black",
                label="best val_weighted_f1",
            )
        axis.grid(alpha=0.25)

    axes[-1].set_xlabel("Epoch")
    fig.tight_layout()

    png_path = output_dir / "multidag_cl_training_diagnosis.png"
    pdf_path = output_dir / "multidag_cl_training_diagnosis.pdf"
    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path)
    plt.close(fig)
    figure_paths.extend([png_path, pdf_path])
    return figure_paths


def main() -> None:
    args = parse_args()
    run_id = str(args.run_id)
    run_dir = RUNS_DIR / run_id
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    logs_dir = run_dir / "logs"
    figures_dir = run_dir / "figures" / "diagnostics"
    figures_dir.mkdir(parents=True, exist_ok=True)

    epoch_metrics_path = logs_dir / "epoch_metrics.csv"
    val_metrics_path = logs_dir / "evaluations" / "val_best_model" / "metrics.csv"
    test_metrics_path = logs_dir / "evaluations" / "test_best_model" / "metrics.csv"
    config_path = logs_dir / "experiment_config.yaml"

    epoch_df = read_csv_required(epoch_metrics_path)
    config = read_yaml(config_path)
    val_eval_df = read_csv_optional(val_metrics_path)
    test_eval_df = read_csv_optional(test_metrics_path)

    diagnosis = build_diagnosis_row(
        run_id=run_id,
        run_dir=run_dir,
        config=config,
        epoch_df=epoch_df,
        val_eval_df=val_eval_df,
        test_eval_df=test_eval_df,
    )

    csv_path = figures_dir / "multidag_cl_training_diagnosis.csv"
    md_path = figures_dir / "multidag_cl_training_diagnosis.md"
    pd.DataFrame([diagnosis]).to_csv(csv_path, index=False, encoding="utf-8-sig")

    figure_paths = plot_diagnostics(
        epoch_df=epoch_df,
        output_dir=figures_dir,
        best_epoch=diagnosis["best_epoch_by_val_weighted_f1"],
    )
    write_markdown(
        path=md_path,
        diagnosis=diagnosis,
        epoch_metrics_path=epoch_metrics_path,
        val_metrics_path=val_metrics_path,
        test_metrics_path=test_metrics_path,
        figure_paths=figure_paths,
    )

    print("=" * 100)
    print("MultiDAG+CL run diagnosis")
    print("=" * 100)
    print("Run ID:", run_id)
    print("Best epoch by val_weighted_f1:", diagnosis["best_epoch_by_val_weighted_f1"])
    print("Best epoch by val_acc:", diagnosis["best_epoch_by_val_acc"])
    print("Final epoch:", diagnosis["final_epoch"])
    print("Val loss worsens after best:", diagnosis["val_loss_worsens_after_best"])
    print(
        "Val weighted F1 worsens after best:",
        diagnosis["val_weighted_f1_worsens_after_best"],
    )
    print("Behavior label:", diagnosis["behavior_label"])
    print("CSV:", csv_path)
    print("Markdown:", md_path)
    if figure_paths:
        print("Figures:")
        for figure_path in figure_paths:
            print(" ", figure_path)
    print("=" * 100)


if __name__ == "__main__":
    main()

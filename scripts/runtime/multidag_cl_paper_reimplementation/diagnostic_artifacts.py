"""Artifact orchestration for optional MultiDAG-CL overfitting diagnostics."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch
from torch.utils.data import DataLoader

from .adapter import ProjectBatchAdapter
from .checkpoint import strict_reload_checkpoint
from .diagnostics import (
    CHECKPOINT_FILENAMES,
    CHECKPOINT_TYPES,
    PREDICTION_FIELDS,
    DiagnosticCheckpointTracker,
    DiagnosticSettings,
    checkpoint_summary_rows,
    named_prediction_rows,
    normalized_confusion,
    parameter_count_row,
    prediction_flip_details,
    prediction_flip_summary,
    reliability_bins,
    write_csv,
    ww_high_confidence_rows,
)
from .evaluation import evaluate_model
from .manifest import RunPaths


def export_diagnostic_artifacts(
    *,
    model: torch.nn.Module,
    val_loader: DataLoader,
    adapter: ProjectBatchAdapter,
    device: torch.device,
    label_ids: list[int],
    label_names: list[str],
    paths: RunPaths,
    expected_checkpoint_identity: Mapping[str, Any],
    tracker: DiagnosticCheckpointTracker,
    condition: str,
    settings: DiagnosticSettings,
) -> list[str]:
    artifacts: list[str] = []
    evaluations: dict[str, dict[str, Any]] = {}
    prediction_rows: dict[str, list[dict[str, Any]]] = {}
    for checkpoint_type in CHECKPOINT_TYPES:
        checkpoint_path = paths.checkpoints / CHECKPOINT_FILENAMES[checkpoint_type]
        strict_reload_checkpoint(
            checkpoint_path,
            model=model,
            device=device,
            expected_identity=expected_checkpoint_identity,
            require_locked=checkpoint_type == "best_val_wf1",
        )
        result = evaluate_model(
            model,
            val_loader,
            adapter=adapter,
            device=device,
            split=f"validation_{checkpoint_type}",
            labels=label_ids,
            diagnostics_enabled=True,
            ece_bins=settings.ece_bins,
            label_names=label_names,
            collect_attention=(
                settings.export_dag_attention
                and checkpoint_type in {"best_val_wf1", "final"}
            ),
        )
        evaluations[checkpoint_type] = result
        prediction_rows[checkpoint_type] = named_prediction_rows(result, label_names)
        if settings.save_selected_val_predictions:
            output = paths.predictions / "diagnostic" / (
                f"val_predictions_{checkpoint_type}.csv"
            )
            write_csv(output, prediction_rows[checkpoint_type], PREDICTION_FIELDS)
            artifacts.append(output.as_posix())
        if settings.export_dag_attention and checkpoint_type in {"best_val_wf1", "final"}:
            attention_path = paths.run_dir / "attention" / (
                f"dag_attention_{checkpoint_type}.csv"
            )
            write_csv(
                attention_path,
                result["attention_rows"],
                [
                    "dialogue_id",
                    "target_utterance_id",
                    "source_utterance_id",
                    "layer_index",
                    "attention_weight",
                ],
            )
            artifacts.append(attention_path.as_posix())

    if settings.save_checkpoint_summary:
        output = paths.reports / "checkpoint_summary.csv"
        write_csv(
            output,
            checkpoint_summary_rows(tracker, condition),
            [
                "condition",
                "checkpoint_type",
                "epoch",
                "val_loss",
                "val_accuracy",
                "val_weighted_f1",
                "val_macro_f1",
                "val_uar",
                "val_nll",
                "val_brier",
                "val_ece",
                "mean_confidence_correct",
                "mean_confidence_wrong",
            ],
        )
        artifacts.append(output.as_posix())

    if settings.save_calibration_summary:
        record_epochs = {
            record.checkpoint_type: record.epoch
            for record in tracker.ordered_records()
        }
        rows = []
        for checkpoint_type in CHECKPOINT_TYPES:
            metrics = evaluations[checkpoint_type]["diagnostic_metrics"]
            rows.append(
                {
                    "condition": condition,
                    "checkpoint_type": checkpoint_type,
                    "epoch": record_epochs[checkpoint_type],
                    "nll": metrics["nll"],
                    "brier": metrics["brier"],
                    "ece": metrics["ece"],
                    "mean_confidence_correct": metrics["mean_confidence_correct"],
                    "mean_confidence_wrong": metrics["mean_confidence_wrong"],
                }
            )
        output = paths.reports / "calibration_summary.csv"
        write_csv(
            output,
            rows,
            [
                "condition",
                "checkpoint_type",
                "epoch",
                "nll",
                "brier",
                "ece",
                "mean_confidence_correct",
                "mean_confidence_wrong",
            ],
        )
        artifacts.append(output.as_posix())

    if settings.save_reliability_bins:
        rows = []
        for checkpoint_type in ("min_val_loss", "best_val_wf1", "final"):
            result = evaluations[checkpoint_type]
            for row in reliability_bins(
                result["diagnostic_probabilities"],
                result["diagnostic_labels"],
                ece_bins=settings.ece_bins,
            ):
                rows.append(
                    {
                        "condition": condition,
                        "checkpoint_type": checkpoint_type,
                        **row,
                    }
                )
        output = paths.reports / "reliability_bins.csv"
        write_csv(
            output,
            rows,
            [
                "condition",
                "checkpoint_type",
                "bin_index",
                "bin_lower",
                "bin_upper",
                "count",
                "mean_confidence",
                "empirical_accuracy",
            ],
        )
        artifacts.append(output.as_posix())

    confusion_dir = paths.reports / "confusion_diagnostic"
    for checkpoint_type in ("min_val_loss", "best_val_wf1", "final"):
        raw = evaluations[checkpoint_type]["confusion_matrix"]
        normalized = normalized_confusion(raw)
        for suffix, matrix in (("raw", raw), ("normalized_by_true", normalized)):
            rows = [
                {
                    "true_label": label_names[index],
                    **{
                        name: matrix[index][column]
                        for column, name in enumerate(label_names)
                    },
                }
                for index in range(len(label_names))
            ]
            output = confusion_dir / f"val_{checkpoint_type}_{suffix}.csv"
            write_csv(output, rows, ["true_label", *label_names])
            artifacts.append(output.as_posix())

    if settings.save_prediction_flip_analysis:
        detail_rows: list[dict[str, Any]] = []
        summary_rows: list[dict[str, Any]] = []
        ww_rows: list[dict[str, Any]] = []
        for early_type in ("min_val_loss", "best_val_wf1"):
            comparison = f"{early_type}->final"
            details = prediction_flip_details(
                condition,
                comparison,
                prediction_rows[early_type],
                prediction_rows["final"],
            )
            detail_rows.extend(details)
            summary_rows.extend(prediction_flip_summary(details))
            ww_rows.extend(ww_high_confidence_rows(details))
        detail_output = paths.reports / "prediction_flip_details.csv"
        write_csv(
            detail_output,
            detail_rows,
            [
                "condition",
                "comparison",
                "dialogue_id",
                "utterance_id",
                "true_label",
                "early_pred",
                "late_pred",
                "flip_type",
                "early_confidence",
                "late_confidence",
                "early_true_probability",
                "late_true_probability",
                "confidence_change",
                "true_probability_change",
            ],
        )
        summary_output = paths.reports / "prediction_flip_summary.csv"
        write_csv(
            summary_output,
            summary_rows,
            [
                "condition",
                "comparison",
                "true_label",
                "flip_type",
                "count",
                "ratio",
                "mean_confidence_early",
                "mean_confidence_late",
                "mean_true_prob_early",
                "mean_true_prob_late",
            ],
        )
        ww_output = paths.reports / "ww_high_confidence_summary.csv"
        write_csv(
            ww_output,
            ww_rows,
            [
                "condition",
                "comparison",
                "true_label",
                "ww_total",
                "confidence_increase_count",
                "final_confidence_gt_080_count",
                "final_confidence_gt_090_count",
                "final_confidence_gt_095_count",
            ],
        )
        artifacts.extend(
            [detail_output.as_posix(), summary_output.as_posix(), ww_output.as_posix()]
        )

    if settings.save_parameter_count:
        output = paths.reports / "parameter_count.csv"
        write_csv(
            output,
            [parameter_count_row(model, condition)],
            [
                "condition",
                "total_parameters",
                "trainable_parameters",
                "encoder_parameters",
                "dag_parameters",
                "classifier_parameters",
                "fusion_parameters",
            ],
        )
        artifacts.append(output.as_posix())
    return artifacts


__all__ = ["export_diagnostic_artifacts"]

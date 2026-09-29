"""Optional overfitting/generalization diagnostics for the UniLSTM track.

The module is deliberately independent of the training control plane.  It
contains pure metric/selection helpers plus small CSV writers so that the
legacy runtime remains unchanged when diagnostics are absent or disabled.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import precision_recall_fscore_support

from utils.metrics import compute_classification_metrics


IEMOCAP_LABEL_NAMES = (
    "happy",
    "sad",
    "neutral",
    "angry",
    "excited",
    "frustrated",
)
_LABEL_ALIASES = {
    "hap": "happy",
    "happy": "happy",
    "sad": "sad",
    "neu": "neutral",
    "neutral": "neutral",
    "ang": "angry",
    "angry": "angry",
    "exc": "excited",
    "excited": "excited",
    "fru": "frustrated",
    "frustrated": "frustrated",
}
CHECKPOINT_TYPES = ("min_val_loss", "best_val_wf1", "best_val_uar", "final")
CHECKPOINT_FILENAMES = {
    "min_val_loss": "min_val_loss_model.pt",
    "best_val_wf1": "best_val_wf1_model.pt",
    "best_val_uar": "best_val_uar_model.pt",
    "final": "final_model.pt",
}
_SETTING_NAMES = (
    "save_extended_epoch_metrics",
    "save_per_class_epoch_metrics",
    "save_checkpoint_summary",
    "save_selected_val_predictions",
    "save_prediction_flip_analysis",
    "save_calibration_summary",
    "save_reliability_bins",
    "save_parameter_count",
    "export_dag_attention",
)


@dataclass(frozen=True)
class DiagnosticSettings:
    enabled: bool = False
    ece_bins: int = 15
    save_extended_epoch_metrics: bool = False
    save_per_class_epoch_metrics: bool = False
    save_checkpoint_summary: bool = False
    save_selected_val_predictions: bool = False
    save_prediction_flip_analysis: bool = False
    save_calibration_summary: bool = False
    save_reliability_bins: bool = False
    save_parameter_count: bool = False
    export_dag_attention: bool = False

    def to_mapping(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "ece_bins": self.ece_bins,
            **{name: getattr(self, name) for name in _SETTING_NAMES},
        }


def parse_diagnostic_settings(config: Mapping[str, Any]) -> DiagnosticSettings:
    diagnostics = config.get("diagnostics")
    if diagnostics is None:
        return DiagnosticSettings()
    if not isinstance(diagnostics, Mapping):
        raise TypeError("diagnostics must be a mapping")
    unknown = sorted(set(diagnostics) - {"overfitting_generalization"})
    if unknown:
        raise ValueError(f"unknown diagnostics sections: {unknown}")
    section = diagnostics.get("overfitting_generalization")
    if not isinstance(section, Mapping):
        raise TypeError("diagnostics.overfitting_generalization must be a mapping")
    allowed = {"enabled", "ece_bins", *_SETTING_NAMES}
    unknown = sorted(set(section) - allowed)
    if unknown:
        raise ValueError(f"unknown overfitting diagnostic fields: {unknown}")
    enabled = section.get("enabled", False)
    if not isinstance(enabled, bool):
        raise TypeError("diagnostics.overfitting_generalization.enabled must be bool")
    ece_bins = section.get("ece_bins", 15)
    if isinstance(ece_bins, bool) or not isinstance(ece_bins, int):
        raise TypeError("diagnostics.overfitting_generalization.ece_bins must be int")
    if ece_bins < 2:
        raise ValueError("diagnostics.overfitting_generalization.ece_bins must be >= 2")
    values: dict[str, bool] = {}
    for name in _SETTING_NAMES:
        value = section.get(name, False)
        if not isinstance(value, bool):
            raise TypeError(f"diagnostics.overfitting_generalization.{name} must be bool")
        if value and not enabled:
            raise ValueError(f"{name}=true requires overfitting diagnostics enabled")
        values[name] = value
    return DiagnosticSettings(enabled=enabled, ece_bins=ece_bins, **values)


def canonical_label_names(label_names: Sequence[str]) -> list[str]:
    resolved = [_LABEL_ALIASES.get(str(value).strip().lower()) for value in label_names]
    if any(value is None for value in resolved):
        raise ValueError(f"unsupported IEMOCAP diagnostic label names: {list(label_names)!r}")
    if tuple(resolved) != IEMOCAP_LABEL_NAMES:
        raise ValueError(
            "IEMOCAP diagnostic label order must be happy,sad,neutral,angry,excited,frustrated"
        )
    return [str(value) for value in resolved]


def condition_name(active_modalities: Sequence[str]) -> str:
    active = set(active_modalities)
    if not active or active - {"text", "audio", "visual"}:
        raise ValueError("active_modalities must be a nonempty T/A/V subset")
    return "".join(
        symbol
        for name, symbol in (("text", "T"), ("audio", "A"), ("visual", "V"))
        if name in active
    )


def _validate_logits_labels(
    logits: torch.Tensor, labels: torch.Tensor, num_classes: int
) -> tuple[torch.Tensor, torch.Tensor]:
    logits = torch.as_tensor(logits, dtype=torch.float64).detach().cpu()
    labels = torch.as_tensor(labels, dtype=torch.long).detach().cpu()
    if logits.dim() != 2 or logits.shape[1] != int(num_classes):
        raise ValueError("logits must have shape [N,C]")
    if labels.dim() != 1 or labels.shape[0] != logits.shape[0] or labels.numel() == 0:
        raise ValueError("labels must be nonempty [N] aligned with logits")
    if not torch.isfinite(logits).all():
        raise ValueError("diagnostic logits must be finite")
    if torch.any((labels < 0) | (labels >= int(num_classes))):
        raise ValueError("diagnostic labels must be in [0,C)")
    return logits, labels


def reliability_bins(
    probabilities: torch.Tensor,
    labels: torch.Tensor,
    *,
    ece_bins: int,
) -> list[dict[str, Any]]:
    probabilities = torch.as_tensor(probabilities, dtype=torch.float64).detach().cpu()
    labels = torch.as_tensor(labels, dtype=torch.long).detach().cpu()
    if probabilities.dim() != 2 or labels.shape != (probabilities.shape[0],):
        raise ValueError("probabilities and labels must have shapes [N,C] and [N]")
    if ece_bins < 2:
        raise ValueError("ece_bins must be >= 2")
    confidences, predictions = probabilities.max(dim=1)
    correct = predictions.eq(labels)
    rows: list[dict[str, Any]] = []
    for index in range(ece_bins):
        lower = index / ece_bins
        upper = (index + 1) / ece_bins
        if index == ece_bins - 1:
            mask = (confidences >= lower) & (confidences <= upper)
        else:
            mask = (confidences >= lower) & (confidences < upper)
        count = int(mask.sum().item())
        rows.append(
            {
                "bin_index": index,
                "bin_lower": lower,
                "bin_upper": upper,
                "count": count,
                "mean_confidence": (
                    float(confidences[mask].mean().item()) if count else math.nan
                ),
                "empirical_accuracy": (
                    float(correct[mask].double().mean().item()) if count else math.nan
                ),
            }
        )
    return rows


def compute_probability_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    num_classes: int,
    ece_bins: int,
) -> dict[str, float]:
    logits, labels = _validate_logits_labels(logits, labels, num_classes)
    probabilities = torch.softmax(logits, dim=1)
    predictions = probabilities.argmax(dim=1)
    shared = compute_classification_metrics(
        labels.numpy(), predictions.numpy(), labels=list(range(num_classes))
    )
    one_hot = F.one_hot(labels, num_classes=num_classes).to(probabilities.dtype)
    confidence = probabilities.max(dim=1).values
    correct = predictions.eq(labels)
    bins = reliability_bins(probabilities, labels, ece_bins=ece_bins)
    ece = 0.0
    total = labels.numel()
    for row in bins:
        if row["count"]:
            ece += (row["count"] / total) * abs(
                row["mean_confidence"] - row["empirical_accuracy"]
            )
    return {
        "accuracy": shared["acc"],
        "weighted_f1": shared["weighted_f1"],
        "macro_f1": shared["macro_f1"],
        "uar": shared["uar"],
        "nll": float(F.cross_entropy(logits, labels, reduction="mean").item()),
        "brier": float(((probabilities - one_hot) ** 2).sum(dim=1).mean().item()),
        "ece": float(ece),
        "mean_confidence_correct": (
            float(confidence[correct].mean().item()) if bool(correct.any()) else math.nan
        ),
        "mean_confidence_wrong": (
            float(confidence[~correct].mean().item()) if bool((~correct).any()) else math.nan
        ),
    }


def compute_per_class_metrics(
    labels: Sequence[int], predictions: Sequence[int], label_names: Sequence[str]
) -> list[dict[str, Any]]:
    label_ids = list(range(len(label_names)))
    precision, recall, f1, support = precision_recall_fscore_support(
        labels,
        predictions,
        labels=label_ids,
        average=None,
        zero_division=0,
    )
    return [
        {
            "class": str(label_name),
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "support": int(support[index]),
        }
        for index, label_name in enumerate(label_names)
    ]


@dataclass(frozen=True)
class DiagnosticCheckpointRecord:
    checkpoint_type: str
    epoch: int
    metrics: dict[str, float]


class DiagnosticCheckpointTracker:
    """Select validation diagnostic checkpoints with deterministic early ties."""

    def __init__(self) -> None:
        self.records: dict[str, DiagnosticCheckpointRecord] = {}

    @staticmethod
    def _key(checkpoint_type: str, epoch: int, metrics: Mapping[str, float]):
        if checkpoint_type == "min_val_loss":
            return (-float(metrics["val_loss"]), float(metrics["val_weighted_f1"]), -epoch)
        if checkpoint_type == "best_val_wf1":
            return (float(metrics["val_weighted_f1"]), -float(metrics["val_loss"]), -epoch)
        if checkpoint_type == "best_val_uar":
            return (float(metrics["val_uar"]), float(metrics["val_weighted_f1"]), -epoch)
        if checkpoint_type == "final":
            return (epoch,)
        raise ValueError(f"unknown checkpoint type: {checkpoint_type}")

    def update(self, *, epoch: int, metrics: Mapping[str, float]) -> tuple[str, ...]:
        required = {
            "val_loss",
            "val_accuracy",
            "val_weighted_f1",
            "val_macro_f1",
            "val_uar",
            "val_nll",
            "val_brier",
            "val_ece",
            "val_mean_confidence_correct",
            "val_mean_confidence_wrong",
        }
        missing = sorted(required - set(metrics))
        if missing:
            raise ValueError(f"checkpoint metrics missing fields: {missing}")
        copied = {name: float(metrics[name]) for name in required}
        improved: list[str] = []
        for checkpoint_type in CHECKPOINT_TYPES:
            candidate = DiagnosticCheckpointRecord(checkpoint_type, int(epoch), copied)
            current = self.records.get(checkpoint_type)
            if current is None or self._key(checkpoint_type, epoch, copied) > self._key(
                checkpoint_type, current.epoch, current.metrics
            ):
                self.records[checkpoint_type] = candidate
                improved.append(checkpoint_type)
        return tuple(improved)

    def ordered_records(self) -> list[DiagnosticCheckpointRecord]:
        missing = [name for name in CHECKPOINT_TYPES if name not in self.records]
        if missing:
            raise RuntimeError(f"diagnostic checkpoint selection incomplete: {missing}")
        return [self.records[name] for name in CHECKPOINT_TYPES]


def annotate_checkpoint_payload(
    payload: Mapping[str, Any], checkpoint_type: str
) -> dict[str, Any]:
    if checkpoint_type not in CHECKPOINT_TYPES:
        raise ValueError(f"unknown checkpoint type: {checkpoint_type}")
    value = dict(payload)
    value["diagnostic_checkpoint_type"] = checkpoint_type
    value["test_evaluation_eligible"] = checkpoint_type == "best_val_wf1"
    return value


def checkpoint_allows_test_evaluation(checkpoint: Mapping[str, Any]) -> bool:
    checkpoint_type = checkpoint.get("diagnostic_checkpoint_type")
    if checkpoint_type is None:
        return checkpoint.get("test_evaluation_eligible", True) is not False
    return (
        checkpoint_type == "best_val_wf1"
        and checkpoint.get("test_evaluation_eligible") is True
    )


def parameter_count_row(model: torch.nn.Module, condition: str) -> dict[str, Any]:
    groups = {"encoder_parameters": 0, "dag_parameters": 0, "classifier_parameters": 0}
    total = 0
    trainable = 0
    for name, parameter in model.named_parameters():
        count = parameter.numel()
        total += count
        if parameter.requires_grad:
            trainable += count
        if name.startswith(("paper_encoder.", "source_input_adapter.", "source_encoder.")):
            groups["encoder_parameters"] += count
        elif name.startswith("graph_layers."):
            groups["dag_parameters"] += count
        elif name.startswith("classifier."):
            groups["classifier_parameters"] += count
        else:
            raise ValueError(f"unclassified model parameter: {name}")
    if sum(groups.values()) != total:
        raise RuntimeError("parameter component counts do not sum to total")
    return {
        "condition": condition,
        "total_parameters": total,
        "trainable_parameters": trainable,
        **groups,
        "fusion_parameters": 0,
    }


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]], fields: Sequence[str]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(fields), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})
    return path


def checkpoint_summary_rows(
    tracker: DiagnosticCheckpointTracker, condition: str
) -> list[dict[str, Any]]:
    rows = []
    for record in tracker.ordered_records():
        metrics = record.metrics
        rows.append(
            {
                "condition": condition,
                "checkpoint_type": record.checkpoint_type,
                "epoch": record.epoch,
                "val_loss": metrics["val_loss"],
                "val_accuracy": metrics["val_accuracy"],
                "val_weighted_f1": metrics["val_weighted_f1"],
                "val_macro_f1": metrics["val_macro_f1"],
                "val_uar": metrics["val_uar"],
                "val_nll": metrics["val_nll"],
                "val_brier": metrics["val_brier"],
                "val_ece": metrics["val_ece"],
                "mean_confidence_correct": metrics["val_mean_confidence_correct"],
                "mean_confidence_wrong": metrics["val_mean_confidence_wrong"],
            }
        )
    return rows


def named_prediction_rows(
    result: Mapping[str, Any], label_names: Sequence[str]
) -> list[dict[str, Any]]:
    names = canonical_label_names(label_names)
    rows: list[dict[str, Any]] = []
    for source in result["predictions"]:
        probabilities = list(source["probabilities"])
        true_id = int(source["true_label"])
        predicted_id = int(source["predicted_label"])
        row = {
            "dialogue_id": source["dialogue_id"],
            "utterance_id": source["utterance_id"],
            "speaker_id": source["speaker_id"],
            "true_label": names[true_id],
            "pred_label": names[predicted_id],
            **{f"prob_{name}": float(probabilities[index]) for index, name in enumerate(names)},
            "pred_confidence": float(source["pred_confidence"]),
            "true_label_probability": float(source["true_label_probability"]),
            "correct": bool(source["correct"]),
        }
        rows.append(row)
    return rows


PREDICTION_FIELDS = [
    "dialogue_id",
    "utterance_id",
    "speaker_id",
    "true_label",
    "pred_label",
    *[f"prob_{name}" for name in IEMOCAP_LABEL_NAMES],
    "pred_confidence",
    "true_label_probability",
    "correct",
]


def prediction_flip_details(
    condition: str,
    comparison: str,
    early_rows: Sequence[Mapping[str, Any]],
    late_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    early = {(row["dialogue_id"], row["utterance_id"]): row for row in early_rows}
    late = {(row["dialogue_id"], row["utterance_id"]): row for row in late_rows}
    if early.keys() != late.keys():
        raise ValueError("prediction flip inputs must contain identical validation samples")
    rows = []
    for key in early:
        left, right = early[key], late[key]
        if left["true_label"] != right["true_label"]:
            raise ValueError("prediction flip true labels do not align")
        flip = ("C" if left["correct"] else "W") + ("C" if right["correct"] else "W")
        rows.append(
            {
                "condition": condition,
                "comparison": comparison,
                "dialogue_id": key[0],
                "utterance_id": key[1],
                "true_label": left["true_label"],
                "early_pred": left["pred_label"],
                "late_pred": right["pred_label"],
                "flip_type": flip,
                "early_confidence": float(left["pred_confidence"]),
                "late_confidence": float(right["pred_confidence"]),
                "early_true_probability": float(left["true_label_probability"]),
                "late_true_probability": float(right["true_label_probability"]),
                "confidence_change": float(right["pred_confidence"])
                - float(left["pred_confidence"]),
                "true_probability_change": float(right["true_label_probability"])
                - float(left["true_label_probability"]),
            }
        )
    return rows


def prediction_flip_summary(details: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    labels = ["ALL", *IEMOCAP_LABEL_NAMES]
    for true_label in labels:
        selected = [
            row for row in details if true_label == "ALL" or row["true_label"] == true_label
        ]
        denominator = len(selected)
        for flip_type in ("CC", "WC", "CW", "WW"):
            group = [row for row in selected if row["flip_type"] == flip_type]
            def mean(name: str) -> float:
                return (
                    float(sum(float(row[name]) for row in group) / len(group))
                    if group
                    else math.nan
                )
            exemplar = details[0] if details else {"condition": "", "comparison": ""}
            rows.append(
                {
                    "condition": exemplar["condition"],
                    "comparison": exemplar["comparison"],
                    "true_label": true_label,
                    "flip_type": flip_type,
                    "count": len(group),
                    "ratio": len(group) / denominator if denominator else math.nan,
                    "mean_confidence_early": mean("early_confidence"),
                    "mean_confidence_late": mean("late_confidence"),
                    "mean_true_prob_early": mean("early_true_probability"),
                    "mean_true_prob_late": mean("late_true_probability"),
                }
            )
    return rows


def ww_high_confidence_rows(details: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    exemplar = details[0] if details else {"condition": "", "comparison": ""}
    rows = []
    for true_label in ("ALL", *IEMOCAP_LABEL_NAMES):
        group = [
            row
            for row in details
            if row["flip_type"] == "WW"
            and (true_label == "ALL" or row["true_label"] == true_label)
        ]
        rows.append(
            {
                "condition": exemplar["condition"],
                "comparison": exemplar["comparison"],
                "true_label": true_label,
                "ww_total": len(group),
                "confidence_increase_count": sum(
                    float(row["confidence_change"]) > 0 for row in group
                ),
                "final_confidence_gt_080_count": sum(
                    float(row["late_confidence"]) > 0.80 for row in group
                ),
                "final_confidence_gt_090_count": sum(
                    float(row["late_confidence"]) > 0.90 for row in group
                ),
                "final_confidence_gt_095_count": sum(
                    float(row["late_confidence"]) > 0.95 for row in group
                ),
            }
        )
    return rows


def normalized_confusion(matrix: Any) -> np.ndarray:
    value = np.asarray(matrix, dtype=np.float64)
    row_sums = value.sum(axis=1, keepdims=True)
    return np.divide(value, row_sums, out=np.zeros_like(value), where=row_sums != 0)


__all__ = [
    "CHECKPOINT_FILENAMES",
    "CHECKPOINT_TYPES",
    "DiagnosticCheckpointTracker",
    "DiagnosticSettings",
    "IEMOCAP_LABEL_NAMES",
    "PREDICTION_FIELDS",
    "annotate_checkpoint_payload",
    "canonical_label_names",
    "checkpoint_allows_test_evaluation",
    "checkpoint_summary_rows",
    "compute_per_class_metrics",
    "compute_probability_metrics",
    "condition_name",
    "named_prediction_rows",
    "normalized_confusion",
    "parameter_count_row",
    "parse_diagnostic_settings",
    "prediction_flip_details",
    "prediction_flip_summary",
    "reliability_bins",
    "write_csv",
    "ww_high_confidence_rows",
]

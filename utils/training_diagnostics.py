"""Opt-in, tensor-light training diagnostics shared by model workflows."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import torch

from utils.metrics import compute_classification_metrics, compute_per_class_recall


DIAGNOSTIC_FIELDS = (
    "epoch",
    "train_loss",
    "val_loss",
    "train_weighted_f1",
    "val_weighted_f1",
    "learning_rate",
    "classification_loss",
    "contrastive_loss",
    "auxiliary_loss",
    "gradient_norm",
    "parameter_update_norm",
    "nonzero_gradient_parameter_count",
    "trainable_parameter_count",
    "logit_mean",
    "logit_std",
    "logit_min",
    "logit_max",
    "prediction_entropy",
    "predicted_class_count",
    "dominant_class_ratio",
    "per_class_recall",
    "effective_batch_count",
    "early_stopping_counter",
    "best_epoch",
    "best_metric",
    "min_epochs",
    "patience",
    "stopped_by_early_stopping",
)


def diagnostics_enabled(config: Mapping[str, Any]) -> bool:
    diagnostics = config.get("diagnostics", {})
    return isinstance(diagnostics, Mapping) and bool(diagnostics.get("enabled", False))


def should_collect_expensive_diagnostics(
    config: Mapping[str, Any],
    epoch: int,
) -> bool:
    """Return whether an epoch should sample an optimizer-step update norm."""

    diagnostics = config.get("diagnostics", {})
    if not isinstance(diagnostics, Mapping) or not diagnostics_enabled(config):
        return False
    full_frequency_epochs = int(diagnostics.get("full_frequency_epochs", 15))
    every_n_epochs = int(diagnostics.get("expensive_every_n_epochs", 5))
    if full_frequency_epochs < 0:
        raise ValueError("diagnostics.full_frequency_epochs must be non-negative")
    if every_n_epochs <= 0:
        raise ValueError("diagnostics.expensive_every_n_epochs must be positive")
    return int(epoch) <= full_frequency_epochs or int(epoch) % every_n_epochs == 0


def optimizer_parameter_audit(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> dict[str, Any]:
    """Compare trainable parameters with optimizer groups by object identity."""

    trainable = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    trainable_by_id = {id(parameter): (name, parameter) for name, parameter in trainable}
    optimizer_parameters = [
        parameter
        for group in optimizer.param_groups
        for parameter in group["params"]
    ]
    optimizer_ids = [id(parameter) for parameter in optimizer_parameters]
    missing_ids = sorted(set(trainable_by_id) - set(optimizer_ids))
    duplicate_reference_count = len(optimizer_ids) - len(set(optimizer_ids))
    groups = []
    for index, group in enumerate(optimizer.param_groups):
        parameters = list(group["params"])
        groups.append(
            {
                "group_index": index,
                "parameter_tensor_count": len(parameters),
                "parameter_element_count": sum(
                    int(parameter.numel()) for parameter in parameters
                ),
                "learning_rate": float(group["lr"]),
                "weight_decay": float(group.get("weight_decay", 0.0)),
            }
        )
    return {
        "trainable_parameter_tensor_count": len(trainable),
        "trainable_parameter_count": sum(
            int(parameter.numel()) for _, parameter in trainable
        ),
        "optimizer_parameter_tensor_count": len(optimizer_parameters),
        "optimizer_parameter_count": sum(
            int(parameter.numel()) for parameter in optimizer_parameters
        ),
        "missing_trainable_parameter_names": [
            trainable_by_id[parameter_id][0] for parameter_id in missing_ids
        ],
        "duplicate_optimizer_parameter_reference_count": duplicate_reference_count,
        "parameter_groups": groups,
    }


class TrainingEpochAccumulator:
    """Aggregate train-side losses, predictions, gradients, and one true update."""

    def __init__(
        self,
        num_classes: int,
        *,
        collect_parameter_update: bool,
    ) -> None:
        self.num_classes = int(num_classes)
        self.collect_parameter_update = bool(collect_parameter_update)
        self.effective_batch_count = 0
        self.valid_count = 0
        self.total_loss = 0.0
        self.classification_loss = 0.0
        self.contrastive_loss = 0.0
        self.auxiliary_loss = 0.0
        self.contrastive_present = False
        self.auxiliary_present = False
        self.y_true: list[int] = []
        self.y_pred: list[int] = []
        self.gradient_norm_sum = 0.0
        self.gradient_sample_count = 0
        self.nonzero_gradient_parameter_count = 0
        self.parameter_update_norm: Optional[float] = None

    def update_batch(
        self,
        output: Mapping[str, Any],
        batch: Mapping[str, Any],
    ) -> None:
        valid = batch["attention_mask"].bool() & (batch["labels"] >= 0)
        count = int(valid.sum().item())
        if count <= 0:
            return
        self.effective_batch_count += 1
        self.valid_count += count
        self.total_loss += float(output["loss"].detach().item()) * count
        self.classification_loss += (
            float(output["classification_loss"].detach().item()) * count
        )
        auxiliary_losses = dict(output.get("aux_losses", {}))
        contrastive = auxiliary_losses.pop("contrastive_loss", None)
        if contrastive is not None:
            self.contrastive_present = True
            self.contrastive_loss += float(contrastive.detach().item()) * count
        if auxiliary_losses:
            self.auxiliary_present = True
            self.auxiliary_loss += (
                sum(float(value.detach().item()) for value in auxiliary_losses.values())
                * count
            )
        predicted = output["logits"].detach().argmax(dim=-1)
        self.y_true.extend(
            int(value) for value in batch["labels"][valid].detach().cpu().tolist()
        )
        self.y_pred.extend(
            int(value) for value in predicted[valid].detach().cpu().tolist()
        )

    def record_gradients(self, model: torch.nn.Module) -> None:
        squared_norm = 0.0
        nonzero_count = 0
        for parameter in model.parameters():
            if not parameter.requires_grad or parameter.grad is None:
                continue
            gradient = parameter.grad.detach()
            squared_norm += float(gradient.float().pow(2).sum().item())
            nonzero_count += int(torch.count_nonzero(gradient).item())
        self.gradient_norm_sum += math.sqrt(max(0.0, squared_norm))
        self.gradient_sample_count += 1
        self.nonzero_gradient_parameter_count = max(
            self.nonzero_gradient_parameter_count,
            nonzero_count,
        )

    def snapshot_parameter_update(
        self,
        model: torch.nn.Module,
    ) -> Optional[dict[str, torch.Tensor]]:
        if not self.collect_parameter_update or self.parameter_update_norm is not None:
            return None
        return {
            name: parameter.detach().cpu().clone()
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }

    def record_parameter_update(
        self,
        model: torch.nn.Module,
        before: Optional[Mapping[str, torch.Tensor]],
    ) -> None:
        if before is None:
            return
        squared_norm = 0.0
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            previous = before[name]
            difference = parameter.detach().cpu() - previous
            squared_norm += float(difference.float().pow(2).sum().item())
        self.parameter_update_norm = math.sqrt(max(0.0, squared_norm))

    def summary(self) -> dict[str, Any]:
        if self.valid_count <= 0:
            raise RuntimeError("training diagnostics observed no valid utterances")
        metrics = compute_classification_metrics(
            self.y_true,
            self.y_pred,
            labels=list(range(self.num_classes)),
        )
        return {
            "loss": self.total_loss / self.valid_count,
            "classification_loss": self.classification_loss / self.valid_count,
            "contrastive_loss": (
                self.contrastive_loss / self.valid_count
                if self.contrastive_present
                else None
            ),
            "auxiliary_loss": (
                self.auxiliary_loss / self.valid_count
                if self.auxiliary_present
                else None
            ),
            "weighted_f1": metrics["weighted_f1"],
            "gradient_norm": (
                self.gradient_norm_sum / self.gradient_sample_count
                if self.gradient_sample_count
                else None
            ),
            "parameter_update_norm": self.parameter_update_norm,
            "nonzero_gradient_parameter_count": (
                self.nonzero_gradient_parameter_count
            ),
            "effective_batch_count": self.effective_batch_count,
        }


class PredictionDiagnosticsAccumulator:
    """Aggregate validation logits and predictions without retaining tensors."""

    def __init__(
        self,
        num_classes: int,
        label_names: Optional[Sequence[str]] = None,
    ) -> None:
        self.num_classes = int(num_classes)
        self.label_names = (
            [str(value) for value in label_names]
            if label_names is not None
            else [str(index) for index in range(self.num_classes)]
        )
        self.logit_count = 0
        self.logit_sum = 0.0
        self.logit_square_sum = 0.0
        self.logit_min = float("inf")
        self.logit_max = float("-inf")
        self.entropy_sum = 0.0
        self.prediction_count = 0
        self.y_true: list[int] = []
        self.y_pred: list[int] = []

    def update(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> None:
        valid = attention_mask.bool() & (labels >= 0)
        valid_logits = logits.detach()[valid].float()
        if valid_logits.numel() == 0:
            return
        probabilities = torch.softmax(valid_logits, dim=-1)
        entropy = -(
            probabilities * probabilities.clamp_min(1e-12).log()
        ).sum(dim=-1)
        predicted = probabilities.argmax(dim=-1)
        true_values = labels.detach()[valid].long()

        self.logit_count += int(valid_logits.numel())
        self.logit_sum += float(valid_logits.sum().item())
        self.logit_square_sum += float(valid_logits.pow(2).sum().item())
        self.logit_min = min(self.logit_min, float(valid_logits.min().item()))
        self.logit_max = max(self.logit_max, float(valid_logits.max().item()))
        self.entropy_sum += float(entropy.sum().item())
        self.prediction_count += int(predicted.numel())
        self.y_true.extend(int(value) for value in true_values.cpu().tolist())
        self.y_pred.extend(int(value) for value in predicted.cpu().tolist())

    def summary(self) -> dict[str, Any]:
        if self.logit_count <= 0 or self.prediction_count <= 0:
            raise RuntimeError("prediction diagnostics observed no valid utterances")
        mean = self.logit_sum / self.logit_count
        variance = max(0.0, self.logit_square_sum / self.logit_count - mean**2)
        predicted_counts = {
            label_id: self.y_pred.count(label_id)
            for label_id in range(self.num_classes)
        }
        present_counts = [count for count in predicted_counts.values() if count > 0]
        recalls = compute_per_class_recall(
            self.y_true,
            self.y_pred,
            labels=list(range(self.num_classes)),
        )
        named_recalls = {
            self.label_names[label_id]: recalls[label_id]
            for label_id in range(self.num_classes)
        }
        return {
            "logit_mean": mean,
            "logit_std": math.sqrt(variance),
            "logit_min": self.logit_min,
            "logit_max": self.logit_max,
            "prediction_entropy": self.entropy_sum / self.prediction_count,
            "predicted_class_count": len(present_counts),
            "dominant_class_ratio": (
                max(present_counts) / self.prediction_count
                if present_counts
                else None
            ),
            "per_class_recall": json.dumps(
                named_recalls,
                ensure_ascii=False,
                sort_keys=True,
            ),
        }


def complete_diagnostic_row(values: Mapping[str, Any]) -> dict[str, Any]:
    """Return one exact-schema row; unavailable values remain empty in CSV."""

    unknown = sorted(set(values) - set(DIAGNOSTIC_FIELDS))
    if unknown:
        raise KeyError(f"unknown diagnostic fields: {unknown}")
    return {field: values.get(field) for field in DIAGNOSTIC_FIELDS}


def write_diagnostic_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Write a stable diagnostic schema and preserve missing losses as blanks."""

    import csv

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=DIAGNOSTIC_FIELDS, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(complete_diagnostic_row(row))


__all__ = [
    "DIAGNOSTIC_FIELDS",
    "PredictionDiagnosticsAccumulator",
    "TrainingEpochAccumulator",
    "complete_diagnostic_row",
    "diagnostics_enabled",
    "optimizer_parameter_audit",
    "should_collect_expensive_diagnostics",
    "write_diagnostic_csv",
]

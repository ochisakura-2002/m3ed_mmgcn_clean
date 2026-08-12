"""Global-utterance evaluation and canonical small-artifact export."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

import torch

from models.multidag_cl.paper_reimplementation.contracts import MISSING_LABEL_INDEX

from utils.metrics import (
    compute_classification_metrics,
    compute_confusion_matrix,
    compute_per_class_recall,
)

from .adapter import ProjectBatchAdapter


def move_batch(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        name: value.to(device) if isinstance(value, torch.Tensor) else value
        for name, value in batch.items()
    }


def model_forward(model: torch.nn.Module, batch: Mapping[str, Any]):
    return model(
        text_features=batch["text_features"],
        audio_features=batch["audio_features"],
        visual_features=batch["visual_features"],
        attention_mask=batch["attention_mask"],
        lengths=batch["lengths"],
        speaker_ids_int=batch["speaker_ids_int"],
        labels=batch.get("labels"),
    )


def evaluate_model(
    model: torch.nn.Module,
    loader: Iterable[Mapping[str, Any]],
    *,
    adapter: ProjectBatchAdapter,
    device: torch.device,
    split: str,
    labels: list[int],
    max_batches: Optional[int] = None,
) -> dict[str, Any]:
    """Aggregate predictions over all valid utterances, never per dialogue."""

    was_training = model.training
    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []
    prediction_rows: list[dict[str, Any]] = []
    loss_numerator = 0.0
    valid_count = 0
    batch_count = 0
    with torch.no_grad():
        for batch_index, raw_batch in enumerate(loader):
            if max_batches is not None and batch_index >= int(max_batches):
                break
            batch = move_batch(raw_batch, device)
            adapter.adapt(batch, split=split, require_labels=True)
            mask = (
                batch["attention_mask"].bool()
                & (batch["labels"] != adapter.config.loss_ignore_index)
                & (batch["labels"] != MISSING_LABEL_INDEX)
            )
            count = int(mask.sum().item())
            if count == 0:
                continue
            output = model_forward(model, batch)
            predictions = output.logits.argmax(dim=-1)
            true_values = batch["labels"][mask].detach().cpu().tolist()
            pred_values = predictions[mask].detach().cpu().tolist()
            y_true.extend(int(value) for value in true_values)
            y_pred.extend(int(value) for value in pred_values)
            if output.loss is None:
                raise RuntimeError("labeled evaluation output is missing loss")
            loss_numerator += float(output.loss.detach().cpu().item()) * count
            valid_count += count
            batch_count += 1
            lengths = batch["lengths"].detach().cpu().tolist()
            prediction_cpu = predictions.detach().cpu()
            labels_cpu = batch["labels"].detach().cpu()
            for dialogue_index, length in enumerate(lengths):
                dialogue_id = str(batch["dialogue_ids"][dialogue_index])
                utterance_ids = batch["utterance_ids"][dialogue_index]
                for utterance_index in range(int(length)):
                    true_label = int(labels_cpu[dialogue_index, utterance_index])
                    if true_label in {
                        MISSING_LABEL_INDEX,
                        adapter.config.loss_ignore_index,
                    }:
                        continue
                    prediction_rows.append(
                        {
                            "split": split,
                            "dialogue_id": dialogue_id,
                            "utterance_id": utterance_ids[utterance_index],
                            "utterance_index": utterance_index,
                            "true_label": true_label,
                            "predicted_label": int(
                                prediction_cpu[dialogue_index, utterance_index]
                            ),
                        }
                    )
    if was_training:
        model.train()
    if valid_count == 0:
        raise ValueError(f"evaluation split {split!r} contained no valid utterances")
    shared = compute_classification_metrics(y_true, y_pred, labels=labels)
    metrics = {
        "accuracy": shared["acc"],
        "weighted_f1": shared["weighted_f1"],
        "macro_f1": shared["macro_f1"],
        "uar": shared["uar"],
        "loss": loss_numerator / valid_count,
        "prediction_count": valid_count,
        "dominant_class_ratio": max(
            y_pred.count(label_id) for label_id in labels
        )
        / valid_count,
    }
    return {
        "split": split,
        "metrics": metrics,
        "per_class_recall": compute_per_class_recall(y_true, y_pred, labels),
        "confusion_matrix": compute_confusion_matrix(y_true, y_pred, labels=labels),
        "predictions": prediction_rows,
        "batch_count": batch_count,
    }


def export_evaluation(
    result: Mapping[str, Any],
    *,
    reports_dir: Path,
    predictions_dir: Path,
    label_names: list[str],
) -> dict[str, str]:
    split = str(result["split"])
    reports_dir = Path(reports_dir)
    predictions_dir = Path(predictions_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = reports_dir / f"{split}_metrics.json"
    metrics_path.write_text(
        json.dumps(result["metrics"], ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    prediction_path = predictions_dir / f"{split}_predictions.tsv"
    _write_rows(
        prediction_path,
        list(result["predictions"]),
        [
            "split",
            "dialogue_id",
            "utterance_id",
            "utterance_index",
            "true_label",
            "predicted_label",
        ],
    )
    confusion_path = reports_dir / f"{split}_confusion_matrix.tsv"
    matrix = result["confusion_matrix"]
    with confusion_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file, delimiter="\t", lineterminator="\n")
        writer.writerow(["true\\pred", *label_names])
        for label_name, row in zip(label_names, matrix.tolist()):
            writer.writerow([label_name, *row])
    per_class_path = reports_dir / f"{split}_per_class_metrics.tsv"
    _write_rows(
        per_class_path,
        [
            {
                "label_id": index,
                "label_name": label_name,
                "recall": result["per_class_recall"][index],
            }
            for index, label_name in enumerate(label_names)
        ],
        ["label_id", "label_name", "recall"],
    )
    return {
        "metrics": metrics_path.as_posix(),
        "predictions": prediction_path.as_posix(),
        "confusion_matrix": confusion_path.as_posix(),
        "per_class_metrics": per_class_path.as_posix(),
    }


def _write_rows(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with Path(path).open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fields,
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


__all__ = ["evaluate_model", "export_evaluation", "model_forward", "move_batch"]

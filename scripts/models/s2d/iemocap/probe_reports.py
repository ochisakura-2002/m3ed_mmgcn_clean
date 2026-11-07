"""Atomic, joinable evidence tables for Q1. No scientific conclusions or plots."""

from __future__ import annotations

import csv
import io
import json
import os
from pathlib import Path
import statistics
import tempfile

import torch
from sklearn.metrics import precision_recall_fscore_support

from scripts.models.s2d.iemocap.probe_bank import LABELS
from utils.metrics import compute_classification_metrics, compute_confusion_matrix


METRICS = ("acc", "weighted_f1", "macro_f1", "uar")
PREDICTION_FIELDS = ("utterance_id", "true_label", "pred_label", *[f"prob_{i}" for i in range(6)])
LONG_FIELDS = ("model", "feature", "probe_type", "seed", "best_epoch", *METRICS)
COMPARISONS = (
    ("C0 - A0", "L-C0", "L-A0", "different branches: adapted MCP diagnostic versus frozen RGB"),
    ("D0 - A0", "L-D0", "L-A0", "MCP static residual correction"),
    ("B - D0", "L-B", "L-D0", "full contextualization (multiple components differ)"),
    ("LinearB - OriginalS2D", "L-B", "OriginalS2D", "classifier/training utilization gap"),
    ("TCND0 - LinearD0", "T-D0", "L-D0", "independent sequence information beyond mean pooling"),
    ("TCNB - LinearB", "T-B", "L-B", "remaining contextual slot sequence information"),
)


def atomic_write(path, writer):
    """fsync a sibling temporary, then atomically link without replacing targets.

    The exclusive run directory owns these files. Hard-link publication also
    prevents clobbering if a target appears concurrently. Unsupported filesystems
    fail closed; there is no non-atomic fallback.
    """
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"Refusing existing output: {path}")
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w+b", dir=path.parent, prefix=f".{path.name}.",
                                         suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            writer(stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def atomic_json(path, value):
    content = (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")
    atomic_write(path, lambda stream: stream.write(content))


def atomic_csv(path, rows, fields):
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    content = buffer.getvalue().encode("utf-8")
    atomic_write(path, lambda stream: stream.write(content))


def classification_report(labels, predictions):
    labels, predictions = torch.as_tensor(labels), torch.as_tensor(predictions)
    if (labels.ndim != 1 or labels.numel() == 0 or labels.shape != predictions.shape
            or labels.dtype != torch.int64 or predictions.dtype != torch.int64
            or bool(((labels < 0) | (labels >= 6) | (predictions < 0) | (predictions >= 6)).any())):
        raise ValueError("Expected non-empty aligned six-class integer labels/predictions")
    true, pred = labels.cpu().numpy(), predictions.cpu().numpy()
    indices = list(range(6))
    result = compute_classification_metrics(true, pred, labels=indices)
    precision, recall, f1, support = precision_recall_fscore_support(
        true, pred, labels=indices, zero_division=0)
    result["per_class"] = [dict(label_index=i, label=label, precision=float(precision[i]),
                                recall=float(recall[i]), f1=float(f1[i]), support=int(support[i]))
                           for i, label in enumerate(LABELS)]
    result["confusion_matrix"] = compute_confusion_matrix(true, pred, labels=indices).tolist()
    result["confusion_matrix_labels"] = list(LABELS)
    result["sample_count"] = len(true)
    return result


def write_evaluation(directory, ids, labels, probabilities, report, *, reference=False):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if (probabilities.dtype != torch.float32 or probabilities.shape != (len(ids), 6)
            or not bool(torch.isfinite(probabilities).all()) or bool((probabilities < 0).any())
            or not torch.allclose(probabilities.sum(-1), torch.ones(len(ids)), atol=1e-6, rtol=1e-6)
            or len(set(ids)) != len(ids) or len(labels) != len(ids)):
        raise ValueError("Invalid Dev prediction schema")
    predictions = probabilities.argmax(-1)
    expected = classification_report(labels, predictions)
    # Loss/selection fields may accompany the exact metric report.
    if any(report.get(k) != v for k, v in expected.items()):
        raise ValueError("Report disagrees with Dev probabilities")
    rows = [dict(utterance_id=uid, true_label=int(labels[i]), pred_label=int(predictions[i]),
                 **{f"prob_{c}": float(probabilities[i, c]) for c in range(6)}) for i, uid in enumerate(ids)]
    predictions_name = "original_s2d_dev_predictions.csv" if reference else "predictions_dev.csv"
    per_class_name = "original_s2d_per_class_metrics.csv" if reference else "per_class_metrics.csv"
    matrix_name = "original_s2d_confusion_matrix.csv" if reference else "confusion_matrix.csv"
    atomic_csv(directory / predictions_name, rows, PREDICTION_FIELDS)
    atomic_csv(directory / per_class_name, report["per_class"], report["per_class"][0].keys())
    write_confusion(directory / matrix_name, report)


def write_confusion(path, report):
    rows = [dict(true_label=i, true_label_name=LABELS[i],
                 **{f"pred_{j}": count for j, count in enumerate(counts)})
            for i, counts in enumerate(report["confusion_matrix"])]
    atomic_csv(path, rows, ("true_label", "true_label_name", *[f"pred_{i}" for i in range(6)]))


def summarize(rows):
    groups = {}
    for row in rows:
        groups.setdefault((row["model"], row["feature"], row["probe_type"]), []).append(row)
    result = []
    for (model, feature, kind), group in groups.items():
        summary = dict(model=model, feature=feature, probe_type=kind, n_seeds=len(group))
        for metric in METRICS:
            values = [r[metric] for r in group]
            summary.update({f"{metric}_mean": statistics.mean(values),
                            f"{metric}_std": statistics.stdev(values) if len(values) > 1 else 0.0})
        result.append(summary)
    return result


def pairwise_deltas(rows):
    groups = {}
    for row in rows:
        group = groups.setdefault(row["model"], {})
        if row["seed"] in group:
            raise ValueError("Duplicate model/seed")
        group[row["seed"]] = row
    result = []
    for comparison, lhs, rhs, interpretation in COMPARISONS:
        if lhs not in groups or rhs not in groups:
            continue  # Explicit preflight-only run has no trainable probe rows.
        left, right = groups[lhs], groups[rhs]
        seeds = sorted(left)
        if rhs != "OriginalS2D" and set(left) != set(right):
            raise ValueError("Paired probes must have identical seed sets")
        for metric in METRICS:
            values = [left[seed][metric] - right["" if rhs == "OriginalS2D" else seed][metric]
                      for seed in seeds]
            base = dict(comparison=comparison, lhs=lhs, rhs=rhs, metric=metric,
                        interpretation_boundary=interpretation, n_pairs=len(values))
            for seed, value in zip(seeds, values):
                result.append(dict(base, row_type="per_seed", seed=seed, delta=value,
                                   delta_mean="", delta_std=""))
            result.append(dict(base, row_type="summary", seed="", delta="",
                               delta_mean=statistics.mean(values),
                               delta_std=statistics.stdev(values) if len(values) > 1 else 0.0))
    return result


def write_summaries(root, rows, per_class, histories, evaluations):
    root = Path(root)
    root.mkdir()
    atomic_csv(root / "metrics_long.csv", rows, LONG_FIELDS)
    summary = summarize(rows)
    atomic_csv(root / "metrics_summary.csv", summary, summary[0].keys())
    deltas = pairwise_deltas(rows)
    atomic_csv(root / "pairwise_deltas.csv", deltas, (
        "comparison", "lhs", "rhs", "metric", "interpretation_boundary", "n_pairs", "row_type",
        "seed", "delta", "delta_mean", "delta_std"))
    atomic_csv(root / "per_class_metrics.csv", per_class, per_class[0].keys())
    (root / "confusion_matrices").mkdir()
    for name, report in evaluations.items():
        write_confusion(root / "confusion_matrices" / f"{name}.csv", report)
    (root / "training_curves").mkdir()
    fields = ("model", "feature", "seed", "epoch", "optimization_loss", "train_loss", "dev_loss",
              *[f"{split}_{metric}" for split in ("train", "dev") for metric in METRICS])
    atomic_csv(root / "training_curves/metrics_by_epoch.csv", histories, fields)

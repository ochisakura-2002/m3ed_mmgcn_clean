"""Shared probability metrics and utterance-level prediction records."""

from __future__ import annotations

import re
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np


_SESSION_PATTERN = re.compile(r"^(Ses0[1-5])[FM]_", re.IGNORECASE)


def parse_iemocap_dialogue_metadata(dialogue_id: Any) -> tuple[str, str]:
    """Return strict session and impro/script type derived from an ID."""

    text = str(dialogue_id)
    match = _SESSION_PATTERN.match(text)
    session_id = match.group(1) if match else "unknown"
    lowered = text.lower()
    if "_impro" in lowered:
        dialogue_type = "impro"
    elif "_script" in lowered:
        dialogue_type = "script"
    else:
        dialogue_type = "unknown"
    return session_id, dialogue_type


def compute_calibration_metrics(
    y_true: Sequence[int],
    probabilities: Sequence[Sequence[float]],
    *,
    num_bins: int = 10,
    epsilon: float = 1.0e-12,
) -> Dict[str, float]:
    """Compute ECE, multiclass NLL/Brier, and confidence summaries."""

    targets = np.asarray(y_true, dtype=np.int64)
    probs = np.asarray(probabilities, dtype=np.float64)
    if targets.ndim != 1 or probs.ndim != 2 or probs.shape[0] != targets.shape[0]:
        raise ValueError("Expected y_true [N] and probabilities [N, C].")
    if targets.size == 0:
        raise ValueError("Calibration metrics require at least one prediction.")
    if num_bins <= 0:
        raise ValueError("num_bins must be positive.")
    if not np.isfinite(probs).all():
        raise ValueError("Probabilities contain NaN or Inf.")
    if np.any(probs < 0.0) or np.any(probs > 1.0):
        raise ValueError("Probabilities must lie in [0, 1].")
    if not np.allclose(probs.sum(axis=1), 1.0, atol=1.0e-6, rtol=0.0):
        raise ValueError("Each probability row must sum to 1.")
    if np.any(targets < 0) or np.any(targets >= probs.shape[1]):
        raise ValueError("y_true contains a class outside the probability columns.")

    predicted = probs.argmax(axis=1)
    confidence = probs.max(axis=1)
    correct = predicted == targets
    bin_ids = np.minimum((confidence * int(num_bins)).astype(np.int64), int(num_bins) - 1)
    ece = 0.0
    for bin_id in range(int(num_bins)):
        in_bin = bin_ids == bin_id
        if np.any(in_bin):
            ece += float(in_bin.mean()) * abs(
                float(correct[in_bin].mean()) - float(confidence[in_bin].mean())
            )

    clipped_true = np.clip(probs[np.arange(targets.size), targets], epsilon, 1.0)
    one_hot = np.eye(probs.shape[1], dtype=np.float64)[targets]

    def subset_mean(mask: np.ndarray) -> float:
        return float(confidence[mask].mean()) if np.any(mask) else float("nan")

    return {
        "ece_10": float(ece),
        "nll": float(-np.log(clipped_true).mean()),
        "brier_score": float(np.square(probs - one_hot).sum(axis=1).mean()),
        "mean_confidence": float(confidence.mean()),
        "mean_confidence_correct": subset_mean(correct),
        "mean_confidence_incorrect": subset_mean(~correct),
    }


def _safe_label_column(label: str) -> str:
    value = re.sub(r"[^0-9A-Za-z]+", "_", str(label)).strip("_")
    return value or "class"


def build_prediction_row(
    *,
    split: str,
    dialogue_id: Any,
    utterance_id: Any,
    utterance_index: int,
    true_label_id: int,
    predicted_label_id: int,
    probabilities: Sequence[float],
    label_list: Sequence[str],
    extra: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Build the canonical prediction row while preserving compatibility IDs."""

    values = [float(value) for value in probabilities]
    if len(values) != len(label_list):
        raise ValueError("Probability count must equal label_list length.")
    true_id = int(true_label_id)
    pred_id = int(predicted_label_id)
    session_id, dialogue_type = parse_iemocap_dialogue_metadata(dialogue_id)
    row: Dict[str, Any] = {
        "split": str(split),
        "dialogue_id": str(dialogue_id),
        "utterance_id": str(utterance_id),
        "utterance_index": int(utterance_index),
        "session_id": session_id,
        "dialogue_type": dialogue_type,
        "true_label": str(label_list[true_id]),
        "predicted_label": str(label_list[pred_id]),
        "true_label_id": true_id,
        "pred_label_id": pred_id,
        "true_label_name": str(label_list[true_id]),
        "pred_label_name": str(label_list[pred_id]),
        "confidence": float(max(values)),
    }
    for class_id, (label, probability) in enumerate(zip(label_list, values)):
        row[f"probability_{class_id}"] = probability
        row[f"probability_{_safe_label_column(label)}"] = probability
    if extra:
        row.update(dict(extra))
    return row


__all__ = [
    "build_prediction_row",
    "compute_calibration_metrics",
    "parse_iemocap_dialogue_metadata",
]

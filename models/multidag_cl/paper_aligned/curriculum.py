"""Paper Algorithm 1 curriculum utilities for MultiDAG+CL."""

from __future__ import annotations

import math
from typing import Sequence

import torch


def dialogue_difficulty_from_sequences(
    labels: Sequence[int],
    speakers: Sequence[int],
) -> float:
    """Equation 8: ``(emotion_shifts + speakers) / (utterances + speakers)``."""

    if len(labels) != len(speakers) or not labels:
        raise ValueError("labels and speakers must be non-empty equal-length sequences.")
    histories: dict[int, list[int]] = {}
    for label, speaker in zip(labels, speakers):
        histories.setdefault(int(speaker), []).append(int(label))
    shifts = sum(
        left != right
        for history in histories.values()
        for left, right in zip(history, history[1:])
    )
    speaker_count = len(histories)
    return float((shifts + speaker_count) / (len(labels) + speaker_count))


def dialogue_difficulty(
    labels: torch.Tensor,
    speakers: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Batch form of the paper difficulty measure."""

    if labels.shape != speakers.shape or labels.shape != attention_mask.shape:
        raise ValueError("labels, speakers, and attention_mask must share [B,T].")
    values = []
    for index in range(labels.shape[0]):
        valid = attention_mask[index].bool()
        values.append(
            dialogue_difficulty_from_sequences(
                labels[index][valid].detach().cpu().tolist(),
                speakers[index][valid].detach().cpu().tolist(),
            )
        )
    return labels.new_tensor(values, dtype=torch.float32)


def curriculum_baby_step_indices(
    difficulties: Sequence[float],
    epoch: int,
    bucket_count: int,
) -> list[int]:
    """Return indices visible at an epoch under Algorithm 1 baby steps."""

    if bucket_count <= 0:
        raise ValueError("bucket_count must be positive when curriculum is enabled.")
    if epoch <= 0:
        raise ValueError("epoch is one-indexed and must be positive.")
    ordered = sorted(range(len(difficulties)), key=lambda i: (float(difficulties[i]), i))
    if not ordered:
        return []
    bucket_size = int(math.ceil(len(ordered) / bucket_count))
    visible_buckets = min(int(epoch), int(bucket_count))
    return ordered[: min(len(ordered), visible_buckets * bucket_size)]


__all__ = [
    "curriculum_baby_step_indices",
    "dialogue_difficulty",
    "dialogue_difficulty_from_sequences",
]

"""Mask-first tensor operations used by strict-causal graph baselines."""

from __future__ import annotations

import torch


def masked_softmax(
    scores: torch.Tensor,
    mask: torch.Tensor,
    dim: int = -1,
) -> torch.Tensor:
    """Apply a boolean mask before softmax and return zero for empty rows."""

    if scores.shape != mask.shape:
        try:
            mask = torch.broadcast_to(mask, scores.shape)
        except RuntimeError as exc:
            raise ValueError("mask is not broadcastable to scores") from exc
    if not scores.is_floating_point():
        raise ValueError("scores must be floating point")

    mask_bool = mask.to(device=scores.device, dtype=torch.bool)
    masked_scores = scores.masked_fill(~mask_bool, torch.finfo(scores.dtype).min)
    probabilities = torch.softmax(masked_scores, dim=dim)
    probabilities = probabilities.masked_fill(~mask_bool, 0.0)
    denominator = probabilities.sum(dim=dim, keepdim=True)
    return torch.where(
        denominator > 0,
        probabilities / denominator.clamp_min(torch.finfo(scores.dtype).tiny),
        torch.zeros_like(probabilities),
    )

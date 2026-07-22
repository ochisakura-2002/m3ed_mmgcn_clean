"""Losses for the isolated project causal GS-MCC-inspired baseline."""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask_float = mask.to(device=values.device, dtype=values.dtype)
    denominator = mask_float.sum()
    if not torch.any(mask):
        return values.sum() * 0.0
    return (values * mask_float).sum() / denominator


def _pairwise_modal_objectives(
    low_frequency_modal_repr: torch.Tensor,
    high_frequency_modal_repr: torch.Tensor,
    attention_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute same-utterance project proxies without labels or future positions."""

    if low_frequency_modal_repr.shape != high_frequency_modal_repr.shape:
        raise ValueError("low/high modal representations must share a shape")
    if low_frequency_modal_repr.dim() != 4:
        raise ValueError("modal representations must have shape [B,T,M,H]")
    if attention_mask.shape != low_frequency_modal_repr.shape[:2]:
        raise ValueError("attention_mask must match modal representations")
    num_modalities = low_frequency_modal_repr.shape[2]
    if num_modalities < 2:
        zero = low_frequency_modal_repr.sum() * 0.0
        return zero, zero

    low_normalized = F.normalize(low_frequency_modal_repr, dim=-1, eps=1e-8)
    high_normalized = F.normalize(high_frequency_modal_repr, dim=-1, eps=1e-8)
    consistency_terms = []
    complementarity_terms = []
    for left in range(num_modalities):
        for right in range(left + 1, num_modalities):
            low_cosine = (low_normalized[:, :, left] * low_normalized[:, :, right]).sum(-1)
            high_cosine = (high_normalized[:, :, left] * high_normalized[:, :, right]).sum(-1)
            consistency_terms.append(1.0 - low_cosine)
            complementarity_terms.append(high_cosine.abs())

    consistency = torch.stack(consistency_terms, dim=0).mean(dim=0)
    complementarity = torch.stack(complementarity_terms, dim=0).mean(dim=0)
    mask = attention_mask.to(dtype=torch.bool)
    return _masked_mean(consistency, mask), _masked_mean(complementarity, mask)


def compute_causal_gsmcc_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    attention_mask: torch.Tensor,
    low_frequency_modal_repr: torch.Tensor,
    high_frequency_modal_repr: torch.Tensor,
    classification_weight: float = 1.0,
    consistency_weight: float = 0.0,
    complementarity_weight: float = 0.0,
) -> Dict[str, torch.Tensor]:
    """Return masked CE and explicitly project-local high/low auxiliary losses."""

    if logits.dim() != 3:
        raise ValueError("logits must have shape [B,T,C]")
    if labels.shape != logits.shape[:2] or attention_mask.shape != logits.shape[:2]:
        raise ValueError("labels and attention_mask must have shape [B,T]")
    for name, value in {
        "classification_weight": classification_weight,
        "consistency_weight": consistency_weight,
        "complementarity_weight": complementarity_weight,
    }.items():
        if value < 0:
            raise ValueError(f"{name} must be non-negative")

    valid = attention_mask.to(dtype=torch.bool) & (labels >= 0)
    if torch.any(valid):
        classification_loss = F.cross_entropy(logits[valid], labels[valid].long())
    else:
        classification_loss = logits.sum() * 0.0
    consistency_loss, complementarity_loss = _pairwise_modal_objectives(
        low_frequency_modal_repr,
        high_frequency_modal_repr,
        attention_mask,
    )
    total_loss = (
        float(classification_weight) * classification_loss
        + float(consistency_weight) * consistency_loss
        + float(complementarity_weight) * complementarity_loss
    )
    return {
        "classification_loss": classification_loss,
        "consistency_loss": consistency_loss,
        "complementarity_loss": complementarity_loss,
        "total_loss": total_loss,
    }

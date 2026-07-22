"""Shared tensor helpers for isolated original-paper reproductions."""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


CAUSAL_GRADE = "noncausal_offline_full_context"


def validate_dialogue_batch(
    text_features: torch.Tensor,
    audio_features: torch.Tensor,
    visual_features: torch.Tensor,
    attention_mask: torch.Tensor,
    lengths: torch.Tensor,
    speaker_ids_int: torch.Tensor,
) -> None:
    if text_features.dim() != 3:
        raise ValueError("text_features must have shape [B,T,D].")
    batch_time = text_features.shape[:2]
    for name, value in (
        ("audio_features", audio_features),
        ("visual_features", visual_features),
    ):
        if value.dim() != 3 or value.shape[:2] != batch_time:
            raise ValueError(f"{name} must share [B,T] with text_features.")
    if attention_mask.shape != batch_time or speaker_ids_int.shape != batch_time:
        raise ValueError("attention_mask and speaker_ids_int must have shape [B,T].")
    if lengths.dim() != 1 or lengths.shape[0] != batch_time[0]:
        raise ValueError("lengths must have shape [B].")
    mask_lengths = attention_mask.long().sum(dim=1)
    if not torch.equal(mask_lengths.cpu(), lengths.long().cpu()):
        raise ValueError("attention_mask sums must exactly match lengths.")


def run_packed_rnn(
    rnn: nn.Module,
    inputs: torch.Tensor,
    lengths: torch.Tensor,
) -> torch.Tensor:
    packed = pack_padded_sequence(
        inputs,
        lengths.detach().cpu(),
        batch_first=True,
        enforce_sorted=False,
    )
    packed_output, _ = rnn(packed)
    output, _ = pad_packed_sequence(
        packed_output,
        batch_first=True,
        total_length=inputs.shape[1],
    )
    return output


def flatten_valid(features: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    return features[attention_mask.bool()]


def scatter_valid(
    valid_features: torch.Tensor,
    attention_mask: torch.Tensor,
    output_dim: int,
) -> torch.Tensor:
    output = valid_features.new_zeros((*attention_mask.shape, int(output_dim)))
    output[attention_mask.bool()] = valid_features
    return output


def masked_cross_entropy(
    logits: torch.Tensor,
    labels: Optional[torch.Tensor],
    attention_mask: torch.Tensor,
    class_weight: Optional[torch.Tensor] = None,
) -> Optional[torch.Tensor]:
    if labels is None:
        return None
    valid = attention_mask.bool() & (labels >= 0)
    if not torch.any(valid):
        raise ValueError("A labeled batch must contain at least one valid utterance.")
    weight = None if class_weight is None else class_weight.to(logits)
    return F.cross_entropy(logits[valid], labels[valid].long(), weight=weight)


def masked_softmax(
    scores: torch.Tensor,
    mask: torch.Tensor,
    dim: int = -1,
) -> torch.Tensor:
    boolean_mask = mask.bool()
    masked = scores.masked_fill(~boolean_mask, torch.finfo(scores.dtype).min)
    probabilities = torch.softmax(masked, dim=dim)
    probabilities = probabilities * boolean_mask.to(probabilities.dtype)
    normalizer = probabilities.sum(dim=dim, keepdim=True)
    return torch.where(normalizer > 0, probabilities / normalizer.clamp_min(1e-12), probabilities)


def active_feature_concat(
    text_features: torch.Tensor,
    audio_features: torch.Tensor,
    visual_features: torch.Tensor,
    active_modalities: Sequence[str],
) -> torch.Tensor:
    lookup = {
        "text": text_features,
        "audio": audio_features,
        "visual": visual_features,
    }
    normalized = [str(value).lower() for value in active_modalities]
    if not normalized or any(value not in lookup for value in normalized):
        raise ValueError("active_modalities must contain text/audio/visual names.")
    return torch.cat([lookup[value] for value in normalized], dim=-1)


def structured_output(
    *,
    logits: torch.Tensor,
    classification_loss: Optional[torch.Tensor],
    aux_losses: Optional[Mapping[str, torch.Tensor]] = None,
    features: Optional[Mapping[str, Any]] = None,
    diagnostics: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    auxiliaries = dict(aux_losses or {})
    total_loss = classification_loss
    for value in auxiliaries.values():
        total_loss = value if total_loss is None else total_loss + value
    merged_diagnostics = {"causal_grade": CAUSAL_GRADE}
    merged_diagnostics.update(dict(diagnostics or {}))
    return {
        "logits": logits,
        "loss": total_loss,
        "classification_loss": classification_loss,
        "aux_losses": auxiliaries,
        "features": dict(features or {}),
        "diagnostics": merged_diagnostics,
    }


def class_weight_tensor(
    values: Optional[Sequence[float]],
    num_classes: Optional[int] = None,
) -> Optional[torch.Tensor]:
    if values is None:
        return None
    tensor = torch.as_tensor(list(values), dtype=torch.float32)
    if tensor.dim() != 1 or tensor.numel() == 0 or torch.any(tensor <= 0):
        raise ValueError("class_weight values must be a non-empty positive vector.")
    if num_classes is not None and tensor.numel() != int(num_classes):
        raise ValueError("class_weight length must equal num_classes.")
    return tensor


__all__ = [
    "CAUSAL_GRADE",
    "active_feature_concat",
    "class_weight_tensor",
    "flatten_valid",
    "masked_cross_entropy",
    "masked_softmax",
    "run_packed_rnn",
    "scatter_valid",
    "structured_output",
    "validate_dialogue_batch",
]

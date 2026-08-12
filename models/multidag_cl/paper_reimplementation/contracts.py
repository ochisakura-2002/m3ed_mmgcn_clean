"""Typed tensor and metadata contracts for the independent Stage-B2 core."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional

import torch

from .config import MultiDAGCLConfig


MISSING_LABEL_INDEX = -1


@dataclass(frozen=True)
class ContextVisibilityIdentity:
    dag_topology_causal: bool
    end_to_end_causal: bool
    end_to_end_causal_assuming_local_text_features: bool
    upstream_text_feature_locality: str
    context_leakage_risk: tuple[str, ...]


@dataclass(frozen=True)
class EncodedModalities:
    audio: torch.Tensor
    visual: torch.Tensor
    text: torch.Tensor
    fused: torch.Tensor
    encoder_profile: str
    modality_order: tuple[str, str, str]
    context_visibility_identity: ContextVisibilityIdentity
    model_math_deviation_list: tuple[str, ...]


@dataclass(frozen=True)
class SpeakerRelations:
    same_speaker: torch.Tensor
    different_speaker: torch.Tensor


@dataclass(frozen=True)
class AttentionResult:
    logits: torch.Tensor
    weights: torch.Tensor
    message: torch.Tensor


@dataclass(frozen=True)
class DualGRUResult:
    node_state: torch.Tensor
    context_state: torch.Tensor
    state: torch.Tensor


@dataclass(frozen=True)
class DAGLayerResult:
    state: torch.Tensor
    attention_logits: Optional[torch.Tensor]
    attention_weights: Optional[torch.Tensor]
    messages: Optional[torch.Tensor]


@dataclass(frozen=True)
class ModelDiagnostics:
    adjacency: torch.Tensor
    relations: SpeakerRelations
    layer_diagnostics: tuple[DAGLayerResult, ...]


@dataclass(frozen=True)
class ModelOutput:
    logits: torch.Tensor
    loss: Optional[torch.Tensor]
    encoded_state: torch.Tensor
    encoded_modalities: Optional[EncodedModalities]
    raw_concat: Optional[torch.Tensor]
    layer_states: tuple[torch.Tensor, ...]
    representation: torch.Tensor
    profile_identity: str
    context_visibility_identity: ContextVisibilityIdentity
    diagnostics: Optional[ModelDiagnostics]


@dataclass(frozen=True)
class DialogueRecord:
    dialogue_id: str
    original_train_index: int
    difficulty: float


@dataclass(frozen=True)
class BucketManifest:
    profile: str
    configured_bucket_count: int
    actual_bucket_count: int
    ordered_dialogue_ids: tuple[str, ...]
    original_indices: tuple[int, ...]
    difficulties: tuple[float, ...]
    bucket_membership: tuple[tuple[str, ...], ...]
    membership_sha256: str


class MultiDAGBatchContract:
    """Validate the named project batch without moving or mutating tensors."""

    _TENSOR_FIELDS = (
        "text_features",
        "audio_features",
        "visual_features",
        "speaker_ids_int",
        "attention_mask",
        "lengths",
    )

    def __init__(self, config: MultiDAGCLConfig) -> None:
        if not isinstance(config, MultiDAGCLConfig):
            raise TypeError("config must be MultiDAGCLConfig")
        self.config = config

    def validate(
        self,
        batch: Mapping[str, Any],
        *,
        require_labels: bool,
        split: str,
    ) -> None:
        if not isinstance(batch, Mapping):
            raise TypeError("batch must be a mapping")
        if not isinstance(require_labels, bool):
            raise TypeError("require_labels must be bool")
        if not isinstance(split, str) or not split.strip():
            raise ValueError("split must be a non-empty explicit string")

        for name in self._TENSOR_FIELDS:
            if name not in batch:
                raise ValueError(f"missing required batch field: {name}")
            if not isinstance(batch[name], torch.Tensor):
                raise TypeError(f"{name} must be a torch.Tensor")
        labels = batch.get("labels")
        if require_labels and labels is None:
            raise ValueError("labels are required for this batch")
        if labels is not None and not isinstance(labels, torch.Tensor):
            raise TypeError("labels must be a torch.Tensor when present")

        text = batch["text_features"]
        audio = batch["audio_features"]
        visual = batch["visual_features"]
        speakers = batch["speaker_ids_int"]
        attention_mask = batch["attention_mask"]
        lengths = batch["lengths"]

        for name, value, expected_dim in (
            ("text_features", text, self.config.text_feature_dim),
            ("audio_features", audio, self.config.audio_feature_dim),
            ("visual_features", visual, self.config.visual_feature_dim),
        ):
            if value.dtype is not torch.float32:
                raise TypeError(f"{name} must have dtype float32")
            if value.dim() != 3:
                raise ValueError(f"{name} must have shape [B,T,D]")
            if value.shape[-1] != expected_dim:
                raise ValueError(
                    f"{name} feature dimension {value.shape[-1]} != configured {expected_dim}"
                )
            if not torch.isfinite(value).all():
                raise ValueError(f"{name} must contain only finite values")

        batch_time = text.shape[:2]
        if audio.shape[:2] != batch_time or visual.shape[:2] != batch_time:
            raise ValueError("all modality tensors must share [B,T]")
        batch_size, max_time = batch_time
        if batch_size < 1 or max_time < 1:
            raise ValueError("batch and time dimensions must be non-empty")

        if speakers.dtype is not torch.int64 or speakers.shape != batch_time:
            raise TypeError("speaker_ids_int must be int64 [B,T]")
        if attention_mask.dtype not in (torch.int64, torch.bool) or attention_mask.shape != batch_time:
            raise TypeError("attention_mask must be int64 or bool [B,T]")
        if lengths.dtype is not torch.int64 or lengths.shape != (batch_size,):
            raise TypeError("lengths must be int64 [B]")
        if labels is not None and (labels.dtype is not torch.int64 or labels.shape != batch_time):
            raise TypeError("labels must be int64 [B,T]")

        devices = {batch[name].device for name in self._TENSOR_FIELDS}
        if labels is not None:
            devices.add(labels.device)
        if len(devices) != 1:
            raise ValueError("all tensor fields must share one device")

        if attention_mask.dtype is torch.int64 and not torch.all(
            (attention_mask == 0) | (attention_mask == 1)
        ):
            raise ValueError("attention_mask may contain only 0/1")
        mask = attention_mask.bool()
        if torch.any(lengths < 1) or torch.any(lengths > max_time):
            raise ValueError("every dialogue must have length in [1,T]")
        if not torch.equal(mask.long().sum(dim=1), lengths):
            raise ValueError("lengths must equal attention_mask sums")
        positions = torch.arange(max_time, device=lengths.device).unsqueeze(0)
        expected_mask = positions < lengths.unsqueeze(1)
        if not torch.equal(mask, expected_mask):
            raise ValueError("attention_mask must be a contiguous valid prefix with right padding")

        padded = ~mask
        for name, value in (
            ("text_features", text),
            ("audio_features", audio),
            ("visual_features", visual),
        ):
            if torch.count_nonzero(value[padded]).item() != 0:
                raise ValueError(f"{name} padded rows must be exact zero")
        if torch.any(speakers[mask] < 0):
            raise ValueError("valid speaker IDs must be non-negative")
        if torch.count_nonzero(speakers[padded]).item() != 0:
            raise ValueError("padded speaker IDs must be zero and excluded by the mask")

        if labels is not None:
            valid_labels = labels[mask]
            legal = (valid_labels == MISSING_LABEL_INDEX) | (
                (valid_labels >= 0) & (valid_labels < self.config.num_classes)
            )
            if not torch.all(legal):
                raise ValueError("valid labels must be -1 or in [0,num_classes)")
            if not torch.all(labels[padded] == self.config.loss_ignore_index):
                raise ValueError("padded labels must equal the configured -100 ignore index")

        dialogue_ids = batch.get("dialogue_ids")
        utterance_ids = batch.get("utterance_ids")
        if not isinstance(dialogue_ids, list) or len(dialogue_ids) != batch_size:
            raise TypeError("dialogue_ids must be list[str] with length B")
        if not all(isinstance(item, str) and item.strip() for item in dialogue_ids):
            raise ValueError("dialogue_ids must be non-empty strings")
        if len(set(dialogue_ids)) != len(dialogue_ids):
            raise ValueError("dialogue_ids must be unique within a batch")
        if not isinstance(utterance_ids, list) or len(utterance_ids) != batch_size:
            raise TypeError("utterance_ids must be list[list[str]] with length B")
        length_values = lengths.detach().cpu().tolist()
        for index, (ids, length) in enumerate(zip(utterance_ids, length_values)):
            if not isinstance(ids, list) or not all(
                isinstance(item, str) and item.strip() for item in ids
            ):
                raise TypeError(f"utterance_ids[{index}] must be list[str]")
            if len(ids) != int(length):
                raise ValueError("utterance_ids lengths must exactly match tensor lengths")
            if len(set(ids)) != len(ids):
                raise ValueError("utterance_ids must be unique within each dialogue")


__all__ = [
    "AttentionResult",
    "BucketManifest",
    "ContextVisibilityIdentity",
    "DAGLayerResult",
    "DialogueRecord",
    "DualGRUResult",
    "EncodedModalities",
    "ModelDiagnostics",
    "ModelOutput",
    "MultiDAGBatchContract",
    "SpeakerRelations",
]

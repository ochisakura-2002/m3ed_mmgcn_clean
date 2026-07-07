"""Device-neutral MultiDAG+CL-style baseline for project dialogue batches.

This module keeps the project-facing contract batch-first:

    text_features:   [B, T, D_text]
    audio_features:  [B, T, D_audio]
    visual_features: [B, T, D_visual]
    attention_mask:  [B, T]
    speaker_ids_int: [B, T]
    labels:          [B, T]

It is not a line-by-line copy of the official MultiDAG+CL implementation. The
port preserves the core shape and modeling direction needed here: feature-level
multimodal input, a directed past dialogue graph, speaker-aware relations,
batch-first padded tensors, masked loss, and no device hardcoding.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


_VALID_MODALITIES = ("text", "audio", "visual")


def build_directed_past_adjacency(
    attention_mask: torch.Tensor,
    speaker_ids_int: torch.Tensor,
    window_past: int,
) -> torch.Tensor:
    """Build a causal adjacency matrix for padded dialogue batches.

    ``adj[b, i, j] = 1`` means utterance ``i`` may read utterance ``j``.
    The matrix includes self-loops for valid utterances and never includes
    future or padded nodes.
    """

    if attention_mask.dim() != 2:
        raise ValueError(
            f"attention_mask must be [B, T], got {tuple(attention_mask.shape)}"
        )
    if speaker_ids_int.shape != attention_mask.shape:
        raise ValueError(
            "speaker_ids_int must have the same [B, T] shape as attention_mask"
        )
    if attention_mask.device != speaker_ids_int.device:
        raise ValueError("attention_mask and speaker_ids_int must share device")
    if int(window_past) < 0:
        raise ValueError("window_past must be non-negative")

    mask = attention_mask.to(dtype=torch.bool)
    _, seq_len = mask.shape
    device = mask.device

    positions = torch.arange(seq_len, device=device)
    row_positions = positions.view(1, seq_len, 1)
    col_positions = positions.view(1, 1, seq_len)
    causal = col_positions <= row_positions

    valid_row = mask.unsqueeze(2)
    valid_col = mask.unsqueeze(1)

    valid_order = mask.to(dtype=torch.long).cumsum(dim=1) - 1
    order_diff = valid_order.unsqueeze(2) - valid_order.unsqueeze(1)
    if int(window_past) == 0:
        in_window = order_diff == 0
    else:
        in_window = (order_diff >= 0) & (order_diff <= int(window_past))

    adjacency = valid_row & valid_col & causal & in_window
    return adjacency.to(dtype=torch.float32)


def build_speaker_relation_masks(
    attention_mask: torch.Tensor,
    speaker_ids_int: Optional[torch.Tensor],
    num_speakers: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return same- and different-speaker masks for valid utterance pairs."""

    if attention_mask.dim() != 2:
        raise ValueError(
            f"attention_mask must be [B, T], got {tuple(attention_mask.shape)}"
        )
    if int(num_speakers) <= 0:
        raise ValueError("num_speakers must be positive")

    mask = attention_mask.to(dtype=torch.bool)
    batch_size, seq_len = mask.shape
    if speaker_ids_int is None:
        speakers = torch.zeros(
            batch_size,
            seq_len,
            device=mask.device,
            dtype=torch.long,
        )
    else:
        if speaker_ids_int.shape != mask.shape:
            raise ValueError(
                "speaker_ids_int must have the same [B, T] shape as attention_mask"
            )
        if speaker_ids_int.device != mask.device:
            raise ValueError("speaker_ids_int and attention_mask must share device")
        speakers = speaker_ids_int.to(dtype=torch.long)

    speakers = speakers.clamp(0, int(num_speakers) - 1)
    valid_pair = mask.unsqueeze(2) & mask.unsqueeze(1)
    same = valid_pair & (speakers.unsqueeze(2) == speakers.unsqueeze(1))
    different = valid_pair & (speakers.unsqueeze(2) != speakers.unsqueeze(1))
    return same.to(dtype=torch.long), different.to(dtype=torch.long)


def _masked_normalized_softmax(
    scores: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    mask = mask.to(dtype=torch.bool, device=scores.device)
    masked_scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
    weights = torch.softmax(masked_scores, dim=-1) * mask.to(dtype=scores.dtype)
    denom = weights.sum(dim=-1, keepdim=True).clamp_min(
        torch.finfo(scores.dtype).tiny
    )
    return weights / denom


class _SpeakerAwareDAGLayer(nn.Module):
    """One causal speaker-aware graph update layer."""

    def __init__(self, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.query = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.key = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.value_same = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.value_different = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.relation_bias = nn.Embedding(2, 1)
        self.gru = nn.GRUCell(self.hidden_dim, self.hidden_dim)
        self.dropout = nn.Dropout(float(dropout))
        self.update_norm = nn.LayerNorm(self.hidden_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.feed_forward_norm = nn.LayerNorm(self.hidden_dim)

    def forward(
        self,
        features: torch.Tensor,
        adjacency: torch.Tensor,
        same_speaker_mask: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, seq_len, hidden_dim = features.shape
        if hidden_dim != self.hidden_dim:
            raise ValueError(
                f"expected hidden_dim={self.hidden_dim}, got {hidden_dim}"
            )

        valid = attention_mask.to(dtype=torch.bool, device=features.device)
        outputs = []
        for index in range(seq_len):
            if outputs:
                previous_states = torch.stack(outputs, dim=1)
                source_states = torch.cat(
                    [previous_states, features[:, index : index + 1, :]],
                    dim=1,
                )
            else:
                source_states = features[:, index : index + 1, :]

            context = self._gather_context(
                query=features[:, index, :],
                source_states=source_states,
                adjacency_row=adjacency[:, index, : index + 1],
                same_speaker_row=same_speaker_mask[:, index, : index + 1],
            )
            updated = self.gru(features[:, index, :], context)
            updated = self.update_norm(features[:, index, :] + self.dropout(updated))
            ff = self.feed_forward(updated)
            updated = self.feed_forward_norm(updated + self.dropout(ff))
            updated = torch.where(
                valid[:, index].unsqueeze(-1),
                updated,
                torch.zeros_like(updated),
            )
            outputs.append(updated)

        if not outputs:
            return features.new_zeros(batch_size, seq_len, self.hidden_dim)
        return torch.stack(outputs, dim=1)

    def _gather_context(
        self,
        query: torch.Tensor,
        source_states: torch.Tensor,
        adjacency_row: torch.Tensor,
        same_speaker_row: torch.Tensor,
    ) -> torch.Tensor:
        query_h = self.query(query).unsqueeze(1)
        key_h = self.key(source_states)
        scores = (query_h * key_h).sum(dim=-1) / math.sqrt(float(self.hidden_dim))

        relation_ids = same_speaker_row.to(device=query.device, dtype=torch.long)
        scores = scores + self.relation_bias(relation_ids).squeeze(-1)

        attention = _masked_normalized_softmax(scores, adjacency_row)
        same_value = self.value_same(source_states)
        different_value = self.value_different(source_states)
        values = torch.where(
            same_speaker_row.to(device=query.device, dtype=torch.bool).unsqueeze(-1),
            same_value,
            different_value,
        )
        return torch.bmm(attention.unsqueeze(1), values).squeeze(1)


class MultiDAGCLBaseline(nn.Module):
    """Project-shaped MultiDAG+CL-style baseline for padded dialogue tensors."""

    def __init__(
        self,
        text_dim: int,
        audio_dim: int,
        visual_dim: int,
        hidden_dim: int,
        num_classes: int,
        num_speakers: int = 2,
        window_past: int = 5,
        dropout: float = 0.1,
        active_modalities: tuple[str, ...] = ("text", "audio", "visual"),
        num_graph_layers: int = 2,
    ) -> None:
        super().__init__()
        self.text_dim = int(text_dim)
        self.audio_dim = int(audio_dim)
        self.visual_dim = int(visual_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_classes = int(num_classes)
        self.num_speakers = int(num_speakers)
        self.window_past = int(window_past)
        self.active_modalities = tuple(active_modalities)
        self.num_graph_layers = int(num_graph_layers)

        self._validate_init()

        self.modality_projections = nn.ModuleDict(
            {
                "text": nn.Linear(self.text_dim, self.hidden_dim, bias=False),
                "audio": nn.Linear(self.audio_dim, self.hidden_dim, bias=False),
                "visual": nn.Linear(self.visual_dim, self.hidden_dim, bias=False),
            }
        )
        self.input_dropout = nn.Dropout(float(dropout))
        self.fusion = nn.Sequential(
            nn.Linear(len(self.active_modalities) * self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.LayerNorm(self.hidden_dim),
        )

        self.graph_layers = nn.ModuleList(
            [
                _SpeakerAwareDAGLayer(self.hidden_dim, float(dropout))
                for _ in range(self.num_graph_layers)
            ]
        )

        classifier_dim = self.hidden_dim * (self.num_graph_layers + 1)
        self.classifier = nn.Sequential(
            nn.Linear(classifier_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.hidden_dim, self.num_classes),
        )

    def forward(
        self,
        text_features: torch.Tensor,
        audio_features: torch.Tensor,
        visual_features: torch.Tensor,
        attention_mask: torch.Tensor,
        speaker_ids_int: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """Run MultiDAG+CL-style forward and optionally compute masked CE."""

        self._validate_inputs(
            text_features=text_features,
            audio_features=audio_features,
            visual_features=visual_features,
            attention_mask=attention_mask,
            speaker_ids_int=speaker_ids_int,
            labels=labels,
        )

        mask_bool = attention_mask.to(dtype=torch.bool)
        valid_scale = mask_bool.unsqueeze(-1).to(dtype=text_features.dtype)
        speaker_ids = self._speaker_indices(mask_bool, speaker_ids_int)

        modality_inputs = {
            "text": text_features,
            "audio": audio_features,
            "visual": visual_features,
        }
        projected = {}
        for name, tensor in modality_inputs.items():
            hidden = F.relu(self.modality_projections[name](tensor))
            projected[name] = self.input_dropout(hidden) * valid_scale

        fused_input = torch.cat(
            [projected[name] for name in self.active_modalities],
            dim=-1,
        )
        fused_features = self.fusion(fused_input) * valid_scale

        adjacency = build_directed_past_adjacency(
            attention_mask=mask_bool,
            speaker_ids_int=speaker_ids,
            window_past=self.window_past,
        ).to(device=text_features.device)
        same_speaker_mask, different_speaker_mask = build_speaker_relation_masks(
            attention_mask=mask_bool,
            speaker_ids_int=speaker_ids,
            num_speakers=self.num_speakers,
        )

        graph_states = [fused_features]
        hidden = fused_features
        for layer in self.graph_layers:
            hidden = layer(
                features=hidden,
                adjacency=adjacency,
                same_speaker_mask=same_speaker_mask,
                attention_mask=mask_bool,
            )
            hidden = hidden * valid_scale
            graph_states.append(hidden)

        contextual_features = torch.cat(graph_states, dim=-1)
        logits = self.classifier(contextual_features)
        logits = logits.masked_fill(~mask_bool.unsqueeze(-1), 0.0)

        loss = None
        aux_losses: Dict[str, torch.Tensor] = {}
        if labels is not None:
            loss = self._masked_cross_entropy(logits, labels, mask_bool)
            aux_losses["ce"] = loss
            aux_losses["total"] = loss

        return {
            "logits": logits,
            "loss": loss,
            "aux_losses": aux_losses,
            "adjacency": adjacency,
            "same_speaker_mask": same_speaker_mask,
            "different_speaker_mask": different_speaker_mask,
            "features": contextual_features,
        }

    def _speaker_indices(
        self,
        attention_mask: torch.Tensor,
        speaker_ids_int: Optional[torch.Tensor],
    ) -> torch.Tensor:
        batch_size, seq_len = attention_mask.shape
        if speaker_ids_int is None:
            speaker_ids_int = torch.zeros(
                batch_size,
                seq_len,
                device=attention_mask.device,
                dtype=torch.long,
            )
        else:
            speaker_ids_int = speaker_ids_int.to(
                device=attention_mask.device,
                dtype=torch.long,
            )
        return speaker_ids_int.clamp(0, self.num_speakers - 1)

    def _masked_cross_entropy(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        labels = labels.to(device=logits.device, dtype=torch.long)
        valid = attention_mask.to(device=logits.device) & (labels >= 0)
        if not valid.any():
            return logits.sum() * 0.0
        return F.cross_entropy(logits[valid], labels[valid])

    def _validate_init(self) -> None:
        dims = {
            "text_dim": self.text_dim,
            "audio_dim": self.audio_dim,
            "visual_dim": self.visual_dim,
            "hidden_dim": self.hidden_dim,
            "num_classes": self.num_classes,
            "num_speakers": self.num_speakers,
        }
        for name, value in dims.items():
            if int(value) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.window_past < 0:
            raise ValueError("window_past must be non-negative")
        if self.num_graph_layers <= 0:
            raise ValueError("num_graph_layers must be positive")
        if not self.active_modalities:
            raise ValueError("active_modalities must not be empty")
        if len(set(self.active_modalities)) != len(self.active_modalities):
            raise ValueError("active_modalities must not contain duplicates")
        unknown = sorted(set(self.active_modalities) - set(_VALID_MODALITIES))
        if unknown:
            raise ValueError(f"unknown active_modalities: {unknown}")

    def _validate_inputs(
        self,
        text_features: torch.Tensor,
        audio_features: torch.Tensor,
        visual_features: torch.Tensor,
        attention_mask: torch.Tensor,
        speaker_ids_int: Optional[torch.Tensor],
        labels: Optional[torch.Tensor],
    ) -> None:
        if text_features.dim() != 3:
            raise ValueError(
                f"text_features must be [B, T, D], got {tuple(text_features.shape)}"
            )
        if audio_features.dim() != 3:
            raise ValueError(
                f"audio_features must be [B, T, D], got {tuple(audio_features.shape)}"
            )
        if visual_features.dim() != 3:
            raise ValueError(
                f"visual_features must be [B, T, D], got {tuple(visual_features.shape)}"
            )

        batch_size, seq_len, text_dim = text_features.shape
        audio_shape = audio_features.shape
        visual_shape = visual_features.shape
        if audio_shape[:2] != (batch_size, seq_len):
            raise ValueError("audio_features must share B and T with text_features")
        if visual_shape[:2] != (batch_size, seq_len):
            raise ValueError("visual_features must share B and T with text_features")
        if text_dim != self.text_dim:
            raise ValueError(f"expected text_dim={self.text_dim}, got {text_dim}")
        if audio_shape[-1] != self.audio_dim:
            raise ValueError(f"expected audio_dim={self.audio_dim}, got {audio_shape[-1]}")
        if visual_shape[-1] != self.visual_dim:
            raise ValueError(
                f"expected visual_dim={self.visual_dim}, got {visual_shape[-1]}"
            )

        if attention_mask.shape != (batch_size, seq_len):
            raise ValueError(
                "attention_mask must be [B, T], "
                f"got {tuple(attention_mask.shape)}"
            )
        if speaker_ids_int is not None and speaker_ids_int.shape != (batch_size, seq_len):
            raise ValueError(
                "speaker_ids_int must be [B, T], "
                f"got {tuple(speaker_ids_int.shape)}"
            )
        if labels is not None and labels.shape != (batch_size, seq_len):
            raise ValueError(f"labels must be [B, T], got {tuple(labels.shape)}")

        devices = {
            text_features.device,
            audio_features.device,
            visual_features.device,
            attention_mask.device,
        }
        if speaker_ids_int is not None:
            devices.add(speaker_ids_int.device)
        if labels is not None:
            devices.add(labels.device)
        if len(devices) != 1:
            raise ValueError("all input tensors must be on the same device")

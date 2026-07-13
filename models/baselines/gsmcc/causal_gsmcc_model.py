"""Project causal GS-MCC-inspired baseline with no official-code equivalence claim."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from models.baselines.causal_graph_common import (
    build_causal_multimodal_adjacency,
    row_normalize_adjacency,
    utterance_time_to_multimodal_node_time,
)
from models.baselines.gsmcc.causal_spectral_ops import CausalDirectedPolynomialLayer


_ENCODERS = ("linear", "causal_gru")
_FUSIONS = ("concat", "gate", "sum")


@dataclass(frozen=True)
class CausalGSMCCConfig:
    """Validated configuration for :class:`CausalGSMCCInspiredBaseline`."""

    text_dim: int
    audio_dim: int
    visual_dim: int
    hidden_dim: int
    num_classes: int
    dropout: float = 0.1
    window_past: Optional[int] = None
    num_filter_steps: int = 2
    num_graph_layers: int = 1
    modality_encoder_type: str = "linear"
    fusion_type: str = "concat"
    classification_weight: float = 1.0
    consistency_weight: float = 0.0
    complementarity_weight: float = 0.0
    context_mode: str = "causal"
    window_future: int = 0
    bidirectional: bool = False
    nodal_attention: str = "none"

    def __post_init__(self) -> None:
        for name in ("text_dim", "audio_dim", "visual_dim", "hidden_dim", "num_classes"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0,1)")
        if self.window_past is not None and self.window_past < 0:
            raise ValueError("window_past must be None or non-negative")
        if self.num_filter_steps <= 0 or self.num_graph_layers <= 0:
            raise ValueError("num_filter_steps and num_graph_layers must be positive")
        if self.modality_encoder_type not in _ENCODERS:
            raise ValueError(f"modality_encoder_type must be one of {_ENCODERS}")
        if self.fusion_type not in _FUSIONS:
            raise ValueError(f"fusion_type must be one of {_FUSIONS}")
        for name in (
            "classification_weight",
            "consistency_weight",
            "complementarity_weight",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")
        if self.context_mode != "causal":
            raise ValueError("context_mode must explicitly be 'causal'")
        if self.window_future > 0:
            raise ValueError("window_future > 0 is forbidden for a causal model")
        if self.window_future != 0:
            raise ValueError("window_future must be exactly 0")
        if self.bidirectional:
            raise ValueError("bidirectional=True is forbidden for a causal model")
        if self.nodal_attention != "none":
            raise ValueError("nodal attention is disabled in this causal implementation")


class _CausalModalityEncoder(nn.Module):
    """Position-wise projection with an optional forward-only GRU."""

    def __init__(self, input_dim: int, config: CausalGSMCCConfig) -> None:
        super().__init__()
        self.encoder_type = config.modality_encoder_type
        self.hidden_dim = config.hidden_dim
        self.projection = nn.Linear(input_dim, config.hidden_dim)
        self.input_norm = nn.LayerNorm(config.hidden_dim)
        self.dropout = nn.Dropout(config.dropout)
        self.gru: Optional[nn.GRU]
        if self.encoder_type == "causal_gru":
            self.gru = nn.GRU(
                config.hidden_dim,
                config.hidden_dim,
                batch_first=True,
                bidirectional=False,
            )
            self.output_norm = nn.LayerNorm(config.hidden_dim)
        else:
            self.gru = None
            self.output_norm = nn.Identity()

    def forward(
        self,
        features: torch.Tensor,
        attention_mask: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        mask_scale = attention_mask.unsqueeze(-1).to(dtype=features.dtype)
        hidden = self.dropout(F.gelu(self.input_norm(self.projection(features))))
        hidden = hidden * mask_scale
        if self.gru is not None:
            packed = pack_padded_sequence(
                hidden,
                lengths.detach().cpu().tolist(),
                batch_first=True,
                enforce_sorted=False,
            )
            encoded_packed, _ = self.gru(packed)
            hidden, _ = pad_packed_sequence(
                encoded_packed,
                batch_first=True,
                total_length=features.shape[1],
            )
            hidden = self.output_norm(hidden)
        return hidden * mask_scale


class CausalGSMCCInspiredBaseline(nn.Module):
    """Strictly causal, pure-PyTorch GS-MCC-inspired project candidate."""

    def __init__(self, config: CausalGSMCCConfig) -> None:
        super().__init__()
        if not isinstance(config, CausalGSMCCConfig):
            raise TypeError("config must be a CausalGSMCCConfig")
        self.config = config
        self.encoders = nn.ModuleDict(
            {
                "text": _CausalModalityEncoder(config.text_dim, config),
                "audio": _CausalModalityEncoder(config.audio_dim, config),
                "visual": _CausalModalityEncoder(config.visual_dim, config),
            }
        )
        self.spectral_layers = nn.ModuleList(
            [
                CausalDirectedPolynomialLayer(
                    config.hidden_dim,
                    config.num_filter_steps,
                    config.dropout,
                )
                for _ in range(config.num_graph_layers)
            ]
        )

        if config.fusion_type == "concat":
            self.fusion_projection: Optional[nn.Module] = nn.Linear(
                2 * config.hidden_dim,
                config.hidden_dim,
            )
            self.fusion_gate: Optional[nn.Module] = None
        elif config.fusion_type == "gate":
            self.fusion_projection = None
            self.fusion_gate = nn.Linear(2 * config.hidden_dim, config.hidden_dim)
        else:
            self.fusion_projection = None
            self.fusion_gate = None
        self.fusion_norm = nn.LayerNorm(config.hidden_dim)
        self.fusion_dropout = nn.Dropout(config.dropout)
        self.classifier = nn.Linear(config.hidden_dim, config.num_classes)

    def forward(
        self,
        text_features: torch.Tensor,
        audio_features: torch.Tensor,
        visual_features: torch.Tensor,
        attention_mask: torch.Tensor,
        lengths: torch.Tensor,
        speaker_ids_int: torch.Tensor,
        return_aux: bool = False,
    ) -> Any:
        """Return ``[B,T,C]`` logits or a dictionary of causal diagnostics."""

        self._validate_inputs(
            text_features,
            audio_features,
            visual_features,
            attention_mask,
            lengths,
            speaker_ids_int,
        )
        mask = attention_mask.to(dtype=torch.bool)
        projected = [
            self.encoders[name](features, mask, lengths)
            for name, features in (
                ("text", text_features),
                ("audio", audio_features),
                ("visual", visual_features),
            )
        ]
        modal_features = torch.stack(projected, dim=2)
        batch_size, seq_len, num_modalities, hidden_dim = modal_features.shape
        node_features = modal_features.reshape(batch_size, seq_len * num_modalities, hidden_dim)
        node_mask = mask.unsqueeze(-1).expand(batch_size, seq_len, num_modalities)
        node_mask = node_mask.reshape(batch_size, seq_len * num_modalities)
        node_scale = node_mask.unsqueeze(-1).to(dtype=node_features.dtype)

        adjacency = build_causal_multimodal_adjacency(
            mask,
            lengths,
            window_past=self.config.window_past,
            num_modalities=num_modalities,
            include_cross_modal_history=True,
            include_same_time_cross_modal=True,
            self_loop=True,
        )
        normalized_adjacency = row_normalize_adjacency(adjacency).to(
            dtype=node_features.dtype
        )

        hidden = node_features
        low_nodes = node_features
        high_nodes = torch.zeros_like(node_features)
        for layer in self.spectral_layers:
            hidden, low_nodes, high_nodes = layer(hidden, normalized_adjacency)
            hidden = hidden * node_scale
            low_nodes = low_nodes * node_scale
            high_nodes = high_nodes * node_scale

        low_modal = low_nodes.reshape(
            batch_size,
            seq_len,
            num_modalities,
            hidden_dim,
        )
        high_modal = high_nodes.reshape(
            batch_size,
            seq_len,
            num_modalities,
            hidden_dim,
        )
        low_frequency_repr = low_modal.mean(dim=2)
        high_frequency_repr = high_modal.mean(dim=2)
        fused_repr = self._fuse(low_frequency_repr, high_frequency_repr)
        fused_repr = fused_repr * mask.unsqueeze(-1).to(dtype=fused_repr.dtype)
        logits = self.classifier(fused_repr)
        logits = logits.masked_fill(~mask.unsqueeze(-1), 0.0)
        if not return_aux:
            return logits

        node_time = utterance_time_to_multimodal_node_time(
            seq_len,
            num_modalities=num_modalities,
            device=logits.device,
        )
        diagnostics: Dict[str, torch.Tensor] = {
            "node_time": node_time,
            "edge_count_per_dialogue": adjacency.sum(dim=(1, 2)),
            "valid_node_count_per_dialogue": node_mask.sum(dim=1),
        }
        return {
            "logits": logits,
            "low_frequency_repr": low_frequency_repr,
            "high_frequency_repr": high_frequency_repr,
            "low_frequency_modal_repr": low_modal,
            "high_frequency_modal_repr": high_modal,
            "fused_repr": fused_repr,
            "adjacency": adjacency,
            "normalized_adjacency": normalized_adjacency,
            "causal_diagnostics": diagnostics,
        }

    def _fuse(self, low: torch.Tensor, high: torch.Tensor) -> torch.Tensor:
        if self.config.fusion_type == "concat":
            assert self.fusion_projection is not None
            fused = self.fusion_projection(torch.cat([low, high], dim=-1))
        elif self.config.fusion_type == "gate":
            assert self.fusion_gate is not None
            gate = torch.sigmoid(self.fusion_gate(torch.cat([low, high], dim=-1)))
            fused = gate * low + (1.0 - gate) * high
        else:
            fused = low + high
        return self.fusion_norm(self.fusion_dropout(fused))

    def _validate_inputs(
        self,
        text_features: torch.Tensor,
        audio_features: torch.Tensor,
        visual_features: torch.Tensor,
        attention_mask: torch.Tensor,
        lengths: torch.Tensor,
        speaker_ids_int: torch.Tensor,
    ) -> None:
        features = (text_features, audio_features, visual_features)
        expected_dims = (
            self.config.text_dim,
            self.config.audio_dim,
            self.config.visual_dim,
        )
        if any(tensor.dim() != 3 for tensor in features):
            raise ValueError("all modality features must have shape [B,T,D]")
        batch_time = text_features.shape[:2]
        for tensor, expected_dim in zip(features, expected_dims):
            if tensor.shape[:2] != batch_time or tensor.shape[-1] != expected_dim:
                raise ValueError("modality feature shapes do not match the configuration")
        if attention_mask.shape != batch_time or speaker_ids_int.shape != batch_time:
            raise ValueError("attention_mask and speaker_ids_int must have shape [B,T]")
        if lengths.shape != (batch_time[0],):
            raise ValueError("lengths must have shape [B]")
        devices = {tensor.device for tensor in features}
        devices.update({attention_mask.device, lengths.device, speaker_ids_int.device})
        if len(devices) != 1:
            raise ValueError("all input tensors must share a device")
        if torch.any(lengths <= 0):
            raise ValueError("every dialogue must contain at least one utterance")
        expected_mask = torch.arange(batch_time[1], device=lengths.device).unsqueeze(0) < lengths.unsqueeze(1)
        if not torch.equal(attention_mask.to(dtype=torch.bool), expected_mask):
            raise ValueError("attention_mask must be a contiguous prefix matching lengths")
        valid_speakers = speaker_ids_int[expected_mask]
        if torch.any(valid_speakers < 0):
            raise ValueError("valid utterances must have non-negative speaker ids")

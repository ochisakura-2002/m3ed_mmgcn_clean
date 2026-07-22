"""DialogueGCN-derived project implementation with an explicit causal contract."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .causal_dialoguegcn_graph import (
    CausalEdgeAttention,
    build_causal_dialoguegcn_graph,
)
from .relational_graph_conv import RelationalGraphConv


_CONTEXT_ENCODERS = ("linear", "causal_gru")
_FUSIONS = ("concat", "gate")


@dataclass(frozen=True)
class CausalDialogueGCNConfig:
    """Validated configuration for the project causal DialogueGCN baseline."""

    text_dim: int
    audio_dim: int
    visual_dim: int
    hidden_dim: int
    context_hidden_dim: int
    graph_hidden_dim: int
    num_classes: int
    dropout: float = 0.1
    window_past: Optional[int] = None
    context_encoder_type: str = "causal_gru"
    num_graph_layers: int = 1
    num_speakers: int = 2
    context_mode: str = "causal"
    fusion_type: str = "concat"
    window_future: int = 0
    bidirectional: bool = False
    nodal_attention: str = "none"

    def __post_init__(self) -> None:
        for name in (
            "text_dim",
            "audio_dim",
            "visual_dim",
            "hidden_dim",
            "context_hidden_dim",
            "graph_hidden_dim",
            "num_classes",
            "num_speakers",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0,1)")
        if self.window_past is not None and self.window_past < 0:
            raise ValueError("window_past must be None or non-negative")
        if self.context_encoder_type not in _CONTEXT_ENCODERS:
            raise ValueError(f"context_encoder_type must be one of {_CONTEXT_ENCODERS}")
        if self.fusion_type not in _FUSIONS:
            raise ValueError(f"fusion_type must be one of {_FUSIONS}")
        if self.num_graph_layers <= 0:
            raise ValueError("num_graph_layers must be positive")
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


class CausalDialogueGCNBaseline(nn.Module):
    """Pure-PyTorch, multimodal, DialogueGCN-derived causal candidate."""

    def __init__(self, config: CausalDialogueGCNConfig) -> None:
        super().__init__()
        if not isinstance(config, CausalDialogueGCNConfig):
            raise TypeError("config must be a CausalDialogueGCNConfig")
        self.config = config
        self.modality_projections = nn.ModuleDict(
            {
                "text": nn.Linear(config.text_dim, config.hidden_dim),
                "audio": nn.Linear(config.audio_dim, config.hidden_dim),
                "visual": nn.Linear(config.visual_dim, config.hidden_dim),
            }
        )
        if config.fusion_type == "concat":
            self.fusion_projection: Optional[nn.Module] = nn.Linear(
                3 * config.hidden_dim,
                config.hidden_dim,
            )
            self.fusion_gate: Optional[nn.Module] = None
        else:
            self.fusion_projection = None
            self.fusion_gate = nn.Linear(3 * config.hidden_dim, 3)
        self.fusion_norm = nn.LayerNorm(config.hidden_dim)
        self.fusion_dropout = nn.Dropout(config.dropout)

        if config.context_encoder_type == "causal_gru":
            self.context_gru: Optional[nn.GRU] = nn.GRU(
                config.hidden_dim,
                config.context_hidden_dim,
                batch_first=True,
                bidirectional=False,
            )
            self.context_linear: Optional[nn.Module] = None
        else:
            self.context_gru = None
            self.context_linear = nn.Linear(config.hidden_dim, config.context_hidden_dim)
        self.context_norm = nn.LayerNorm(config.context_hidden_dim)
        self.context_dropout = nn.Dropout(config.dropout)

        self.edge_attention = CausalEdgeAttention(config.context_hidden_dim)
        graph_layers = []
        input_dim = config.context_hidden_dim
        for _ in range(config.num_graph_layers):
            graph_layers.append(
                RelationalGraphConv(
                    input_dim,
                    config.graph_hidden_dim,
                    config.num_speakers ** 2,
                    config.dropout,
                )
            )
            input_dim = config.graph_hidden_dim
        self.graph_layers = nn.ModuleList(graph_layers)
        self.classifier = nn.Linear(config.graph_hidden_dim, config.num_classes)

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
        """Return ``[B,T,C]`` logits or graph/context auxiliary tensors."""

        self._validate_inputs(
            text_features,
            audio_features,
            visual_features,
            attention_mask,
            lengths,
            speaker_ids_int,
        )
        mask = attention_mask.to(dtype=torch.bool)
        mask_scale = mask.unsqueeze(-1).to(dtype=text_features.dtype)
        modal = []
        for name, features in (
            ("text", text_features),
            ("audio", audio_features),
            ("visual", visual_features),
        ):
            projected_steps = [
                F.gelu(self.modality_projections[name](features[:, time_index]))
                for time_index in range(features.shape[1])
            ]
            modal.append(torch.stack(projected_steps, dim=1) * mask_scale)
        fused = self._fuse_modalities(modal) * mask_scale
        context_repr = self._encode_context(fused, mask, lengths) * mask_scale

        adjacency, relation_ids, relation_mapping = build_causal_dialoguegcn_graph(
            mask,
            lengths,
            speaker_ids_int,
            self.config.num_speakers,
            self.config.window_past,
        )
        edge_attention = self.edge_attention(context_repr, adjacency)
        graph_repr = context_repr
        for layer in self.graph_layers:
            graph_repr = layer(
                graph_repr,
                adjacency,
                relation_ids,
                edge_attention,
                mask,
            )
        logits = self.classifier(graph_repr)
        logits = logits.masked_fill(~mask.unsqueeze(-1), 0.0)
        if not return_aux:
            return logits
        return {
            "logits": logits,
            "adjacency": adjacency,
            "edge_type": relation_ids,
            "relation_ids": relation_ids,
            "relation_mapping": relation_mapping,
            "edge_attention": edge_attention,
            "context_repr": context_repr,
            "graph_repr": graph_repr,
        }

    def _fuse_modalities(self, modal: list[torch.Tensor]) -> torch.Tensor:
        fused_steps = []
        for time_index in range(modal[0].shape[1]):
            current = [value[:, time_index] for value in modal]
            concatenated = torch.cat(current, dim=-1)
            if self.config.fusion_type == "concat":
                assert self.fusion_projection is not None
                fused = self.fusion_projection(concatenated)
            else:
                assert self.fusion_gate is not None
                gates = torch.softmax(self.fusion_gate(concatenated), dim=-1)
                stacked = torch.stack(current, dim=1)
                fused = (stacked * gates.unsqueeze(-1)).sum(dim=1)
            fused_steps.append(
                self.fusion_dropout(F.gelu(self.fusion_norm(fused)))
            )
        return torch.stack(fused_steps, dim=1)

    def _encode_context(
        self,
        fused: torch.Tensor,
        attention_mask: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        if self.context_gru is not None:
            batch_size = fused.shape[0]
            hidden = fused.new_zeros(
                self.context_gru.num_layers,
                batch_size,
                self.context_gru.hidden_size,
            )
            outputs = []
            for time_index in range(fused.shape[1]):
                step_output, candidate_hidden = self.context_gru(
                    fused[:, time_index : time_index + 1],
                    hidden,
                )
                valid = attention_mask[:, time_index].view(1, batch_size, 1)
                hidden = torch.where(valid, candidate_hidden, hidden)
                outputs.append(
                    step_output[:, 0]
                    * valid[0].to(dtype=step_output.dtype)
                )
            encoded = torch.stack(outputs, dim=1)
        else:
            assert self.context_linear is not None
            encoded = F.gelu(self.context_linear(fused))
        encoded = self.context_dropout(self.context_norm(encoded))
        return encoded * attention_mask.unsqueeze(-1).to(dtype=encoded.dtype)

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
        if torch.any(valid_speakers < 0) or torch.any(valid_speakers >= self.config.num_speakers):
            raise ValueError("valid utterances contain an out-of-range speaker id")

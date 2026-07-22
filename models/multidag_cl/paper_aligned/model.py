"""Project reproduction of MultiDAG and MultiDAG+CL."""

from __future__ import annotations

from typing import Any, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.common.paper_aligned import (
    class_weight_tensor,
    masked_cross_entropy,
    masked_softmax,
    run_packed_rnn,
    structured_output,
    validate_dialogue_batch,
)
from .curriculum import dialogue_difficulty


def build_get_adj_v1(
    speaker_ids_int: torch.Tensor,
    attention_mask: torch.Tensor,
    window_past: int,
) -> torch.Tensor:
    """Released ``get_adj_v1`` predecessor selection.

    For each target, scan backwards and include every utterance until the
    ``window_past``-th same-speaker predecessor is encountered.  ``-1`` means
    all predecessors.  The diagonal and all future positions remain zero.
    """

    if speaker_ids_int.shape != attention_mask.shape:
        raise ValueError("speaker_ids_int and attention_mask must share [B,T].")
    if window_past == 0 or window_past < -1:
        raise ValueError("window_past must be -1 or a positive integer.")
    batch_size, max_len = speaker_ids_int.shape
    adjacency = torch.zeros(
        (batch_size, max_len, max_len),
        dtype=torch.bool,
        device=speaker_ids_int.device,
    )
    lengths = attention_mask.long().sum(dim=1).tolist()
    for batch_index, length in enumerate(lengths):
        for target in range(int(length)):
            same_count = 0
            target_speaker = int(speaker_ids_int[batch_index, target].item())
            for source in range(target - 1, -1, -1):
                adjacency[batch_index, target, source] = True
                if int(speaker_ids_int[batch_index, source].item()) == target_speaker:
                    same_count += 1
                    if window_past != -1 and same_count == window_past:
                        break
    return adjacency


class _DAGLayer(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.attention = nn.Linear(hidden_dim * 2, 1)
        self.same_speaker = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.different_speaker = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.node_gru = nn.GRUCell(hidden_dim, hidden_dim)
        self.context_gru = nn.GRUCell(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        previous: torch.Tensor,
        adjacency: torch.Tensor,
        same_speaker: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, max_len, hidden_dim = previous.shape
        outputs: list[torch.Tensor] = []
        attention_rows = previous.new_zeros((batch_size, max_len, max_len))
        for target in range(max_len):
            query = previous[:, target]
            if target == 0:
                message = torch.zeros_like(query)
            else:
                keys = torch.stack(outputs, dim=1)
                expanded_query = query.unsqueeze(1).expand(-1, target, -1)
                scores = self.attention(torch.cat([expanded_query, keys], dim=-1)).squeeze(-1)
                edge_mask = adjacency[:, target, :target]
                weights = masked_softmax(scores, edge_mask, dim=-1)
                relation = same_speaker[:, target, :target].unsqueeze(-1)
                values = (
                    relation * self.same_speaker(keys)
                    + (1.0 - relation) * self.different_speaker(keys)
                )
                message = torch.bmm(weights.unsqueeze(1), values).squeeze(1)
                attention_rows[:, target, :target] = weights
            node_state = self.node_gru(query, message)
            context_state = self.context_gru(message, query)
            current = self.dropout(node_state + context_state)
            current = current * attention_mask[:, target].unsqueeze(-1).to(current.dtype)
            outputs.append(current)
        return torch.stack(outputs, dim=1), attention_rows


class _NodalAttention(nn.Module):
    def __init__(self, feature_dim: int, mode: Optional[str]) -> None:
        super().__init__()
        if mode not in {None, "global", "past"}:
            raise ValueError("nodal_attention must be null, global, or past.")
        self.mode = mode
        self.transform = nn.Linear(feature_dim, feature_dim)

    def forward(self, features: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if self.mode is None:
            return features
        scores = torch.bmm(self.transform(features), features.transpose(1, 2))
        mask = attention_mask.bool().unsqueeze(1).expand_as(scores)
        if self.mode == "past":
            causal = torch.ones_like(scores, dtype=torch.bool).tril()
            mask = mask & causal
        weights = masked_softmax(torch.tanh(scores), mask, dim=-1)
        return torch.bmm(weights, features)


class OriginalReproMultiDAGCL(nn.Module):
    model_key = "original_repro_multidag_cl"
    causal_grade = "noncausal_offline_full_context"

    def __init__(
        self,
        text_feature_dim: int,
        audio_feature_dim: int,
        visual_feature_dim: int,
        num_classes: int = 6,
        hidden_dim: int = 300,
        graph_layers: int = 4,
        dropout: float = 0.4,
        window_past: int = 1,
        use_curriculum_learning: bool = True,
        curriculum_bucket_count: int = 5,
        nodal_attention: Optional[str] = None,
        class_weight: Optional[Sequence[float]] = None,
    ) -> None:
        super().__init__()
        if hidden_dim <= 0 or hidden_dim % 2:
            raise ValueError("hidden_dim must be a positive even integer.")
        if graph_layers <= 0:
            raise ValueError("graph_layers must be positive.")
        self.num_classes = int(num_classes)
        self.hidden_dim = int(hidden_dim)
        self.window_past = int(window_past)
        self.use_curriculum_learning = bool(use_curriculum_learning)
        self.curriculum_bucket_count = int(curriculum_bucket_count)

        self.text_projection = nn.Linear(int(text_feature_dim), self.hidden_dim)
        self.text_encoder = nn.LSTM(
            self.hidden_dim,
            self.hidden_dim // 2,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=float(dropout),
        )
        self.audio_encoder = nn.Linear(int(audio_feature_dim), self.hidden_dim)
        self.visual_encoder = nn.Linear(int(visual_feature_dim), self.hidden_dim)
        fused_dim = self.hidden_dim * 3
        self.initial_projection = nn.Linear(fused_dim, self.hidden_dim)
        self.graph_layers = nn.ModuleList(
            [_DAGLayer(self.hidden_dim, float(dropout)) for _ in range(int(graph_layers))]
        )
        classifier_dim = fused_dim + self.hidden_dim * (len(self.graph_layers) + 1)
        self.nodal_attention = _NodalAttention(classifier_dim, nodal_attention)
        self.classifier = nn.Sequential(
            nn.Linear(classifier_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.hidden_dim, self.num_classes),
        )
        weight = class_weight_tensor(class_weight, self.num_classes)
        self.register_buffer("class_weight", weight, persistent=weight is not None)

    def forward(
        self,
        text_features: torch.Tensor,
        audio_features: torch.Tensor,
        visual_features: torch.Tensor,
        attention_mask: torch.Tensor,
        lengths: torch.Tensor,
        speaker_ids_int: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        **_: Any,
    ) -> dict[str, Any]:
        validate_dialogue_batch(
            text_features,
            audio_features,
            visual_features,
            attention_mask,
            lengths,
            speaker_ids_int,
        )
        text = run_packed_rnn(
            self.text_encoder, self.text_projection(text_features), lengths
        )
        audio = self.audio_encoder(audio_features)
        visual = self.visual_encoder(visual_features)
        fused_input = torch.cat([audio, visual, text], dim=-1)
        initial = F.relu(self.initial_projection(fused_input))
        adjacency = build_get_adj_v1(
            speaker_ids_int, attention_mask, self.window_past
        )
        same_speaker = (
            speaker_ids_int.unsqueeze(2) == speaker_ids_int.unsqueeze(1)
        ).to(initial.dtype)
        states = [initial * attention_mask.unsqueeze(-1).to(initial.dtype)]
        attention_layers = []
        for layer in self.graph_layers:
            state, edge_attention = layer(
                states[-1], adjacency, same_speaker, attention_mask
            )
            states.append(state)
            attention_layers.append(edge_attention)
        fusion = torch.cat(states + [fused_input], dim=-1)
        fusion = self.nodal_attention(fusion, attention_mask)
        logits = self.classifier(fusion)
        logits = logits * attention_mask.unsqueeze(-1).to(logits.dtype)
        classification_loss = masked_cross_entropy(
            logits, labels, attention_mask, self.class_weight
        )
        difficulty = (
            None
            if labels is None
            else dialogue_difficulty(labels, speaker_ids_int, attention_mask)
        )
        return structured_output(
            logits=logits,
            classification_loss=classification_loss,
            features={
                "modality_encoder_output": {"text": text, "audio": audio, "visual": visual},
                "context_output": initial,
                "graph_output": states[-1],
                "fusion_output": fusion,
                "logits": logits,
                "base_classification_loss": classification_loss,
                "auxiliary_losses": {},
            },
            diagnostics={
                "model_key": self.model_key,
                "adjacency": adjacency,
                "same_speaker": same_speaker,
                "edge_attention": attention_layers,
                "dag_direction": "past_to_future",
                "use_curriculum_learning": self.use_curriculum_learning,
                "curriculum_bucket_count": self.curriculum_bucket_count,
                "dialogue_difficulty": difficulty,
            },
        )


__all__ = ["OriginalReproMultiDAGCL", "build_get_adj_v1"]

"""Project reproduction of the original bidirectional DialogueGCN."""

from __future__ import annotations

from typing import Any, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.baselines.original_repro.common import (
    active_feature_concat,
    class_weight_tensor,
    masked_cross_entropy,
    masked_softmax,
    run_packed_rnn,
    structured_output,
    validate_dialogue_batch,
)


OFFICIAL_IEMOCAP_CLASS_WEIGHT = (
    1 / 0.086747,
    1 / 0.144406,
    1 / 0.227883,
    1 / 0.160585,
    1 / 0.127711,
    1 / 0.252668,
)


def dialoguegcn_relation_id(
    target_speaker: int,
    source_speaker: int,
    target_index: int,
    source_index: int,
    num_speakers: int,
) -> int:
    """Encode target/source speaker pair and temporal direction."""

    direction = 0 if target_index < source_index else 1
    return ((int(target_speaker) * int(num_speakers) + int(source_speaker)) * 2) + direction


def build_dialoguegcn_graph(
    attention_mask: torch.Tensor,
    speaker_ids_int: torch.Tensor,
    num_speakers: int,
    window_past: int,
    window_future: int,
) -> tuple[torch.Tensor, torch.Tensor, dict[int, str]]:
    """Build directed past/future edges and all ``2*M^2`` relation IDs."""

    if attention_mask.shape != speaker_ids_int.shape:
        raise ValueError("attention_mask and speaker_ids_int must share [B,T].")
    if window_past < -1 or window_future < -1:
        raise ValueError("DialogueGCN windows must be -1 or non-negative.")
    batch_size, max_len = attention_mask.shape
    adjacency = torch.zeros(
        (batch_size, max_len, max_len),
        dtype=torch.bool,
        device=attention_mask.device,
    )
    relation_ids = torch.full(
        (batch_size, max_len, max_len),
        -1,
        dtype=torch.long,
        device=attention_mask.device,
    )
    for batch_index, length_value in enumerate(attention_mask.long().sum(dim=1).tolist()):
        length = int(length_value)
        for target in range(length):
            left = 0 if window_past == -1 else max(0, target - window_past)
            right = length if window_future == -1 else min(length, target + window_future + 1)
            for source in range(left, right):
                adjacency[batch_index, target, source] = True
                relation_ids[batch_index, target, source] = dialoguegcn_relation_id(
                    int(speaker_ids_int[batch_index, target].item()),
                    int(speaker_ids_int[batch_index, source].item()),
                    target,
                    source,
                    num_speakers,
                )
    mapping = {
        ((target * num_speakers + source) * 2) + direction: (
            f"target_speaker={target}|source_speaker={source}|"
            f"{'future_source' if direction == 0 else 'past_or_self_source'}"
        )
        for target in range(num_speakers)
        for source in range(num_speakers)
        for direction in range(2)
    }
    return adjacency, relation_ids, mapping


class _EdgeAttention(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.transform = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, context: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        # Paper Eq. (1): alpha_ij = softmax(g_i^T W_e g_j).
        scores = torch.bmm(context, self.transform(context).transpose(1, 2))
        return masked_softmax(scores, adjacency, dim=-1)


class DialogueGCNRelationalGraphNetwork(nn.Module):
    """Dense, dependency-free implementation of DialogueGCN Eqs. (2)-(3)."""

    def __init__(
        self,
        input_dim: int,
        graph_hidden_dim: int,
        num_relations: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.num_relations = int(num_relations)
        self.relation_weights = nn.Parameter(
            torch.empty(self.num_relations, input_dim, graph_hidden_dim)
        )
        nn.init.xavier_uniform_(self.relation_weights)
        self.root = nn.Linear(input_dim, graph_hidden_dim, bias=False)
        self.second_neighbor = nn.Linear(
            graph_hidden_dim, graph_hidden_dim, bias=False
        )
        self.second_root = nn.Linear(graph_hidden_dim, graph_hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        features: torch.Tensor,
        adjacency: torch.Tensor,
        relation_ids: torch.Tensor,
        edge_attention: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        transformed = torch.einsum("bsi,rih->brsh", features, self.relation_weights)
        relations = torch.arange(self.num_relations, device=features.device)
        identity = torch.eye(
            adjacency.shape[-1], dtype=torch.bool, device=features.device
        ).unsqueeze(0)
        neighbor_mask = adjacency.bool() & ~identity
        relation_mask = (
            relation_ids.unsqueeze(1) == relations.view(1, -1, 1, 1)
        ) & neighbor_mask.unsqueeze(1)
        relation_count = relation_mask.sum(dim=-1, keepdim=True).clamp_min(1)
        weights = (
            edge_attention.unsqueeze(1)
            * relation_mask.to(edge_attention.dtype)
            / relation_count.to(edge_attention.dtype)
        )
        first = torch.einsum("brts,brsh->bth", weights, transformed)
        root_attention = edge_attention.diagonal(dim1=-2, dim2=-1).unsqueeze(-1)
        first = F.relu(first + root_attention * self.root(features))

        transformed_neighbors = self.second_neighbor(first)
        second = torch.bmm(neighbor_mask.to(first.dtype), transformed_neighbors)
        second = F.relu(second + self.second_root(first))
        second = self.dropout(second)
        return second * attention_mask.unsqueeze(-1).to(second.dtype)


class _NodalAttention(nn.Module):
    def __init__(self, feature_dim: int) -> None:
        super().__init__()
        self.transform = nn.Linear(feature_dim, feature_dim, bias=False)

    def forward(self, features: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        scores = torch.bmm(self.transform(features), features.transpose(1, 2))
        source_mask = attention_mask.bool().unsqueeze(1).expand_as(scores)
        weights = masked_softmax(scores, source_mask, dim=-1)
        return torch.bmm(weights, features)


class OriginalReproDialogueGCN(nn.Module):
    model_key = "original_repro_dialoguegcn"
    causal_grade = "noncausal_offline_full_context"

    def __init__(
        self,
        text_feature_dim: int,
        audio_feature_dim: int,
        visual_feature_dim: int,
        num_classes: int = 6,
        context_hidden_dim: int = 100,
        graph_hidden_dim: int = 100,
        num_speakers: int = 2,
        dropout: float = 0.4,
        use_nodal_attention: bool = True,
        use_class_weight: bool = True,
        base_model: str = "LSTM",
        window_past: int = 10,
        window_future: int = 10,
        active_modalities: Sequence[str] = ("text",),
        class_weight: Optional[Sequence[float]] = None,
    ) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.num_speakers = int(num_speakers)
        self.window_past = int(window_past)
        self.window_future = int(window_future)
        self.use_nodal_attention = bool(use_nodal_attention)
        self.use_class_weight = bool(use_class_weight)
        self.base_model = str(base_model).upper()
        self.active_modalities = tuple(str(value).lower() for value in active_modalities)
        dimensions = {
            "text": int(text_feature_dim),
            "audio": int(audio_feature_dim),
            "visual": int(visual_feature_dim),
        }
        input_dim = sum(dimensions[value] for value in self.active_modalities)
        if self.base_model == "GRU":
            self.context_encoder: nn.Module = nn.GRU(
                input_dim,
                int(context_hidden_dim),
                num_layers=2,
                batch_first=True,
                bidirectional=True,
                dropout=float(dropout),
            )
            context_dim = int(context_hidden_dim) * 2
        elif self.base_model == "LSTM":
            self.context_encoder = nn.LSTM(
                input_dim,
                int(context_hidden_dim),
                num_layers=2,
                batch_first=True,
                bidirectional=True,
                dropout=float(dropout),
            )
            context_dim = int(context_hidden_dim) * 2
        elif self.base_model == "NONE":
            self.context_encoder = nn.Linear(input_dim, int(context_hidden_dim) * 2)
            context_dim = int(context_hidden_dim) * 2
        else:
            raise ValueError("base_model must be GRU, LSTM, or None.")

        self.edge_attention = _EdgeAttention(context_dim)
        self.graph_network = DialogueGCNRelationalGraphNetwork(
            context_dim,
            int(graph_hidden_dim),
            num_relations=2 * self.num_speakers**2,
            dropout=float(dropout),
        )
        node_dim = context_dim + int(graph_hidden_dim)
        self.nodal_attention = _NodalAttention(node_dim)
        self.classifier = nn.Sequential(
            nn.Linear(node_dim, int(context_hidden_dim)),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(context_hidden_dim), self.num_classes),
        )
        configured_weight = class_weight
        if configured_weight is None and self.use_class_weight and self.num_classes == 6:
            configured_weight = OFFICIAL_IEMOCAP_CLASS_WEIGHT
        weight = class_weight_tensor(configured_weight, self.num_classes)
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
        model_input = active_feature_concat(
            text_features,
            audio_features,
            visual_features,
            self.active_modalities,
        )
        if self.base_model == "NONE":
            context = self.context_encoder(model_input)
        else:
            context = run_packed_rnn(self.context_encoder, model_input, lengths)
        context = context * attention_mask.unsqueeze(-1).to(context.dtype)
        adjacency, relation_ids, relation_mapping = build_dialoguegcn_graph(
            attention_mask,
            speaker_ids_int,
            self.num_speakers,
            self.window_past,
            self.window_future,
        )
        edge_attention = self.edge_attention(context, adjacency)
        graph = self.graph_network(
            context,
            adjacency,
            relation_ids,
            edge_attention,
            attention_mask,
        )
        residual_nodes = torch.cat([context, graph], dim=-1)
        fusion = (
            self.nodal_attention(residual_nodes, attention_mask)
            if self.use_nodal_attention
            else residual_nodes
        )
        logits = self.classifier(fusion)
        logits = logits * attention_mask.unsqueeze(-1).to(logits.dtype)
        weight = self.class_weight if self.use_class_weight else None
        classification_loss = masked_cross_entropy(logits, labels, attention_mask, weight)
        return structured_output(
            logits=logits,
            classification_loss=classification_loss,
            features={
                "modality_encoder_output": model_input,
                "context_output": context,
                "graph_output": graph,
                "fusion_output": fusion,
                "logits": logits,
                "base_classification_loss": classification_loss,
                "auxiliary_losses": {},
            },
            diagnostics={
                "model_key": self.model_key,
                "adjacency": adjacency,
                "relation_ids": relation_ids,
                "relation_mapping": relation_mapping,
                "edge_attention": edge_attention,
                "base_model": self.base_model,
                "window_past": self.window_past,
                "window_future": self.window_future,
                "use_nodal_attention": self.use_nodal_attention,
                "use_class_weight": self.use_class_weight,
            },
        )


__all__ = [
    "DialogueGCNRelationalGraphNetwork",
    "OFFICIAL_IEMOCAP_CLASS_WEIGHT",
    "OriginalReproDialogueGCN",
    "build_dialoguegcn_graph",
    "dialoguegcn_relation_id",
]

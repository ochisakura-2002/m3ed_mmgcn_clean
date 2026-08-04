"""Sequential MultiDAG layer and the frozen swapped dual-GRU update."""

from __future__ import annotations

import torch
import torch.nn as nn

from .attention import RelationAwareAttention
from .contracts import DAGLayerResult, DualGRUResult, SpeakerRelations


class DualGRUNodeUpdate(nn.Module):
    """Apply two independent GRUCells with the normative PyTorch argument roles."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        if isinstance(hidden_dim, bool) or not isinstance(hidden_dim, int):
            raise TypeError("hidden_dim must be int")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        self.hidden_dim = hidden_dim
        self.node_gru = nn.GRUCell(hidden_dim, hidden_dim)
        self.context_gru = nn.GRUCell(hidden_dim, hidden_dim)

    def forward(
        self,
        previous_layer_state: torch.Tensor,
        message: torch.Tensor,
    ) -> DualGRUResult:
        if previous_layer_state.dim() != 2 or previous_layer_state.shape[1] != self.hidden_dim:
            raise ValueError("previous_layer_state must have shape [B,H]")
        if message.shape != previous_layer_state.shape:
            raise ValueError("message must share [B,H] with previous_layer_state")
        if previous_layer_state.device != message.device or previous_layer_state.dtype != message.dtype:
            raise ValueError("previous_layer_state and message must share dtype/device")
        node_state = self.node_gru(previous_layer_state, message)
        context_state = self.context_gru(message, previous_layer_state)
        if node_state.shape != previous_layer_state.shape or context_state.shape != previous_layer_state.shape:
            raise ValueError("GRU cells must return [B,H]")
        state = node_state + context_state
        return DualGRUResult(
            node_state=node_state,
            context_state=context_state,
            state=state,
        )


class MultiDAGLayer(nn.Module):
    """Compute current-layer nodes left-to-right from completed predecessors."""

    def __init__(self, hidden_dim: int, *, collect_diagnostics: bool = False) -> None:
        super().__init__()
        if not isinstance(collect_diagnostics, bool):
            raise TypeError("collect_diagnostics must be bool")
        self.hidden_dim = hidden_dim
        self.collect_diagnostics = collect_diagnostics
        self.attention = RelationAwareAttention(hidden_dim)
        self.node_update = DualGRUNodeUpdate(hidden_dim)

    def forward(
        self,
        previous_layer: torch.Tensor,
        adjacency: torch.Tensor,
        relations: SpeakerRelations,
        attention_mask: torch.Tensor,
    ) -> DAGLayerResult:
        if previous_layer.dim() != 3 or previous_layer.shape[2] != self.hidden_dim:
            raise ValueError("previous_layer must have shape [B,T,H]")
        batch_size, max_time, _ = previous_layer.shape
        if adjacency.dtype is not torch.bool or adjacency.shape != (batch_size, max_time, max_time):
            raise TypeError("adjacency must be bool [B,T,T]")
        if not isinstance(relations, SpeakerRelations):
            raise TypeError("relations must be SpeakerRelations")
        for name, value in (
            ("same_speaker", relations.same_speaker),
            ("different_speaker", relations.different_speaker),
        ):
            if value.dtype is not torch.bool or value.shape != adjacency.shape:
                raise TypeError(f"{name} must be bool [B,T,T]")
        if attention_mask.dtype not in (torch.int64, torch.bool) or attention_mask.shape != (batch_size, max_time):
            raise TypeError("attention_mask must be int64 or bool [B,T]")
        devices = {
            previous_layer.device,
            adjacency.device,
            relations.same_speaker.device,
            relations.different_speaker.device,
            attention_mask.device,
        }
        if len(devices) != 1:
            raise ValueError("DAG layer inputs must share a device")
        mask = attention_mask.bool()
        if torch.any(torch.triu(adjacency, diagonal=0)):
            raise ValueError("adjacency may not contain self or future-source edges")
        valid_pairs = mask.unsqueeze(2) & mask.unsqueeze(1)
        if torch.any(adjacency & ~valid_pairs):
            raise ValueError("adjacency may not contain padded rows or columns")
        if torch.any(relations.same_speaker & relations.different_speaker):
            raise ValueError("speaker relations must be mutually exclusive")
        if torch.any((relations.same_speaker | relations.different_speaker) & ~valid_pairs):
            raise ValueError("invalid speaker-relation pairs must be false")

        outputs: list[torch.Tensor] = []
        diagnostic_logits: list[torch.Tensor] = []
        diagnostic_weights: list[torch.Tensor] = []
        diagnostic_messages: list[torch.Tensor] = []

        for target in range(max_time):
            valid_target = mask[:, target]
            valid_indices = torch.nonzero(valid_target, as_tuple=False).squeeze(-1)
            current = previous_layer.new_zeros((batch_size, self.hidden_dim))
            row_logits = previous_layer.new_zeros((batch_size, max_time))
            row_weights = previous_layer.new_zeros((batch_size, max_time))
            row_messages = previous_layer.new_zeros((batch_size, self.hidden_dim))

            if valid_indices.numel() > 0:
                query = previous_layer.index_select(0, valid_indices)[:, target, :]
                if target == 0:
                    predecessor_states = previous_layer.new_zeros(
                        (valid_indices.numel(), 0, self.hidden_dim)
                    )
                else:
                    completed = torch.stack(outputs, dim=1)
                    predecessor_states = completed.index_select(0, valid_indices)
                predecessor_mask = adjacency.index_select(0, valid_indices)[:, target, :target]
                same_mask = relations.same_speaker.index_select(0, valid_indices)[:, target, :target]
                attention_result = self.attention(
                    query,
                    predecessor_states,
                    predecessor_mask,
                    same_mask,
                )
                update_result = self.node_update(query, attention_result.message)
                current = current.index_copy(0, valid_indices, update_result.state)
                if self.collect_diagnostics:
                    row_messages = row_messages.index_copy(
                        0, valid_indices, attention_result.message
                    )
                    if target > 0:
                        row_logits[valid_indices, :target] = attention_result.logits
                        row_weights[valid_indices, :target] = attention_result.weights
            outputs.append(current)
            if self.collect_diagnostics:
                diagnostic_logits.append(row_logits)
                diagnostic_weights.append(row_weights)
                diagnostic_messages.append(row_messages)

        state = torch.stack(outputs, dim=1)
        state = state * mask.unsqueeze(-1).to(state.dtype)
        if self.collect_diagnostics:
            attention_logits = torch.stack(diagnostic_logits, dim=1)
            attention_weights = torch.stack(diagnostic_weights, dim=1)
            messages = torch.stack(diagnostic_messages, dim=1)
        else:
            attention_logits = None
            attention_weights = None
            messages = None
        return DAGLayerResult(
            state=state,
            attention_logits=attention_logits,
            attention_weights=attention_weights,
            messages=messages,
        )


__all__ = ["DualGRUNodeUpdate", "MultiDAGLayer"]

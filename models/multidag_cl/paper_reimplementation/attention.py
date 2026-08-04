"""Concat-linear, relation-on-values attention from the frozen equations."""

from __future__ import annotations

import torch
import torch.nn as nn

from .contracts import AttentionResult


class RelationAwareAttention(nn.Module):
    """Attend only to explicit predecessors and transform values by relation."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        if isinstance(hidden_dim, bool) or not isinstance(hidden_dim, int):
            raise TypeError("hidden_dim must be int")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        self.hidden_dim = hidden_dim
        self.score_linear = nn.Linear(2 * hidden_dim, 1, bias=True)
        self.W_same = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_different = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(
        self,
        query: torch.Tensor,
        predecessor_states: torch.Tensor,
        predecessor_mask: torch.Tensor,
        same_mask: torch.Tensor,
    ) -> AttentionResult:
        if query.dim() != 2 or query.shape[1] != self.hidden_dim:
            raise ValueError("query must have shape [B,H]")
        if predecessor_states.dim() != 3 or predecessor_states.shape[0] != query.shape[0]:
            raise ValueError("predecessor_states must have shape [B,P,H]")
        if predecessor_states.shape[2] != self.hidden_dim:
            raise ValueError("predecessor_states hidden dimension mismatch")
        batch_size, predecessor_count, _ = predecessor_states.shape
        expected_mask_shape = (batch_size, predecessor_count)
        for name, value in (("predecessor_mask", predecessor_mask), ("same_mask", same_mask)):
            if value.dtype not in (torch.int64, torch.bool) or value.shape != expected_mask_shape:
                raise TypeError(f"{name} must be int64 or bool [B,P]")
            if value.dtype is torch.int64 and not torch.all((value == 0) | (value == 1)):
                raise ValueError(f"{name} may contain only 0/1")
        devices = {
            query.device,
            predecessor_states.device,
            predecessor_mask.device,
            same_mask.device,
        }
        if len(devices) != 1:
            raise ValueError("attention inputs must share a device")
        if query.dtype != predecessor_states.dtype or not query.dtype.is_floating_point:
            raise TypeError("query and predecessor_states must share a floating dtype")
        if not torch.isfinite(query).all() or not torch.isfinite(predecessor_states).all():
            raise ValueError("attention states must be finite")

        edge_mask = predecessor_mask.bool()
        relation_same = same_mask.bool()
        if torch.any(relation_same & ~edge_mask):
            relation_same = relation_same & edge_mask

        if predecessor_count == 0:
            return AttentionResult(
                logits=query.new_zeros((batch_size, 0)),
                weights=query.new_zeros((batch_size, 0)),
                message=query.new_zeros((batch_size, self.hidden_dim)),
            )

        expanded_query = query.unsqueeze(1).expand(-1, predecessor_count, -1)
        raw_logits = self.score_linear(
            torch.cat((expanded_query, predecessor_states), dim=-1)
        ).squeeze(-1)
        weights = raw_logits.new_zeros(raw_logits.shape)
        nonempty = edge_mask.any(dim=-1)
        if bool(nonempty.any().item()):
            active_logits = raw_logits[nonempty]
            active_mask = edge_mask[nonempty]
            dtype_min = torch.finfo(active_logits.dtype).min
            masked_logits = active_logits.masked_fill(~active_mask, dtype_min)
            active_weights = torch.softmax(masked_logits, dim=-1)
            active_weights = active_weights * active_mask.to(active_weights.dtype)
            normalizer = active_weights.sum(dim=-1, keepdim=True)
            active_weights = active_weights / normalizer.clamp_min(
                torch.finfo(active_weights.dtype).tiny
            )
            weights[nonempty] = active_weights

        same_values = self.W_same(predecessor_states)
        different_values = self.W_different(predecessor_states)
        transformed_values = torch.where(
            relation_same.unsqueeze(-1),
            same_values,
            different_values,
        )
        message = torch.bmm(weights.unsqueeze(1), transformed_values).squeeze(1)
        masked_logits = torch.where(edge_mask, raw_logits, torch.zeros_like(raw_logits))
        if not torch.isfinite(weights).all() or not torch.isfinite(message).all():
            raise RuntimeError("relation-aware attention produced non-finite values")
        return AttentionResult(
            logits=masked_logits,
            weights=weights,
            message=message,
        )


__all__ = ["RelationAwareAttention"]

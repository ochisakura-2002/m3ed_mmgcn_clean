"""Pure-PyTorch directed relational graph convolution."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class RelationalGraphConv(nn.Module):
    """Aggregate causal neighbors with one learned transform per relation."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_relations: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if input_dim <= 0 or output_dim <= 0 or num_relations <= 0:
            raise ValueError("graph dimensions and num_relations must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0,1)")
        self.num_relations = int(num_relations)
        self.relation_weights = nn.Parameter(
            torch.empty(num_relations, input_dim, output_dim)
        )
        nn.init.xavier_uniform_(self.relation_weights)
        self.self_projection = nn.Linear(input_dim, output_dim)
        self.residual_projection: nn.Module
        if input_dim == output_dim:
            self.residual_projection = nn.Identity()
        else:
            self.residual_projection = nn.Linear(input_dim, output_dim, bias=False)
        self.output_norm = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        features: torch.Tensor,
        adjacency: torch.Tensor,
        relation_ids: torch.Tensor,
        edge_attention: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Apply a relation-specific causal message-passing layer."""

        if features.dim() != 3:
            raise ValueError("features must have shape [B,T,H]")
        expected_graph_shape = (features.shape[0], features.shape[1], features.shape[1])
        if adjacency.shape != expected_graph_shape:
            raise ValueError("adjacency must have shape [B,T,T]")
        if relation_ids.shape != expected_graph_shape or edge_attention.shape != expected_graph_shape:
            raise ValueError("relation_ids and edge_attention must match adjacency")
        if attention_mask.shape != features.shape[:2]:
            raise ValueError("attention_mask must have shape [B,T]")
        active_relations = relation_ids[adjacency.to(dtype=torch.bool)]
        if torch.any(active_relations < 0) or torch.any(active_relations >= self.num_relations):
            raise ValueError("active edge has an invalid relation id")

        transformed = torch.einsum("bsi,rih->brsh", features, self.relation_weights)
        relation_range = torch.arange(self.num_relations, device=features.device)
        relation_mask = relation_ids.unsqueeze(1) == relation_range.view(1, -1, 1, 1)
        weights = edge_attention.unsqueeze(1) * relation_mask.to(dtype=edge_attention.dtype)
        aggregated = torch.einsum("brts,brsh->bth", weights, transformed)
        updated = aggregated + self.self_projection(features)
        updated = F.gelu(updated)
        updated = self.dropout(updated) + self.residual_projection(features)
        updated = self.output_norm(updated)
        return updated * attention_mask.unsqueeze(-1).to(dtype=updated.dtype)

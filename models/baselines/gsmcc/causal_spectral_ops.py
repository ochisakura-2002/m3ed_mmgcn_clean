"""Directed polynomial high/low filters that preserve causal edge direction."""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def causal_directed_polynomial_filter(
    features: torch.Tensor,
    normalized_adjacency: torch.Tensor,
    num_steps: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return causal low/high signals from repeated left multiplication by ``S``."""

    if features.dim() != 3:
        raise ValueError("features must have shape [B,N,H]")
    if normalized_adjacency.shape != (
        features.shape[0],
        features.shape[1],
        features.shape[1],
    ):
        raise ValueError("normalized_adjacency must have shape [B,N,N]")
    if num_steps <= 0:
        raise ValueError("num_steps must be positive")

    previous = features
    high_components = []
    for _ in range(num_steps):
        current = torch.bmm(normalized_adjacency, previous)
        high_components.append(previous - current)
        previous = current
    low_frequency = previous
    high_frequency = torch.stack(high_components, dim=0).mean(dim=0)
    return low_frequency, high_frequency


class CausalDirectedPolynomialLayer(nn.Module):
    """Learned wrapper around a strictly directed polynomial graph filter."""

    def __init__(self, hidden_dim: int, num_steps: int, dropout: float) -> None:
        super().__init__()
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if num_steps <= 0:
            raise ValueError("num_steps must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0,1)")
        self.num_steps = int(num_steps)
        self.low_projection = nn.Linear(hidden_dim, hidden_dim)
        self.high_projection = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.output_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        features: torch.Tensor,
        normalized_adjacency: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Filter, project, and residually update node representations."""

        low_raw, high_raw = causal_directed_polynomial_filter(
            features,
            normalized_adjacency,
            self.num_steps,
        )
        low_projected = F.gelu(self.low_projection(low_raw))
        high_projected = F.gelu(self.high_projection(high_raw))
        low = self.output_norm(features + self.dropout(low_projected))
        high = self.output_norm(features + self.dropout(high_projected))
        updated = 0.5 * (low + high)
        return updated, low, high

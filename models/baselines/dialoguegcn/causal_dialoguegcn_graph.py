"""Graph construction and mask-first edge attention for causal DialogueGCN."""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from models.baselines.causal_graph_common import (
    build_causal_utterance_adjacency,
    build_speaker_pair_relation_ids,
    masked_softmax,
)


def build_causal_dialoguegcn_graph(
    attention_mask: torch.Tensor,
    lengths: torch.Tensor,
    speaker_ids_int: torch.Tensor,
    num_speakers: int,
    window_past: Optional[int],
) -> Tuple[torch.Tensor, torch.Tensor, Dict[int, str]]:
    """Return causal adjacency and source/target speaker-pair relation ids."""

    adjacency = build_causal_utterance_adjacency(
        attention_mask,
        lengths,
        window_past=window_past,
        self_loop=True,
    )
    relation_ids, relation_mapping = build_speaker_pair_relation_ids(
        speaker_ids_int,
        adjacency,
        num_speakers,
    )
    return adjacency, relation_ids, relation_mapping


class CausalEdgeAttention(nn.Module):
    """Normalize target/source scores over legal historical sources only."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        self.hidden_dim = int(hidden_dim)
        self.query = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.key = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(
        self,
        context_repr: torch.Tensor,
        adjacency: torch.Tensor,
    ) -> torch.Tensor:
        """Return ``[B,target,source]`` edge weights with zero invalid entries."""

        if context_repr.dim() != 3:
            raise ValueError("context_repr must have shape [B,T,H]")
        if adjacency.shape != (
            context_repr.shape[0],
            context_repr.shape[1],
            context_repr.shape[1],
        ):
            raise ValueError("adjacency must have shape [B,T,T]")
        query = self.query(context_repr)
        key = self.key(context_repr)
        scores = torch.bmm(query, key.transpose(1, 2)) / math.sqrt(self.hidden_dim)
        return masked_softmax(scores, adjacency.to(dtype=torch.bool), dim=-1)

"""Exact causal-shift PDCN on each observed dialogue prefix (Rey 2025)."""

import torch
from torch import nn

from .common import GraphCore, row_operator


def causal_shift_factors(adjacency, mask):
    """Return W, W^-1 and boolean ancestor selectors D_k.

    These are derived from graph structure only. No transitive edges are added
    to the canonical DAG, and no future degree or learned edge is consulted.
    """
    b, t, _ = adjacency.shape
    identity = torch.eye(t, dtype=torch.float32, device=adjacency.device).expand(b, -1, -1)
    inverse = identity - row_operator(adjacency, mask)
    closure = torch.linalg.solve_triangular(inverse, identity, upper=False, unitriangular=True)
    ancestors = adjacency | torch.diag_embed(mask)
    rows = []
    for target in range(t):
        row = ancestors[:, target].clone()
        if target:
            completed = torch.stack(rows, dim=1)
            row = row | (adjacency[:, target, :target, None] & completed).any(dim=1)
        rows.append(row)
    return closure, inverse, torch.stack(rows, dim=1)


class DCNCore(GraphCore):
    """PDCN shared MLP, with sum readouts at four depths on each prefix."""

    def __init__(self, hidden_dim=300):
        super().__init__(hidden_dim)
        self.layers = nn.ModuleList(nn.Linear(hidden_dim, hidden_dim, bias=False) for _ in range(4))

    def forward(self, h0, adjacency, relations, attention_mask):
        mask = self.validate(h0, adjacency, relations, attention_mask)
        closure, inverse, ancestors = causal_shift_factors(adjacency, mask)
        causes = torch.bmm(inverse, h0)
        # [B, branch k, receiver i, H], exactly W D_k W^-1 h0.
        branches = torch.einsum("bij,bkj,bjh->bkih", closure, ancestors.to(h0.dtype), causes)
        t = mask.shape[1]
        index = torch.arange(t, device=h0.device)
        # Original PDCN on the induced prefix ending at each receiver, not on
        # future-indexed branches of the full dialogue. No length normalization.
        eligible = ((index[:, None] <= index[None, :])[None]
                    & mask[:, :, None] & mask[:, None, :])
        states = []
        for layer in self.layers:
            branches = torch.relu(layer(branches)) * eligible[..., None]
            states.append(branches.sum(dim=1) * mask[..., None])
        return tuple(states)

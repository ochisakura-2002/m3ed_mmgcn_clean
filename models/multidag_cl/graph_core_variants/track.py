"""Persistent speaker-origin tracks inspired by MTGCN, not its category model."""

import torch
from torch import nn

from .common import GraphCore, row_operator


class TrackCore(GraphCore):
    def __init__(self, hidden_dim=300, *, initial_residual=0.1):
        super().__init__(hidden_dim)
        self.initial_residual = initial_residual
        self.readouts = nn.ModuleList(nn.Linear(2 * hidden_dim, hidden_dim, bias=False)
                                      for _ in range(4))

    def forward(self, h0, adjacency, relations, attention_mask):
        mask = self.validate(h0, adjacency, relations, attention_mask)
        # Anchoring to the first observed speaker avoids numeric speaker-ID assumptions.
        first = relations.same_speaker[:, 0, :] & mask
        same_group = first[:, :, None] == first[:, None, :]
        pairs = mask[:, :, None] & mask[:, None, :]
        if not torch.equal(same_group & pairs, relations.same_speaker):
            raise ValueError("Track v0 supports at most two speaker groups")
        affiliation = torch.stack((first, mask & ~first), dim=1)
        initial = h0[:, None] * affiliation[..., None]
        messages = initial
        operator = row_operator(adjacency, mask, self_terms=True)
        states = []
        for readout in self.readouts:
            messages = (torch.einsum("bij,bkjh->bkih", operator, messages)
                        + self.initial_residual * initial)
            track0, track1 = messages.unbind(1)
            same = torch.where(first[..., None], track0, track1)
            other = torch.where(first[..., None], track1, track0)
            states.append(readout(torch.cat((same, other), dim=-1)) * mask[..., None])
        return tuple(states)

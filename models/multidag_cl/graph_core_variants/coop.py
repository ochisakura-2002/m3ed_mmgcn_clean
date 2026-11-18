"""Deterministic soft four-action communication on the original MultiDAG edges."""

import torch
from torch import nn

from models.multidag_cl.paper_reimplementation.attention import RelationAwareAttention
from models.multidag_cl.paper_reimplementation.contracts import AttentionResult
from models.multidag_cl.paper_reimplementation.dag_layer import MultiDAGLayer
from .common import GraphCore


class CooperativeAttention(RelationAwareAttention):
    def __init__(self, hidden_dim):
        super().__init__(hidden_dim)
        # Order: STANDARD, LISTEN, BROADCAST, ISOLATE. No stochastic sampling.
        self.policy = nn.Linear(hidden_dim, 4)

    def forward(self, query, predecessor_states, predecessor_mask, same_mask):
        result = super().forward(query, predecessor_states, predecessor_mask, same_mask)
        if predecessor_states.shape[1] == 0:
            return result
        receiver = torch.softmax(self.policy(query), -1)
        sender = torch.softmax(self.policy(predecessor_states), -1)
        listen = receiver[:, 0] + receiver[:, 1]
        broadcast = sender[..., 0] + sender[..., 2]
        weights = result.weights * broadcast * listen[:, None]
        # Do not renormalize: that would erase receiver communication control.
        values = torch.where(same_mask.bool()[..., None],
                             self.W_same(predecessor_states),
                             self.W_different(predecessor_states))
        message = torch.bmm(weights[:, None], values).squeeze(1)
        return AttentionResult(logits=result.logits, weights=weights, message=message)


class CoopCore(GraphCore):
    def __init__(self, hidden_dim=300):
        super().__init__(hidden_dim)
        self.layers = nn.ModuleList(MultiDAGLayer(hidden_dim) for _ in range(4))
        for layer in self.layers:
            layer.attention = CooperativeAttention(hidden_dim)

    def forward(self, h0, adjacency, relations, attention_mask):
        self.validate(h0, adjacency, relations, attention_mask)
        states = []
        state = h0
        for layer in self.layers:
            state = layer(state, adjacency, relations, attention_mask).state
            states.append(state)
        return tuple(states)

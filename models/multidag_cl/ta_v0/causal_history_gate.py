"""Gate only existing causal history edges; diagnostics retain scalar moments."""

from contextlib import contextmanager
import math

import torch
from torch import nn


def distance_bucket(distances):
    """Map positive integer dialogue-turn distances to the six frozen buckets."""
    if distances.dtype != torch.int64:
        raise TypeError("distances must be int64")
    if torch.any(distances < 1):
        raise ValueError("history distances must be positive (j < i)")
    boundaries = distances.new_tensor([1, 2, 3, 5, 10])
    return torch.bucketize(distances.contiguous(), boundaries, right=False)


class GateDiagnostics:
    """Population moments over edge occurrences, across batches and DAG layers.

    Empty groups have count=0 and null mean/std. No edge tensors are retained.
    """

    names = ("all", "same_speaker", "different_speaker",
             *(f"distance_bucket_{i}" for i in range(6)))

    def __init__(self):
        self.moments = None

    @torch.no_grad()
    def update(self, gates, same_speaker, buckets):
        values = gates.detach().double()
        groups = torch.cat((torch.zeros_like(buckets),
                            torch.where(same_speaker, 1, 2), 3 + buckets))
        values = values.repeat(3)
        moments = torch.stack((torch.ones_like(values), values, values.square()), dim=1)
        if self.moments is None:
            self.moments = values.new_zeros((len(self.names), 3))
        self.moments.index_add_(0, groups, moments)

    def summary(self):
        rows = (self.moments.cpu().tolist() if self.moments is not None
                else [[0, 0, 0] for _ in self.names])
        result = {}
        for name, (count, total, squares) in zip(self.names, rows):
            mean = total / count if count else None
            std = math.sqrt(max(0., squares / count - mean * mean)) if count else None
            result[name] = {"edge_count": int(count), "mean": mean, "std": std}
        return result


class CausalHistoryGate(nn.Module):
    """A same/different relation embedding, distance embedding and sigmoid MLP."""

    def __init__(self, hidden_dim, speaker_embedding_dim=8, distance_embedding_dim=8,
                 gate_hidden_dim=64):
        super().__init__()
        self.speaker_embedding = nn.Embedding(2, speaker_embedding_dim)
        self.distance_embedding = nn.Embedding(6, distance_embedding_dim)
        self.mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim + speaker_embedding_dim + distance_embedding_dim,
                      gate_hidden_dim), nn.ReLU(), nn.Linear(gate_hidden_dim, 1), nn.Sigmoid(),
        )
        self.diagnostics = None

    def forward(self, query, history, same_speaker, distances):
        """Inputs describe E legal edges: [E,H], [E,H], bool [E], int64 [E]."""
        if (query.ndim != 2 or history.shape != query.shape
                or same_speaker.dtype != torch.bool or same_speaker.shape != query.shape[:1]
                or distances.shape != same_speaker.shape):
            raise ValueError("invalid CHG edge shapes/relation dtype")
        buckets = distance_bucket(distances)
        features = torch.cat((query, history, self.speaker_embedding(same_speaker.long()),
                              self.distance_embedding(buckets)), dim=-1)
        gates = self.mlp(features).squeeze(-1)
        if self.diagnostics is not None and not self.training:
            self.diagnostics.update(gates, same_speaker, buckets)
        return gates

    def reweight(self, query, history_prefix, predecessor_mask, same_mask, semantic_weights):
        """MultiDAG passes completed positions [0,i); its width is the target i.

        Gather only true entries in the existing predecessor row. No second DAG,
        full gate matrix, self/future features or speaker identity vocabulary.
        """
        batch, source = predecessor_mask.nonzero(as_tuple=True)
        if source.numel() == 0:
            return semantic_weights
        target = history_prefix.shape[1]
        gates = self(query[batch], history_prefix[batch, source], same_mask[batch, source],
                     target - source)
        gated = torch.zeros_like(semantic_weights)
        gated[batch, source] = semantic_weights[batch, source] * gates
        return gated / (gated.sum(dim=-1, keepdim=True) + 1e-8)


@contextmanager
def collect_gate_diagnostics(model):
    """Explicit inference scope; restore collectors even if evaluation fails."""
    gates = [module for module in model.modules() if isinstance(module, CausalHistoryGate)]
    if any(gate.diagnostics is not None for gate in gates):
        raise ValueError("nested CHG diagnostics are not supported")
    diagnostics = GateDiagnostics()
    try:
        for gate in gates:
            gate.diagnostics = diagnostics
        yield diagnostics
    finally:
        for gate in gates:
            gate.diagnostics = None

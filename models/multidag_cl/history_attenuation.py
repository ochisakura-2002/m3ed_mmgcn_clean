"""H1 weights for the existing normalized TrackCore predecessor contributions."""

from functools import partial

import torch
from torch import nn
from torch.nn import functional as F


COSINE_EPS = 1e-8
LAMBDA_MAX = 0.5
PARAMETER_COUNT = 2


class HistoryAttenuation(nn.Module):
    def __init__(self):
        super().__init__()
        self.raw_lambda_same = nn.Parameter(torch.tensor(0.0))
        self.raw_lambda_diff = nn.Parameter(torch.tensor(0.0))

    def lambdas(self):
        return (self.raw_lambda_same.clamp(0.0, LAMBDA_MAX),
                self.raw_lambda_diff.clamp(0.0, LAMBDA_MAX))

    def forward(self, operator, h0, adjacency, relations, *, observe=None):
        edges = adjacency.nonzero(as_tuple=True)
        batch, receiver, sender = edges
        detached = h0.detach()
        cosine = F.cosine_similarity(detached[batch, receiver], detached[batch, sender],
                                     dim=-1, eps=COSINE_EPS).clamp(-1.0, 1.0)
        distance = (1.0 - cosine) * 0.5
        same = relations.same_speaker[edges]
        lambda_same, lambda_diff = self.lambdas()
        factor = 1.0 - torch.where(same, lambda_same, lambda_diff) * distance
        if observe is not None:
            observe(same.detach(), factor.detach(), cosine.detach(), distance.detach())
        # GraphCore already validates the strict-past adjacency. Keep the
        # original denominators, diagonal, and all non-edge entries.
        return operator.index_put(edges, operator[edges] * factor)


class FrozenHistoryModel(nn.Module):
    """Own the two scalars separately from the original F0 checkpoint schema."""

    def __init__(self, f0, *, history_attenuation_enabled=True):
        super().__init__()
        self.f0 = f0.requires_grad_(False).eval()
        self.attenuation = HistoryAttenuation()
        self.history_attenuation_enabled = history_attenuation_enabled

    def forward(self, *, history_attenuation_enabled=None, observe=None, **inputs):
        enabled = self.history_attenuation_enabled if history_attenuation_enabled is None else history_attenuation_enabled
        if not enabled:
            return self.f0(**inputs)
        callback = self.attenuation if observe is None else partial(self.attenuation, observe=observe)
        return self.f0(history_attenuation=callback, **inputs)

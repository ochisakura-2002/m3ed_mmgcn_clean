"""Text-centered directed matching shared by Hybrid and Fusion-A."""

from math import sqrt

import torch
from torch import nn
from torch.nn import functional as F

from .base import EPS, WIDTH, UtteranceFusion


EDGES = {
    "text_from_audio": (2, 0, "T<-A"),
    "text_from_visual": (2, 1, "T<-V"),
    "audio_from_text": (0, 2, "A<-T"),
    "visual_from_text": (1, 2, "V<-T"),
}


class DirectedMatcher(nn.Module):
    def __init__(self, rank):
        super().__init__()
        self.query = (nn.Linear(WIDTH, WIDTH, bias=False) if rank is None else nn.Sequential(
            nn.Linear(WIDTH, rank, bias=False), nn.Linear(rank, WIDTH, bias=False)))
        self.value = nn.Linear(WIDTH, WIDTH, bias=False)

    def forward(self, receiver, source, value):
        score = (self.query(receiver) * source).sum(dim=-1) / sqrt(WIDTH)
        gate = score.sigmoid()
        return gate, gate[..., None] * self.value(value)


class TextCenteredMatching(UtteranceFusion):
    def __init__(self, *, rank=None, residual_scale):
        super().__init__()
        self.matchers = nn.ModuleDict({name: DirectedMatcher(rank) for name in EDGES})
        self.residual_scale = residual_scale

    def mix(self, blocks, active, diagnostics):
        values = blocks.unbind(dim=-2)
        normalized = F.layer_norm(blocks, (WIDTH,)).unbind(dim=-2)
        messages = [torch.zeros_like(values[0]) for _ in values]
        for name, (receiver, source, label) in EDGES.items():
            gate, delta = self.matchers[name](normalized[receiver], normalized[source], values[source])
            valid_edge = active[..., receiver] & active[..., source]
            messages[receiver] = messages[receiver] + delta.masked_fill(~valid_edge[..., None], 0.0)
            if diagnostics is not None:
                message_l2 = delta[valid_edge].norm(dim=-1)
                receiver_l2 = values[receiver][valid_edge].norm(dim=-1)
                diagnostics[label] = torch.stack((gate[valid_edge], message_l2, receiver_l2,
                                                  message_l2 / receiver_l2.clamp_min(EPS)), dim=-1)
        residual = self.residual_scale * torch.stack(messages, dim=-2)
        self.record_activity(diagnostics, blocks, residual, active)
        return blocks + residual

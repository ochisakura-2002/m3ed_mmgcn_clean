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
EDGE_LABELS = tuple(label for _, _, label in EDGES.values())


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
    def __init__(self, *, rank=None, residual_scale, active_edges=None):
        super().__init__()
        if active_edges is not None and (
            rank is None or not isinstance(active_edges, (tuple, list)) or not active_edges
            or any(label not in EDGE_LABELS for label in active_edges)
            or len(set(active_edges)) != len(active_edges)
        ):
            raise ValueError("active_edges must contain unique low-rank matching edge labels")
        self.active_edges = tuple(label for label in EDGE_LABELS if active_edges is None or label in active_edges)
        self.matchers = nn.ModuleDict({
            name: DirectedMatcher(rank) for name, (_, _, label) in EDGES.items() if label in self.active_edges
        })
        self.record_edge_residuals = active_edges is not None
        self.residual_scale = residual_scale

    def mix(self, blocks, active, diagnostics):
        values = blocks.unbind(dim=-2)
        normalized = F.layer_norm(blocks, (WIDTH,)).unbind(dim=-2)
        messages = [torch.zeros_like(values[0]) for _ in values]
        edge_receivers = {}
        for name, matcher in self.matchers.items():
            receiver, source, label = EDGES[name]
            gate, delta = matcher(normalized[receiver], normalized[source], values[source])
            valid_edge = active[..., receiver] & active[..., source]
            messages[receiver] = messages[receiver] + delta.masked_fill(~valid_edge[..., None], 0.0)
            if diagnostics is not None:
                message_l2 = delta[valid_edge].norm(dim=-1)
                receiver_l2 = values[receiver][valid_edge].norm(dim=-1)
                diagnostics[label] = torch.stack((gate[valid_edge], message_l2, receiver_l2,
                                                  message_l2 / receiver_l2.clamp_min(EPS)), dim=-1)
                if self.record_edge_residuals:
                    edge_receivers[label] = (receiver, valid_edge, (self.residual_scale * delta[valid_edge]).norm(dim=-1))
        residual = self.residual_scale * torch.stack(messages, dim=-2)
        for label, (receiver, valid_edge, applied_edge_l2) in edge_receivers.items():
            samples = diagnostics[label]
            denominator = samples[:, 2].clamp_min(EPS)
            applied_receiver_l2 = residual[..., receiver, :][valid_edge].norm(dim=-1)
            diagnostics[label] = torch.cat((samples, (applied_edge_l2 / denominator)[:, None],
                                            (applied_receiver_l2 / denominator)[:, None]), dim=-1)
        self.record_activity(diagnostics, blocks, residual, active)
        return blocks + residual

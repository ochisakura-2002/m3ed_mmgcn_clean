"""Utterance-level joint router and modality-specific residual experts."""

import torch
from torch import nn
from torch.nn import functional as F

from .base import MODALITIES, OUTPUT_WIDTH, WIDTH, UtteranceFusion


class UtteranceModalityExpertRouter(UtteranceFusion):
    def __init__(self, *, router_hidden_dim, expert_hidden_dim, residual_scale):
        super().__init__()
        self.residual_scale = residual_scale
        self.router = nn.Sequential(nn.Linear(OUTPUT_WIDTH, router_hidden_dim), nn.GELU(),
                                    nn.Linear(router_hidden_dim, len(MODALITIES)))
        self.experts = nn.ModuleList(nn.Sequential(nn.Linear(WIDTH, expert_hidden_dim), nn.GELU(),
                                                   nn.Linear(expert_hidden_dim, WIDTH))
                                     for _ in MODALITIES)

    def mix(self, blocks, active, diagnostics):
        normalized = F.layer_norm(blocks, (WIDTH,))
        logits = self.router(normalized.flatten(-2)).masked_fill(~active, -torch.inf)
        count = active.sum(dim=-1, keepdim=True)
        probabilities = logits.masked_fill(count == 0, 0.0).softmax(dim=-1).masked_fill(~active, 0.0)
        updates = torch.stack([expert(block) for expert, block in zip(self.experts, normalized.unbind(dim=-2))],
                              dim=-2)
        residual = self.residual_scale * (count * probabilities)[..., None] * updates
        if diagnostics is not None:
            diagnostics["router"] = torch.cat((probabilities, residual.norm(dim=-1)), dim=-1)
        self.record_activity(diagnostics, blocks, residual, active)
        return blocks + residual

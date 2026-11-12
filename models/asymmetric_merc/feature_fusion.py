"""Frozen-representation B1 and scalar VRG-v0, without source expert imports."""

import torch
from torch import nn


class ConcatLinear(nn.Module):
    def __init__(self):
        super().__init__()
        self.classifier = nn.Linear(1500 + 768, 6)

    def forward(self, z_ta, z_v):
        return self.classifier(torch.cat((z_ta, z_v), dim=-1))


class VisualReliabilityGate(nn.Module):
    def __init__(self):
        super().__init__()
        self.ta_projection = nn.Linear(1500, 300)
        self.v_projection = nn.Linear(768, 300)
        self.gate = nn.Sequential(nn.Linear(1200, 64), nn.ReLU(), nn.Linear(64, 1), nn.Sigmoid())
        self.classifier = nn.Linear(300, 6)

    def forward(self, z_ta, z_v, *, return_details=False):
        t = torch.relu(self.ta_projection(z_ta))
        v = torch.relu(self.v_projection(z_v))
        q = torch.cat((t, v, torch.abs(t - v), t * v), dim=-1)
        r = self.gate(q)
        logits = self.classifier(t + r * v)
        return {"t": t, "v": v, "r": r, "logits": logits} if return_details else logits


def build_model(*, vrg_enabled=False):
    """YAML vrg.enabled=false is exactly the B1 model, including state schema."""
    if type(vrg_enabled) is not bool:
        raise ValueError("vrg.enabled must be a boolean")
    return VisualReliabilityGate() if vrg_enabled else ConcatLinear()

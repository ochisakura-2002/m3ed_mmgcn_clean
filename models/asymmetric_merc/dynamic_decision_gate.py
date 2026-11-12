"""Probability-only TA-preserving scalar decision fusion (833 parameters)."""

import math

import torch
from torch import nn

EPSILON = 1e-12
INITIAL_VISUAL_WEIGHT = 0.12


def gate_input(p_ta, p_v):
    if p_ta.ndim != 2 or p_ta.shape[1] != 6 or p_v.shape != p_ta.shape:
        raise ValueError("Expert probabilities must both have shape [B,6]")
    # Expert outputs are frozen, including when callers supply requires_grad=True.
    ta = p_ta.detach().clamp(min=EPSILON)
    visual = p_v.detach().clamp(min=EPSILON)
    return ta, visual, torch.cat((ta, visual, (ta - visual).abs(), ta * visual), dim=1)


class ConstantDecisionGate(nn.Module):
    """Untrained B0 sanity control; zero parameters."""

    def forward(self, p_ta, p_v, *, return_details=False):
        ta, visual, q = gate_input(p_ta, p_v)
        g = ta.new_full((len(ta), 1), INITIAL_VISUAL_WEIGHT)
        scores = (1 - g) * ta.log() + g * visual.log()
        return {"scores": scores, "g": g, "q": q} if return_details else scores


class DynamicDecisionGate(nn.Module):
    def __init__(self):
        super().__init__()
        # Float64 preserves the probability artifact precision and B0 contract.
        self.hidden = nn.Linear(24, 32, dtype=torch.float64)
        self.final_gate_layer = nn.Linear(32, 1, dtype=torch.float64)
        nn.init.zeros_(self.final_gate_layer.weight)
        nn.init.constant_(self.final_gate_layer.bias,
                          math.log(INITIAL_VISUAL_WEIGHT / (1 - INITIAL_VISUAL_WEIGHT)))

    def forward(self, p_ta, p_v, *, return_details=False):
        ta, visual, q = gate_input(p_ta, p_v)
        g = torch.sigmoid(self.final_gate_layer(torch.relu(self.hidden(q))))
        scores = (1 - g) * ta.log() + g * visual.log()
        return {"scores": scores, "g": g, "q": q} if return_details else scores


def build_model(*, enabled=True):
    if type(enabled) is not bool:
        raise ValueError("dynamic_decision_gate.enabled must be boolean")
    return DynamicDecisionGate() if enabled else ConstantDecisionGate()

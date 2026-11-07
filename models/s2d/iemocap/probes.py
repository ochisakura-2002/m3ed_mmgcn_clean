"""Small classifiers over frozen bank tensors; independent of the S2D model."""

from __future__ import annotations

import os
import random

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


SEEDS = (100, 101, 102, 103, 104)


def seed_probe(seed: int) -> None:
    """Repeatable within one device/software stack; unsupported kernels fail."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def mean_pool(sequence: torch.Tensor) -> torch.Tensor:
    if sequence.ndim != 3 or tuple(sequence.shape[1:]) != (16, 768):
        raise ValueError("Expected frozen sequence [N,16,768]")
    return sequence.mean(dim=1)


class LinearProbe(nn.Module):
    def __init__(self):
        super().__init__()
        self.classifier = nn.Linear(768, 6)

    def forward(self, pooled):
        if pooled.ndim != 2 or pooled.shape[1] != 768:
            raise ValueError("Linear probe expects mean-pooled [N,768]")
        return self.classifier(pooled)


class CausalResidualBlock(nn.Module):
    def __init__(self, hidden: int, dilation: int, dropout: float):
        super().__init__()
        self.left_padding = 2 * dilation
        self.conv = nn.Conv1d(hidden, hidden, kernel_size=3, dilation=dilation)
        self.dropout = nn.Dropout(dropout)

    def forward(self, value):
        update = self.conv(F.pad(value, (self.left_padding, 0)))
        return F.gelu(value + self.dropout(F.gelu(update)))


class TemporalProbe(nn.Module):
    """Causal convolutions, followed by whole-utterance pooling for diagnosis.

    B inputs already contain non-causal TMA. Neither this pooling nor the
    original crop pipeline implies a causal robot deployment contract.
    """

    def __init__(self):
        super().__init__()
        self.projection = nn.Conv1d(768, 128, kernel_size=1)
        self.blocks = nn.Sequential(CausalResidualBlock(128, 1, 0.1),
                                    CausalResidualBlock(128, 2, 0.1))
        self.classifier = nn.Linear(128, 6)

    def encode_slots(self, sequence):
        if sequence.ndim != 3 or tuple(sequence.shape[1:]) != (16, 768):
            raise ValueError("Temporal probe expects frozen [N,16,768]")
        return self.blocks(self.projection(sequence.transpose(1, 2)))

    def forward(self, sequence):
        return self.classifier(self.encode_slots(sequence).mean(dim=-1))

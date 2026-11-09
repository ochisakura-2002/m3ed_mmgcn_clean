"""Reusable current-protocol TA-only causal MultiDAG component."""

from .config import LABEL_VOCAB, TAV0Config
from .model import TAOutput, TAV0

__all__ = ["LABEL_VOCAB", "TAV0Config", "TAOutput", "TAV0"]

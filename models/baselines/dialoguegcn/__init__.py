"""Isolated DialogueGCN-derived strict-causal project baseline."""

from .causal_dialoguegcn_model import (
    CausalDialogueGCNBaseline,
    CausalDialogueGCNConfig,
)

__all__ = ["CausalDialogueGCNBaseline", "CausalDialogueGCNConfig"]

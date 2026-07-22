"""Compatibility wrapper for :mod:`models.dialoguegcn.unified.causal_dialoguegcn_graph`."""

from models.dialoguegcn.unified.causal_dialoguegcn_graph import (
    CausalEdgeAttention,
    build_causal_dialoguegcn_graph,
)

__all__ = ["CausalEdgeAttention", "build_causal_dialoguegcn_graph"]

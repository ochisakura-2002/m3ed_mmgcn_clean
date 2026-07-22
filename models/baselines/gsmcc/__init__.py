"""Compatibility wrapper for the causal GS-MCC project variant."""

from models.gsmcc.project_variant.causal import (
    CausalGSMCCConfig,
    CausalGSMCCInspiredBaseline,
    compute_causal_gsmcc_loss,
)

__all__ = [
    "CausalGSMCCConfig",
    "CausalGSMCCInspiredBaseline",
    "compute_causal_gsmcc_loss",
]

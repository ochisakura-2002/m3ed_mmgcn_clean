"""Compatibility wrapper for the causal GS-MCC project-variant model."""

from models.gsmcc.project_variant.causal.causal_gsmcc_model import (
    CausalGSMCCConfig,
    CausalGSMCCInspiredBaseline,
)

__all__ = ["CausalGSMCCConfig", "CausalGSMCCInspiredBaseline"]

"""Compatibility wrapper for the causal GS-MCC project-variant loss."""

from models.gsmcc.project_variant.causal.causal_gsmcc_losses import compute_causal_gsmcc_loss

__all__ = ["compute_causal_gsmcc_loss"]

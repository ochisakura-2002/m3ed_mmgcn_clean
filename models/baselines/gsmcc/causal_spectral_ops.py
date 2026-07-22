"""Compatibility wrapper for the causal GS-MCC project-variant spectral ops."""

from models.gsmcc.project_variant.causal.causal_spectral_ops import (
    CausalDirectedPolynomialLayer,
    causal_directed_polynomial_filter,
)

__all__ = ["CausalDirectedPolynomialLayer", "causal_directed_polynomial_filter"]

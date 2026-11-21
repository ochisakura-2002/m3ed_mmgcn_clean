"""Shared encoded-input, availability and h0 contract."""

import torch
from torch import nn

from models.multidag_cl.graph_core_variants.common import valid_prefix


MODALITIES = "AVT"
WIDTH = 100
OUTPUT_WIDTH = len(MODALITIES) * WIDTH
EPS = 1e-8


class UtteranceFusion(nn.Module):
    def forward(self, *, text, audio, visual, attention_mask, availability=None, diagnostics=None):
        """Return [B,T,300] in A,V,T order. Availability is bool [B,T,3]."""
        valid = valid_prefix(attention_mask)
        inputs = (audio, visual, text)
        if any(x.shape != (*valid.shape, WIDTH) or x.dtype != torch.float32 for x in inputs):
            raise TypeError("encoded modalities must be float32 [B,T,100]")
        if any(x.device != valid.device for x in inputs):
            raise ValueError("encoded modalities and mask must share a device")
        active = valid[..., None].expand(-1, -1, len(MODALITIES))
        if availability is not None:
            if availability.dtype != torch.bool or availability.shape != active.shape:
                raise TypeError("availability must be bool [B,T,3] in A,V,T order")
            if availability.device != valid.device:
                raise ValueError("availability and mask must share a device")
            active = active & availability
        blocks = torch.stack(inputs, dim=-2).masked_fill(~active[..., None], 0.0)
        return self.mix(blocks, active, diagnostics).masked_fill(~active[..., None], 0.0).flatten(-2)

    def mix(self, blocks, active, diagnostics):
        raise NotImplementedError

    @staticmethod
    def record_activity(diagnostics, blocks, residual, active):
        if diagnostics is not None:
            receiver = blocks.norm(dim=-1).clamp_min(EPS)
            diagnostics["activity"] = (residual.norm(dim=-1) / receiver)[active]


class DirectConcatFusion(UtteranceFusion):
    def mix(self, blocks, active, diagnostics):
        return blocks

"""External bounded refinement for the frozen six-class top two."""

from dataclasses import dataclass

import torch
from torch import nn


STATE_WIDTH = 300
HIDDEN_WIDTH = 16
CLASS_COUNT = 6
PAIR_COUNT = 15
PARAMETER_COUNT = 9856


def pair_index_and_sign(top1, top2):
    """Map distinct class IDs to lexicographic unordered-pair rows."""
    low = torch.minimum(top1, top2)
    high = torch.maximum(top1, top2)
    index = low * (2 * CLASS_COUNT - 1 - low) // 2 + high - low - 1
    return index, torch.where(top1 == low, 1, -1)


class PairwiseRefinementHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.normalize = nn.LayerNorm(STATE_WIDTH, eps=1e-5, elementwise_affine=False)
        self.projection = nn.Linear(2 * STATE_WIDTH, HIDDEN_WIDTH)
        self.pair_embeddings = nn.Parameter(torch.zeros(PAIR_COUNT, HIDDEN_WIDTH))
        nn.init.xavier_uniform_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)

    def forward(self, h0, h4, top1, top2):
        index, sign = pair_index_and_sign(top1, top2)
        features = torch.cat((self.normalize(h0.detach()), self.normalize(h4.detach())), dim=-1)
        projected = torch.tanh(self.projection(features))
        score = (self.pair_embeddings[index] * projected).sum(dim=-1)
        delta = torch.tanh(sign * score)
        # Floating-point tanh may saturate at exactly ±1.
        inner_bound = 1 - torch.finfo(delta.dtype).eps / 2
        return delta.clamp(-inner_bound, inner_bound)


@dataclass(frozen=True)
class Refinement:
    logits: torch.Tensor
    delta: torch.Tensor
    corrected_margin: torch.Tensor
    written: torch.Tensor


def bounded_writeback(logits, top1, top2, gate, margin, delta):
    """Anchor both candidates at the frozen runner-up score before writing."""
    corrected = margin + delta
    rows = logits[gate]
    a, b = top1[gate], top2[gate]
    row = torch.arange(len(rows), device=logits.device)
    anchor = rows[row, b]
    gap = corrected[gate]
    # Test the stored value after conversion back to the F0 logit dtype.
    winner_logit = (anchor + gap.abs()).to(logits.dtype)
    write = (delta[gate] != 0) & (gap != 0) & (winner_logit > anchor)
    updated = rows.clone()
    updated[row, a] = torch.where(gap > 0, winner_logit, anchor)
    updated[row, b] = torch.where(gap < 0, winner_logit, anchor)
    final = logits.clone()
    final[gate] = torch.where(write[:, None], updated, rows)
    written = torch.zeros_like(gate)
    written[gate] = write
    return Refinement(final, delta, corrected, written)

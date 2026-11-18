"""Unified four-depth contract and strict graph-boundary validation."""

import torch
from torch import nn

from models.multidag_cl.paper_reimplementation.contracts import SpeakerRelations


def valid_prefix(mask):
    if mask.ndim != 2 or mask.dtype not in (torch.bool, torch.int64):
        raise TypeError("attention_mask must be bool/int64 [B,T]")
    if min(mask.shape) < 1 or not torch.all((mask == 0) | (mask == 1)):
        raise ValueError("empty/invalid attention_mask")
    valid = mask.bool()
    lengths = valid.sum(-1)
    if torch.any(lengths < 1) or not torch.equal(
            valid, torch.arange(mask.shape[1], device=mask.device)[None] < lengths[:, None]):
        raise ValueError("attention_mask must be a nonempty contiguous prefix")
    return valid


class GraphCore(nn.Module):
    """core(h0, DAG, speaker relations, mask) -> four [B,T,H] tensors.

    Production H=300. Smaller H is useful for equation-level unit tests only.
    Implementations must not add/repair edges or retain state across calls.
    """

    def __init__(self, hidden_dim=300):
        super().__init__()
        if type(hidden_dim) is not int or hidden_dim < 1:
            raise ValueError("hidden_dim must be positive integer")
        self.hidden_dim = hidden_dim

    def validate(self, h0, adjacency, relations, attention_mask):
        mask = valid_prefix(attention_mask)
        b, t = mask.shape
        if h0.dtype != torch.float32 or h0.shape != (b, t, self.hidden_dim):
            raise TypeError("h0 must be float32 [B,T,H]")
        if not torch.isfinite(h0).all() or torch.count_nonzero(h0[~mask]):
            raise ValueError("h0 must be finite with exact zero padding")
        if adjacency.dtype != torch.bool or adjacency.shape != (b, t, t):
            raise TypeError("adjacency must be bool [B,T,T]")
        if not isinstance(relations, SpeakerRelations):
            raise TypeError("relations must be SpeakerRelations")
        same, other = relations.same_speaker, relations.different_speaker
        if any(x.dtype != torch.bool or x.shape != adjacency.shape for x in (same, other)):
            raise TypeError("relations must be bool [B,T,T]")
        if len({x.device for x in (h0, adjacency, same, other, mask)}) != 1:
            raise ValueError("GraphCore inputs must share device")
        pairs = mask[:, :, None] & mask[:, None, :]
        if torch.any(torch.triu(adjacency)) or torch.any(adjacency & ~pairs):
            raise ValueError("DAG may not contain self/future/padded edges")
        if (torch.any(same & other) or not torch.equal(same | other, pairs)
                or not torch.equal(same, same.transpose(1, 2))
                or not torch.equal(same.diagonal(dim1=1, dim2=2), mask)):
            raise ValueError("speaker relations must partition valid pairs symmetrically")
        return mask


def row_operator(adjacency, mask, *, self_terms=False):
    """Normalize only the receiver's observed incoming edges."""
    values = adjacency.to(torch.float32)
    if self_terms:
        values = values + torch.diag_embed(mask.to(values.dtype))
    return values / values.sum(-1, keepdim=True).clamp_min(1)

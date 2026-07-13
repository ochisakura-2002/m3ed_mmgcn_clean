"""Batched graph builders using the convention ``adj[target, source]``."""

from __future__ import annotations

from typing import Optional

import torch


def _validated_mask(
    attention_mask: torch.Tensor,
    lengths: torch.Tensor,
) -> torch.Tensor:
    """Validate a prefix-style dialogue mask and return it as boolean."""

    if attention_mask.dim() != 2:
        raise ValueError("attention_mask must have shape [B, T]")
    if lengths.dim() != 1 or lengths.shape[0] != attention_mask.shape[0]:
        raise ValueError("lengths must have shape [B]")
    if lengths.device != attention_mask.device:
        raise ValueError("attention_mask and lengths must share a device")

    mask = attention_mask.to(dtype=torch.bool)
    lengths_long = lengths.to(dtype=torch.long)
    batch_size, seq_len = mask.shape
    if torch.any(lengths_long < 0) or torch.any(lengths_long > seq_len):
        raise ValueError("every length must be between 0 and T")

    time = torch.arange(seq_len, device=mask.device).unsqueeze(0)
    expected = time < lengths_long.unsqueeze(1)
    if not torch.equal(mask, expected):
        raise ValueError("attention_mask must be a contiguous prefix matching lengths")
    if batch_size == 0:
        raise ValueError("empty batches are not supported")
    return mask


def build_causal_utterance_adjacency(
    attention_mask: torch.Tensor,
    lengths: torch.Tensor,
    window_past: Optional[int] = None,
    self_loop: bool = True,
) -> torch.Tensor:
    """Build ``[B,T,T]`` causal adjacency with ``source_time <= target_time``."""

    mask = _validated_mask(attention_mask, lengths)
    if window_past is not None and window_past < 0:
        raise ValueError("window_past must be None or non-negative")

    _, seq_len = mask.shape
    target_time = torch.arange(seq_len, device=mask.device).view(1, seq_len, 1)
    source_time = torch.arange(seq_len, device=mask.device).view(1, 1, seq_len)
    causal = source_time <= target_time
    if window_past is not None:
        causal = causal & ((target_time - source_time) <= window_past)
    if not self_loop:
        causal = causal & (source_time != target_time)

    valid = mask.unsqueeze(2) & mask.unsqueeze(1)
    adjacency = valid & causal
    assert_no_future_edges(adjacency, torch.arange(seq_len, device=mask.device))
    return adjacency


def utterance_time_to_multimodal_node_time(
    seq_len: int,
    num_modalities: int = 3,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Return utterance-major node times for ``[u0_m0, u0_m1, ...]``."""

    if seq_len < 0:
        raise ValueError("seq_len must be non-negative")
    if num_modalities <= 0:
        raise ValueError("num_modalities must be positive")
    return torch.arange(seq_len, device=device).repeat_interleave(num_modalities)


def build_causal_multimodal_adjacency(
    attention_mask: torch.Tensor,
    lengths: torch.Tensor,
    window_past: Optional[int] = None,
    num_modalities: int = 3,
    include_cross_modal_history: bool = True,
    include_same_time_cross_modal: bool = True,
    self_loop: bool = True,
) -> torch.Tensor:
    """Build a causal utterance-modality graph with approximately ``M*T`` nodes."""

    mask = _validated_mask(attention_mask, lengths)
    if window_past is not None and window_past < 0:
        raise ValueError("window_past must be None or non-negative")
    if num_modalities <= 0:
        raise ValueError("num_modalities must be positive")

    batch_size, seq_len = mask.shape
    node_time = utterance_time_to_multimodal_node_time(
        seq_len,
        num_modalities=num_modalities,
        device=mask.device,
    )
    node_modality = torch.arange(num_modalities, device=mask.device).repeat(seq_len)
    target_time = node_time.view(1, -1, 1)
    source_time = node_time.view(1, 1, -1)
    target_modality = node_modality.view(1, -1, 1)
    source_modality = node_modality.view(1, 1, -1)

    allowed = source_time <= target_time
    if window_past is not None:
        allowed = allowed & ((target_time - source_time) <= window_past)

    same_time = source_time == target_time
    same_modality = source_modality == target_modality
    if not include_cross_modal_history:
        allowed = allowed & (same_modality | same_time)
    if not include_same_time_cross_modal:
        allowed = allowed & (~same_time | same_modality)
    if not self_loop:
        node_index = torch.arange(seq_len * num_modalities, device=mask.device)
        allowed = allowed & (node_index.view(1, -1, 1) != node_index.view(1, 1, -1))

    node_mask = mask.unsqueeze(-1).expand(batch_size, seq_len, num_modalities)
    node_mask = node_mask.reshape(batch_size, seq_len * num_modalities)
    adjacency = allowed & node_mask.unsqueeze(2) & node_mask.unsqueeze(1)
    assert_no_future_edges(adjacency, node_time)
    return adjacency


def build_full_context_negative_control_adjacency(
    attention_mask: torch.Tensor,
    lengths: torch.Tensor,
    num_modalities: int = 1,
    self_loop: bool = True,
) -> torch.Tensor:
    """Build an explicitly noncausal full graph for tests only."""

    mask = _validated_mask(attention_mask, lengths)
    if num_modalities <= 0:
        raise ValueError("num_modalities must be positive")
    batch_size, seq_len = mask.shape
    node_mask = mask.unsqueeze(-1).expand(batch_size, seq_len, num_modalities)
    node_mask = node_mask.reshape(batch_size, seq_len * num_modalities)
    adjacency = node_mask.unsqueeze(2) & node_mask.unsqueeze(1)
    if not self_loop:
        eye = torch.eye(adjacency.shape[-1], dtype=torch.bool, device=mask.device)
        adjacency = adjacency & ~eye.unsqueeze(0)
    return adjacency


def row_normalize_adjacency(adjacency: torch.Tensor) -> torch.Tensor:
    """Normalize incoming source weights independently for every target row."""

    if adjacency.dim() not in (2, 3) or adjacency.shape[-1] != adjacency.shape[-2]:
        raise ValueError("adjacency must have shape [N,N] or [B,N,N]")
    weights = adjacency if adjacency.is_floating_point() else adjacency.to(torch.float32)
    denominator = weights.sum(dim=-1, keepdim=True)
    return torch.where(
        denominator > 0,
        weights / denominator.clamp_min(torch.finfo(weights.dtype).tiny),
        torch.zeros_like(weights),
    )


def assert_no_future_edges(
    adjacency: torch.Tensor,
    node_time: torch.Tensor,
) -> None:
    """Raise when a nonzero edge has a future source for its target."""

    if adjacency.dim() not in (2, 3) or adjacency.shape[-1] != adjacency.shape[-2]:
        raise ValueError("adjacency must have shape [N,N] or [B,N,N]")
    if node_time.dim() == 1:
        if node_time.shape[0] != adjacency.shape[-1]:
            raise ValueError("node_time length must match adjacency")
        target_time = node_time.view(-1, 1)
        source_time = node_time.view(1, -1)
    elif node_time.dim() == 2 and adjacency.dim() == 3:
        if node_time.shape != adjacency.shape[:2]:
            raise ValueError("batched node_time must have shape [B,N]")
        target_time = node_time.unsqueeze(2)
        source_time = node_time.unsqueeze(1)
    else:
        raise ValueError("node_time must have shape [N] or [B,N]")

    future_edge = (adjacency != 0) & (source_time > target_time)
    if torch.any(future_edge):
        first = future_edge.nonzero(as_tuple=False)[0].tolist()
        raise AssertionError(
            "future edge violates source_time <= target_time at index " + str(first)
        )

"""
Official-aligned dense multimodal graph utilities for M3ED-MMGCN.

This module adapts the official MMGCN adjacency construction to the local
M3ED dialogue-batch format.

Official MMGCN idea:
    1. Each valid utterance has one node per modality.
    2. Same-modality utterance edges are weighted by cosine similarity:
           sim = 1 - acos(cosine) / pi
    3. Cross-modality edges connect the same utterance across modalities,
       also weighted by cosine similarity.
    4. The final adjacency is normalized as:
           D^{-1/2} A D^{-1/2}

Local extension:
    context_mode = "full"
        use all allowed utterances inside each dialogue.

    context_mode = "causal"
        target utterance can only receive source utterances from history
        where source_time <= target_time.

Node layout:
    modality order is [audio, visual, text], matching the official MMGCN
    convention [a, v, l].

    audio node  = 0 * N + utterance_global_index
    visual node = 1 * N + utterance_global_index
    text node   = 2 * N + utterance_global_index
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import torch


def lengths_to_list(lengths) -> List[int]:
    if torch.is_tensor(lengths):
        return [int(x) for x in lengths.detach().cpu().tolist()]
    return [int(x) for x in lengths]


def normalize_window(value: Optional[int]) -> Optional[int]:
    if value is None:
        return None

    if isinstance(value, str):
        if value.lower() in {"none", "null", ""}:
            return None
        return int(value)

    return int(value)


def build_utterance_index(
    lengths,
    max_len: int,
    device: torch.device,
) -> Tuple[torch.Tensor, List[Tuple[int, int]]]:
    """
    Build mapping from padded [B, T] positions to compact valid utterance index.

    Returns:
        global_index: [B, T], invalid positions are -1
        positions: list of (batch_index, time_index)
    """
    length_list = lengths_to_list(lengths)
    batch_size = len(length_list)

    global_index = torch.full(
        (batch_size, int(max_len)),
        fill_value=-1,
        dtype=torch.long,
        device=device,
    )

    positions: List[Tuple[int, int]] = []
    cursor = 0

    for batch_index, length in enumerate(length_list):
        valid_length = min(int(length), int(max_len))

        for time_index in range(valid_length):
            global_index[batch_index, time_index] = cursor
            positions.append((batch_index, time_index))
            cursor += 1

    return global_index, positions


def context_allowed(
    target_time: int,
    source_time: int,
    context_mode: str,
    window_past: Optional[int],
    window_future: Optional[int],
) -> bool:
    context_mode = str(context_mode).lower()

    if context_mode not in {"full", "causal"}:
        raise ValueError(
            f"Unsupported context_mode={context_mode}. Use 'full' or 'causal'."
        )

    if context_mode == "causal" and source_time > target_time:
        return False

    if window_past is not None:
        if source_time < target_time - int(window_past):
            return False

    if window_future is not None:
        if source_time > target_time + int(window_future):
            return False

    return True


def cosine_to_mm_similarity(cosine: torch.Tensor) -> torch.Tensor:
    """
    Convert cosine similarity to official MMGCN edge similarity.

    Official form:
        sim = 1 - acos(cosine * 0.99999) / pi

    Here we clamp for numerical stability.
    """
    cosine = cosine.clamp(min=-0.99999, max=0.99999)
    return 1.0 - torch.acos(cosine) / torch.pi


def pairwise_mm_similarity(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Pairwise official-style similarity matrix for one modality.

    Args:
        x: [L, H]

    Returns:
        sim: [L, L]
    """
    norm = x.norm(dim=1, keepdim=True).clamp_min(eps)
    x_norm = x / norm
    cosine = torch.matmul(x_norm, x_norm.transpose(0, 1))
    return cosine_to_mm_similarity(cosine)


def paired_mm_similarity(
    x: torch.Tensor,
    y: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Official-style similarity between paired utterances from two modalities.

    Args:
        x: [L, H]
        y: [L, H]

    Returns:
        sim: [L]
    """
    x_norm = x / x.norm(dim=1, keepdim=True).clamp_min(eps)
    y_norm = y / y.norm(dim=1, keepdim=True).clamp_min(eps)
    cosine = torch.sum(x_norm * y_norm, dim=1)
    return cosine_to_mm_similarity(cosine)


def build_official_like_multimodal_adjacency(
    audio_nodes: torch.Tensor,
    visual_nodes: torch.Tensor,
    text_nodes: torch.Tensor,
    lengths,
    max_len: int,
    context_mode: str = "full",
    window_past: Optional[int] = None,
    window_future: Optional[int] = None,
    gamma: float = 0.7,
) -> Tuple[torch.Tensor, torch.Tensor, List[Tuple[int, int]]]:
    """
    Build official-aligned multimodal adjacency.

    Args:
        audio_nodes:  [N, H]
        visual_nodes: [N, H]
        text_nodes:   [N, H]
        lengths: dialogue lengths
        max_len: padded max dialogue length

    Returns:
        adj: [3N, 3N]
        global_index: [B, T]
        positions: list of valid utterance positions
    """
    device = audio_nodes.device
    dtype = audio_nodes.dtype

    window_past = normalize_window(window_past)
    window_future = normalize_window(window_future)
    gamma = float(gamma)

    global_index, positions = build_utterance_index(
        lengths=lengths,
        max_len=max_len,
        device=device,
    )

    num_utterances = len(positions)
    modal_features = [audio_nodes, visual_nodes, text_nodes]
    modal_count = 3
    num_nodes = modal_count * num_utterances

    if num_nodes == 0:
        empty_adj = torch.zeros((0, 0), dtype=dtype, device=device)
        return empty_adj, global_index, positions

    adj = torch.zeros(
        (num_nodes, num_nodes),
        dtype=dtype,
        device=device,
    )

    length_list = lengths_to_list(lengths)

    start = 0

    for dialogue_length in length_list:
        valid_length = min(int(dialogue_length), int(max_len))

        if valid_length <= 0:
            continue

        end = start + valid_length

        # Same-modality edges.
        for modal_index, features in enumerate(modal_features):
            sub_features = features[start:end]
            sub_adj = pairwise_mm_similarity(sub_features)

            for target_time in range(valid_length):
                target_global = start + target_time
                target_node = modal_index * num_utterances + target_global

                for source_time in range(valid_length):
                    if not context_allowed(
                        target_time=target_time,
                        source_time=source_time,
                        context_mode=context_mode,
                        window_past=window_past,
                        window_future=window_future,
                    ):
                        continue

                    source_global = start + source_time
                    source_node = modal_index * num_utterances + source_global
                    adj[target_node, source_node] = sub_adj[target_time, source_time]

        # Cross-modality edges for the same utterance.
        for target_modal in range(modal_count):
            for source_modal in range(modal_count):
                if target_modal == source_modal:
                    continue

                target_features = modal_features[target_modal][start:end]
                source_features = modal_features[source_modal][start:end]
                diag_sim = paired_mm_similarity(target_features, source_features)

                for time_index in range(valid_length):
                    utterance_global = start + time_index
                    target_node = target_modal * num_utterances + utterance_global
                    source_node = source_modal * num_utterances + utterance_global
                    adj[target_node, source_node] = gamma * diag_sim[time_index]

        start = end

    degree = adj.sum(dim=1).clamp_min(1e-8)
    degree_inv_sqrt = torch.pow(degree, -0.5)
    adj = degree_inv_sqrt.view(-1, 1) * adj * degree_inv_sqrt.view(1, -1)

    return adj, global_index, positions


def adjacency_density(adj: torch.Tensor) -> float:
    if adj.numel() == 0:
        return 0.0

    return float((adj > 0).float().mean().item())
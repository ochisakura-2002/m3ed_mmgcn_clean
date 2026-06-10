"""
Official-aligned dense multimodal graph utilities.

本文件负责构造 MMGCN 使用的 dense multimodal adjacency。

节点布局固定为官方 MMGCN 顺序：
    audio node  = 0 * N + utterance_global_index
    visual node = 1 * N + utterance_global_index
    text node   = 2 * N + utterance_global_index

为什么固定保留三类节点：
    1. classifier 输入维度保持 3H 不变；
    2. 旧 checkpoint 结构不变；
    3. 测试时缺失模态可以复用三模态 checkpoint；
    4. full 三模态默认行为尽量保持旧逻辑。

active_modalities 只控制哪些模态参与建边：
    full:  audio + visual + text，行为等价旧版本；
    缺失: inactive 模态节点保留，但对应 row/col 没有边。
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import torch


MODALITY_ORDER: Tuple[str, str, str] = ("audio", "visual", "text")
FULL_MODALITIES: Tuple[str, str, str] = ("text", "audio", "visual")


def lengths_to_list(lengths) -> List[int]:
    """把 lengths 转成 Python int list。"""
    if torch.is_tensor(lengths):
        return [int(x) for x in lengths.detach().cpu().tolist()]
    return [int(x) for x in lengths]


def normalize_window(value: Optional[int]) -> Optional[int]:
    """把 window 配置统一成 int 或 None。"""
    if value is None:
        return None

    if isinstance(value, str):
        if value.lower() in {"none", "null", ""}:
            return None
        return int(value)

    return int(value)


def normalize_active_modalities(
    active_modalities: Optional[Sequence[str]] = None,
) -> Tuple[str, ...]:
    """
    标准化 active_modalities。

    输入可以是：
        None
        ["text", "audio"]
        ("audio", "text")

    输出固定按照 text/audio/visual 顺序：
        ("text", "audio")
    """
    if active_modalities is None:
        return FULL_MODALITIES

    valid = set(FULL_MODALITIES)
    normalized = []

    for name in active_modalities:
        value = str(name).strip().lower()

        if value == "":
            continue

        if value not in valid:
            raise ValueError(
                f"Unknown modality={name!r}. "
                "Supported modalities are: text, audio, visual."
            )

        if value not in normalized:
            normalized.append(value)

    if len(normalized) == 0:
        raise ValueError("active_modalities cannot be empty.")

    return tuple(name for name in FULL_MODALITIES if name in normalized)


def active_modalities_to_node_mask(
    active_modalities: Optional[Sequence[str]] = None,
) -> Tuple[bool, bool, bool]:
    """
    转成图节点顺序 [audio, visual, text] 的 bool mask。

    返回：
        audio_active, visual_active, text_active
    """
    active = set(normalize_active_modalities(active_modalities))
    return tuple(name in active for name in MODALITY_ORDER)  # type: ignore[return-value]


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
    """判断 source_time 是否允许向 target_time 传信息。"""
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
        sim = 1 - acos(cosine) / pi

    这里 clamp 是为了避免 acos 的数值越界。
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
    active_modalities: Optional[Sequence[str]] = None,
) -> Tuple[torch.Tensor, torch.Tensor, List[Tuple[int, int]]]:
    """
    Build official-aligned multimodal adjacency.

    Args:
        audio_nodes: [N, H]
        visual_nodes: [N, H]
        text_nodes: [N, H]
        lengths: [B]
        max_len: padded max dialogue length
        active_modalities:
            None 或 ["text", "audio", "visual"] 表示完整三模态。
            例如 ["text", "audio"] 表示 visual 节点保留但不参与建边。

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

    # 图内部顺序固定为 [audio, visual, text]。
    modal_features = [audio_nodes, visual_nodes, text_nodes]
    modal_active = active_modalities_to_node_mask(active_modalities)

    modal_count = 3
    num_nodes = modal_count * num_utterances

    if num_utterances == 0:
        empty_adj = torch.zeros((0, 0), dtype=dtype, device=device)
        return empty_adj, global_index, positions

    adj = torch.zeros(
        (num_nodes, num_nodes),
        dtype=dtype,
        device=device,
    )

    length_list = lengths_to_list(lengths)
    start = 0

    for length in length_list:
        dialogue_length = min(int(length), int(max_len))
        end = start + dialogue_length

        if dialogue_length <= 0:
            start = end
            continue

        # 1. 同模态边。
        for modal_index, features in enumerate(modal_features):
            if not modal_active[modal_index]:
                continue

            sub_features = features[start:end]
            sub_adj = pairwise_mm_similarity(sub_features)

            for target_time in range(dialogue_length):
                target_global = start + target_time
                target_node = modal_index * num_utterances + target_global

                for source_time in range(dialogue_length):
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

        # 2. 跨模态边，只连接同一句 utterance 的不同模态。
        for target_modal in range(modal_count):
            if not modal_active[target_modal]:
                continue

            for source_modal in range(modal_count):
                if target_modal == source_modal:
                    continue

                if not modal_active[source_modal]:
                    continue

                target_features = modal_features[target_modal][start:end]
                source_features = modal_features[source_modal][start:end]
                diag_sim = paired_mm_similarity(target_features, source_features)

                for time_index in range(dialogue_length):
                    utterance_global = start + time_index
                    target_node = target_modal * num_utterances + utterance_global
                    source_node = source_modal * num_utterances + utterance_global
                    adj[target_node, source_node] = gamma * diag_sim[time_index]

        start = end

    # 3. D^{-1/2} A D^{-1/2}
    # inactive 节点 degree=0，所以这里必须 clamp，避免 NaN。
    degree = adj.sum(dim=1).clamp_min(1e-8)
    degree_inv_sqrt = torch.pow(degree, -0.5)
    adj = degree_inv_sqrt.view(-1, 1) * adj * degree_inv_sqrt.view(1, -1)

    return adj, global_index, positions


def adjacency_density(adj: torch.Tensor) -> float:
    """返回非零边密度，用于 debug 和日志。"""
    if adj.numel() == 0:
        return 0.0

    return float((adj > 0).float().mean().item())

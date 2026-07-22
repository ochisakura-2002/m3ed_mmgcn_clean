"""
Dialogue-level official-aligned MMGCN baseline.

这个文件实现当前项目使用的 MMGCN 适配版本。

输入是项目统一的 padded dialogue batch：
    text_features:   [B, T, D_text]
    audio_features:  [B, T, D_audio]
    visual_features: [B, T, D_visual]

核心机制保留官方 MMGCN 主路径：
    1. 三模态分别线性投影到 hidden_dim；
    2. speaker embedding 只加到 text/language 分支；
    3. modal embedding 加到 audio/visual/text 分支；
    4. 每个 utterance-modality pair 作为图节点；
    5. 使用 similarity-weighted multimodal adjacency；
    6. 使用 GCNII-style 图传播；
    7. 拼接 [audio, visual, text] 图后特征做 utterance-level 分类。

新增能力：
    active_modalities 可选控制哪些模态参与计算。
    不写 active_modalities 时默认完整三模态，尽量保持旧行为。
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from .dense_graph import (
    adjacency_density,
    active_modalities_to_node_mask,
    build_official_like_multimodal_adjacency,
    normalize_active_modalities,
)


FULL_MODALITIES: Tuple[str, str, str] = ("text", "audio", "visual")


class GraphConvolution(nn.Module):
    """
    Official MMGCN / GCNII-style graph convolution.

    公式对应官方 GraphConvolution 主逻辑：
        theta = log(lamda / layer + 1)
        hi = A @ input
        support = (1 - alpha) * hi + alpha * h0
        output = theta * support @ W + (1 - theta) * support
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        residual: bool = False,
        variant: bool = False,
    ) -> None:
        super().__init__()

        self.variant = bool(variant)
        self.in_features = int(in_features) * 2 if self.variant else int(in_features)
        self.out_features = int(out_features)
        self.residual = bool(residual)

        self.weight = Parameter(
            torch.empty(self.in_features, self.out_features)
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        stdv = 1.0 / math.sqrt(self.out_features)
        self.weight.data.uniform_(-stdv, stdv)

    def forward(
        self,
        input_features: torch.Tensor,
        adj: torch.Tensor,
        h0: torch.Tensor,
        lamda: float,
        alpha: float,
        layer_index: int,
    ) -> torch.Tensor:
        theta = math.log(float(lamda) / int(layer_index) + 1.0)

        hi = torch.matmul(adj, input_features)

        if self.variant:
            support = torch.cat([hi, h0], dim=1)
            residual_mix = (1.0 - float(alpha)) * hi + float(alpha) * h0
        else:
            support = (1.0 - float(alpha)) * hi + float(alpha) * h0
            residual_mix = support

        output = (
            theta * torch.matmul(support, self.weight)
            + (1.0 - theta) * residual_mix
        )

        if self.residual:
            output = output + input_features

        return output


class GCNIIBackbone(nn.Module):
    """
    Official-style GCNII backbone adapted to dense adjacency.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        lamda: float,
        alpha: float,
        variant: bool = False,
        use_residual: bool = False,
    ) -> None:
        super().__init__()

        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.dropout = float(dropout)
        self.lamda = float(lamda)
        self.alpha = float(alpha)
        self.variant = bool(variant)
        self.use_residual = bool(use_residual)

        self.convs = nn.ModuleList(
            [
                GraphConvolution(
                    in_features=self.hidden_dim,
                    out_features=self.hidden_dim,
                    residual=self.use_residual,
                    variant=self.variant,
                )
                for _ in range(self.num_layers)
            ]
        )

        self.input_fc = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.act_fn = nn.ReLU()

    def forward(
        self,
        x: torch.Tensor,
        adj: torch.Tensor,
    ) -> torch.Tensor:
        layer_inner = F.dropout(
            x,
            self.dropout,
            training=self.training,
        )
        layer_inner = self.act_fn(self.input_fc(layer_inner))
        h0 = layer_inner

        for layer_index, conv in enumerate(self.convs, start=1):
            layer_inner = F.dropout(
                layer_inner,
                self.dropout,
                training=self.training,
            )
            layer_inner = conv(
                input_features=layer_inner,
                adj=adj,
                h0=h0,
                lamda=self.lamda,
                alpha=self.alpha,
                layer_index=layer_index,
            )
            layer_inner = self.act_fn(layer_inner)

        return layer_inner


class M3EDMMGCN(nn.Module):
    """
    Official-aligned dialogue-level MMGCN adapter.

    类名保留 M3EDMMGCN 是历史原因。
    只要输入符合统一 batch 接口，它也可以用于 IEMOCAP。
    """

    def __init__(
        self,
        text_dim: int,
        audio_dim: int,
        visual_dim: int,
        hidden_dim: int,
        num_classes: int,
        num_layers: int = 2,
        dropout: float = 0.3,
        lamda: float = 0.5,
        alpha: float = 0.1,
        gamma: float = 0.7,
        use_speaker: bool = True,
        use_modal: bool = True,
        use_residual: bool = False,
        context_mode: str = "full",
        window_past: Optional[int] = None,
        window_future: Optional[int] = None,
        active_modalities: Optional[Sequence[str]] = None,
    ) -> None:
        super().__init__()

        self.text_dim = int(text_dim)
        self.audio_dim = int(audio_dim)
        self.visual_dim = int(visual_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_classes = int(num_classes)
        self.num_layers = int(num_layers)

        self.dropout_value = float(dropout)
        self.lamda = float(lamda)
        self.alpha = float(alpha)
        self.gamma = float(gamma)

        self.use_speaker = bool(use_speaker)
        self.use_modal = bool(use_modal)
        self.use_residual = bool(use_residual)

        self.context_mode = str(context_mode)
        self.window_past = window_past
        self.window_future = window_future

        # active_modalities 只是 Python 属性，不进入 state_dict。
        # 这样旧 checkpoint 仍然可以 strict=True 加载。
        self.active_modalities = normalize_active_modalities(active_modalities)

        # Official MMGCN uses modality-specific Linear layers.
        self.audio_fc = nn.Linear(self.audio_dim, self.hidden_dim)
        self.visual_fc = nn.Linear(self.visual_dim, self.hidden_dim)
        self.text_fc = nn.Linear(self.text_dim, self.hidden_dim)

        self.modal_embeddings = nn.Embedding(3, self.hidden_dim)

        # M3ED/IEMOCAP 都是 dyadic speaker 为主。
        # 留 16 个槽位是为了避免预处理里 speaker id 超出 {0, 1} 时直接崩。
        self.speaker_embeddings = nn.Embedding(16, self.hidden_dim)

        self.graph_net = GCNIIBackbone(
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=self.dropout_value,
            lamda=self.lamda,
            alpha=self.alpha,
            variant=False,
            use_residual=self.use_residual,
        )

        # 固定 3H，inactive 模态用零向量占位。
        # 这样 checkpoint 结构不变，三模态训练 checkpoint 也能用于测试时缺失模态评估。
        self.final_fc = nn.Linear(self.hidden_dim * 3, self.num_classes)

    @staticmethod
    def _is_full_modalities(active_modalities: Sequence[str]) -> bool:
        """判断当前是否为完整三模态。"""
        return tuple(active_modalities) == FULL_MODALITIES

    def _make_hidden_zeros_like(
        self,
        reference: torch.Tensor,
    ) -> torch.Tensor:
        """构造 [B, T, H] 的零 hidden。"""
        batch_size, max_len = reference.shape[0], reference.shape[1]
        return reference.new_zeros((batch_size, max_len, self.hidden_dim))

    def _build_node_active_mask(
        self,
        num_utterances: int,
        active_modalities: Sequence[str],
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """
        构造图节点级 mask。

        图节点顺序固定为 [audio block, visual block, text block]。
        返回形状：
            [3N, 1]
        """
        audio_active, visual_active, text_active = active_modalities_to_node_mask(
            active_modalities
        )

        values = [
            1.0 if audio_active else 0.0,
            1.0 if visual_active else 0.0,
            1.0 if text_active else 0.0,
        ]

        blocks = [
            torch.full(
                (num_utterances, 1),
                fill_value=value,
                dtype=dtype,
                device=device,
            )
            for value in values
        ]

        return torch.cat(blocks, dim=0)

    def _gather_valid_utterance_features(
        self,
        features: torch.Tensor,
        positions: List[Tuple[int, int]],
    ) -> torch.Tensor:
        if len(positions) == 0:
            return features.new_zeros((0, self.hidden_dim))

        gathered = [
            features[batch_index, time_index]
            for batch_index, time_index in positions
        ]

        return torch.stack(gathered, dim=0)

    def _project_modalities(
        self,
        text_features: torch.Tensor,
        audio_features: torch.Tensor,
        visual_features: torch.Tensor,
        speaker_ids_int: Optional[torch.Tensor],
        active_modalities: Sequence[str],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Project raw features to hidden dimension.

        Internal modality order follows official MMGCN:
            audio, visual, text

        full 三模态时走旧逻辑。
        非 full 时，inactive 模态直接置零，避免 fc bias/modal embedding 把缺失模态复活。
        """
        active = set(active_modalities)

        if self._is_full_modalities(active_modalities):
            audio_h = self.audio_fc(audio_features)
            visual_h = self.visual_fc(visual_features)
            text_h = self.text_fc(text_features)

            if self.use_speaker and speaker_ids_int is not None:
                speaker_ids = speaker_ids_int.long().clamp(
                    min=0,
                    max=self.speaker_embeddings.num_embeddings - 1,
                )
                speaker_h = self.speaker_embeddings(speaker_ids)

                # Official MM_GCN adds speaker embedding only to language branch.
                text_h = text_h + speaker_h

            if self.use_modal:
                modal_ids = torch.tensor(
                    [0, 1, 2],
                    dtype=torch.long,
                    device=text_features.device,
                )
                modal_h = self.modal_embeddings(modal_ids)

                audio_h = audio_h + modal_h[0].view(1, 1, -1)
                visual_h = visual_h + modal_h[1].view(1, 1, -1)
                text_h = text_h + modal_h[2].view(1, 1, -1)

            return audio_h, visual_h, text_h

        # 非 full 模式：inactive 模态不经过 Linear，不吃 bias，不加 embedding。
        if "audio" in active:
            audio_h = self.audio_fc(audio_features)
        else:
            audio_h = self._make_hidden_zeros_like(text_features)

        if "visual" in active:
            visual_h = self.visual_fc(visual_features)
        else:
            visual_h = self._make_hidden_zeros_like(text_features)

        if "text" in active:
            text_h = self.text_fc(text_features)
        else:
            text_h = self._make_hidden_zeros_like(text_features)

        if "text" in active and self.use_speaker and speaker_ids_int is not None:
            speaker_ids = speaker_ids_int.long().clamp(
                min=0,
                max=self.speaker_embeddings.num_embeddings - 1,
            )
            speaker_h = self.speaker_embeddings(speaker_ids)
            text_h = text_h + speaker_h

        if self.use_modal:
            modal_ids = torch.tensor(
                [0, 1, 2],
                dtype=torch.long,
                device=text_features.device,
            )
            modal_h = self.modal_embeddings(modal_ids)

            if "audio" in active:
                audio_h = audio_h + modal_h[0].view(1, 1, -1)
            if "visual" in active:
                visual_h = visual_h + modal_h[1].view(1, 1, -1)
            if "text" in active:
                text_h = text_h + modal_h[2].view(1, 1, -1)

        return audio_h, visual_h, text_h

    def forward(
        self,
        text_features: torch.Tensor,
        audio_features: torch.Tensor,
        visual_features: torch.Tensor,
        lengths: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        speaker_ids_int: Optional[torch.Tensor] = None,
        return_graph: bool = False,
        active_modalities: Optional[Sequence[str]] = None,
    ) -> Dict[str, torch.Tensor]:
        if text_features.dim() != 3:
            raise ValueError(
                f"text_features must be [B, T, D], got {tuple(text_features.shape)}"
            )

        active_modalities = (
            self.active_modalities
            if active_modalities is None
            else normalize_active_modalities(active_modalities)
        )
        is_full_modalities = self._is_full_modalities(active_modalities)

        batch_size, max_len, _ = text_features.shape
        device = text_features.device

        audio_h, visual_h, text_h = self._project_modalities(
            text_features=text_features,
            audio_features=audio_features,
            visual_features=visual_features,
            speaker_ids_int=speaker_ids_int,
            active_modalities=active_modalities,
        )

        # 先用图工具得到有效 utterance 位置。
        # 后面再 gather compact node features。
        from .dense_graph import build_utterance_index

        _, positions = build_utterance_index(
            lengths=lengths,
            max_len=max_len,
            device=device,
        )

        num_utterances = len(positions)

        logits = text_features.new_zeros(
            (batch_size, max_len, self.num_classes)
        )

        if num_utterances == 0:
            output: Dict[str, torch.Tensor] = {
                "logits": logits,
                "logits_flat": logits.new_zeros((0, self.num_classes)),
            }

            if return_graph:
                output.update(
                    {
                        "adjacency": logits.new_zeros((0, 0)),
                        "node_features": logits.new_zeros((0, self.hidden_dim)),
                        "utterance_features": logits.new_zeros(
                            (0, self.hidden_dim * 3)
                        ),
                    }
                )

            return output

        audio_nodes = self._gather_valid_utterance_features(audio_h, positions)
        visual_nodes = self._gather_valid_utterance_features(visual_h, positions)
        text_nodes = self._gather_valid_utterance_features(text_h, positions)

        graph_active_modalities = None if is_full_modalities else active_modalities

        adj, global_index, positions = build_official_like_multimodal_adjacency(
            audio_nodes=audio_nodes,
            visual_nodes=visual_nodes,
            text_nodes=text_nodes,
            lengths=lengths,
            max_len=max_len,
            context_mode=self.context_mode,
            window_past=self.window_past,
            window_future=self.window_future,
            gamma=self.gamma,
            active_modalities=graph_active_modalities,
        )

        # Official node order: [a, v, l].
        x = torch.cat(
            [
                audio_nodes,
                visual_nodes,
                text_nodes,
            ],
            dim=0,
        )

        node_active_mask: Optional[torch.Tensor] = None

        if not is_full_modalities:
            node_active_mask = self._build_node_active_mask(
                num_utterances=num_utterances,
                active_modalities=active_modalities,
                device=device,
                dtype=x.dtype,
            )

            # 进入 GCN 前先置零一次。
            x = x * node_active_mask

        graph_features = self.graph_net(
            x=x,
            adj=adj,
        )

        if node_active_mask is not None:
            # GCNII 的 input_fc bias、h0 injection、residual 都可能让 inactive 节点重新非零。
            # 所以图传播后必须再置零一次。是的，神经网络很会诈尸。
            graph_features = graph_features * node_active_mask

        audio_out = graph_features[0:num_utterances]
        visual_out = graph_features[num_utterances : 2 * num_utterances]
        text_out = graph_features[2 * num_utterances : 3 * num_utterances]

        # Official MM_GCN concatenates [a, v, l] after graph propagation.
        utterance_features = torch.cat(
            [
                audio_out,
                visual_out,
                text_out,
            ],
            dim=-1,
        )

        logits_flat = self.final_fc(utterance_features)

        for utterance_index, (batch_index, time_index) in enumerate(positions):
            logits[batch_index, time_index] = logits_flat[utterance_index]

        output = {
            "logits": logits,
            "logits_flat": logits_flat,
            "utterance_features": utterance_features,
        }

        if return_graph:
            graph_output = {
                "adjacency": adj,
                "global_index": global_index,
                "node_features": graph_features,
                "node_features_initial": x,
                "num_utterance_nodes": torch.tensor(
                    num_utterances,
                    device=device,
                    dtype=torch.long,
                ),
                "num_graph_nodes": torch.tensor(
                    graph_features.shape[0],
                    device=device,
                    dtype=torch.long,
                ),
                "adjacency_density": torch.tensor(
                    adjacency_density(adj),
                    device=device,
                    dtype=torch.float32,
                ),
            }

            if node_active_mask is not None:
                graph_output["node_active_mask"] = node_active_mask

            output.update(graph_output)

        return output

"""
M3ED-compatible official-aligned MMGCN baseline.

This module adapts the official MMGCN implementation to the local M3ED
dialogue-batch training interface.

It is not a byte-for-byte copy of the official code, because the local training
script passes padded dialogue tensors:
    text_features:   [B, T, D_text]
    audio_features:  [B, T, D_audio]
    visual_features: [B, T, D_visual]

The model preserves the main official mechanisms:
    1. Modality-specific linear projection.
    2. Speaker embedding added only to language/text branch.
    3. Modal embedding added to audio/visual/text branches.
    4. Similarity-weighted multimodal adjacency.
    5. GCNII-style propagation.
    6. Concatenation of audio, visual, text graph features for classification.

Output:
    {
        "logits": [B, T, num_classes],
        "logits_flat": [N, num_classes],
        ...
    }
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from .dense_graph import (
    adjacency_density,
    build_official_like_multimodal_adjacency,
)


class GraphConvolution(nn.Module):
    """
    Official MMGCN / GCNII-style graph convolution.

    Formula follows the official GraphConvolution:
        theta = log(lamda / layer + 1)
        hi = A @ input
        support = (1 - alpha) * hi + alpha * h0
        output = theta * support @ W + (1 - theta) * support

    If variant=True, support is concat([hi, h0]) and r remains the mixed state.
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

        output = theta * torch.matmul(support, self.weight) + (1.0 - theta) * residual_mix

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
    Official-aligned MMGCN adapter for local M3ED features.
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

        # Official MMGCN uses modality-specific Linear layers.
        self.audio_fc = nn.Linear(self.audio_dim, self.hidden_dim)
        self.visual_fc = nn.Linear(self.visual_dim, self.hidden_dim)
        self.text_fc = nn.Linear(self.text_dim, self.hidden_dim)

        self.modal_embeddings = nn.Embedding(3, self.hidden_dim)

        # M3ED is basically dyadic, but a slightly larger table avoids crashes
        # if preprocessing uses ids beyond {0, 1}.
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

        self.final_fc = nn.Linear(self.hidden_dim * 3, self.num_classes)

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
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Project raw features to hidden dimension.

        Internal modality order follows official MMGCN:
            audio, visual, text
        """
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

    def forward(
        self,
        text_features: torch.Tensor,
        audio_features: torch.Tensor,
        visual_features: torch.Tensor,
        lengths: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        speaker_ids_int: Optional[torch.Tensor] = None,
        return_graph: bool = False,
    ) -> Dict[str, torch.Tensor]:
        if text_features.dim() != 3:
            raise ValueError(
                f"text_features must be [B, T, D], got {tuple(text_features.shape)}"
            )

        batch_size, max_len, _ = text_features.shape
        device = text_features.device

        audio_h, visual_h, text_h = self._project_modalities(
            text_features=text_features,
            audio_features=audio_features,
            visual_features=visual_features,
            speaker_ids_int=speaker_ids_int,
        )

        # First get valid positions using graph utility. We pass placeholder
        # compact nodes later after positions are known.
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
                        "utterance_features": logits.new_zeros((0, self.hidden_dim * 3)),
                    }
                )

            return output

        audio_nodes = self._gather_valid_utterance_features(audio_h, positions)
        visual_nodes = self._gather_valid_utterance_features(visual_h, positions)
        text_nodes = self._gather_valid_utterance_features(text_h, positions)

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

        graph_features = self.graph_net(
            x=x,
            adj=adj,
        )

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
            output.update(
                {
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
            )

        return output
"""Project reproduction of the original full-context MMGCN."""

from __future__ import annotations

from typing import Any, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.baselines.mmgcn.dense_graph import (
    adjacency_density,
    build_official_like_multimodal_adjacency,
    build_utterance_index,
)
from models.baselines.mmgcn.mm_gcn import GCNIIBackbone
from models.baselines.original_repro.common import (
    class_weight_tensor,
    flatten_valid,
    masked_cross_entropy,
    run_packed_rnn,
    scatter_valid,
    structured_output,
    validate_dialogue_batch,
)


class OriginalReproMMGCN(nn.Module):
    """MMGCN with the official text BiLSTM and full multimodal graph.

    The class is intentionally isolated from :class:`M3EDMMGCN`.  The latter
    retains the project's causal mode, while this class always represents an
    offline, full-context paper reproduction.
    """

    model_key = "original_repro_mmgcn"
    causal_grade = "noncausal_offline_full_context"

    def __init__(
        self,
        text_feature_dim: int,
        audio_feature_dim: int,
        visual_feature_dim: int,
        num_classes: int = 6,
        hidden_dim: int = 200,
        graph_layers: int = 4,
        dropout: float = 0.4,
        lamda: float = 0.5,
        alpha: float = 0.1,
        gamma: float = 1.0,
        num_speakers: int = 2,
        use_speaker: bool = True,
        use_modal: bool = False,
        use_residual: bool = True,
        class_weight: Optional[Sequence[float]] = None,
    ) -> None:
        super().__init__()
        if hidden_dim <= 0 or hidden_dim % 2:
            raise ValueError("hidden_dim must be a positive even integer for the text BiLSTM.")
        if graph_layers <= 0:
            raise ValueError("graph_layers must be positive.")
        self.num_classes = int(num_classes)
        self.hidden_dim = int(hidden_dim)
        self.gamma = float(gamma)
        self.use_speaker = bool(use_speaker)
        self.use_modal = bool(use_modal)

        self.text_projection = nn.Linear(int(text_feature_dim), self.hidden_dim)
        self.text_context_encoder = nn.LSTM(
            input_size=self.hidden_dim,
            hidden_size=self.hidden_dim // 2,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=float(dropout),
        )
        self.audio_encoder = nn.Linear(int(audio_feature_dim), self.hidden_dim)
        self.visual_encoder = nn.Linear(int(visual_feature_dim), self.hidden_dim)
        self.speaker_embeddings = nn.Embedding(int(num_speakers), self.hidden_dim)
        self.modal_embeddings = nn.Embedding(3, self.hidden_dim)
        self.graph_net = GCNIIBackbone(
            hidden_dim=self.hidden_dim,
            num_layers=int(graph_layers),
            dropout=float(dropout),
            lamda=float(lamda),
            alpha=float(alpha),
            variant=True,
            use_residual=bool(use_residual),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(float(dropout)),
            nn.ReLU(),
            nn.Linear(self.hidden_dim * 3, self.num_classes),
        )
        weight = class_weight_tensor(class_weight, self.num_classes)
        self.register_buffer("class_weight", weight, persistent=weight is not None)

    def forward(
        self,
        text_features: torch.Tensor,
        audio_features: torch.Tensor,
        visual_features: torch.Tensor,
        attention_mask: torch.Tensor,
        lengths: torch.Tensor,
        speaker_ids_int: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        **_: Any,
    ) -> dict[str, Any]:
        validate_dialogue_batch(
            text_features,
            audio_features,
            visual_features,
            attention_mask,
            lengths,
            speaker_ids_int,
        )
        text_projected = self.text_projection(text_features)
        text_context = run_packed_rnn(self.text_context_encoder, text_projected, lengths)
        audio_encoded = self.audio_encoder(audio_features)
        visual_encoded = self.visual_encoder(visual_features)

        if self.use_speaker:
            speaker = self.speaker_embeddings(
                speaker_ids_int.long().clamp(0, self.speaker_embeddings.num_embeddings - 1)
            )
            # This follows the released effective path: speaker identity is
            # injected into the language branch before graph construction.
            text_context = text_context + speaker
        if self.use_modal:
            modal = self.modal_embeddings(torch.arange(3, device=text_features.device))
            audio_encoded = audio_encoded + modal[0]
            visual_encoded = visual_encoded + modal[1]
            text_context = text_context + modal[2]

        audio_nodes = flatten_valid(audio_encoded, attention_mask)
        visual_nodes = flatten_valid(visual_encoded, attention_mask)
        text_nodes = flatten_valid(text_context, attention_mask)
        adjacency, global_index, positions = build_official_like_multimodal_adjacency(
            audio_nodes=audio_nodes,
            visual_nodes=visual_nodes,
            text_nodes=text_nodes,
            lengths=lengths,
            max_len=text_features.shape[1],
            context_mode="full",
            window_past=None,
            window_future=None,
            gamma=self.gamma,
        )
        initial_nodes = torch.cat([audio_nodes, visual_nodes, text_nodes], dim=0)
        graph_nodes = self.graph_net(initial_nodes, adjacency)
        utterance_count = audio_nodes.shape[0]
        fusion = torch.cat(
            [
                graph_nodes[:utterance_count],
                graph_nodes[utterance_count : 2 * utterance_count],
                graph_nodes[2 * utterance_count :],
            ],
            dim=-1,
        )
        logits_valid = self.classifier(fusion)
        logits = scatter_valid(logits_valid, attention_mask, self.num_classes)
        classification_loss = masked_cross_entropy(
            logits, labels, attention_mask, self.class_weight
        )
        return structured_output(
            logits=logits,
            classification_loss=classification_loss,
            features={
                "modality_encoder_output": {
                    "text": text_context,
                    "audio": audio_encoded,
                    "visual": visual_encoded,
                },
                "context_output": text_context,
                "graph_output": graph_nodes,
                "fusion_output": fusion,
                "logits": logits,
                "base_classification_loss": classification_loss,
                "auxiliary_losses": {},
            },
            diagnostics={
                "model_key": self.model_key,
                "full_context": True,
                "num_graph_nodes": int(graph_nodes.shape[0]),
                "adjacency_density": adjacency_density(adjacency),
                "adjacency": adjacency,
                "global_index": global_index,
                "positions": positions,
            },
        )


__all__ = ["OriginalReproMMGCN"]

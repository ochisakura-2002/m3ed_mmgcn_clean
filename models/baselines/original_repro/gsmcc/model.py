"""Paper-oriented GS-MCC implementation.

The released GS-MCC repository does not expose the paper's explicit low/high
frequency dual branches or its cross-frequency contrastive objective.  This
implementation follows the paper description for those mechanisms while
keeping the released feature encoders and sliding multimodal graph as the
closest code references.  The resolution is recorded in the source-audit
documentation rather than silently treating paper and code as identical.
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

from ..common import (
    CAUSAL_GRADE,
    class_weight_tensor,
    masked_cross_entropy,
    run_packed_rnn,
    structured_output,
    validate_dialogue_batch,
)


def angular_similarity(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Non-negative angular similarity used by the multimodal graph."""

    cosine = F.cosine_similarity(left, right, dim=-1, eps=1e-8).clamp(-1.0, 1.0)
    return 1.0 - torch.acos(cosine) / math.pi


def build_sliding_multimodal_graph(
    modality_features: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    valid_mask: torch.Tensor,
    window: int,
    cross_modal_scale: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a normalized 3-modality graph with local temporal edges.

    Nodes are ordered ``audio, visual, text``.  Same-modality nodes connect
    inside the symmetric temporal window, and the three modalities for one
    utterance form a fully connected cross-modal block.
    """

    if window < 0:
        raise ValueError("GS-MCC sliding window must be non-negative")
    audio, visual, text = modality_features
    batch_size, max_length, _ = text.shape
    nodes = torch.cat((audio, visual, text), dim=1)
    node_mask = valid_mask.repeat(1, 3)
    adjacency = nodes.new_zeros((batch_size, 3 * max_length, 3 * max_length))

    for batch_index in range(batch_size):
        length = int(valid_mask[batch_index].sum().item())
        for modality_index in range(3):
            offset = modality_index * max_length
            features = modality_features[modality_index][batch_index]
            for target in range(length):
                start = max(0, target - window)
                stop = min(length, target + window + 1)
                targets = features[target].expand(stop - start, -1)
                weights = angular_similarity(targets, features[start:stop])
                adjacency[batch_index, offset + target, offset + start : offset + stop] = weights

        for utterance in range(length):
            for left_modality in range(3):
                left_node = left_modality * max_length + utterance
                for right_modality in range(3):
                    if left_modality == right_modality:
                        continue
                    right_node = right_modality * max_length + utterance
                    weight = angular_similarity(
                        modality_features[left_modality][batch_index, utterance].unsqueeze(0),
                        modality_features[right_modality][batch_index, utterance].unsqueeze(0),
                    ).squeeze(0)
                    adjacency[batch_index, left_node, right_node] = cross_modal_scale * weight

    adjacency = 0.5 * (adjacency + adjacency.transpose(1, 2))
    adjacency = adjacency * node_mask.unsqueeze(1) * node_mask.unsqueeze(2)
    degree = adjacency.sum(dim=-1).clamp_min(1e-8)
    inv_sqrt_degree = degree.rsqrt()
    normalized = inv_sqrt_degree.unsqueeze(-1) * adjacency * inv_sqrt_degree.unsqueeze(-2)
    return normalized, node_mask


class FourierGraphOperator(nn.Module):
    """One learnable graph-frequency filtering block.

    The graph low/high operators separate smooth and residual components.  A
    real FFT along the node axis then applies learnable complex channel gains,
    which makes the implementation an actual Fourier graph branch rather than
    a relabelled polynomial graph convolution.
    """

    def __init__(self, hidden_dim: int, frequency: str, dropout: float) -> None:
        super().__init__()
        if frequency not in {"low", "high"}:
            raise ValueError(f"unsupported frequency branch: {frequency}")
        self.frequency = frequency
        self.real_gain = nn.Parameter(torch.ones(hidden_dim))
        self.imag_gain = nn.Parameter(torch.zeros(hidden_dim))
        self.mix = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        features: torch.Tensor,
        adjacency: torch.Tensor,
        node_mask: torch.Tensor,
    ) -> torch.Tensor:
        identity_component = features
        propagated = torch.bmm(adjacency, features)
        if self.frequency == "low":
            filtered = identity_component + propagated
        else:
            filtered = identity_component - propagated

        spectrum = torch.fft.rfft(filtered, dim=1, norm="ortho")
        gain = torch.complex(self.real_gain, self.imag_gain).view(1, 1, -1)
        transformed = torch.fft.irfft(
            spectrum * gain,
            n=filtered.shape[1],
            dim=1,
            norm="ortho",
        )
        update = F.gelu(self.mix(transformed))
        result = self.norm(features + self.dropout(update))
        return result * node_mask.unsqueeze(-1)


def cross_frequency_contrastive_loss(
    low_features: torch.Tensor,
    high_features: torch.Tensor,
    node_mask: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """Contrast each branch against the other branch without augmentation.

    Following the paper's no-augmentation construction, a node is its own
    positive within a frequency branch while all valid nodes in the opposite
    branch are negatives.  The two frequency directions are averaged.
    """

    if temperature <= 0:
        raise ValueError("contrastive temperature must be positive")
    valid = node_mask.bool()
    low = F.normalize(low_features[valid], dim=-1)
    high = F.normalize(high_features[valid], dim=-1)
    if low.numel() == 0:
        return low_features.sum() * 0.0

    positive = torch.ones(low.shape[0], device=low.device, dtype=low.dtype) / temperature
    negative_logits = torch.matmul(low, high.transpose(0, 1)) / temperature
    low_denominator = torch.logsumexp(torch.cat((positive.unsqueeze(1), negative_logits), dim=1), dim=1)
    high_denominator = torch.logsumexp(
        torch.cat((positive.unsqueeze(1), negative_logits.transpose(0, 1)), dim=1),
        dim=1,
    )
    return 0.5 * ((low_denominator - positive).mean() + (high_denominator - positive).mean())


class ProjectPaperOrientedGSMCC(nn.Module):
    """Project variant inspired by GS-MCC; not a paper reproduction."""

    model_key = "project_paper_oriented_gsmcc"
    causal_grade = CAUSAL_GRADE
    fidelity_status = "PROJECT_VARIANT_NOT_PAPER_REPRODUCTION"

    def __init__(
        self,
        text_feature_dim: int,
        audio_feature_dim: int,
        visual_feature_dim: int,
        num_classes: int = 6,
        hidden_dim: int = 100,
        num_speakers: int = 2,
        graph_layers: int = 4,
        window: int = 10,
        dropout: float = 0.4,
        cross_modal_scale: float = 1.0,
        use_contrastive_loss: bool = True,
        contrastive_temperature: float = 0.1,
        contrastive_loss_weight: float = 1.0,
        class_weight: list[float] | None = None,
    ) -> None:
        super().__init__()
        if graph_layers < 1:
            raise ValueError("graph_layers must be at least one")
        if hidden_dim <= 0 or hidden_dim % 2:
            raise ValueError("hidden_dim must be a positive even integer")
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.window = window
        self.cross_modal_scale = cross_modal_scale
        self.use_contrastive_loss = use_contrastive_loss
        self.contrastive_temperature = contrastive_temperature
        self.contrastive_loss_weight = contrastive_loss_weight

        self.text_input = nn.Linear(text_feature_dim, hidden_dim)
        self.text_encoder = nn.GRU(
            hidden_dim,
            hidden_dim // 2,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=dropout,
        )
        self.audio_input = nn.Linear(audio_feature_dim, hidden_dim)
        self.audio_encoder = nn.GRU(
            hidden_dim,
            hidden_dim // 2,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=dropout,
        )
        self.visual_input = nn.Linear(visual_feature_dim, hidden_dim)
        self.visual_encoder = nn.GRU(
            hidden_dim,
            hidden_dim // 2,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=dropout,
        )
        self.speaker_embedding = nn.Embedding(num_speakers, hidden_dim)
        self.low_layers = nn.ModuleList(
            FourierGraphOperator(hidden_dim, "low", dropout) for _ in range(graph_layers)
        )
        self.high_layers = nn.ModuleList(
            FourierGraphOperator(hidden_dim, "high", dropout) for _ in range(graph_layers)
        )
        self.low_projector = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.high_projector = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.classifier = nn.Sequential(
            nn.Linear(6 * hidden_dim, 2 * hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * hidden_dim, num_classes),
        )
        weight = class_weight_tensor(class_weight, num_classes)
        self.register_buffer("class_weight", weight, persistent=weight is not None)

    def forward(
        self,
        text_features: torch.Tensor,
        audio_features: torch.Tensor,
        visual_features: torch.Tensor,
        attention_mask: torch.Tensor,
        lengths: torch.Tensor,
        speaker_ids_int: torch.Tensor,
        labels: torch.Tensor | None = None,
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
        valid_mask = attention_mask.bool()
        speaker_ids = speaker_ids_int.long().clamp(0, self.speaker_embedding.num_embeddings - 1)
        speaker = self.speaker_embedding(speaker_ids)

        text_input = F.gelu(self.text_input(text_features)) + speaker
        text = run_packed_rnn(self.text_encoder, text_input, lengths)
        audio_input = F.gelu(self.audio_input(audio_features)) + speaker
        visual_input = F.gelu(self.visual_input(visual_features)) + speaker
        audio = run_packed_rnn(self.audio_encoder, audio_input, lengths)
        visual = run_packed_rnn(self.visual_encoder, visual_input, lengths)
        mask = valid_mask.unsqueeze(-1)
        text, audio, visual = text * mask, audio * mask, visual * mask

        adjacency, node_mask = build_sliding_multimodal_graph(
            (audio, visual, text),
            valid_mask,
            self.window,
            self.cross_modal_scale,
        )
        nodes = torch.cat((audio, visual, text), dim=1)
        low, high = nodes, nodes
        for low_layer, high_layer in zip(self.low_layers, self.high_layers):
            low = low_layer(low, adjacency, node_mask)
            high = high_layer(high, adjacency, node_mask)

        max_length = valid_mask.shape[1]
        modality_fusion: list[torch.Tensor] = []
        for modality_index in range(3):
            start = modality_index * max_length
            stop = start + max_length
            modality_fusion.extend((low[:, start:stop], high[:, start:stop]))
        fused = torch.cat(modality_fusion, dim=-1)
        logits = self.classifier(fused) * mask
        classification_loss = masked_cross_entropy(
            logits,
            labels,
            attention_mask,
            self.class_weight,
        )

        auxiliary_losses: dict[str, torch.Tensor] = {}
        raw_contrastive = logits.sum() * 0.0
        if self.use_contrastive_loss:
            raw_contrastive = cross_frequency_contrastive_loss(
                self.low_projector(low),
                self.high_projector(high),
                node_mask,
                self.contrastive_temperature,
            )
            auxiliary_losses["contrastive_loss"] = self.contrastive_loss_weight * raw_contrastive

        output = structured_output(
            logits=logits,
            classification_loss=classification_loss,
            aux_losses=auxiliary_losses,
            features={
                "modality_encoder_output": {"text": text, "audio": audio, "visual": visual},
                "context_output": nodes,
                "low_frequency": low,
                "high_frequency": high,
                "graph_output": {"low_frequency": low, "high_frequency": high},
                "fusion_output": fused,
                "logits": logits,
                "base_classification_loss": classification_loss,
                "auxiliary_losses": auxiliary_losses,
            },
            diagnostics={
                "adjacency": adjacency,
                "node_mask": node_mask,
                "window": self.window,
                "contrastive_enabled": self.use_contrastive_loss,
                "contrastive_temperature": self.contrastive_temperature,
                "raw_contrastive_loss": raw_contrastive.detach(),
                "fidelity_status": self.fidelity_status,
                "paper_reproduction_eligible": False,
            },
        )
        return output


__all__ = [
    "FourierGraphOperator",
    "ProjectPaperOrientedGSMCC",
    "angular_similarity",
    "build_sliding_multimodal_graph",
    "cross_frequency_contrastive_loss",
]

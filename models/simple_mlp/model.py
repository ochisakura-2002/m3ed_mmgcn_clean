"""
Simple multimodal MLP baseline for M3ED-style dialogue features.

Input:
    text_features:   [B, T, text_dim]
    audio_features:  [B, T, audio_dim]
    visual_features: [B, T, visual_dim]

Output:
    {
        "logits": [B, T, num_classes],
        "features": [B, T, hidden_dim],
    }
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn


class M3EDConcatMLP(nn.Module):
    """
    Utterance-level multimodal concatenation MLP.

    This model does not use dialogue context. It receives a full dialogue-shaped
    batch only because the dataset and collator are dialogue-based.
    """

    def __init__(
        self,
        text_dim: int,
        audio_dim: int,
        visual_dim: int,
        hidden_dim: int,
        num_classes: int,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()

        input_dim = int(text_dim) + int(audio_dim) + int(visual_dim)

        self.feature_mlp = nn.Sequential(
            nn.Linear(input_dim, int(hidden_dim)),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
        )

        self.classifier = nn.Linear(int(hidden_dim), int(num_classes))

    def forward(
        self,
        text_features: torch.Tensor,
        audio_features: torch.Tensor,
        visual_features: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        if text_features.dim() != 3:
            raise ValueError(
                f"text_features must be [B, T, D], got {tuple(text_features.shape)}"
            )

        x = torch.cat(
            [
                text_features,
                audio_features,
                visual_features,
            ],
            dim=-1,
        )

        features = self.feature_mlp(x)
        logits = self.classifier(features)

        return {
            "logits": logits,
            "features": features,
        }
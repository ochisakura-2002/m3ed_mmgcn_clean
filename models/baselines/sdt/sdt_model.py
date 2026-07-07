"""Device-neutral SDT baseline port for project dialogue batches.

This module keeps the project-facing contract batch-first:

    text_features:   [B, T, D_text]
    audio_features:  [B, T, D_audio]
    visual_features: [B, T, D_visual]
    attention_mask:  [B, T]
    speaker_ids_int: [B, T]
    labels:          [B, T]

It is not a line-by-line copy of the official SDT implementation. The port keeps
the main modeling ingredients: transformer-based intra/cross-modal contextual
encoding, gated fusion, modality-specific auxiliary heads, and optional
self-distillation from the fused prediction to unimodal predictions.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class _CrossModalTransformerLayer(nn.Module):
    """One batch-first transformer layer for self- or cross-modal attention."""

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attention_dropout = nn.Dropout(dropout)
        self.attention_norm = nn.LayerNorm(hidden_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.feed_forward_dropout = nn.Dropout(dropout)
        self.feed_forward_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        query: torch.Tensor,
        source: torch.Tensor,
        key_padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        # MultiheadAttention returns NaN if every key in a row is masked. That
        # should not occur for real dialogues, but this keeps the layer robust.
        attention_padding_mask = key_padding_mask
        if key_padding_mask.any() and key_padding_mask.all(dim=1).any():
            attention_padding_mask = key_padding_mask.clone()
            attention_padding_mask[attention_padding_mask.all(dim=1)] = False

        context, _ = self.attention(
            query=query,
            key=source,
            value=source,
            key_padding_mask=attention_padding_mask,
            need_weights=False,
        )
        x = self.attention_norm(query + self.attention_dropout(context))
        ff = self.feed_forward(x)
        x = self.feed_forward_norm(x + self.feed_forward_dropout(ff))
        return x.masked_fill(key_padding_mask.unsqueeze(-1), 0.0)


class _UnimodalGatedFusion(nn.Module):
    """Feature-wise gate used on each transformer branch."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.gate = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.gate(x)) * x


class _MultimodalGatedFusion(nn.Module):
    """Fuse text/audio/visual contextual states with feature-wise modality weights."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.score = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(
        self,
        text_context: torch.Tensor,
        audio_context: torch.Tensor,
        visual_context: torch.Tensor,
    ) -> torch.Tensor:
        stacked = torch.stack(
            [text_context, audio_context, visual_context],
            dim=-2,
        )
        weights = torch.softmax(self.score(stacked), dim=-2)
        return torch.sum(weights * stacked, dim=-2)


def _sinusoidal_position_encoding(
    length: int,
    hidden_dim: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    position = torch.arange(length, device=device, dtype=dtype).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, hidden_dim, 2, device=device, dtype=dtype)
        * -(math.log(10000.0) / hidden_dim)
    )
    encoding = torch.zeros(1, length, hidden_dim, device=device, dtype=dtype)
    encoding[0, :, 0::2] = torch.sin(position * div_term)
    if hidden_dim > 1:
        encoding[0, :, 1::2] = torch.cos(
            position * div_term[: encoding[0, :, 1::2].shape[-1]]
        )
    return encoding


class SDTBaseline(nn.Module):
    """Project-shaped SDT baseline for padded dialogue feature tensors."""

    def __init__(
        self,
        text_dim: int,
        audio_dim: int,
        visual_dim: int,
        hidden_dim: int,
        num_classes: int,
        num_speakers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
        temperature: float = 1.0,
        use_self_distillation: bool = True,
        distillation_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.text_dim = int(text_dim)
        self.audio_dim = int(audio_dim)
        self.visual_dim = int(visual_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_classes = int(num_classes)
        self.num_speakers = int(num_speakers)
        self.num_heads = int(num_heads)
        self.temperature = float(temperature)
        self.use_self_distillation = bool(use_self_distillation)
        self.distillation_weight = float(distillation_weight)

        if self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if self.num_heads <= 0 or self.hidden_dim % self.num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        if self.num_classes <= 0:
            raise ValueError("num_classes must be positive")
        if self.num_speakers <= 0:
            raise ValueError("num_speakers must be positive")
        if self.temperature <= 0.0:
            raise ValueError("temperature must be positive")

        self.padding_speaker_idx = self.num_speakers
        self.speaker_embeddings = nn.Embedding(
            self.num_speakers + 1,
            self.hidden_dim,
            padding_idx=self.padding_speaker_idx,
        )

        self.text_projection = nn.Linear(self.text_dim, self.hidden_dim, bias=False)
        self.audio_projection = nn.Linear(self.audio_dim, self.hidden_dim, bias=False)
        self.visual_projection = nn.Linear(self.visual_dim, self.hidden_dim, bias=False)
        self.input_dropout = nn.Dropout(dropout)

        self.context_layers = nn.ModuleDict(
            {
                "tt": _CrossModalTransformerLayer(self.hidden_dim, self.num_heads, dropout),
                "at": _CrossModalTransformerLayer(self.hidden_dim, self.num_heads, dropout),
                "vt": _CrossModalTransformerLayer(self.hidden_dim, self.num_heads, dropout),
                "aa": _CrossModalTransformerLayer(self.hidden_dim, self.num_heads, dropout),
                "ta": _CrossModalTransformerLayer(self.hidden_dim, self.num_heads, dropout),
                "va": _CrossModalTransformerLayer(self.hidden_dim, self.num_heads, dropout),
                "vv": _CrossModalTransformerLayer(self.hidden_dim, self.num_heads, dropout),
                "tv": _CrossModalTransformerLayer(self.hidden_dim, self.num_heads, dropout),
                "av": _CrossModalTransformerLayer(self.hidden_dim, self.num_heads, dropout),
            }
        )
        self.unimodal_gates = nn.ModuleDict(
            {
                key: _UnimodalGatedFusion(self.hidden_dim)
                for key in self.context_layers.keys()
            }
        )

        self.text_reduce = nn.Linear(3 * self.hidden_dim, self.hidden_dim)
        self.audio_reduce = nn.Linear(3 * self.hidden_dim, self.hidden_dim)
        self.visual_reduce = nn.Linear(3 * self.hidden_dim, self.hidden_dim)
        self.multimodal_fusion = _MultimodalGatedFusion(self.hidden_dim)

        self.text_classifier = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, self.num_classes),
        )
        self.audio_classifier = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, self.num_classes),
        )
        self.visual_classifier = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, self.num_classes),
        )
        self.fused_classifier = nn.Linear(self.hidden_dim, self.num_classes)

    def forward(
        self,
        text_features: torch.Tensor,
        audio_features: torch.Tensor,
        visual_features: torch.Tensor,
        attention_mask: torch.Tensor,
        speaker_ids_int: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """Run SDT forward and optionally compute masked CE/distillation loss."""

        self._validate_inputs(
            text_features=text_features,
            audio_features=audio_features,
            visual_features=visual_features,
            attention_mask=attention_mask,
            speaker_ids_int=speaker_ids_int,
            labels=labels,
        )

        mask_bool = attention_mask.to(dtype=torch.bool)
        key_padding_mask = ~mask_bool
        speaker_idx = self._speaker_indices(
            attention_mask=mask_bool,
            speaker_ids_int=speaker_ids_int,
        )

        text_h = self.text_projection(text_features)
        audio_h = self.audio_projection(audio_features)
        visual_h = self.visual_projection(visual_features)

        position = _sinusoidal_position_encoding(
            length=text_h.size(1),
            hidden_dim=self.hidden_dim,
            device=text_h.device,
            dtype=text_h.dtype,
        )
        speaker_emb = self.speaker_embeddings(speaker_idx).to(dtype=text_h.dtype)
        valid_scale = mask_bool.unsqueeze(-1).to(dtype=text_h.dtype)

        text_h = self.input_dropout(text_h + position + speaker_emb) * valid_scale
        audio_h = self.input_dropout(audio_h + position + speaker_emb) * valid_scale
        visual_h = self.input_dropout(visual_h + position + speaker_emb) * valid_scale

        tt = self._branch("tt", text_h, text_h, key_padding_mask)
        at = self._branch("at", text_h, audio_h, key_padding_mask)
        vt = self._branch("vt", text_h, visual_h, key_padding_mask)
        aa = self._branch("aa", audio_h, audio_h, key_padding_mask)
        ta = self._branch("ta", audio_h, text_h, key_padding_mask)
        va = self._branch("va", audio_h, visual_h, key_padding_mask)
        vv = self._branch("vv", visual_h, visual_h, key_padding_mask)
        tv = self._branch("tv", visual_h, text_h, key_padding_mask)
        av = self._branch("av", visual_h, audio_h, key_padding_mask)

        text_context = self.text_reduce(torch.cat([tt, at, vt], dim=-1)) * valid_scale
        audio_context = self.audio_reduce(torch.cat([aa, ta, va], dim=-1)) * valid_scale
        visual_context = self.visual_reduce(torch.cat([vv, tv, av], dim=-1)) * valid_scale
        fused_context = self.multimodal_fusion(
            text_context,
            audio_context,
            visual_context,
        ) * valid_scale

        text_logits = self.text_classifier(text_context)
        audio_logits = self.audio_classifier(audio_context)
        visual_logits = self.visual_classifier(visual_context)
        logits = self.fused_classifier(fused_context)

        loss = None
        aux_losses: Dict[str, torch.Tensor] = {}
        if labels is not None:
            ce_loss = self._masked_cross_entropy(logits, labels, mask_bool)
            aux_losses["ce"] = ce_loss
            loss = ce_loss

            if self.use_self_distillation and self.distillation_weight != 0.0:
                distillation_loss = torch.stack(
                    [
                        self._masked_kl_divergence(text_logits, logits, mask_bool),
                        self._masked_kl_divergence(audio_logits, logits, mask_bool),
                        self._masked_kl_divergence(visual_logits, logits, mask_bool),
                    ]
                ).mean()
                aux_losses["distillation"] = distillation_loss
                loss = loss + self.distillation_weight * distillation_loss

            aux_losses["total"] = loss

        return {
            "logits": logits,
            "loss": loss,
            "aux_losses": aux_losses,
            "features": fused_context,
            "modality_logits": {
                "text": text_logits,
                "audio": audio_logits,
                "visual": visual_logits,
            },
        }

    def _branch(
        self,
        key: str,
        query: torch.Tensor,
        source: torch.Tensor,
        key_padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        output = self.context_layers[key](
            query=query,
            source=source,
            key_padding_mask=key_padding_mask,
        )
        return self.unimodal_gates[key](output)

    def _speaker_indices(
        self,
        attention_mask: torch.Tensor,
        speaker_ids_int: Optional[torch.Tensor],
    ) -> torch.Tensor:
        batch_size, seq_len = attention_mask.shape
        if speaker_ids_int is None:
            speaker_ids_int = torch.zeros(
                batch_size,
                seq_len,
                device=attention_mask.device,
                dtype=torch.long,
            )
        else:
            speaker_ids_int = speaker_ids_int.to(
                device=attention_mask.device,
                dtype=torch.long,
            )

        padding = torch.full_like(speaker_ids_int, self.padding_speaker_idx)
        speaker_ids_int = speaker_ids_int.clamp(0, self.num_speakers - 1)
        return torch.where(attention_mask, speaker_ids_int, padding)

    def _masked_cross_entropy(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        labels = labels.to(device=logits.device, dtype=torch.long)
        valid = attention_mask.to(device=logits.device) & (labels >= 0)
        if not valid.any():
            return logits.sum() * 0.0
        return F.cross_entropy(logits[valid], labels[valid])

    def _masked_kl_divergence(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        valid = attention_mask.to(device=student_logits.device)
        if not valid.any():
            return student_logits.sum() * 0.0

        temperature = self.temperature
        student_log_prob = F.log_softmax(
            student_logits[valid] / temperature,
            dim=-1,
        )
        teacher_prob = F.softmax(
            teacher_logits.detach()[valid] / temperature,
            dim=-1,
        )
        return F.kl_div(
            student_log_prob,
            teacher_prob,
            reduction="batchmean",
        ) * (temperature * temperature)

    def _validate_inputs(
        self,
        text_features: torch.Tensor,
        audio_features: torch.Tensor,
        visual_features: torch.Tensor,
        attention_mask: torch.Tensor,
        speaker_ids_int: Optional[torch.Tensor],
        labels: Optional[torch.Tensor],
    ) -> None:
        if text_features.dim() != 3:
            raise ValueError(
                f"text_features must be [B, T, D], got {tuple(text_features.shape)}"
            )
        if audio_features.dim() != 3:
            raise ValueError(
                f"audio_features must be [B, T, D], got {tuple(audio_features.shape)}"
            )
        if visual_features.dim() != 3:
            raise ValueError(
                f"visual_features must be [B, T, D], got {tuple(visual_features.shape)}"
            )

        batch_size, seq_len, text_dim = text_features.shape
        audio_shape = audio_features.shape
        visual_shape = visual_features.shape
        if audio_shape[:2] != (batch_size, seq_len):
            raise ValueError("audio_features must share B and T with text_features")
        if visual_shape[:2] != (batch_size, seq_len):
            raise ValueError("visual_features must share B and T with text_features")
        if text_dim != self.text_dim:
            raise ValueError(f"expected text_dim={self.text_dim}, got {text_dim}")
        if audio_shape[-1] != self.audio_dim:
            raise ValueError(f"expected audio_dim={self.audio_dim}, got {audio_shape[-1]}")
        if visual_shape[-1] != self.visual_dim:
            raise ValueError(
                f"expected visual_dim={self.visual_dim}, got {visual_shape[-1]}"
            )

        if attention_mask.shape != (batch_size, seq_len):
            raise ValueError(
                "attention_mask must be [B, T], "
                f"got {tuple(attention_mask.shape)}"
            )
        if speaker_ids_int is not None and speaker_ids_int.shape != (batch_size, seq_len):
            raise ValueError(
                "speaker_ids_int must be [B, T], "
                f"got {tuple(speaker_ids_int.shape)}"
            )
        if labels is not None and labels.shape != (batch_size, seq_len):
            raise ValueError(f"labels must be [B, T], got {tuple(labels.shape)}")

        devices = {
            text_features.device,
            audio_features.device,
            visual_features.device,
            attention_mask.device,
        }
        if speaker_ids_int is not None:
            devices.add(speaker_ids_int.device)
        if labels is not None:
            devices.add(labels.device)
        if len(devices) != 1:
            raise ValueError("all input tensors must be on the same device")

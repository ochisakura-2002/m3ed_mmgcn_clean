"""Independently authored encoders for the two frozen conformance profiles."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from .config import ConformanceProfile, EncoderProfile, MultiDAGCLConfig
from .contracts import ContextVisibilityIdentity, EncodedModalities


def _validate_masked_modalities(
    config: MultiDAGCLConfig,
    text_features: torch.Tensor,
    audio_features: torch.Tensor,
    visual_features: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    for name, value, expected_dim in (
        ("text_features", text_features, config.text_feature_dim),
        ("audio_features", audio_features, config.audio_feature_dim),
        ("visual_features", visual_features, config.visual_feature_dim),
    ):
        if value.dtype is not torch.float32 or value.dim() != 3:
            raise TypeError(f"{name} must be float32 [B,T,D]")
        if value.shape[-1] != expected_dim:
            raise ValueError(f"{name} has the wrong configured feature dimension")
        if not torch.isfinite(value).all():
            raise ValueError(f"{name} must be finite")
    batch_time = text_features.shape[:2]
    if audio_features.shape[:2] != batch_time or visual_features.shape[:2] != batch_time:
        raise ValueError("modalities must share [B,T]")
    if attention_mask.dtype not in (torch.int64, torch.bool) or attention_mask.shape != batch_time:
        raise TypeError("attention_mask must be int64 or bool [B,T]")
    devices = {
        text_features.device,
        audio_features.device,
        visual_features.device,
        attention_mask.device,
    }
    if len(devices) != 1:
        raise ValueError("modalities and attention_mask must share a device")
    if attention_mask.dtype is torch.int64 and not torch.all(
        (attention_mask == 0) | (attention_mask == 1)
    ):
        raise ValueError("attention_mask may contain only 0/1")
    mask = attention_mask.bool()
    padded = ~mask
    for name, value in (
        ("text_features", text_features),
        ("audio_features", audio_features),
        ("visual_features", visual_features),
    ):
        if torch.count_nonzero(value[padded]).item() != 0:
            raise ValueError(f"{name} padded rows must be exact zero")
    return mask


class PaperFormulaModalityEncoder(nn.Module):
    """Paper-formula A/V/T encoder with an explicit dialogue-axis text choice."""

    def __init__(self, config: MultiDAGCLConfig) -> None:
        super().__init__()
        if not isinstance(config, MultiDAGCLConfig):
            raise TypeError("config must be MultiDAGCLConfig")
        if config.conformance_profile is not ConformanceProfile.PAPER_FORMULA_BEHAVIOR:
            raise ValueError("PaperFormulaModalityEncoder requires paper_formula_behavior")
        if config.encoder_profile is not EncoderProfile.PAPER_MODALITY_SPECIFIC:
            raise ValueError("PaperFormulaModalityEncoder requires paper_modality_specific")
        self.config = config
        self.audio_projection = nn.Linear(config.audio_feature_dim, config.audio_output_dim)
        self.visual_projection = nn.Linear(config.visual_feature_dim, config.visual_output_dim)
        self.text_projection = nn.Linear(config.text_feature_dim, config.text_output_dim)
        text_hidden = config.text_output_dim // 2 if config.text_bidirectional else config.text_output_dim
        self.text_encoder = nn.LSTM(
            input_size=config.text_output_dim,
            hidden_size=text_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=config.text_bidirectional,
        )

    def forward(
        self,
        *,
        text_features: torch.Tensor,
        audio_features: torch.Tensor,
        visual_features: torch.Tensor,
        lengths: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> EncodedModalities:
        mask = _validate_masked_modalities(
            self.config,
            text_features,
            audio_features,
            visual_features,
            attention_mask,
        )
        batch_size, max_time = mask.shape
        if lengths.dtype is not torch.int64 or lengths.shape != (batch_size,):
            raise TypeError("lengths must be int64 [B]")
        if lengths.device != attention_mask.device:
            raise ValueError("lengths and attention_mask must share a device")
        if torch.any(lengths < 1) or torch.any(lengths > max_time):
            raise ValueError("lengths must lie in [1,T]")
        if not torch.equal(mask.long().sum(dim=1), lengths):
            raise ValueError("lengths must equal attention_mask sums")
        positions = torch.arange(max_time, device=lengths.device).unsqueeze(0)
        if not torch.equal(mask, positions < lengths.unsqueeze(1)):
            raise ValueError("attention_mask must be a contiguous valid prefix")

        audio = F.relu(self.audio_projection(audio_features))
        visual = F.relu(self.visual_projection(visual_features))
        projected_text = self.text_projection(text_features)
        packed = pack_padded_sequence(
            projected_text,
            lengths.detach().cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        packed_text, _ = self.text_encoder(packed)
        text, _ = pad_packed_sequence(
            packed_text,
            batch_first=True,
            total_length=max_time,
        )
        mask_values = mask.unsqueeze(-1)
        audio = audio * mask_values.to(audio.dtype)
        visual = visual * mask_values.to(visual.dtype)
        text = text * mask_values.to(text.dtype)
        fused = torch.cat((audio, visual, text), dim=-1)
        fused = fused * mask_values.to(fused.dtype)

        if self.config.causal_text_ablation:
            context_identity = ContextVisibilityIdentity(
                dag_topology_causal=True,
                end_to_end_causal=False,
                end_to_end_causal_assuming_local_text_features=True,
                upstream_text_feature_locality="not_evaluated_by_model_core",
                context_leakage_risk=("unverified_upstream_text_feature_context",),
            )
            deviations = ("causal_unidirectional_text_encoder_ablation",)
        else:
            context_identity = ContextVisibilityIdentity(
                dag_topology_causal=True,
                end_to_end_causal=False,
                end_to_end_causal_assuming_local_text_features=False,
                upstream_text_feature_locality="not_evaluated_by_model_core",
                context_leakage_risk=("paper_dialogue_axis_text_bilstm",),
            )
            deviations = ()
        return EncodedModalities(
            audio=audio,
            visual=visual,
            text=text,
            fused=fused,
            encoder_profile=self.config.encoder_profile.value,
            modality_order=("audio", "visual", "text"),
            context_visibility_identity=context_identity,
            model_math_deviation_list=deviations,
        )


class OfficialSourceInputAdapter:
    """Named T/A/V packer for the isolated released-source behavior profile."""

    def __init__(self, config: MultiDAGCLConfig) -> None:
        if not isinstance(config, MultiDAGCLConfig):
            raise TypeError("config must be MultiDAGCLConfig")
        if config.conformance_profile is not ConformanceProfile.OFFICIAL_SOURCE_BEHAVIOR:
            raise ValueError("source input adapter requires official_source_behavior")
        self.config = config

    def pack(
        self,
        *,
        text_features: torch.Tensor,
        audio_features: torch.Tensor,
        visual_features: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        mask = _validate_masked_modalities(
            self.config,
            text_features,
            audio_features,
            visual_features,
            attention_mask,
        )
        raw_concat = torch.cat((text_features, audio_features, visual_features), dim=-1)
        return raw_concat * mask.unsqueeze(-1).to(raw_concat.dtype)


class OfficialSourceSingleProjectionEncoder(nn.Module):
    """One position-wise 300-wide projection for source-behavior conformance."""

    def __init__(self, config: MultiDAGCLConfig) -> None:
        super().__init__()
        if not isinstance(config, MultiDAGCLConfig):
            raise TypeError("config must be MultiDAGCLConfig")
        if config.conformance_profile is not ConformanceProfile.OFFICIAL_SOURCE_BEHAVIOR:
            raise ValueError("source projection encoder requires official_source_behavior")
        self.config = config
        raw_dim = config.text_feature_dim + config.audio_feature_dim + config.visual_feature_dim
        self.projection = nn.Linear(raw_dim, config.hidden_dim)

    def forward(self, packed: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        expected_dim = (
            self.config.text_feature_dim
            + self.config.audio_feature_dim
            + self.config.visual_feature_dim
        )
        if packed.dtype is not torch.float32 or packed.dim() != 3:
            raise TypeError("packed must be float32 [B,T,D_sum]")
        if packed.shape[-1] != expected_dim:
            raise ValueError("packed source input has the wrong feature dimension")
        if attention_mask.dtype not in (torch.int64, torch.bool) or attention_mask.shape != packed.shape[:2]:
            raise TypeError("attention_mask must be int64 or bool [B,T]")
        if attention_mask.device != packed.device:
            raise ValueError("packed and attention_mask must share a device")
        if not torch.isfinite(packed).all():
            raise ValueError("packed source input must be finite")
        mask = attention_mask.bool()
        if torch.count_nonzero(packed[~mask]).item() != 0:
            raise ValueError("packed padded rows must be exact zero")
        encoded = F.relu(self.projection(packed))
        return encoded * mask.unsqueeze(-1).to(encoded.dtype)


__all__ = [
    "OfficialSourceInputAdapter",
    "OfficialSourceSingleProjectionEncoder",
    "PaperFormulaModalityEncoder",
]

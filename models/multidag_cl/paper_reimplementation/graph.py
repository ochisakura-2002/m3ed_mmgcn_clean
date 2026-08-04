"""Past-only predecessor topology and semantic speaker relations."""

from __future__ import annotations

import torch

from .config import PredecessorProfile
from .contracts import SpeakerRelations


def _validated_graph_inputs(
    speaker_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    if not isinstance(speaker_ids, torch.Tensor) or not isinstance(attention_mask, torch.Tensor):
        raise TypeError("speaker_ids and attention_mask must be tensors")
    if speaker_ids.dtype is not torch.int64 or speaker_ids.dim() != 2:
        raise TypeError("speaker_ids must be int64 [B,T]")
    if attention_mask.dtype not in (torch.int64, torch.bool) or attention_mask.shape != speaker_ids.shape:
        raise TypeError("attention_mask must be int64 or bool [B,T]")
    if speaker_ids.device != attention_mask.device:
        raise ValueError("speaker_ids and attention_mask must share a device")
    if attention_mask.dtype is torch.int64 and not torch.all(
        (attention_mask == 0) | (attention_mask == 1)
    ):
        raise ValueError("attention_mask may contain only 0/1")
    mask = attention_mask.bool()
    if torch.any(mask.sum(dim=1) < 1):
        raise ValueError("every dialogue must have at least one valid utterance")
    max_time = mask.shape[1]
    lengths = mask.long().sum(dim=1)
    positions = torch.arange(max_time, device=mask.device).unsqueeze(0)
    if not torch.equal(mask, positions < lengths.unsqueeze(1)):
        raise ValueError("attention_mask must be a contiguous valid prefix")
    if torch.any(speaker_ids[mask] < 0):
        raise ValueError("valid speaker IDs must be non-negative")
    return mask


class CausalPredecessorBuilder:
    """Build receiver/source adjacency by the frozen backward speaker scan."""

    def __init__(
        self,
        window_past_same_speaker: int = 1,
        profile: PredecessorProfile = PredecessorProfile.OFFICIAL_SAME_SPEAKER_COUNT_WINDOW,
    ) -> None:
        if isinstance(window_past_same_speaker, bool) or not isinstance(
            window_past_same_speaker, int
        ):
            raise TypeError("window_past_same_speaker must be int")
        if window_past_same_speaker < 1:
            raise ValueError("window_past_same_speaker must be at least 1")
        try:
            self.profile = (
                profile
                if isinstance(profile, PredecessorProfile)
                else PredecessorProfile(profile)
            )
        except (TypeError, ValueError) as error:
            raise ValueError(f"unknown predecessor profile: {profile!r}") from error
        self.window_past_same_speaker = window_past_same_speaker

    def build(
        self,
        speaker_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        mask = _validated_graph_inputs(speaker_ids, attention_mask)
        batch_size, max_time = speaker_ids.shape
        adjacency = torch.zeros(
            (batch_size, max_time, max_time),
            dtype=torch.bool,
            device=speaker_ids.device,
        )
        lengths = mask.long().sum(dim=1).detach().cpu().tolist()
        for batch_index, length_value in enumerate(lengths):
            length = int(length_value)
            for target in range(length):
                target_speaker = int(speaker_ids[batch_index, target].item())
                same_count = 0
                for source in range(target - 1, -1, -1):
                    adjacency[batch_index, target, source] = True
                    if int(speaker_ids[batch_index, source].item()) == target_speaker:
                        same_count += 1
                        if same_count == self.window_past_same_speaker:
                            break
        return adjacency


class SpeakerRelationBuilder:
    """Build named same/different-speaker metadata over valid pairs."""

    @staticmethod
    def build(
        speaker_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> SpeakerRelations:
        mask = _validated_graph_inputs(speaker_ids, attention_mask)
        valid_pair = mask.unsqueeze(2) & mask.unsqueeze(1)
        equality = speaker_ids.unsqueeze(2) == speaker_ids.unsqueeze(1)
        same_speaker = valid_pair & equality
        different_speaker = valid_pair & ~equality
        return SpeakerRelations(
            same_speaker=same_speaker,
            different_speaker=different_speaker,
        )


__all__ = ["CausalPredecessorBuilder", "SpeakerRelationBuilder"]

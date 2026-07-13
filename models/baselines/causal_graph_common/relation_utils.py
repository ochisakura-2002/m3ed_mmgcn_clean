"""Speaker-relation utilities for causal DialogueGCN-derived graphs."""

from __future__ import annotations

from typing import Dict, Tuple

import torch


def build_speaker_pair_relation_ids(
    speaker_ids_int: torch.Tensor,
    adjacency: torch.Tensor,
    num_speakers: int,
) -> Tuple[torch.Tensor, Dict[int, str]]:
    """Encode ``source speaker -> target speaker`` on legal utterance edges."""

    if num_speakers <= 0:
        raise ValueError("num_speakers must be positive")
    if speaker_ids_int.dim() != 2:
        raise ValueError("speaker_ids_int must have shape [B,T]")
    if adjacency.dim() != 3 or adjacency.shape != (
        speaker_ids_int.shape[0],
        speaker_ids_int.shape[1],
        speaker_ids_int.shape[1],
    ):
        raise ValueError("adjacency must have shape [B,T,T]")
    if speaker_ids_int.device != adjacency.device:
        raise ValueError("speaker_ids_int and adjacency must share a device")

    speakers = speaker_ids_int.to(dtype=torch.long)
    active_speakers = adjacency.any(dim=1) | adjacency.any(dim=2)
    invalid = active_speakers & ((speakers < 0) | (speakers >= num_speakers))
    if torch.any(invalid):
        raise ValueError("valid graph nodes contain an out-of-range speaker id")

    target_speaker = speakers.unsqueeze(2)
    source_speaker = speakers.unsqueeze(1)
    relation_ids = source_speaker * num_speakers + target_speaker
    relation_ids = relation_ids.masked_fill(~adjacency.to(dtype=torch.bool), -1)
    mapping = {
        source * num_speakers + target: f"source_{source}->target_{target}"
        for source in range(num_speakers)
        for target in range(num_speakers)
    }
    return relation_ids, mapping

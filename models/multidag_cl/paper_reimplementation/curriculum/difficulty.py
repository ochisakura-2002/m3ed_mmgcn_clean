"""Pure deterministic implementation of the dialogue difficulty measure."""

from __future__ import annotations

from typing import Sequence


class DialogueDifficultyScorer:
    """Score one training dialogue from valid gold labels and speakers only."""

    @staticmethod
    def score(labels: Sequence[int], speakers: Sequence[int]) -> float:
        if not isinstance(labels, Sequence) or isinstance(labels, (str, bytes)):
            raise TypeError("labels must be a sequence of valid gold integer labels")
        if not isinstance(speakers, Sequence) or isinstance(speakers, (str, bytes)):
            raise TypeError("speakers must be a sequence of valid integer speaker IDs")
        if len(labels) != len(speakers):
            raise ValueError("labels and speakers must have equal length")
        if len(labels) == 0:
            raise ValueError("a dialogue must contain at least one utterance")
        for name, values in (("labels", labels), ("speakers", speakers)):
            if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
                raise TypeError(f"{name} must contain only integers")
            if any(value < 0 for value in values):
                raise ValueError(f"{name} must not contain padding or negative values")

        histories: dict[int, list[int]] = {}
        for label, speaker in zip(labels, speakers):
            histories.setdefault(speaker, []).append(label)
        shift_count = sum(
            left != right
            for history in histories.values()
            for left, right in zip(history, history[1:])
        )
        speaker_count = len(histories)
        numerator = shift_count + speaker_count
        denominator = len(labels) + speaker_count
        return float(numerator) / float(denominator)


__all__ = ["DialogueDifficultyScorer"]

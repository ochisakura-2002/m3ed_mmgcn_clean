"""Leakage-safe IEMOCAP splits for the original MERC reproductions.

The legacy IEMOCAP feature pickle stores Ses01--Ses04 in ``trainVid`` and
Ses05 in ``testVid``. The official-split track preserves that boundary, while
the two fair five-fold tracks combine the lists only to rotate the outer test
session.
"""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Any, Mapping, Sequence

import numpy as np

from .split_utils import IEMOCAP_SESSION_IDS, parse_iemocap_session_id


@dataclass(frozen=True)
class OuterSessionSplit:
    """Resolved dialogue IDs for one nested IEMOCAP fold."""

    outer_test_session: str
    train_dialogue_ids: tuple[str, ...]
    inner_val_dialogue_ids: tuple[str, ...]
    test_dialogue_ids: tuple[str, ...]
    split_seed: int
    inner_val_ratio: float
    test_split_used_for_selection: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "outer_test_session": self.outer_test_session,
            "train_dialogue_ids": list(self.train_dialogue_ids),
            "inner_val_dialogue_ids": list(self.inner_val_dialogue_ids),
            "test_dialogue_ids": list(self.test_dialogue_ids),
            "split_seed": self.split_seed,
            "inner_val_ratio": self.inner_val_ratio,
            "test_split_used_for_selection": self.test_split_used_for_selection,
        }


@dataclass(frozen=True)
class OfficialTrainSplit:
    """Original ``trainVid``/``testVid`` with safe inner validation."""

    train_dialogue_ids: tuple[str, ...]
    inner_val_dialogue_ids: tuple[str, ...]
    test_dialogue_ids: tuple[str, ...]
    split_seed: int
    inner_val_ratio: float
    test_split_used_for_selection: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "train_dialogue_ids": list(self.train_dialogue_ids),
            "inner_val_dialogue_ids": list(self.inner_val_dialogue_ids),
            "test_dialogue_ids": list(self.test_dialogue_ids),
            "split_seed": self.split_seed,
            "inner_val_ratio": self.inner_val_ratio,
            "test_split_used_for_selection": self.test_split_used_for_selection,
        }


def _dialogue_profile(
    dialogue_id: str,
    labels_by_dialogue: Mapping[str, Sequence[int]],
    speakers_by_dialogue: Mapping[str, Sequence[Any]],
    num_labels: int,
    speaker_values: Sequence[str],
) -> np.ndarray:
    labels = [int(value) for value in labels_by_dialogue[dialogue_id]]
    speakers = [str(value) for value in speakers_by_dialogue[dialogue_id]]
    label_counts = np.bincount(labels, minlength=num_labels).astype(np.float64)
    speaker_counts = np.asarray(
        [sum(value == speaker for value in speakers) for speaker in speaker_values],
        dtype=np.float64,
    )
    return np.concatenate([label_counts, speaker_counts, [float(len(labels))]])


def _stratified_validation_ids(
    candidate_ids: Sequence[str],
    labels_by_dialogue: Mapping[str, Sequence[int]],
    speakers_by_dialogue: Mapping[str, Sequence[Any]],
    ratio: float,
    seed: int,
) -> list[str]:
    """Greedily approximate label/speaker counts at dialogue granularity.

    IEMOCAP dialogues are multi-label groups, so ordinary single-label
    stratification is not applicable.  This deterministic greedy selection
    targets the aggregate label, speaker, and utterance-count profile while
    never splitting a dialogue.
    """

    if not 0.0 < float(ratio) < 1.0:
        raise ValueError("inner_val_ratio must be in (0, 1).")
    if len(candidate_ids) < 2:
        raise ValueError("At least two non-test dialogues are required.")

    n_val = max(1, int(round(len(candidate_ids) * float(ratio))))
    n_val = min(n_val, len(candidate_ids) - 1)
    max_label = max(
        int(label)
        for dialogue_id in candidate_ids
        for label in labels_by_dialogue[dialogue_id]
    )
    num_labels = max_label + 1
    speaker_values = sorted(
        {
            str(speaker)
            for dialogue_id in candidate_ids
            for speaker in speakers_by_dialogue[dialogue_id]
        }
    )
    profiles = {
        dialogue_id: _dialogue_profile(
            dialogue_id,
            labels_by_dialogue,
            speakers_by_dialogue,
            num_labels,
            speaker_values,
        )
        for dialogue_id in candidate_ids
    }
    target = sum(profiles.values()) * (n_val / len(candidate_ids))
    scale = np.maximum(target, 1.0)

    rng = random.Random(int(seed))
    tie_order = list(candidate_ids)
    rng.shuffle(tie_order)
    tie_rank = {dialogue_id: index for index, dialogue_id in enumerate(tie_order)}

    selected: list[str] = []
    selected_profile = np.zeros_like(target)
    remaining = set(candidate_ids)
    while len(selected) < n_val:
        best_id = min(
            remaining,
            key=lambda dialogue_id: (
                float(
                    np.square(
                        (selected_profile + profiles[dialogue_id] - target) / scale
                    ).sum()
                ),
                tie_rank[dialogue_id],
                dialogue_id,
            ),
        )
        selected.append(best_id)
        selected_profile += profiles[best_id]
        remaining.remove(best_id)
    return sorted(selected)


def build_outer_session_split(
    dialogue_ids: Sequence[str],
    labels_by_dialogue: Mapping[str, Sequence[int]],
    speakers_by_dialogue: Mapping[str, Sequence[Any]],
    outer_test_session: str,
    inner_val_ratio: float = 0.1,
    split_seed: int = 42,
) -> OuterSessionSplit:
    """Build one outer-session test fold plus an inner validation split."""

    outer = str(outer_test_session).strip().capitalize()
    if outer not in IEMOCAP_SESSION_IDS:
        raise ValueError(
            f"outer_test_session must be one of {IEMOCAP_SESSION_IDS}; got {outer!r}."
        )

    ordered_ids = list(dict.fromkeys(str(value) for value in dialogue_ids))
    if not ordered_ids:
        raise ValueError("dialogue_ids cannot be empty.")
    missing_labels = [value for value in ordered_ids if value not in labels_by_dialogue]
    missing_speakers = [value for value in ordered_ids if value not in speakers_by_dialogue]
    if missing_labels or missing_speakers:
        raise ValueError(
            "Missing split metadata: "
            f"labels={missing_labels[:3]}, speakers={missing_speakers[:3]}."
        )

    test_ids = [
        dialogue_id
        for dialogue_id in ordered_ids
        if parse_iemocap_session_id(dialogue_id) == outer
    ]
    remaining_ids = [value for value in ordered_ids if value not in set(test_ids)]
    if not test_ids or not remaining_ids:
        raise ValueError(
            f"outer fold {outer} produced test={len(test_ids)}, remaining={len(remaining_ids)}."
        )

    val_ids = _stratified_validation_ids(
        remaining_ids,
        labels_by_dialogue,
        speakers_by_dialogue,
        ratio=float(inner_val_ratio),
        seed=int(split_seed),
    )
    val_set = set(val_ids)
    train_ids = [value for value in remaining_ids if value not in val_set]

    train_set, test_set = set(train_ids), set(test_ids)
    if train_set & val_set or train_set & test_set or val_set & test_set:
        raise RuntimeError("Constructed IEMOCAP split contains overlapping dialogues.")
    if train_set | val_set | test_set != set(ordered_ids):
        raise RuntimeError("Constructed IEMOCAP split does not cover the dialogue pool.")

    return OuterSessionSplit(
        outer_test_session=outer,
        train_dialogue_ids=tuple(train_ids),
        inner_val_dialogue_ids=tuple(val_ids),
        test_dialogue_ids=tuple(test_ids),
        split_seed=int(split_seed),
        inner_val_ratio=float(inner_val_ratio),
    )


def build_official_train_split(
    train_dialogue_ids: Sequence[str],
    test_dialogue_ids: Sequence[str],
    labels_by_dialogue: Mapping[str, Sequence[int]],
    speakers_by_dialogue: Mapping[str, Sequence[Any]],
    inner_val_ratio: float = 0.1,
    split_seed: int = 42,
) -> OfficialTrainSplit:
    """Keep official test IDs untouched and split only official train IDs."""

    train_pool = list(dict.fromkeys(str(value) for value in train_dialogue_ids))
    test_ids = list(dict.fromkeys(str(value) for value in test_dialogue_ids))
    if not train_pool or not test_ids:
        raise ValueError("Official trainVid and testVid must both be non-empty.")
    overlap = set(train_pool) & set(test_ids)
    if overlap:
        raise ValueError(f"Official trainVid/testVid overlap: {sorted(overlap)}")
    if any(parse_iemocap_session_id(value) == "Ses05" for value in train_pool):
        raise ValueError("Official trainVid unexpectedly contains Ses05 dialogues.")
    if any(parse_iemocap_session_id(value) != "Ses05" for value in test_ids):
        raise ValueError("Official testVid must contain only Ses05 dialogues.")

    missing_labels = [value for value in train_pool if value not in labels_by_dialogue]
    missing_speakers = [value for value in train_pool if value not in speakers_by_dialogue]
    if missing_labels or missing_speakers:
        raise ValueError(
            "Missing official-train split metadata: "
            f"labels={missing_labels[:3]}, speakers={missing_speakers[:3]}."
        )
    val_ids = _stratified_validation_ids(
        train_pool,
        labels_by_dialogue,
        speakers_by_dialogue,
        ratio=float(inner_val_ratio),
        seed=int(split_seed),
    )
    val_set = set(val_ids)
    resolved_train = [value for value in train_pool if value not in val_set]
    if set(resolved_train) & val_set or (set(resolved_train) | val_set) != set(train_pool):
        raise RuntimeError("Official train/validation split is not a disjoint partition.")
    if (set(resolved_train) | val_set) & set(test_ids):
        raise RuntimeError("Official test dialogues leaked into train or validation.")
    return OfficialTrainSplit(
        train_dialogue_ids=tuple(resolved_train),
        inner_val_dialogue_ids=tuple(val_ids),
        test_dialogue_ids=tuple(test_ids),
        split_seed=int(split_seed),
        inner_val_ratio=float(inner_val_ratio),
    )


__all__ = [
    "OfficialTrainSplit",
    "OuterSessionSplit",
    "build_official_train_split",
    "build_outer_session_split",
]

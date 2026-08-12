from __future__ import annotations

import pytest

from models.multidag_cl.paper_reimplementation.config import (
    CurriculumPartitionProfile,
    CurriculumScheduleProfile,
)
from models.multidag_cl.paper_reimplementation.contracts import DialogueRecord
from models.multidag_cl.paper_reimplementation.curriculum import (
    CurriculumBucketPartitioner,
    CurriculumSchedule,
    DialogueDifficultyScorer,
)


@pytest.mark.parametrize(
    ("speakers", "labels", "expected"),
    [
        ([0], [0], 1.0 / 2.0),
        ([0, 0, 0], [0, 0, 0], 1.0 / 4.0),
        ([0, 0, 0], [0, 1, 0], 3.0 / 4.0),
        ([0, 1, 0, 1], [0, 1, 0, 1], 1.0 / 3.0),
        ([0, 1, 0, 1, 0], [0, 2, 1, 2, 0], 4.0 / 7.0),
        ([0, 1, 0, 1], [0, 0, 1, 1], 2.0 / 3.0),
    ],
)
def test_all_frozen_dmf_hand_fixtures(speakers, labels, expected: float) -> None:
    assert DialogueDifficultyScorer.score(labels, speakers) == expected


def test_dmf_rejects_empty_padding_predictions_and_split_inputs() -> None:
    with pytest.raises(ValueError, match="at least one"):
        DialogueDifficultyScorer.score([], [])
    with pytest.raises(ValueError, match="missing sentinel"):
        DialogueDifficultyScorer.score([0, -100], [0, 0])
    with pytest.raises(TypeError):
        DialogueDifficultyScorer.score(predictions=[0], speakers=[0])
    with pytest.raises(TypeError):
        DialogueDifficultyScorer.score([0], [0], split="validation")


def test_dmf_ignores_official_missing_label_as_emotion_class() -> None:
    assert DialogueDifficultyScorer.score([0, -1, 0], [0, 0, 0]) == 1.0 / 4.0


def _records(count: int) -> list[DialogueRecord]:
    return [
        DialogueRecord(
            dialogue_id=f"d{index}",
            original_train_index=index,
            difficulty=float(index) / 10.0,
        )
        for index in range(count)
    ]


def test_primary_divisible_and_nondivisible_balanced_membership() -> None:
    partitioner = CurriculumBucketPartitioner(
        CurriculumPartitionProfile.BALANCED_STABLE_CONTIGUOUS
    )
    divisible = partitioner.partition(_records(10), 5)
    assert divisible.actual_bucket_count == 5
    assert [len(bucket) for bucket in divisible.bucket_membership] == [2, 2, 2, 2, 2]
    assert divisible.bucket_membership[0] == ("d0", "d1")

    nondivisible = partitioner.partition(_records(7), 5)
    assert [len(bucket) for bucket in nondivisible.bucket_membership] == [2, 2, 1, 1, 1]
    assert nondivisible.bucket_membership == (
        ("d0", "d1"),
        ("d2", "d3"),
        ("d4",),
        ("d5",),
        ("d6",),
    )


def test_ties_use_original_train_index_and_ignore_input_shuffle() -> None:
    records = [
        DialogueRecord("d2", 2, 0.5),
        DialogueRecord("d0", 0, 0.5),
        DialogueRecord("d3", 3, 0.7),
        DialogueRecord("d1", 1, 0.5),
        DialogueRecord("d4", 4, 0.8),
    ]
    partitioner = CurriculumBucketPartitioner("balanced_stable_contiguous")
    first = partitioner.partition(records, 5)
    second = partitioner.partition(list(reversed(records)), 5)
    assert first.ordered_dialogue_ids == ("d0", "d1", "d2", "d3", "d4")
    assert first.bucket_membership == second.bucket_membership
    assert first.membership_sha256 == second.membership_sha256


def test_source_ceiling_chunks_record_configured_and_smaller_actual_count() -> None:
    source = CurriculumBucketPartitioner(
        CurriculumPartitionProfile.SOURCE_CEILING_CHUNKS
    ).partition(_records(5), 4)
    assert source.configured_bucket_count == 4
    assert source.actual_bucket_count == 3
    assert source.bucket_membership == (("d0", "d1"), ("d2", "d3"), ("d4",))
    assert all(source.bucket_membership)


def test_partition_rejects_k_greater_than_n_and_illegal_k() -> None:
    partitioner = CurriculumBucketPartitioner("balanced_stable_contiguous")
    with pytest.raises(ValueError, match="may not exceed"):
        partitioner.partition(_records(4), 5)
    with pytest.raises(ValueError, match="at least 1"):
        partitioner.partition(_records(4), 0)


def test_schedule_epochs_resume_disabled_and_actual_count_saturation() -> None:
    schedule = CurriculumSchedule(
        CurriculumScheduleProfile.OFFICIAL_ONE_BUCKET_PER_EPOCH
    )
    assert schedule.visible_bucket_count(1, 5) == 1
    assert schedule.visible_bucket_count(5, 5) == 5
    assert schedule.visible_bucket_count(30, 5) == 5
    assert schedule.visible_bucket_count(4, 5) == 4
    assert schedule.visible_bucket_count(30, 3) == 3

    disabled = CurriculumSchedule("official_one_bucket_per_epoch", enabled=False)
    assert disabled.visible_bucket_count(1, 5) == 5
    assert disabled.visible_bucket_count(30, 3) == 3


def test_schedule_rejects_zero_based_epoch_and_empty_actual_manifest() -> None:
    schedule = CurriculumSchedule("official_one_bucket_per_epoch")
    with pytest.raises(ValueError, match="one-based"):
        schedule.visible_bucket_count(0, 5)
    with pytest.raises(ValueError, match="positive"):
        schedule.visible_bucket_count(1, 0)

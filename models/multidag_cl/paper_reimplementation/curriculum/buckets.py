"""Deterministic profile-specific curriculum bucket partitioning."""

from __future__ import annotations

import hashlib
import math
from typing import Sequence

from ..config import CurriculumPartitionProfile
from ..contracts import BucketManifest, DialogueRecord


class CurriculumBucketPartitioner:
    """Partition immutable dialogue records without sampler or loader state."""

    def __init__(self, profile: CurriculumPartitionProfile) -> None:
        try:
            self.profile = (
                profile
                if isinstance(profile, CurriculumPartitionProfile)
                else CurriculumPartitionProfile(profile)
            )
        except (TypeError, ValueError) as error:
            raise ValueError(f"unknown curriculum partition profile: {profile!r}") from error

    def partition(
        self,
        records: Sequence[DialogueRecord],
        bucket_count: int,
    ) -> BucketManifest:
        if not isinstance(records, Sequence) or isinstance(records, (str, bytes)):
            raise TypeError("records must be a sequence of DialogueRecord")
        if isinstance(bucket_count, bool) or not isinstance(bucket_count, int):
            raise TypeError("bucket_count must be int")
        if bucket_count < 1:
            raise ValueError("bucket_count must be at least 1")
        if len(records) == 0:
            raise ValueError("at least one dialogue record is required")
        if bucket_count > len(records):
            raise ValueError("bucket_count K may not exceed dialogue count N")
        for record in records:
            if not isinstance(record, DialogueRecord):
                raise TypeError("records must contain DialogueRecord values")
            if not record.dialogue_id.strip():
                raise ValueError("dialogue IDs must be non-empty")
            if record.original_train_index < 0:
                raise ValueError("original_train_index must be non-negative")
            if not math.isfinite(record.difficulty):
                raise ValueError("difficulty must be finite")
        if len({record.dialogue_id for record in records}) != len(records):
            raise ValueError("dialogue IDs must be unique")
        if len({record.original_train_index for record in records}) != len(records):
            raise ValueError("original_train_index values must be unique")

        if self.profile is CurriculumPartitionProfile.BALANCED_STABLE_CONTIGUOUS:
            ordered = sorted(
                records,
                key=lambda record: (record.difficulty, record.original_train_index),
            )
            quotient, remainder = divmod(len(ordered), bucket_count)
            sizes = [
                quotient + (1 if bucket_index < remainder else 0)
                for bucket_index in range(bucket_count)
            ]
        else:
            ordered = sorted(records, key=lambda record: record.difficulty)
            chunk_size = int(math.ceil(len(ordered) / bucket_count))
            sizes = [
                min(chunk_size, len(ordered) - start)
                for start in range(0, len(ordered), chunk_size)
            ]

        buckets: list[tuple[DialogueRecord, ...]] = []
        cursor = 0
        for size in sizes:
            if size <= 0:
                raise RuntimeError("partition algorithm attempted to create an empty bucket")
            buckets.append(tuple(ordered[cursor : cursor + size]))
            cursor += size
        if cursor != len(ordered):
            raise RuntimeError("partition algorithm did not assign every dialogue exactly once")

        membership_lines = [
            "bucket_index\tbucket_position\tdialogue_id\toriginal_train_index\tdifficulty"
        ]
        for bucket_index, bucket in enumerate(buckets, start=1):
            for bucket_position, record in enumerate(bucket):
                membership_lines.append(
                    f"{bucket_index}\t{bucket_position}\t{record.dialogue_id}\t"
                    f"{record.original_train_index}\t{record.difficulty:.17g}"
                )
        membership_bytes = ("\n".join(membership_lines) + "\n").encode("utf-8")
        membership_sha256 = hashlib.sha256(membership_bytes).hexdigest()
        return BucketManifest(
            profile=self.profile.value,
            configured_bucket_count=bucket_count,
            actual_bucket_count=len(buckets),
            ordered_dialogue_ids=tuple(record.dialogue_id for record in ordered),
            original_indices=tuple(record.original_train_index for record in ordered),
            difficulties=tuple(float(record.difficulty) for record in ordered),
            bucket_membership=tuple(
                tuple(record.dialogue_id for record in bucket) for bucket in buckets
            ),
            membership_sha256=membership_sha256,
        )


__all__ = ["CurriculumBucketPartitioner"]

"""Training-only curriculum orchestration around the Stage-B2 pure core."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

import torch
from torch.utils.data import DataLoader, Dataset, Subset

from models.multidag_cl.paper_reimplementation.config import (
    CurriculumPartitionProfile,
    CurriculumScheduleProfile,
)
from models.multidag_cl.paper_reimplementation.contracts import (
    BucketManifest,
    DialogueRecord,
)
from models.multidag_cl.paper_reimplementation.curriculum import (
    CurriculumBucketPartitioner,
    CurriculumSchedule,
    DialogueDifficultyScorer,
)


@dataclass(frozen=True)
class CurriculumMembershipRow:
    dialogue_id: str
    original_train_index: int
    difficulty: float
    bucket_index: int
    visible_from_epoch: int


class CurriculumRuntime:
    """Own immutable train membership and derive visibility from global epoch."""

    def __init__(
        self,
        manifest: BucketManifest,
        rows: Sequence[CurriculumMembershipRow],
        *,
        schedule_profile: CurriculumScheduleProfile,
        enabled: bool,
    ) -> None:
        self.manifest = manifest
        self.rows = tuple(rows)
        self.schedule = CurriculumSchedule(schedule_profile, enabled=enabled)
        self.enabled = bool(enabled)
        if len(self.rows) != len(self.manifest.ordered_dialogue_ids):
            raise ValueError("curriculum rows and bucket manifest size differ")
        self._index_by_dialogue = {
            row.dialogue_id: row.original_train_index for row in self.rows
        }

    @classmethod
    def from_training_dataset(
        cls,
        dataset: Dataset,
        *,
        split: str,
        bucket_count: int,
        partition_profile: CurriculumPartitionProfile,
        schedule_profile: CurriculumScheduleProfile,
        enabled: bool = True,
    ) -> "CurriculumRuntime":
        if str(split).strip().lower() != "train":
            raise ValueError("curriculum may only be computed from the training split")
        records: list[DialogueRecord] = []
        for index in range(len(dataset)):
            item = dataset[index]
            if not isinstance(item, Mapping):
                raise TypeError("training dataset items must be mappings")
            dialogue_id = str(item.get("dialogue_id", "")).strip()
            if not dialogue_id:
                raise ValueError("training item is missing dialogue_id")
            labels = _valid_sequence(item.get("labels"), item, "labels")
            speakers = _valid_sequence(item.get("speaker_ids_int"), item, "speaker_ids_int")
            difficulty = DialogueDifficultyScorer.score(labels, speakers)
            records.append(
                DialogueRecord(
                    dialogue_id=dialogue_id,
                    original_train_index=index,
                    difficulty=difficulty,
                )
            )
        partitioner = CurriculumBucketPartitioner(partition_profile)
        manifest = partitioner.partition(records, bucket_count)
        by_id = {record.dialogue_id: record for record in records}
        rows: list[CurriculumMembershipRow] = []
        for bucket_index, bucket in enumerate(manifest.bucket_membership, start=1):
            for dialogue_id in bucket:
                record = by_id[dialogue_id]
                rows.append(
                    CurriculumMembershipRow(
                        dialogue_id=record.dialogue_id,
                        original_train_index=record.original_train_index,
                        difficulty=record.difficulty,
                        bucket_index=bucket_index,
                        visible_from_epoch=1 if not enabled else bucket_index,
                    )
                )
        return cls(
            manifest,
            rows,
            schedule_profile=schedule_profile,
            enabled=enabled,
        )

    def visible_bucket_count(self, global_epoch: int) -> int:
        return self.schedule.visible_bucket_count(
            global_epoch,
            self.manifest.actual_bucket_count,
        )

    def visible_indices(self, global_epoch: int) -> list[int]:
        visible_count = self.visible_bucket_count(global_epoch)
        visible_ids = [
            dialogue_id
            for bucket in self.manifest.bucket_membership[:visible_count]
            for dialogue_id in bucket
        ]
        return [self._index_by_dialogue[dialogue_id] for dialogue_id in visible_ids]

    def visible_dialogue_ids(self, global_epoch: int) -> tuple[str, ...]:
        visible_count = self.visible_bucket_count(global_epoch)
        return tuple(
            dialogue_id
            for bucket in self.manifest.bucket_membership[:visible_count]
            for dialogue_id in bucket
        )

    def build_loader(
        self,
        dataset: Dataset,
        *,
        global_epoch: int,
        batch_size: int,
        seed: int,
        collate_fn: Callable,
        num_workers: int = 0,
        pin_memory: bool = False,
    ) -> DataLoader:
        indices = self.visible_indices(global_epoch)
        subset = Subset(dataset, indices)
        generator = torch.Generator()
        generator.manual_seed(int(seed) + int(global_epoch))
        return DataLoader(
            subset,
            batch_size=int(batch_size),
            shuffle=True,
            generator=generator,
            num_workers=int(num_workers),
            pin_memory=bool(pin_memory),
            collate_fn=collate_fn,
        )

    def export_manifest(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(
                file,
                fieldnames=[
                    "dialogue_id",
                    "original_train_index",
                    "difficulty",
                    "bucket_index",
                    "visible_from_epoch",
                ],
                delimiter="\t",
                lineterminator="\n",
            )
            writer.writeheader()
            for row in self.rows:
                writer.writerow(
                    {
                        "dialogue_id": row.dialogue_id,
                        "original_train_index": row.original_train_index,
                        "difficulty": f"{row.difficulty:.17g}",
                        "bucket_index": row.bucket_index,
                        "visible_from_epoch": row.visible_from_epoch,
                    }
                )

    def resume_global_epoch(
        self,
        completed_epoch: int,
        persisted_membership_sha256: str,
    ) -> int:
        if persisted_membership_sha256 != self.manifest.membership_sha256:
            raise ValueError("resume curriculum membership SHA256 mismatch")
        if isinstance(completed_epoch, bool) or int(completed_epoch) < 0:
            raise ValueError("completed_epoch must be a non-negative integer")
        return int(completed_epoch) + 1


def _valid_sequence(value: Any, item: Mapping[str, Any], name: str) -> list[int]:
    if isinstance(value, torch.Tensor):
        values = value.detach().cpu().tolist()
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        values = list(value)
    else:
        raise TypeError(f"training item {name} must be a tensor or sequence")
    length_value: Optional[int] = None
    for key in ("length", "num_utterances"):
        if key in item:
            length_value = int(item[key])
            break
    if length_value is not None:
        values = values[:length_value]
    return [int(element) for element in values]


__all__ = ["CurriculumMembershipRow", "CurriculumRuntime"]

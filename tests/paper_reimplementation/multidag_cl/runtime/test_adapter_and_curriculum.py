from __future__ import annotations

import csv
from pathlib import Path

import pytest
import torch

from scripts.runtime.multidag_cl_paper_reimplementation import (
    CurriculumRuntime,
    ProjectBatchAdapter,
)

from ._helpers import (
    assert_tensor_values_equal,
    core_config,
    feature_metadata,
    synthetic_batch,
    synthetic_dataset,
)


def test_adapter_returns_same_mapping_and_tensor_objects_without_value_changes() -> None:
    batch = synthetic_batch()
    before = {name: value.clone() for name, value in batch.items() if torch.is_tensor(value)}
    tensor_ids = {name: id(value) for name, value in batch.items() if torch.is_tensor(value)}
    adapter = ProjectBatchAdapter(core_config(), feature_metadata())
    adapted = adapter.adapt(batch, split="train")
    assert adapted is batch
    assert {name: id(batch[name]) for name in tensor_ids} == tensor_ids
    assert_tensor_values_equal(before, batch)


def test_adapter_preserves_labels_speakers_mask_and_metadata() -> None:
    batch = synthetic_batch()
    adapter = ProjectBatchAdapter(core_config(), feature_metadata())
    labels = batch["labels"].clone()
    speakers = batch["speaker_ids_int"].clone()
    mask = batch["attention_mask"].clone()
    adapter.adapt(batch, split="validation")
    torch.testing.assert_close(labels, batch["labels"], rtol=0, atol=0)
    torch.testing.assert_close(speakers, batch["speaker_ids_int"], rtol=0, atol=0)
    torch.testing.assert_close(mask, batch["attention_mask"], rtol=0, atol=0)
    metadata = adapter.manifest_metadata()
    assert metadata["feature_registry"] == "synthetic_v1"
    assert len(metadata["feature_sha256"]) == 64
    assert metadata["feature_dimensions"] == {"text": 8, "audio": 6, "visual": 5}


def test_adapter_rejects_non_right_padding() -> None:
    batch = synthetic_batch()
    batch["attention_mask"][0] = torch.tensor([1, 0, 1])
    batch["lengths"][0] = 2
    with pytest.raises(ValueError, match="right padding|contiguous"):
        ProjectBatchAdapter(core_config(), feature_metadata()).adapt(batch, split="train")


def test_adapter_rejects_feature_dimension_mismatch() -> None:
    batch = synthetic_batch()
    batch["text_features"] = batch["text_features"][..., :-1]
    with pytest.raises(ValueError, match="feature dimension"):
        ProjectBatchAdapter(core_config(), feature_metadata()).adapt(batch, split="train")


def build_curriculum() -> CurriculumRuntime:
    config = core_config()
    return CurriculumRuntime.from_training_dataset(
        synthetic_dataset("train"),
        split="train",
        bucket_count=config.bucket_count,
        partition_profile=config.curriculum_partition,
        schedule_profile=config.curriculum_schedule,
        enabled=True,
    )


def test_curriculum_rejects_validation_and_test_membership() -> None:
    config = core_config()
    for split in ("validation", "test"):
        with pytest.raises(ValueError, match="training split"):
            CurriculumRuntime.from_training_dataset(
                synthetic_dataset("val"),
                split=split,
                bucket_count=config.bucket_count,
                partition_profile=config.curriculum_partition,
                schedule_profile=config.curriculum_schedule,
            )


def test_curriculum_visible_buckets_and_resume_use_global_epoch() -> None:
    runtime = build_curriculum()
    assert runtime.visible_bucket_count(1) == 1
    assert runtime.visible_bucket_count(5) == 5
    assert runtime.visible_bucket_count(30) == 5
    assert len(runtime.visible_indices(1)) == 1
    assert len(runtime.visible_indices(5)) == 5
    assert runtime.resume_global_epoch(3, runtime.manifest.membership_sha256) == 4
    with pytest.raises(ValueError, match="SHA256"):
        runtime.resume_global_epoch(3, "0" * 64)


def test_curriculum_manifest_has_frozen_fields(tmp_path: Path) -> None:
    runtime = build_curriculum()
    path = tmp_path / "curriculum_bucket_manifest.tsv"
    runtime.export_manifest(path)
    rows = list(csv.DictReader(path.open(encoding="utf-8"), delimiter="\t"))
    assert len(rows) == 5
    assert list(rows[0]) == [
        "dialogue_id",
        "original_train_index",
        "difficulty",
        "bucket_index",
        "visible_from_epoch",
    ]
    assert runtime.manifest.configured_bucket_count == 5
    assert runtime.manifest.actual_bucket_count == 5
    assert len(runtime.manifest.membership_sha256) == 64


def test_loader_shuffle_never_changes_visible_membership() -> None:
    runtime = build_curriculum()
    dataset = synthetic_dataset("train")
    expected = set(runtime.visible_dialogue_ids(5))
    orders = []
    for seed in (10, 11):
        loader = runtime.build_loader(
            dataset,
            global_epoch=5,
            batch_size=2,
            seed=seed,
            collate_fn=lambda items: items,
        )
        order = [item["dialogue_id"] for batch in loader for item in batch]
        orders.append(order)
        assert set(order) == expected
    assert set(orders[0]) == set(orders[1])

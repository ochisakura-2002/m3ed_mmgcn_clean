from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader

from datasets.iemocap.official_feature_dataset import iemocap_dialogue_collate_fn
from models.registry.paper_reimplementation import build_paper_reimplementation_model
from scripts.runtime.multidag_cl_paper_reimplementation.adapter import ProjectBatchAdapter
from scripts.runtime.multidag_cl_paper_reimplementation.checkpoint import (
    save_checkpoint_atomic,
)
from scripts.runtime.multidag_cl_paper_reimplementation.curriculum import CurriculumRuntime
from scripts.runtime.multidag_cl_paper_reimplementation.manifest import (
    append_resume_history,
    build_run_manifest,
    prepare_run_paths,
    write_manifest,
)
from scripts.runtime.multidag_cl_paper_reimplementation.optimizer import build_optimizer
from scripts.runtime.multidag_cl_paper_reimplementation.trainer import (
    ValidationCleanCoordinator,
    _checkpoint_payload,
    _clip_gradients,
    _lock_best_checkpoint,
    _train_epoch,
    _training_summary_counts,
)
from scripts.runtime.multidag_cl_paper_reimplementation.validation import (
    validate_runtime_config,
)
from utils.run_metadata import compute_file_sha256

from ._helpers import (
    FORMAL_CONFIG,
    REAL_CONFIG,
    ROOT,
    SYNTHETIC_CONFIG,
    core_config,
    feature_metadata,
    load_config,
    synthetic_dataset,
)


def _gradient_clipping(config: dict) -> dict:
    return config["runtime"]["gradient_clipping"]


def test_all_stage_b3_configs_execute_explicit_frozen_gradient_clipping() -> None:
    expected = {
        "mode": "global_norm",
        "max_norm": 5.0,
        "norm_type": 2.0,
        "error_if_nonfinite": True,
    }
    for path in (FORMAL_CONFIG, SYNTHETIC_CONFIG, REAL_CONFIG):
        config = load_config(path)
        assert _gradient_clipping(config) == expected
        core, _ = validate_runtime_config(
            config, mode="check", project_root=ROOT, verify_checksum=False
        )
        assert core.gradient_clip_norm == expected["max_norm"]
        assert "runtime_gradient_clipping_none_stage_b3_override" not in str(config)


def _controlled_train_epoch(maximum_optimizer_steps: int):
    config = load_config()
    model = build_paper_reimplementation_model(config)
    optimizer = build_optimizer(model, config["runtime"]["optimizer"])
    loader = DataLoader(
        synthetic_dataset("train"),
        batch_size=2,
        shuffle=False,
        collate_fn=iemocap_dialogue_collate_fn,
    )
    result, global_step = _train_epoch(
        model,
        loader,
        adapter=ProjectBatchAdapter(core_config(config), feature_metadata(config)),
        optimizer=optimizer,
        device=torch.device("cpu"),
        max_batches=1,
        maximum_optimizer_steps=maximum_optimizer_steps,
        global_step=0,
        gradient_clipping=_gradient_clipping(config),
    )
    return result, global_step, optimizer


def test_clipping_occurs_once_immediately_before_synthetic_optimizer_step(
    monkeypatch,
) -> None:
    events: list[str] = []
    original_clip = torch.nn.utils.clip_grad_norm_

    def recording_clip(*args, **kwargs):
        events.append("clip")
        return original_clip(*args, **kwargs)

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", recording_clip)
    config = load_config()
    model = build_paper_reimplementation_model(config)
    optimizer = build_optimizer(model, config["runtime"]["optimizer"])
    original_step = optimizer.step

    def recording_step(*args, **kwargs):
        events.append("step")
        return original_step(*args, **kwargs)

    monkeypatch.setattr(optimizer, "step", recording_step)
    loader = DataLoader(
        synthetic_dataset("train"),
        batch_size=2,
        shuffle=False,
        collate_fn=iemocap_dialogue_collate_fn,
    )
    result, global_step = _train_epoch(
        model,
        loader,
        adapter=ProjectBatchAdapter(core_config(config), feature_metadata(config)),
        optimizer=optimizer,
        device=torch.device("cpu"),
        max_batches=1,
        maximum_optimizer_steps=1,
        global_step=0,
        gradient_clipping=_gradient_clipping(config),
    )
    assert events == ["clip", "step"]
    assert result["gradient_clip_count"] == 1
    assert result["optimizer_steps"] == 1
    assert result["maximum_pre_clip_grad_norm"] >= result[
        "maximum_post_clip_grad_norm"
    ]
    assert result["maximum_post_clip_grad_norm"] <= 5.0 + 1.0e-5
    assert result["nonfinite_gradient_count"] == 0
    assert global_step == 1


def test_zero_step_probe_performs_no_gradient_clipping() -> None:
    result, global_step, _ = _controlled_train_epoch(0)
    assert result["batch_count"] == 1
    assert result["optimizer_steps"] == 0
    assert result["gradient_clip_count"] == 0
    assert result["maximum_pre_clip_grad_norm"] == 0.0
    assert result["maximum_post_clip_grad_norm"] == 0.0
    assert global_step == 0


def test_nonfinite_gradient_fails_before_optimizer_step() -> None:
    model = torch.nn.Linear(2, 1)
    for parameter in model.parameters():
        parameter.grad = torch.full_like(parameter, float("inf"))
    with pytest.raises(FloatingPointError, match="non-finite gradient norm"):
        _clip_gradients(model, _gradient_clipping(load_config()))


def test_training_counter_distinguishes_two_epochs_six_batches_and_steps() -> None:
    counts = _training_summary_counts(
        epoch_rows=[{"epoch": 1}, {"epoch": 2}],
        total_train_batches=6,
        total_optimizer_steps=6,
    )
    assert counts == {
        "epoch_count": 2,
        "train_batch_count": 6,
        "optimizer_step_count": 6,
    }
    assert counts["train_batch_count"] != counts["epoch_count"]


def _curriculum(config: dict) -> CurriculumRuntime:
    core = core_config(config)
    return CurriculumRuntime.from_training_dataset(
        synthetic_dataset("train"),
        split="train",
        bucket_count=core.bucket_count,
        partition_profile=core.curriculum_partition,
        schedule_profile=core.curriculum_schedule,
        enabled=core.curriculum_enabled,
    )


def test_locked_best_checkpoint_keeps_selected_epoch_global_step(tmp_path: Path) -> None:
    config = load_config()
    config["dataset"]["split_membership_sha256"] = "4" * 64
    core = core_config(config)
    feature = feature_metadata(config)
    curriculum = _curriculum(config)
    model = build_paper_reimplementation_model(config)
    optimizer = build_optimizer(model, config["runtime"]["optimizer"])
    coordinator = ValidationCleanCoordinator(test_evaluation_count=1)
    coordinator.complete_validation(
        epoch=1, val_weighted_f1=0.7, val_loss=0.5, global_step=3
    )
    best_path = tmp_path / "best_model.pt"
    save_checkpoint_atomic(
        best_path,
        _checkpoint_payload(
            model=model,
            optimizer=optimizer,
            config=config,
            core=core,
            feature=feature,
            curriculum=curriculum,
            coordinator=coordinator,
            epoch=1,
            global_step=3,
            resolved_config_sha256="5" * 64,
            checkpoint_locked=False,
        ),
    )
    coordinator.complete_validation(
        epoch=2, val_weighted_f1=0.6, val_loss=0.4, global_step=7
    )
    best = coordinator.finish_and_lock()
    assert best.epoch == 1
    assert best.global_step == 3
    _lock_best_checkpoint(
        best_path=best_path,
        model=model,
        optimizer=optimizer,
        config=config,
        core=core,
        feature=feature,
        curriculum=curriculum,
        coordinator=coordinator,
        resolved_config_sha256="5" * 64,
    )
    locked = torch.load(best_path, map_location="cpu", weights_only=False)
    assert locked["epoch"] == locked["best_epoch"] == 1
    assert locked["global_step"] == 3
    assert locked["best_validation_weighted_f1"] == 0.7
    assert locked["best_validation_loss"] == 0.5
    assert locked["selector_state"]["best"]["global_step"] == 3


def _resume_fixture(tmp_path: Path):
    config = load_config()
    config["dataset"]["split_membership_sha256"] = "6" * 64
    curriculum = _curriculum(config)
    resolved, paths = prepare_run_paths(
        config,
        config_path=SYNTHETIC_CONFIG,
        project_root=ROOT,
        output_root_override=tmp_path / "smoke_outputs",
        experiment_date="20260804",
        experiment_group=None,
    )
    manifest = build_run_manifest(
        config=resolved,
        paths=paths,
        project_root=ROOT,
        feature=feature_metadata(config),
        bucket_membership_sha256=curriculum.manifest.membership_sha256,
        configured_bucket_count=curriculum.manifest.configured_bucket_count,
        actual_bucket_count=curriculum.manifest.actual_bucket_count,
        optimizer_audit={},
        command_entrypoint="scripts/models/multidag_cl/paper_reimplementation/train.py",
    )
    write_manifest(paths.run_manifest, manifest)
    core = core_config(config)
    expected_identity = {
        "registry_key": config["registry_key"],
        "implementation_identity": core.implementation_identity,
        "conformance_profile": core.conformance_profile.value,
        "data_track": core.data_track.value,
        "feature_sha256": config["dataset"]["feature_sha256"],
        "split_protocol": config["dataset"]["split_protocol"],
        "split_membership_sha256": config["dataset"]["split_membership_sha256"],
        "bucket_membership_sha256": curriculum.manifest.membership_sha256,
        "resolved_config_sha256": manifest["resolved_config_sha256"],
    }
    checkpoint = {
        "registry_key": expected_identity["registry_key"],
        "implementation_identity": expected_identity["implementation_identity"],
        "conformance_profile": expected_identity["conformance_profile"],
        "data_track": expected_identity["data_track"],
        "feature_sha256": expected_identity["feature_sha256"],
        "split_protocol": expected_identity["split_protocol"],
        "split_membership_sha256": expected_identity["split_membership_sha256"],
        "curriculum_membership_sha256": expected_identity[
            "bucket_membership_sha256"
        ],
        "resolved_config_sha256": expected_identity["resolved_config_sha256"],
        "epoch": 3,
        "global_step": 11,
        "checkpoint_locked": False,
    }
    checkpoint_path = paths.checkpoints / "last_model.pt"
    save_checkpoint_atomic(checkpoint_path, checkpoint)
    return config, paths, manifest, expected_identity, checkpoint, checkpoint_path


def test_resume_keeps_resolved_config_bytes_and_appends_history(tmp_path: Path) -> None:
    config, paths, initial, identity, checkpoint, checkpoint_path = _resume_fixture(
        tmp_path
    )
    before_bytes = paths.resolved_config.read_bytes()
    before_sha = compute_file_sha256(paths.resolved_config)
    resumed, resumed_paths = prepare_run_paths(
        config,
        config_path=SYNTHETIC_CONFIG,
        project_root=ROOT,
        output_root_override=tmp_path / "smoke_outputs",
        experiment_date=None,
        experiment_group=None,
        resume_run_dir=paths.run_dir,
    )
    assert resumed == load_config_from_bytes(before_bytes)
    assert resumed_paths == paths
    assert paths.resolved_config.read_bytes() == before_bytes
    assert compute_file_sha256(paths.resolved_config) == before_sha
    for index in (1, 2):
        entry = append_resume_history(
            paths.run_manifest,
            checkpoint_path=checkpoint_path,
            checkpoint=checkpoint,
            run_dir=paths.run_dir,
            project_root=ROOT,
            expected_identity=identity,
            resume_timestamp=f"2026-08-04T00:00:0{index}Z",
        )
        assert entry["completed_epoch"] == 3
        assert entry["resumed_global_epoch"] == 4
        assert entry["checkpoint_global_step"] == 11
        assert entry["checkpoint_path"] == "checkpoints/last_model.pt"
    updated = json.loads(paths.run_manifest.read_text(encoding="utf-8"))
    assert len(updated["resume_history"]) == 2
    assert updated["resumed_global_epoch"] == 4
    for name in identity:
        assert updated[name] == initial[name]
    assert paths.resolved_config.read_bytes() == before_bytes


def load_config_from_bytes(value: bytes) -> dict:
    import yaml

    loaded = yaml.safe_load(value.decode("utf-8"))
    assert isinstance(loaded, dict)
    return loaded


def test_resume_rejects_config_and_curriculum_mismatch(tmp_path: Path) -> None:
    config, paths, _, identity, checkpoint, checkpoint_path = _resume_fixture(tmp_path)
    changed = deepcopy(config)
    changed["runtime"]["batch_size"] = 3
    with pytest.raises(ValueError, match="immutable resolved_config.yaml"):
        prepare_run_paths(
            changed,
            config_path=SYNTHETIC_CONFIG,
            project_root=ROOT,
            output_root_override=tmp_path / "smoke_outputs",
            experiment_date=None,
            experiment_group=None,
            resume_run_dir=paths.run_dir,
        )
    wrong_curriculum = deepcopy(checkpoint)
    wrong_curriculum["curriculum_membership_sha256"] = "0" * 64
    wrong_path = paths.checkpoints / "wrong_curriculum.pt"
    save_checkpoint_atomic(wrong_path, wrong_curriculum)
    with pytest.raises(ValueError, match="curriculum_membership_sha256 mismatch"):
        append_resume_history(
            paths.run_manifest,
            checkpoint_path=wrong_path,
            checkpoint=wrong_curriculum,
            run_dir=paths.run_dir,
            project_root=ROOT,
            expected_identity=identity,
        )
    locked = deepcopy(checkpoint)
    locked["checkpoint_locked"] = True
    locked_path = paths.checkpoints / "locked_best.pt"
    save_checkpoint_atomic(locked_path, locked)
    with pytest.raises(RuntimeError, match="locked best checkpoint"):
        append_resume_history(
            paths.run_manifest,
            checkpoint_path=locked_path,
            checkpoint=locked,
            run_dir=paths.run_dir,
            project_root=ROOT,
            expected_identity=identity,
        )
    current = json.loads(paths.run_manifest.read_text(encoding="utf-8"))
    assert current["resume_history"] == []

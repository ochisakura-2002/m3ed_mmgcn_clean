from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import torch

from models.registry.paper_reimplementation import build_paper_reimplementation_model
from scripts.runtime.multidag_cl_paper_reimplementation import (
    TestEvaluationGate,
    ValidationCheckpointSelector,
    ValidationCleanCoordinator,
    build_optimizer,
    strict_reload_checkpoint,
)
from scripts.runtime.multidag_cl_paper_reimplementation.checkpoint import (
    save_checkpoint_atomic,
)
from scripts.runtime.multidag_cl_paper_reimplementation.optimizer import (
    audit_optimizer_parameters,
)

from ._helpers import core_config, feature_metadata, load_config


def test_optimizer_is_explicit_single_group_adamw_with_complete_coverage() -> None:
    config = load_config()
    model = build_paper_reimplementation_model(config)
    optimizer = build_optimizer(model, config["runtime"]["optimizer"])
    assert isinstance(optimizer, torch.optim.AdamW)
    assert len(optimizer.param_groups) == 1
    group = optimizer.param_groups[0]
    assert group["lr"] == 5.0e-4
    assert group["betas"] == (0.9, 0.999)
    assert group["eps"] == 1.0e-6
    assert group["weight_decay"] == 0.0
    assert group["amsgrad"] is False
    audit = audit_optimizer_parameters(model, optimizer)
    assert audit["missing_parameter_count"] == 0
    assert audit["duplicate_parameter_count"] == 0
    assert audit["optimized_parameter_tensor_count"] == audit["trainable_parameter_tensor_count"]


@pytest.mark.parametrize(
    ("candidates", "expected_epoch"),
    [
        ([(1, 0.4, 0.9), (2, 0.5, 2.0)], 2),
        ([(1, 0.5, 0.9), (2, 0.5, 0.8)], 2),
        ([(1, 0.5, 0.8), (2, 0.5, 0.8)], 1),
    ],
)
def test_selector_validation_lexicographic_rules(candidates, expected_epoch) -> None:
    selector = ValidationCheckpointSelector()
    for epoch, weighted_f1, loss in candidates:
        selector.update(epoch=epoch, val_weighted_f1=weighted_f1, val_loss=loss)
    assert selector.best is not None
    assert selector.best.epoch == expected_epoch


def test_selector_lock_is_irreversible_and_has_no_test_metric_input() -> None:
    selector = ValidationCheckpointSelector()
    selector.update(epoch=1, val_weighted_f1=0.5, val_loss=0.7)
    selector.lock()
    assert selector.state_dict()["test_split_used_for_selection"] is False
    with pytest.raises(RuntimeError, match="locked"):
        selector.update(epoch=2, val_weighted_f1=1.0, val_loss=0.0)
    with pytest.raises(TypeError):
        selector.update(
            epoch=2,
            val_weighted_f1=1.0,
            val_loss=0.0,
            test_weighted_f1=1.0,
        )


def test_test_gate_requires_finish_lock_reload_and_only_once() -> None:
    gate = TestEvaluationGate(maximum_evaluations=1)
    with pytest.raises(RuntimeError):
        gate.before_test()
    gate.mark_training_finished()
    gate.mark_checkpoint_locked()
    with pytest.raises(RuntimeError):
        gate.before_test()
    gate.mark_checkpoint_reloaded()
    gate.before_test()
    assert gate.test_evaluation_count == 1
    with pytest.raises(RuntimeError, match="limit"):
        gate.before_test()


def _fake_30_epoch_best(test_metric_bias: float) -> tuple[int, int]:
    coordinator = ValidationCleanCoordinator(test_evaluation_count=1)
    test_loader_calls = 0
    for epoch in range(1, 31):
        validation_score = 0.8 if epoch == 7 else 0.5
        coordinator.complete_validation(
            epoch=epoch,
            val_weighted_f1=validation_score,
            val_loss=1.0 / epoch,
        )
        test_metric_bias += epoch
        assert test_loader_calls == 0
    best = coordinator.finish_and_lock()
    assert test_loader_calls == 0
    coordinator.mark_strict_reload()
    coordinator.before_test()
    test_loader_calls += 1
    return best.epoch, test_loader_calls


def test_fake_30_epoch_test_leakage_proof() -> None:
    first = _fake_30_epoch_best(-1000.0)
    second = _fake_30_epoch_best(1000.0)
    assert first == second == (7, 1)


def _checkpoint_payload(config: dict, locked: bool) -> tuple[dict, torch.nn.Module]:
    model = build_paper_reimplementation_model(config)
    optimizer = build_optimizer(model, config["runtime"]["optimizer"])
    core = core_config(config)
    feature = feature_metadata(config)
    payload = {
        "registry_key": config["registry_key"],
        "implementation_identity": core.implementation_identity,
        "conformance_profile": core.conformance_profile.value,
        "data_track": core.data_track.value,
        "model_config": core.to_mapping(),
        "num_classes": core.num_classes,
        "feature_dimensions": feature.feature_dimensions,
        "feature_registry": feature.registry_key,
        "feature_sha256": feature.feature_sha256,
        "split_protocol": config["dataset"]["split_protocol"],
        "split_membership_sha256": "2" * 64,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "curriculum_membership_sha256": "1" * 64,
        "resolved_config_sha256": "3" * 64,
        "checkpoint_locked": locked,
    }
    return payload, model


def test_strict_reload_validates_identity_profile_config_and_lock(tmp_path: Path) -> None:
    config = load_config()
    payload, model = _checkpoint_payload(config, locked=True)
    path = tmp_path / "best_model.pt"
    save_checkpoint_atomic(path, payload)
    expected = {
        name: payload[name]
        for name in (
            "registry_key",
            "implementation_identity",
            "conformance_profile",
            "data_track",
            "model_config",
            "num_classes",
            "feature_dimensions",
            "feature_registry",
            "feature_sha256",
            "split_protocol",
        )
    }
    strict_reload_checkpoint(
        path,
        model=model,
        device=torch.device("cpu"),
        expected_identity=expected,
        require_locked=True,
    )
    wrong = deepcopy(expected)
    wrong["conformance_profile"] = "official_source_behavior"
    with pytest.raises(ValueError, match="conformance_profile"):
        strict_reload_checkpoint(
            path,
            model=model,
            device=torch.device("cpu"),
            expected_identity=wrong,
            require_locked=True,
        )


def test_strict_reload_rejects_unlocked_checkpoint(tmp_path: Path) -> None:
    config = load_config()
    payload, model = _checkpoint_payload(config, locked=False)
    path = tmp_path / "best_model.pt"
    save_checkpoint_atomic(path, payload)
    expected = {name: payload[name] for name in payload if name not in {"model_state_dict", "optimizer_state_dict", "curriculum_membership_sha256", "checkpoint_locked"}}
    with pytest.raises(ValueError, match="locked"):
        strict_reload_checkpoint(
            path,
            model=model,
            device=torch.device("cpu"),
            expected_identity=expected,
            require_locked=True,
        )

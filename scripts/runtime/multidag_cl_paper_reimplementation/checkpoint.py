"""Validation-clean selection, checkpoint compatibility, and test leakage guard."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

import torch


@dataclass(frozen=True)
class ValidationCandidate:
    epoch: int
    val_weighted_f1: float
    val_loss: float
    global_step: int = 0


class ValidationCheckpointSelector:
    """Lexicographic validation-only selector with an irreversible lock."""

    def __init__(self) -> None:
        self.best: Optional[ValidationCandidate] = None
        self.locked = False

    @staticmethod
    def _key(candidate: ValidationCandidate) -> tuple[float, float, int]:
        return (
            float(candidate.val_weighted_f1),
            -float(candidate.val_loss),
            -int(candidate.epoch),
        )

    def update(
        self,
        *,
        epoch: int,
        val_weighted_f1: float,
        val_loss: float,
        global_step: int = 0,
    ) -> bool:
        if self.locked:
            raise RuntimeError("checkpoint selector is locked")
        candidate = ValidationCandidate(
            epoch=int(epoch),
            val_weighted_f1=float(val_weighted_f1),
            val_loss=float(val_loss),
            global_step=int(global_step),
        )
        if candidate.epoch < 1:
            raise ValueError("candidate epoch must be one-based")
        if candidate.global_step < 0:
            raise ValueError("candidate global_step must be non-negative")
        if self.best is None or self._key(candidate) > self._key(self.best):
            self.best = candidate
            return True
        return False

    def lock(self) -> ValidationCandidate:
        if self.best is None:
            raise RuntimeError("cannot lock without a validation candidate")
        self.locked = True
        return self.best

    def state_dict(self) -> dict[str, Any]:
        return {
            "locked": self.locked,
            "best": None if self.best is None else asdict(self.best),
            "selection_fields": [
                "val_weighted_f1_desc",
                "val_loss_asc",
                "epoch_asc",
            ],
            "test_split_used_for_selection": False,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        best = state.get("best")
        self.best = None if best is None else ValidationCandidate(**best)
        self.locked = bool(state.get("locked", False))


class TestEvaluationGate:
    """Enforce finish -> lock -> strict reload -> one final test evaluation."""

    __test__ = False

    def __init__(self, *, maximum_evaluations: int) -> None:
        if maximum_evaluations not in (0, 1):
            raise ValueError("maximum test evaluations must be 0 or 1")
        self.maximum_evaluations = int(maximum_evaluations)
        self.training_finished = False
        self.best_checkpoint_locked = False
        self.best_checkpoint_reloaded = False
        self.test_evaluation_count = 0

    def mark_training_finished(self) -> None:
        self.training_finished = True

    def mark_checkpoint_locked(self) -> None:
        if not self.training_finished:
            raise RuntimeError("best checkpoint cannot lock before training finishes")
        self.best_checkpoint_locked = True

    def mark_checkpoint_reloaded(self) -> None:
        if not self.best_checkpoint_locked:
            raise RuntimeError("best checkpoint cannot reload before lock")
        self.best_checkpoint_reloaded = True

    def before_test(self) -> None:
        if not (
            self.training_finished
            and self.best_checkpoint_locked
            and self.best_checkpoint_reloaded
        ):
            raise RuntimeError(
                "test requires training_finished, best_checkpoint_locked, "
                "and best_checkpoint_reloaded"
            )
        if self.test_evaluation_count >= self.maximum_evaluations:
            raise RuntimeError("test evaluation count limit exceeded")
        self.test_evaluation_count += 1

    def state_dict(self) -> dict[str, Any]:
        return {
            "training_finished": self.training_finished,
            "best_checkpoint_locked": self.best_checkpoint_locked,
            "best_checkpoint_reloaded": self.best_checkpoint_reloaded,
            "test_evaluation_count": self.test_evaluation_count,
            "test_split_used_for_selection": False,
        }


def save_checkpoint_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(dict(payload), temporary)
    temporary.replace(path)


def load_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    try:
        value = torch.load(Path(path), map_location=device, weights_only=False)
    except TypeError:
        value = torch.load(Path(path), map_location=device)
    if not isinstance(value, dict):
        raise TypeError("checkpoint must contain a mapping")
    return value


def strict_reload_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    device: torch.device,
    expected_identity: Mapping[str, Any],
    optimizer: Optional[torch.optim.Optimizer] = None,
    require_locked: bool = False,
) -> dict[str, Any]:
    checkpoint = load_checkpoint(path, device)
    required = {
        "registry_key",
        "implementation_identity",
        "conformance_profile",
        "data_track",
        "model_config",
        "num_classes",
        "feature_dimensions",
        "model_state_dict",
        "optimizer_state_dict",
        "curriculum_membership_sha256",
        "resolved_config_sha256",
        "split_membership_sha256",
    }
    missing = sorted(required - set(checkpoint))
    if missing:
        raise ValueError(f"checkpoint missing compatibility fields: {missing}")
    for name, expected in expected_identity.items():
        if checkpoint.get(name) != expected:
            raise ValueError(
                f"checkpoint {name} mismatch: "
                f"checkpoint={checkpoint.get(name)!r}, expected={expected!r}"
            )
    if require_locked and checkpoint.get("checkpoint_locked") is not True:
        raise ValueError("evaluation requires a locked best checkpoint")
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint


__all__ = [
    "TestEvaluationGate",
    "ValidationCandidate",
    "ValidationCheckpointSelector",
    "load_checkpoint",
    "save_checkpoint_atomic",
    "strict_reload_checkpoint",
]

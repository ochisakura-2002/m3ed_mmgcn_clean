from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
import shutil
import uuid

import pytest
import torch
import yaml

from scripts.runtime.paper_aligned import (
    normalized_training_config,
    validate_runtime_config,
)
from scripts.workflows.benchmarks.prepare_full_repair import (
    prepare_full_repair_matrix,
)
from utils.training_control import EarlyStoppingController
from utils.training_diagnostics import (
    DIAGNOSTIC_FIELDS,
    PredictionDiagnosticsAccumulator,
    TrainingEpochAccumulator,
    complete_diagnostic_row,
    diagnostics_enabled,
    should_collect_expensive_diagnostics,
    write_diagnostic_csv,
)


ROOT = Path(__file__).resolve().parents[2]
MATRIX_PATH = Path(
    "configs/benchmarks/repairs/dialoguegcn_gsmcc_full/repair_matrix.yaml"
)
FORMAL_BASES = (
    ROOT
    / "configs/benchmarks/long_training/iemocap_clean/base/"
    "dialoguegcn_full_context.yaml",
    ROOT
    / "configs/benchmarks/long_training/iemocap_clean/base/"
    "gsmcc_project_variant_full_context.yaml",
)
REQUIRED_REPAIR_METADATA = {
    "run_id",
    "model_family",
    "implementation_identity",
    "context_mode",
    "validation_session",
    "test_session",
    "seed",
    "candidate_type",
    "changed_variables",
    "unchanged_variables",
    "control_group",
    "diagnostic_question",
    "base_config",
    "resolved_config",
    "entrypoint",
    "max_epochs",
    "min_epochs",
    "patience",
    "learning_rate",
    "selection_metric",
    "test_selection_leakage",
    "experiment_group",
    "output_root",
}


@pytest.fixture
def workspace_tmp_dir() -> Path:
    temp_parent = ROOT / "tests" / "_tmp_full_repair"
    temp_parent.mkdir(parents=True, exist_ok=True)
    path = temp_parent / f"pytest_full_repair_{uuid.uuid4().hex}"
    path.mkdir()
    try:
        yield path
    finally:
        shutil.rmtree(path)
        try:
            temp_parent.rmdir()
        except OSError:
            pass


def _load_yaml(path: Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def test_diagnostics_default_off_and_frequency_contract() -> None:
    state = torch.random.get_rng_state().clone()
    assert diagnostics_enabled({}) is False
    assert diagnostics_enabled({"diagnostics": {"enabled": False}}) is False
    assert should_collect_expensive_diagnostics({}, 1) is False
    assert torch.equal(state, torch.random.get_rng_state())

    enabled = {
        "diagnostics": {
            "enabled": True,
            "full_frequency_epochs": 15,
            "expensive_every_n_epochs": 5,
        }
    }
    assert all(
        should_collect_expensive_diagnostics(enabled, epoch)
        for epoch in range(1, 16)
    )
    assert should_collect_expensive_diagnostics(enabled, 20)
    assert not should_collect_expensive_diagnostics(enabled, 16)


def test_diagnostic_fields_and_missing_losses_are_not_zero(
    workspace_tmp_dir: Path,
) -> None:
    expected = {
        "epoch",
        "train_loss",
        "val_loss",
        "train_weighted_f1",
        "val_weighted_f1",
        "learning_rate",
        "classification_loss",
        "contrastive_loss",
        "auxiliary_loss",
        "gradient_norm",
        "parameter_update_norm",
        "nonzero_gradient_parameter_count",
        "trainable_parameter_count",
        "logit_mean",
        "logit_std",
        "logit_min",
        "logit_max",
        "prediction_entropy",
        "predicted_class_count",
        "dominant_class_ratio",
        "per_class_recall",
        "effective_batch_count",
        "early_stopping_counter",
        "best_epoch",
        "best_metric",
        "min_epochs",
        "patience",
        "stopped_by_early_stopping",
    }
    assert set(DIAGNOSTIC_FIELDS) == expected

    accumulator = TrainingEpochAccumulator(
        2,
        collect_parameter_update=True,
    )
    model = torch.nn.Linear(2, 2, bias=False)
    model.weight.grad = torch.ones_like(model.weight)
    accumulator.record_gradients(model)
    before = accumulator.snapshot_parameter_update(model)
    with torch.no_grad():
        model.weight.add_(0.25)
    accumulator.record_parameter_update(model, before)
    output = {
        "loss": torch.tensor(0.5),
        "classification_loss": torch.tensor(0.5),
        "aux_losses": {},
        "logits": torch.tensor([[[2.0, 0.0], [0.0, 2.0]]]),
    }
    batch = {
        "labels": torch.tensor([[0, 1]]),
        "attention_mask": torch.tensor([[1, 1]]),
    }
    accumulator.update_batch(output, batch)
    summary = accumulator.summary()
    assert summary["contrastive_loss"] is None
    assert summary["auxiliary_loss"] is None
    assert summary["parameter_update_norm"] == pytest.approx(0.5)
    assert summary["nonzero_gradient_parameter_count"] == 4

    row = complete_diagnostic_row(
        {
            "epoch": 1,
            "contrastive_loss": summary["contrastive_loss"],
            "auxiliary_loss": summary["auxiliary_loss"],
        }
    )
    path = workspace_tmp_dir / "training_diagnostics.csv"
    write_diagnostic_csv(path, [row])
    with path.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    assert set(rows[0]) == expected
    assert rows[0]["contrastive_loss"] == ""
    assert rows[0]["auxiliary_loss"] == ""


def test_prediction_diagnostics_are_structured() -> None:
    accumulator = PredictionDiagnosticsAccumulator(2, ["Negative", "Positive"])
    accumulator.update(
        torch.tensor([[[2.0, 0.0], [0.0, 2.0], [0.0, 0.0]]]),
        torch.tensor([[0, 1, -100]]),
        torch.tensor([[1, 1, 0]]),
    )
    summary = accumulator.summary()
    assert summary["predicted_class_count"] == 2
    assert summary["dominant_class_ratio"] == pytest.approx(0.5)
    assert summary["logit_std"] > 0
    assert json.loads(summary["per_class_recall"]) == {
        "Negative": 1.0,
        "Positive": 1.0,
    }


def test_min_epochs_default_preserves_old_patience_behavior() -> None:
    controller = EarlyStoppingController(patience=2)
    assert controller.min_epochs == 0
    assert controller.update(1, 0.5).improved
    assert not controller.update(2, 0.4).should_stop
    decision = controller.update(3, 0.3)
    assert decision.should_stop
    assert decision.counter == 2


def test_min_epochs_blocks_stop_but_allows_best_checkpoint_signal() -> None:
    controller = EarlyStoppingController(patience=1, min_epochs=3)
    first = controller.update(1, 0.2)
    second = controller.update(2, 0.4)
    assert first.improved and not first.should_stop
    assert second.improved and not second.should_stop
    assert second.best_epoch == 2
    assert second.best_metric == pytest.approx(0.4)

    third = controller.update(3, 0.3)
    assert third.should_stop
    assert third.counter == 1


def test_repair_prepare_materializes_twelve_valid_configs_without_training(
    workspace_tmp_dir: Path,
) -> None:
    before = {path: path.read_bytes() for path in FORMAL_BASES}
    resolved_root = workspace_tmp_dir / "resolved_configs"
    result = prepare_full_repair_matrix(
        MATRIX_PATH,
        "prepare",
        "20260727",
        root=ROOT,
        resolved_root=resolved_root,
        batch_id="pytest_full_repair",
    )

    assert result["expanded_run_count"] == 12
    assert result["expanded_run_count"] <= 12
    assert result["model_counts"] == Counter(
        {"dialoguegcn": 4, "gsmcc_project_variant": 8}
    )
    assert result["runtime_validation_count"] == 12
    assert result["output_collision_count"] == 0
    assert result["formal_sources_unchanged"] is True
    assert result["formal_training_started"] == 0
    assert len(set(record["run_id"] for record in result["records"])) == 12
    assert len(set(record["output_root"] for record in result["records"])) == 12
    assert len(set(result["commands"])) == 12

    for name in (
        "commands.txt",
        "matrix.yaml",
        "matrix.csv",
        "git_commit.txt",
        "preparation_metadata.json",
    ):
        assert (workspace_tmp_dir / name).is_file()

    for record in result["records"]:
        config = _load_yaml(record["resolved_path"])
        repair = config["repair_experiment"]
        assert REQUIRED_REPAIR_METADATA <= set(repair)
        assert repair["test_session"] == "Ses05"
        assert repair["selection_metric"] == "val_weighted_f1"
        assert repair["test_selection_leakage"] is False
        assert config["protocol"]["test_split_used_for_selection"] is False
        assert config["training"]["select_best_by"] == "val_weighted_f1"
        assert config["diagnostics"]["enabled"] is True
        assert config["diagnostics"]["full_frequency_epochs"] >= 15
        assert config["output"]["experiment_group"] == (
            "dialoguegcn_gsmcc_full_repair"
        )
        assert config["output"]["root"].startswith(
            "outputs/20260727/dialoguegcn_gsmcc_full_repair/runs/"
        )
        assert "outputs/long_training" not in record["command"]
        assert "outputs/launcher_logs" not in record["command"]
        validate_runtime_config(normalized_training_config(config))

    assert all(path.read_bytes() == before[path] for path in FORMAL_BASES)


def test_repair_candidate_controlled_variables() -> None:
    result = prepare_full_repair_matrix(
        MATRIX_PATH,
        "check",
        "20260727",
        root=ROOT,
        batch_id="pytest_full_repair_check",
    )
    by_id = {record["run_id"]: record["config"] for record in result["records"]}

    for run_id in (
        "dialoguegcn_ses04_original_diagnostic",
        "dialoguegcn_ses03_original_diagnostic",
    ):
        training = by_id[run_id]["training"]
        assert training.get("early_stopping_min_epochs", 0) == 0
        assert training["early_stopping_patience"] == 10
        assert by_id[run_id]["optimizer"]["learning_rate"] == pytest.approx(3e-4)

    for run_id in (
        "dialoguegcn_ses04_delayed_early_stop",
        "dialoguegcn_ses03_delayed_early_stop",
    ):
        training = by_id[run_id]["training"]
        assert training["early_stopping_min_epochs"] == 30
        assert training["early_stopping_patience"] == 20
        assert by_id[run_id]["optimizer"]["learning_rate"] == pytest.approx(3e-4)

    for session in ("ses01", "ses02"):
        original = by_id[f"gsmcc_{session}_original_diagnostic"]
        delayed = by_id[f"gsmcc_{session}_delayed_early_stop"]
        lr_candidate = by_id[f"gsmcc_{session}_lr_candidate"]
        assert original["training"].get("early_stopping_min_epochs", 0) == 0
        assert original["training"]["early_stopping_patience"] == 20
        assert original["optimizer"]["learning_rate"] == pytest.approx(1e-5)
        assert delayed["training"]["early_stopping_min_epochs"] == 90
        assert delayed["training"]["early_stopping_patience"] == 40
        assert delayed["optimizer"]["learning_rate"] == pytest.approx(1e-5)
        assert lr_candidate["training"].get("early_stopping_min_epochs", 0) == 0
        assert lr_candidate["training"]["early_stopping_patience"] == 20
        assert lr_candidate["optimizer"]["learning_rate"] == pytest.approx(3e-5)

    for session in ("ses03", "ses04"):
        control = by_id[f"gsmcc_{session}_best_candidate_control"]
        assert control["training"]["early_stopping_min_epochs"] == 90
        assert control["training"]["early_stopping_patience"] == 40
        assert control["optimizer"]["learning_rate"] == pytest.approx(1e-5)

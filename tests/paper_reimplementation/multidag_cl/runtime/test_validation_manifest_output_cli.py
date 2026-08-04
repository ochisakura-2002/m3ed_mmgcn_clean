from __future__ import annotations

import inspect
import json
from copy import deepcopy
from pathlib import Path

import pytest
import torch

from models.registry.paper_reimplementation import (
    MODEL_METADATA,
    MODEL_REGISTRY,
    REGISTRY_KEY,
    get_model_metadata,
)
from scripts.runtime.multidag_cl_paper_reimplementation import (
    RuntimeValidationError,
    run_runtime,
    validate_runtime_config,
)
from scripts.runtime.multidag_cl_paper_reimplementation.manifest import (
    REQUIRED_MANIFEST_FIELDS,
)
from scripts.runtime.multidag_cl_paper_reimplementation.validation import (
    SMOKE_OUTPUT_ROOT,
)
import scripts.models.multidag_cl.paper_reimplementation.train as cli

from ._helpers import FORMAL_CONFIG, ROOT, SYNTHETIC_CONFIG, load_config, mutated_config


def validate(config: dict, mode: str = "check"):
    return validate_runtime_config(
        config,
        mode=mode,
        project_root=ROOT,
        verify_checksum=False,
    )


def test_registry_identity_is_unique_complete_and_not_causal_or_official() -> None:
    assert list(MODEL_REGISTRY) == [REGISTRY_KEY]
    metadata = get_model_metadata()
    required = {
        "canonical_name",
        "implementation_identity",
        "conformance_profile",
        "data_track",
        "paper_title",
        "paper_venue",
        "paper_year",
        "paper_sha256",
        "official_repo_url",
        "official_commit",
        "paper_data_track_status",
        "dag_topology_causal",
        "end_to_end_causal",
    }
    assert required <= set(metadata)
    assert metadata["implementation_identity"] == "paper_reimplementation"
    assert metadata["dag_topology_causal"] is True
    assert metadata["end_to_end_causal"] is False
    assert "author_official" not in json.dumps(metadata)


def test_formal_config_check_validates_registry_split_and_model_without_asset() -> None:
    core, feature = validate(load_config(FORMAL_CONFIG), "check")
    assert core.graph_layers == 4
    assert feature.registry_key == "clean_roberta_v1"
    assert feature.feature_dimensions == {"text": 768, "audio": 1582, "visual": 342}


def _mutate(path, value, *, source=FORMAL_CONFIG):
    config = mutated_config(source)
    target = config
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    return config


@pytest.mark.parametrize(
    "config",
    [
        _mutate(("model_core", "identity", "implementation_identity"), "author_official"),
        _mutate(("model_core", "data", "track"), "paper_data"),
        _mutate(("experiment", "context_label"), "causal_context"),
        _mutate(("model_core", "graph", "window_future"), 1),
        _mutate(("model_core", "graph", "allow_self_edge"), True),
        _mutate(("model_core", "graph", "allow_future_edge"), True),
        _mutate(("model_core", "graph", "global_nodal_attention"), True),
        _mutate(("runtime", "early_stopping"), True),
        _mutate(("runtime", "epochs"), 29),
        _mutate(("model_core", "curriculum", "bucket_count"), 4),
        _mutate(("checkpoint", "primary_metric"), "test_weighted_f1"),
        _mutate(("checkpoint", "secondary_tiebreak"), "test_loss_lower"),
        _mutate(("checkpoint", "test_evaluation_count"), 2),
        _mutate(("dataset", "feature_dimensions", "text"), 767),
        _mutate(("dataset", "feature_sha256"), ""),
        _mutate(("dataset", "validation_session"), "Ses05"),
        _mutate(("dataset", "test_session"), "Ses04"),
        _mutate(("registry_key",), "unknown_registry_key"),
        _mutate(("model_core", "identity", "conformance_profile"), "official_source_behavior"),
    ],
)
def test_runtime_validation_rejects_forbidden_formal_cases(config: dict) -> None:
    with pytest.raises((RuntimeValidationError, ValueError, KeyError, TypeError)):
        validate(config, "check")


def test_runtime_validation_rejects_smoke_output_in_outputs() -> None:
    config = mutated_config(SYNTHETIC_CONFIG)
    config["runtime"]["smoke_output_root"] = "outputs"
    with pytest.raises(RuntimeValidationError, match="smoke_output_root"):
        validate(config, "synthetic-smoke")


def test_synthetic_runtime_writes_complete_canonical_artifacts_and_one_step() -> None:
    output_root = SMOKE_OUTPUT_ROOT / "pt"
    result = run_runtime(
        SYNTHETIC_CONFIG,
        mode="synthetic-smoke",
        output_root_override=output_root,
        experiment_date="20260804",
        experiment_group="mdcl_pr_pytest",
        device_override="cpu",
    )
    assert result["status"] == "PASS"
    assert result["optimizer_steps"] == 1
    assert result["optimizer_step_count"] == 1
    assert result["epoch_count"] == 1
    assert result["train_batch_count"] == 1
    assert result["gradient_clip_count"] == 1
    assert result["nonfinite_gradient_count"] == 0
    assert result["test_evaluation_count"] == 1
    assert result["test_split_used_for_selection"] is False
    assert result["formal_training_started"] == 0
    run_dir = Path(result["run_dir"])
    assert "outputs" not in run_dir.parts
    expected = [
        run_dir / "checkpoints/best_model.pt",
        run_dir / "checkpoints/last_model.pt",
        run_dir / "logs/epoch_metrics.tsv",
        run_dir / "logs/run_summary.json",
        run_dir / "reports/validation_best_metrics.json",
        run_dir / "reports/fake_test_metrics.json",
        run_dir / "predictions/validation_best_predictions.tsv",
        run_dir / "predictions/fake_test_predictions.tsv",
        run_dir / "manifests/curriculum_bucket_manifest.tsv",
        run_dir / "manifests/run_manifest.json",
        run_dir / "resolved_config.yaml",
    ]
    assert all(path.is_file() for path in expected)
    manifest = json.loads((run_dir / "manifests/run_manifest.json").read_text(encoding="utf-8"))
    assert REQUIRED_MANIFEST_FIELDS <= set(manifest)
    assert manifest["formal_experiment"] is False
    assert manifest["smoke_only"] is True
    assert manifest["test_split_used_for_selection"] is False
    assert manifest["final_evaluation"]["test_evaluation_count_actual"] == 1
    assert not Path(manifest["resolved_config_path"]).is_absolute()
    checkpoint = torch.load(
        run_dir / "checkpoints/best_model.pt",
        map_location="cpu",
        weights_only=False,
    )
    assert checkpoint["checkpoint_locked"] is True
    assert checkpoint["training_finished"] is True

    manifest_before_evaluation = (run_dir / "manifests/run_manifest.json").read_bytes()
    checkpoint_before_evaluation = torch.load(
        run_dir / "checkpoints/best_model.pt",
        map_location="cpu",
        weights_only=False,
    )
    evaluation = run_runtime(
        SYNTHETIC_CONFIG,
        mode="evaluate",
        device_override="cpu",
        resume_checkpoint=run_dir / "checkpoints/best_model.pt",
    )
    assert evaluation["status"] == "PASS"
    assert evaluation["evaluation_only"] is True
    assert evaluation["checkpoint_locked"] is True
    assert evaluation["test_split_used_for_selection"] is False
    assert evaluation["training_manifest_mutated"] is False
    checkpoint_after_evaluation = torch.load(
        run_dir / "checkpoints/best_model.pt",
        map_location="cpu",
        weights_only=False,
    )
    assert checkpoint_after_evaluation["selector_state"] == checkpoint_before_evaluation[
        "selector_state"
    ]
    assert (run_dir / "manifests/run_manifest.json").read_bytes() == (
        manifest_before_evaluation
    )


def test_cli_check_never_trains_or_allocates_a_run() -> None:
    result = cli.main(
        [
            "--mode",
            "check",
            "--config",
            str(SYNTHETIC_CONFIG),
            "--device",
            "cpu",
        ]
    )
    assert result["status"] == "PASS"
    assert result["optimizer_steps"] == 0
    assert "run_dir" not in result


def test_cli_train_mode_dispatch_can_be_tested_without_training(monkeypatch) -> None:
    calls = []

    def fake_run_runtime(config_path, **kwargs):
        calls.append((config_path, kwargs))
        return {"status": "BUILT_ONLY", "optimizer_steps": 0}

    monkeypatch.setattr(cli, "run_runtime", fake_run_runtime)
    result = cli.main(["--mode", "train", "--config", str(FORMAL_CONFIG)])
    assert result["status"] == "BUILT_ONLY"
    assert calls[0][1]["mode"] == "train"


def test_cli_evaluate_requires_checkpoint() -> None:
    with pytest.raises(ValueError, match="locked checkpoint"):
        cli.main(["--mode", "evaluate", "--config", str(SYNTHETIC_CONFIG)])


def test_runtime_source_is_isolated_from_existing_multidag_cores() -> None:
    runtime_root = ROOT / "scripts/runtime/multidag_cl_paper_reimplementation"
    text = "\n".join(
        path.read_text(encoding="utf-8") for path in runtime_root.glob("*.py")
    )
    assert "models.multidag_cl.unified" not in text
    assert "models.multidag_cl.paper_aligned" not in text
    assert "third_party" not in text

from copy import deepcopy
from datetime import datetime as RealDatetime
import json
from pathlib import Path

import pytest
import torch
import yaml

from models.multidag_cl.graph_core_variants import GraphCoreConfig, GraphCoreTA, VARIANTS
from models.multidag_cl.graph_core_variants.config import PARENT_BASELINE, baseline_config
from scripts.models.multidag_cl.graph_core_variants.train import main
from scripts.models.multidag_cl.ta_v0.export_dev import export_dev
from scripts.runtime.multidag_cl_graph_core_variants import formal
from scripts.runtime.multidag_cl_graph_core_variants.training import GraphCoreRuntime
from scripts.runtime.multidag_cl_ta_v0.training import PROTOCOL
from scripts.runtime.multidag_cl_paper_reimplementation.checkpoint import load_checkpoint, save_checkpoint_atomic


def raw_config(variant):
    return yaml.safe_load(Path(formal.config_path(variant)).read_text(encoding="utf-8"))


@pytest.mark.parametrize("variant", VARIANTS)
def test_formal_configs_freeze_every_non_graph_condition(variant):
    raw = raw_config(variant)
    graph = formal.validate_formal_config(raw)
    assert graph == GraphCoreConfig(variant)
    parent = yaml.safe_load(Path(formal.PARENT_CONFIG).read_text(encoding="utf-8"))
    for section in ("model", "dataset", "runtime", "training", "checkpoint"):
        assert raw[section] == parent[section]
    assert raw["training"] == PROTOCOL
    assert raw["graph_core"] == graph.resolved()


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("section,key,value", [
    ("training", "seed", 42), ("training", "max_epochs", 29),
    ("training", "batch_size", True), ("training", "gradient_clip_norm", 1.),
    ("training", "curriculum", True), ("training", "class_weight", [1]*6),
    ("training", "label_smoothing", .1), ("training", "early_stopping", True),
    ("training", "scheduler", "cosine"), ("training", "optimizer", {"learning_rate": .01}),
    ("model", "classifier_dropout", .2), ("model", "text_dim", 768),
    ("model", "diagnostic_t1024_ab1", False), ("model", "use_causal_history_gate", True),
    ("runtime", "splits", ["train", "dev", "test"]), ("dataset", "label_vocab", ["changed"]),
    ("checkpoint", "selection_metric", "test_weighted_f1"),
    ("checkpoint", "test_evaluation_count", 1), ("graph_core", "enabled", False),
    ("identity", "model_name", "multidag_combined"), ("output", "experiment_group", "other"),
])
def test_formal_protocol_mutation_fails(variant, section, key, value):
    raw = raw_config(variant)
    raw[section][key] = value
    with pytest.raises((TypeError, ValueError)):
        formal.validate_formal_config(raw)


@pytest.mark.parametrize("raw", [
    {"variant": "track+coop"}, {"variant": "dune"}, {"variant": "track", "enabled": 1},
    {"variant": "track", "track_initial_residual": float("nan")},
    {"variant": "track", "track_initial_residual": True},
    {"variant": "coop", "track_initial_residual": .2},
    {"variant": "dcn", "future_branches": True},
])
def test_invalid_or_combined_core_config_is_rejected(raw):
    with pytest.raises((TypeError, ValueError)):
        GraphCoreConfig.from_mapping(raw)


@pytest.fixture(params=tuple(VARIANTS), scope="module")
def completed(request, tmp_path_factory):
    variant = request.param
    runtime = formal.synthetic_runtime(GraphCoreConfig(variant))
    manifest = formal.execute(runtime, raw_config(variant), paths={},
        git={"git_head": "synthetic-head", "git_dirty": True},
        output_root=tmp_path_factory.mktemp(variant), experiment_date="20261118", synthetic_epochs=2)
    return variant, runtime, Path(manifest["run_dir"]), manifest


def test_real_synthetic_fit_checkpoints_manifest_resolved_and_dev_exports(completed):
    variant, runtime, directory, manifest = completed
    assert directory.parent.name == "runs"
    assert directory.parent.parent.name == f"multidag_{variant}_t1024_formal"
    assert directory.parent.parent.parent.name == "20261118"
    assert directory.name.startswith(f"multidag_{variant}_t1024_seed100_")
    assert manifest["status"] == "completed" and manifest["optimizer_steps"] == 2
    assert manifest["best_checkpoint_reloaded"]
    assert manifest["test_loader_built"] == manifest["test_evaluation_count"] == 0
    assert manifest["graph_topology_changed"] is manifest["future_edges_added"] is False
    for key, value in VARIANTS[variant].items():
        assert manifest[key] == value
    assert manifest["parent_baseline"] == PARENT_BASELINE
    assert manifest["graph_core_variant"] == variant
    assert manifest["text_dim"] == 1024 and manifest["audio_dim"] == 1582
    assert manifest["hidden_dim"] == 300 and manifest["graph_layers"] == 4
    assert manifest["parameter_count"] == sum(p.numel() for p in runtime.model.parameters())
    assert manifest["git_head"] == "synthetic-head" and manifest["git_dirty"] is True
    assert manifest["training_config"] == PROTOCOL and manifest["seed"] == 100
    assert manifest["data_identity"] == {"kind": "synthetic_train_dev_only_v1"}
    resolved = yaml.safe_load((directory / "resolved_config.yaml").read_text(encoding="utf-8"))
    assert resolved["graph_core"] == manifest["graph_core"] == GraphCoreConfig(variant).resolved()
    assert manifest == json.loads((directory / "manifests" / "run_manifest.json").read_text())
    payload = load_checkpoint(directory / "checkpoints" / "last_model.pt", torch.device("cpu"))
    assert payload["model_lineage"] == VARIANTS[variant]["model_lineage"]
    assert payload["schema"] == "multidag_graph_core_train_dev_checkpoint_v1"
    assert payload["training_config"] == PROTOCOL and payload["global_step"] == 2
    assert payload["test_evaluation_count"] == payload["test_loader_built"] == 0
    metadata = json.loads((directory / "representations" / "validation_best_h_ta.metadata.json").read_text())
    assert metadata == manifest["dev_export"]
    assert metadata["representation_name"] == f"{VARIANTS[variant]['model_lineage']}_preclassifier"
    assert metadata["rows"] == 7 and metadata["dimension"] == 1500
    assert len(list((directory / "reports").glob("validation_best_*"))) == 3
    assert not any("test" in p.name.lower() for p in directory.rglob("*"))


def test_resume_next_optimizer_step_reproduces_and_cross_identity_rejected(completed):
    variant, _, directory, _ = completed
    source = directory / "checkpoints" / "last_model.pt"
    first = formal.synthetic_runtime(GraphCoreConfig(variant))
    first.resume(source)
    first.train_step(next(iter(first.loader("train", epoch=3))))
    second = formal.synthetic_runtime(GraphCoreConfig(variant))
    second.resume(source)
    second.train_step(next(iter(second.loader("train", epoch=3))))
    for left, right in zip(first.model.parameters(), second.model.parameters()):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    other = next(v for v in VARIANTS if v != variant)
    with pytest.raises(ValueError, match="identity"):
        formal.synthetic_runtime(GraphCoreConfig(other)).resume(source)


def test_tampered_test_counter_or_core_checkpoint_rejected_before_copy(completed, tmp_path):
    variant, _, directory, _ = completed
    for field in ("test_loader_built", "test_evaluation_count", "graph_core"):
        payload = load_checkpoint(directory / "checkpoints" / "last_model.pt", torch.device("cpu"))
        payload[field] = 1 if field != "graph_core" else {"variant": "other"}
        path = tmp_path / f"{field}.pt"
        save_checkpoint_atomic(path, payload)
        runtime = formal.synthetic_runtime(GraphCoreConfig(variant))
        before = runtime.model.classifier[0].weight.detach().clone()
        with pytest.raises(ValueError, match="mismatch"):
            runtime.resume(path)
        torch.testing.assert_close(before, runtime.model.classifier[0].weight, rtol=0, atol=0)


@pytest.mark.parametrize("variant", VARIANTS)
def test_test_paths_fail_before_factory_or_generator_access(variant, tmp_path):
    runtime = GraphCoreRuntime(GraphCoreTA(GraphCoreConfig(variant)),
        lambda split: pytest.fail("dataset factory accessed"), synthetic_only=True)
    def forbidden():
        pytest.fail("batch generator accessed")
        yield
    calls = [lambda: runtime.loader("test"), lambda: runtime.evaluate(forbidden(), split="test"),
             lambda: runtime.evaluate_test(), lambda: runtime.test_loader, lambda: runtime.test_predictions,
             lambda: export_dev(runtime, forbidden(), tmp_path / "forbidden", expected_ids=["id"],
                                checkpoint_identity="id", config_identity="id", split="test")]
    for call in calls:
        with pytest.raises(ValueError, match="Test"):
            call()
    assert runtime.optimizer_steps == 0


def test_explicit_flag_and_config_validation_precede_real_data_access(monkeypatch):
    monkeypatch.setattr(formal, "load_train_dev", lambda *a, **kw: pytest.fail("real data access"))
    with pytest.raises(ValueError, match="formal-train-dev"):
        formal.run_formal(config_path="missing", asset_path="missing", model_manifest_path="missing",
                          provenance_manifest_path="missing")


@pytest.mark.parametrize("arguments", [[], ["--config", "missing"],
    ["--config", "missing", "--formal-train-dev"],
    ["--config", "missing", "--synthetic-dry-run", "--asset-path", "missing"],
    ["--config", "missing", "--synthetic-dry-run", "--output-root", "missing"]])
def test_cli_rejects_missing_authorization_or_mixed_modes(arguments):
    with pytest.raises(SystemExit):
        main(arguments)


@pytest.mark.parametrize("variant", VARIANTS)
def test_cli_synthetic_forward_backward_dev_only(variant, capsys):
    assert main(["--config", formal.config_path(variant), "--synthetic-dry-run"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "PASS" and result["synthetic_only"]
    assert result["optimizer_steps"] == result["test_evaluation_count"] == result["test_loader_built"] == 0


def test_output_collision_fails_before_optimizer_or_loader(tmp_path, monkeypatch):
    class FixedDatetime:
        @staticmethod
        def now():
            return RealDatetime(2026, 11, 18, 12, 0).astimezone()
    monkeypatch.setattr(formal, "datetime", FixedDatetime)
    graph = GraphCoreConfig("track")
    raw = raw_config("track")
    first = formal.synthetic_runtime(graph)
    result = formal.execute(first, raw, paths={}, git={}, output_root=tmp_path,
                           experiment_date="20261118", synthetic_epochs=1)
    second = formal.synthetic_runtime(graph)
    monkeypatch.setattr(second, "loader", lambda *a, **kw: pytest.fail("loader after collision"))
    with pytest.raises(FileExistsError):
        formal.execute(second, raw, paths={}, git={}, output_root=tmp_path,
                       experiment_date="20261118", synthetic_epochs=1)
    assert second.optimizer_steps == 0
    assert Path(result["run_dir"]).exists()


@pytest.mark.parametrize("variant", VARIANTS)
def test_formal_orchestration_with_injected_synthetic_data_only(variant, tmp_path, monkeypatch):
    source = formal.synthetic_runtime(GraphCoreConfig(variant))
    datasets = {split: source.dataset_factory(split) for split in ("train", "dev")}
    calls = []
    def injected(*args, **kwargs):
        calls.append(kwargs)
        return datasets, source.data_identity, {}
    monkeypatch.setattr(formal, "load_train_dev", injected)
    monkeypatch.setattr(formal, "git_identity", lambda: {"git_head": "synthetic", "git_dirty": False})
    result = formal.run_formal(formal_train_dev=True, config_path=formal.config_path(variant),
        asset_path="injected", model_manifest_path="injected", provenance_manifest_path="injected",
        output_root=tmp_path, experiment_date="20261118", one_step_smoke=True)
    assert calls == [{"config": baseline_config()}]
    assert result["optimizer_steps"] == 1 and result["status"] == "smoke_completed"
    assert result["test_loader_built"] == result["test_evaluation_count"] == 0


def test_defaults_are_resolved_and_recorded_without_mutating_input(tmp_path):
    raw = raw_config("track")
    raw["graph_core"] = {"variant": "track"}
    original = deepcopy(raw)
    graph = formal.validate_formal_config(raw)
    runtime = formal.synthetic_runtime(graph)
    result = formal.execute(runtime, raw, paths={}, git={}, output_root=tmp_path,
                            experiment_date="20261118", synthetic_epochs=1)
    assert raw == original
    resolved = yaml.safe_load((Path(result["run_dir"]) / "resolved_config.yaml").read_text())
    assert result["graph_core"] == resolved["graph_core"] == GraphCoreConfig("track").resolved()


@pytest.mark.parametrize("variant", VARIANTS)
def test_yaml_off_switch_runs_the_parent_synthetic_path(variant, tmp_path, capsys):
    raw = raw_config(variant)
    raw["graph_core"]["enabled"] = False
    config = tmp_path / "off.yaml"
    config.write_text(yaml.safe_dump(raw), encoding="utf-8")
    assert main(["--config", str(config), "--synthetic-dry-run"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["model_lineage"] == PARENT_BASELINE
    assert result["graph_core"]["enabled"] is False
    assert result["optimizer_steps"] == 0

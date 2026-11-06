"""Real tiny checkpoint IO and synthetic historical runs; no real Test assets."""

from collections.abc import Mapping
import copy
import hashlib
import json
from pathlib import Path
import shutil
from types import SimpleNamespace
from unittest.mock import Mock
import uuid

import pytest
import torch
from torch import nn
import yaml

import models.s2d.iemocap.model as model_module
from models.s2d.iemocap.config import load_config
from datasets.iemocap.s2d_faces import PREPROCESSING_VERSION
from scripts.models.s2d.iemocap import checkpoints, finalize_run as entry, train


ROOT = Path(__file__).resolve().parents[3]
CONFIG = ROOT / "configs/s2d/iemocap/v0.yaml"
BEST = .34355913724532466


@pytest.fixture
def tmp_path():
    base = (ROOT / "tmp/assistant_work/s2d_finalization_tests").resolve()
    path = base / uuid.uuid4().hex
    path.mkdir(parents=True)
    try:
        yield path
    finally:
        assert path.resolve().parent == base
        shutil.rmtree(path)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def test_production_checkpoint_real_roundtrip_is_pure_tensor_state(tmp_path):
    source = nn.Sequential(nn.Linear(3, 4), nn.BatchNorm1d(4), nn.Linear(4, 6))
    target = copy.deepcopy(source)
    with torch.no_grad():
        for parameter in target.parameters():
            parameter.zero_()
        source[1].running_mean.fill_(.25)
    path = tmp_path / "best_model.pt"
    checkpoints.save_model_weights(source, path)  # Actual production helper, no IO mocks.
    loaded = torch.load(path, map_location="cpu", weights_only=True)
    assert isinstance(loaded, Mapping) and loaded
    assert all(isinstance(key, str) and isinstance(value, torch.Tensor) for key, value in loaded.items())
    assert not {"model", "config", "label_vocab", "epoch", "optimizer", "scheduler"} & loaded.keys()
    checkpoints.reload_model_weights(target, path)
    assert source.state_dict().keys() == target.state_dict().keys()
    assert all(torch.equal(value, target.state_dict()[key]) for key, value in source.state_dict().items())


@pytest.fixture
def historical(tmp_path, monkeypatch):
    config, vocab = load_config(CONFIG)
    run_dir = tmp_path / "20261105/TEST_SYNTHETIC_seed100_es30"
    for name in ("checkpoints", "logs", "metrics", "predictions"):
        (run_dir / name).mkdir(parents=True, exist_ok=True)
    config["system"].update(device="cpu", output_dir=str(tmp_path / "TEST_SYNTHETIC_output_base"))
    config["output"].update(run_id=run_dir.name, experiment_date="20261105", root=str(run_dir))
    config["dataset"].update(data_root=str(tmp_path), manifest_path="TEST_SYNTHETIC_manifest.csv")
    (tmp_path / config["dataset"]["manifest_path"]).touch()
    for section, field in (("face", "cache_path"), ("model", "pretrained_checkpoint"),
                           ("model", "landmark_checkpoint")):
        path = tmp_path / f"TEST_SYNTHETIC_{field}"
        path.touch()
        config[section][field] = str(path)
    (run_dir / "resolved_config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    metadata = {
        "training_completed": False, "test_evaluation_count": 0, "test_split_used_for_selection": False,
        "early_stopping": config["early_stopping"], "checkpoint_selection_metric": "val_weighted_f1",
        "seed": 100, "run_id": run_dir.name, "label_vocab": vocab.as_dict(),
        "data_root": config["dataset"]["data_root"], "manifest_path": config["dataset"]["manifest_path"],
        "face_cache_path": config["face"]["cache_path"], "face_preprocessing_version": PREPROCESSING_VERSION,
        "checkpoints": {name: {"path": config["model"][field], "expected_sha256": config["model"][f"{field}_sha256"],
                                "sha256_verified": True}
                        for name, field in (("s2d", "pretrained_checkpoint"), ("landmark", "landmark_checkpoint"))}}
    write_json(run_dir / "run_metadata.json", metadata)
    write_json(run_dir / "logs/label_vocab.json", vocab.as_dict())
    write_json(run_dir / "metrics/coverage.json", {"face_preprocessing_version": PREPROCESSING_VERSION})
    selection = train.EarlyStopping(True, 10, 8)
    records = []
    for epoch, score in enumerate([.1, .2, .25, .28, .30, .32, BEST] + [.3] * 8, 1):
        selection.update(epoch, score)
        records.append({"epoch": epoch, "train": {"loss": .5}, "dev": {"weighted_f1": score},
                        "best_epoch": selection.best_epoch, "best_val_weighted_f1": selection.best_val_weighted_f1,
                        "epochs_without_improvement": selection.epochs_without_improvement})
    (run_dir / "logs/epochs.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    source = nn.Linear(3, 6)
    checkpoint = run_dir / "checkpoints/best_model_weights_only.pt"
    checkpoints.save_model_weights(source, checkpoint)
    # Keep the old container in place; the finalizer must never auto-convert/overwrite it.
    torch.save({"model": source.state_dict(), "epoch": 7, "val_weighted_f1": BEST,
                "label_vocab": vocab.as_dict(), "config": config}, run_dir / "checkpoints/best_model.pt")
    events = []

    class Target(nn.Linear):
        def load_state_dict(self, state, strict=True):
            events.append(("reload", strict))
            return super().load_state_dict(state, strict=strict)

    target = Target(3, 6)
    builder = Mock(return_value=target)
    monkeypatch.setattr(model_module, "build_s2d", builder)
    inputs = Mock(return_value="TEST_SYNTHETIC_inputs")
    monkeypatch.setattr(entry, "prepare_evaluation_inputs", inputs)

    def loader(config, inputs):
        assert events == [("reload", True)]
        assert (run_dir / entry.ATTEMPT_FILE).is_file()
        events.append(("loader", "test"))
        return "TEST_SYNTHETIC_loader"

    test_loader = Mock(side_effect=loader)
    monkeypatch.setattr(entry, "make_test_loader", test_loader)

    def evaluate(model, loader, device, current_vocab, *, collect):
        assert model is target and loader == "TEST_SYNTHETIC_loader" and collect
        assert events[-1] == ("loader", "test")
        assert all(torch.equal(value, target.state_dict()[key]) for key, value in source.state_dict().items())
        events.append(("evaluate", "test"))
        report = train.classification_report([0, 1], [0, 0], current_vocab)
        report.update(loss=.5, sample_count=2)
        batches = [{"utterance_id": ["TEST_SYNTHETIC_test_001"], "label_index": torch.tensor([0]),
                    "h_v": torch.zeros(1, 768), "logits_v": torch.ones(1, 6), "prob_v": torch.ones(1, 6) / 6}]
        return report, batches

    evaluate_mock = Mock(side_effect=evaluate)
    monkeypatch.setattr(train, "run_epoch", evaluate_mock)
    monkeypatch.setattr(train, "train", Mock(side_effect=AssertionError("Finalizer must not train")))
    return SimpleNamespace(run_dir=run_dir, checkpoint=checkpoint, config=config, vocab=vocab,
                           metadata=metadata, records=records, events=events, builder=builder,
                           loader=test_loader, evaluate=evaluate_mock, inputs=inputs)


def test_historical_state_recovered_without_metadata_results(historical):
    h = historical
    before = (h.run_dir / "run_metadata.json").read_bytes()
    *_, state, _ = entry.validate_run(h.run_dir, h.checkpoint)
    assert state == {"stop_reason": "early_stopping", "stopped_epoch": 15, "best_epoch": 7,
                     "best_val_weighted_f1": BEST, "min_epochs": 10, "patience": 8,
                     "epochs_without_improvement_at_stop": 8}
    assert (h.run_dir / "run_metadata.json").read_bytes() == before
    assert not (h.run_dir / entry.ATTEMPT_FILE).exists()
    h.builder.assert_not_called()
    h.loader.assert_not_called()


@pytest.mark.parametrize("representations", [False, True])
def test_finalization_success_test_exactly_once_exports_and_atomic_metadata(historical, representations, monkeypatch):
    h = historical
    h.config["export"]["representations"] = representations
    (h.run_dir / "resolved_config.yaml").write_text(yaml.safe_dump(h.config), encoding="utf-8")
    original_mixed = (h.run_dir / "checkpoints/best_model.pt").read_bytes()
    real_replace = entry.os.replace
    replacements = []

    def replace(source, destination):
        metadata = entry.read_json(destination)
        assert metadata["training_completed"] is False and metadata["test_evaluation_count"] == 0
        assert all((h.run_dir / name).is_file() for name in entry.FINAL_OUTPUTS[:2])
        if representations:
            assert (h.run_dir / entry.FINAL_OUTPUTS[2]).is_file()
        replacements.append(destination)
        real_replace(source, destination)

    monkeypatch.setattr(entry.os, "replace", replace)
    # Exercise the real CLI and its explicit checkpoint argument.
    entry.main(["--run-dir", str(h.run_dir), "--checkpoint", str(h.checkpoint)])
    assert h.evaluate.call_count == h.loader.call_count == 1
    metadata = entry.read_json(h.run_dir / "run_metadata.json")
    assert metadata["training_completed"] is True and metadata["test_evaluation_count"] == 1
    assert metadata["best_epoch"] == 7 and metadata["stopped_epoch"] == 15
    assert metadata["best_val_weighted_f1"] == BEST and metadata["epochs_without_improvement_at_stop"] == 8
    assert metadata["test_split_used_for_selection"] is False
    assert metadata["finalization_mode"] == "existing_run"
    assert metadata["finalization_checkpoint_path"] == str(h.checkpoint)
    assert metadata["finalization_checkpoint_sha256"] == hashlib.sha256(h.checkpoint.read_bytes()).hexdigest()
    report = entry.read_json(h.run_dir / "metrics/test_final.json")
    assert {"acc", "weighted_f1", "macro_f1", "uar", "per_class", "confusion_matrix", "loss"} <= report.keys()
    assert len(report["confusion_matrix"]) == len(report["per_class"]) == 6
    assert report["selected_epoch"] == 7 and report["selected_val_weighted_f1"] == BEST
    assert "TEST_SYNTHETIC_test_001" in (h.run_dir / "predictions/test_predictions.csv").read_text()
    assert (h.run_dir / entry.FINAL_OUTPUTS[2]).exists() is representations
    if representations:
        payload = torch.load(h.run_dir / entry.FINAL_OUTPUTS[2], weights_only=True)
        assert payload["batches"][0]["h_v"].shape == (1, 768)
    assert len(replacements) == 1
    assert (h.run_dir / "checkpoints/best_model.pt").read_bytes() == original_mixed
    with pytest.raises(ValueError, match="test_evaluation_count"):
        entry.finalize_run(h.run_dir, h.checkpoint)
    assert h.evaluate.call_count == 1


@pytest.mark.parametrize("field,value", [
    ("test_evaluation_count", 1), ("test_evaluation_count", -1), ("test_evaluation_count", False),
    ("test_evaluation_count", None), ("test_split_used_for_selection", True),
    ("training_completed", True), ("best_epoch", 8), ("stopped_epoch", 14),
    ("best_val_weighted_f1", .9), ("seed", 42), ("checkpoint_selection_metric", "test_weighted_f1"),
    ("manifest_path", "TEST_SYNTHETIC_other.csv"), ("face_preprocessing_version", "old"),
])
def test_inconsistent_metadata_aborts_before_model_and_test(historical, field, value):
    h = historical
    h.metadata[field] = value
    write_json(h.run_dir / "run_metadata.json", h.metadata)
    with pytest.raises(ValueError, match="Inconsistent"):
        entry.finalize_run(h.run_dir, h.checkpoint)
    h.builder.assert_not_called()
    h.loader.assert_not_called()
    h.evaluate.assert_not_called()
    assert not (h.run_dir / entry.ATTEMPT_FILE).exists()


@pytest.mark.parametrize("name", ["run_metadata.json", "resolved_config.yaml", "logs/epochs.jsonl",
                                  "logs/label_vocab.json", "metrics/coverage.json", "checkpoints/best_model_weights_only.pt"])
def test_missing_required_file_aborts_before_test(historical, name):
    h = historical
    (h.run_dir / name).unlink()
    with pytest.raises(FileNotFoundError):
        entry.finalize_run(h.run_dir, h.checkpoint)
    h.builder.assert_not_called()
    h.loader.assert_not_called()
    h.evaluate.assert_not_called()


@pytest.mark.parametrize("name", [*entry.FINAL_OUTPUTS, entry.ATTEMPT_FILE])
def test_prior_final_output_or_interrupted_attempt_refuses_overwrite(historical, name):
    h = historical
    path = h.run_dir / name
    path.parent.mkdir(exist_ok=True)
    path.write_text("TEST_SYNTHETIC_preserve", encoding="utf-8")
    with pytest.raises(FileExistsError):
        entry.finalize_run(h.run_dir, h.checkpoint)
    assert path.read_text() == "TEST_SYNTHETIC_preserve"
    h.loader.assert_not_called()
    h.evaluate.assert_not_called()


@pytest.mark.parametrize("kind", ["incomplete", "duplicate_epoch", "counter", "best_score", "extra_after_stop", "nonfinite"])
def test_bad_epoch_history_aborts_before_test(historical, kind):
    h = historical
    if kind == "incomplete":
        h.records.pop()
    elif kind == "duplicate_epoch":
        h.records[4]["epoch"] = 4
    elif kind == "counter":
        h.records[-1]["epochs_without_improvement"] = 7
    elif kind == "best_score":
        h.records[-1]["best_val_weighted_f1"] = .9
    elif kind == "nonfinite":
        h.records[-1]["dev"]["weighted_f1"] = float("nan")
    else:
        h.records.append({**h.records[-1], "epoch": 16})
    (h.run_dir / "logs/epochs.jsonl").write_text("".join(json.dumps(r) + "\n" for r in h.records), encoding="utf-8")
    with pytest.raises(ValueError):
        entry.finalize_run(h.run_dir, h.checkpoint)
    h.loader.assert_not_called()
    h.evaluate.assert_not_called()


@pytest.mark.parametrize("kind", ["missing", "unexpected", "shape"])
def test_strict_reload_failure_aborts_before_test_construction(historical, kind):
    h = historical
    state = torch.load(h.checkpoint, weights_only=True)
    if kind == "missing":
        state.pop("bias")
    elif kind == "unexpected":
        state["TEST_SYNTHETIC_extra"] = torch.zeros(1)
    else:
        state["weight"] = torch.zeros(1, 1)
    torch.save(state, h.checkpoint)
    with pytest.raises(RuntimeError):
        entry.finalize_run(h.run_dir, h.checkpoint)
    assert h.events == [("reload", True)]
    h.loader.assert_not_called()
    h.evaluate.assert_not_called()
    assert not (h.run_dir / entry.ATTEMPT_FILE).exists()
    assert entry.read_json(h.run_dir / "run_metadata.json")["training_completed"] is False


@pytest.mark.parametrize("unsafe", [False, True])
def test_historical_mixed_checkpoint_never_uses_unsafe_fallback(historical, monkeypatch, unsafe):
    h = historical
    if unsafe:
        torch.save({"model": {"weight": torch.ones(6, 3)}, "config": {"path": Path("TEST_SYNTHETIC")}},
                   h.run_dir / "checkpoints/best_model.pt")
    loader = Mock(wraps=torch.load)
    monkeypatch.setattr(torch, "load", loader)
    with pytest.raises(Exception):
        entry.finalize_run(h.run_dir)  # No implicit selection of the converted sibling.
    assert loader.call_count == 1
    assert loader.call_args.kwargs == {"map_location": "cpu", "weights_only": True}
    h.loader.assert_not_called()
    h.evaluate.assert_not_called()


@pytest.mark.parametrize("stage", ["test", "predictions", "representations", "metadata"])
def test_test_or_export_failure_keeps_metadata_incomplete_and_blocks_retry(historical, monkeypatch, stage):
    h = historical
    if stage == "test":
        h.evaluate.side_effect = RuntimeError("TEST_SYNTHETIC_test_failure")
    elif stage == "predictions":
        monkeypatch.setattr(train, "export_predictions", Mock(side_effect=RuntimeError("TEST_SYNTHETIC_export_failure")))
    elif stage == "representations":
        h.config["export"]["representations"] = True
        (h.run_dir / "resolved_config.yaml").write_text(yaml.safe_dump(h.config), encoding="utf-8")
        monkeypatch.setattr(torch, "save", Mock(side_effect=RuntimeError("TEST_SYNTHETIC_representation_failure")))
    else:
        monkeypatch.setattr(entry.os, "replace", Mock(side_effect=RuntimeError("TEST_SYNTHETIC_metadata_failure")))
    before = (h.run_dir / "run_metadata.json").read_bytes()
    with pytest.raises(RuntimeError, match="TEST_SYNTHETIC"):
        entry.finalize_run(h.run_dir, h.checkpoint)
    assert h.evaluate.call_count == 1
    assert (h.run_dir / "run_metadata.json").read_bytes() == before
    assert (h.run_dir / entry.ATTEMPT_FILE).is_file()
    with pytest.raises(FileExistsError):
        entry.finalize_run(h.run_dir, h.checkpoint)
    assert h.evaluate.call_count == 1
    assert not list(h.run_dir.glob(".finalization_metadata_*"))


def test_runtime_overrides_use_existing_resolver_without_rewriting_provenance(historical):
    h = historical
    replacement = h.run_dir.parent / "TEST_SYNTHETIC_relocated_manifest.csv"
    replacement.touch()
    before = (h.run_dir / "resolved_config.yaml").read_bytes()
    result = entry.validate_run(h.run_dir, h.checkpoint, overrides={"manifest_path": str(replacement)})
    assert result[2]["dataset"]["manifest_path"] == str(replacement)
    assert (h.run_dir / "resolved_config.yaml").read_bytes() == before
    assert not Path(h.config["system"]["output_dir"]).exists()
    with pytest.raises(ValueError, match="only existing input"):
        entry.validate_run(h.run_dir, h.checkpoint, overrides={"epochs": 1})


def test_coverage_mismatch_aborts_before_test(historical, monkeypatch):
    h = historical
    h.inputs.side_effect = ValueError("TEST_SYNTHETIC_coverage_mismatch")
    with pytest.raises(ValueError, match="coverage"):
        entry.finalize_run(h.run_dir, h.checkpoint)
    h.builder.assert_not_called()
    h.loader.assert_not_called()
    h.evaluate.assert_not_called()


def test_test_once_gate_is_rechecked_after_slow_preflight(historical):
    h = historical

    def competing_test(*args):
        h.metadata["test_evaluation_count"] = 1
        write_json(h.run_dir / "run_metadata.json", h.metadata)
        return "TEST_SYNTHETIC_inputs"

    h.inputs.side_effect = competing_test
    with pytest.raises(ValueError, match="Test-once gate"):
        entry.finalize_run(h.run_dir, h.checkpoint)
    h.loader.assert_not_called()
    h.evaluate.assert_not_called()


@pytest.mark.parametrize("corruption", [None, "counts", "v_identity", "face_usable", "missing_cache_id"])
def test_real_manifest_cache_preflight_and_loader_preserve_population(tmp_path, monkeypatch, corruption):
    from datasets.iemocap.s2d_faces import FaceBBoxCache, FaceCacheEntry
    from datasets.iemocap.s2d_dataset import IEMOCAPS2DDataset

    config, vocab = load_config(CONFIG)
    config["dataset"].update(data_root=str(tmp_path), columns={}, manifest_path=str(
        ROOT / "tests/models/s2d/fixtures/synthetic_manifest.csv"))
    config["train"]["num_workers"] = 0
    cache_path = tmp_path / "TEST_SYNTHETIC_cache.json"
    config["face"]["cache_path"] = str(cache_path)
    entries = {}
    for split in ("train", "dev", "test"):
        uid = f"SYNTHETIC_{split}_001"
        usable = split != "dev"
        entries[uid] = FaceCacheEntry(
            utterance_id=uid, frame_indices=tuple(range(16)), frame_sizes=((100, 80),) * 16,
            selected_bboxes=((20., 20., 60., 60.) if usable else None,) * 16,
            detection_probabilities=(.99 if usable else None,) * 16,
            detection_valid=(usable,) * 16, fallback_used=(False,) * 16,
            reasons=("TEST_SYNTHETIC_detected" if usable else "TEST_SYNTHETIC_all_failed",) * 16,
            valid_detection_count=16 if usable else 0, face_usable=usable)
    if corruption == "missing_cache_id":
        entries.pop("SYNTHETIC_test_001")
    FaceBBoxCache(entries).save(cache_path)
    coverage = {
        "counts": {"train": {"universe": 1, "v_available": 1, "v_unavailable": 0},
                   "dev": {"universe": 1, "v_available": 1, "v_unavailable": 0},
                   "test": {"universe": 2, "v_available": 1, "v_unavailable": 1}},
        "v_unavailable_utterance_ids": ["SYNTHETIC_test_no_video"],
        "face_coverage": {split: {"face_usable": 0 if split == "dev" else 1,
                                 "face_unusable_utterance_ids": ["SYNTHETIC_dev_001"] if split == "dev" else [],
                                 "valid_detection_count": 0 if split == "dev" else 16,
                                 "fallback_frame_count": 0} for split in ("train", "dev", "test")}}
    if corruption == "counts":
        coverage["counts"]["test"]["universe"] = 3
    elif corruption == "v_identity":
        coverage["v_unavailable_utterance_ids"] = []
    elif corruption == "face_usable":
        coverage["face_coverage"]["dev"]["face_usable"] = 1
    monkeypatch.setattr(IEMOCAPS2DDataset, "__getitem__", Mock(side_effect=AssertionError("Video must not open in preflight")))
    if corruption is not None:
        with pytest.raises(ValueError):
            entry.prepare_evaluation_inputs(config, vocab, coverage)
        return
    inputs = entry.prepare_evaluation_inputs(config, vocab, coverage)
    assert len(inputs[0].records) == 4
    loader = entry.make_test_loader(config, inputs)
    assert [row.utterance_id for row in loader.dataset.records] == ["SYNTHETIC_test_001"]
    assert loader.num_workers == 0 and not loader.drop_last
    assert loader.dataset.face_cache.entries["SYNTHETIC_dev_001"].face_usable is False


@pytest.mark.parametrize("enabled,epochs", [(True, 30), (False, 30)])
def test_recovery_accepts_max_epoch_completion_and_keeps_earliest_tie(enabled, epochs):
    config, _ = load_config(CONFIG)
    config["early_stopping"]["enabled"] = enabled
    metadata = {"early_stopping": config["early_stopping"], "checkpoint_selection_metric": "val_weighted_f1"}
    selection = train.EarlyStopping(enabled, 10, 8)
    records = []
    for epoch in range(1, epochs + 1):
        score = epoch / 100 if enabled else .2
        selection.update(epoch, score)
        records.append({"epoch": epoch, "dev": {"weighted_f1": score}, "best_epoch": selection.best_epoch,
                        "best_val_weighted_f1": selection.best_val_weighted_f1,
                        "epochs_without_improvement": selection.epochs_without_improvement})
    state = entry.recover_training_state(config, metadata, records)
    assert state["stop_reason"] == "max_epochs" and state["stopped_epoch"] == 30
    assert state["best_epoch"] == (30 if enabled else 1)

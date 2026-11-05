"""Synthetic control-flow tests only: no real data, weights, detector or training."""

import csv
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

import datasets.iemocap.s2d_dataset as dataset_module
import datasets.iemocap.s2d_faces as face_module
import datasets.iemocap.s2d_manifest as manifest_module
import models.s2d.iemocap.model as model_module
from models.s2d.iemocap.config import load_config
import scripts.models.s2d.iemocap.train as entry


ROOT = Path(__file__).resolve().parents[3]
CONFIG = ROOT / "configs/s2d/iemocap/v0.yaml"


@pytest.fixture
def tmp_path():
    # Ordinary workspace directories avoid the managed Windows pytest-temp ACL issue.
    base = (ROOT / "tmp/assistant_work/s2d_early_stopping_tests").resolve()
    path = base / uuid.uuid4().hex
    path.mkdir(parents=True)
    try:
        yield path
    finally:
        assert path.resolve().parent == base
        shutil.rmtree(path)


def test_improvement_resets_patience():
    state = entry.EarlyStopping(True, 10, 8)
    assert state.update(1, .2)
    assert not state.update(2, .1)
    assert state.epochs_without_improvement == 1
    assert state.update(3, .3)
    assert (state.best_epoch, state.best_val_weighted_f1, state.epochs_without_improvement) == (3, .3, 0)


def test_no_improvement_increments_and_ties_keep_earliest_best():
    state = entry.EarlyStopping(True, 10, 8)
    state.update(1, .3)
    for epoch, score in enumerate([.2, .3, .1], 2):
        assert not state.update(epoch, score)
        assert state.epochs_without_improvement == epoch - 1
        assert (state.best_epoch, state.best_val_weighted_f1) == (1, .3)


def test_min_epochs_gate_counts_non_improving_warmup():
    state = entry.EarlyStopping(True, 10, 2)
    for epoch in range(1, 11):
        state.update(epoch, .3)
        assert state.should_stop(epoch) is (epoch == 10)
    assert state.epochs_without_improvement == 9


def test_patience_triggers_on_exact_boundary():
    state = entry.EarlyStopping(True, 2, 8)
    for epoch in range(1, 10):
        state.update(epoch, .3)
        assert state.should_stop(epoch) is (epoch == 9)


@pytest.mark.parametrize("score", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_dev_score_aborts(score):
    with pytest.raises(FloatingPointError, match="Dev Weighted-F1"):
        entry.EarlyStopping(True, 10, 8).update(1, score)


def test_frozen_formal_config():
    config, _ = load_config(CONFIG)
    assert {key: config["train"][key] for key in ("learning_rate", "weight_decay", "epochs")} == {
        "learning_rate": 1e-5, "weight_decay": .05, "epochs": 30}
    assert config["early_stopping"] == {
        "enabled": True, "min_epochs": 10, "patience": 8,
        "metric": "weighted_f1", "split": "dev", "mode": "max"}
    assert "scheduler" not in config and "scheduler" not in config["train"]


def test_check_config_opens_only_config_and_vocab_and_creates_nothing(monkeypatch, capsys):
    real_open = Path.open
    allowed = {CONFIG.resolve(), (ROOT / "configs/_shared/data/iemocap/benchmark_label_vocab_v1.yaml").resolve()}
    opened = []

    def guarded_open(path, *args, **kwargs):
        assert path.resolve() in allowed, f"Unexpected static-check file access: {path}"
        opened.append(path.resolve())
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)
    monkeypatch.setattr(Path, "mkdir", Mock(side_effect=AssertionError("Static check created output")))
    monkeypatch.setattr(torch, "load", Mock(side_effect=AssertionError("Static check opened weights")))
    monkeypatch.setattr(entry, "train", Mock(side_effect=AssertionError("Static check entered train")))
    entry.main(["--config", str(CONFIG), "--check-config"])
    summary = json.loads(capsys.readouterr().out)
    assert summary["status"] == "CONFIG_VALID_ONLY" and not summary["data_or_weights_opened"]
    assert summary["training"] == {"learning_rate": 1e-5, "weight_decay": .05, "epochs": 30}
    assert summary["early_stopping"]["patience"] == 8
    assert set(opened) == allowed


@pytest.mark.parametrize("field,value", [
    ("enabled", "true"), ("min_epochs", 0), ("min_epochs", True), ("min_epochs", 31),
    ("patience", 0), ("patience", -1), ("patience", 1.5), ("patience", True),
    ("metric", "acc"), ("metric", "macro_f1"), ("split", "test"), ("mode", "min"),
])
def test_invalid_early_stopping_config_rejected(tmp_path, field, value):
    config, _ = load_config(CONFIG)
    config["early_stopping"][field] = value
    path = tmp_path / "TEST_SYNTHETIC_invalid.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(ValueError, match="early.stopping"):
        load_config(path)


@pytest.fixture
def harness(tmp_path, monkeypatch):
    """Run the production train() orchestration with mocked epochs and tiny checkpoints."""
    config, vocab = load_config(CONFIG)
    config["system"].update(device="cpu", output_dir=str(tmp_path / "TEST_SYNTHETIC_outputs"))
    config["dataset"].update(data_root=str(tmp_path), manifest_path="TEST_SYNTHETIC_missing.csv")
    config["output"].update(experiment_date="20261105", run_id="TEST_SYNTHETIC_early_stopping")
    events, saved_epochs = [], []
    state = SimpleNamespace(epoch=0, reloaded=False, scores=[], corrupt=False)

    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.marker = nn.Parameter(torch.zeros(1))
            self.load_reports = {name: {"sha256_verified": True} for name in ("s2d", "landmark")}

        def load_state_dict(self, weights, strict=True):
            events.append(("reload", strict))
            result = super().load_state_dict(weights, strict=strict)
            state.reloaded = True
            return result

    model = TinyModel()
    monkeypatch.setattr(entry, "validate_runtime_paths", Mock())
    monkeypatch.setattr(model_module, "build_s2d", lambda *args: model)
    monkeypatch.setattr(face_module, "build_face_processor", lambda config: object())
    cache = SimpleNamespace(entries={}, require_ids=lambda ids: None)
    monkeypatch.setattr(face_module.FaceBBoxCache, "load", lambda path: cache)
    manifest = SimpleNamespace(records=[], counts=lambda: {}, for_split=lambda split: [])
    monkeypatch.setattr(manifest_module, "S2DManifest", lambda *args: manifest)

    class FakeDataset:
        def __init__(self, manifest, split, *args, **kwargs):
            events.append(("dataset", split))
            if split == "test":
                assert state.reloaded, "Test dataset must be constructed after strict best reload"
            self.split = split

        def __len__(self):
            return 1

    monkeypatch.setattr(dataset_module, "IEMOCAPS2DDataset", FakeDataset)
    monkeypatch.setattr(torch.utils.data, "DataLoader", lambda dataset, **kwargs: dataset)
    real_save, real_load = torch.save, torch.load

    def save(payload, path):
        if Path(path).name == "best_model.pt":
            saved_epochs.append(payload["epoch"])
            if state.corrupt:
                payload = {**payload, "model": {}}
        real_save(payload, path)

    def load(path, **kwargs):
        events.append(("load", Path(path).name, kwargs))
        return real_load(path, **kwargs)

    monkeypatch.setattr(torch, "save", save)
    monkeypatch.setattr(torch, "load", load)

    def run_epoch(current_model, loader, device, current_vocab, optimizer=None, *, collect=False):
        assert current_model is model
        split = loader.split
        events.append(("epoch", split))
        report = entry.classification_report([0, 1], [0, 0], vocab)
        report.update(loss=.5, sample_count=2)
        if split == "train":
            assert optimizer is not None and not collect
            assert not state.reloaded
            state.epoch += 1
            with torch.no_grad():
                model.marker.fill_(state.epoch)
        elif split == "dev":
            assert optimizer is None and not collect
            report["weighted_f1"] = state.scores[state.epoch - 1]
            # Every other Dev metric improves even when the selection score stalls.
            report.update(acc=state.epoch / 30, macro_f1=state.epoch / 30, uar=state.epoch / 30)
        else:
            assert split == "test" and collect and optimizer is None and state.reloaded
            assert int(model.marker.item()) == saved_epochs[-1]
            batches = [{"utterance_id": ["TEST_SYNTHETIC_final"], "label_index": torch.tensor([0]),
                        "h_v": torch.zeros(1, 8), "logits_v": torch.ones(1, 6),
                        "prob_v": torch.ones(1, 6) / 6}]
            return report, batches
        return report, []

    monkeypatch.setattr(entry, "run_epoch", run_epoch)
    return SimpleNamespace(config=config, vocab=vocab, events=events, state=state,
                           saved_epochs=saved_epochs, model=model,
                           run_dir=Path(config["system"]["output_dir"]) / "20261105/TEST_SYNTHETIC_early_stopping")


@pytest.mark.parametrize("scores,min_epochs,patience,enabled,reason,best,stopped,count", [
    ([.1, .2, .25, .28, .30, .32, .34355913724532466] + [.3] * 23,
     10, 8, True, "early_stopping", 7, 15, 8),
    ([.3] * 30, 10, 2, True, "early_stopping", 1, 10, 9),
    ([i / 100 for i in range(1, 31)], 10, 8, True, "max_epochs", 30, 30, 0),
    ([.1, .2, .2, .19, .3] + [.3] * 25, 10, 8, True, "early_stopping", 5, 13, 8),
    ([.3] * 30, 10, 8, False, "max_epochs", 1, 30, 29),
    ([i / 100 for i in range(1, 28)] + [.27] * 3, 10, 8, True, "max_epochs", 27, 30, 3),
    ([i / 100 for i in range(1, 23)] + [.22] * 8, 10, 8, True, "early_stopping", 22, 30, 8),
])
@pytest.mark.parametrize("export", [False, True])
def test_training_selection_test_once_and_completion_metadata(
        harness, scores, min_epochs, patience, enabled, reason, best, stopped, count, export):
    h = harness
    h.state.scores = scores
    h.config["early_stopping"].update(enabled=enabled, min_epochs=min_epochs, patience=patience)
    h.config["export"]["representations"] = export
    entry.train(h.config, h.vocab)
    expected_saves, best_score = [], -float("inf")
    for epoch, score in enumerate(scores[:stopped], 1):
        if score > best_score:
            expected_saves.append(epoch)
            best_score = score
    assert h.saved_epochs == expected_saves  # Equal scores never overwrite best_model.pt.
    epochs = [event[1] for event in h.events if event[0] == "epoch"]
    assert epochs == ["train", "dev"] * stopped + ["test"]
    assert [event for event in h.events if event[0] == "reload"] == [("reload", True)]
    assert [event for event in h.events if event[0] == "load"] == [
        ("load", "best_model.pt", {"map_location": "cpu", "weights_only": True})]
    assert h.events.index(("reload", True)) < h.events.index(("dataset", "test"))
    assert h.state.epoch == stopped and int(h.model.marker.item()) == best
    metadata = json.loads((h.run_dir / "run_metadata.json").read_text(encoding="utf-8"))
    expected = {"training_completed": True, "stop_reason": reason, "stopped_epoch": stopped,
                "best_epoch": best, "best_val_weighted_f1": scores[best - 1],
                "min_epochs": min_epochs, "patience": patience,
                "epochs_without_improvement_at_stop": count,
                "test_evaluation_count": 1, "test_split_used_for_selection": False}
    assert {key: metadata[key] for key in expected} == expected
    records = [json.loads(row) for row in (h.run_dir / "logs/epochs.jsonl").read_text().splitlines()]
    assert len(records) == stopped
    assert records[-1]["epochs_without_improvement"] == count
    assert records[-1]["best_epoch"] == best
    required_metrics = {"acc", "weighted_f1", "macro_f1", "uar", "per_class", "confusion_matrix", "loss"}
    for record in records:
        assert required_metrics <= record["dev"].keys()
        assert len(record["dev"]["confusion_matrix"]) == len(record["dev"]["per_class"]) == 6
        assert {"precision", "recall", "f1", "support"} <= record["dev"]["per_class"]["hap"].keys()
    final = json.loads((h.run_dir / "metrics/test_final.json").read_text())
    assert required_metrics <= final.keys()
    assert final["selected_epoch"] == best and final["selected_val_weighted_f1"] == scores[best - 1]
    with (h.run_dir / "predictions/test_predictions.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 1 and rows[0]["utterance_id"] == "TEST_SYNTHETIC_final"
    representation = h.run_dir / "representations/test_visual_outputs.pt"
    assert representation.exists() is export
    if export:
        payload = torch.load(representation, weights_only=True)
        assert payload["split"] == "test"
        assert {"h_v", "logits_v", "prob_v"} <= payload["batches"][0].keys()


def test_strict_reload_failure_prevents_test_and_completion(harness):
    h = harness
    h.state.scores = [.3] * 30
    h.state.corrupt = True
    with pytest.raises(RuntimeError, match="Missing key"):
        entry.train(h.config, h.vocab)
    assert ("reload", True) in h.events
    assert ("dataset", "test") not in h.events and ("epoch", "test") not in h.events
    metadata = json.loads((h.run_dir / "run_metadata.json").read_text(encoding="utf-8"))
    assert metadata["training_completed"] is False
    assert metadata["test_evaluation_count"] == 0
    assert not (h.run_dir / "metrics/test_final.json").exists()

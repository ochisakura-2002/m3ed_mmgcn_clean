from copy import deepcopy
import csv
import json
from pathlib import Path
import pickle

import numpy as np
import pytest
import torch
import yaml

from datasets.iemocap.ta_v0_dataset import SCHEMA, VERSION, TAV0Dataset
from models.multidag_cl.ta_v0 import LABEL_VOCAB, TAV0, TAV0Config
from scripts.models.multidag_cl.ta_v0.export_dev import PREDICTION_COLUMNS, export_dev
from scripts.models.multidag_cl.ta_v0.train import MODEL_INPUTS, main, synthetic_package
from scripts.runtime.multidag_cl_paper_reimplementation.checkpoint import (
    ValidationCheckpointSelector, load_checkpoint, save_checkpoint_atomic,
)
from scripts.runtime.multidag_cl_ta_v0 import formal
from scripts.runtime.multidag_cl_ta_v0.training import (
    PROTOCOL, TrainDevRuntime, classification_report, masked_cross_entropy, seed_everything,
)

FORMAL_CONFIG = ("configs/multidag_cl/ta_v0/iemocap/causal_context/"
                 "asymmetric_merc_v1/formal_train_dev.yaml")


@pytest.fixture(autouse=True, scope="module")
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def synthetic_runtime():
    seed_everything("cpu")
    package = synthetic_package()
    datasets = {s: TAV0Dataset(*package, split=s, expected_rows=len(package[1]))
                for s in ("train", "dev")}
    return TrainDevRuntime(TAV0(), datasets.__getitem__, synthetic_only=True,
                          data_identity={"sha256": {"asset": "synthetic-package-v1"}})


def dev_ids(runtime):
    return [uid for batch in runtime.loader("dev") for dialogue in batch["utterance_ids"] for uid in dialogue]


@pytest.fixture(scope="module")
def completed(tmp_path_factory):
    runtime = synthetic_runtime()
    directory = tmp_path_factory.mktemp("runtime") / "run"
    manifest = formal.fit(runtime, directory, resolved_config={"synthetic": True},
                          manifest={"synthetic_only": True, "formal_train_dev": False},
                          expected_dev_ids=dev_ids(runtime), synthetic_epochs=2)
    return runtime, directory, manifest


def test_masked_ce_ignores_poisoned_padding_and_gradient():
    logits = torch.randn(2, 3, 6, requires_grad=True)
    mask = torch.tensor([[True, True, False], [True, False, False]])
    targets = torch.tensor([[0, 1, 999], [2, -456, 73]])
    with torch.no_grad():
        logits[~mask] = float("nan")
    loss, count = masked_cross_entropy(logits, targets, mask)
    assert count == 3 and torch.isfinite(loss)
    torch.testing.assert_close(loss, torch.nn.functional.cross_entropy(logits[mask], targets[mask]))
    loss.backward()
    assert torch.count_nonzero(logits.grad[~mask]) == 0


@pytest.mark.parametrize("case", ["empty", "label", "loss"])
def test_masked_ce_rejects_invalid_valid_targets(case):
    logits = torch.zeros(1, 1, 6)
    targets, mask = torch.zeros(1, 1, dtype=torch.int64), torch.ones(1, 1, dtype=torch.bool)
    if case == "empty":
        mask[:] = False
    elif case == "label":
        targets[:] = 6
    else:
        logits[:] = float("nan")
    with pytest.raises(ValueError):
        masked_cross_entropy(logits, targets, mask)


def test_optimizer_step_changes_parameters_and_clips(monkeypatch):
    runtime = synthetic_runtime()
    before = runtime.model.classifier[-1].weight.detach().clone()
    calls = []
    original = torch.nn.utils.clip_grad_norm_
    def clip(parameters, limit, **kwargs):
        parameters = list(parameters)
        result = original(parameters, limit, **kwargs)
        post_norm = torch.linalg.vector_norm(torch.stack([p.grad.norm() for p in parameters]))
        calls.append((limit, float(post_norm)))
        return result
    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", clip)
    result = runtime.train_step(next(iter(runtime.loader("train"))))
    assert result["valid_target_count"] == 5 and result["optimizer_steps"] == 1
    assert np.isfinite(result["loss"]) and np.isfinite(result["gradient_norm"])
    assert not torch.equal(before, runtime.model.classifier[-1].weight)
    assert calls[0][0] == 5.0 and calls[0][1] <= 5.00001


def test_nonfinite_gradients_fail_before_optimizer():
    runtime = synthetic_runtime()
    handle = runtime.model.classifier[-1].weight.register_hook(lambda g: g * float("nan"))
    with pytest.raises(ValueError, match="gradients"):
        runtime.train_step(next(iter(runtime.loader("train"))))
    handle.remove()
    assert runtime.optimizer_steps == 0


def test_selector_uses_only_weighted_f1_loss_then_earlier_epoch():
    selector = ValidationCheckpointSelector()
    assert selector.update(epoch=5, val_weighted_f1=.5, val_loss=2)
    assert selector.update(epoch=6, val_weighted_f1=.6, val_loss=9)
    assert selector.update(epoch=7, val_weighted_f1=.6, val_loss=1)
    assert not selector.update(epoch=8, val_weighted_f1=.6, val_loss=1)
    assert selector.update(epoch=4, val_weighted_f1=.6, val_loss=1)
    assert not selector.update(epoch=1, val_weighted_f1=.59, val_loss=0)
    assert selector.best.epoch == 4


def test_metrics_use_all_six_classes_and_global_support():
    report = classification_report([0, 0, 1], [0, 1, 1], loss=1)
    metrics = report["metrics"]
    assert metrics["accuracy"] == pytest.approx(2 / 3)
    assert metrics["weighted_f1"] == pytest.approx(2 / 3)
    assert metrics["uar"] == pytest.approx((.5 + 1) / 6)
    assert metrics["macro_f1"] == pytest.approx((2 / 3 + 2 / 3) / 6)
    assert [row["support"] for row in report["per_class"]] == [2, 1, 0, 0, 0, 0]
    assert np.asarray(report["confusion_matrix"]).shape == (6, 6)


def test_best_last_weights_logs_manifest_and_exports(completed):
    runtime, directory, manifest = completed
    checkpoints = directory / "checkpoints"
    best = load_checkpoint(checkpoints / "best_model.pt", torch.device("cpu"))
    last = load_checkpoint(checkpoints / "last_model.pt", torch.device("cpu"))
    assert best["epoch"] == runtime.selector.best.epoch
    assert last["epoch"] == 2 and last["global_step"] == 2
    assert last["best_checkpoint"]["epoch"] == best["epoch"]
    assert best["training_config"] == PROTOCOL and best["label_vocab"] == list(LABEL_VOCAB)
    assert best["model_config"] == yaml.safe_load(
        Path(FORMAL_CONFIG).read_text(encoding="utf-8"))["model"]
    assert best["optimizer_state_dict"]["state"]
    assert best["test_evaluation_count"] == best["test_loader_built"] == 0
    weights = torch.load(checkpoints / "best_model_weights_only.pt", map_location="cpu", weights_only=True)
    restored, structured = TAV0().eval(), TAV0().eval()
    restored.load_state_dict(weights, strict=True)
    structured.load_state_dict(best["model_state_dict"], strict=True)
    batch = next(iter(runtime.loader("dev")))
    with torch.no_grad():
        reference = runtime.model.eval()(**{k: batch[k] for k in MODEL_INPUTS})
        for model in (restored, structured):
            reloaded = model(**{k: batch[k] for k in MODEL_INPUTS})
            torch.testing.assert_close(reference.logits_ta, reloaded.logits_ta, rtol=0, atol=0)
            torch.testing.assert_close(reference.h_ta, reloaded.h_ta, rtol=0, atol=0)
    with (directory / "logs" / "epoch_metrics.tsv").open(newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        assert set(formal.EPOCH_COLUMNS) == set(reader.fieldnames)
        assert [int(row["epoch"]) for row in reader] == [1, 2]
    assert manifest["optimizer_steps"] == manifest["optimizer_steps_this_invocation"] == 2
    assert manifest["status"] == "completed" and manifest["best_checkpoint_reloaded"]
    assert manifest["test_loader_built"] == manifest["test_evaluation_count"] == 0
    assert json.loads((directory / "manifests" / "run_manifest.json").read_text()) == manifest
    with (directory / "predictions" / "validation_best_predictions.tsv").open(newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        assert tuple(reader.fieldnames) == PREDICTION_COLUMNS
        rows = list(reader)
    with np.load(directory / "representations" / "validation_best_h_ta.npz", allow_pickle=False) as data:
        assert data["h_ta"].shape == (4, 1500) and np.isfinite(data["h_ta"]).all()
        assert data["utterance_ids"].tolist() == [row["utterance_id"] for row in rows]
        assert len(set(data["utterance_ids"])) == 4
        assert data["labels"].tolist() == [int(row["true_label"]) for row in rows]
    probabilities = np.array([[float(row[f"prob_{i}"]) for i in range(6)] for row in rows])
    assert np.isfinite(probabilities).all()
    np.testing.assert_allclose(probabilities.sum(1), 1, atol=1e-6)
    metadata = json.loads((directory / "representations" / "validation_best_h_ta.metadata.json").read_text())
    assert metadata["representation_name"] == "ta_v0_contextual_preclassifier"
    assert metadata["dimension"] == 1500 and metadata["label_vocab"] == list(LABEL_VOCAB)
    assert metadata["checkpoint_identity"] == formal.sha256_file(checkpoints / "best_model.pt")
    assert metadata["config_identity"] == formal.sha256_file(directory / "resolved_config.yaml")
    assert metadata["prediction_representation_order_identical"]
    assert len(list((directory / "reports").glob("validation_best_*"))) == 3
    assert not any("test" in path.name.lower() for path in directory.rglob("*"))


def test_existing_run_is_never_overwritten(completed):
    _, directory, _ = completed
    marker = (directory / "resolved_config.yaml").read_bytes()
    runtime = synthetic_runtime()
    with pytest.raises(FileExistsError):
        formal.fit(runtime, directory, resolved_config={}, manifest={},
                   expected_dev_ids=dev_ids(runtime), synthetic_epochs=1)
    assert runtime.optimizer_steps == 0
    assert (directory / "resolved_config.yaml").read_bytes() == marker


@pytest.mark.parametrize("arguments", [[], ["--config", "missing"],
    ["--synthetic-dry-run", "--asset-path", "missing"],
    ["--formal-train-dev"], ["--formal-train-dev", "--synthetic-dry-run"]])
def test_cli_requires_explicit_formal_flag_and_paths(arguments):
    with pytest.raises(SystemExit):
        main(arguments)


def test_no_real_data_access_without_formal_flag(monkeypatch):
    monkeypatch.setattr(formal, "load_train_dev", lambda *a: pytest.fail("feature retrieval"))
    with pytest.raises(ValueError, match="formal-train-dev"):
        formal.run_formal(config_path="missing", asset_path="missing", model_manifest_path="missing",
                          provenance_manifest_path="missing", output_dir="missing")
    with pytest.raises(ValueError, match="formal-train-dev"):
        TrainDevRuntime(TAV0(), lambda split: pytest.fail("factory called"))


@pytest.mark.parametrize("access", ["loader", "evaluate", "evaluate_test", "test_loader", "test_predictions", "export"])
def test_test_paths_fail_before_retrieval(access, tmp_path):
    runtime = TrainDevRuntime(TAV0(), lambda split: pytest.fail("factory called"), synthetic_only=True)
    def batches():
        pytest.fail("feature retrieval")
        yield
    with pytest.raises(ValueError, match="Test"):
        if access == "loader":
            runtime.loader("test")
        elif access == "evaluate":
            runtime.evaluate(batches(), split="test")
        elif access in ("test_loader", "test_predictions"):
            getattr(runtime, access)
        elif access == "export":
            export_dev(runtime, batches(), tmp_path / "forbidden", expected_ids=["id"],
                       checkpoint_identity="id", config_identity="config", split="test")
        else:
            runtime.evaluate_test()


def test_seeded_shuffle_independent_of_rng():
    first, second = synthetic_runtime(), synthetic_runtime()
    class Dataset:
        split = "train"
        def __len__(self):
            return 100
    first.dataset_factory = second.dataset_factory = lambda split: Dataset()
    a = list(first.loader("train", epoch=2).sampler)
    torch.rand(100)
    b = list(second.loader("train", epoch=2).sampler)
    c = list(second.loader("train", epoch=3).sampler)
    assert a == b and a != c and sorted(a) == list(range(100))
    runtime = synthetic_runtime()
    assert isinstance(runtime.loader("dev").sampler, torch.utils.data.SequentialSampler)


@pytest.mark.parametrize("field", ["model_config", "label_vocab", "seed", "training_config", "data_identity",
                                   "optimizer_state_dict", "test_evaluation_count"])
def test_resume_identity_mismatch_fails_before_parameters_change(completed, tmp_path, field):
    _, directory, _ = completed
    payload = load_checkpoint(directory / "checkpoints" / "last_model.pt", torch.device("cpu"))
    if field == "optimizer_state_dict":
        payload[field]["param_groups"][0]["lr"] = .9
    elif field == "test_evaluation_count":
        payload[field] = 1
    else:
        payload[field] = "mismatch"
    path = tmp_path / "bad.pt"
    save_checkpoint_atomic(path, payload)
    runtime = synthetic_runtime()
    before = runtime.model.classifier[-1].weight.detach().clone()
    with pytest.raises(ValueError, match="mismatch"):
        runtime.resume(path)
    torch.testing.assert_close(before, runtime.model.classifier[-1].weight, rtol=0, atol=0)
    assert runtime.optimizer_steps == 0


def test_resume_reproduces_next_optimizer_step(completed):
    _, directory, _ = completed
    source = directory / "checkpoints" / "last_model.pt"
    first, second = synthetic_runtime(), synthetic_runtime()
    first.resume(source)
    first.train_step(next(iter(first.loader("train", epoch=3))))
    second.resume(source)
    second.train_step(next(iter(second.loader("train", epoch=3))))
    assert first.optimizer_steps == second.optimizer_steps == 3
    for a, b in zip(first.model.parameters(), second.model.parameters()):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_resume_preserves_earlier_best_when_next_epoch_is_worse(tmp_path, monkeypatch):
    runtime = synthetic_runtime()
    checkpoints = tmp_path / "old"
    runtime.train_step(next(iter(runtime.loader("train", epoch=1))))
    runtime.save_epoch(checkpoints, epoch=1, dev={"metrics": {"weighted_f1": .9, "loss": 1.0}})
    inherited = deepcopy(runtime.model.state_dict())
    runtime.train_step(next(iter(runtime.loader("train", epoch=2))))
    runtime.save_epoch(checkpoints, epoch=2, dev={"metrics": {"weighted_f1": .8, "loss": .1}})
    resumed = synthetic_runtime()
    resumed.resume(checkpoints / "last_model.pt")
    dev = resumed.evaluate()
    dev["metrics"].update(weighted_f1=0.0, loss=99.0)
    monkeypatch.setattr(resumed, "evaluate", lambda: dev)
    directory = tmp_path / "resumed"
    manifest = formal.fit(resumed, directory, resolved_config={}, manifest={},
                          expected_dev_ids=dev_ids(resumed), synthetic_epochs=3)
    assert manifest["optimizer_steps_this_invocation"] == 1 and manifest["optimizer_steps"] == 3
    assert manifest["best_epoch"] == 1
    best = load_checkpoint(directory / "checkpoints" / "best_model.pt", torch.device("cpu"))
    assert best["epoch"] == 1 and best["global_step"] == 1
    for key, value in inherited.items():
        if isinstance(value, torch.Tensor):
            torch.testing.assert_close(value, best["model_state_dict"][key], rtol=0, atol=0)
            torch.testing.assert_close(value, resumed.model.state_dict()[key], rtol=0, atol=0)


def test_one_step_smoke_is_bounded_and_cannot_resume(tmp_path):
    runtime = synthetic_runtime()
    directory = tmp_path / "smoke"
    manifest = formal.fit(runtime, directory, resolved_config={}, manifest={},
                          expected_dev_ids=dev_ids(runtime), one_step_smoke=True)
    assert manifest["optimizer_steps"] == 1 and manifest["status"] == "smoke_completed"
    assert not manifest["training_finished"]
    with pytest.raises(ValueError, match="completed epoch"):
        synthetic_runtime().resume(directory / "checkpoints" / "last_model.pt")


@pytest.mark.parametrize("section,key,value", [
    ("training", "seed", 42), ("training", "max_epochs", 1), ("training", "class_weight", [1] * 6),
    ("training", "label_smoothing", .1), ("training", "curriculum", True),
    ("training", "early_stopping", True), ("training", "batch_size", True),
    ("runtime", "splits", ["train", "dev", "test"]), ("model", "classifier_dropout", .2),
    ("checkpoint", "test_evaluation_count", 1),
])
def test_formal_protocol_is_frozen(section, key, value):
    raw = yaml.safe_load(Path(FORMAL_CONFIG).read_text(encoding="utf-8"))
    assert formal.validate_formal_config(raw) == TAV0Config()
    raw[section][key] = value
    with pytest.raises(ValueError, match="frozen"):
        formal.validate_formal_config(raw)


def write_package(directory, asset, model_rows, provenance_rows):
    directory.mkdir()
    paths = [directory / "iemocap_asym_ta_v1.pkl", directory / "model_input_manifest_v1.csv",
             directory / "iemocap_asym_manifest_v1.csv"]
    with paths[0].open("wb") as stream:
        pickle.dump(asset, stream)
    for path, rows in zip(paths[1:], (model_rows, provenance_rows)):
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    return paths


@pytest.fixture(scope="module")
def portable_files(tmp_path_factory):
    """Full-size synthetic identity universe; Test has no feature arrays."""
    dialogues, model, provenance = {}, [], []
    splits = {s: [] for s in ("train", "dev", "test")}
    number = 0
    for split, utterances, dialogue_count in (("train", 4830, 100), ("dev", 980, 20), ("test", 1623, 31)):
        for index in range(dialogue_count):
            number += 1
            did = f"Ses01F_impro{number:03d}"
            n = utterances // dialogue_count + int(index < utterances % dialogue_count)
            ids = [f"{did}_{'F' if i % 2 == 0 else 'M'}{i:03d}" for i in range(n)]
            speakers = [uid.rsplit("_", 1)[1][0] for uid in ids]
            labels = [i % 6 for i in range(n)]
            splits[split].append(did)
            dialogues[did] = {
                "utterance_ids": ids, "speakers": speakers,
                "benchmark_labels": [LABEL_VOCAB[i] for i in labels], "benchmark_label_ids": labels,
                "sentences": ["synthetic"] * n,
                "text_features": np.zeros((n, 768), dtype=np.float32) if split != "test" else "FORBIDDEN",
                "audio_features": np.zeros((n, 1582), dtype=np.float32) if split != "test" else "FORBIDDEN",
            }
            for i, uid in enumerate(ids):
                row = {"utterance_id": uid, "dialogue_id": did, "ta_dialogue_id": did,
                       "speaker": speakers[i], "split": split, "benchmark_label": LABEL_VOCAB[labels[i]],
                       "benchmark_label_id": labels[i], "ta_row_index": i}
                model.append(dict(row, session="Session1", ta_asset_relative_path="ta/iemocap_asym_ta_v1.pkl"))
                provenance.append(dict(row, canonical_dialogue_order=n - i))
    asset = {"schema": SCHEMA, "dataset_version": VERSION,
             "feature_spec": {"text": {"dimension": 768}, "audio_ta": {"dimension": 1582}},
             "label_vocab": {"itos": list(LABEL_VOCAB)}, "dialogues": dialogues, "splits": splits}
    paths = write_package(tmp_path_factory.mktemp("portable") / "input", asset, model, provenance)
    return paths


def test_formal_data_counts_hashes_canonical_order_and_test_feature_isolation(portable_files):
    paths = portable_files
    before = [formal.sha256_file(p) for p in paths]
    datasets, identity, recorded_paths = formal.load_train_dev(*paths)
    assert set(datasets) == {"train", "dev"}
    assert identity["package_utterances"] == 7433
    assert identity["split_counts"] == formal.SPLIT_COUNTS
    assert list(identity["sha256"].values()) == before
    assert list(recorded_paths.values()) == [p.as_posix() for p in paths]
    assert [formal.sha256_file(p) for p in paths] == before
    assert identity["dataset_schema"] == SCHEMA and identity["dataset_version"] == VERSION
    assert identity["label_vocab"] == list(LABEL_VOCAB)
    assert datasets["train"][0]["canonical_order"] == tuple(range(1, 50))
    assert datasets["train"][0]["ta_row_indices"] == tuple(reversed(range(49)))
    assert len(datasets["train"]) == 100 and len(datasets["dev"]) == 20


def test_formal_data_rejects_changed_split_counts(portable_files, tmp_path):
    with portable_files[0].open("rb") as stream:
        asset = pickle.load(stream)
    rows = []
    for path in portable_files[1:]:
        with path.open(encoding="utf-8", newline="") as stream:
            rows.append(list(csv.DictReader(stream)))
    dialogue = asset["splits"]["train"].pop()
    asset["splits"]["dev"].append(dialogue)
    for manifest in rows:
        for row in manifest:
            if row["ta_dialogue_id"] == dialogue:
                row["split"] = "dev"
    paths = write_package(tmp_path / "changed", asset, *rows)
    with pytest.raises(ValueError, match="counts mismatch"):
        formal.load_train_dev(*paths)


def test_default_980_row_formal_dev_export(portable_files, tmp_path):
    datasets, identity, _ = formal.load_train_dev(*portable_files)
    seed_everything("cpu")
    runtime = TrainDevRuntime(TAV0(), datasets.__getitem__, synthetic_only=True, data_identity=identity)
    directory = tmp_path / "full_dev"
    expected = dev_ids(runtime)
    metadata = export_dev(runtime, runtime.loader("dev"), directory, expected_ids=expected,
                          checkpoint_identity="synthetic-untrained-980", config_identity="synthetic-config",
                          formal_layout=True)
    assert metadata["rows"] == metadata["unique_utterance_ids"] == 980
    assert metadata["duplicate_ids"] == metadata["padding_rows"] == 0
    with np.load(directory / "representations" / "validation_best_h_ta.npz", allow_pickle=False) as arrays:
        assert arrays["h_ta"].shape == (980, 1500) and np.isfinite(arrays["h_ta"]).all()
        ids = arrays["utterance_ids"].tolist()
    with (directory / "predictions" / "validation_best_predictions.tsv").open(newline="") as stream:
        predictions = list(csv.DictReader(stream, delimiter="\t"))
    assert ids == [row["utterance_id"] for row in predictions] == sorted(expected)
    probabilities = np.array([[float(row[f"prob_{i}"]) for i in range(6)] for row in predictions])
    assert np.isfinite(probabilities).all()
    np.testing.assert_allclose(probabilities.sum(1), 1, atol=1e-6)
    assert runtime.optimizer_steps == 0


def test_formal_orchestration_and_cli_use_only_injected_synthetic_data(tmp_path, monkeypatch):
    source = synthetic_runtime()
    datasets = {split: source.dataset_factory(split) for split in ("train", "dev")}
    paths = {"asset": "synthetic.pkl", "model_manifest": "synthetic-model.csv",
             "provenance_manifest": "synthetic-provenance.csv"}
    monkeypatch.setattr(formal, "load_train_dev", lambda *args: (datasets, source.data_identity, paths))
    group = tmp_path / "group"
    result = formal.run_formal(
        formal_train_dev=True, config_path=FORMAL_CONFIG, asset_path="synthetic.pkl",
        model_manifest_path="synthetic-model.csv", provenance_manifest_path="synthetic-provenance.csv",
        output_dir=group, device="cpu", one_step_smoke=True)
    assert result["run_id"].startswith("ta_v0_seed100_")
    assert result["optimizer_steps"] == 1 and result["test_evaluation_count"] == 0
    assert result["model_parameter_count"] == 5975510
    assert result["data_paths"] == paths and result["data_identity"] == source.data_identity
    assert result["training_config"] == PROTOCOL
    assert result["checkpoint_tiebreak_rules"] == ["weighted_f1_desc", "dev_loss_asc", "epoch_asc"]
    calls = []
    monkeypatch.setattr(formal, "run_formal", lambda **kwargs: calls.append(kwargs) or {"status": "injected"})
    assert main(["--formal-train-dev", "--config", FORMAL_CONFIG, "--asset-path", "synthetic.pkl",
                 "--model-manifest-path", "model.csv", "--provenance-manifest-path", "provenance.csv",
                 "--output-dir", str(tmp_path / "cli"), "--smoke-one-step"]) == 0
    assert calls[0]["formal_train_dev"] is True and calls[0]["one_step_smoke"] is True

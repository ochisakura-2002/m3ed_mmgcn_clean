"""Synthetic-only frozen representation export checks; no training or real assets."""

import csv
from dataclasses import replace
import json

import numpy as np
import pytest
import torch

from datasets.iemocap.ta_v0_dataset import TAV0Dataset
from models.multidag_cl.ta_v0 import TAV0, TAV0Config
from scripts.models.multidag_cl.ta_v0 import export_representations as exporter
from scripts.models.multidag_cl.ta_v0.export_dev import export_dev, PREDICTION_COLUMNS
from scripts.models.multidag_cl.ta_v0.train import MODEL_INPUTS, synthetic_package, synthetic_runner
from scripts.runtime.multidag_cl_paper_reimplementation.checkpoint import save_checkpoint_atomic
from scripts.runtime.multidag_cl_ta_v0.training import checkpoint_identity


@pytest.fixture(autouse=True, scope="module")
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


class ForbiddenTestFeatures:
    def __getitem__(self, key):
        raise AssertionError("Test feature access forbidden")

    def __array__(self, *args, **kwargs):
        raise AssertionError("Test feature access forbidden")


@pytest.fixture(scope="module")
def frozen(tmp_path_factory):
    package = synthetic_package()
    asset = package[0]
    # Dialogue order deliberately differs from lexicographic ID order; storage
    # rows within each dialogue are already scrambled by the shared fixture.
    for split in ("train", "dev"):
        asset["splits"][split].reverse()
    for did in asset["splits"]["test"]:
        for key in ("text_features", "audio_features"):
            asset["dialogues"][did][key] = ForbiddenTestFeatures()
    datasets = {split: TAV0Dataset(*package, split=split, expected_rows=len(package[1]))
                for split in ("train", "dev")}
    identity = {"synthetic_only": True}
    torch.manual_seed(100)
    model = TAV0()
    checkpoint = {
        **checkpoint_identity(model.config, identity),
        "model_state_dict": model.state_dict(), "epoch": 18, "epoch_complete": True,
        "best_dev_weighted_f1": .5, "best_dev_loss": 1.0,
        "selector": {"best": {"epoch": 18, "val_weighted_f1": .5, "val_loss": 1.0},
                     "test_split_used_for_selection": False},
        "test_loader_built": 0, "test_evaluation_count": 0, "git_head": "synthetic",
    }
    run = tmp_path_factory.mktemp("frozen")
    path = run / "checkpoints" / "best_model.pt"
    save_checkpoint_atomic(path, checkpoint)
    return datasets, identity, checkpoint, path


def provenance():
    return {"checkpoint_path": "synthetic/best_model.pt", "checkpoint_best_epoch": 18,
            "seed": 100, "git_commit": "synthetic"}


@pytest.mark.parametrize("split,count,prefix", [
    ("train", 5, "train"), ("dev", 4, "validation"), ("validation", 4, "validation"),
])
def test_exports_exact_canonical_ids_labels_features_logits_probs(frozen, tmp_path, split, count, prefix):
    datasets, identity, _, path = frozen
    canonical_split = exporter.normalize_split(split)
    dataset = datasets[canonical_split]
    model, _ = exporter.load_frozen_model(path, data_identity=identity)
    # Verify true inference mode on every export batch, including Train.
    modes = []
    hook = model.register_forward_pre_hook(
        lambda module, inputs: modes.append((module.training, torch.is_grad_enabled())))
    directory = tmp_path / split
    result = exporter.export_representations(
        model, dataset, directory, split=split, expected_rows=count, provenance=provenance())
    hook.remove()
    assert modes and all(mode == (False, False) for mode in modes)
    assert all(parameter.grad is None for parameter in model.parameters())
    expected_ids, expected_dialogues, expected_labels = [], [], []
    expected_features, expected_logits, expected_probs = [], [], []
    with torch.inference_mode():
        from datasets.iemocap.ta_v0_dataset import ta_v0_collate
        for sample in dataset:
            expected_ids.extend(sample["utterance_ids"])
            expected_dialogues.extend([sample["dialogue_id"]] * len(sample["utterance_ids"]))
            expected_labels.extend(sample["labels"].tolist())
            batch = ta_v0_collate([sample])
            output = model(**{key: batch[key] for key in MODEL_INPUTS})
            expected_features.append(output.h_ta[0].numpy())
            expected_logits.append(output.logits_ta[0].numpy())
            expected_probs.append(output.prob_ta[0].numpy())
    assert expected_ids != sorted(expected_ids)
    with np.load(directory / f"{prefix}_best_h_ta.npz", allow_pickle=False) as arrays:
        assert set(arrays.files) == {"utterance_id", "dialogue_id", "label", "h_ta", "logits_ta", "prob_ta"}
        assert arrays["utterance_id"].tolist() == expected_ids
        assert arrays["dialogue_id"].tolist() == expected_dialogues
        assert arrays["label"].tolist() == expected_labels
        assert arrays["h_ta"].shape == (count, 1500)
        assert arrays["logits_ta"].shape == arrays["prob_ta"].shape == (count, 6)
        np.testing.assert_allclose(arrays["h_ta"], np.concatenate(expected_features), atol=1e-6)
        np.testing.assert_allclose(arrays["logits_ta"], np.concatenate(expected_logits), atol=1e-6)
        np.testing.assert_allclose(arrays["prob_ta"], np.concatenate(expected_probs), atol=1e-6)
        probabilities = arrays["prob_ta"].copy()
        predictions = arrays["logits_ta"].argmax(-1)
    with (directory / f"{prefix}_best_predictions.tsv").open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        assert tuple(reader.fieldnames) == exporter.PREDICTION_COLUMNS
        rows = list(reader)
    assert [row["utterance_id"] for row in rows] == expected_ids
    assert [row["dialogue_id"] for row in rows] == expected_dialogues
    assert [int(row["true_label"]) for row in rows] == expected_labels
    assert [int(row["pred_label"]) for row in rows] == predictions.tolist()
    np.testing.assert_allclose(
        [[float(row[f"prob_{i}"]) for i in range(6)] for row in rows], probabilities)
    manifest = json.loads((directory / "representation_export_manifest.json").read_text())
    assert manifest == result
    assert manifest["test_access"] is False and manifest["row_count"] == count
    assert manifest["split"] == canonical_split
    assert manifest["h_ta_dim"] == 1500 and manifest["num_classes"] == 6
    assert manifest["checkpoint_best_epoch"] == 18 and manifest["seed"] == 100
    original = (directory / f"{prefix}_best_h_ta.npz").read_bytes()
    with pytest.raises(FileExistsError):
        exporter.export_representations(
            model, dataset, directory, split=split, expected_rows=count, provenance=provenance())
    assert (directory / f"{prefix}_best_h_ta.npz").read_bytes() == original


def test_legacy_dev_contract_is_unchanged(tmp_path):
    runner = synthetic_runner()
    batches = list(runner.loader("dev"))
    ids = [uid for batch in batches for dialogue in batch["utterance_ids"] for uid in dialogue]
    directory = tmp_path / "legacy"
    export_dev(runner, reversed(batches), directory, expected_ids=ids, expected_rows=4,
               checkpoint_identity="synthetic", config_identity="synthetic")
    with np.load(directory / "h_ta.npz", allow_pickle=False) as arrays:
        assert set(arrays.files) == {"utterance_ids", "labels", "h_ta", "split"}
        assert arrays["utterance_ids"].tolist() == sorted(ids)
    with (directory / "predictions.csv").open(newline="") as stream:
        assert tuple(csv.reader(stream).__next__()) == PREDICTION_COLUMNS
    assert json.loads((directory / "metadata.json").read_text())["split"] == "dev"


@pytest.mark.parametrize("split", ["test", "validation_test", "all"])
def test_forbidden_split_fails_before_any_access(monkeypatch, tmp_path, split, capsys):
    def forbidden(*args, **kwargs):
        raise AssertionError("data/checkpoint access before split rejection")
    monkeypatch.setattr(exporter, "load_train_dev", forbidden)
    monkeypatch.setattr(exporter, "load_frozen_model", forbidden)
    directory = tmp_path / "forbidden"
    with pytest.raises(SystemExit) as error:
        exporter.main(["--split", split, "--checkpoint", "missing.pt",
                       "--asset-path", "missing.pkl", "--model-manifest-path", "missing.csv",
                       "--provenance-manifest-path", "missing.csv", "--output-dir", str(directory)])
    assert error.value.code == 2
    assert "invalid choice" in capsys.readouterr().err
    with pytest.raises(ValueError, match="Test access forbidden"):
        exporter.export_representations(None, None, directory, split=split, provenance={})
    assert not directory.exists()


@pytest.mark.parametrize("split", ["train", "dev"])
@pytest.mark.parametrize("case", ["duplicate_id", "empty_id", "label", "h_ta", "logits", "prob", "prob_sum", "argmax"])
def test_invalid_rows_fail_before_writing(frozen, tmp_path, split, case):
    datasets, identity, _, path = frozen
    model, _ = exporter.load_frozen_model(path, data_identity=identity)
    dataset = datasets[split]

    class CorruptDataset:
        def __init__(self):
            self.split = split
        def __len__(self):
            return len(dataset)
        def __getitem__(self, index):
            sample = dict(dataset[index])
            if case in ("duplicate_id", "empty_id") and index == 1:
                ids = list(sample["utterance_ids"])
                ids[0] = dataset[0]["utterance_ids"][0] if case == "duplicate_id" else " "
                sample["utterance_ids"] = tuple(ids)
            if case == "label":
                sample["labels"] = sample["labels"].clone()
                sample["labels"][0] = 6
            return sample

    def corrupt(module, args, output):
        if case in ("h_ta", "logits", "prob"):
            key = {"logits": "logits_ta", "prob": "prob_ta"}.get(case, case)
            value = getattr(output, key).clone()
            value[0, 0, 0] = float("nan")
            return replace(output, **{key: value})
        if case == "prob_sum":
            return replace(output, prob_ta=output.prob_ta * .5)
        if case == "argmax":
            value = torch.zeros_like(output.prob_ta)
            value.scatter_(-1, ((output.logits_ta.argmax(-1) + 1) % 6).unsqueeze(-1), 1)
            return replace(output, prob_ta=value)
        return output

    hook = model.register_forward_hook(corrupt)
    directory = tmp_path / "invalid"
    try:
        with pytest.raises(ValueError):
            exporter.export_representations(
                model, CorruptDataset(), directory, split=split,
                expected_rows=5 if split == "train" else 4, provenance=provenance())
    finally:
        hook.remove()
    assert not directory.exists()


@pytest.mark.parametrize("split", ["train", "dev"])
def test_real_row_counts_are_enforced(frozen, tmp_path, split):
    datasets, identity, _, path = frozen
    model, _ = exporter.load_frozen_model(path, data_identity=identity)
    directory = tmp_path / "wrong_count"
    with pytest.raises(ValueError, match=f"expected={4830 if split == 'train' else 980}"):
        exporter.export_representations(
            model, datasets[split], directory, split=split, provenance=provenance())
    assert not directory.exists()


def test_off_chg_frozen_identity_loads_without_optimizer(frozen, monkeypatch):
    _, identity, checkpoint, path = frozen
    original = exporter.sha256_file(path)
    def forbidden(*args, **kwargs):
        raise AssertionError("optimizer construction forbidden")
    monkeypatch.setattr(torch.optim, "AdamW", forbidden)
    model, loaded = exporter.load_frozen_model(path, data_identity=identity)
    assert loaded["epoch"] == 18 and loaded["seed"] == 100
    assert model.config == TAV0Config()
    assert "use_causal_history_gate" not in loaded["model_config"]
    assert not any("history_gate" in key for key in model.state_dict())
    assert not model.training and not any(p.requires_grad for p in model.parameters())
    for key, value in model.state_dict().items():
        expected = checkpoint["model_state_dict"][key]
        if isinstance(value, torch.Tensor):
            torch.testing.assert_close(value, expected, rtol=0, atol=0)
        else:
            assert value == expected
    assert exporter.sha256_file(path) == original


@pytest.mark.parametrize("field,value", [("seed", 101), ("epoch", 19), ("test_evaluation_count", 1)])
def test_wrong_frozen_checkpoint_is_rejected(frozen, tmp_path, field, value):
    _, identity, checkpoint, _ = frozen
    bad = dict(checkpoint, **{field: value})
    path = tmp_path / "bad.pt"
    save_checkpoint_atomic(path, bad)
    with pytest.raises(ValueError):
        exporter.load_frozen_model(path, data_identity=identity)


@pytest.mark.parametrize("source,split", [("--run-dir", None), ("--checkpoint", "train"),
                                         ("--checkpoint", "validation")])
def test_cli_uses_common_data_and_checkpoint_runtime(frozen, monkeypatch, tmp_path, source, split):
    datasets, identity, _, path = frozen
    original = exporter.sha256_file(path)
    calls = []
    def load(*args):
        calls.append(args)
        return datasets, identity, {}
    monkeypatch.setattr(exporter, "load_train_dev", load)
    monkeypatch.setattr(exporter, "SPLIT_COUNTS",
                        {"train": {"utterances": 5}, "dev": {"utterances": 4}})
    monkeypatch.setattr(exporter, "git_identity",
                        lambda: {"git_head": "synthetic-exporter", "git_dirty": True})
    def forbidden(*args, **kwargs):
        raise AssertionError("training/optimizer forbidden")
    monkeypatch.setattr(torch.optim, "AdamW", forbidden)
    directory = tmp_path / "cli"
    args = [source, str(path.parents[1] if source == "--run-dir" else path),
            "--asset-path", "synthetic.pkl", "--model-manifest-path", "model.csv",
            "--provenance-manifest-path", "provenance.csv", "--output-dir", str(directory)]
    if split is not None:
        args += ["--split", split]
    assert exporter.main(args) == 0
    assert calls == [("synthetic.pkl", "model.csv", "provenance.csv")]
    manifest = json.loads((directory / "representation_export_manifest.json").read_text())
    assert manifest["split"] == ("train" if split == "train" else "dev")
    assert manifest["checkpoint_sha256"] == original
    assert manifest["git_commit"] == "synthetic-exporter"
    assert manifest["test_access"] is False
    assert exporter.sha256_file(path) == original


def test_existing_formal_best_export_reuses_strict_reload_without_training(frozen, tmp_path):
    from scripts.models.multidag_cl.ta_v0.train import DevelopmentRunner
    from scripts.runtime.multidag_cl_ta_v0 import formal
    datasets, _, checkpoint, _ = frozen
    directory = tmp_path / "formal"
    save_checkpoint_atomic(directory / "checkpoints" / "best_model.pt", checkpoint)
    (directory / "resolved_config.yaml").write_text("synthetic: true\n", encoding="utf-8")
    runner = DevelopmentRunner(TAV0(), datasets.__getitem__)
    validated = []
    def validate(payload):
        assert payload["epoch"] == 18 and payload["seed"] == 100
        validated.append(True)
    runner.validate_checkpoint = validate
    ids = [uid for sample in datasets["dev"] for uid in sample["utterance_ids"]]
    metadata, _ = formal.export_best(runner, directory, expected_ids=ids, expected_rows=4)
    assert validated == [True] and metadata["split"] == "dev"
    with np.load(directory / "representations" / "validation_best_h_ta.npz",
                 allow_pickle=False) as arrays:
        assert set(arrays.files) == {"utterance_ids", "labels", "h_ta", "split"}
        assert arrays["utterance_ids"].tolist() == sorted(ids)
    with (directory / "predictions" / "validation_best_predictions.tsv").open(
            encoding="utf-8", newline="") as stream:
        assert tuple(csv.reader(stream, delimiter="\t").__next__()) == PREDICTION_COLUMNS

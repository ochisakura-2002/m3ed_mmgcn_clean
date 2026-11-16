"""AB-1 single-factor guards and synthetic-only readiness; no real assets or training."""

import csv
from dataclasses import replace
from pathlib import Path
import pickle

import numpy as np
import pytest
import torch
import yaml

from datasets.iemocap.ta_v0_dataset import TAV0Dataset, ta_v0_collate
from models.multidag_cl.ta_v0 import TAV0, TAV0Config
from scripts.models.multidag_cl.ta_v0.train import MODEL_INPUTS, synthetic_package
from scripts.runtime.multidag_cl_ta_v0 import formal
from scripts.runtime.multidag_cl_ta_v0.training import PROTOCOL, TrainDevRuntime

BASE_CONFIG = ("configs/multidag_cl/ta_v0/iemocap/causal_context/"
               "asymmetric_merc_v1/formal_train_dev.yaml")
AB1_CONFIG = BASE_CONFIG.replace("formal_train_dev.yaml", "formal_train_dev_t1024_ab1.yaml")


@pytest.fixture(autouse=True, scope="module")
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def raw_config(path=AB1_CONFIG):
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def ab1():
    return formal.validate_formal_config(raw_config())


def test_baseline_default_and_explicit_off_preserve_checkpoint_identity():
    raw = raw_config(BASE_CONFIG)
    config = formal.validate_formal_config(raw)
    assert config.text_dim == 768 and config.diagnostic_t1024_ab1 is False
    assert config.checkpoint_config() == raw["model"]
    assert config.lineage == "ta_v0"
    raw["model"]["diagnostic_t1024_ab1"] = False
    assert formal.validate_formal_config(raw) == config
    assert TAV0().get_extra_state()["config"] == config.checkpoint_config()


def test_ab1_yaml_has_exactly_two_semantic_changes_and_preserves_formatting():
    baseline, diagnostic = raw_config(BASE_CONFIG), raw_config()
    assert formal.validate_formal_config(diagnostic) == TAV0Config(
        text_dim=1024, diagnostic_t1024_ab1=True)
    diagnostic["model"].pop("diagnostic_t1024_ab1")
    diagnostic["model"]["text_dim"] = 768
    assert diagnostic == baseline
    original = Path(BASE_CONFIG).read_bytes()
    newline = b"\r\n" if b"\r\n" in original else b"\n"
    expected = original.replace(b"  text_dim: 768" + newline,
        b"  text_dim: 1024" + newline + b"  diagnostic_t1024_ab1: true" + newline)
    assert Path(AB1_CONFIG).read_bytes() == expected


@pytest.mark.parametrize("flag", [None, False])
def test_1024_without_opt_in_is_rejected(flag):
    raw = raw_config(BASE_CONFIG)
    raw["model"]["text_dim"] = 1024
    if flag is not None:
        raw["model"]["diagnostic_t1024_ab1"] = flag
    with pytest.raises(ValueError, match="freezes text_dim=768"):
        formal.validate_formal_config(raw)


@pytest.mark.parametrize("dimension", [512, 768, 2048, 1024.0, True])
def test_override_has_exact_1024_integer_allowlist(dimension):
    raw = raw_config()
    raw["model"]["text_dim"] = dimension
    with pytest.raises(ValueError, match="freezes text_dim=1024"):
        formal.validate_formal_config(raw)


@pytest.mark.parametrize("flag", [1, "true", None])
def test_diagnostic_opt_in_requires_bool(flag):
    raw = raw_config()
    raw["model"]["diagnostic_t1024_ab1"] = flag
    with pytest.raises(TypeError, match="diagnostic_t1024_ab1 must be bool"):
        formal.validate_formal_config(raw)


@pytest.mark.parametrize("field,value", [
    ("audio_dim", 1583), ("encoder_dim", 128), ("dag_hidden_dim", 100),
    ("dag_layers", 3), ("num_classes", 7), ("window_past_same_speaker", 2),
    ("classifier_dropout", .2), ("use_causal_history_gate", True),
    ("chg_speaker_embedding_dim", 9), ("chg_distance_embedding_dim", 9), ("chg_hidden_dim", 65),
])
def test_override_cannot_change_any_other_model_field(field, value):
    raw = raw_config()
    raw["model"][field] = value
    with pytest.raises(ValueError, match="freezes"):
        formal.validate_formal_config(raw)
    with pytest.raises(ValueError, match="freezes"):
        replace(ab1(), **{field: value})


@pytest.mark.parametrize("section,field,value", [
    ("training", "seed", 42), ("training", "max_epochs", 1),
    ("training", "batch_size", 8), ("training", "curriculum", True),
    ("training", "optimizer", {"name": "SGD", "learning_rate": .1}),
    ("training", "early_stopping", True), ("training", "class_weight", [1] * 6),
    ("runtime", "seed", 42), ("runtime", "curriculum", True),
    ("runtime", "splits", ["train", "dev", "test"]),
    ("checkpoint", "selection_metric", "dev_loss"), ("checkpoint", "test_evaluation_count", 1),
    ("checkpoint", "test_split_used_for_selection", True),
    ("dataset", "label_vocab", ["neu", "sad", "hap", "ang", "exc", "fru"]),
])
def test_ab1_formal_protocol_stays_frozen_before_data_access(monkeypatch, section, field, value):
    raw = raw_config()
    raw[section][field] = value
    monkeypatch.setattr(formal.yaml, "safe_load", lambda _: raw)
    monkeypatch.setattr(formal, "load_train_dev", lambda *a, **kw: pytest.fail("data accessed"))
    with pytest.raises(ValueError, match="frozen"):
        formal.run_formal(formal_train_dev=True, config_path=AB1_CONFIG, asset_path="FORBIDDEN",
                          model_manifest_path="FORBIDDEN", provenance_manifest_path="FORBIDDEN",
                          output_dir="FORBIDDEN")


class ForbiddenTestFeatures:
    def __getitem__(self, key):
        raise AssertionError("Test feature access forbidden")

    def __array__(self, *args, **kwargs):
        raise AssertionError("Test feature access forbidden")


def diagnostic_package():
    package = synthetic_package()
    asset = package[0]
    asset["feature_spec"]["text"]["dimension"] = 1024
    rng = np.random.default_rng(100)
    for split in ("train", "dev"):
        for did in asset["splits"][split]:
            row = asset["dialogues"][did]
            row["text_features"] = rng.normal(size=(len(row["utterance_ids"]), 1024)).astype(np.float32)
    for did in asset["splits"]["test"]:
        for key in ("text_features", "audio_features"):
            asset["dialogues"][did][key] = ForbiddenTestFeatures()
    return package


def datasets_for(package, config):
    return {s: TAV0Dataset(*package, split=s, expected_rows=len(package[1]), config=config)
            for s in ("train", "dev")}


def test_adapter_collate_and_unilstm_forward_change_only_text_input_width():
    config = ab1()
    package = diagnostic_package()
    datasets = datasets_for(package, config)
    baseline = synthetic_package()
    for split, dataset in datasets.items():
        original = TAV0Dataset(*baseline, split=split, expected_rows=len(baseline[1]))
        for left, right in zip(dataset, original):
            assert left.keys() == right.keys()
            for key in left.keys() - {"text_features"}:
                if isinstance(left[key], torch.Tensor):
                    torch.testing.assert_close(left[key], right[key], rtol=0, atol=0)
                else:
                    assert left[key] == right[key]
    with pytest.raises(ValueError, match="wrong feature shape"):
        TAV0Dataset(*package, split="train", expected_rows=len(package[1]))
    model = TAV0(config).eval()
    runtime = TrainDevRuntime(model, datasets.__getitem__, synthetic_only=True)
    batch = next(iter(runtime.loader("dev")))
    assert batch["text_features"].shape == (2, 3, 1024)
    assert batch["audio_features"].shape == (2, 3, 1582)
    with pytest.raises(RuntimeError):
        ta_v0_collate([datasets["dev"][0]])
    with torch.inference_mode():
        text, audio = model.encoders(batch["text_features"], batch["audio_features"],
                                     batch["lengths"], batch["attention_mask"])
        output, loss = runtime.forward_batch(batch, split="dev")
    assert text.shape == audio.shape == (2, 3, 100)
    assert model.encoders.text_encoder.hidden_size == 100
    assert model.encoders.text_encoder.num_layers == 1 and not model.encoders.text_encoder.bidirectional
    assert output.h_ta.shape == (2, 3, 1500) and output.logits_ta.shape == (2, 3, 6)
    assert torch.isfinite(loss) and torch.isfinite(output.prob_ta).all()
    assert not torch.count_nonzero(output.h_ta[~output.mask])
    assert not any("visual" in name or "history_gate" in name for name in model.state_dict())
    baseline_state, diagnostic_state = TAV0().state_dict(), model.state_dict()
    assert baseline_state.keys() == diagnostic_state.keys()
    changed = [name for name, value in diagnostic_state.items()
               if isinstance(value, torch.Tensor) and value.shape != baseline_state[name].shape]
    assert changed == ["encoders.text_projection.weight"]
    for target, source in ((TAV0(), diagnostic_state), (model, baseline_state)):
        with pytest.raises(ValueError, match="checkpoint identity"):
            target.load_state_dict(source, strict=False)
    TAV0(config).load_state_dict(diagnostic_state, strict=True)
    full = {k: batch[k][:1].clone() for k in MODEL_INPUTS}
    changed = {k: v.clone() for k, v in full.items()}
    changed["text_features"][:, 1:] += 100
    changed["audio_features"][:, 1:] -= 100
    with torch.inference_mode():
        reference, perturbed = model(**full), model(**changed)
    torch.testing.assert_close(reference.logits_ta[:, :1], perturbed.logits_ta[:, :1])
    assert runtime.optimizer_steps == 0 and all(p.grad is None for p in model.parameters())


@pytest.mark.parametrize("case", ["text_metadata", "audio_metadata", "audio_features", "split", "label", "order"])
def test_ab1_preserves_adapter_contract_checks(case):
    package = diagnostic_package()
    asset, model, provenance = package
    did = asset["splits"]["train"][0]
    if case == "text_metadata":
        asset["feature_spec"]["text"]["dimension"] = 768
    elif case == "audio_metadata":
        asset["feature_spec"]["audio_ta"]["dimension"] = 1583
    elif case == "audio_features":
        asset["dialogues"][did]["audio_features"] = np.zeros((3, 1583), dtype=np.float32)
    elif case == "split":
        model[0]["split"] = "dev"
    elif case == "label":
        model[0]["benchmark_label_id"] = 5
    else:
        provenance[1]["canonical_dialogue_order"] = provenance[0]["canonical_dialogue_order"]
    with pytest.raises(ValueError):
        datasets_for(package, ab1())


def test_ab1_manifest_resolved_config_and_data_loading_are_explicit(monkeypatch, tmp_path):
    config = ab1()
    datasets = datasets_for(diagnostic_package(), config)
    paths = {"asset": "synthetic.pkl", "model_manifest": "synthetic-model.csv",
             "provenance_manifest": "synthetic-provenance.csv"}
    seen = []
    def load(*args, **kwargs):
        assert kwargs == {"config": config}
        seen.append(args)
        return datasets, {"synthetic_only": True}, paths
    def inspect_fit(runtime, run_dir, *, resolved_config, manifest, **kwargs):
        assert runtime.optimizer_steps == 0
        assert manifest["model_lineage"] == "ta_v0_t1024_ab1_diagnostic"
        assert manifest["model_config"]["diagnostic_t1024_ab1"] is True
        assert resolved_config["model"]["diagnostic_t1024_ab1"] is True
        assert manifest["run_id"].startswith("ta_v0_t1024_ab1_diagnostic_seed100_")
        assert manifest["training_config"] == resolved_config["training"] == PROTOCOL
        assert resolved_config["runtime"]["splits"] == ["train", "dev"]
        assert manifest["checkpoint_metric"] == "dev_weighted_f1"
        assert manifest["checkpoint_tiebreak_rules"] == ["weighted_f1_desc", "dev_loss_asc", "epoch_asc"]
        assert not Path(run_dir).exists()
        return {"inspected": True}
    monkeypatch.setattr(formal, "load_train_dev", load)
    monkeypatch.setattr(formal, "fit", inspect_fit)
    monkeypatch.setattr(formal, "git_identity", lambda: {"git_head": "synthetic", "git_dirty": False})
    assert formal.run_formal(formal_train_dev=True, config_path=AB1_CONFIG, asset_path="synthetic.pkl",
        model_manifest_path="synthetic-model.csv", provenance_manifest_path="synthetic-provenance.csv",
        output_dir=tmp_path) == {"inspected": True}
    assert len(seen) == 1


def test_formal_loader_propagates_opt_in_with_synthetic_files(monkeypatch, tmp_path):
    package = diagnostic_package()
    paths = [tmp_path / name for name in ("synthetic.pkl", "model.csv", "provenance.csv")]
    paths[0].write_bytes(pickle.dumps(package[0]))
    for path, rows in zip(paths[1:], package[1:]):
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    before = [formal.sha256_file(path) for path in paths]
    monkeypatch.setattr(formal, "PACKAGE_ROWS", len(package[1]))
    counts = {"train": {"utterances": 5, "dialogues": 2},
              "dev": {"utterances": 4, "dialogues": 2}}
    monkeypatch.setattr(formal, "SPLIT_COUNTS", counts)
    datasets, identity, _ = formal.load_train_dev(*paths, config=ab1())
    assert set(datasets) == {"train", "dev"} and identity["split_counts"] == counts
    assert datasets["train"][0]["text_features"].shape == (3, 1024)
    assert datasets["train"][0]["canonical_order"] == (1, 2, 3)
    assert list(identity["sha256"].values()) == before
    assert [formal.sha256_file(path) for path in paths] == before
    with pytest.raises(ValueError, match="wrong feature shape"):
        formal.load_train_dev(*paths)


def test_ab1_rejects_test_before_factory_or_files(monkeypatch):
    config = ab1()
    runtime = TrainDevRuntime(TAV0(config), lambda _: pytest.fail("factory accessed"), synthetic_only=True)
    with pytest.raises(ValueError, match="Test"):
        runtime.loader("test")
    monkeypatch.setattr(Path, "open", lambda *a, **kw: pytest.fail("file accessed"))
    with pytest.raises(ValueError, match="Test"):
        TAV0Dataset.from_files("FORBIDDEN", "FORBIDDEN", "FORBIDDEN", split="test", config=config)
    assert runtime.optimizer_steps == 0

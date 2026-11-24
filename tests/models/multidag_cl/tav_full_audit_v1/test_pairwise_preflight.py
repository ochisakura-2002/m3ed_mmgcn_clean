import json
import pickle
from types import SimpleNamespace
from zipfile import ZipFile

import numpy as np
import pytest
import torch

from datasets.iemocap.tav_full_audit_dataset import COUNTS, DIMS, SCHEMA, VERSION, fingerprint, identity_sha
from models.multidag_cl.fusion.model import FusionTAV
from scripts.analysis.models.multidag_cl import pairwise_preflight_inputs as inputs
from scripts.analysis.models.multidag_cl import run_pairwise_preflight as runner
from scripts.analysis.models.multidag_cl.failure_audit_metrics import metadata, quartiles
from scripts.analysis.models.multidag_cl.pairwise_preflight_metrics import (
    METRICS, analyze, decision, gate, oracle, scores, train_threshold,
)
from scripts.runtime.multidag_cl_fusion.spec import make_final_config
from scripts.runtime.multidag_cl_ta_v0.formal import json_identity, sha256_file, write_json_atomic
from scripts.runtime.multidag_cl_ta_v0.training import classification_report
from scripts.runtime.multidag_cl_tav_full_audit_v1.spec import read_csv, write_csv


def test_mad_and_stable_class_id_order():
    data = scores([[0.] * 6, [0., 1000., 0., 0., 0., 0.], [0., 5., 0., 5., 0., 5.]], [5, 0, 5])
    assert data.mad[:2] == pytest.approx([1, 0])
    assert ((0 <= data.mad) & (data.mad <= 1)).all()
    assert data.top1.tolist() == [0, 1, 1]
    assert data.top2.tolist() == [1, 0, 3]
    assert data.true_rank.tolist() == [6, 2, 3]
    assert data.margin.tolist() == [0, 1000, 0]


def test_train_order_statistic_and_gate_boundaries():
    assert train_threshold([0.8, 0.2, 0.6, 0.1, 0.4]) == 0.6
    mad = np.array([0.59, 0.6, 0.6, 0.7, 0.9, 0.61])
    margin = np.array([0.5, 0., 1., 1.1, 0.5, 0.999])
    valid = np.array([True, True, True, True, False, True])
    assert gate(mad, margin, valid, 0.6).tolist() == [False, False, False, False, False, True]
    assert gate(mad, np.full_like(mad, 0.1), valid, 0.6).tolist() == [False, True, True, True, False, True]


def test_oracle_flips_only_reachable_wrong_rows_and_uses_project_metrics():
    z = np.tile([1., 0.5, 0.25, 0., 0., 0.], (6, 1))
    data = scores(z, [1, 0, 2, 1, 3, 1])
    selected = np.array([True, True, True, False, False, True])
    prediction, recovered = oracle(data, selected)
    assert recovered.tolist() == [True, False, False, False, False, True]
    assert prediction.tolist() == [1, 0, 0, 0, 0, 1]
    np.testing.assert_array_equal(prediction[~recovered], data.top1[~recovered])

    q4 = np.array([True, False, False, False, False, True])
    summary, tables = analyze(data, data, q4, {"SHIFT": q4, "H_ONLY": None})
    # All rows pass the frozen gate in this fixture. Gold rank 2 is corrected.
    expected = classification_report(data.labels, np.array([1, 0, 0, 1, 0, 1]), loss=0)["metrics"]
    base = classification_report(data.labels, data.top1, loss=0)["metrics"]
    for key in METRICS:
        assert summary[f"oracle_{key}"] == expected[key]
        assert summary[f"oracle_delta_{key}_pp"] == pytest.approx(100 * (expected[key] - base[key]))
    assert summary["recoverable_f"] == summary["n_y_eq_b"] == 3
    assert summary["q4_oracle_net_recovery"] == 2
    assert summary["wrong_true_rank2_rate"] == pytest.approx(0.6)
    assert summary["wrong_true_rank3plus_rate"] == pytest.approx(0.4)
    assert len(tables["ORACLE_PAIR_ANALYSIS.csv"]) == len(tables["TRAIN_PAIR_SUPPORT.csv"]) == 15
    assert len(tables["TRAIN_PAIR_DIRECTION_SUPPORT.csv"]) == 30
    assert len(tables["TRAIN_CLASS_SUPPORT.csv"]) == 6
    assert len(tables["TRUE_RANK_DISTRIBUTION.csv"]) == 42
    assert tables["ORACLE_FAILURE_SUBGROUPS.csv"][1]["status"] == "NOT_AVAILABLE"


@pytest.mark.parametrize(("n", "delta", "expected"), [(8, .75, "PASS"), (7, 3, "STOP"), (11, .749999, "STOP")])
def test_only_the_two_registered_conditions_control_the_decision(n, delta, expected):
    row = decision(n, delta)
    assert row["ZERO_TRAINING_PREFLIGHT"] == expected
    assert (row["PAIRWISE_HEAD_IMPLEMENTATION"] == "ELIGIBLE") == (expected == "PASS")


@pytest.fixture
def projection(tmp_path, monkeypatch):
    directory = tmp_path / "certified"
    directory.mkdir()
    source_data = {"schema": SCHEMA, "sha256": {"asset": "1" * 64, "model_manifest": "2" * 64,
                                               "provenance_manifest": "3" * 64}, "split_counts": COUNTS,
                   "feature_dimensions": inputs.FEATURE_DIMENSIONS, "row_order_identity_sha256": "4" * 64}
    split_counts = dict.fromkeys(inputs.SPLITS, {"utterances": 3, "dialogues": 1})
    monkeypatch.setattr(inputs, "TRAIN_DEV_COUNTS", split_counts)
    asset = {"schema": SCHEMA, "dataset_version": VERSION, "label_vocab": {"itos": list(inputs.CLASS_ORDER)},
             "feature_spec": {key: {"dimension": dim, "dtype": "float32"} for key, dim in DIMS.items()},
             "splits": {}, "dialogues": {}}
    rows, populations, pairing = [], {}, {}
    for split, did, labels in (("train", "Ses01F_impro01", [0, 1, 0]), ("dev", "Ses04F_impro01", [0, 1, 1])):
        utterances = tuple(f"{did}_F{i:03}" for i in range(3))
        asset["splits"][split] = [did]
        asset["dialogues"][did] = {"utterance_ids": utterances, "speakers": ["F"] * 3, "benchmark_label_ids": labels,
                                   "benchmark_labels": [inputs.CLASS_ORDER[label] for label in labels],
                                   **{key: np.ones((3, dim), dtype=np.float32) for key, dim in DIMS.items()}}
        for index, (uid, label) in enumerate(zip(utterances, labels)):
            rows.append({"utterance_id": uid, "dialogue_id": did, "split": split, "speaker": "F",
                         "canonical_dialogue_order": index + 1, "asset_dialogue_id": did, "asset_row_index": index,
                         "benchmark_label": inputs.CLASS_ORDER[label], "benchmark_label_id": label,
                         "audio_sha256": fingerprint(asset["dialogues"][did]["audio_features"][index])})
        populations[split] = tuple((uid, did, label) for uid, label in zip(utterances, labels))
        pairing[split] = {**split_counts[split], "sha256": json_identity([(did, utterances)])}
    (directory / inputs.ASSET).write_bytes(pickle.dumps(asset))
    write_csv(directory / inputs.MODEL_MANIFEST, rows)
    certificate = {**inputs.HOLD, "schema": inputs.PROJECTION_SCHEMA, "source_data_identity": source_data,
                   "splits": list(inputs.SPLITS), "split_counts": split_counts, "row_order_identity_sha256": identity_sha(rows),
                   "sha256": {name: sha256_file(directory / name) for name in (inputs.ASSET, inputs.MODEL_MANIFEST)}}
    write_json_atomic(directory / inputs.PROJECTION_MANIFEST, certificate)
    source = SimpleNamespace(manifest={"data_identity": source_data, "batch_pairing": {"population": pairing}},
                             population=populations["dev"])
    return directory, certificate, source


def test_forward_reuses_real_f0_on_train_dev_only(projection, monkeypatch):
    directory, certificate, source = projection
    datasets_built, loaders_built = [], []
    dataset_factory, data_loader = inputs.TAVAuditDataset, inputs.DataLoader

    def record_dataset(*args, split, **kwargs):
        datasets_built.append(split)
        return dataset_factory(*args, split=split, **kwargs)

    def record_loader(dataset, **kwargs):
        loaders_built.append(dataset.split)
        return data_loader(dataset, **kwargs)

    monkeypatch.setattr(inputs, "TAVAuditDataset", record_dataset)
    monkeypatch.setattr(inputs, "DataLoader", record_loader)
    datasets, actual_certificate = inputs.load_projected_train_dev(directory, source)
    assert actual_certificate == certificate
    assert inputs.validate_population(datasets, source) == source.manifest["batch_pairing"]["population"]
    model = FusionTAV(make_final_config("F0", 100)).requires_grad_(False).eval()
    calls = []

    def check_forward(_module, _args, kwargs):
        assert torch.is_inference_mode_enabled()
        assert "labels" not in kwargs
        calls.append(True)

    threads = torch.get_num_threads()
    handle = model.register_forward_pre_hook(check_forward, with_kwargs=True)
    torch.set_num_threads(1)
    try:
        train = inputs.frozen_forward(model, datasets["train"], "train", device="cpu", batch_size=16)
        dev = inputs.frozen_forward(model, datasets["dev"], "dev", device="cpu", batch_size=16)
        with pytest.raises(ValueError, match="Test factory forbidden"):
            inputs.frozen_forward(model, None, "test", device="cpu", batch_size=16)
    finally:
        handle.remove()
        torch.set_num_threads(threads)
    assert train.logits.shape == dev.logits.shape == (3, 6)
    assert dev.population == source.population
    assert datasets_built == loaders_built == ["train", "dev"] and len(calls) == 2
    assert not model.training and all(p.grad is None for p in model.parameters())


def test_rejects_unscoped_or_tampered_input_before_pickle_or_loader(projection, monkeypatch):
    directory, certificate, source = projection

    def forbidden(*_args, **_kwargs):
        pytest.fail("unverified package reached a dataset, pickle or loader")

    monkeypatch.setattr(inputs.pickle, "load", forbidden)
    monkeypatch.setattr(inputs, "TAVAuditDataset", forbidden)
    monkeypatch.setattr(inputs, "DataLoader", forbidden)
    write_json_atomic(directory / inputs.PROJECTION_MANIFEST, {**certificate, "splits": ["train", "dev", "test"]})
    with pytest.raises(ValueError, match="Train/Dev-only projection"):
        inputs.load_projected_train_dev(directory, source)
    write_json_atomic(directory / inputs.PROJECTION_MANIFEST, certificate)
    (directory / inputs.ASSET).write_bytes(b"tampered")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        inputs.load_projected_train_dev(directory, source)


@pytest.mark.parametrize("counter", inputs.HOLD_COUNTERS)
def test_every_hold_counter_is_a_hard_check(counter):
    with pytest.raises(inputs.TestHoldViolation, match=counter):
        inputs.validate_hold({**inputs.HOLD, counter: 1})
    damaged = dict(inputs.HOLD)
    del damaged[counter]
    with pytest.raises(inputs.TestHoldViolation, match=counter):
        inputs.validate_hold(damaged)


def test_read_f0_requires_historical_checkpoint_not_a_matched_control(monkeypatch, tmp_path):
    config = make_final_config("F0", 100)
    identity = dict(inputs.F0_IDENTITY)
    manifest = {"architecture_identity": json_identity({key: config[key] for key in (
        "architecture_lineage", "frontend_source", "graph_core", "fusion")}),
        "context": "past_all_causal", "graph_core": config["graph_core"], "fusion": config["fusion"],
        "active_modalities": "TAV", "feature_dimensions": inputs.FEATURE_DIMENSIONS,
        "best_dev_metrics": {"valid_target_count": 980, "accuracy": 693 / 980, "weighted_f1": .699627}}
    source = SimpleNamespace(identity=identity, config=config, manifest=manifest, population=[None] * 980,
                             checkpoint=tmp_path / "best_model.pt")
    requested = []

    def only_seed100(_path, *, seeds):
        requested.extend(seeds)
        return {100: source}

    monkeypatch.setattr(inputs, "read_json", lambda _path: inputs.HOLD)
    monkeypatch.setattr(inputs, "sources_from_runs", only_seed100)
    monkeypatch.setattr(inputs, "sha256_file", lambda _path: inputs.F0_IDENTITY["checkpoint_sha256"])
    assert inputs.read_f0(tmp_path) is source
    assert requested == [100]
    identity["checkpoint_sha256"] = "8c9277977efa3c7ee42c76ac1a7d102ffa0e42299203be8a2356612854957104"
    with pytest.raises(ValueError, match="identity mismatch"):
        inputs.read_f0(tmp_path)


def test_round1_trace_is_joined_by_key_and_can_supply_q4_without_history(tmp_path, projection):
    _, _, source = projection
    source.identity = dict(inputs.F0_IDENTITY)
    source.logits = np.tile([1., .5, .2, 0., 0., 0.], (3, 1))
    data = scores(source.logits, [label for _, _, label in source.population])
    rows = [{"seed": 100, **row, "mad": mad, "mad_quartile": rank, "predicted_label": inputs.CLASS_ORDER[top1]}
            for row, mad, rank, top1 in zip(metadata(source.population), data.mad, quartiles(data.mad), data.top1)]
    trace = tmp_path / "AMBIGUITY_AUDIT.csv"
    write_csv(trace, rows[::-1])
    manifest = {**inputs.HOLD, "sources": [source.identity], "data_identity": source.manifest["data_identity"],
                "outputs_sha256": {trace.name: sha256_file(trace)}}
    write_json_atomic(tmp_path / "failure_audit_manifest.json", manifest)
    q4, groups, status = inputs.audit_subgroups(source, tmp_path)
    np.testing.assert_array_equal(q4, quartiles(data.mad) == 4)
    assert status["Q4_SOURCE"] == "VERIFIED_ROUND1_TRACE"
    assert status["SHIFT_SUBGROUP_STATUS"] == "AVAILABLE" and status["H_SUBGROUP_STATUS"] == "NOT_AVAILABLE"
    assert groups["SHIFT"].tolist() == [False, True, False] and groups["H_ONLY"] is None
    trace.write_text("tampered", encoding="utf-8")
    _, groups, status = inputs.audit_subgroups(source, tmp_path)
    assert status["Q4_SOURCE"] == "ARCHIVED_F0_DEV_STABLE_MAD" and all(group is None for group in groups.values())


def test_reconstructed_history_groups_reproduce_both_registered_budgets(tmp_path):
    did = "Ses04F_impro01"
    gold = np.zeros(980, dtype=int)
    gold[:102] = 1
    gold[245:349] = 1
    archived = scores(np.tile([1., .5, .25, 0., 0., 0.], (980, 1)), gold)
    meta = metadata([(f"{did}_F{i:03}", did, label) for i, label in enumerate(gold)])
    h_high = set(range(245, 490))
    rows = [{"seed": 100, "row_type": "sample", "condition": condition, **row,
             "predicted_label": "sad" if condition != "FULL_HISTORY" and i in h_high else "hap"}
            for condition in inputs.HISTORY for i, row in enumerate(meta)]
    trace = tmp_path / "HISTORY_UTILITY_AUDIT.csv"
    write_csv(trace, rows)
    manifest = {"outputs_sha256": {trace.name: sha256_file(trace)}}
    groups = inputs.history_groups(tmp_path, manifest, meta, archived, archived.mad)
    assert [int(groups[name].sum()) for name in inputs.H_GROUPS] == [245, 245, 0, 490]
    with pytest.raises(ValueError, match="capture counts"):
        inputs.history_groups(tmp_path, manifest, meta, archived, np.arange(980))


@pytest.mark.parametrize("hold_violation", [False, True])
def test_identity_failure_stops_before_any_forward(tmp_path, monkeypatch, capsys, hold_violation):
    def reject(_path):
        if hold_violation:
            inputs.validate_hold({**inputs.HOLD, "test_forward_count": 1})
        raise ValueError("synthetic identity mismatch")

    def forbidden(*_args):
        pytest.fail("identity failure reached a dataset or forward")

    monkeypatch.setattr(runner, "read_f0", reject)
    monkeypatch.setattr(runner, "load_projected_train_dev", forbidden)
    monkeypatch.setattr(runner, "forward_f0", forbidden)
    args = SimpleNamespace(batch_id="test-batch", frozen_runs=tmp_path)
    output, archive = tmp_path / "result", tmp_path / "result.zip"
    assert runner.execute(args, output, archive, device="cpu", git={}, experiment_date="20261124") == 2
    evidence = json.loads((output / runner.DECISION).read_text(encoding="utf-8"))
    assert evidence["PREFLIGHT"] == ("INVALID" if hold_violation else "REVIEW_INVALID_IDENTITY")
    assert evidence["ZERO_TRAINING_PREFLIGHT"] == "INVALID" and evidence["PAIRWISE_HEAD_IMPLEMENTATION"] == "NOT_AUTHORIZED"
    assert not archive.exists() and not (output / runner.SUMMARY).exists()
    assert f"TEST_TOUCHED = {'YES' if hold_violation else 'NO'}" in capsys.readouterr().out
    if hold_violation:
        counters = json.loads((output / runner.HOLD_STATUS).read_text(encoding="utf-8"))
        assert counters["rejected_input_hold_evidence"] == {"test_forward_count": 1}


@pytest.mark.parametrize(("repaired", "decision_name"), [(6, "STOP"), (8, "PASS")])
def test_successful_reporting_path_exports_a_lightweight_verified_bundle(tmp_path, monkeypatch, capsys,
                                                                         repaired, decision_name):
    config = make_final_config("F0", 100)
    z = np.tile([1., .5, .2, 0., 0., 0.], (12, 1))
    labels = [1] * repaired + [0] * (12 - repaired)
    population = tuple((f"u{i}", "d", label) for i, label in enumerate(labels))
    source = SimpleNamespace(config=config, identity=inputs.F0_IDENTITY, logits=z.copy(),
                             manifest={"config_sha256": "c", "architecture_identity": "a", "data_identity": {}})
    split = inputs.FrozenSplit(z, population)
    monkeypatch.setattr(runner, "read_f0", lambda _path: source)
    monkeypatch.setattr(runner, "load_projected_train_dev", lambda *_args: ({}, inputs.HOLD))
    monkeypatch.setattr(runner, "validate_population", lambda *_args: {})
    monkeypatch.setattr(runner, "forward_f0", lambda *_args: (split, split))
    monkeypatch.setattr(runner, "audit_subgroups", lambda *_args: (
        np.arange(12) >= 9, dict.fromkeys(inputs.H_GROUPS), {"Q4_SOURCE": "SYNTHETIC",
        "SHIFT_SUBGROUP_STATUS": "NOT_AVAILABLE", "H_SUBGROUP_STATUS": "NOT_AVAILABLE"}))
    monkeypatch.setattr(runner, "TRAIN_DEV_COUNTS", dict.fromkeys(inputs.SPLITS, {"utterances": 12}))
    monkeypatch.setattr(runner, "Q4_SUPPORT", 3)
    args = SimpleNamespace(batch_id="test-batch", frozen_runs=tmp_path, train_dev_package=tmp_path, round1_dir=None)
    output, archive = tmp_path / "result", tmp_path / "result.zip"
    (tmp_path / "unrelated.pt").write_bytes(b"not an input")
    assert runner.execute(args, output, archive, device="cpu", git={}, experiment_date="20261124") == 0
    summary = read_csv(output / runner.SUMMARY)[0]
    assert summary["recoverable_f"] == str(repaired) and summary["q4_support"] == "3"
    manifest = json.loads((output / runner.MANIFEST).read_text(encoding="utf-8"))
    assert manifest["ZERO_TRAINING_PREFLIGHT"] == decision_name and manifest["HANDOFF_READY"] == "YES"
    with ZipFile(archive) as bundle:
        assert bundle.testzip() is None
        assert set(bundle.namelist()) == {*manifest["outputs_sha256"], runner.MANIFEST}
        assert not any(name.endswith((".pkl", ".pt", ".npz")) for name in bundle.namelist())
    for name, digest in manifest["outputs_sha256"].items():
        assert sha256_file(output / name) == digest
    assert f"ZERO_TRAINING_PREFLIGHT = {decision_name}" in capsys.readouterr().out
    with pytest.raises(FileExistsError):
        runner.execute(args, output, archive, device="cpu", git={}, experiment_date="20261124")

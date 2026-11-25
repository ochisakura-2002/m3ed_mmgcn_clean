import csv
from copy import deepcopy
import gzip
import pickle

import numpy as np
import pytest
import torch

from datasets.iemocap.tav_full_audit_dataset import DIMS, MANIFEST_FIELDS, SCHEMA, VERSION, fingerprint, identity_sha
from models.multidag_cl.fusion.base import DirectConcatFusion
from models.multidag_cl.fusion.model import FusionTAV
from models.multidag_cl.ta_v0 import LABEL_VOCAB
from scripts.analysis.models.multidag_cl import pairwise_preflight_inputs as projection_loader
from scripts.analysis.models.multidag_cl.post_ph.internal_protocol import write_rows
from scripts.models.multidag_cl import group_isolated as g0
from scripts.runtime.multidag_cl_group_isolated import protocol
from scripts.runtime.multidag_cl_group_isolated.evaluation import checkpoint_path, summarize
from scripts.runtime.multidag_cl_ta_v0.formal import sha256_file, write_json_atomic
from scripts.runtime.multidag_cl_tav_full_audit_v1.runtime import AuditRuntime


@pytest.mark.parametrize(("session", "fit", "held_out", "steps"), [
    ("Ses01", (72, 3457), (28, 1373), 5),
    ("Ses02", (70, 3474), (30, 1356), 5),
    ("Ses03", (68, 3261), (32, 1569), 5),
    ("Ses04", (90, 4298), (10, 532), 6),
])
def test_frozen_fold_identity(session, fit, held_out, steps):
    frozen, raw = protocol.read_frozen(g0.ROOT)
    path = g0.ROOT / protocol.CONFIG_DIR / f"{session.lower()}_seed100.yaml"
    assert protocol.load_config(path) == protocol.fold_config(session)
    fold = frozen["folds"][session]
    assert tuple(fold["fit"][key] for key in ("dialogues", "utterances")) == fit
    assert tuple(fold["held_out"][key] for key in ("dialogues", "utterances")) == held_out
    assert g0.steps_per_epoch(fold["fit"], raw) == steps
    assert not set(fold["fit"]["actors"]) & set(fold["held_out"]["actors"])
    assert not set(fold["fit"]["dialogue_ids"]) & set(fold["held_out"]["dialogue_ids"])
    assert frozen["certified_projection"]["sha256"]["train_dev_tav.pkl"] == (
        "a9f23d0fb2b59abf156222a93e29896a203cf4948908ec3bbbb99cf37744c466")


def test_actor_overlap_rejected():
    frozen, _ = protocol.read_frozen(g0.ROOT)
    fold = deepcopy(frozen["folds"]["Ses03"])
    fold["held_out"]["actors"][0] = fold["fit"]["actors"][0]
    with pytest.raises(ValueError, match="actor"):
        protocol.assert_isolation(fold["fit"], fold["held_out"], "Ses03")


@pytest.fixture
def certified(tmp_path, monkeypatch):
    frozen, raw = protocol.read_frozen(g0.ROOT)
    config = protocol.fold_config("Ses03")
    rng = np.random.default_rng(100)
    splits = {"train": [f"{s}F_impro{i:02d}" for s in protocol.SESSIONS for i in range(1, 4)],
              "dev": ["Ses04M_impro04"]}
    dialogues, manifest, populations = {}, [], {}
    for split, ids in splits.items():
        populations[split] = []
        for did in ids:
            speakers = ["F", "F", "M", "M", "F", "M"]
            uids = [f"{did}_{speaker}{idx:03d}" for idx, speaker in enumerate(speakers)]
            labels = list(range(len(uids)))
            dialogue = {
                "utterance_ids": uids, "speakers": speakers, "benchmark_label_ids": labels,
                "benchmark_labels": list(LABEL_VOCAB),
                **{key: rng.standard_normal((len(uids), dim), dtype=np.float32) for key, dim in DIMS.items()},
            }
            dialogues[did] = dialogue
            data = [(uid, did, label) for uid, label in zip(uids, labels)]
            populations[split].extend(reversed(data))
            for index, (uid, _, label) in enumerate(data):
                manifest.append({
                    "utterance_id": uid, "dialogue_id": did, "split": split, "speaker": speakers[index],
                    "canonical_dialogue_order": len(uids) - index, "asset_dialogue_id": did, "asset_row_index": index,
                    "benchmark_label": LABEL_VOCAB[label], "benchmark_label_id": label,
                    "audio_sha256": fingerprint(dialogue["audio_features"][index]),
                })
    asset = {
        "schema": SCHEMA, "dataset_version": VERSION, "splits": splits, "dialogues": dialogues,
        "feature_spec": {key: {"dimension": dim, "dtype": "float32"} for key, dim in DIMS.items()},
        "label_vocab": {"itos": list(LABEL_VOCAB)},
    }
    projected = tmp_path / projection_loader.ASSET
    projected.write_bytes(pickle.dumps(asset, protocol=pickle.HIGHEST_PROTOCOL))
    model = tmp_path / projection_loader.MODEL_MANIFEST
    write_rows(model, manifest, MANIFEST_FIELDS)
    descriptions = {split: protocol.describe(rows) for split, rows in populations.items()}
    split_counts = {split: {key: entry[key] for key in ("dialogues", "utterances")}
                    for split, entry in descriptions.items()}
    identity = {
        "schema": SCHEMA, "split_counts": {**split_counts, "test": {"dialogues": 1, "utterances": 1}},
        "feature_dimensions": projection_loader.FEATURE_DIMENSIONS,
        "sha256": {"asset": "a" * 64, "model_manifest": "b" * 64, "provenance_manifest": "c" * 64},
    }
    frozen["formal_f0"].update(
        data_identity=identity,
        population={split: {**split_counts[split], "sha256": desc["order_sha256"]}
                    for split, desc in descriptions.items()},
        labeled_tape_sha256={split: desc["labeled_tape_sha256"] for split, desc in descriptions.items()},
    )
    entries = {role: [row for row in populations["train"] if (row[1].startswith("Ses03")) == (role == "held_out")]
               for role in ("fit", "held_out")}
    frozen["folds"]["Ses03"] = {role: protocol.describe(rows) for role, rows in entries.items()}
    frozen["certified_projection"] = {
        "schema": projection_loader.PROJECTION_SCHEMA, "row_order_identity_sha256": identity_sha(manifest),
        "sha256": {projected.name: sha256_file(projected), model.name: sha256_file(model)},
    }
    certificate = {**frozen["certified_projection"], **protocol.HOLD, "splits": list(splits),
                   "source_data_identity": identity, "split_counts": split_counts}
    write_json_atomic(tmp_path / protocol.PROJECTION_MANIFEST, certificate)
    monkeypatch.setattr(protocol, "TRAIN_DEV_COUNTS", split_counts)
    monkeypatch.setattr(projection_loader, "TRAIN_DEV_COUNTS", split_counts)
    monkeypatch.setattr(projection_loader, "COUNTS", identity["split_counts"])
    return config, raw, frozen, entries, asset


def test_projection_membership_and_order(certified, tmp_path):
    config, _, frozen, expected, asset = certified
    views, entries, data = protocol.load_fold(config, frozen, tmp_path)
    assert entries == expected
    dev_ids = set(asset["splits"]["dev"])
    assert dev_ids.isdisjoint({item["dialogue_id"] for view in views.values() for item in view})
    assert data["test_loader_built"] == 0
    for role, view in views.items():
        for item in view:
            original = asset["dialogues"][item["dialogue_id"]]
            assert item["utterance_ids"] == tuple(reversed(original["utterance_ids"]))
            assert item["labels"].tolist() == list(reversed(original["benchmark_label_ids"]))
            for key in DIMS:
                assert np.array_equal(item[key].numpy(), original[key][::-1])
        assert view.split == {"fit": "train", "held_out": "dev"}[role]


def test_training_materializes_fit_only(certified, tmp_path, monkeypatch):
    config, _, frozen, _, _ = certified
    getitem = projection_loader.TAVAuditDataset.__getitem__
    seen = []

    def observe(dataset, index):
        did = dataset.dialogue_ids[index]
        assert dataset.split == "train" and not did.startswith(config["held_out_session"])
        seen.append(did)
        return getitem(dataset, index)

    monkeypatch.setattr(projection_loader.TAVAuditDataset, "__getitem__", observe)
    views, _, _ = protocol.load_fold(config, frozen, tmp_path, fit_only=True)
    assert list(views) == ["fit"]
    assert seen == frozen["folds"]["Ses03"]["fit"]["dialogue_ids"]


@pytest.mark.parametrize(("fault", "error"), [
    ("changed_certificate", "J/K record"), ("changed_pickle", "SHA-256 mismatch"),
    ("canonical_source", "distinct Train/Dev-only"), ("hold_violation", "Test HOLD"),
])
def test_projection_fails_before_unpickle(certified, tmp_path, monkeypatch, fault, error):
    config, _, frozen, _, _ = certified
    path = tmp_path / protocol.PROJECTION_MANIFEST
    certificate = protocol.read_json(path)
    asset = tmp_path / projection_loader.ASSET
    if fault == "changed_certificate":
        certificate["sha256"][asset.name] = "d" * 64
    elif fault == "changed_pickle":
        asset.write_bytes(b"unverified pickle")
    elif fault == "canonical_source":
        canonical_sha = frozen["formal_f0"]["data_identity"]["sha256"]["asset"]
        certificate["sha256"][asset.name] = frozen["certified_projection"]["sha256"][asset.name] = canonical_sha
    else:
        certificate["test_labels_inspected"] = 1
    write_json_atomic(path, certificate)
    monkeypatch.setattr(projection_loader.pickle, "load", lambda _: pytest.fail("unverified pickle was opened"))
    with pytest.raises(ValueError, match=error):
        protocol.load_fold(config, frozen, tmp_path)


def test_changed_gold_tape_rejected(certified, tmp_path):
    config, _, frozen, _, _ = certified
    frozen["formal_f0"]["labeled_tape_sha256"]["dev"] = "0" * 64
    with pytest.raises(ValueError, match="certified dev order, IDs or gold"):
        protocol.load_fold(config, frozen, tmp_path)


@pytest.mark.parametrize(("session", "args", "expected"), [
    ("ses03", [], "requires --execute"),
    ("ses01", ["--execute"], "matching --release-fold"),
    ("ses04", ["--execute", "--release-fold", "Ses04"], "Ubuntu CUDA"),
])
def test_real_training_cli_gates_before_assets(monkeypatch, capsys, session, args, expected):
    cfg = g0.ROOT / protocol.CONFIG_DIR / f"{session}_seed100.yaml"
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(g0, "load_fold", lambda *a: pytest.fail("real assets opened after a gate failure"))
    with pytest.raises(SystemExit) as exc:
        g0.main(["train", "--config", str(cfg), "--projection-dir", "unused", "--experiment-date", "20261125", *args])
    assert exc.value.code == 2
    assert expected in capsys.readouterr().err


def test_fixed_endpoint_and_offline_export(certified, tmp_path, monkeypatch):
    config, raw, frozen, expected, _ = certified
    views, entries, data = protocol.load_fold(config, frozen, tmp_path)
    run_dir = tmp_path / "run"
    original_threads = torch.get_num_threads()
    evaluate = AuditRuntime.evaluate
    torch.set_num_threads(1)
    try:
        with monkeypatch.context() as m:
            m.setattr(AuditRuntime, "evaluate", lambda *args, **kwargs: pytest.fail("evaluation ran during fit"))
            m.setattr(AuditRuntime, "save_epoch", lambda *args, **kwargs: pytest.fail("selector ran during fit"))
            result = g0.train(run_dir, config, raw, data, views["fit"], entries["fit"],
                              device="cpu", git={"git_head": "synthetic"}, synthetic=True)
        assert AuditRuntime.evaluate is evaluate
        assert result["last_completed_epoch"] == result["optimizer_steps"] == 30
        assert result["status"] == "completed" and result["synthetic_only"]
        assert not (run_dir / "checkpoints" / "best_model.pt").exists()
        assert not (run_dir / "reports").exists()
        runtime, _ = g0.build_runtime(raw, {"train": views["fit"]}, "cpu", synthetic=True)
        assert isinstance(runtime.model, FusionTAV) and isinstance(runtime.model.fusion, DirectConcatFusion)
        assert len(runtime.model.graph_core.readouts) == 4
        assert runtime.model.classifier[4].p == 0.4
        opt = runtime.optimizer.param_groups[0]
        assert (opt["lr"], opt["weight_decay"], opt["betas"], opt["eps"], opt["foreach"]) == (
            5e-4, 0, (0.9, 0.999), 1e-6, False)
        assert len(runtime.curriculum.visible_indices(1)) == len(runtime.curriculum.visible_indices(30)) == 9
        with pytest.raises(ValueError, match="completed exact fold"):
            g0.evaluate(run_dir, config, raw, {**data, "changed": True}, views, entries,
                        device="cpu", synthetic=True)
        first = checkpoint_path(run_dir, 1)
        checkpoint = first.read_bytes()
        first.write_bytes(b"corrupt trajectory checkpoint")
        with pytest.raises(ValueError, match="SHA-256 mismatch at epoch 1"):
            g0.evaluate(run_dir, config, raw, data, views, entries, device="cpu", synthetic=True)
        assert not (run_dir / "reports" / "evaluation").exists()
        first.write_bytes(checkpoint)
        summary = g0.evaluate(run_dir, config, raw, data, views, entries, device="cpu", synthetic=True)
        with pytest.raises(FileExistsError):
            g0.evaluate(run_dir, config, raw, data, views, entries, device="cpu", synthetic=True)
        with pytest.raises(FileExistsError):
            g0.train(run_dir, config, raw, data, views["fit"], entries["fit"],
                     device="cpu", git={"git_head": "synthetic"}, synthetic=True)
    finally:
        torch.set_num_threads(original_threads)
    proof = protocol.read_json(summary / "evaluation_manifest.json")
    assert proof["optimizer_steps_performed"] == 0 and proof["endpoint_epoch"] == 30
    assert proof["endpoint_sha256"] == sha256_file(checkpoint_path(run_dir, 30))
    for name, digest in proof["files_sha256"].items():
        assert sha256_file(summary / name) == digest
    with (summary / "epoch_trends.csv").open(newline="") as stream:
        trend = list(csv.DictReader(stream))
    assert [(int(row["epoch"]), row["population"]) for row in trend] == [
        (epoch, role) for epoch in range(1, 31) for role in ("fit", "held_out")]
    with gzip.open(summary / "trajectory_predictions.csv.gz", "rt", newline="") as stream:
        trajectory = list(csv.DictReader(stream))
    assert len(trajectory) == 29 * sum(map(len, expected.values()))
    assert all(f"logit_{label}" in trajectory[0] for label in LABEL_VOCAB)
    for role, expected_rows in expected.items():
        with (summary / f"{role}_epoch30_predictions.csv").open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        assert [(r["utterance_id"], r["dialogue_id"], int(r["true_label_id"])) for r in rows] == expected_rows
        assert all(f"logit_{label}" in rows[0] for label in LABEL_VOCAB)
        metrics = protocol.read_json(summary / f"{role}_epoch30_metrics.json")
        assert sum(map(sum, metrics["confusion_matrix"])) == len(expected_rows)
        assert metrics["metrics"]["valid_target_count"] == len(expected_rows)


def test_reliability_edge_cases_use_full_population_and_stable_nll():
    did = "Ses03F_impro01"
    expected = [(f"{did}_F{i:03}", did, gold) for i, gold in enumerate((0, 0, 1))]
    rows = [{"utterance_id": uid, "dialogue_id": did, "true_label": label, "predicted_label": 0,
             "logits": [1000.0, -1000.0, 0.0, 0.0, 0.0, 0.0]} for uid, did, label in expected]
    result, export, bins, _, coverage = summarize(rows, expected)
    assert result["metrics"]["accuracy"] == pytest.approx(2 / 3)
    assert result["auxiliary"]["nll"] == pytest.approx(2000 / 3)
    assert result["auxiliary"]["brier"] == pytest.approx(2 / 3)
    assert result["auxiliary"]["ece10"] == pytest.approx(1 / 3)
    assert result["auxiliary"]["error_detection"]["auroc"] == 0.5
    assert result["auxiliary"]["error_detection"]["auprc"] == pytest.approx(1 / 3)
    assert result["auxiliary"]["high_confidence"]["error_support"] == 1
    assert [r["shift_group"] for r in export] == ["NO_HISTORY", "STABLE", "SHIFT"]
    assert sum(r["support"] for r in bins) == 3
    assert coverage[-1]["coverage"] == 1.0 and coverage[-1]["risk"] == pytest.approx(1 / 3)

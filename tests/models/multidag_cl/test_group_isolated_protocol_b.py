import csv
from copy import deepcopy
import gzip
import pickle

import numpy as np
import pytest
import torch
import yaml

from datasets.iemocap.tav_full_audit_dataset import DIMS, MANIFEST_FIELDS, SCHEMA, VERSION, collate, fingerprint, identity_sha
from models.multidag_cl.fusion.base import DirectConcatFusion
from models.multidag_cl.fusion.model import FusionTAV
from models.multidag_cl.ta_v0 import LABEL_VOCAB
from scripts.analysis.models.multidag_cl import group_isolated_comparison as comparison
from scripts.analysis.models.multidag_cl import pairwise_preflight_inputs as projection_loader
from scripts.analysis.models.multidag_cl.post_ph.internal_protocol import write_rows
from scripts.models.multidag_cl import group_isolated as g0
from scripts.runtime.multidag_cl_group_isolated import protocol
from scripts.runtime.multidag_cl_group_isolated.evaluation import TRAJECTORY_FIELDS, checkpoint_path, export_role, summarize
from scripts.runtime.multidag_cl_group_isolated.provenance import source_hashes
from scripts.runtime.multidag_cl_ta_v0.formal import sha256_file, write_json_atomic
from scripts.runtime.multidag_cl_ta_v0.training import masked_cross_entropy
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
    ls_path = g0.ROOT / protocol.G1_CONFIG_DIR / path.name
    assert protocol.load_config(ls_path) == protocol.fold_config(session, g1=True)
    assert g0.output_path(protocol.load_config(ls_path), "20261125") != g0.output_path(protocol.load_config(path), "20261125")
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


@pytest.mark.parametrize("g1", [False, True])
def test_training_materializes_fit_only(certified, tmp_path, monkeypatch, g1):
    config, _, frozen, _, _ = certified
    config = protocol.fold_config(config["held_out_session"], g1=g1)
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
@pytest.mark.parametrize("g1", [False, True])
def test_real_training_cli_gates_before_assets(monkeypatch, capsys, session, args, expected, g1):
    cfg = g0.ROOT / (protocol.G1_CONFIG_DIR if g1 else protocol.CONFIG_DIR) / f"{session}_seed100.yaml"
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(g0, "load_fold", lambda *a: pytest.fail("real assets opened after a gate failure"))
    with pytest.raises(SystemExit) as exc:
        g0.main(["train", "--config", str(cfg), "--projection-dir", "unused", "--experiment-date", "20261125", *args])
    assert exc.value.code == 2
    assert expected in capsys.readouterr().err


def test_g1_first_fold_explicit_release_and_config_lock(tmp_path, monkeypatch, capsys):
    path = g0.ROOT / protocol.G1_CONFIG_DIR / "ses03_seed100.yaml"
    monkeypatch.setattr(g0, "load_fold", lambda *args, **kwargs: pytest.fail("projection accessed before release"))
    with pytest.raises(SystemExit):
        g0.main(["train", "--config", str(path), "--projection-dir", "unused", "--experiment-date", "20261125",
                 "--execute", "--release-fold", "Ses02"])
    assert "matching --release-fold" in capsys.readouterr().err
    modified = tmp_path / "modified.yaml"
    cfg = protocol.load_config(path)
    for change in ({"label_smoothing": 0.05}, {"checkpoint_epoch": 18}, {"held_out_session": "Ses05"},
                   {"output": protocol.fold_config("Ses03")["output"]}):
        modified.write_text(yaml.safe_dump({**cfg, **change}), encoding="utf-8")
        with pytest.raises(ValueError):
            protocol.load_config(modified)


def test_masked_label_smoothing_matches_uniform_formula_and_legacy_gradient():
    logits = torch.tensor([[[2., 1., 0., -1., -2., -3.],
                            [float("nan")] * 6, [0., 2., 0., 4., 0., 0.]]], requires_grad=True)
    targets = torch.tensor([[0, 999, 4]])
    mask = torch.tensor([[True, False, True]])
    ce = torch.nn.functional.cross_entropy(logits[mask], targets[mask])
    default, count = masked_cross_entropy(logits, targets, mask)
    explicit, _ = masked_cross_entropy(logits, targets, mask, label_smoothing=0.0)
    assert count == 2 and torch.equal(default, explicit) and torch.equal(ce, explicit)
    legacy_grad = torch.autograd.grad(ce, logits)[0]
    assert torch.equal(legacy_grad, torch.autograd.grad(default, logits)[0])
    assert torch.equal(legacy_grad, torch.autograd.grad(explicit, logits)[0])
    smoothed, _ = masked_cross_entropy(logits, targets, mask, label_smoothing=0.1)
    soft = 0.9 * torch.nn.functional.one_hot(targets[mask], 6) + 0.1 / 6
    assert torch.allclose(soft.sum(-1), torch.ones(2))
    assert torch.allclose(soft[0], torch.tensor([.9 + .1 / 6, *([.1 / 6] * 5)]))
    expected = -(soft * logits[mask].log_softmax(-1)).sum(-1).mean()
    assert torch.allclose(smoothed, expected)
    grad = torch.autograd.grad(smoothed, logits)[0]
    assert torch.allclose(grad[mask], (logits[mask].softmax(-1) - soft) / count)
    assert torch.count_nonzero(grad[~mask]) == 0


def test_same_step_at_zero_and_g1_only_changes_fit_objective(certified, tmp_path):
    config, raw, frozen, _, _ = certified
    views, _, _ = protocol.load_fold(config, frozen, tmp_path)
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        default, default_init = g0.build_runtime(raw, {"train": views["fit"]}, "cpu", synthetic=True)
        batch = collate([views["fit"][0]])
        original = default.train_step(batch)
        zero, zero_init = g0.build_runtime(raw, {"train": views["fit"]}, "cpu", synthetic=True, train_label_smoothing=0.0)
        assert zero_init == default_init
        assert zero.train_step(batch) == original
        for p, q in zip(default.model.parameters(), zero.model.parameters()):
            assert torch.equal(p, q)
            assert (p.grad is None and q.grad is None) or torch.equal(p.grad, q.grad)
            assert default.optimizer.state[p].keys() == zero.optimizer.state[q].keys()
            for name in default.optimizer.state[p]:
                assert torch.equal(default.optimizer.state[p][name], zero.optimizer.state[q][name])
        smooth, ls_init = g0.build_runtime(raw, {"train": views["fit"]}, "cpu", synthetic=True, train_label_smoothing=.1)
        assert ls_init == default_init
        assert smooth.train_step(batch)["loss"] != original["loss"]
        assert smooth.model.state_dict().keys() == zero.model.state_dict().keys()
        smooth.model.load_state_dict(default.model.state_dict(), strict=True)
        eval_batch = collate([views["held_out"][0]])
        actual, nll = smooth.forward_batch(eval_batch, split="dev")
        baseline, ce = default.forward_batch(eval_batch, split="dev")
        assert torch.equal(actual.logits_ta, baseline.logits_ta) and torch.equal(nll, ce)
        changed = {**eval_batch, **{name: tensor.clone() for name, tensor in eval_batch.items() if name in DIMS}}
        for name in DIMS:
            changed[name][:, 3:, :] += 25
        prefix, _ = smooth.forward_batch(changed, split="dev")
        assert torch.equal(actual.logits_ta[:, :3], prefix.logits_ta[:, :3])
        assert not torch.equal(actual.logits_ta[:, 3:], prefix.logits_ta[:, 3:])
    finally:
        torch.set_num_threads(threads)


@pytest.mark.parametrize("g1", [False, True])
def test_fixed_endpoint_and_offline_export(certified, tmp_path, monkeypatch, g1):
    config, raw, frozen, expected, _ = certified
    config = protocol.fold_config(config["held_out_session"], g1=g1)
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
    if g1:
        assert result["fit_objective"] == {"name": "uniform_label_smoothed_ce", "label_smoothing": .1}
        assert result["evaluation_objective"] == "hard_label_nll"
        assert {"hard_label_nll", "brier", "ece10"}.issubset(trend[-1])
        assert "train_update_loss" not in trend[-1] and "loss" not in trend[-1]
        with (run_dir / "logs" / "fit_updates.csv").open(newline="") as stream:
            updates = list(csv.DictReader(stream))
        assert len(updates) == 30 and "train_update_loss" in updates[-1] and "hard_label_nll" not in updates[-1]
        held = protocol.read_json(summary / "held_out_epoch30_metrics.json")
        assert float(trend[-1]["hard_label_nll"]) == pytest.approx(held["auxiliary"]["nll"])
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


def test_source_inventory_includes_imports_and_frozen_inputs():
    config = (g0.ROOT / protocol.G1_CONFIG_DIR / "ses03_seed100.yaml").resolve()
    frozen_config = (g0.ROOT / protocol.CONFIG_DIR / "fold_identities.json").resolve()
    hashes = source_hashes(g0.ROOT, (config, frozen_config))
    assert hashes[config.relative_to(g0.ROOT).as_posix()] == sha256_file(config)
    assert hashes[frozen_config.relative_to(g0.ROOT).as_posix()] == sha256_file(frozen_config)
    assert "models/multidag_cl/fusion/model.py" in hashes
    assert "scripts/runtime/multidag_cl_ta_v0/training.py" in hashes


def test_paired_report_joins_by_id_and_rejects_changed_label():
    did = "Ses03F_impro01"
    rows = [{"utterance_id": f"{did}_F{gold:03}", "dialogue_id": did, "true_label": gold,
             "predicted_label": gold, "logits": [float(c == gold) for c in range(6)]} for gold in range(6)]
    corrected = {"predicted_label": 1, "logits": [0., 1., 0., 0., 0., 0.]}
    before = [{**rows[0], **corrected}, *rows[1:]]
    summary, details = comparison.paired_report(before, list(reversed(rows)))
    assert summary["paired"]["ALL"]["r"] == 1 and summary["paired"]["ALL"]["d"] == 0
    assert details[0]["recovered"] == 1 and len(summary["pair_confusions"]) == 6
    assert summary["g1"]["auxiliary"]["high_confidence"]["error_rate"] is None
    with pytest.raises(ValueError, match="order, IDs and labels"):
        comparison.paired_report(before, [{**rows[0], "true_label": 1}, *rows[1:]])


def test_pooled_comparison_reuses_four_authenticated_export_contracts(tmp_path, monkeypatch):
    _, raw = protocol.read_frozen(g0.ROOT)
    frozen = {"folds": {}}
    ctrl_root, g1_root = tmp_path / protocol.GROUP, tmp_path / protocol.G1_GROUP
    for session in protocol.SESSIONS:
        did = f"{session}F_impro01"
        expected = [(f"{did}_F{gold:03}", did, gold) for gold in range(6)]
        predictions = [
            {"utterance_id": uid, "dialogue_id": did, "true_label": gold,
             "predicted_label": gold, "logits": [float(c == gold) for c in range(6)]}
            for uid, did, gold in expected
        ]
        frozen["folds"][session] = {"held_out": protocol.describe(expected), "fit": {"dialogues": 48}}
        for is_g1, root in ((False, ctrl_root), (True, g1_root)):
            cfg = protocol.fold_config(session, g1=is_g1)
            run = root / "runs" / cfg["run_id"]
            export_dir = run / "reports" / "evaluation"
            export_dir.mkdir(parents=True)
            (run / "manifests").mkdir()
            (run / "logs").mkdir()
            identity = g0.run_identity(cfg, raw, {})
            header = {**protocol.HOLD, "TEST_TOUCHED": "NO", "run_identity": identity}
            manifest = {**header, "status": "completed", "synthetic_only": False, "f0_recipe": raw,
                        "seed": 100, "data_identity": {}, "initialization": {"fixture": True},
                        "steps_per_epoch": 3, "optimizer_steps": 90, "last_completed_epoch": 30,
                        "checkpoint_sha256": {"30": "fixture"}}
            write_json_atomic(run / "manifests" / "run_manifest.json", manifest)
            export_role(export_dir, "held_out", predictions, expected)
            with gzip.open(export_dir / "trajectory_predictions.csv.gz", "wt", encoding="utf-8", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(TRAJECTORY_FIELDS)
                for epoch in range(1, 30):
                    writer.writerows((epoch, "held_out", row["utterance_id"], did, row["true_label"],
                                      row["predicted_label"], *row["logits"]) for row in predictions)
            write_json_atomic(export_dir / "evaluation_manifest.json", {
                **header, "optimizer_steps_performed": 0, "calibration": None, "endpoint_epoch": 30,
                "checkpoint_selection": "fixed_epoch", "endpoint_sha256": "fixture",
                "files_sha256": {path.name: sha256_file(path) for path in export_dir.iterdir()},
            })
            updates = [{"epoch": e, "optimizer_steps": e * 3, "batch_tape_sha256": "fixture"} for e in range(1, 31)]
            write_rows(run / "logs" / "fit_updates.csv", updates, tuple(updates[0]))
    monkeypatch.setattr(comparison, "read_frozen", lambda _: (frozen, raw))
    output = comparison.compare(ctrl_root, g1_root, "fixture")
    summary = protocol.read_json(output / "paired_summary.json")
    assert set(summary) == {*protocol.SESSIONS, "pooled_oof"}
    assert summary["pooled_oof"]["g1"]["metrics"]["accuracy"] == 1.0
    assert summary["pooled_oof"]["paired"]["ALL"]["support"] == 24
    with (output / "pooled_oof_epoch_trends.csv").open(newline="") as stream:
        assert len(list(csv.DictReader(stream))) == 60
    with pytest.raises(FileExistsError):
        comparison.compare(ctrl_root, g1_root, "fixture")
    trajectory = next(g1_root.glob("runs/*/reports/evaluation/trajectory_predictions.csv.gz"))
    trajectory.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        comparison.compare(ctrl_root, g1_root, "corrupt")

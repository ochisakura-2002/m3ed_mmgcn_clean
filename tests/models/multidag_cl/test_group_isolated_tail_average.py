import csv
from copy import deepcopy
import gzip
from shutil import copy2

import pytest
import torch

from datasets.iemocap.tav_full_audit_dataset import DIMS, collate
from models.multidag_cl.fusion.base import DirectConcatFusion
from models.multidag_cl.fusion.model import FusionTAV
from models.multidag_cl.ta_v0 import LABEL_VOCAB
from scripts.analysis.models.multidag_cl.post_ph.internal_protocol import write_rows
from scripts.models.multidag_cl import group_isolated as g0
from scripts.models.multidag_cl import group_isolated_tail_average as g2
from scripts.runtime.multidag_cl_group_isolated import fixed_tail_average as tail, protocol
from scripts.runtime.multidag_cl_group_isolated.evaluation import TRAJECTORY_FIELDS, checkpoint_path, export_role
from scripts.runtime.multidag_cl_ta_v0.formal import sha256_file, write_json_atomic

from test_group_isolated_protocol_b import certified


def run_manifest(root, config, raw, data):
    run = root / "runs" / config["run_id"]
    (run / "manifests").mkdir(parents=True)
    steps = g0.steps_per_epoch(data["fold"]["fit"], raw)
    manifest = {
        **protocol.HOLD, "TEST_TOUCHED": "NO", "status": "completed", "seed": 100,
        "run_id": config["run_id"], "run_identity": g0.run_identity(config, raw, data),
        "f0_recipe": raw, "data_identity": data, "synthetic_only": False, "last_completed_epoch": 30,
        "endpoint_epoch": 30, "checkpoint_selection": "fixed_epoch", "initialization": {"fixture": True},
        "parameter_count": protocol.PARAMETERS, "steps_per_epoch": steps, "optimizer_steps": steps * 30,
        "checkpoint_sha256": {str(epoch): "a" * 64 for epoch in range(1, 31)},
    }
    if config["schema"] == protocol.G1_SCHEMA:
        manifest.update(fit_objective={"name": "uniform_label_smoothed_ce", "label_smoothing": 0.1},
                        evaluation_objective="hard_label_nll")
    write_json_atomic(run / "manifests" / "run_manifest.json", manifest)
    return run, manifest


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_same_state_equivalence_and_equal_formula(dtype):
    reference = {"weight": torch.tensor([1.123, -2.718, 0.114], dtype=dtype)}
    for count in (1, 4):
        result = tail.equal_average(iter([reference] * count), reference)
        assert torch.equal(result["weight"], reference["weight"])
        assert result["weight"].data_ptr() != reference["weight"].data_ptr()
    tensors = [torch.tensor(values, dtype=dtype) for values in ([2, -8, 1], [4, -2, 2], [6, 6, 3], [24, 20, 5])]
    result = tail.equal_average(({"weight": tensor} for tensor in tensors), reference)
    assert result["weight"].dtype == dtype
    assert torch.equal(result["weight"], torch.tensor([9., 4., 2.75], dtype=dtype))


@pytest.mark.parametrize("invalid", ["key", "shape", "dtype", "nonfloat", "nan", "extra"])
def test_unexpected_checkpoint_state_fails(invalid):
    reference = {"w": torch.arange(2, dtype=torch.float32), "_extra_state": {"schema": "fixture"}}
    bad = deepcopy(reference)
    if invalid == "key":
        bad["buffer"] = torch.ones(1)
    elif invalid == "shape":
        bad["w"] = torch.zeros(1)
    elif invalid == "dtype":
        bad["w"] = bad["w"].double()
    elif invalid == "nonfloat":
        bad["w"] = bad["w"].long()
    elif invalid == "nan":
        bad["w"][0] = torch.nan
    else:
        bad["_extra_state"]["schema"] = "changed"
    with pytest.raises(ValueError, match="checkpoint"):
        tail.equal_average([reference, bad], reference)


def test_ambiguous_model_state_refused():
    with pytest.raises(ValueError, match="BatchNorm"):
        tail.parameter_schema(torch.nn.BatchNorm1d(3))
    with pytest.raises(ValueError, match="unexpected model state"):
        tail.parameter_schema(torch.nn.Linear(3, 2))


@pytest.fixture
def trajectory(certified, tmp_path, monkeypatch):
    torch.set_num_threads(1)
    config, raw, frozen, _, _ = certified
    views, entries, data = protocol.load_fold(config, frozen, tmp_path)
    monkeypatch.setattr(g2, "read_frozen", lambda _: (frozen, raw))
    runtime, _ = g0.build_runtime(raw, {"train": views["fit"], "dev": views["held_out"]}, "cpu", synthetic=True)
    model = runtime.model
    assert isinstance(model, FusionTAV) and isinstance(model.fusion, DirectConcatFusion)
    assert len(model.graph_core.readouts) == 4 and not tuple(model.named_buffers())
    initial = tail.parameter_schema(model)
    assert len(initial) == 21 and initial["_extra_state"]["schema"] == "ct_multidag_fusion_model_v1"
    ce_root = tmp_path / protocol.GROUP
    run, manifest = run_manifest(ce_root, config, raw, data)
    weights = []
    for index, epoch in enumerate(tail.AVERAGING_EPOCHS):
        state = deepcopy(initial)
        state["classifier.0.bias"].add_(index / 8)
        weights.append(state["classifier.0.bias"])
        file = checkpoint_path(run, epoch)
        file.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"identity": manifest["run_identity"], "epoch": epoch,
                    "optimizer_steps": epoch * manifest["steps_per_epoch"], "model_state_dict": state}, file)
        manifest["checkpoint_sha256"][str(epoch)] = sha256_file(file)
    write_json_atomic(run / "manifests" / "run_manifest.json", manifest)
    return ce_root, run, manifest, frozen, raw, data, entries, views, runtime, weights


def test_evaluation_end_to_end_is_frozen_and_read_only(trajectory, tmp_path, monkeypatch):
    ce_root, run, _, _, raw, _, entries, views, runtime, weights = trajectory
    target = tmp_path / g2.G2_GROUP / "runs" / g2.g2_run_id("Ses03")
    initial_files = {p: (sha256_file(p), p.stat().st_mtime_ns) for p in run.rglob("*") if p.is_file()}
    projection = tmp_path / "projection"
    projection.mkdir()
    for name in (protocol.PROJECTION_MANIFEST, *protocol.read_json(tmp_path / protocol.PROJECTION_MANIFEST)["sha256"]):
        copy2(tmp_path / name, projection / name)

    def forbidden(*args, **kwargs):
        pytest.fail("G2 called optimizer.step or train_step")

    monkeypatch.setattr(torch.optim.AdamW, "step", forbidden)
    monkeypatch.setattr(runtime, "train_step", forbidden)
    monkeypatch.setattr(g2, "build_runtime", lambda *args: (runtime, {}))
    assert g2.evaluate(ce_root, target, "Ses03", projection, device="cpu") == target
    assert {p: (sha256_file(p), p.stat().st_mtime_ns) for p in initial_files} == initial_files
    manifest = protocol.read_json(target / "manifests" / "evaluation_manifest.json")
    protocol.validate_hold(manifest)
    assert manifest["identity"]["averaging_epochs"] == [21, 24, 27, 30]
    assert manifest["new_train_runs"] == manifest["optimizer_steps_performed"] == 0
    assert not runtime.optimizer.state and all(p.grad is None for p in runtime.model.parameters())
    saved = torch.load(target / g2.AVERAGED_CHECKPOINT, weights_only=True, map_location="cpu")
    assert saved["identity"] == manifest["identity"]
    assert torch.allclose(saved["model_state_dict"]["classifier.0.bias"], torch.stack(weights).mean(0))
    for name, value in runtime.model.named_parameters():
        assert torch.equal(value, saved["model_state_dict"][name])
    assert saved["model_state_dict"]["_extra_state"]["audit_config"] == raw
    fold = views["held_out"]
    batch = collate([fold[0]])
    changed = {**batch, **{key: batch[key].clone() for key in DIMS}}
    for key in DIMS:
        changed[key][:, 3:] += 8
    with torch.inference_mode():
        output, _ = runtime.forward_batch(batch, split="dev")
        perturbed, _ = runtime.forward_batch(changed, split="dev")
    assert torch.equal(output.logits_ta[:, :3], perturbed.logits_ta[:, :3])
    assert not torch.equal(output.logits_ta[:, 3:], perturbed.logits_ta[:, 3:])
    assert len(entries["held_out"]) == protocol.read_json(target / "reports" / "held_out_metrics.json")[
        "metrics"]["valid_target_count"]
    with pytest.raises(FileExistsError):
        g2.evaluate(ce_root, target, "Ses03", projection, device="cpu")


def test_hash_is_checked_before_torch_load(trajectory, monkeypatch):
    _, run, manifest, *_ = trajectory
    source = tail.source_inventory(run, manifest)[0]
    (run / source["path"]).write_bytes(b"tampered")
    monkeypatch.setattr(torch, "load", lambda *a, **kw: pytest.fail("unverified checkpoint deserialized"))
    with pytest.raises(ValueError, match="SHA-256 mismatch at epoch 21"):
        tail.read_checkpoint(run, source, manifest["run_identity"])


@pytest.mark.parametrize("fault", ["wrong_fold", "wrong_epoch", "wrong_steps"])
def test_authenticated_file_with_wrong_identity_is_rejected(trajectory, fault):
    _, run, manifest, *_ = trajectory
    source = tail.source_inventory(run, manifest)[0]
    path = run / source["path"]
    saved = torch.load(path, weights_only=True, map_location="cpu")
    if fault == "wrong_fold":
        saved["identity"]["run_id"] = protocol.fold_config("Ses02")["run_id"]
    elif fault == "wrong_epoch":
        saved["epoch"] = 24
    else:
        saved["optimizer_steps"] += 1
    torch.save(saved, path)
    with pytest.raises(ValueError, match="identity mismatch"):
        tail.read_checkpoint(run, {**source, "sha256": sha256_file(path)}, manifest["run_identity"])


@pytest.mark.parametrize("fault", ["test", "fold", "ls"])
def test_bad_manifest_cannot_open_projection_or_checkpoint(trajectory, tmp_path, monkeypatch, fault):
    root, run, manifest, *_ = trajectory
    if fault == "test":
        manifest["test_forward_count"] = 1
    elif fault == "fold":
        manifest["data_identity"]["held_out_session"] = "Ses04"
    else:
        manifest["fit_objective"] = {"name": "uniform_label_smoothed_ce", "label_smoothing": 0.1}
    write_json_atomic(run / "manifests" / "run_manifest.json", manifest)
    monkeypatch.setattr(g2, "load_fold", lambda *a: pytest.fail("opened projection after manifest failure"))
    monkeypatch.setattr(torch, "load", lambda *a, **kw: pytest.fail("opened checkpoint after manifest failure"))
    with pytest.raises(ValueError, match="Test HOLD|run_manifest|hard-label CE"):
        g2.evaluate(root, tmp_path / "new", "Ses03", tmp_path / "projection", device="cpu")


def export_reference(run, manifest, predictions, expected):
    export = run / "reports" / "evaluation"
    export.mkdir(parents=True)
    export_role(export, "held_out", predictions, expected)
    with gzip.open(export / "trajectory_predictions.csv.gz", "wt", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(TRAJECTORY_FIELDS)
        for epoch in range(1, 30):
            writer.writerows((epoch, "held_out", row["utterance_id"], row["dialogue_id"], row["true_label"],
                              row["predicted_label"], *row["logits"]) for row in predictions)
    write_json_atomic(export / "evaluation_manifest.json", {
        **protocol.HOLD, "TEST_TOUCHED": "NO", "run_identity": manifest["run_identity"],
        "optimizer_steps_performed": 0, "calibration": None, "endpoint_epoch": 30,
        "checkpoint_selection": "fixed_epoch", "endpoint_sha256": manifest["checkpoint_sha256"]["30"],
        "files_sha256": {path.name: sha256_file(path) for path in export.iterdir()},
    })
    (run / "logs").mkdir()
    steps = manifest["steps_per_epoch"]
    tape = [{"epoch": e, "optimizer_steps": e * steps, "batch_tape_sha256": "a" * 64} for e in range(1, 31)]
    write_rows(run / "logs" / "fit_updates.csv", tape, tuple(tape[0]))


def test_four_fold_comparison_from_existing_exports(certified, tmp_path, monkeypatch):
    config, raw, frozen, _, _ = certified
    _, _, data = protocol.load_fold(config, frozen, tmp_path)
    ce_root, ls_root = tmp_path / protocol.GROUP, tmp_path / protocol.G1_GROUP
    avg_root = tmp_path / g2.G2_GROUP
    for session in protocol.SESSIONS:
        did = f"{session}F_impro01"
        expected = [(f"{did}_{'M' if gold % 2 else 'F'}{gold:03}", did, gold) for gold in range(6)]
        fold = {**frozen["folds"][session], "held_out": protocol.describe(expected)}
        frozen["folds"][session] = fold
        identity = {**data, "held_out_session": session, "fold": fold}
        ce_run, ce = run_manifest(ce_root, protocol.fold_config(session), raw, identity)
        ls_run, ls = run_manifest(ls_root, protocol.fold_config(session, g1=True), raw, identity)
        arms = []
        for error in (True, False):
            predictions = []
            for uid, did, gold in expected:
                guess = 1 if error and gold == 0 else gold
                predictions.append({"utterance_id": uid, "dialogue_id": did, "true_label": gold,
                                    "predicted_label": guess, "logits": [8. * (c == guess) for c in range(6)]})
            arms.append(predictions)
        export_reference(ce_run, ce, arms[0], expected)
        export_reference(ls_run, ls, arms[1], expected)
        avg = avg_root / "runs" / g2.g2_run_id(session)
        (avg / "manifests").mkdir(parents=True)
        (avg / "reports").mkdir()
        g2.export_scoring(avg / "reports", "held_out", arms[1], expected)
        write_json_atomic(avg / "manifests" / "evaluation_manifest.json", {
            **protocol.HOLD, "TEST_TOUCHED": "NO", "data_identity": identity, "calibration": None,
            "identity": {"schema": g2.G2_SCHEMA, "held_out_session": session, "seed": 100, "run_id": avg.name,
                         "source_run_identity": ce["run_identity"], "sources": tail.source_inventory(ce_run, ce),
                         "source_manifest_sha256": sha256_file(ce_run / "manifests" / "run_manifest.json"),
                         "parameter_count": protocol.PARAMETERS, "averaging_epochs": list(tail.AVERAGING_EPOCHS),
                         "averaging_rule": "elementwise_equal_mean_fp32"},
            "optimizer_steps_performed": 0, "new_train_runs": 0, "scored_populations": ["held_out"],
            "timing_seconds": {"checkpoint_read_average_strict_load": 1.0,
                               "held_out_inference_and_prediction_collection": 2.0}, "device": "cpu",
            "files_sha256": {g2.G2_PREDICTIONS.as_posix(): sha256_file(avg / g2.G2_PREDICTIONS)},
        })
    monkeypatch.setattr(g2, "read_frozen", lambda _: (frozen, raw))
    monkeypatch.setattr(g2, "load_fold", lambda *a: pytest.fail("comparison opened projection"))
    monkeypatch.setattr(torch, "load", lambda *a, **kw: pytest.fail("comparison opened a checkpoint"))
    dest = avg_root / "analysis" / "batches" / "fixture"
    assert g2.compare(ce_root, ls_root, avg_root, dest) == dest
    summary = protocol.read_json(dest / "summary.json")
    assert set(summary) == {*protocol.SESSIONS, "pooled_oof"}
    pooled = summary["pooled_oof"]
    assert pooled["arms"][g2.AVG]["metrics"]["valid_target_count"] == 24
    pair = pooled["avg_vs"][g2.CE30]
    assert pair["paired"]["ALL"]["net"] == 4
    assert pair["paired_dialogue_bootstrap"][0]["resamples"] == 2000
    assert pair["paired_dialogue_bootstrap"][0]["seed"] == 7100
    assert pair["per_class_r_d_net"][LABEL_VOCAB[0]]["r"] == 4
    assert pooled["avg_vs"][g2.LS30]["delta_pp"]["weighted_f1"] == 0.0
    assert protocol.read_json(dest / "decision.json")["status"] == "DEVELOPMENT_GO"
    assert protocol.read_json(dest / "comparison_manifest.json")["total_seconds"][
        "held_out_inference_and_prediction_collection"] == 8.0
    avg_file = avg / g2.G2_PREDICTIONS
    avg_file.write_bytes(b"tampered")
    rejected = dest.with_name("tampered")
    with pytest.raises(ValueError, match="G2 prediction SHA-256 mismatch"):
        g2.compare(ce_root, ls_root, avg_root, rejected)
    assert not rejected.exists()

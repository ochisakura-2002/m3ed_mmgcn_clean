from copy import deepcopy

import numpy as np
import pytest
import torch

from scripts.analysis.models.multidag_cl.pairwise_fastkill_metrics import bootstrap
from scripts.analysis.models.multidag_cl.post_ph.internal_protocol import write_rows
from scripts.models.multidag_cl import group_isolated as g0
from scripts.models.multidag_cl import group_isolated_ls_factorial as cli
from scripts.models.multidag_cl import group_isolated_tail_average as g2
from scripts.runtime.multidag_cl_fusion.runtime import training_rng_identity
from scripts.runtime.multidag_cl_group_isolated import fixed_ls_factorial as g4, fixed_tail_average as tail, protocol
from scripts.runtime.multidag_cl_group_isolated.evaluation import ROLES, checkpoint_path, export_role
from scripts.runtime.multidag_cl_ta_v0.formal import json_identity, sha256_file, write_json_atomic
from scripts.runtime.multidag_cl_tav_full_audit_v1.runtime import AuditRuntime
from test_group_isolated_protocol_b import certified  # noqa: F401 - reused pytest fixture
from test_group_isolated_tail_average import run_manifest


@pytest.fixture(scope="module")
def reference():
    return cli.historical_imports()


def predictions(expected, *, correct):
    rows = []
    for uid, did, gold in expected:
        guess = (gold + 1) % 6 if not correct and int(did[-2:]) % 2 else gold
        rows.append({"utterance_id": uid, "dialogue_id": did, "true_label": gold, "predicted_label": guess,
                     "logits": [8. * (label == guess) for label in range(6)]})
    return rows


def write_endpoint(run, manifest, expected, updates, *, correct):
    export = run / "reports" / "evaluation"
    export.mkdir(parents=True)
    export_role(export, "held_out", predictions(expected, correct=correct), expected)
    write_json_atomic(export / "evaluation_manifest.json", {
        **protocol.HOLD, "TEST_TOUCHED": "NO", "run_id": run.name, "run_identity": manifest["run_identity"],
        "optimizer_steps_performed": 0, "calibration": None, "endpoint_epoch": 30,
        "scored_populations": list(ROLES), "checkpoint_selection": "fixed_epoch",
        "endpoint_sha256": manifest["checkpoint_sha256"]["30"],
        "files_sha256": {path.name: sha256_file(path) for path in export.iterdir()},
    })
    (run / "logs").mkdir()
    write_rows(run / "logs" / "fit_updates.csv", updates, tuple(updates[0]))


@pytest.fixture
def history(certified, reference, tmp_path, monkeypatch):
    original = torch.get_num_threads()
    torch.set_num_threads(1)
    _, raw, frozen, _, asset = certified
    ce_root, ls_root, g2_root = (tmp_path / group for group in (protocol.GROUP, protocol.G1_GROUP, g2.G2_GROUP))
    all_rows = []
    for did in asset["splits"]["train"]:
        item = asset["dialogues"][did]
        all_rows.extend(reversed([(uid, did, gold) for uid, gold in zip(item["utterance_ids"],
                                                                       item["benchmark_label_ids"])]))
    for session in protocol.SESSIONS:
        frozen["folds"][session] = {
            "fit": protocol.describe([row for row in all_rows if row[1][:5] != session]),
            "held_out": protocol.describe([row for row in all_rows if row[1][:5] == session]),
        }
    runs, biases = {}, {}
    for session in protocol.SESSIONS:
        config = protocol.fold_config(session, g1=True)
        views, entries, data = protocol.load_fold(config, frozen, tmp_path)
        runtime, init = g0.build_runtime(raw, {"train": views["fit"]}, "cpu", synthetic=True)
        rng = training_rng_identity("cpu")
        steps = g0.steps_per_epoch(frozen["folds"][session]["fit"], raw)
        tape = cli.batch_tape(runtime, entries["fit"], steps, steps * 30)
        updates = [{"epoch": e, "optimizer_steps": steps * e,
                    "batch_tape_sha256": json_identity([row["batch_sha256"] for row in tape[(e - 1) * steps:e * steps]])}
                   for e in range(1, 31)]
        ce_run, ce = run_manifest(ce_root, protocol.fold_config(session), raw, data)
        ls_run, ls = run_manifest(ls_root, config, raw, data)
        runs[session] = ls_run
        configs = {path.relative_to(g0.ROOT).as_posix(): sha256_file(path) for path in g4.source_configs(session, raw)}
        ce["initialization"] = init
        ls.update(initialization=init, training_initial_rng=rng, device="cpu", git={"git_head": "a" * 40},
                  runtime_provenance={"git_head": "a" * 40, "files_sha256": {**reference, **configs},
                                      "late_imports_sha256": {}})
        weights = tail.parameter_schema(runtime.model)
        biases[session] = []
        for index, epoch in enumerate(tail.AVERAGING_EPOCHS):
            state = deepcopy(weights)
            state["classifier.0.bias"].add_(index / 8)
            biases[session].append(state["classifier.0.bias"])
            path = checkpoint_path(ls_run, epoch)
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"identity": ls["run_identity"], "epoch": epoch, "optimizer_steps": epoch * steps,
                        "model_state_dict": state}, path)
            ls["checkpoint_sha256"][str(epoch)] = sha256_file(path)
        for run, manifest, correct in ((ce_run, ce, False), (ls_run, ls, True)):
            write_json_atomic(run / "manifests/run_manifest.json", manifest)
            write_endpoint(run, manifest, entries["held_out"], updates, correct=correct)
        avg = g2_root / "runs" / g2.g2_run_id(session)
        (avg / "manifests").mkdir(parents=True)
        (avg / "reports").mkdir()
        expected = entries["held_out"]
        g2.export_scoring(avg / "reports", "held_out", predictions(expected, correct=True), expected)
        write_json_atomic(avg / "manifests/evaluation_manifest.json", {
            **cli.ZERO_UPDATES, "data_identity": data, "calibration": None, "scored_populations": ["held_out"],
            "identity": g2.average_identity(ce, sha256_file(ce_run / "manifests/run_manifest.json"), session,
                                            tail.source_inventory(ce_run, ce)),
            "files_sha256": {g2.G2_PREDICTIONS.as_posix(): sha256_file(avg / g2.G2_PREDICTIONS)},
            "timing_seconds": {}, "device": "cpu",
        })
    projection = tmp_path / "projection"
    projection.mkdir()
    cert = tmp_path / protocol.PROJECTION_MANIFEST
    for name in (cert.name, *protocol.read_json(cert)["sha256"]):
        (projection / name).write_bytes((tmp_path / name).read_bytes())
    monkeypatch.setattr(cli, "historical_imports", lambda: reference)
    yield (ce_root, ls_root, g2_root), runs, biases, projection, frozen, raw
    torch.set_num_threads(original)


def test_fixed_four_fold_end_to_end(history, tmp_path, monkeypatch):
    roots, runs, biases, projection, frozen, raw = history
    assert g4.recipe()["averaging_epochs"] == [21, 24, 27, 30]
    paths = {path: (sha256_file(path), path.stat().st_mtime_ns) for root in roots
             for path in root.rglob("*") if path.is_file()}
    reads, scored = [], []
    reader, evaluate = tail.read_checkpoint, AuditRuntime.evaluate

    def read(run, source, identity):
        reads.append((run, source["epoch"]))
        return reader(run, source, identity)

    def score(runtime, *, split, **kwargs):
        assert len(reads) == 16 and split == "dev" and torch.is_inference_mode_enabled()
        scored.append(runtime)
        return evaluate(runtime, split=split, **kwargs)

    def forbidden(*args, **kwargs):
        pytest.fail("G4 ran a training step")

    monkeypatch.setattr(tail, "read_checkpoint", read)
    monkeypatch.setattr(AuditRuntime, "evaluate", score)
    monkeypatch.setattr(AuditRuntime, "train_step", forbidden)
    monkeypatch.setattr(torch.optim.AdamW, "step", forbidden)
    evaluation, result = tmp_path / "evaluation", tmp_path / "result"
    cli.evaluate(*roots, evaluation, projection, frozen, raw, device="cpu")
    assert reads == [(run, epoch) for run in runs.values() for epoch in tail.AVERAGING_EPOCHS]
    assert len(scored) == 4
    for session, runtime in zip(protocol.SESSIONS, scored):
        saved = torch.load(evaluation / session / cli.AVERAGED_CHECKPOINT, weights_only=True, map_location="cpu")
        assert torch.equal(saved["model_state_dict"]["classifier.0.bias"], torch.stack(biases[session]).mean(0))
        assert all(torch.equal(value, saved["model_state_dict"][key]) for key, value in runtime.model.named_parameters())
        assert saved["identity"]["fit_objective"]["label_smoothing"] == 0.1
        with pytest.raises(ValueError, match="CE trajectory"):
            tail.load_fixed_tail(runs[session], protocol.read_json(runs[session] / "manifests/run_manifest.json"), runtime.model)

    monkeypatch.setattr(torch, "load", lambda *a, **kw: pytest.fail("comparison loaded weights"))
    monkeypatch.setattr(cli, "load_fold", lambda *a, **kw: pytest.fail("comparison loaded features"))
    assert cli.compare(*roots, evaluation, result, frozen, raw) == result
    summary = protocol.read_json(result / "summary.json")
    assert list(summary) == [*protocol.SESSIONS, "pooled_oof"]
    pooled = summary["pooled_oof"]
    assert set(pooled["arms"]) == set(g4.ARMS)
    assert pooled["arms"][g4.LS_AVG]["metrics"]["valid_target_count"] == 72
    assert set(pooled["comparisons"][g4.PRIMARY]["paired"]) == {"ALL", "SHIFT", "STABLE", "NO_HISTORY"}
    assert len(pooled["comparisons"][g4.PRIMARY]["per_class_r_d_net"]) == 6
    assert g4.valid_sha(pooled["bootstrap_draws_sha256"])
    ref = tmp_path / "g3"
    ref.mkdir()
    hashes = {}
    for session in protocol.SESSIONS:
        (ref / session).mkdir()
        for arm in (g2.CE30, g2.LS30):
            name = f"{session}/{arm}_predictions.csv"
            (ref / name).write_bytes((result / name).read_bytes())
            hashes[name] = sha256_file(ref / name)
    write_json_atomic(ref / "summary.json", {
        scope: {"arms": {"ce_u_star": {"fixture": True}, "ls_u_star": {"fixture": True}}} for scope in summary
    })
    hashes["summary.json"] = sha256_file(ref / "summary.json")
    write_json_atomic(ref / "comparison_manifest.json", {
        **protocol.HOLD, "schema": cli.G3_SCHEMA, "output_sha256": hashes,
        "outer_cost_and_positions": {session: {"selected_position": {"optimizer_steps": steps}}
                                     for session, steps in zip(protocol.SESSIONS, (54, 72, 72, 48))},
    })
    with_ref = result.with_name("with_ref")
    cli.compare(*roots, evaluation, with_ref, frozen, raw, g3_analysis=ref)
    assert (with_ref / "summary.json").read_bytes() == (result / "summary.json").read_bytes()
    assert set(protocol.read_json(with_ref / "g3_descriptive_reference.json")["scopes"]) == set(summary)
    assert {p: (sha256_file(p), p.stat().st_mtime_ns) for p in paths} == paths
    (evaluation / "Ses04" / cli.PREDICTIONS).write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="prediction SHA-256"):
        cli.compare(*roots, evaluation, tmp_path / "invalid", frozen, raw)
    assert not (tmp_path / "invalid").exists()


@pytest.mark.parametrize(("fault", "message"), [
    ("hash", "SHA-256 mismatch"), ("missing", "epoch27.pt"), ("objective", "run_manifest differs"),
    ("source", "source inventory"), ("initialization", "initialization"), ("rng", "CPU RNG"),
    ("steps", "epoch or optimizer-step identity"), ("schema", "checkpoint tensor schema"),
])
def test_source_failure_precedes_any_evaluation(history, tmp_path, monkeypatch, fault, message):
    roots, runs, _, projection, frozen, raw = history
    run = runs["Ses04"]
    checkpoint, manifest_path = checkpoint_path(run, 27), run / "manifests/run_manifest.json"
    if fault == "hash":
        checkpoint.write_bytes(b"different")
    elif fault == "missing":
        checkpoint.unlink()
    else:
        manifest = protocol.read_json(manifest_path)
        if fault == "objective":
            manifest["fit_objective"]["label_smoothing"] = 0.2
        elif fault == "source":
            manifest["runtime_provenance"]["files_sha256"]["scripts/models/multidag_cl/group_isolated.py"] = "0" * 64
        elif fault == "initialization":
            manifest["initialization"]["shared_initial_state_sha256"] = "0" * 64
        elif fault == "rng":
            manifest["training_initial_rng"]["torch_cpu"] = "0" * 64
        else:
            saved = torch.load(checkpoint, weights_only=True, map_location="cpu")
            if fault == "steps":
                saved["optimizer_steps"] += 1
            elif fault == "schema":
                saved["model_state_dict"]["classifier.0.bias"] = torch.zeros(2)
            torch.save(saved, checkpoint)
            manifest["checkpoint_sha256"]["27"] = sha256_file(checkpoint)
        write_json_atomic(manifest_path, manifest)
    monkeypatch.setattr(AuditRuntime, "evaluate", lambda *a, **kw: pytest.fail("scored an unauthenticated batch"))
    destination = tmp_path / "stopped"
    with pytest.raises((ValueError, FileNotFoundError), match=message):
        cli.evaluate(*roots, destination, projection, frozen, raw, device="cpu")
    assert not destination.exists()


def test_factorial_bootstrap_uses_common_draws_and_correct_interaction(tmp_path):
    expected = [(f"{did}_F{gold:03}", did, gold) for session in protocol.SESSIONS[:2]
                for i in range(1, 4) for did in [f"{session}F_impro{i:02}"] for gold in range(6)]
    bad, good = (predictions(expected, correct=correct) for correct in (False, True))
    arms = dict(zip(g4.ARMS, (bad, good, bad, good)))
    same = g4.report_scope(tmp_path / "same", arms, expected)
    assert all(value == 0 for value in same["interaction"]["delta_pp"].values())
    assert all(value == 0 for ci in same["interaction"]["paired_dialogue_bootstrap"].values() for value in ci.values())
    arms[g4.LS_AVG] = bad
    different = g4.report_scope(tmp_path / "different", arms, expected)
    assert same["bootstrap_draws_sha256"] == different["bootstrap_draws_sha256"]
    labels = np.array([row[2] for row in expected])
    guesses = [np.array([row["predicted_label"] for row in arm]) for arm in (good, bad)]
    historical = bootstrap(labels, *guesses, [r[1] for r in expected], strata=[r[1][:5] for r in expected])
    for row in historical:
        actual = different["interaction"]["paired_dialogue_bootstrap"][row["metric"]]
        assert actual == {key: row[key] for key in actual}


def test_decision_keeps_reliability_signal_separate():
    def scope(wf1, macro=0.0, uar=0.0):
        return {"comparisons": {g4.PRIMARY: {"delta_pp": {"weighted_f1": wf1, "macro_f1": macro, "uar": uar},
                                             "reliability_delta": {"nll": -0.03}}}}

    summary = {session: scope(value) for session, value in zip(protocol.SESSIONS, (.01, .50, .40, -1.0))}
    summary["pooled_oof"] = scope(.5, -.30, -.30)
    assert g4.decision(summary)["status"] == "DEVELOPMENT_GO"
    summary["pooled_oof"] = scope(.49)
    decision = g4.decision(summary)
    assert decision["status"] == "NO_CLASSIFICATION_GO" and decision["reliability_review_signal"]

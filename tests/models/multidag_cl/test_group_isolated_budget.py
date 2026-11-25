import csv
from copy import deepcopy

import numpy as np
import pytest
import torch

from datasets.iemocap.tav_full_audit_dataset import DIMS
from scripts.models.multidag_cl import group_isolated_budget as cli
from scripts.runtime.multidag_cl_group_isolated import budget as g3
from scripts.runtime.multidag_cl_group_isolated import budget_history as history
from scripts.runtime.multidag_cl_group_isolated.evaluation import checkpoint_path
from scripts.runtime.multidag_cl_group_isolated.protocol import FoldView, SESSIONS, describe, read_frozen, read_json
from scripts.runtime.multidag_cl_paper_reimplementation.checkpoint import save_checkpoint_atomic
from scripts.runtime.multidag_cl_ta_v0.formal import json_identity, sha256_file, write_json_atomic
from scripts.runtime.multidag_cl_ta_v0.training import cpu_snapshot


@pytest.fixture(autouse=True)
def compact_cpu_runtime():
    original = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(original)


def make_population(dialogues):
    items, rows = [], []
    random = np.random.default_rng(21)
    for did in dialogues:
        uids = tuple(f"{did}_{'F' if i % 2 == 0 else 'M'}{i:03d}" for i in range(6))
        rows.extend((uid, did, label) for label, uid in enumerate(uids))
        items.append({
            "dialogue_id": did, "utterance_ids": uids, "speaker_ids_int": torch.tensor([0, 1] * 3),
            "labels": torch.arange(6), **{
                key: torch.from_numpy(random.standard_normal((6, dim), dtype=np.float32)) for key, dim in DIMS.items()
            },
        })
    return items, rows


def test_frozen_plan_and_optimizer_cost():
    frozen, raw = read_frozen(cli.ROOT)
    plans = g3.plan(frozen, raw)
    members = [{"spec": plan, "tape": [1] * max(plan["candidates"]), "identity": {"fit": plan["fit"]}}
               for plan in plans]
    groups, decision = g3.qualify_shared(members)
    assert len(plans) == 12 and decision == "VERIFIED" and len(groups) == 6
    assert sum(max(p["candidates"]) for p in plans) == 1170
    assert sum(max(len(x["tape"]) for x in group) for group in groups) == 630
    assert sum(len(set().union(*(x["spec"]["candidates"] for x in group))) for group in groups) == 42
    assert sum(len(x["spec"]["candidates"]) for x in members) == 60
    for key in ("identity", "tape"):
        changed = deepcopy(members)
        changed[0][key] = "changed"
        runs, status = g3.qualify_shared(changed)
        assert len(runs) == 12 and status == "NOT_SUPPORTED"


def test_session_macro_selection_and_exact_ties():
    frozen, _ = read_frozen(cli.ROOT)
    outer = "Ses01"
    fits = [s for s in SESSIONS if s != outer]
    tables = []
    for index, session in enumerate(fits):
        held = frozen["folds"][session]["held_out"]
        tables.append({
            "outer": outer, "validation": session, "fit": sorted(set(fits) - {session}),
            "validation_population": held,
            "scores": [{"optimizer_steps": step, "valid_targets": held["utterances"], "hard_nll": nll}
                       for step, nll in zip(g3.candidates(outer), ((1, 2, 4, 4, 4) if index < 2
                                                                  else (5, 1, 4, 4, 4)))],
        })
    assert g3.select(outer, tables, frozen)["selected_updates"] == 36
    tables[0]["scores"][0]["hard_nll"] = 0
    tables[2]["scores"][0]["hard_nll"] = 4
    assert g3.select(outer, tables, frozen)["selected_updates"] == 18
    tables[0]["scores"][0]["valid_targets"] -= 1
    with pytest.raises(ValueError, match="population or scoring"):
        g3.select(outer, tables, frozen)


def test_longer_fit_has_exact_earlier_checkpoints_and_batch_positions(tmp_path):
    _, raw = read_frozen(cli.ROOT)
    items, rows = make_population([f"Ses02F_impro{i:02d}" for i in range(17)])
    fit = FoldView(items, "train")
    short, long = (tmp_path / name for name in ("short", "long"))
    prepared = g3.prepare(raw, fit, rows, {"certified": "synthetic"}, {}, 3, device="cpu", synthetic=True)
    first = g3.train_fixed(short, prepared, [1, 3])
    prepared = g3.prepare(raw, fit, rows, {"certified": "synthetic"}, {}, 4, device="cpu", synthetic=True)
    second = g3.train_fixed(long, prepared, [1, 3, 4])
    assert first["identity"] == second["identity"]
    assert first["batch_tape"] == second["batch_tape"][:3]
    for step in ("1", "3"):
        states = [g3.load_state(path, manifest["identity"], manifest["checkpoints"][step])
                  for path, manifest in ((short, first), (long, second))]
        for key, tensor in states[0].items():
            assert torch.equal(tensor, states[1][key]) if torch.is_tensor(tensor) else tensor == states[1][key]
    position = second["checkpoints"]["3"]
    assert (position["epoch"], position["batch_in_epoch"], position["dialogue_exposure"]) == (2, 1, 33)
    assert position["valid_target_exposure"] == 33 * 6
    assert not position["end_of_epoch"]
    checkpoint = long / position["path"]
    checkpoint.write_bytes(b"wrong bytes")
    with pytest.raises(ValueError, match="SHA-256"):
        g3.load_state(long, second["identity"], position)


@pytest.fixture
def mini_folds(monkeypatch):
    frozen, raw = read_frozen(cli.ROOT)
    items, rows = make_population([f"{session}F_impro0{i}" for session in SESSIONS for i in range(1, 5)])
    for session in SESSIONS:
        frozen["folds"][session] = {
            role: describe([row for row in rows if (row[1][:5] == session) == (role == "held_out")])
            for role in ("fit", "held_out")
        }
    calls = []

    def load(config, _, projection, *, fit_only=False):
        session = config["held_out_session"]
        calls.append((session, fit_only))
        groups = {"fit": [s for s in SESSIONS if s != session], "held_out": [session]}
        fit = FoldView(items, "train")
        values = {key: g3.subset(fit, rows, groups[key], split)
                  for key, split in (("fit", "train"), ("held_out", "dev")) if key == "fit" or not fit_only}
        return ({key: value[0] for key, value in values.items()}, {key: value[1] for key, value in values.items()},
                {"outer": session, "projection": {"certified": "synthetic"}})

    monkeypatch.setattr(cli, "load_fold", load)
    monkeypatch.setattr(g3, "candidates", lambda _: (1, 2))
    monkeypatch.setattr(g3, "plan", lambda *_: [
        {"outer": outer, "validation": val, "fit": list(pair), "steps_per_epoch": 1, "candidates": [1, 2]}
        for outer, val, pair in g3.assignments()
    ])
    monkeypatch.setattr(cli, "ENDPOINT", 3)
    return frozen, raw, calls


def test_inner_lock_outer_refit_and_offline_report(tmp_path, monkeypatch, mini_folds):
    frozen, raw, calls = mini_folds
    inner, outer, projection = (tmp_path / name for name in ("inner", "outer", "projection"))
    events = []
    train, score = g3.train_fixed, g3.score_state

    def observe_train(*args, **kwargs):
        events.append("train")
        return train(*args, **kwargs)

    def observe_score(*args, **kwargs):
        events.append("score")
        return score(*args, **kwargs)

    monkeypatch.setattr(g3, "train_fixed", observe_train)
    monkeypatch.setattr(g3, "score_state", observe_score)
    source = {"files_sha256": {}, "environment": {"fixture": True}}
    lock = cli.run_inner(inner, projection, frozen, raw, device="cpu", independent=False, source=source, synthetic=True)
    assert calls == [(session, True) for session in SESSIONS]
    assert events == ["train"] * 6 + ["score"] * 24
    assert lock["training_updates"] == 12 and lock["inner_reuse"] == "VERIFIED"
    assert len(list((inner / "selection").glob("*/*/scores.json"))) == 12
    digest = sha256_file(inner / "budget_lock.json")
    assert cli.read_lock(inner, digest, frozen, raw, synthetic=True)["selections"] == lock["selections"]
    with pytest.raises(ValueError, match="SHA-256"):
        cli.read_lock(inner, "0" * 64, frozen, raw, synthetic=True)
    with pytest.raises(ValueError, match="same execution mode"):
        cli.read_lock(inner, digest, frozen, raw)
    calls.clear()
    events.clear()
    result = cli.run_outer(outer, projection, lock, digest, frozen, raw, None,
                           source=source, device="cpu", synthetic=True)
    assert calls == [(s, flag) for flag in (True, False) for s in SESSIONS]
    assert events == ["train"] * 8 + ["score"] * 16
    for s, fold in result.items():
        assert fold["mode"] == "fresh_paired_refit"
        assert fold["selected_position"]["optimizer_steps"] == lock["selections"][s]["selected_updates"]
        assert sum(x["updates"] for x in fold["cost"]) == 6
    # Exercise the offline G2 reader interface using synthetic G2 evidence.
    monkeypatch.setattr(cli, "authenticated_run", lambda *_: ({}, "fixture"))

    def load_avg(run, *args):
        expected = args[-1]
        rows = [{"utterance_id": uid, "dialogue_id": did, "true_label": label,
                 "predicted_label": label, "logits": [float(k == label) for k in range(6)]}
                for uid, did, label in expected]
        return rows, {"synthetic_reference": run.name}

    monkeypatch.setattr(cli, "read_average", load_avg)
    report = tmp_path / "analysis"
    cli.compare(report, outer, tmp_path / "ce", tmp_path / "g2", lock, digest, frozen, raw, synthetic=True)
    pooled = read_json(report / "summary.json")["pooled_oof"]
    assert set(pooled["arms"]) == {*cli.ARMS, cli.AVG}
    assert pooled["arms"][cli.CE_U]["metrics"]["valid_target_count"] == 96
    assert f"{cli.LS_U}_vs_{cli.CE_U}" in pooled["comparisons"]
    table = next((inner / "selection").glob("*/*/scores.json"))
    table.write_bytes(table.read_bytes() + b" ")
    with pytest.raises(ValueError, match="differs from the independently scored tables"):
        cli.read_lock(inner, digest, frozen, raw, synthetic=True)


def test_history_requires_source_rng_tapes_and_checkpoint_bytes(tmp_path, monkeypatch):
    frozen, raw = read_frozen(cli.ROOT)
    items, rows = make_population([f"Ses02F_impro{i:02d}" for i in range(17)])
    environment = {"torch_deterministic_algorithms": True, "cublas_workspace_config": ":4096:8"}
    source = {"environment": environment}
    prepared = g3.prepare(raw, FoldView(items, "train"), rows, {}, source, 60, device="cpu", synthetic=True)
    config = cli.fold_config("Ses01")
    run = tmp_path / "runs" / config["run_id"]
    dependency = "scripts/runtime/multidag_cl_group_isolated/protocol.py"
    reference = {dependency: sha256_file(cli.ROOT / dependency)}
    manifest = {
        "initialization": prepared.initialization, "training_initial_rng": prepared.initial_rng,
        "runtime_provenance": {"environment": environment, "files_sha256": reference, "late_imports_sha256": {}},
        "steps_per_epoch": 2, "run_id": config["run_id"], "run_identity": {"fixture": config["run_id"]},
        "checkpoint_sha256": {},
    }
    for epoch in (1, 30):
        path = checkpoint_path(run, epoch)
        path.parent.mkdir(parents=True, exist_ok=True)
        save_checkpoint_atomic(path, {"identity": manifest["run_identity"], "epoch": epoch, "optimizer_steps": epoch * 2,
                                      "model_state_dict": cpu_snapshot(prepared.runtime.model.state_dict())})
        manifest["checkpoint_sha256"][str(epoch)] = sha256_file(path)
    log = run / "logs" / "fit_updates.csv"
    log.parent.mkdir()
    with log.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("epoch", "optimizer_steps", "batch_tape_sha256"))
        for e in range(1, 31):
            writer.writerow((e, 2 * e, json_identity([r["batch_sha256"] for r in prepared.tape[2 * e - 2:2 * e]])))
    monkeypatch.setattr(history, "authenticated_run", lambda *_: (manifest, "fixture"))
    args = (tmp_path, "Ses01", False, frozen, raw, prepared, source, reference)
    direct = history.inspect_history(*args, 2)
    assert direct["selected"] is not None and direct["evidence"]["selected"]["optimizer_steps"] == 2
    prefix = history.inspect_history(*args, 3)
    assert prefix["selected"] is None and list(prefix["anchors"]) == [1]
    history.verify_anchor(prefix, 1, prepared.runtime.model)
    with torch.no_grad():
        next(prepared.runtime.model.parameters()).add_(1)
    with pytest.raises(ValueError, match="differs from historical weights"):
        history.verify_anchor(prefix, 1, prepared.runtime.model)
    manifest["training_initial_rng"] = {}
    with pytest.raises(ValueError, match="initial RNG"):
        history.inspect_history(*args, 2)
    manifest["training_initial_rng"] = prepared.initial_rng
    with pytest.raises(ValueError, match="source inventory"):
        history.inspect_history(*args[:-1], {"missing.py": "0" * 64}, 2)
    log.write_text("epoch,optimizer_steps,batch_tape_sha256\n", encoding="utf-8")
    with pytest.raises(ValueError, match="batch tape differs"):
        history.inspect_history(*args, 2)


@pytest.mark.parametrize("selected,mode,runs,updates", [
    (2, "verified_historical_checkpoints", 0, 0),
    (3, "verified_prefix_replay", 2, 6),
    (3, "fresh_paired_refit", 3, 14),
])
def test_outer_pair_routes(tmp_path, monkeypatch, selected, mode, runs, updates):
    frozen, raw = read_frozen(cli.ROOT)
    monkeypatch.setattr(cli, "ENDPOINT", 3)
    items, rows = make_population([f"Ses02F_impro{i:02d}" for i in range(17)])
    fit = FoldView(items, "train")
    source, projection = {"fixture": True}, {"fixture": "projection"}
    expected = {}
    for is_ls in (False, True):
        directory = tmp_path / ("historical_ls" if is_ls else "historical_ce")
        prepared = g3.prepare(raw, fit, rows, projection, source, 6, smoothing=0.1 if is_ls else 0.0,
                              device="cpu", synthetic=True)
        record = g3.train_fixed(directory, prepared, {2, 3, 6})
        expected[is_ls] = {u: g3.load_state(directory, record["identity"], record["checkpoints"][str(u)])
                           for u in (2, 3, 6)}

    def inspect(_root, _outer, is_ls, *_args):
        return {"e30": expected[is_ls][6], "selected": expected[is_ls][2] if selected == 2 else None,
                "evidence": {"fixture": is_ls}, "is_ls": is_ls}

    def anchor(evidence, epoch, model):
        assert epoch == 1
        for key, value in model.state_dict().items():
            reference = expected[evidence["is_ls"]][2][key]
            assert torch.equal(value, reference) if torch.is_tensor(value) else value == reference
        if mode == "fresh_paired_refit":
            raise ValueError("fixture anchor mismatch")

    monkeypatch.setattr(cli, "inspect_history", inspect)
    monkeypatch.setattr(cli, "verify_anchor", anchor)
    actual, summary = cli.paired_fit(tmp_path / "outer", "Ses01", selected, fit, rows, {"projection": projection},
                                    frozen, raw, (tmp_path / "ce", tmp_path / "ls"), source, {},
                                    device="cpu", synthetic=True)
    assert summary["mode"] == mode
    assert len(summary["cost"]) == runs and sum(c["updates"] for c in summary["cost"]) == updates
    for is_ls, pair in actual.items():
        for state, step in zip(pair, (6, selected)):
            for key, value in state.items():
                reference = expected[is_ls][step][key]
                assert torch.equal(value, reference) if torch.is_tensor(value) else value == reference

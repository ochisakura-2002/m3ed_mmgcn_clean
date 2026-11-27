import csv
from pathlib import Path
import tarfile
from types import SimpleNamespace

import pytest

from scripts.analysis.models.multidag_cl import original_dev_analysis as analysis
from scripts.analysis.models.multidag_cl.plot_original_dev_alignment import render
from scripts.runtime.multidag_cl_fusion import original_dev_alignment as alignment
from scripts.runtime.multidag_cl_fusion.runtime import paired_runtime
from scripts.runtime.multidag_cl_group_isolated.protocol import read_frozen
from scripts.runtime.multidag_cl_ta_v0.formal import sha256_file, write_json_atomic
from scripts.runtime.multidag_cl_tav_full_audit_v1.formal import execute
from scripts.runtime.multidag_cl_tav_full_audit_v1.runtime import SyntheticDialogues
from scripts.workflows.benchmarks import run_original_dev_alignment as workflow


def synthetic():
    datasets, entries = {}, {}
    for session, split in enumerate(("train", "dev"), 1):
        dataset = SyntheticDialogues(split)
        if split == "train":
            dataset.items = [dict(dataset.items[i % len(dataset.items)]) for i in range(17)]
        rows = []
        for index, item in enumerate(dataset.items):
            did = f"Ses{session:02d}F_impro{index:02d}"
            ids = tuple(f"{did}_{'FM'[s]}{i:03d}" for i, s in enumerate(item["speaker_ids_int"]))
            item.update(dialogue_id=did, utterance_ids=ids)
            rows.extend((uid, did, int(gold)) for uid, gold in zip(ids, item["labels"]))
        datasets[split], entries[split] = dataset, rows
    return datasets, entries


def test_reuse_official_loop_and_end_to_end_analysis(artifact_dir, monkeypatch):
    datasets, entries = synthetic()
    frozen, raw = read_frozen(alignment.ROOT)
    config = alignment.read_config(frozen)
    git = {"git_head": "synthetic"}
    projection = {"sha256": {}}
    options = {"synthetic_only": True, "git_head": git["git_head"]}
    original, _ = paired_runtime(raw, datasets, device="cpu", **options)
    reference = execute(original, raw, artifact_dir / "baseline", git=git, synthetic_epochs=2)
    lock = {"status": "LOCKED", "selected_updates": 1, "scores": [], "means": [{"updates": 1, "hard_nll": 1.0}]}
    runs = {}
    for arm in ("D0", "D1"):
        run = artifact_dir / arm.lower()
        runtime = workflow.make_runtime(raw, datasets, run, config, arm, lock, reference, git,
                                        expected=reference, synthetic=True)
        assert alignment.preflight(runtime, reference["batch_pairing"], entries, epochs=2)["label_perturbation"] == "PASS"
        workflow.fit(runtime, raw, git, projection, synthetic_epochs=2)
        runs[arm] = workflow.checked_arm(runtime, synthetic_epochs=2)
        assert set(runs[arm]["snapshots"]) == {"u_star", "epoch_01", "epoch_02"}
        assert runs[arm]["snapshots"]["u_star"]["optimizer_steps"] == 1
        assert runs[arm]["snapshots"]["u_star"]["epoch_complete"] is False
        assert runs[arm]["run"]["shared_initialization"] == reference["shared_initialization"]
        assert runs[arm]["run"]["batch_pairing"] == reference["batch_pairing"]
    assert alignment.epoch_rows(artifact_dir / "d0") == alignment.epoch_rows(artifact_dir / "baseline")
    assert runs["D0"]["best_evidence"]["predictions"] != runs["D1"]["best_evidence"]["predictions"]
    assert runs["D0"]["best_evidence"]["predictions"] == alignment.read_json(
        artifact_dir / "baseline/reports/validation_best_dev_evidence.json")["predictions"]

    monkeypatch.setattr(analysis, "ENDPOINT", 2)
    endpoints, sources, learning = {}, {}, []
    for arm, record in runs.items():
        run = artifact_dir / arm.lower()
        fresh = workflow.make_runtime(raw, datasets, run, config, arm, lock, reference, git,
                                      expected=reference, synthetic=True)
        results, provenance, curves = analysis.replay(fresh, run, record, entries, [1, 2])
        endpoints.update(results)
        sources.update(provenance)
        learning.extend(curves)
    tables = artifact_dir / "bundle/tables"
    tables.mkdir(parents=True)
    analysis.stage_tables(tables, endpoints, sources, learning, entries["dev"], lock)
    with (tables / "overall_metrics.csv").open(encoding="utf-8", newline="") as stream:
        assert {r["arm"] for r in csv.DictReader(stream) if r["status"] == "SCORED"} == set(analysis.ENDPOINTS)
    with (tables / "r_d_net.csv").open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            assert int(row["net"]) == int(row["r"]) - int(row["d"])
    with (tables / "paired_bootstrap.csv").open(encoding="utf-8", newline="") as stream:
        pairs = {(r["contrast"], r["metric"]): float(r["delta"]) for r in csv.DictReader(stream)}
        for metric in analysis.METRICS:
            assert pairs[(analysis.INTERACTION, metric)] == pytest.approx(
                pairs[("LS AVG − LS E30", metric)] - pairs[("CE AVG − CE E30", metric)])
    render(tables, artifact_dir / "bundle/figures")
    archive = artifact_dir / "result.tar.gz"
    assert workflow.ship(artifact_dir / "bundle", archive, "PASS") == sha256_file(archive)
    with tarfile.open(archive) as result:
        names = result.getnames()
    assert len([name for name in names if name.endswith(".svg")]) == 13
    assert not any(name.endswith((".pt", ".pkl")) for name in names)

    partial = artifact_dir / "partial"
    partial.mkdir()
    partial_ends = {k: v for k, v in endpoints.items() if not k.endswith("U*")}
    partial_sources = {k: sources[k] for k in partial_ends}
    missing = {"status": "UNAVAILABLE", "reason": "synthetic missing historical assets"}
    scores, intervals = analysis.stage_tables(partial, partial_ends, partial_sources, learning, entries["dev"], missing)
    with (partial / "overall_metrics.csv").open(encoding="utf-8", newline="") as stream:
        assert [r["arm"] for r in csv.DictReader(stream) if r["status"] == "UNAVAILABLE"] == ["CE U*", "LS U*"]
    figure_result = render(partial, artifact_dir / "partial_figures")
    assert figure_result["figures"] == 12
    analysis.write_results(artifact_dir / "RESULTS.md", scores, intervals, missing)
    assert "D3 状态为 UNAVAILABLE" in (artifact_dir / "RESULTS.md").read_text(encoding="utf-8")


def test_budget_uses_common_steps_and_session_macro_nll(monkeypatch):
    frozen, raw = read_frozen(alignment.ROOT)
    config = alignment.read_config(frozen)
    folds = alignment.SESSIONS
    rows = {}
    for count, (session, steps) in enumerate(zip(folds, (5, 5, 5, 6)), 1):
        target_logits = {30: 1 if count == 4 else 0, 60: 0 if count == 4 else 0.5}
        rows[session] = {}
        for epoch in range(1, 31):
            logit = target_logits.get(epoch * steps, -1)
            rows[session][epoch] = [{"logits": [logit, 0, 0, 0, 0, 0],
                                     "true_label": 0, "predicted_label": int(logit < 0)}] * count

    def manifest(session):
        return {"steps_per_epoch": 6 if session == "Ses04" else 5,
                "checkpoint_sha256": {str(i): "a" * 64 for i in range(1, 31)}}

    monkeypatch.setattr(alignment, "authenticated_run", lambda run, fold, *args:
                        (manifest(fold["held_out_session"]), alignment.BUDGET_EXPORT_SHA256[fold["held_out_session"]][0]))

    def export(root, fold, *args):
        session = fold["held_out_session"]
        main, evaluation, updates, trajectory, endpoint = alignment.BUDGET_EXPORT_SHA256[session]
        proof = {"run_manifest_sha256": main, "evaluation_manifest_sha256": evaluation, "fit_updates_sha256": updates,
                 "prediction_files_sha256": {"trajectory_predictions.csv.gz": trajectory,
                                             "held_out_epoch30_predictions.csv": endpoint}}
        return None, rows[session], manifest(session), None, proof

    monkeypatch.setattr(alignment, "read_export", export)
    lock = alignment.select_budget(config, frozen, raw, Path("unused"))
    assert lock["status"] == "LOCKED" and lock["selected_updates"] == 60
    assert {(r["session"], r["epoch"], r["n"]) for r in lock["scores"] if r["updates"] == 60} == {
        ("Ses01", 12, 1), ("Ses02", 12, 2), ("Ses03", 12, 3), ("Ses04", 10, 4),
    }
    micro = {u: sum(r["n"] * r["hard_nll"] for r in lock["scores"] if r["updates"] == u) for u in (30, 60)}
    assert micro[30] < micro[60]
    assert alignment.select_budget(config, frozen, raw, None)["status"] == "UNAVAILABLE"

    def missing(*args):
        raise FileNotFoundError("trajectory_predictions.csv.gz")

    def checkpoint_replay(run, manifest, fold, raw, train, epochs, device):
        session = fold["held_out"]["sessions"][0]
        assert train == "train only"
        assert epochs == [step // manifest["steps_per_epoch"] for step in config["budget_candidate_updates"]]
        return {e: rows[session][e] for e in epochs}, {"checkpoint": "a" * 64}

    monkeypatch.setattr(alignment, "read_export", missing)
    monkeypatch.setattr(alignment, "replay_budget_weights", checkpoint_replay)
    replayed = alignment.select_budget(config, frozen, raw, Path("unused"), train="train only")
    assert replayed["selected_updates"] == 60
    assert {source["mode"] for source in replayed["sources"].values()} == {"historical_checkpoint_replay"}

    monkeypatch.setattr(alignment, "read_export", lambda *args: (None, [], {}, None, {}))
    assert alignment.select_budget(config, frozen, raw, Path("unused"))["status"] == "UNAVAILABLE"


def test_tampered_stage_a_receipt_stops_approval(artifact_dir):
    path = artifact_dir / "stage_a.json"
    write_json_atomic(path, {"status": "PASS"})
    approved = sha256_file(path)
    write_json_atomic(path, {"status": "FAIL"})
    with pytest.raises(ValueError, match="receipt SHA-256"):
        workflow.checked_approval(SimpleNamespace(manifest_root=artifact_dir), approved, {})


def test_budget_lock_cannot_change_after_review(artifact_dir):
    paths = SimpleNamespace(manifest_root=artifact_dir / "manifests", runs_root=artifact_dir / "runs")
    paths.manifest_root.mkdir()
    run = workflow.arm_run(paths, "D0")
    for name in workflow.RUN_EVIDENCE:
        path = run / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("synthetic", encoding="utf-8")
    config = {"budget_candidate_updates": [30, 60]}
    lock_path = paths.manifest_root / "budget_lock.json"
    lock = {**workflow.HOLD, "candidate_updates": [30, 60], "selected_updates": 30, "status": "LOCKED"}
    write_json_atomic(lock_path, lock)
    receipt = paths.manifest_root / "stage_a.json"
    write_json_atomic(receipt, {**workflow.HOLD, "run_id": run.name, "status": "PASS",
        "config_sha256": workflow.json_identity(config), "d3_status": "LOCKED",
        "numerical_audit": {"numerical_identity": True}, "budget_sha256": sha256_file(lock_path),
        "run_evidence_sha256": {name: sha256_file(run / name) for name in workflow.RUN_EVIDENCE}})
    approved = sha256_file(receipt)
    assert workflow.checked_approval(paths, approved, config)[1] == lock
    write_json_atomic(lock_path, {**lock, "selected_updates": 60})
    with pytest.raises(ValueError, match="budget lock changed"):
        workflow.checked_approval(paths, approved, config)

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


@pytest.fixture
def projected_inputs(artifact_dir, monkeypatch):
    def rows(*ids):
        return [(uid, uid.rsplit("_", 1)[0], i % 6) for i, uid in enumerate(ids)]

    entries = {
        "train": rows("Ses01F_impro01_F000", "Ses04F_impro01_F000", "Ses04F_impro01_M001"),
        "dev": rows("Ses04F_impro02_F000", "Ses04F_impro02_M001"),
    }
    summaries = {split: alignment.describe(tape) for split, tape in entries.items()}
    counts = {split: {key: summary[key] for key in ("dialogues", "utterances")}
              for split, summary in summaries.items()}
    historical = {
        "data_identity": {"sha256": {"asset": "a" * 64}},
        "batch_pairing": {"population": {split: {**counts[split], "sha256": summary["order_sha256"]}
                                         for split, summary in summaries.items()}},
    }
    evidence = {"predictions": [
        {"utterance_id": uid, "dialogue_id": did, "true_label": gold} for uid, did, gold in entries["dev"]
    ]}
    frozen = {
        "certified_projection": {"schema": "ct_multidag_train_dev_projection_v1",
                                 "sha256": {"train_dev_tav.pkl": "b" * 64}, "row_order_identity_sha256": "c" * 64},
        "formal_f0": {"labeled_tape_sha256": {split: summary["labeled_tape_sha256"]
                                             for split, summary in summaries.items()}},
    }
    projection = {**frozen["certified_projection"], **alignment.HOLD, "splits": ["train", "dev"],
                  "split_counts": counts, "source_data_identity": historical["data_identity"]}
    manifest = artifact_dir / alignment.PROJECTION_MANIFEST
    write_json_atomic(manifest, projection)
    monkeypatch.setattr(alignment, "TRAIN_DEV_COUNTS", counts)

    def load_projection(_directory, _source):
        datasets = {}
        for split, tape in entries.items():
            ordered = {}
            for uid, did, _ in tape:
                ordered.setdefault(did, []).append((None, None, uid))
            datasets[split] = SimpleNamespace(
                dialogue_ids=list(ordered), ordered=ordered,
                rows={uid: {"dialogue_id": did, "benchmark_label_id": gold} for uid, did, gold in tape},
            )
        return datasets, projection

    monkeypatch.setattr(alignment, "load_projected_train_dev", load_projection)
    return SimpleNamespace(entries=entries, rows=rows, projection=projection, manifest=manifest,
                           load=lambda: alignment.load_inputs(artifact_dir, frozen, historical, evidence))


def test_load_inputs_accepts_frozen_f0_group_overlap(projected_inputs):
    _, entries, projection, summaries = projected_inputs.load()
    assert entries == projected_inputs.entries
    assert projection == projected_inputs.projection
    assert summaries["dev"]["sessions"] == ["Ses04"]
    assert summaries["dev"]["actors"] == ["Ses04F", "Ses04M"]


@pytest.mark.parametrize(("extra_dev_id", "reason"), [
    ("Ses01F_impro02_F000", "shared sessions mismatch"),
    ("Ses02F_impro02_F000", "Dev sessions mismatch"),
    ("Ses04F_impro01_F003", "dialogue IDs overlap"),
    ("Ses04F_impro01_F000", "utterance IDs overlap"),
])
def test_load_inputs_rejects_unexpected_overlap(projected_inputs, extra_dev_id, reason):
    projected_inputs.entries["dev"].extend(projected_inputs.rows(extra_dev_id))
    with pytest.raises(ValueError, match=reason):
        projected_inputs.load()


def test_load_inputs_rejects_wrong_shared_session(projected_inputs):
    projected_inputs.entries["dev"] = projected_inputs.rows("Ses01F_impro02_F000", "Ses01F_impro02_M001")
    with pytest.raises(ValueError, match=r"shared sessions mismatch.*Ses01"):
        projected_inputs.load()


def test_load_inputs_rejects_missing_shared_actor(projected_inputs):
    projected_inputs.entries["dev"][-1] = projected_inputs.rows("Ses04F_impro02_F001")[0]
    with pytest.raises(ValueError, match="shared actors mismatch"):
        projected_inputs.load()


@pytest.mark.parametrize(("split", "fault", "reason"), [
    ("train", "label", "train labeled population SHA-256 mismatch"),
    ("dev", "label", "Dev projection order/IDs/gold differ"),
    ("train", "order", "train canonical population differs"),
    ("train", "population", "train canonical population differs"),
])
def test_load_inputs_rejects_changed_population(projected_inputs, split, fault, reason):
    tape = projected_inputs.entries[split]
    if fault == "label":
        uid, did, gold = tape[0]
        tape[0] = (uid, did, gold + 1)
    elif fault == "order":
        tape[:] = tape[1:] + tape[:1]
    else:
        del tape[0]
    with pytest.raises(ValueError, match=reason):
        projected_inputs.load()


@pytest.mark.parametrize("field", ["sha256", "row_order_identity_sha256", "source_data_identity"])
def test_load_inputs_rejects_changed_certificate(projected_inputs, field):
    write_json_atomic(projected_inputs.manifest, {**projected_inputs.projection, field: None})
    with pytest.raises(ValueError, match="^projection differs from the authenticated"):
        projected_inputs.load()


def test_load_inputs_rejects_changed_loaded_projection(projected_inputs):
    projected_inputs.projection["sha256"] = {}
    with pytest.raises(ValueError, match="^loaded projection differs from the authenticated"):
        projected_inputs.load()


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

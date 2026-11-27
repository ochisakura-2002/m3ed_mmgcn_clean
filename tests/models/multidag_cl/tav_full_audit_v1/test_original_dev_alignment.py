import csv
import json
from pathlib import Path
import subprocess
import sys
import tarfile
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.analysis.models.multidag_cl import original_dev_analysis as analysis
from scripts.analysis.models.multidag_cl import plot_original_dev_alignment as plotting
from scripts.analysis.models.multidag_cl.plot_original_dev_alignment import render
from scripts.runtime.multidag_cl_fusion import original_dev_alignment as alignment
from scripts.runtime.multidag_cl_fusion.runtime import paired_runtime
from scripts.runtime.multidag_cl_group_isolated.protocol import read_frozen
from scripts.runtime.multidag_cl_ta_v0.formal import sha256_file, write_json_atomic
from scripts.runtime.multidag_cl_tav_full_audit_v1.formal import execute
from scripts.runtime.multidag_cl_tav_full_audit_v1.runtime import SyntheticDialogues
from scripts.workflows.benchmarks import original_dev_stage_b as gates
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


def synthetic(train_size=17):
    datasets, entries = {}, {}
    for session, split in enumerate(("train", "dev"), 1):
        dataset = SyntheticDialogues(split)
        if split == "train":
            dataset.items = [dict(dataset.items[i % len(dataset.items)]) for i in range(train_size)]
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


def test_environment_canonicalization():
    live = {"gpu_compute_capability": (np.int64(7), 0), "nested": {"v": (None, np.bool_(True), np.float32(0.5))},
            "torch_build": "Intel oneAPI MKL 2023.1 Build 20230303"}
    old = json.loads(gates.canonical(live))
    gates.check_environment(old, live)
    assert json.loads(gates.canonical(old)) == old
    for actual in ({**old, "torch_build": "Intel oneAPI MKL 2024.2.0 Build 20240605"},
                   {**old, "gpu_compute_capability": [8, 0]},
                   {**old, "nested": {"v": [None, 1, 0.5]}},
                   {**old, "nested": {}}, {**old, "new_field": None}):
        with pytest.raises(ValueError, match="environment differs"):
            gates.check_environment(old, actual)
    with pytest.raises(ValueError, match="Out of range"):
        gates.canonical({"threads": float("nan")})


def test_independent_initialization_and_plot_worker_without_display(monkeypatch):
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    monkeypatch.setenv("MPLBACKEND", "TkAgg")
    command = [sys.executable, "-B", "-m", workflow.__name__, "environment-probe", "--probe-device", "cpu"]

    def probe(*args):
        return json.loads(subprocess.check_output([*command, *args], cwd=workflow.ROOT, encoding="utf-8"))

    first, second = probe(), probe()
    gates.check_environment(first["environment"], second["environment"])
    assert first["diagnostics"] == second["diagnostics"]
    assert "matplotlib" not in first["diagnostics"]["import_order"]
    assert "matplotlib" in probe("--probe-plot-first")["diagnostics"]["import_order"]
    worker = workflow.plot_worker(None)
    assert worker["backend"].lower() == "agg"
    family = plotting.select_font(None)
    if family:
        font = Path(plotting.font_manager.findfont(family))
        assert workflow.plot_worker(font)["language"] == "zh"
    fonts = plotting.font_manager.fontManager.ttflist
    monkeypatch.setattr(plotting.font_manager.fontManager, "ttflist", [f for f in fonts if f.name == "DejaVu Sans"])
    assert plotting.plot_probe(None)["language"] == "en"


def reviewed_sources(current):
    policy = alignment.read_json(workflow.ROOT / gates.SOURCE_POLICY)
    historical = dict(current)
    for path, record in policy["files"].items():
        assert current[path] == record["new_sha256"]
        if record["old_sha256"] is None:
            del historical[path]
        else:
            historical[path] = record["old_sha256"]
    return historical


def test_source_policy_exact_signed_scope():
    current = workflow.provenance("cpu")["files_sha256"]
    historical = reviewed_sources(current)
    certificate = gates.certify_sources(workflow.ROOT, "stage_a_locked", historical, current)
    assert certificate["stage_a_sha256"] == "stage_a_locked"
    assert set(certificate["changes"]) == set(gates.SOURCE_ROLES)
    core = "scripts/runtime/multidag_cl_ta_v0/training.py"
    for changed in ({**current, core: "0" * 64}, {k: v for k, v in current.items() if k != core},
                    {**current, next(iter(gates.SOURCE_ROLES)): "0" * 64}):
        with pytest.raises(ValueError, match="unapproved Stage A source difference"):
            gates.certify_sources(workflow.ROOT, "stage_a_locked", historical, changed)


def test_stage_b_preflight_recovery_with_real_cpu_fit_and_sealed_bundle(artifact_dir, monkeypatch):
    frozen, raw = read_frozen(alignment.ROOT)
    config = alignment.read_config(frozen)
    datasets, entries = synthetic(train_size=97)
    git = {"git_head": "synthetic-stage-a"}
    projection = {"sha256": {}}
    paths = SimpleNamespace(manifest_root=artifact_dir / "manifests" / "synthetic90",
                            runs_root=artifact_dir / "runs", analysis_root=artifact_dir / "analysis" / "synthetic90")
    paths.manifest_root.mkdir(parents=True)
    lock = {**workflow.HOLD, "status": "LOCKED", "candidate_updates": config["budget_candidate_updates"],
            "selected_updates": 90, "scores": [], "means": [{"updates": 90, "hard_nll": 1.0}]}
    lock_file = paths.manifest_root / "budget_lock.json"
    write_json_atomic(lock_file, lock)
    original, _ = paired_runtime(raw, datasets, device="cpu", synthetic_only=True, git_head=git["git_head"])
    reference = execute(original, raw, artifact_dir / "reference", git=git, synthetic_epochs=30)
    d0, d1 = (workflow.arm_run(paths, arm) for arm in ("D0", "D1"))
    baseline = workflow.make_runtime(raw, datasets, d0, config, "D0", lock, reference, git, expected=reference,
                                     synthetic=True)
    workflow.fit(baseline, raw, git, projection, synthetic_epochs=30)
    d0_record = workflow.checked_arm(baseline, synthetic_epochs=30)
    assert d0_record["snapshots"]["u_star"] | {"path": None, "sha256": None} == {
        "epoch": 13, "optimizer_steps": 90, "epoch_complete": False, "path": None, "sha256": None,
    }
    stage_a = paths.manifest_root / "stage_a.json"
    status = paths.manifest_root / "stage_b.json"
    provenance = workflow.provenance("cpu")
    legacy = {"files_sha256": reviewed_sources(provenance["files_sha256"]), "environment": provenance["environment"]}
    write_json_atomic(stage_a, {**workflow.HOLD, "status": "PASS", "run_id": d0.name, "d3_status": lock["status"],
        "config_sha256": workflow.json_identity(config), "budget_sha256": sha256_file(lock_file),
        "numerical_audit": {"numerical_identity": True}, "projection": projection, "runtime": legacy,
        "run_evidence_sha256": workflow.training_evidence(d0)})
    args = SimpleNamespace(approved_stage_a_sha256=sha256_file(stage_a), figure_font=None, resume_analysis=False,
                           preflight_only=True)
    monkeypatch.setattr(workflow, "prepare_stage", lambda *args: (reference, {}, [], datasets, entries, projection, {}))
    monkeypatch.setattr(workflow, "git_identity", lambda: git)
    before_assets = {p: sha256_file(p) for p in (stage_a, lock_file, d0 / "checkpoints/best_model.pt")}

    def invoke():
        with gates.batch_lock(paths.manifest_root / ".stage_b.lock", readonly=args.preflight_only):
            workflow.execute_stage_b(args, paths, frozen, raw, config, synthetic_epochs=30)

    def drift(_device):
        return {**provenance, "environment": {**provenance["environment"], "torch_build": "other MKL"}}

    with monkeypatch.context() as patch:
        patch.setattr(workflow, "provenance", drift)
        for preflight_only in (True, False):
            args.preflight_only = preflight_only
            with pytest.raises(ValueError, match="other MKL"):
                invoke()
            assert not d1.exists() and not status.exists() and not paths.analysis_root.exists()
    paths.analysis_root.mkdir(parents=True)
    write_json_atomic(status, {**workflow.HOLD, "protocol": workflow.PROTOCOL, "d3_status": lock["status"],
                               "stage_a_sha256": args.approved_stage_a_sha256, "status": "FAIL",
                               "error": "previous pre-fit failure"})
    old_failure = status.read_bytes()
    args.preflight_only = True
    invoke()
    assert status.read_bytes() == old_failure and not list(paths.analysis_root.iterdir())
    args.preflight_only = False
    plot_worker = workflow.plot_worker

    def failed_render(font, tables=None, destination=None):
        if tables is not None:
            raise RuntimeError("injected renderer failure")
        return plot_worker(font)

    with monkeypatch.context() as patch:
        patch.setattr(workflow, "plot_worker", failed_render)
        with pytest.raises(RuntimeError, match="injected renderer failure"):
            invoke()
    assert not paths.analysis_root.exists()
    result = alignment.read_json(status)
    assert result["phase"] == "trained" and result["status"] == "FAIL"
    assert (paths.manifest_root / result["archived_attempt"] / "stage_b.json").read_bytes() == old_failure
    asset = d1 / "checkpoints/alignment/u_star.pt"
    d1_assets = {p: sha256_file(p) for p in (asset, d1 / "checkpoints/best_model.pt")}
    with monkeypatch.context() as patch:
        patch.setattr(workflow, "fit", lambda *args, **kwargs: pytest.fail("existing D1 must not be overwritten"))
        args.resume_analysis = True
        with monkeypatch.context() as interrupted:
            real_publish = workflow.publish

            def stop_during_publish(state, destination, archive, status_file):
                directory, _ = workflow.checked_publication(state, destination, archive)
                directory.rename(destination)
                raise RuntimeError("injected interruption after moving analysis")

            interrupted.setattr(workflow, "publish", stop_during_publish)
            with pytest.raises(RuntimeError, match="interruption after moving analysis"):
                invoke()
        assert alignment.read_json(status)["phase"] == "publishing"
        assert workflow.publish is real_publish
        invoke()
        invoke()
    final = alignment.read_json(status)
    assert final["status"] == "PASS" and final["phase"] == "published" and final["endpoint_count"] == 8
    assert final["d1_evidence_sha256"] == workflow.training_evidence(d1)
    assert {p: sha256_file(p) for p in before_assets | d1_assets} == before_assets | d1_assets
    package = Path(final["archive"])
    assert sha256_file(package) == final["archive_sha256"]
    with tarfile.open(package) as archive:
        members = set(archive.getnames())
        assert {"RESULTS.md", "SHA256SUMS.txt", "figures/FIGURE_INDEX.md", "stage_a_source_compatibility.json"} <= members
        assert not any(name.endswith((".pt", ".pkl")) for name in members)
        assert len([name for name in members if name.endswith(".png")]) == 13
        assert len([name for name in members if name.endswith(".svg")]) == 13
        assert archive.extractfile("figures/overview.png").read(8) == b"\x89PNG\r\n\x1a\n"
        assert b"<svg" in archive.extractfile("figures/overview.svg").read(1000)
    checksums = (paths.analysis_root / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines()
    for line in checksums:
        digest, name = line.split("  ", 1)
        assert sha256_file(paths.analysis_root / name) == digest
    with (paths.analysis_root / "tables/overall_metrics.csv").open(encoding="utf-8", newline="") as stream:
        assert {row["arm"] for row in csv.DictReader(stream)} == set(analysis.ENDPOINTS)
    evidence = alignment.read_json(paths.analysis_root / "tables/analysis_protocol.json")
    for tag in ("CE", "LS"):
        assert evidence["sources"][f"{tag} U*"]["sources"][0]["optimizer_steps"] == 90
        assert [item["epoch"] for item in evidence["sources"][f"{tag} AVG"]["sources"]] == [21, 24, 27, 30]
    for arm in ("D0", "D1"):
        assert alignment.read_json(workflow.arm_run(paths, arm) / "manifests/run_manifest.json")["optimizer_steps"] == 210


def test_unknown_or_incomplete_d1_blocks_recovery(artifact_dir):
    receipt = artifact_dir / "stage_b.json"
    destination, archive, d1 = (artifact_dir / name for name in ("analysis", "report.tar.gz", "d1"))
    write_json_atomic(receipt, {"status": "FAIL", "stage_a_sha256": "reviewed"})
    d1.mkdir()
    checkpoint = d1 / "partial.pt"
    checkpoint.write_bytes(b"incomplete")
    with pytest.raises(ValueError, match="training assets exist"):
        gates.inspect_attempt(receipt, destination, archive, d1, "reviewed")
    assert checkpoint.read_bytes() == b"incomplete"
    write_json_atomic(receipt, {"status": "FAIL", "phase": "fit_started", "stage_a_sha256": "reviewed"})
    (d1 / ".training.lock").touch()
    with pytest.raises(ValueError, match="training lock is present"):
        gates.inspect_attempt(receipt, destination, archive, d1, "reviewed")


def test_undefined_curves_and_empty_groups_export_with_headers(artifact_dir, monkeypatch):
    did = "Ses04F_impro01"
    uid = f"{did}_F000"
    expected = [(uid, did, 0)]
    predictions = [{"utterance_id": uid, "dialogue_id": did, "true_label": 0, "predicted_label": 0,
                    "logits": [1.0, 0, 0, 0, 0, 0]}]
    fields = {"accuracy": 1, "weighted_f1": 1, "macro_f1": 1 / 6, "uar": 1 / 6,
              "loss": 1, "valid_target_count": 1}
    endpoints = dict.fromkeys(analysis.ENDPOINTS[:-2], predictions)
    sources = dict.fromkeys(endpoints, {"formal_metrics": fields})
    learning = [{"arm": arm, "epoch": 1, "population": role, "weighted_f1": 1, "hard_label_nll": 1}
                for arm in ("CE", "LS") for role in ("Fit", "Dev")]
    budget = {"status": "UNAVAILABLE"}
    monkeypatch.setattr(analysis, "ENDPOINT", 1)
    analysis.stage_tables(artifact_dir, endpoints, sources, learning, expected, budget)
    assert (artifact_dir / "roc_pr.csv").read_text().splitlines() == ["arm,curve,fpr,tpr,precision,recall,threshold"]
    assert (artifact_dir / "class_roc_pr.csv").read_text().splitlines() == ["arm,label,curve,x,y"]
    with (artifact_dir / "shift.csv").open(newline="") as stream:
        shift = list(csv.DictReader(stream))
    assert all(row["valid_target_count"] == "0" for row in shift if row["group"] in ("SHIFT", "STABLE"))
    with pytest.raises(ValueError, match="unique utterances"):
        analysis.stage_tables(artifact_dir, endpoints, sources, learning, expected * 2, budget)
    with pytest.raises(ValueError, match="trajectory rows"):
        analysis.stage_tables(artifact_dir, endpoints, sources, learning[:-1], expected, budget)
    predictions[0]["logits"][0] = float("nan")
    with pytest.raises(ValueError, match="nonfinite frozen logits"):
        analysis.stage_tables(artifact_dir, endpoints, sources, learning, expected, budget)

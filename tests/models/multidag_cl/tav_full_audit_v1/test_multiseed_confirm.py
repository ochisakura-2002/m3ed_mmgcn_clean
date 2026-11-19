import csv
import json
import os
from pathlib import Path
import tarfile
from types import SimpleNamespace

import pytest
import torch
import yaml

from scripts.analysis.models.multidag_cl import collect_graphcore_stage_evidence_v1 as evidence
from scripts.analysis.models.multidag_cl import summarize_tav_multiseed_confirm_v1 as summary
from scripts.runtime.multidag_cl_paper_reimplementation.checkpoint import load_checkpoint
from scripts.runtime.multidag_cl_ta_v0.formal import json_identity, sha256_file
from scripts.runtime.multidag_cl_tav_full_audit_v1 import confirm as validation
from scripts.runtime.multidag_cl_tav_full_audit_v1.formal import execute
from scripts.runtime.multidag_cl_tav_full_audit_v1.runtime import synthetic_runtime
from scripts.runtime.multidag_cl_tav_full_audit_v1.spec import confirm_config, make_config
from scripts.workflows.benchmarks import run_tav_multiseed_confirm_v1 as runner
from scripts.workflows.benchmarks.run_tav_full_audit_v1 import plan as full_audit_plan

DATE = "20261119"
HOLD = {"test_loader_built": 0, "test_evaluation_count": 0, "test_used_for_selection": False}
METRICS = dict(accuracy=0.5, weighted_f1=0.4, macro_f1=0.3, uar=0.2, loss=1.0, valid_target_count=980)


def write_json(path, record):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record), encoding="utf-8")


def run_fixture(directory, raw, data, *, ta_lineage=None):
    # Small written artifacts exercise orchestration only. No real data or checkpoint is opened.
    for name in ("checkpoints", "logs"):
        (directory / name).mkdir(parents=True)
    checkpoint = directory / "checkpoints/best_model.pt"
    checkpoint.write_bytes(b"test fixture checkpoint placeholder")
    manifest = {"status": "completed", "training_finished": True, "best_checkpoint_reloaded": True,
                "last_completed_epoch": 30, "synthetic_only": False, "data_identity": data,
                "best_epoch": 2, "best_dev_metrics": METRICS, "best_checkpoint_sha256": sha256_file(checkpoint),
                "seed": 100, "parameter_count": 99, "git_head": validation.AUDIT_REFERENCE, "git_dirty": False, **HOLD}
    if ta_lineage:
        manifest.update(model_lineage=ta_lineage, model_parameter_count=99, run_id=directory.name)
        metrics = {**METRICS, "checkpoint_identity": manifest["best_checkpoint_sha256"]}
        resolved = raw
    else:
        manifest.update(run_id=raw["experiment_id"], experiment_id=raw["experiment_id"], schema=raw["schema"],
                        model_name=raw["model_name"], architecture_lineage=raw["architecture_lineage"],
                        architecture_identity=validation.architecture_sha(raw), config_sha256=json_identity(raw),
                        seed=raw["training"]["seed"], active_modalities=raw["active_modalities"], graph_core=raw["graph_core"],
                        context=raw["context_mode"], curriculum=raw["curriculum"], training_config=raw["training"])
        metrics = METRICS
        resolved = {"audit_config": raw, "data_identity": data}
    (directory / "resolved_config.yaml").write_text(yaml.safe_dump(resolved), encoding="utf-8")
    write_json(directory / "manifests/run_manifest.json", manifest)
    write_json(directory / "reports/validation_best_metrics.json", metrics)
    (directory / "reports/test_metrics.json").write_text("not read or packaged", encoding="utf-8")
    (directory / "logs/epoch_metrics.tsv").write_text(
        "epoch\tdev_weighted_f1\tdev_loss\tdev_accuracy\tdev_macro_f1\tdev_uar\n"
        + "".join(f"{epoch}\t0.4\t{1.0 if epoch >= 2 else 1.1}\t0.5\t0.3\t0.2\n" for epoch in range(1, 31)),
        encoding="utf-8")
    return manifest


def test_plan_and_paired_seed_reaches_model_shuffle_checkpoint_and_export(artifact_dir):
    date, jobs, configs, sources = runner.plan(output_root=artifact_dir / "unused", experiment_date=DATE)
    assert date == DATE and len(jobs) == 40 and len(sources) == 8
    assert len({job["comparison_id"] for job in jobs}) == 5
    assert sum(bool(job["source_seed100_run"]) for job in jobs) == 8
    assert not (artifact_dir / "unused").exists()
    raw, paired = (configs[row["experiment_id"]] for row in (jobs[0], jobs[2]))
    a, b = synthetic_runtime(raw), synthetic_runtime(paired)
    assert raw["active_modalities"] == "TAV" and paired["active_modalities"] == "TA"
    assert a.loader("train", epoch=1).generator.initial_seed() == b.loader("train", epoch=1).generator.initial_seed() == 43
    assert all(torch.equal(left, right) for left, right in zip(a.model.parameters(), b.model.parameters()))
    different_seed = synthetic_runtime(configs[jobs[8]["experiment_id"]])
    assert different_seed.loader("train", epoch=1).generator.initial_seed() == 101
    assert any(not torch.equal(left, right) for left, right in zip(a.model.parameters(), different_seed.model.parameters()))
    with pytest.raises(ValueError, match="outside"):
        confirm_config(make_config("coop"), 42)
    run = artifact_dir / "fit"
    manifest = execute(a, raw, run, synthetic_epochs=1, export_dev_evidence=True)
    checkpoint = load_checkpoint(run / "checkpoints/best_model.pt", torch.device("cpu"))
    assert checkpoint["seed"] == checkpoint["training_config"]["seed"] == manifest["seed"] == 42
    dev = json.loads((run / "reports/validation_best_dev_evidence.json").read_text(encoding="utf-8"))
    assert dev["split"] == "dev" and dev["optimizer_steps_performed"] == dev["test_evaluation_count"] == 0
    assert len(dev["predictions"]) == dev["metrics"]["valid_target_count"]
    with pytest.raises(ValueError, match="seed identity mismatch|training_config identity mismatch|audit_config identity mismatch"):
        different_seed.validate_checkpoint(checkpoint)


def test_full_seed_validation_rejects_partial_and_invalid_history(artifact_dir, monkeypatch):
    _, jobs, configs, sources = runner.plan(output_root=artifact_dir, experiment_date=DATE)
    job = jobs[8]
    raw = sources[job["config_path"]]
    directory = artifact_dir / "historical"
    data = {"sha256": {"asset": "fixture"}}
    manifest = run_fixture(directory, raw, data)
    monkeypatch.setattr(validation, "compatible_historical_source", lambda _: True)
    kwargs = (jobs, configs, sources, data, {job["source_seed100_run"]: directory})
    assert runner.resolve_jobs(*kwargs)[8]["reuse_status"] == "reused_seed100"
    different_data = {"sha256": {"asset": "other fixture"}}
    assert runner.resolve_jobs(jobs, configs, sources, different_data, kwargs[-1])[8]["reuse_status"] == "rerun_seed100"
    (directory / "checkpoints/best_model.pt").write_bytes(b"tampered")
    assert runner.resolve_jobs(*kwargs)[8]["reuse_status"] == "rerun_seed100"
    path = directory / "manifests/run_manifest.json"
    write_json(path, dict(manifest, last_completed_epoch=29))
    with pytest.raises(ValueError, match="incomplete"):
        runner.resolve_jobs(*kwargs)
    write_json(path, dict(manifest, test_evaluation_count=1))
    with pytest.raises(ValueError, match="Test HOLD"):
        runner.resolve_jobs(*kwargs)
    (Path(jobs[-1]["planned_run_dir"]) / "manifests").mkdir(parents=True)
    write_json(Path(jobs[-1]["planned_run_dir"]) / "manifests/run_manifest.json", dict(manifest, status="failed"))
    with pytest.raises(ValueError, match="incomplete"):
        runner.resolve_jobs(jobs, configs, sources, data, {})


def test_tables_join_by_seed_and_compute_paired_student_t():
    seedwise = [{"configuration": row["configuration"], "seed": int(row["seed"]), "best_epoch": 4,
                 **dict.fromkeys(summary.SCORES, 0.5)} for row in validation.matrix_design()]
    for row, delta in zip((row for row in seedwise if row["configuration"] == "track_tav_cloff"), (-2, -1, 0, 1, 2)):
        row["Weighted-F1"] += delta / 100
    tables = summary.tables(seedwise[::-1])
    assert [len(rows) for rows in tables.values()] == [40, 40, 32]
    row = tables[summary.TABLE_NAMES[2]][1]
    assert [row[f"delta_s{seed}"] for seed in validation.CONFIRM_SEEDS] == pytest.approx([-2, -1, 0, 1, 2])
    assert (row["wins"], row["ties"], row["losses"]) == (2, 1, 2)
    assert row["SD"] == pytest.approx(2.5 ** 0.5)
    assert row["ci95_high"] == pytest.approx(1.96324316147756)
    assert row["ci95_low"] == pytest.approx(-row["ci95_high"])
    with pytest.raises(ValueError, match="matched"):
        summary.tables(seedwise[:-1])


def fixture_provenance(directory):
    model, prov, bridge = (directory / name for name in ("model.csv", "provenance.json", "bridge.audit.json"))
    model.write_text("synthetic model manifest fixture\n", encoding="utf-8")
    keys = ("canonical_asset", "historical_asset", "model_manifest", "order_manifest", "historical_split_manifest", "fingerprint_bridge")
    sources = {key: {"sha256": str(index) * 64} for index, key in enumerate(keys, 1)}
    ta_sha = sources["canonical_asset"]["sha256"]
    write_json(bridge, {"schema": "audio_fingerprint_bridge_v1", "status": "complete", "bridge_rows": 7433,
                        "bridge_sha256": sources["fingerprint_bridge"]["sha256"], "sources": sources, **HOLD})
    data = {"schema": "iemocap_asym_tav_t1024_v1", "row_order_identity_sha256": "7" * 64,
            "sha256": {"asset": "8" * 64, "model_manifest": sha256_file(model)}}
    write_json(prov, {"schema": data["schema"], "status": "complete", "asset_sha256": data["sha256"]["asset"],
                      "model_manifest_sha256": data["sha256"]["model_manifest"],
                      "row_order_identity_sha256": data["row_order_identity_sha256"], "sources": sources, **HOLD})
    data["sha256"]["provenance_manifest"] = sha256_file(prov)
    return model, prov, bridge, ta_sha, data


def fixture_audit(output, data):
    _, training, evaluations = full_audit_plan(output_root=output, experiment_date=DATE)
    manifest_path = output / DATE / "tav_full_audit_v1/manifests/batches/a/batch_manifest.json"
    manifests = {}
    for row in training:
        raw = make_config(row["graph_core"], row["modalities"], row["curriculum"], row["context_mode"])
        manifests[row["experiment_id"]] = run_fixture(Path(row["run_dir"]), raw, data)
    for row in evaluations:
        source_id = row["source_experiment_id"]
        source = next(job for job in training if job["experiment_id"] == source_id)
        path = output / DATE / "tav_full_audit_v1/analysis/batches/a/missing_modality" / (row["evaluation_id"] + ".json")
        write_json(path, {**row, "status": "completed", "metrics": METRICS, "best_epoch": 2, "parameter_count": 99,
                          "checkpoint_sha256": manifests[source_id]["best_checkpoint_sha256"], "data_identity": data,
                          "optimizer_steps_performed": 0, "source_run_dir": source["run_dir"], **HOLD})
    write_json(manifest_path, {"schema": "tav_full_audit_batch_v1", "status": "completed", "batch_id": "a",
                               "experiment_date": DATE, "training_jobs": training, "evaluation_jobs": evaluations,
                               "matrix_identity": json_identity({"training": training, "evaluation": evaluations}),
                               "completed": list(manifests), "skipped": [],
                               "evaluated": [row["evaluation_id"] for row in evaluations], "data_identity": data, **HOLD})
    return manifest_path


def test_archive_roundtrip_full_fixture_and_deny_large_artifacts(artifact_dir, monkeypatch):
    model, prov, bridge, ta_sha, data = fixture_provenance(artifact_dir)
    output = artifact_dir / "out"
    audit_path = fixture_audit(output, data)
    observed_seeds = []
    monkeypatch.setattr(runner, "os", SimpleNamespace(name="posix", path=os.path))
    monkeypatch.setattr(runner.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(runner, "load_train_dev", lambda *_: ({}, data, {}))
    monkeypatch.setattr(runner, "seed_everything", lambda _, seed: observed_seeds.append(seed))
    monkeypatch.setattr(runner, "TAVFullAudit", lambda raw: raw)
    monkeypatch.setattr(runner, "AuditRuntime", lambda *args, **kwargs: None)
    monkeypatch.setattr(runner, "execute", lambda _, raw, path, **kwargs: run_fixture(Path(path), raw, data))
    batch_options = {"asset_path": None, "model_manifest_path": model, "provenance_manifest_path": prov, "batch_id": "b",
                     "full_audit_batch_manifest": audit_path, "output_root": output, "experiment_date": DATE}
    state = runner.run(**batch_options)
    assert state["reused_seed100"] == 8 and state["new_runs"] == 32
    assert observed_seeds == [seed for seed in validation.CONFIRM_SEEDS if seed != 100 for _ in range(8)]
    with pytest.raises(FileExistsError, match="already reserved"):
        runner.run(**batch_options)
    replay = runner.run(**{**batch_options, "batch_id": "replay"})
    assert replay["new_runs"] == 0 and replay["reused_seed100"] == 8
    assert len(observed_seeds) == 32
    batch = output / DATE / validation.CONFIRM_GROUP / "manifests/batches/b/batch_manifest.json"
    ta = {}
    for key, lineage in evidence.TA_RUNS.items():
        path = artifact_dir / key
        raw = {"model": {"text_dim": 768 if key == "t768" else 1024}, "training": {"curriculum": False},
               "checkpoint": {"test_split_used_for_selection": False}}
        run_fixture(path, raw, {"sha256": {"asset": ta_sha}}, ta_lineage=lineage)
        ta[key] = path.as_posix()
    inputs = artifact_dir / "historical.json"
    write_json(inputs, {"runs": ta})
    options = {"confirm_batch_manifest": batch, "full_audit_batch_manifest": audit_path, "historical_inputs": inputs,
               "model_manifest_path": model, "provenance_manifest_path": prov, "bridge_audit_path": bridge,
               "output_root": output}
    result = evidence.collect(**options)
    assert (result["COMPLETE_IDENTITIES"], result["REUSED_SEED100"], result["NEW_RUNS"]) == (40, 8, 32)
    assert result["EVIDENCE_PACKAGE_SHA256"] == sha256_file(result["EVIDENCE_PACKAGE"])
    with tarfile.open(result["EVIDENCE_PACKAGE"], "r:gz") as archive:
        members = archive.getnames()
        assert len(members) == result["EVIDENCE_FILE_COUNT"]
        assert all(not name.endswith((".pt", ".pkl")) and "test_metrics.json" not in name for name in members)
        assert sum("/03_tav_full_audit/missing_modality/" in name for name in members) == 56
        seedwise = next(name for name in members if name.endswith(summary.TABLE_NAMES[0]))
        assert len(list(csv.DictReader(archive.extractfile(seedwise).read().decode().splitlines()))) == 40
        filelist = next(name for name in members if name.endswith("/FILELIST.txt"))
        assert archive.extractfile(filelist).read().decode().splitlines() == [name.split("/", 1)[1] for name in members]
        archive.extractall(artifact_dir / "unpacked")
    unpacked = artifact_dir / "unpacked" / members[0].split("/", 1)[0]
    checksums = (unpacked / evidence.INVENTORY / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines()
    for line in checksums:
        digest, relative = line.split("  ", 1)
        assert sha256_file(unpacked / relative) == digest
    with pytest.raises(FileExistsError):
        evidence.collect(**options)
    payload = artifact_dir / "representation.pkl"
    payload.write_bytes(b"fixture")
    with pytest.raises(ValueError, match="lightweight"):
        evidence.Evidence(artifact_dir / "denied").copy(payload, "denied.pkl", {}, "raw")

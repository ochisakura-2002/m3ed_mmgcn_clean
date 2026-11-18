import json
from pathlib import Path

import pytest

from scripts.analysis.models.multidag_cl.summarize_tav_full_audit_v1 import collect, summarize
from scripts.runtime.multidag_cl_ta_v0.formal import sha256_file
from scripts.runtime.multidag_cl_tav_full_audit_v1.formal import completed_manifest
from scripts.runtime.multidag_cl_tav_full_audit_v1.spec import design, evaluation_design, make_config
from scripts.workflows.benchmarks.run_tav_full_audit_v1 import plan


def fixture_batch(artifact_dir):
    output_root = artifact_dir / "outputs"
    date, jobs, evaluations = plan(output_root=output_root, experiment_date="20261118")
    group = output_root / date / "tav_full_audit_v1"
    directory = group / "manifests/batches/synthetic_fixture"
    directory.mkdir(parents=True)
    state = {"batch_id": "synthetic_fixture", "experiment_date": date, "training_jobs": jobs,
             "evaluation_jobs": evaluations, "data_identity": {"kind": "synthetic_fixture"},
             "test_loader_built": 0, "test_evaluation_count": 0}
    path = directory / "batch_manifest.json"
    path.write_text(json.dumps(state), encoding="utf-8")
    return path, state, group


def fake_completed_job(state, index=0):
    job = state["training_jobs"][index]
    directory = Path(job["run_dir"])
    for name in ("manifests", "reports", "checkpoints"):
        (directory / name).mkdir(parents=True)
    (directory / "checkpoints/best_model.pt").write_bytes(b"synthetic fixture, no real checkpoint")
    (directory / "reports/test_metrics.json").write_text("FORBIDDEN invalid JSON")
    metrics = {"accuracy": .5, "weighted_f1": .4, "macro_f1": .3, "uar": .2, "loss": 1.7}
    manifest = {"status": "completed", "training_finished": True, "last_completed_epoch": 30,
                "synthetic_only": False, "config_sha256": job["config_sha256"], "data_identity": state["data_identity"],
                "parameter_count": 123, "best_epoch": 3, "test_loader_built": 0, "test_evaluation_count": 0,
                "best_checkpoint_sha256": sha256_file(directory / "checkpoints/best_model.pt"), "best_dev_metrics": metrics}
    (directory / "manifests/run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (directory / "reports/validation_best_metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    return job, directory, manifest


def test_six_tables_keep_pending_rows_and_read_only_dev_artifacts(artifact_dir):
    path, state, group = fixture_batch(artifact_dir)
    job, run_dir, manifest = fake_completed_job(state)
    row = state["evaluation_jobs"][0]
    evaluation_dir = group / "analysis/batches/synthetic_fixture/missing_modality"
    evaluation_dir.mkdir(parents=True)
    (evaluation_dir / (row["evaluation_id"] + ".json")).write_text(json.dumps({
        **row, "status": "completed", "metrics": manifest["best_dev_metrics"], "best_epoch": 3,
        "parameter_count": 123, "checkpoint_sha256": manifest["best_checkpoint_sha256"],
        "data_identity": state["data_identity"], "optimizer_steps_performed": 0,
        "test_loader_built": 0, "test_evaluation_count": 0}), encoding="utf-8")
    result = summarize(path, output_root=artifact_dir / "summary_outputs")
    assert list(result["rows"].values()) == [24, 8, 20, 14, 5, 56]
    _, training, missing = collect(path)
    assert training[0]["Weighted-F1"] == .4 and missing[0]["Accuracy"] == .5
    assert training[1]["status"] == "pending" and training[1]["Weighted-F1"] == ""
    assert result["test_evaluation_count"] == 0
    with pytest.raises(FileExistsError):
        summarize(path, output_root=artifact_dir / "summary_outputs")


def test_completed_skip_requires_identity_30_epochs_and_checkpoint_integrity(artifact_dir):
    _, state, _ = fixture_batch(artifact_dir)
    job, directory, manifest = fake_completed_job(state)
    raw = make_config()
    assert completed_manifest(directory, raw, state["data_identity"])["status"] == "completed"
    with pytest.raises(ValueError, match="cannot skip"):
        completed_manifest(directory, raw, {"kind": "other dataset"})
    for key, value in (("status", "failed"), ("last_completed_epoch", 29), ("test_evaluation_count", 1)):
        altered = dict(manifest, **{key: value})
        (directory / "manifests/run_manifest.json").write_text(json.dumps(altered))
        with pytest.raises(ValueError, match="cannot skip"):
            completed_manifest(directory, raw, state["data_identity"])
    (directory / "manifests/run_manifest.json").write_text(json.dumps(manifest))
    (directory / "checkpoints/best_model.pt").write_bytes(b"corrupt fixture")
    with pytest.raises(ValueError, match="cannot skip"):
        completed_manifest(directory, raw, state["data_identity"])


def test_summary_refuses_test_counter_and_matrix_drift(artifact_dir):
    path, state, _ = fixture_batch(artifact_dir)
    state["test_evaluation_count"] = 1
    path.write_text(json.dumps(state))
    with pytest.raises(ValueError, match="Test isolation"):
        collect(path)
    state["test_evaluation_count"] = 0
    state["training_jobs"] = state["training_jobs"][::-1]
    path.write_text(json.dumps(state))
    with pytest.raises(ValueError, match="matrix identity"):
        collect(path)

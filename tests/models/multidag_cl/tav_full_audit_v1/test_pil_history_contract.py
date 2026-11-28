"""Exercise the D Stage B schema with synthetic values and independent receipts."""

import csv
from io import BytesIO, StringIO
import tarfile
from types import SimpleNamespace

import pytest
import yaml

from models.multidag_cl.ta_v0 import LABEL_VOCAB
from scripts.runtime.multidag_cl_fusion import original_dev_alignment as alignment
from scripts.runtime.multidag_cl_group_isolated.evaluation import summarize
from scripts.runtime.multidag_cl_group_isolated.protocol import HOLD, read_frozen
from scripts.runtime.multidag_cl_ta_v0.formal import json_identity, sha256_file, write_json_atomic
from scripts.runtime.multidag_cl_ta_v0.training import classification_report
from scripts.workflows.benchmarks import run_pil_screen as pil


# Ordered headers from the reviewed D Stage B archive, independently of PIL's reader.
TRAJECTORY = (
    "arm", "epoch", "uar", "macro_f1", "weighted_f1", "accuracy", "valid_target_count", "hard_label_nll",
    "brier", "ece10", "population", "mean_confidence", "mean_confidence_correct", "mean_confidence_wrong",
    "high_confidence_coverage", "high_confidence_error_share", "formal_hard_ce", "train_update_loss",
    "max_preclip_gradient_norm", "optimizer_steps",
)
PREDICTIONS = (
    "arm", "utterance_id", "dialogue_id", "target_index", "speaker", "true_label", "shift_group",
    "previous_same_speaker_label", "session", "actor", "true_label_id", "predicted_label_id", "predicted_label",
    "is_error", "confidence", "nll", "brier",
    *(f"{prefix}_{label}" for prefix in ("logit", "prob") for label in ("hap", "sad", "neu", "ang", "exc", "fru")),
)
EPOCH_TSV = (
    "epoch", "train_loss", "dev_loss", "dev_accuracy", "dev_weighted_f1", "dev_macro_f1", "dev_uar",
    "gradient_norm", "train_dialogues", "train_targets", "is_best", "optimizer_steps", "dev_acc", "dev_wf1",
)


def table_bytes(fields, rows, *, delimiter=","):
    buffer = StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields, delimiter=delimiter, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")


def write_archive(fixture, *, trajectory_fields=TRAJECTORY, duplicate_member=False):
    tables = {
        "tables/trajectory.csv": table_bytes(trajectory_fields, fixture.trajectory),
        "tables/predictions.csv": table_bytes(PREDICTIONS, fixture.predictions),
        "d0/logs/epoch_metrics.tsv": (fixture.run / "logs/epoch_metrics.tsv").read_bytes(),
        "stage_a.json": fixture.stage_path.read_bytes(),
    }
    fixture.archive.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fixture.archive, "w:gz") as bundle:
        names = [*tables, "tables/trajectory.csv"] if duplicate_member else tables
        for name in names:
            member = tarfile.TarInfo(name)
            member.size = len(tables[name])
            bundle.addfile(member, BytesIO(tables[name]))


@pytest.fixture
def history(artifact_dir, monkeypatch):
    monkeypatch.setattr(pil, "D0_ROOT", artifact_dir / "outputs/20261127/original_dev_alignment")
    monkeypatch.setattr(pil, "D0_HEAD", "f" * 40)
    monkeypatch.setattr(pil, "F0_RUN", artifact_dir / "f0_reference")
    paths = pil.history_paths()
    run = pil.d.arm_run(paths, "D0")
    archive = paths.analysis_root.with_name(f"{pil.D0_BATCH}_stage_b.tar.gz")
    stage_path = paths.manifest_root / "stage_a.json"
    lock_path = paths.manifest_root / "budget_lock.json"
    projection_dir = artifact_dir / "projection"
    frozen, raw = read_frozen(pil.ROOT)
    config = pil.d.read_config(frozen)
    dev = [(f"{dialogue}_{'FM'[i % 2]}{i:03}", dialogue, (j + i) % 6)
           for j in range(20) for dialogue in (f"Ses04F_impro{j:02}",) for i in range(49)]
    truth = [gold for _, _, gold in dev]

    def predictions(correct):
        return [{"utterance_id": uid, "dialogue_id": did, "true_label": gold, "predicted_label": label,
                 "logits": [2.0 if k == label else 0.0 for k in range(len(LABEL_VOCAB))],
                 "probabilities": [.6 if k == label else .08 for k in range(len(LABEL_VOCAB))]}
                for i, (uid, did, gold) in enumerate(dev) for label in (gold if i < correct else (gold + 1) % 6,)]

    best_rows = predictions(693)
    e30_rows = predictions(640)
    best = {**classification_report(truth, [row["predicted_label"] for row in best_rows], loss=.4),
            "predictions": best_rows, "split": "dev", "optimizer_steps_performed": 0, **HOLD}
    e30, prediction_table, *_ = summarize(e30_rows, dev)
    e30_metrics = {**e30["metrics"], "loss": .5}
    logs = []
    trajectory = []
    for epoch in range(1, 31):
        metrics = best["metrics"] if epoch == 21 else e30_metrics
        logs.append({"epoch": epoch, "train_loss": .75, "dev_loss": metrics["loss"],
                     **{f"dev_{key}": metrics[key] for key in alignment.CLASSIFICATION},
                     "gradient_norm": 2.5, "train_dialogues": 100, "train_targets": 4830,
                     "is_best": epoch == 21, "optimizer_steps": epoch * 7, "dev_acc": metrics["accuracy"],
                     "dev_wf1": metrics["weighted_f1"]})
        for role, count in (("Fit", 4830), ("Dev", 980)):
            trajectory.append({**metrics, "epoch": epoch, "population": role, "valid_target_count": count,
                               "hard_label_nll": .25, "brier": .2, "ece10": .1, "mean_confidence": .8,
                               "mean_confidence_correct": .8, "mean_confidence_wrong": .8,
                               "high_confidence_coverage": 0, "high_confidence_error_share": 0,
                               "formal_hard_ce": metrics["loss"], "train_update_loss": .75,
                               "max_preclip_gradient_norm": 2.5, "optimizer_steps": epoch * 7})
    prediction_arms = ("CE Best", "CE E30", "LS Best", "LS E30", "CE AVG", "LS AVG", "CE U*", "LS U*")
    fixture = SimpleNamespace(
        config=config, raw=raw, frozen=frozen, run=run, archive=archive, stage_path=stage_path,
        projection_dir=projection_dir, entries={"dev": dev, "train": []},
        trajectory=[{"arm": arm, **row} for arm in ("CE", "LS") for row in trajectory],
        predictions=[{"arm": arm, **row} for arm in prediction_arms for row in prediction_table],
    )
    for folder in ("manifests", "reports", "logs", "checkpoints"):
        (run / folder).mkdir(parents=True)
        (pil.F0_RUN / folder).mkdir(parents=True)
    stage_path.parent.mkdir(parents=True)
    projection_dir.mkdir()
    lock = {**HOLD, "status": "LOCKED", "candidate_updates": config["budget_candidate_updates"],
            "selected_updates": config["budget_candidate_updates"][0]}
    write_json_atomic(lock_path, lock)
    checkpoint = run / "checkpoints/best_model.pt"
    checkpoint.write_bytes(b"PIL synthetic D0 identity; no serialized model")
    best["checkpoint_sha256"] = sha256_file(checkpoint)
    fixture.best = best
    experiment = {**HOLD, "protocol": pil.d.PROTOCOL, "arm": "D0", "run_id": run.name,
                  "train_label_smoothing": 0.0, "budget_sha256": json_identity(lock)}
    formal = frozen["formal_f0"]
    manifest = {**HOLD, "run_id": run.name, "status": "completed", "training_finished": True,
                "best_checkpoint_reloaded": True, "synthetic_only": False, "last_completed_epoch": 30,
                "optimizer_steps": 210, "best_epoch": 21, "parameter_count": 1_638_306,
                "git_head": pil.D0_HEAD, "experiment": experiment, "best_checkpoint_sha256": best["checkpoint_sha256"],
                "best_dev_metrics": best["metrics"], "config_sha256": json_identity(raw),
                "data_identity": formal["data_identity"], "architecture_identity": formal["architecture_sha256"],
                "shared_initialization": {}, "curriculum_membership_sha256": "synthetic_membership",
                "batch_pairing": {"population": formal["population"]}, "training_config": raw["training"],
                "active_modalities": raw["active_modalities"]}
    fixture.manifest = manifest
    (run / "resolved_config.yaml").write_text(yaml.safe_dump({"experiment": experiment}), encoding="utf-8")
    write_json_atomic(run / "manifests/run_manifest.json", manifest)
    write_json_atomic(run / "reports/validation_best_dev_evidence.json", best)
    write_json_atomic(run / "manifests/alignment_snapshots.json", {})
    (run / "manifests/curriculum_bucket_manifest.tsv").write_bytes(b"synthetic\n")
    epoch_bytes = table_bytes(EPOCH_TSV, logs, delimiter="\t")
    (run / "logs/epoch_metrics.tsv").write_bytes(epoch_bytes)

    reference_checkpoint = pil.F0_RUN / "checkpoints/best_model.pt"
    reference_checkpoint.write_bytes(b"PIL synthetic F0 identity; no serialized model")
    reference_identity = {**alignment.F0_IDENTITY, "git_head": "e" * 40,
                          "checkpoint_sha256": sha256_file(reference_checkpoint)}
    reference_manifest = {**manifest, "git_head": reference_identity["git_head"],
                          "best_checkpoint_sha256": reference_identity["checkpoint_sha256"]}
    reference_best = {**best, "checkpoint_sha256": reference_identity["checkpoint_sha256"]}
    write_json_atomic(pil.F0_RUN / "manifests/run_manifest.json", reference_manifest)
    write_json_atomic(pil.F0_RUN / "reports/validation_best_dev_evidence.json", reference_best)
    (pil.F0_RUN / "resolved_config.yaml").write_text(yaml.safe_dump({"audit_config": raw}), encoding="utf-8")
    (pil.F0_RUN / "logs/epoch_metrics.tsv").write_bytes(epoch_bytes)
    monkeypatch.setattr(alignment, "F0_IDENTITY", reference_identity)
    monkeypatch.setattr(alignment, "REFERENCE_FILES", {name: sha256_file(pil.F0_RUN / name)
                                                      for name in alignment.REFERENCE_FILES})
    fixture.reference = pil.d.read_reference(pil.F0_RUN, frozen, raw)

    projection = {**HOLD, "sha256": {pil.ASSET: "a" * 64, pil.MODEL_MANIFEST: "b" * 64}}
    write_json_atomic(projection_dir / pil.PROJECTION_MANIFEST, projection)
    monkeypatch.setattr(pil, "PROJECTION_SHA", projection["sha256"])
    fixture.stage = {**HOLD, "status": "PASS", "numerical_audit": {"numerical_identity": True},
                     "run_id": run.name, "config_sha256": json_identity(config), "d3_status": lock["status"],
                     "budget_sha256": sha256_file(lock_path), "projection": projection,
                     "run_evidence_sha256": {}}
    fixture.projection = projection
    seal_run(fixture, monkeypatch)
    seal_archive(fixture, monkeypatch)
    return fixture


def seal_run(fixture, monkeypatch):
    fixture.stage["run_evidence_sha256"] = {name: sha256_file(fixture.run / name) for name in pil.d.RUN_EVIDENCE}
    write_json_atomic(fixture.stage_path, fixture.stage)
    monkeypatch.setattr(pil, "STAGE_A_SHA", sha256_file(fixture.stage_path))


def seal_archive(fixture, monkeypatch, **kwargs):
    write_archive(fixture, **kwargs)
    monkeypatch.setattr(pil, "STAGE_B_SHA", sha256_file(fixture.archive))


def baseline(fixture):
    return pil.select_baseline(fixture.config, fixture.stage, *fixture.reference, fixture.entries)


def read_archive(fixture, logs=None):
    return pil.read_history_archive(fixture.archive, fixture.entries["dev"],
                                    pil.d.epoch_rows(fixture.run) if logs is None else logs, fixture.best)


def test_read_reviewed_history_schema(history):
    result = read_archive(history)
    assert result["Best"] == history.best
    assert len(result["history"]) == 30 and len(result["E30"]["predictions"]) == 980
    assert result["history"][0]["fit"]["valid_target_count"] == 4830
    assert result["history"][20]["dev"] == history.best["metrics"]
    assert result["history"][-1]["dev"] == result["E30"]["metrics"]
    assert result["history"][-1]["historical_train_update_loss"] == .75


def test_certify_reuse_and_actual_setup_with_cpu_boundaries(history, monkeypatch):
    datasets = {"train": object(), "dev": object()}
    monkeypatch.setattr(pil.d, "load_inputs", lambda *_: (datasets, history.entries, history.projection, {}))
    monkeypatch.setattr(pil, "build_target_tape", lambda *_: {"changed_count": 1})
    monkeypatch.setattr(pil, "PILRuntime", lambda *_, **__: SimpleNamespace(audit_sampler=lambda _: {}))
    monkeypatch.setattr(pil.d, "check_runtime", lambda *_: None)
    monkeypatch.setattr(pil, "source_environment", lambda _: ({}, {}))
    monkeypatch.setattr(pil, "device_parity", lambda *_: {})
    args = SimpleNamespace(projection_dir=history.projection_dir, batch_id="synthetic", experiment_date="20261128")
    *_, d0, identity, _, _ = pil.setup(args)
    assert identity["baseline"]["status"] == "CERTIFIED_REUSE"
    assert d0["best_epoch"] == 21 and d0["parameter_count"] == 1_638_306
    assert d0["checkpoints"] == {str(history.run / "checkpoints/best_model.pt"): history.best["checkpoint_sha256"]}
    history.archive.unlink()
    *_, d0, identity, _, _ = pil.setup(args)
    assert d0 is None and identity["baseline"]["status"] == "ONE_SHARED_OFF_REQUIRED"
    assert "ASSET_UNAVAILABLE" in identity["baseline"]["d0_reuse_blocker"]


@pytest.mark.parametrize("corruption", [
    "archive_sha", "header_missing", "header_duplicate", "file_duplicate", "epoch_missing", "epoch_duplicate",
    "population", "log_order", "malformed_number", "nonfinite_fit", "dev_metric", "train_objective",
    "e30_id", "e30_duplicate", "e30_order", "e30_logit", "e30_probability", "e30_metric",
])
def test_archive_contract_rejects_corruption(history, monkeypatch, corruption):
    logs = pil.d.epoch_rows(history.run)
    fields = TRAJECTORY
    if corruption == "archive_sha":
        monkeypatch.setattr(pil, "STAGE_B_SHA", "0" * 64)
    elif corruption == "header_missing":
        fields = tuple(field for field in fields if field != "weighted_f1")
    elif corruption == "header_duplicate":
        fields = (*fields, "accuracy")
    elif corruption == "epoch_missing":
        history.trajectory.pop(0)
    elif corruption == "epoch_duplicate":
        history.trajectory[1] = history.trajectory[0]
    elif corruption == "population":
        history.trajectory[0]["valid_target_count"] = 4829
    elif corruption == "log_order":
        logs[0], logs[1] = logs[1], logs[0]
    elif corruption == "malformed_number":
        history.trajectory[0]["formal_hard_ce"] = "invalid"
    elif corruption == "nonfinite_fit":
        history.trajectory[0]["accuracy"] = float("nan")
    elif corruption == "dev_metric":
        history.trajectory[1]["weighted_f1"] = .999
    elif corruption == "train_objective":
        history.trajectory[0]["train_update_loss"] = .8
    else:
        row = history.predictions[980]
        if corruption == "e30_id":
            row["utterance_id"] = "unexpected"
        elif corruption == "e30_duplicate":
            history.predictions[981] = row
        elif corruption == "e30_order":
            history.predictions[980], history.predictions[981] = history.predictions[981], row
        elif corruption == "e30_logit":
            row["logit_hap"] = float("inf")
        elif corruption == "e30_probability":
            row["prob_hap"] = float("nan")
        elif corruption == "e30_metric":
            row.update(predicted_label_id=1, logit_hap=0.0, logit_sad=2.0)
    if corruption != "archive_sha":
        seal_archive(history, monkeypatch, trajectory_fields=fields, duplicate_member=corruption == "file_duplicate")
    with pytest.raises(ValueError):
        read_archive(history, logs)


@pytest.mark.parametrize("corruption", ["best_epoch", "head", "resolved", "checkpoint_sha", "best_logits",
                                       "split", "hold", "archive_sha", "stage_sha", "parameter_count"])
def test_invalid_history_never_selects_off(history, monkeypatch, corruption):
    manifest_path = history.run / "manifests/run_manifest.json"
    best_path = history.run / "reports/validation_best_dev_evidence.json"
    if corruption in ("best_epoch", "head", "parameter_count"):
        key, value = {"best_epoch": ("best_epoch", 30), "head": ("git_head", "0" * 40),
                      "parameter_count": ("parameter_count", 42)}[corruption]
        write_json_atomic(manifest_path, {**history.manifest, key: value})
    elif corruption == "resolved":
        (history.run / "resolved_config.yaml").write_text("experiment: {}\n", encoding="utf-8")
    elif corruption == "checkpoint_sha":
        (history.run / "checkpoints/best_model.pt").write_bytes(b"altered checkpoint identity")
    elif corruption == "best_logits":
        history.best["predictions"][0]["logits"][0] += .01
        write_json_atomic(best_path, history.best)
    elif corruption in ("split", "hold"):
        changed = {"split": "test"} if corruption == "split" else {"test_evaluation_count": 1}
        write_json_atomic(best_path, {**history.best, **changed})
    seal_run(history, monkeypatch)
    if corruption in ("archive_sha", "stage_sha"):
        monkeypatch.setattr(pil, "STAGE_B_SHA" if corruption == "archive_sha" else "STAGE_A_SHA", "0" * 64)
    with pytest.raises(ValueError):
        baseline(history)


@pytest.mark.parametrize("error", [
    AttributeError("PIL symbol bug"), KeyError("unexpected field"), PermissionError("cannot read D0"),
    FileNotFoundError(2, "missing unexpected dependency", "unrelated-missing-file"),
    tarfile.ReadError("unreadable tar"),
])
def test_unexpected_history_error_is_propagated(history, monkeypatch, error):
    def broken_read(*_):
        raise error

    monkeypatch.setattr(pil, "read_history_archive", broken_read)
    with pytest.raises(type(error)) as raised:
        baseline(history)
    assert raised.value is error


def test_missing_stage_a_must_stop_instead_of_selecting_off(history):
    history.stage_path.unlink()
    with pytest.raises(FileNotFoundError):
        baseline(history)


def test_shared_off_identity_exact_and_ordered(history):
    _, best, logs = history.reference
    records = [{"epoch": int(row["epoch"]),
                "dev": {key: float(row[f"dev_{key}"]) for key in ("loss", *alignment.CLASSIFICATION)},
                "steps": [{"objective": float(row["train_loss"]), "gradient_norm": float(row["gradient_norm"]),
                           "valid_targets": int(row["train_targets"]), "step": int(row["optimizer_steps"])}]}
               for row in logs]
    result = {"history": records, "Best": best, "best_epoch": 21}
    pil.verify_shared_control(result, logs, best)
    records[0]["dev"]["loss"] += 1e-12
    with pytest.raises(ValueError, match="exact D tolerance"):
        pil.verify_shared_control(result, logs, best)
    records[0]["epoch"] = 2
    with pytest.raises(ValueError, match="selection or budget"):
        pil.verify_shared_control(result, logs, best)

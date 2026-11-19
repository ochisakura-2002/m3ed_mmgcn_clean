"""Exclusive output creation, same-run resume and best-Dev reload for TAV."""

from copy import deepcopy
import csv
import json
import os
from pathlib import Path

import torch
import yaml

from scripts.runtime.multidag_cl_ta_v0.formal import git_identity, json_identity, sha256_file, write_json_atomic
from scripts.runtime.multidag_cl_ta_v0.training import reload_model_checkpoint, seed_everything
from utils.output_paths import resolve_experiment_date, resolve_output_paths
from models.multidag_cl.tav_full_audit_v1 import TAVFullAudit
from .runtime import AuditRuntime, load_train_dev
from .spec import validate_config


def run_path(raw, *, output_root=None, experiment_date=None):
    return resolve_output_paths(output_base=output_root or raw["output"]["root"],
                                experiment_date=resolve_experiment_date(experiment_date, raw),
                                experiment_group=raw["output"]["experiment_group"],
                                run_id=raw["experiment_id"]).run_root


def completed_manifest(run_dir, raw, data_identity):
    """Skip only an exact, fully finished run with an intact selected checkpoint."""
    path = Path(run_dir) / "manifests" / "run_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if (manifest.get("status") != "completed" or manifest.get("training_finished") is not True
            or manifest.get("last_completed_epoch") != 30 or manifest.get("synthetic_only") is not False
            or manifest.get("config_sha256") != json_identity(raw)
            or manifest.get("data_identity") != data_identity
            or manifest.get("test_loader_built") != 0 or manifest.get("test_evaluation_count") != 0
            or manifest.get("best_checkpoint_sha256") != sha256_file(Path(run_dir) / "checkpoints" / "best_model.pt")):
        raise ValueError("existing run is failed/incomplete or has a different identity; cannot skip/overwrite")
    return manifest


def execute(runtime, raw, run_dir, *, data_paths=None, git=None, resume=None, synthetic_epochs=None,
            export_dev_evidence=False):
    raw = validate_config(raw)
    if raw != runtime.model.audit_config:
        raise ValueError("runtime/config mismatch")
    if os.name == "nt" and not runtime.synthetic_only:
        raise ValueError("formal TAV training is remote-only")
    if synthetic_epochs is not None and not runtime.synthetic_only:
        raise ValueError("epoch override is synthetic-only")
    epochs = 30 if synthetic_epochs is None else synthetic_epochs
    run_dir = Path(run_dir)
    if resume is None and run_dir.exists():
        raise FileExistsError(run_dir)
    if resume is not None:
        resume = Path(resume)
        if resume.resolve() != (run_dir / "checkpoints" / "last_model.pt").resolve():
            raise ValueError("resume must use last_model.pt in the original run")
        previous = json.loads((run_dir / "manifests" / "run_manifest.json").read_text(encoding="utf-8"))
        if (previous["config_sha256"] != json_identity(raw) or previous["data_identity"] != runtime.data_identity
                or previous["status"] == "completed"):
            raise ValueError("resume run identity/status mismatch")
        resolved = yaml.safe_load((run_dir / "resolved_config.yaml").read_text(encoding="utf-8"))
        if (resolved["audit_config"] != raw or resolved["data_identity"] != runtime.data_identity
                or Path(resolved["run_dir"]).resolve() != run_dir.resolve()):
            raise ValueError("resume resolved config identity mismatch")
        runtime.resume(resume)
        if previous.get("last_completed_epoch") != runtime.epoch:
            raise ValueError("manifest/checkpoint completed epoch mismatch")
        manifest = previous
    else:
        run_dir.mkdir(parents=True, exist_ok=False)
        for name in ("checkpoints", "logs", "manifests", "reports"):
            (run_dir / name).mkdir()
        (run_dir / "resolved_config.yaml").write_text(yaml.safe_dump(
            {"audit_config": raw, "data_paths": data_paths or {}, "data_identity": runtime.data_identity,
             "run_dir": run_dir.as_posix()}, sort_keys=False), encoding="utf-8")
        runtime.curriculum.export_manifest(run_dir / "manifests" / "curriculum_bucket_manifest.tsv")
        count = sum(parameter.numel() for parameter in runtime.model.parameters())
        manifest = {"schema": raw["schema"], "experiment_id": raw["experiment_id"], "run_id": raw["experiment_id"],
                    "model_name": raw["model_name"], "architecture_lineage": raw["architecture_lineage"],
                    "architecture_identity": json_identity({k: v for k, v in raw.items()
                                                           if k in ("architecture_lineage", "frontend_source", "graph_core")}),
                    "config_sha256": json_identity(raw), "data_identity": deepcopy(runtime.data_identity),
                    "data_paths": data_paths or {}, **(git or {"git_head": runtime.git_head}),
                    "seed": raw["training"]["seed"],
                    "feature_dimensions": {"text": 1024, "audio": 1582, "visual": 342},
                    "active_modalities": raw["active_modalities"], "graph_core": raw["graph_core"],
                    "curriculum": raw["curriculum"], "context": raw["context_mode"],
                    "curriculum_membership_sha256": runtime.curriculum.manifest.membership_sha256,
                    "training_config": raw["training"], "parameter_count": count,
                    "test_loader_built": 0, "test_evaluation_count": 0, "test_labels_inspected": 0,
                    "test_used_for_selection": False,
                    "synthetic_only": runtime.synthetic_only, "status": "initialized", "training_finished": False,
                    "last_completed_epoch": 0, "optimizer_steps": 0}
    if type(epochs) is not int or not 1 <= epochs <= 30 or runtime.epoch > epochs:
        raise ValueError("invalid training/resume epoch limit")
    # Exclusive lock protects a run from concurrent resumes as well as fresh fits.
    lock = run_dir / ".training.lock"
    descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.close(descriptor)
    manifest_path = run_dir / "manifests" / "run_manifest.json"
    try:
        write_json_atomic(manifest_path, manifest)
        columns = ("epoch", "train_loss", "dev_loss", "dev_accuracy", "dev_weighted_f1", "dev_macro_f1",
                   "dev_uar", "gradient_norm", "train_dialogues", "train_targets", "is_best", "optimizer_steps")
        epoch_path = run_dir / "logs" / "epoch_metrics.tsv"
        if resume is not None:
            with epoch_path.open(encoding="utf-8", newline="") as stream:
                old = list(csv.DictReader(stream, delimiter="\t"))
            if [int(row["epoch"]) for row in old] != list(range(1, runtime.epoch + 1)):
                raise ValueError("resume epoch log/checkpoint mismatch")
        with epoch_path.open("a" if resume else "x", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns, delimiter="\t", lineterminator="\n")
            if resume is None:
                writer.writeheader()
            for epoch in range(runtime.epoch + 1, epochs + 1):
                total, count, norm = 0.0, 0, 0.0
                manifest["status"] = "training"
                write_json_atomic(manifest_path, manifest)
                for batch in runtime.loader("train", epoch=epoch):
                    step = runtime.train_step(batch)
                    total += step["loss"] * step["valid_target_count"]
                    count += step["valid_target_count"]
                    norm = max(norm, step["gradient_norm"])
                visible = runtime.curriculum.visible_indices(epoch)
                expected = sum(len(runtime.dataset("train")[i]["utterance_ids"]) for i in visible)
                if count != expected or count == 0:
                    raise ValueError("CL-visible Train target count mismatch")
                dev = runtime.evaluate()
                if not runtime.synthetic_only and dev["metrics"]["valid_target_count"] != 980:
                    raise ValueError("canonical Dev target count mismatch")
                improved = runtime.save_epoch(run_dir / "checkpoints", epoch=epoch, dev=dev)
                metrics = dev["metrics"]
                writer.writerow({"epoch": epoch, "train_loss": total / count,
                                 **{f"dev_{key}": metrics[key] for key in ("loss", "accuracy", "weighted_f1", "macro_f1", "uar")},
                                 "gradient_norm": norm, "train_dialogues": len(visible), "train_targets": count,
                                 "is_best": improved, "optimizer_steps": runtime.optimizer_steps})
                stream.flush()
                manifest.update(last_completed_epoch=epoch, optimizer_steps=runtime.optimizer_steps,
                                best_epoch=runtime.selector.best.epoch)
                write_json_atomic(manifest_path, manifest)
                print(json.dumps({"experiment_id": raw["experiment_id"], "epoch": epoch,
                                  "dev_weighted_f1": metrics["weighted_f1"], "test_evaluation_count": 0}), flush=True)
        best_path = run_dir / "checkpoints" / "best_model.pt"
        reload_model_checkpoint(runtime.model, best_path, validate=runtime.validate_checkpoint)
        steps_before = runtime.optimizer_steps
        report = runtime.evaluate(include_predictions=True) if export_dev_evidence else runtime.evaluate()
        checkpoint_sha = sha256_file(best_path)
        if export_dev_evidence:
            if steps_before != runtime.optimizer_steps:
                raise ValueError("best-Dev export changed optimizer steps")
            write_json_atomic(run_dir / "reports" / "validation_best_dev_evidence.json",
                              {**report, "split": "dev", "checkpoint_sha256": checkpoint_sha,
                               "export_mode": "final_selected_checkpoint_reload", "optimizer_steps_performed": 0,
                               "test_loader_built": 0, "test_evaluation_count": 0})
        write_json_atomic(run_dir / "reports" / "validation_best_metrics.json", report["metrics"])
        manifest.update(status="synthetic_completed" if runtime.synthetic_only else "completed",
                        training_finished=True, best_checkpoint_reloaded=True, best_dev_metrics=report["metrics"],
                        best_checkpoint_sha256=checkpoint_sha, optimizer_steps=runtime.optimizer_steps)
        write_json_atomic(manifest_path, manifest)
        return manifest
    except Exception as error:
        manifest.update(status="failed", training_finished=False, error_type=type(error).__name__, error=str(error))
        write_json_atomic(manifest_path, manifest)
        raise
    finally:
        lock.unlink()


def real_runtime(raw, *, asset_path, model_manifest_path, provenance_manifest_path, device):
    datasets, identity, paths = load_train_dev(asset_path, model_manifest_path, provenance_manifest_path)
    seed_everything(device, raw["training"]["seed"])
    git = git_identity()
    runtime = AuditRuntime(TAVFullAudit(raw), datasets.__getitem__, formal_train_dev=True, device=device,
                           data_identity=identity, git_head=git["git_head"])
    return runtime, paths, git


def run_formal(*, formal_train_dev=False, config_path, asset_path, model_manifest_path,
               provenance_manifest_path, device="cuda", output_root=None, experiment_date=None, resume=None):
    if formal_train_dev is not True:
        raise ValueError("explicit --formal-train-dev required before data access")
    if os.name == "nt":
        raise ValueError("formal TAV training is remote-only; use --synthetic-smoke on Windows")
    raw = validate_config(yaml.safe_load(Path(config_path).read_text(encoding="utf-8")))
    if device != "cuda" or not torch.cuda.is_available():
        raise ValueError("formal audit requires explicit available CUDA")
    directory = Path(resume).resolve().parents[1] if resume else run_path(raw, output_root=output_root, experiment_date=experiment_date)
    if resume is None and directory.exists():
        raise FileExistsError(directory)
    runtime, paths, git = real_runtime(raw, asset_path=asset_path, model_manifest_path=model_manifest_path,
                                      provenance_manifest_path=provenance_manifest_path, device=device)
    return execute(runtime, raw, directory, data_paths=paths, git=git, resume=resume)

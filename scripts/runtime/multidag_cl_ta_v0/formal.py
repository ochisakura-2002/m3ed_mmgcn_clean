"""Frozen formal TA-v0 orchestration and provenance, restricted to Train/Dev."""

from copy import deepcopy
from datetime import datetime
import csv
import hashlib
import json
from pathlib import Path
import pickle
import subprocess

import numpy as np
import torch
import yaml

from datasets.iemocap.ta_v0_dataset import (
    MODEL_MANIFEST_FIELDS, PROVENANCE_MANIFEST_FIELDS, PACKAGE_ROWS, SCHEMA, VERSION,
    TAV0Dataset, read_columns,
)
from models.multidag_cl.ta_v0 import LABEL_VOCAB, TAV0, TAV0Config
from scripts.models.multidag_cl.ta_v0.export_dev import export_dev
from scripts.runtime.multidag_cl_paper_reimplementation.checkpoint import (
    save_checkpoint_atomic,
)
from .training import (
    PROTOCOL, TrainDevRuntime, classification_report, reload_model_checkpoint, seed_everything,
)

SPLIT_COUNTS = {"train": {"utterances": 4830, "dialogues": 100},
                "dev": {"utterances": 980, "dialogues": 20}}
ORDER_POLICY = "provenance.canonical_dialogue_order -> utterance_id -> validated ta_row_index"
EPOCH_COLUMNS = ("epoch", "train_loss", "dev_loss", "dev_accuracy", "dev_weighted_f1",
                 "dev_macro_f1", "dev_uar", "learning_rate", "gradient_norm", "is_best",
                 "optimizer_steps", "train_valid_targets", "dev_valid_targets")


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def validate_formal_config(raw):
    if set(raw) != {"model", "dataset", "runtime", "training", "checkpoint", "output"}:
        raise ValueError("unexpected/missing formal config sections")
    config = TAV0Config(**raw["model"])
    if config not in (TAV0Config(), TAV0Config(use_causal_history_gate=True)):
        raise ValueError("formal model protocol is frozen except for CHG v0")
    expected_model = config.checkpoint_config()
    if not config.use_causal_history_gate:
        for key in ("use_causal_history_gate", "chg_speaker_embedding_dim",
                    "chg_distance_embedding_dim", "chg_hidden_dim"):
            if key in raw["model"]:
                expected_model[key] = getattr(config, key)
    experiment_group = "ta_v0_chg_v0_formal" if config.use_causal_history_gate else "ta_v0_formal"
    expected = {
        "model": expected_model,
        "dataset": {"package": VERSION, "label_vocab": list(LABEL_VOCAB),
                    "asset_path": None, "model_manifest_path": None, "provenance_manifest_path": None},
        "runtime": {"mode": "formal_train_dev", "synthetic_only": False, "device": "cpu",
                    "splits": ["train", "dev"], "curriculum": False, "seed": 100},
        "training": PROTOCOL,
        "checkpoint": {"selection_metric": "dev_weighted_f1", "test_evaluation_count": 0,
                       "test_split_used_for_selection": False},
        "output": {"root": "outputs", "experiment_date": None, "experiment_group": experiment_group},
    }
    if json_identity(raw) != json_identity(expected):
        raise ValueError("formal model/data/training protocol is frozen; no tuning or Test access")
    return config


def load_train_dev(asset_path, model_manifest_path, provenance_manifest_path):
    paths = {"asset": Path(asset_path), "model_manifest": Path(model_manifest_path),
             "provenance_manifest": Path(provenance_manifest_path)}
    hashes = {key: sha256_file(path) for key, path in paths.items()}
    model_rows = read_columns(paths["model_manifest"], MODEL_MANIFEST_FIELDS)
    provenance_rows = read_columns(paths["provenance_manifest"], PROVENANCE_MANIFEST_FIELDS)
    with paths["asset"].open("rb") as stream:
        asset = pickle.load(stream)
    # Package-wide identity metadata is checked by the adapter; Test features
    # are never indexed, validated, collated, evaluated or exported.
    datasets = {split: TAV0Dataset(asset, model_rows, provenance_rows, split=split,
                                  expected_rows=PACKAGE_ROWS) for split in ("train", "dev")}
    counts = {split: {"utterances": sum(r["split"] == split for r in model_rows),
                      "dialogues": len(dataset)} for split, dataset in datasets.items()}
    if counts != SPLIT_COUNTS:
        raise ValueError(f"frozen Train/Dev counts mismatch: {counts}")
    if hashes != {key: sha256_file(path) for key, path in paths.items()}:
        raise ValueError("data package changed while loading")
    identity = {"sha256": hashes, "dataset_schema": SCHEMA, "dataset_version": VERSION,
                "package_utterances": PACKAGE_ROWS, "split_counts": counts,
                "label_vocab": list(LABEL_VOCAB), "canonical_order_policy": ORDER_POLICY}
    return datasets, identity, {key: path.as_posix() for key, path in paths.items()}


def write_json_atomic(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
                         encoding="utf-8")
    temporary.replace(path)


def git_identity():
    root = Path(__file__).resolve().parents[3]
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    dirty = bool(subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=normal"], cwd=root,
        stderr=subprocess.DEVNULL, text=True).strip())
    return {"git_head": head, "git_dirty": dirty}


def export_best(runtime, run_dir, *, expected_ids, expected_rows):
    checkpoint_path = run_dir / "checkpoints" / "best_model.pt"
    reload_model_checkpoint(runtime.model, checkpoint_path, validate=runtime.validate_checkpoint)
    metadata = export_dev(
        runtime, runtime.loader("dev"), run_dir, expected_ids=expected_ids,
        expected_rows=expected_rows, checkpoint_identity=sha256_file(checkpoint_path),
        config_identity=sha256_file(run_dir / "resolved_config.yaml"), formal_layout=True,
    )
    # Reports derive from exactly the exported best-Dev inference, including its
    # ordering and labels, rather than a separate model pass or checkpoint.
    with (run_dir / "predictions" / "validation_best_predictions.tsv").open(
            encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    logits = torch.tensor([[float(row[f"logit_{i}"]) for i in range(6)] for row in rows],
                          dtype=torch.float64)
    labels = torch.tensor([int(row["true_label"]) for row in rows])
    loss = float(torch.nn.functional.cross_entropy(logits, labels))
    report = classification_report(labels.numpy(), logits.argmax(-1).numpy(), loss=loss)
    reports = run_dir / "reports"
    reports.mkdir(exist_ok=False)
    write_json_atomic(reports / "validation_best_metrics.json",
                      dict(report["metrics"], checkpoint_identity=metadata["checkpoint_identity"]))
    with (reports / "validation_best_per_class_metrics.tsv").open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(report["per_class"][0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(report["per_class"])
    with (reports / "validation_best_confusion_matrix.tsv").open("x", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t")
        writer.writerow(("true_label", *LABEL_VOCAB))
        for label, row in zip(LABEL_VOCAB, report["confusion_matrix"]):
            writer.writerow((label, *row))
    return metadata, report["metrics"]


def fit(runtime, run_dir, *, resolved_config, manifest, expected_dev_ids,
        synthetic_epochs=None, one_step_smoke=False):
    """Fresh run only; explicit resumes use a new directory and preserved best."""
    if synthetic_epochs is not None and not runtime.synthetic_only:
        raise ValueError("epoch override is synthetic-only")
    epochs = 30 if synthetic_epochs is None else synthetic_epochs
    if type(epochs) is not int or not runtime.epoch < epochs <= 30:
        raise ValueError("no remaining training epochs")
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=False)
    for name in ("checkpoints", "logs", "manifests"):
        (run_dir / name).mkdir()
    (run_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump(resolved_config, sort_keys=False), encoding="utf-8")
    manifest = deepcopy(manifest)
    initial_steps = runtime.optimizer_steps
    manifest.update(training_started=False, optimizer_steps=initial_steps,
                    optimizer_steps_this_invocation=0, training_finished=False,
                    one_step_smoke=one_step_smoke, status="initialized",
                    test_loader_built=0, test_evaluation_count=0)
    manifest_path = run_dir / "manifests" / "run_manifest.json"
    write_json_atomic(manifest_path, manifest)
    if runtime.best_checkpoint is not None:
        save_checkpoint_atomic(run_dir / "checkpoints" / "best_model.pt", runtime.best_checkpoint)
        save_checkpoint_atomic(run_dir / "checkpoints" / "best_model_weights_only.pt",
                               runtime.best_checkpoint["model_state_dict"])
    try:
        with (run_dir / "logs" / "epoch_metrics.tsv").open("x", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=EPOCH_COLUMNS, delimiter="\t")
            writer.writeheader()
            for epoch in range(runtime.epoch + 1, epochs + 1):
                numerator, count, max_norm = 0.0, 0, 0.0
                for batch in runtime.loader("train", epoch=epoch):
                    step = runtime.train_step(batch)
                    numerator += step["loss"] * step["valid_target_count"]
                    count += step["valid_target_count"]
                    max_norm = max(max_norm, step["gradient_norm"])
                    manifest.update(training_started=True, optimizer_steps=runtime.optimizer_steps,
                                    optimizer_steps_this_invocation=runtime.optimizer_steps - initial_steps,
                                    status="training")
                    write_json_atomic(manifest_path, manifest)
                    if one_step_smoke:
                        break
                if count == 0:
                    raise ValueError("empty Train epoch")
                dev = runtime.evaluate()
                if "chg_gate_diagnostics" in dev:
                    with (run_dir / "logs" / "chg_gate_diagnostics.jsonl").open(
                            "a", encoding="utf-8") as diagnostics_stream:
                        diagnostics_stream.write(json.dumps(
                            {"epoch": epoch, "split": "dev", "groups": dev["chg_gate_diagnostics"]},
                            sort_keys=True, allow_nan=False) + "\n")
                if dev["metrics"]["valid_target_count"] != len(expected_dev_ids):
                    raise ValueError("Dev valid target count mismatch")
                if not one_step_smoke and not runtime.synthetic_only and count != 4830:
                    raise ValueError("Train valid target count mismatch")
                improved = runtime.save_epoch(run_dir / "checkpoints", epoch=epoch, dev=dev,
                                              epoch_complete=not one_step_smoke)
                metrics = dev["metrics"]
                writer.writerow({
                    "epoch": epoch, "train_loss": numerator / count,
                    **{f"dev_{key}": metrics[key] for key in
                       ("loss", "accuracy", "weighted_f1", "macro_f1", "uar")},
                    "learning_rate": runtime.optimizer.param_groups[0]["lr"],
                    "gradient_norm": max_norm, "is_best": improved,
                    "optimizer_steps": runtime.optimizer_steps,
                    "train_valid_targets": count, "dev_valid_targets": metrics["valid_target_count"],
                })
                stream.flush()
                manifest.update(last_completed_epoch=epoch,
                                best_epoch=runtime.selector.best.epoch,
                                best_dev_weighted_f1=runtime.selector.best.val_weighted_f1,
                                best_dev_loss=runtime.selector.best.val_loss)
                write_json_atomic(manifest_path, manifest)
                print(json.dumps({"epoch": epoch, "optimizer_steps": runtime.optimizer_steps,
                                  "dev_weighted_f1": metrics["weighted_f1"], "is_best": improved}), flush=True)
                if one_step_smoke:
                    break
        metadata, best_metrics = export_best(runtime, run_dir, expected_ids=expected_dev_ids,
                                              expected_rows=len(expected_dev_ids))
        manifest.update(status="smoke_completed" if one_step_smoke else "completed",
                        training_finished=not one_step_smoke, best_checkpoint_reloaded=True,
                        best_checkpoint_sha256=metadata["checkpoint_identity"],
                        best_dev_metrics=best_metrics, dev_export=metadata)
        write_json_atomic(manifest_path, manifest)
    except Exception as error:
        manifest.update(status="failed", optimizer_steps=runtime.optimizer_steps,
                        optimizer_steps_this_invocation=runtime.optimizer_steps - initial_steps,
                        training_started=runtime.optimizer_steps > initial_steps,
                        error_type=type(error).__name__, error=str(error))
        write_json_atomic(manifest_path, manifest)
        raise
    return manifest


def run_formal(*, formal_train_dev=False, config_path, asset_path, model_manifest_path,
               provenance_manifest_path, output_dir, device="cpu", resume=None,
               one_step_smoke=False):
    if formal_train_dev is not True:
        raise ValueError("--formal-train-dev is required before real data access")
    raw = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    config = validate_formal_config(raw)
    device = torch.device(device)
    if device.type not in ("cpu", "cuda"):
        raise ValueError("formal device must be cpu or cuda")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("explicit CUDA device is unavailable")
    if one_step_smoke and resume is not None:
        raise ValueError("one-step smoke cannot resume a partial epoch")
    datasets, data_identity, paths = load_train_dev(asset_path, model_manifest_path, provenance_manifest_path)
    seed_everything(device)
    git = git_identity()
    runtime = TrainDevRuntime(TAV0(config), datasets.__getitem__, device=device,
                             formal_train_dev=True, data_identity=data_identity, git_head=git["git_head"])
    if resume is not None:
        runtime.resume(resume)
    stamp = datetime.now().astimezone().strftime("%Y%m%dT%H%M%S%f%z")
    run_id = f"{config.lineage}_seed100_{stamp}_{json_identity(data_identity)[:8]}"
    run_dir = Path(output_dir) / "runs" / run_id
    resolved = deepcopy(raw)
    resolved["dataset"].update(asset_path=paths["asset"], model_manifest_path=paths["model_manifest"],
                               provenance_manifest_path=paths["provenance_manifest"])
    resolved["runtime"].update(device=str(device), formal_train_dev=True, one_step_smoke=one_step_smoke)
    resolved["output"].update(run_id=run_id, run_dir=run_dir.as_posix())
    expected_ids = [uid for did in datasets["dev"].dialogue_ids
                    for _, _, uid in datasets["dev"]._ordered[did]]
    manifest = {
        "run_id": run_id, "run_dir": run_dir.as_posix(), **git, "seed": 100,
        "device": str(device), "formal_train_dev": True, "synthetic_only": False,
        "model_parameter_count": sum(p.numel() for p in runtime.model.parameters()),
        "model_architecture_identity": json_identity(config.checkpoint_config()),
        "model_lineage": config.lineage, "model_config": config.checkpoint_config(),
        "data_paths": paths, "data_identity": data_identity, "label_vocab": list(LABEL_VOCAB),
        "training_config": deepcopy(PROTOCOL),
        "checkpoint_metric": "dev_weighted_f1",
        "checkpoint_tiebreak_rules": ["weighted_f1_desc", "dev_loss_asc", "epoch_asc"],
        "resume": None if resume is None else {"path": str(resume), "sha256": sha256_file(resume),
                                                "epoch": runtime.epoch, "global_step": runtime.optimizer_steps},
    }
    return fit(runtime, run_dir, resolved_config=resolved, manifest=manifest,
               expected_dev_ids=expected_ids, one_step_smoke=one_step_smoke)

"""Remote S2D-V0 training; --check-config is local, read-only, and data-free."""

from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass
import argparse
import csv
import json
import math
import os
import random
import secrets
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.s2d.iemocap.config import config_summary, load_config, resolve_path, validate_runtime_paths
from scripts.models.s2d.iemocap.checkpoints import reload_model_weights, save_model_weights


FINAL_TEST_ATTEMPT_FILE = "logs/finalization_attempt.json"
FINAL_TEST_OUTPUTS = ("metrics/test_final.json", "predictions/test_predictions.csv",
                      "representations/test_visual_outputs.pt")


@dataclass
class EarlyStopping:
    """Dev Weighted-F1 only; count every non-improving epoch, including warmup."""

    enabled: bool
    min_epochs: int
    patience: int
    best_epoch: int | None = None
    best_val_weighted_f1: float = -math.inf
    epochs_without_improvement: int = 0

    def update(self, epoch, score):
        if not math.isfinite(score):
            raise FloatingPointError("Non-finite Dev Weighted-F1")
        if score > self.best_val_weighted_f1:
            self.best_epoch, self.best_val_weighted_f1 = epoch, score
            self.epochs_without_improvement = 0
            return True
        self.epochs_without_improvement += 1
        return False

    def should_stop(self, epoch):
        return (self.enabled and epoch >= self.min_epochs
                and self.epochs_without_improvement >= self.patience)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                          encoding="utf-8")


def reserve_test_evaluation(run_dir, evidence):
    """Fail closed across concurrent finalizers, interrupted Test, and export failure."""
    run_dir = Path(run_dir)
    metadata = json.loads((run_dir / "run_metadata.json").read_text(encoding="utf-8"))
    if (type(metadata.get("test_evaluation_count")) is not int or metadata["test_evaluation_count"] != 0
            or metadata.get("training_completed") is not False
            or metadata.get("test_split_used_for_selection") is not False):
        raise ValueError("Test-once gate: invalid or already completed evaluation state")
    for name in FINAL_TEST_OUTPUTS:
        if (run_dir / name).exists():
            raise FileExistsError(f"Refusing to overwrite {name}")
    # Retain this exclusive, durable marker on both success and failure. Never auto-reset it.
    with (run_dir / FINAL_TEST_ATTEMPT_FILE).open("x", encoding="utf-8") as stream:
        json.dump({**evidence, "test_evaluation_reserved": True}, stream, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())


def worker_seed(worker_id):
    import numpy as np
    import torch

    seed = torch.initial_seed() % (2 ** 32)
    random.seed(seed)
    np.random.seed(seed)


def classification_report(y_true, y_pred, vocab):
    from sklearn.metrics import precision_recall_fscore_support
    from utils.metrics import compute_classification_metrics, compute_confusion_matrix

    labels = list(range(vocab.num_classes))
    result = compute_classification_metrics(y_true, y_pred, labels=labels)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0)
    result["per_class"] = {
        label: {"precision": float(precision[i]), "recall": float(recall[i]),
                "f1": float(f1[i]), "support": int(support[i])}
        for i, label in enumerate(vocab.index_to_label)}
    result["confusion_matrix"] = compute_confusion_matrix(y_true, y_pred, labels=labels).tolist()
    result["confusion_matrix_labels"] = list(vocab.index_to_label)
    return result


def run_epoch(model, loader, device, vocab, optimizer=None, *, collect=False):
    import torch
    from torch.nn import functional as F

    training = optimizer is not None
    model.train(training)
    true, predicted, exported = [], [], []
    loss_sum, count = 0.0, 0
    with torch.set_grad_enabled(training):
        for batch in loader:
            if not bool(batch["v_valid"].all()):
                raise ValueError("Invalid visual sample reached S2D forward")
            labels = batch["label_index"].to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            output = model(batch["video"].to(device))
            loss = F.cross_entropy(output["logits_v"], labels)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("Non-finite CrossEntropy")
            if training:
                loss.backward()
                optimizer.step()
            loss_sum += loss.item() * len(labels)
            count += len(labels)
            true.extend(labels.detach().cpu().tolist())
            predicted.extend(output["logits_v"].detach().argmax(-1).cpu().tolist())
            if collect:
                exported.append({"utterance_id": list(batch["utterance_id"]),
                                 "label_index": labels.detach().cpu(),
                                 **{key: value.detach().cpu() for key, value in output.items()}})
    if not count:
        raise ValueError("No V-available samples in this split")
    report = classification_report(true, predicted, vocab)
    report.update({"loss": loss_sum / count, "sample_count": count})
    return report, exported


def export_predictions(batches, destination, vocab):
    """Identity-preserving final predictions; V-unavailable rows stay in coverage.json."""
    with Path(destination).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["utterance_id", "v_valid", "benchmark_label", "label_index", "predicted_index",
                         *[f"logit_{label}" for label in vocab.index_to_label],
                         *[f"prob_{label}" for label in vocab.index_to_label]])
        for batch in batches:
            for i, uid in enumerate(batch["utterance_id"]):
                label = int(batch["label_index"][i])
                writer.writerow([uid, True, vocab.index_to_label[label], label,
                                 int(batch["logits_v"][i].argmax()),
                                 *batch["logits_v"][i].tolist(), *batch["prob_v"][i].tolist()])


def prepare_s2d_run_environment(config, experiment_date=None, experiment_group=None):
    """Allocate <S2D root>/<date>/<run_id> without changing shared layouts."""
    from utils.io import make_run_id, save_yaml
    from utils.output_paths import configured_run_id, resolve_experiment_date, resolve_experiment_group
    from utils.run_metadata import write_run_metadata

    output_root = resolve_path(config["system"].get("output_dir"))
    if output_root is None:
        raise ValueError("system.output_dir must be a nonempty S2D output root")
    frozen_date = resolve_experiment_date(experiment_date, config, env={})
    group = resolve_experiment_group(experiment_group, config, default="s2d_iemocap_v0")
    fixed_id = configured_run_id(config)
    for _ in range(100):
        run_id = fixed_id or f"{make_run_id(config['project']['experiment_name'])}_{secrets.token_hex(3)}"
        run_dir = output_root / frozen_date / run_id
        configured_root = config.get("output", {}).get("root") if fixed_id else None
        if configured_root is not None and resolve_path(configured_root) != run_dir:
            raise ValueError(f"Configured output.root does not match the S2D run root: {run_dir}")
        try:
            run_dir.mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            if fixed_id:
                raise
            continue
        break
    else:
        raise RuntimeError("Unable to allocate a unique S2D run directory after 100 attempts")
    run = {"run_id": run_id, "run_dir": run_dir,
           **{f"{name}_dir": run_dir / name
              for name in ("checkpoints", "logs", "predictions", "metrics")}}
    if config["export"]["representations"]:
        run["representations_dir"] = run_dir / "representations"
    for key, path in run.items():
        if key.endswith("_dir"):
            path.mkdir(exist_ok=True)
    config["system"]["seed"] = config["train"]["seed"]
    config["system"]["output_dir"] = str(output_root)
    config.setdefault("output", {}).update({
        "root": str(run_dir), "output_base": str(output_root), "output_root": str(output_root),
        "experiment_root": str(output_root), "day_output_root": str(output_root / frozen_date),
        "experiment_date": frozen_date, "experiment_group": group, "run_id": run_id,
        "run_root": str(run_dir), "log_dir": str(run["logs_dir"]),
        **{key: str(value) for key, value in run.items() if key.endswith("_dir")},
    })
    save_yaml(config, run_dir / "resolved_config.yaml")
    save_yaml(config, run["logs_dir"] / "experiment_config.yaml")
    write_run_metadata(config, run_dir / "run_metadata.json", PROJECT_ROOT)
    return run


def write_s2d_run_provenance(config, vocab, run, load_reports):
    from datasets.iemocap.s2d_faces import PREPROCESSING_VERSION

    metadata_path = run["run_dir"] / "run_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT,
                            capture_output=True, text=True, check=True).stdout.strip()
    branch = subprocess.run(["git", "branch", "--show-current"], cwd=PROJECT_ROOT,
                            capture_output=True, text=True, check=True).stdout.strip()
    metadata.update({"git_commit": commit, "git_branch": branch, "run_id": run["run_id"],
                     "training_completed": False, "test_evaluation_count": 0,
                     "test_split_used_for_selection": False,
                     "early_stopping": dict(config["early_stopping"]),
                     "checkpoint_selection_metric": "val_weighted_f1",
                     "upstream_s2d_commit": json.loads((PROJECT_ROOT / "third_party/S2D/UPSTREAM.json").read_text())["commit"],
                     "manifest_path": config["dataset"]["manifest_path"],
                     "data_root": config["dataset"]["data_root"], "label_vocab": vocab.as_dict(),
                     "checkpoints": {name: {
                         "path": config["model"][field],
                         "expected_sha256": config["model"].get(f"{field}_sha256"),
                         "sha256_verified": load_reports[name]["sha256_verified"],
                     } for name, field in (("s2d", "pretrained_checkpoint"), ("landmark", "landmark_checkpoint"))},
                     "face_cache_path": config["face"]["cache_path"],
                     "face_preprocessing_version": PREPROCESSING_VERSION,
                     "predictions_dir": str(run["predictions_dir"]),
                     "representations_dir": str(run["representations_dir"]) if "representations_dir" in run else None})
    write_json(metadata_path, metadata)
    write_json(run["logs_dir"] / "label_vocab.json", vocab.as_dict())
    write_json(run["logs_dir"] / "pretrained_load_report.json", load_reports)


def train(config, vocab, *, experiment_date=None, experiment_group=None):
    validate_runtime_paths(config)

    import torch
    from torch.utils.data import DataLoader

    from datasets.iemocap.s2d_dataset import IEMOCAPS2DDataset
    from datasets.iemocap.s2d_faces import FaceBBoxCache, PREPROCESSING_VERSION, build_face_processor
    from datasets.iemocap.s2d_manifest import S2DManifest
    from models.s2d.iemocap.model import build_s2d
    from utils.seed import set_seed

    if not config["model"]["enabled"]:
        raise ValueError("S2D branch disabled; use the unchanged TA baseline entrypoint")
    set_seed(config["train"]["seed"])
    device = torch.device(config["system"]["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; run real S2D smoke/training remotely")
    data = config["dataset"]
    root = resolve_path(data["data_root"])
    manifest = S2DManifest(resolve_path(data["manifest_path"], root), root, vocab, data.get("columns"))
    face = build_face_processor(config["face"])
    face_cache = FaceBBoxCache.load(resolve_path(config["face"]["cache_path"]))
    face_cache.require_ids(r.utterance_id for r in manifest.records if r.v_available)
    generator = torch.Generator().manual_seed(config["train"]["seed"])

    def make_loader(split):
        dataset = IEMOCAPS2DDataset(
            manifest, split, face, face_cache=face_cache,
            require_face_cache_for_training=config["face"].get("require_cache_for_training", True))
        if not len(dataset):
            raise ValueError(f"No available V rows for {split}")
        return DataLoader(dataset, batch_size=config["train"]["batch_size"],
                          num_workers=config["train"]["num_workers"], shuffle=(split == "train"),
                          worker_init_fn=worker_seed, generator=generator, drop_last=False)

    train_loader, dev_loader = make_loader("train"), make_loader("dev")
    model_config = dict(config["model"])
    for key in ("pretrained_checkpoint", "landmark_checkpoint"):
        model_config[key] = str(resolve_path(model_config[key]))
    model = build_s2d(vocab, model_config).to(device)
    print(json.dumps({"pretrained_load_reports": model.load_reports}, indent=2), flush=True)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                 lr=config["train"]["learning_rate"],
                                 weight_decay=config["train"]["weight_decay"])
    run = prepare_s2d_run_environment(config, experiment_date, experiment_group)
    write_s2d_run_provenance(config, vocab, run, model.load_reports)
    write_json(run["metrics_dir"] / "coverage.json", {
        "counts": manifest.counts(), "metric_population": "V-available and cached face_usable rows only; decode/cache mismatch aborts",
        "face_preprocessing_version": PREPROCESSING_VERSION,
        "face_coverage": {split: {
            "face_usable": sum(face_cache.entries[r.utterance_id].face_usable for r in manifest.for_split(split)),
            "face_unusable_utterance_ids": [r.utterance_id for r in manifest.for_split(split)
                                            if not face_cache.entries[r.utterance_id].face_usable],
            "valid_detection_count": sum(face_cache.entries[r.utterance_id].valid_detection_count
                                         for r in manifest.for_split(split)),
            "fallback_frame_count": sum(sum(face_cache.entries[r.utterance_id].fallback_used)
                                        for r in manifest.for_split(split))}
            for split in ("train", "dev", "test")},
        "v_unavailable_utterance_ids": [r.utterance_id for r in manifest.records if not r.v_available]})
    early = config["early_stopping"]
    selection = EarlyStopping(early["enabled"], early["min_epochs"], early["patience"])
    stop_reason = "max_epochs"
    best_path = run["checkpoints_dir"] / "best_model.pt"
    for epoch in range(1, config["train"]["epochs"] + 1):
        training, _ = run_epoch(model, train_loader, device, vocab, optimizer)
        validation, _ = run_epoch(model, dev_loader, device, vocab)
        score = validation["weighted_f1"]
        if selection.update(epoch, score):  # Strict improvement; earliest epoch wins ties.
            save_model_weights(model, best_path)
        record = {"epoch": epoch, "train": training, "dev": validation,
                  "best_epoch": selection.best_epoch,
                  "best_val_weighted_f1": selection.best_val_weighted_f1,
                  "epochs_without_improvement": selection.epochs_without_improvement}
        with (run["logs_dir"] / "epochs.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, allow_nan=False) + "\n")
        print(f"epoch={epoch} dev_weighted_f1={score:.6f} best_epoch={selection.best_epoch}", flush=True)
        if selection.should_stop(epoch):
            stop_reason = "early_stopping"
            break
    stopped_epoch = epoch
    reload_model_weights(model, best_path)
    reserve_test_evaluation(run["run_dir"], {"finalization_mode": "training_run",
                                           "selected_epoch": selection.best_epoch})
    # Test loader and frames are accessed only after validation selection is finished.
    test_evaluation_count = 0
    test_report, batches = run_epoch(model, make_loader("test"), device, vocab, collect=True)
    test_evaluation_count += 1
    test_report.update({"selected_epoch": selection.best_epoch,
                        "selected_val_weighted_f1": selection.best_val_weighted_f1})
    write_json(run["metrics_dir"] / "test_final.json", test_report)
    export_predictions(batches, run["predictions_dir"] / "test_predictions.csv", vocab)
    if config["export"]["representations"]:
        torch.save({"label_vocab": vocab.as_dict(), "split": "test", "batches": batches},
                   run["representations_dir"] / "test_visual_outputs.pt")
    metadata_path = run["run_dir"] / "run_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata.update({"training_completed": True, "stop_reason": stop_reason,
                     "stopped_epoch": stopped_epoch, "best_epoch": selection.best_epoch,
                     "best_val_weighted_f1": selection.best_val_weighted_f1,
                     "min_epochs": selection.min_epochs, "patience": selection.patience,
                     "epochs_without_improvement_at_stop": selection.epochs_without_improvement,
                     "test_evaluation_count": test_evaluation_count,
                     "test_split_used_for_selection": False})
    write_json(metadata_path, metadata)
    print(f"Final test and predictions saved in {run['run_dir']}")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/s2d/iemocap/v0.yaml")
    parser.add_argument("--check-config", action="store_true", help="Read YAML/vocab only; no data/weights/output")
    parser.add_argument("--experiment-date")
    parser.add_argument("--experiment-group")
    parser.add_argument("--run-id", help="Fixed run ID; an existing destination is an error")
    parser.add_argument("--data-root", help="IEMOCAP data directory; relative to project root or absolute")
    parser.add_argument("--manifest-path", help="Manifest CSV; relative to data root or absolute")
    parser.add_argument("--face-cache", help="Face bbox cache JSON; relative to project root or absolute")
    parser.add_argument("--pretrained-checkpoint", help="External S2D checkpoint path")
    parser.add_argument("--landmark-checkpoint", help="External MobileFaceNet checkpoint path")
    parser.add_argument("--output-dir", help="S2D output root; appends <YYYYMMDD>/<run_id>/")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    config, vocab = load_config(args.config, overrides=vars(args))
    if args.check_config:
        print(json.dumps(config_summary(config, vocab), ensure_ascii=False, indent=2))
        return
    train(config, vocab)


if __name__ == "__main__":
    main()

"""Remote S2D-V0 training; --check-config is local, read-only, and data-free."""

from pathlib import Path
import argparse
import csv
import json
import math
import random
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.s2d.iemocap.config import config_summary, load_config, resolve_path


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                          encoding="utf-8")


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


def train(config, vocab, *, experiment_date=None, experiment_group=None):
    import torch
    from torch.utils.data import DataLoader

    from datasets.iemocap.s2d_dataset import IEMOCAPS2DDataset
    from datasets.iemocap.s2d_faces import build_face_processor
    from datasets.iemocap.s2d_manifest import S2DManifest
    from models.s2d.iemocap.model import build_s2d
    from utils.io import prepare_run_environment
    from utils.seed import set_seed

    missing = config_summary(config, vocab)["REMOTE_REQUIRED_FIELDS"]
    if missing:
        raise ValueError(f"Configure remote inputs before training: {missing}")
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
    generator = torch.Generator().manual_seed(config["train"]["seed"])

    def make_loader(split):
        dataset = IEMOCAPS2DDataset(manifest, split, face)
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
    config["system"]["seed"] = config["train"]["seed"]
    run = prepare_run_environment(config, experiment_date, experiment_group)
    metadata_path = run["run_dir"] / "run_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT,
                            capture_output=True, text=True, check=True).stdout.strip()
    metadata.update({"git_commit": commit, "checkpoint_selection_metric": "val_weighted_f1",
                     "upstream_s2d_commit": json.loads((PROJECT_ROOT / "third_party/S2D/UPSTREAM.json").read_text())["commit"],
                     "manifest_path": str(manifest.manifest_path), "data_root": str(root),
                     "label_vocab": vocab.as_dict()})
    write_json(metadata_path, metadata)
    write_json(run["artifacts_dir"] / "label_vocab.json", vocab.as_dict())
    write_json(run["artifacts_dir"] / "pretrained_load_report.json", model.load_reports)
    write_json(run["metrics_dir"] / "coverage.json", {
        "counts": manifest.counts(), "metric_population": "V-available rows only; face/decode failure aborts",
        "v_unavailable_utterance_ids": [r.utterance_id for r in manifest.records if not r.v_available]})
    best, best_epoch = -math.inf, None
    best_path = run["checkpoints_dir"] / "best_model.pt"
    for epoch in range(1, config["train"]["epochs"] + 1):
        training, _ = run_epoch(model, train_loader, device, vocab, optimizer)
        validation, _ = run_epoch(model, dev_loader, device, vocab)
        score = validation["weighted_f1"]
        if score > best:  # Deterministic earliest epoch wins ties; test never consulted.
            best, best_epoch = score, epoch
            torch.save({"model": model.state_dict(), "epoch": epoch,
                        "val_weighted_f1": best, "label_vocab": vocab.as_dict(),
                        "config": config}, best_path)
        record = {"epoch": epoch, "train": training, "dev": validation,
                  "best_epoch": best_epoch, "best_val_weighted_f1": best}
        with (run["logs_dir"] / "epochs.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, allow_nan=False) + "\n")
        print(f"epoch={epoch} dev_weighted_f1={score:.6f} best_epoch={best_epoch}", flush=True)
    checkpoint = torch.load(best_path, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model"], strict=True)
    # Test loader and frames are accessed only after validation selection is finished.
    test_report, batches = run_epoch(model, make_loader("test"), device, vocab, collect=True)
    test_report.update({"selected_epoch": best_epoch, "selected_val_weighted_f1": best})
    write_json(run["metrics_dir"] / "test_final.json", test_report)
    export_predictions(batches, run["artifacts_dir"] / "test_predictions.csv", vocab)
    if config["export"]["representations"]:
        torch.save({"label_vocab": vocab.as_dict(), "split": "test", "batches": batches},
                   run["artifacts_dir"] / "test_visual_outputs.pt")
    print(f"Final test and predictions saved in {run['run_dir']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/s2d/iemocap/v0.yaml")
    parser.add_argument("--check-config", action="store_true", help="Read YAML/vocab only; no data/weights/output")
    parser.add_argument("--experiment-date")
    parser.add_argument("--experiment-group")
    args = parser.parse_args()
    config, vocab = load_config(args.config)
    if args.check_config:
        print(json.dumps(config_summary(config, vocab), ensure_ascii=False, indent=2))
        return
    train(config, vocab, experiment_date=args.experiment_date, experiment_group=args.experiment_group)


if __name__ == "__main__":
    main()

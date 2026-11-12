"""Fit one B1/B2 seed on a validated frozen Train/Dev cache, without Test IO."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import platform
import subprocess
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from sklearn.metrics import precision_recall_fscore_support
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import yaml

from models.asymmetric_merc.feature_fusion import VisualReliabilityGate, build_model
from scripts.analysis.models.asymmetric_merc.fusion_report import write_csv, write_json
from scripts.analysis.models.asymmetric_merc.vrg_report import write_gate_report
from scripts.analysis.models.asymmetric_merc.weighted_fusion import LABEL_NAMES, metrics
from scripts.models.asymmetric_merc.frozen_tav_cache import (
    V_PRED_SOURCE, assert_unchanged, attach_probe, guard_input, identity, load_cache,
    reserve_output, standardize,
)
from utils.run_metadata import compute_file_sha256
from utils.seed import set_seed

DEFAULT_CONFIG = PROJECT_ROOT / "configs/asymmetric_merc/frozen_tav/seed100.yaml"
PROTOCOL_VERSION = "asymmetric_merc_frozen_tav_vrg_v0"
FROZEN_TRAIN = {"seed": 100, "loss": "CrossEntropy", "optimizer": "AdamW", "lr": 1e-3,
                "weight_decay": 1e-2, "batch_size": 128, "max_epochs": 100, "patience": 10,
                "train_shuffle": True, "dev_shuffle": False,
                "selection": ["dev_weighted_f1_desc", "dev_loss_asc", "epoch_asc"]}


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True, help="New run directory; no overwrite/resume")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--model", choices=("concat_linear", "vrg"))
    parser.add_argument("--seed", type=int, help="One seed per invocation; first formal round requires 100")
    parser.add_argument("--split", choices=("dev",), default="dev", help="Only Dev evaluation is supported")
    parser.add_argument("--v-quadrant-predictions", type=Path, help="Explicit frozen Linear-B seed100 Dev CSV/TSV")
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    parser.add_argument("--synthetic", action="store_true", help="Requires a synthetic-marked cache")
    parser.add_argument("--smoke-epochs", type=int, help="1..3 epochs, allowed only with --synthetic")
    return parser


def load_protocol(args):
    guard_input(args.config)
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if (not isinstance(config, dict) or set(config) != {"protocol_version", "splits", "vrg", "train"}
            or config.get("protocol_version") != PROTOCOL_VERSION or config.get("splits") != ["train", "dev"]
            or config.get("train") != FROZEN_TRAIN or not isinstance(config.get("vrg"), dict)
            or set(config["vrg"]) != {"enabled"} or type(config["vrg"]["enabled"]) is not bool):
        raise ValueError("Frozen Train/Dev protocol mismatch; Test and hyperparameter changes are forbidden")
    if args.split != "dev":
        raise ValueError("Only Dev evaluation allowed; Test is forbidden")
    if args.model is not None:
        config["vrg"]["enabled"] = args.model == "vrg"
    if args.seed is not None:
        config["train"]["seed"] = args.seed
    if not args.synthetic and config["train"]["seed"] != 100:
        raise ValueError("First formal round is seed100 only")
    if args.smoke_epochs is not None:
        if not args.synthetic or not 1 <= args.smoke_epochs <= 3:
            raise ValueError("--smoke-epochs requires --synthetic and 1..3 epochs")
        config["train"]["max_epochs"] = args.smoke_epochs
    config.update(synthetic=args.synthetic, evaluation_split="dev", device=args.device)
    return config


def selection_key(row):
    return (-row["dev_weighted_f1"], row["dev_loss"], row["epoch"])


def evaluate(model, loader, device):
    model.eval()
    logits, labels, gates = [], [], []
    loss_sum = 0.0
    with torch.no_grad():
        for ta, visual, target in loader:
            ta, visual, target = ta.to(device), visual.to(device), target.to(device)
            if isinstance(model, VisualReliabilityGate):
                result = model(ta, visual, return_details=True)
                output = result["logits"]
                gates.append(result["r"].cpu().numpy().reshape(-1))
            else:
                output = model(ta, visual)
            loss_sum += float(nn.functional.cross_entropy(output, target, reduction="sum").item())
            logits.append(output.cpu())
            labels.append(target.cpu())
    logits, labels = torch.cat(logits), torch.cat(labels).numpy()
    if not bool(torch.isfinite(logits).all()) or not np.isfinite(loss_sum):
        raise ValueError("Non-finite Dev logits/loss")
    prob = torch.softmax(logits, dim=1).numpy()
    pred = prob.argmax(1)
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, pred, labels=list(range(6)), zero_division=0)
    results = {**metrics(labels, pred), "loss": loss_sum / len(labels),
               "per_class": [{"class_id": k, "class_name": name, "precision": float(precision[k]),
                              "recall": float(recall[k]), "f1": float(f1[k]), "support": int(support[k])}
                             for k, name in enumerate(LABEL_NAMES)]}
    return results, prob, np.concatenate(gates) if gates else None


def _git(*args):
    return subprocess.run(["git", *args], cwd=PROJECT_ROOT, check=True, capture_output=True,
                          text=True, encoding="utf-8").stdout.rstrip()


def run(args):
    # Preflight all supplied input paths before any config/cache IO.
    for path in (args.config, args.cache_dir, args.v_quadrant_predictions):
        if path is not None:
            guard_input(path)
    if args.split != "dev":
        raise ValueError("Only Dev evaluation allowed; Test is forbidden")
    config = load_protocol(args)
    data, normalization, cache_manifest, input_records = load_cache(args.cache_dir, synthetic=args.synthetic)
    input_records.append(identity(args.config))
    if args.v_quadrant_predictions is not None:
        previous = cache_manifest.get("v_quadrant_predictions")
        record = identity(args.v_quadrant_predictions)
        if previous is not None and record["sha256"] != previous["sha256"]:
            raise ValueError("Explicit predictions differ from cached frozen Linear-B source")
        data["dev"], probe = attach_probe(data["dev"], args.v_quadrant_predictions)
        input_records.append(record)
    else:
        probe = cache_manifest.get("v_quadrant_predictions")
    if config["vrg"]["enabled"] and probe is None:
        raise ValueError("VRG requires --v-quadrant-predictions or cached frozen Linear-B seed100 predictions")
    prepared = {}
    for split in ("train", "dev"):
        ta, visual = standardize(data[split], normalization)
        if not np.isfinite(ta).all() or not np.isfinite(visual).all():
            raise ValueError("Non-finite standardized features")
        prepared[split] = TensorDataset(torch.from_numpy(ta), torch.from_numpy(visual),
                                        torch.from_numpy(data[split]["label"]))
    training = config["train"]
    set_seed(training["seed"])
    device = torch.device(args.device)
    generator = torch.Generator().manual_seed(training["seed"])
    loaders = {split: DataLoader(dataset, batch_size=training["batch_size"], shuffle=split == "train",
                                 generator=generator if split == "train" else None, num_workers=0)
               for split, dataset in prepared.items()}
    model = build_model(vrg_enabled=config["vrg"]["enabled"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=training["lr"], weight_decay=training["weight_decay"])
    output = reserve_output(args.output_dir, "run")
    config.update(model="vrg" if config["vrg"]["enabled"] else "concat_linear",
                  cache_dir=args.cache_dir.as_posix(), output_dir=output.as_posix(),
                  v_quadrant_predictions=probe, normalization=cache_manifest["normalization"],
                  source_v=cache_manifest["source_v"], source_ta=cache_manifest["source_ta"])
    (output / "resolved_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    manifest = {"status": "RUNNING", "protocol_version": PROTOCOL_VERSION,
                "started_at_utc": datetime.now(timezone.utc).isoformat(), "synthetic": args.synthetic,
                "git_head": _git("rev-parse", "HEAD"), "git_status": _git("status", "--short"),
                "runtime": {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__},
                "model": config["model"], "seed": training["seed"],
                "parameter_count": sum(p.numel() for p in model.parameters()),
                "test_access": False, "test_loader_built": 0, "test_evaluation_count": 0,
                "inputs": input_records, "v_quadrant_source": probe,
                "common_rows": {split: len(prepared[split]) for split in prepared},
                "cache_joins": cache_manifest["joins"],
                "code_sha256": {str(path.relative_to(PROJECT_ROOT)): compute_file_sha256(path) for path in (
                    Path(__file__), PROJECT_ROOT / "models/asymmetric_merc/feature_fusion.py",
                    PROJECT_ROOT / "scripts/models/asymmetric_merc/frozen_tav_cache.py",
                    PROJECT_ROOT / "scripts/analysis/models/asymmetric_merc/vrg_report.py")}}
    write_json(output / "run_manifest.json", manifest)
    epochs, best_row, waiting = [], None, 0
    try:
        for epoch in range(1, training["max_epochs"] + 1):
            model.train()
            loss_sum, count = 0.0, 0
            for ta, visual, labels in loaders["train"]:
                optimizer.zero_grad(set_to_none=True)
                loss = nn.functional.cross_entropy(model(ta.to(device), visual.to(device)), labels.to(device))
                if not bool(torch.isfinite(loss)):
                    raise ValueError("Non-finite Train loss")
                loss.backward()
                if any(p.grad is not None and not bool(torch.isfinite(p.grad).all()) for p in model.parameters()):
                    raise ValueError("Non-finite gradients")
                optimizer.step()
                loss_sum += float(loss.item()) * len(labels)
                count += len(labels)
            dev_metrics, _, _ = evaluate(model, loaders["dev"], device)
            row = {"epoch": epoch, "train_loss": loss_sum / count,
                   **{f"dev_{key}": value for key, value in dev_metrics.items() if key != "per_class"}}
            epochs.append(row)
            if best_row is None or selection_key(row) < selection_key(best_row):
                best_row, waiting = row, 0
                torch.save({"model": model.state_dict(), "model_name": config["model"], "epoch": epoch,
                            "seed": training["seed"], "dev_metrics": dev_metrics,
                            "cache_manifest_sha256": input_records[0]["sha256"]}, output / "best_model.pt")
            else:
                waiting += 1
            write_csv(output / "epoch_metrics.csv", epochs)
            if waiting >= training["patience"]:
                break
        checkpoint = torch.load(output / "best_model.pt", map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["model"], strict=True)
        best_metrics, prob, gates = evaluate(model, loaders["dev"], device)
        best_metrics["best_epoch"] = checkpoint["epoch"]
        write_json(output / "best_dev_metrics.json", best_metrics)
        dev = data["dev"]
        predictions = [{"utterance_id": str(uid), "dialogue_id": str(dev["dialogue_id"][i]), "split": "dev",
                        "true_label": int(dev["label"][i]), "predicted_label": int(prob[i].argmax()),
                        **{f"prob_{k}": float(prob[i, k]) for k in range(6)}}
                       for i, uid in enumerate(dev["utterance_id"])]
        write_csv(output / "best_dev_predictions.csv", predictions)
        write_json(output / "ta_common_dev_metrics.json", metrics(dev["label"], dev["ta_prob"].argmax(1)))
        if config["vrg"]["enabled"]:
            write_gate_report(output, dev, prob.argmax(1), gates, source=V_PRED_SOURCE, synthetic=args.synthetic)
        assert_unchanged(input_records)
        manifest.update(status="complete", best_epoch=checkpoint["epoch"], best_dev=best_metrics,
                        completed_epochs=len(epochs), stopped_early=waiting >= training["patience"],
                        artifacts={path.name: compute_file_sha256(path) for path in output.iterdir()
                                   if path.is_file() and path.name != "run_manifest.json"})
    except Exception as error:
        manifest.update(status="FAILED", error=str(error))
        raise
    finally:
        write_json(output / "run_manifest.json", manifest)
    return manifest


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        result = run(args)
    except (ValueError, OSError, KeyError, TypeError, RuntimeError) as error:
        print(f"FUSION FAILED: {error}", file=sys.stderr)
        return 1
    print({"status": result["status"], "model": result["model"], "best_epoch": result["best_epoch"],
           "synthetic": result["synthetic"]})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

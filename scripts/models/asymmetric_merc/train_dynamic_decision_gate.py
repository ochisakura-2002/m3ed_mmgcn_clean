"""Train one probability-only seed100 gate; fail closed before any Test input IO."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import subprocess
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import yaml

from models.asymmetric_merc.dynamic_decision_gate import ConstantDecisionGate, build_model
from scripts.analysis.models.asymmetric_merc.dynamic_decision_report import (
    CRITERIA, LIMITATION, write_decision_report,
)
from scripts.analysis.models.asymmetric_merc.fusion_report import write_csv, write_json
from scripts.analysis.models.asymmetric_merc.weighted_fusion import (
    ISOLATION, exact_join, fusion_scores, load_predictions, metrics,
)
from utils.output_paths import resolve_output_paths
from utils.run_metadata import compute_file_sha256
from utils.seed import set_seed

DEFAULT_CONFIG = PROJECT_ROOT / "configs/asymmetric_merc/dynamic_decision_gate/seed100.yaml"
PROTOCOL_VERSION = "asymmetric_merc_dynamic_decision_gate_v0"
METRIC_KEYS = ("accuracy", "weighted_f1", "macro_f1", "uar")
METRIC_ATOL = 1e-10
FROZEN_TRAIN = {"seed": 100, "loss": "CrossEntropy", "optimizer": "AdamW", "lr": 1e-3,
                "weight_decay": 1e-2, "batch_size": 128, "max_epochs": 100, "patience": 10,
                "train_shuffle": True, "dev_shuffle": False,
                "selection": "dev_weighted_f1_strict_improvement_earliest_epoch"}
INPUTS = ("ta_train", "ta_dev", "v_train", "v_dev", "b0_predictions", "b0_metrics")


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in INPUTS[:4]:
        parser.add_argument("--" + name.replace("_", "-"), required=True, type=Path)
    parser.add_argument("--b0-predictions", type=Path, help="Frozen Q2-2 fusion/seed100_predictions.csv (score_0..5)")
    parser.add_argument("--b0-metrics", type=Path, help="Frozen Q2-2 fusion/seed100_metrics.json (full precision)")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, required=True, help="New run directory; no overwrite/resume")
    parser.add_argument("--split", choices=("dev",), default="dev")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--synthetic", action="store_true", help="Artificial probability fixtures only")
    parser.add_argument("--smoke-epochs", type=int, help="1..3; requires --synthetic")
    return parser


def guard_path(path):
    path = Path(path)
    for candidate in (path, path.resolve()):
        parts = re.split(r"[\\/]", str(candidate))
        if any(re.search(r"(^|[^a-z0-9])test(?:ing)?([^a-z0-9]|$)", part.lower()) for part in parts):
            raise ValueError("Test/testing input or output path is forbidden before IO")
    return path


def preflight(args):
    if args.split != "dev":
        raise ValueError("Only Dev evaluation is allowed; Test is forbidden before IO")
    for name in (*INPUTS, "config", "output_dir"):
        path = getattr(args, name)
        if path is not None:
            guard_path(path)
            if name in INPUTS[:5] and path.suffix.lower() not in {".csv", ".tsv"}:
                raise ValueError("Only CSV/TSV predictions accepted; no features or expert checkpoints")
    if (args.b0_predictions is None) != (args.b0_metrics is None):
        raise ValueError("Supply both frozen B0 prediction and metric artifacts")
    if not args.synthetic and args.b0_predictions is None:
        raise ValueError("Formal mode requires frozen B0 predictions and full-precision metrics; rounded constants are insufficient")
    if args.smoke_epochs is not None and (not args.synthetic or not 1 <= args.smoke_epochs <= 3):
        raise ValueError("--smoke-epochs requires --synthetic and 1..3 epochs")


def load_protocol(args):
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if (not isinstance(config, dict)
            or set(config) != {"protocol_version", "splits", "dynamic_decision_gate", "train"}
            or config["protocol_version"] != PROTOCOL_VERSION or config["splits"] != ["train", "dev"]
            or config["train"] != FROZEN_TRAIN or not isinstance(config["dynamic_decision_gate"], dict)
            or set(config["dynamic_decision_gate"]) != {"enabled"}
            or type(config["dynamic_decision_gate"]["enabled"]) is not bool):
        raise ValueError("Frozen seed100 Train/Dev protocol mismatch")
    config.update(synthetic=args.synthetic, device=args.device, evaluation_split="dev", dtype="float64")
    config["smoke_epochs"] = args.smoke_epochs
    return config


@dataclass
class CheckpointSelection:
    """Epoch zero is a real candidate. No loss or tolerance tie-break."""

    best_epoch: int
    best_weighted_f1: float
    waiting: int = 0

    def consider(self, epoch, weighted_f1):
        if not np.isfinite(weighted_f1):
            raise ValueError("Non-finite Dev Weighted-F1")
        improved = weighted_f1 > self.best_weighted_f1
        if improved:
            self.best_epoch, self.best_weighted_f1, self.waiting = epoch, weighted_f1, 0
        else:
            self.waiting += 1
        return improved

    def should_stop(self, patience=10):
        return self.waiting >= patience


def reserve_output(path):
    resolved = path.resolve()
    try:
        parts = resolved.relative_to(PROJECT_ROOT / "outputs").parts
    except ValueError:
        parts = None
    if parts is not None:
        if len(parts) != 4 or parts[2] != "runs":
            raise ValueError("Output must be outputs/<date>/<group>/runs/<run_id>")
        expected = resolve_output_paths(output_base=PROJECT_ROOT / "outputs", experiment_date=parts[0],
                                        experiment_group=parts[1], run_id=parts[3]).run_root
        if expected != resolved:
            raise ValueError("Non-canonical run output path")
    resolved.mkdir(parents=True, exist_ok=False)
    return resolved


def evaluate(model, loader, device):
    model.eval()
    scores, gates, labels = [], [], []
    loss_sum = 0.0
    with torch.no_grad():
        for ta, visual, true in loader:
            result = model(ta.to(device), visual.to(device), return_details=True)
            score = result["scores"]
            loss_sum += float(nn.functional.cross_entropy(score, true.to(device), reduction="sum"))
            scores.append(score.cpu())
            gates.append(result["g"].cpu())
            labels.append(true)
    scores, gates = torch.cat(scores).numpy(), torch.cat(gates).numpy().reshape(-1)
    if not np.isfinite(scores).all() or not np.isfinite(gates).all() or not np.isfinite(loss_sum):
        raise ValueError("Non-finite Dev score, gate or loss")
    return {**metrics(torch.cat(labels).numpy(), scores.argmax(1)), "loss": loss_sum / len(gates)}, scores, gates


def epoch0_reproduction(joined, epoch0, scores, gates, constant, reference=None, reference_metrics=None,
                        *, synthetic=False):
    """No rounding and no fallback to the observed metrics as a formal reference."""
    fixed = fusion_scores(joined, .88)
    fixed_pred = fixed.argmax(1)
    fixed_metrics = metrics(joined.true, fixed_pred)
    checks = {
        "dev_count": synthetic or len(joined.ids) == 978,
        "epoch0_vs_fixed_predictions_exact": np.array_equal(scores.argmax(1), fixed_pred),
        "constant_vs_epoch0_scores_exact": np.array_equal(constant, scores),
        "initial_gate": bool(np.allclose(gates, .12, atol=1e-15, rtol=0)),
        "initial_gate_std": float(gates.std()) <= 1e-15,
    }
    differences = {key: abs(epoch0[key] - fixed_metrics[key]) for key in METRIC_KEYS}
    checks["epoch0_vs_fixed_metrics"] = all(value <= METRIC_ATOL for value in differences.values())
    if reference is not None and reference_metrics is not None:
        checks.update(reference_ids_exact=reference.ids == joined.ids,
                      reference_labels_exact=np.array_equal(reference.true, joined.true),
                      reference_predictions_exact=np.array_equal(reference.predicted, scores.argmax(1)))
        measured = metrics(reference.true, reference.predicted)
        artifact_differences = {key: abs(epoch0[key] - reference_metrics[key]) for key in METRIC_KEYS}
        checks["frozen_metrics_match_reference_predictions"] = all(
            abs(measured[key] - reference_metrics[key]) <= METRIC_ATOL for key in METRIC_KEYS)
        checks["frozen_metrics_match_epoch0"] = all(value <= METRIC_ATOL for value in artifact_differences.values())
    else:
        checks["frozen_reference_present"] = synthetic
        artifact_differences = None
    return {"status": "PASS" if all(checks.values()) else "ABORT_BEFORE_TRAINING",
            "checks": {key: bool(value) for key, value in checks.items()}, "metric_atol": METRIC_ATOL,
            "common_rows": len(joined.ids), "epoch0_metrics": epoch0, "fixed_alpha_metrics": fixed_metrics,
            "metric_differences": differences, "frozen_artifact_metric_differences": artifact_differences,
            "max_abs_score_difference_vs_fixed": float(np.abs(scores - fixed).max()),
            "gate_mean": float(gates.mean()), "gate_std": float(gates.std()),
            "reference_mode": "frozen_Q2_2_artifacts" if reference is not None else "synthetic_formula_only",
            "synthetic": synthetic, **ISOLATION}


def _git(*args):
    return subprocess.run(["git", *args], cwd=PROJECT_ROOT, capture_output=True, check=True,
                          text=True, encoding="utf-8").stdout.rstrip()


def _record(path):
    return {"path": path.as_posix(), "sha256": compute_file_sha256(path)}


def _assert_unchanged(records):
    if any(compute_file_sha256(Path(row["path"])) != row["sha256"] for row in records):
        raise ValueError("Frozen input changed during run")


def _reference(args):
    if args.b0_predictions is None:
        return None, None, []
    records = [_record(args.b0_predictions), _record(args.b0_metrics)]
    reference = load_predictions(args.b0_predictions, score_columns=True)
    artifact = json.loads(args.b0_metrics.read_text(encoding="utf-8"))
    if artifact.get("alpha") != .88 or artifact.get("common_rows") != len(reference.ids):
        raise ValueError("Frozen B0 artifact alpha/count mismatch")
    values = artifact.get("fusion", {})
    if any(type(values.get(key)) not in {int, float} or not np.isfinite(values[key])
           or not 0 <= values[key] <= 1 for key in METRIC_KEYS):
        raise ValueError("Frozen B0 artifact requires all four full-precision fusion metrics")
    if not args.synthetic and artifact.get("synthetic") is not False:
        raise ValueError("Formal frozen B0 reference must declare synthetic=false")
    _assert_unchanged(records)
    return reference, values, records


def run(args):
    preflight(args)  # Every explicit path/split is checked before opening any input.
    config_record = _record(args.config)
    config = load_protocol(args)
    tables = {name: load_predictions(getattr(args, name), split="train" if name.endswith("train") else "dev")
              for name in INPUTS[:4]}
    joined = {split: exact_join(tables[f"ta_{split}"], tables[f"v_{split}"]) for split in ("train", "dev")}
    train_ids = set(tables["ta_train"].ids) | set(tables["v_train"].ids)
    dev_ids = set(tables["ta_dev"].ids) | set(tables["v_dev"].ids)
    if train_ids & dev_ids:
        raise ValueError("Train/Dev utterance ID overlap")
    if not args.synthetic and any(len(joined[split].ids) != n for split, n in (("train", 4818), ("dev", 978))):
        raise ValueError("Formal common subset must be Train=4818, Dev=978")
    reference, reference_metrics, reference_records = _reference(args)
    records = [config_record, *(table.manifest for table in tables.values()), *reference_records]
    _assert_unchanged(records)
    set_seed(100)
    device = torch.device(args.device)
    loaders = {}
    for split, data in joined.items():
        dataset = TensorDataset(torch.from_numpy(data.ta_probabilities), torch.from_numpy(data.v_probabilities),
                                torch.from_numpy(data.true))
        loaders[split] = DataLoader(dataset, batch_size=128, shuffle=split == "train", num_workers=0,
                                   generator=torch.Generator().manual_seed(100) if split == "train" else None)
    model = build_model(enabled=config["dynamic_decision_gate"]["enabled"]).to(device)
    output = reserve_output(args.output_dir)
    config.update(inputs={name: str(getattr(args, name)) for name in INPUTS if getattr(args, name) is not None},
                  output_dir=output.as_posix())
    (output / "resolved_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    manifest = {"status": "RUNNING", "workstream": "ASYMMETRIC_MERC", "protocol_version": PROTOCOL_VERSION,
                "git_head": _git("rev-parse", "HEAD"), "git_status": _git("status", "--short"),
                "started_at_utc": datetime.now(timezone.utc).isoformat(), "synthetic": args.synthetic,
                "runtime": {"torch": torch.__version__, "numpy": np.__version__, "python": sys.version},
                "seed": 100, "parameter_count": sum(p.numel() for p in model.parameters()),
                "base_train_predictions": "in_sample", "base_prediction_training_status": {"TA": "in_sample", "V": "in_sample"},
                "cross_fitted": False, "limitation": LIMITATION, "training": FROZEN_TRAIN,
                "sources": {"TA": "TA-v0 seed100 epoch18 train_best_predictions.tsv / validation_best_predictions.tsv",
                            "V": "Frozen Linear-B seed100 predictions_train.csv / predictions_dev.csv",
                            "identity_evidence": "caller declarations; CSV alone does not prove checkpoint identity"},
                "inputs": records, "joins": {split: data.audit for split, data in joined.items()},
                "probabilities_renormalized": False, "dtype": "float64", "criteria": CRITERIA,
                "code_sha256": {str(path.relative_to(PROJECT_ROOT)): compute_file_sha256(path) for path in (
                    Path(__file__), PROJECT_ROOT / "models/asymmetric_merc/dynamic_decision_gate.py",
                    PROJECT_ROOT / "scripts/analysis/models/asymmetric_merc/dynamic_decision_report.py",
                    PROJECT_ROOT / "scripts/analysis/models/asymmetric_merc/weighted_fusion.py",
                    PROJECT_ROOT / "scripts/analysis/models/asymmetric_merc/vrg_report.py",
                    PROJECT_ROOT / "utils/metrics.py")}, **ISOLATION}
    write_json(output / "run_manifest.json", manifest)
    try:
        epoch0, scores, gates = evaluate(model, loaders["dev"], device)
        _, constant, _ = evaluate(ConstantDecisionGate().to(device), loaders["dev"], device)
        reproduction = epoch0_reproduction(joined["dev"], epoch0, scores, gates, constant,
                                           reference, reference_metrics, synthetic=args.synthetic)
        write_json(output / "epoch0_reproduction.json", reproduction)
        if reproduction["status"] != "PASS":
            raise ValueError("ABORT BEFORE TRAINING: epoch0 frozen B0 reproduction failed")
        selection = CheckpointSelection(0, epoch0["weighted_f1"])

        def save_best(epoch, values):
            torch.save({"model": model.state_dict(), "epoch": epoch, "dev_metrics": values, "seed": 100,
                        "model_enabled": config["dynamic_decision_gate"]["enabled"], "protocol_version": PROTOCOL_VERSION},
                       output / "best_model.pt")

        def row(epoch, values, gate, train_loss):
            return {"epoch": epoch, "train_loss": train_loss,
                    **{f"dev_{key}": value for key, value in values.items()},
                    "gate_mean": float(gate.mean()), "gate_std": float(gate.std())}

        save_best(0, epoch0)
        epochs = [row(0, epoch0, gates, None)]
        write_csv(output / "epoch_metrics.csv", epochs)
        enabled = config["dynamic_decision_gate"]["enabled"]
        # No optimizer is constructed until the complete reproduction gate passes.
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2) if enabled else None
        maximum = args.smoke_epochs if args.smoke_epochs is not None else 100
        train_epochs = maximum if enabled else 0
        for epoch in range(1, train_epochs + 1):
            model.train()
            loss_sum = 0.0
            for ta, visual, true in loaders["train"]:
                optimizer.zero_grad(set_to_none=True)
                loss = nn.functional.cross_entropy(model(ta.to(device), visual.to(device)), true.to(device))
                if not bool(torch.isfinite(loss)):
                    raise ValueError("Non-finite Train loss")
                loss.backward()
                if any(p.grad is not None and not bool(torch.isfinite(p.grad).all()) for p in model.parameters()):
                    raise ValueError("Non-finite gate gradient")
                optimizer.step()
                loss_sum += float(loss.detach()) * len(true)
            values, _, gate = evaluate(model, loaders["dev"], device)
            epochs.append(row(epoch, values, gate, loss_sum / len(loaders["train"].dataset)))
            if selection.consider(epoch, values["weighted_f1"]):
                save_best(epoch, values)
            write_csv(output / "epoch_metrics.csv", epochs)
            if selection.should_stop():
                break
        checkpoint = torch.load(output / "best_model.pt", map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["model"], strict=True)
        best, scores, gates = evaluate(model, loaders["dev"], device)
        if best["weighted_f1"] < epoch0["weighted_f1"] or best["weighted_f1"] != selection.best_weighted_f1:
            raise ValueError("Selected checkpoint violated the epoch0 Dev floor")
        best["best_epoch"] = checkpoint["epoch"]
        write_json(output / "best_dev_metrics.json", best)
        prob = torch.softmax(torch.from_numpy(scores), dim=1).numpy()
        data = joined["dev"]
        predicted = scores.argmax(1)
        write_csv(output / "best_dev_predictions.csv", [
            {"utterance_id": uid, "split": "dev", "true_label": int(data.true[i]),
             "predicted_label": int(predicted[i]), **{f"prob_{k}": float(prob[i, k]) for k in range(6)}}
            for i, uid in enumerate(data.ids)])
        report = write_decision_report(output, data, predicted, gates, best_metrics=best,
                                       baseline=reproduction["fixed_alpha_metrics"], synthetic=args.synthetic)
        _assert_unchanged(records)
        manifest.update(status="complete", best_epoch=checkpoint["epoch"], best_dev=best,
                        trained_epochs=len(epochs) - 1, stopped_early=selection.should_stop(),
                        optimizer_steps_allowed=enabled, verdict=report["verdict"],
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
    except (ValueError, OSError, KeyError, TypeError, RuntimeError, csv.Error, subprocess.SubprocessError) as error:
        print(f"DECISION_GATE=FAIL: {error}", file=sys.stderr)
        return 1
    print(f"DECISION_GATE=PASS best_epoch={result['best_epoch']} synthetic={result['synthetic']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

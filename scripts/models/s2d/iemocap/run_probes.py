"""Q1 frozen-bank Train/Dev diagnosis. No Test, S2D, audio, fusion or sampling."""

from __future__ import annotations

import argparse
import copy
import math
from pathlib import Path
import platform
import subprocess
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import sklearn
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
import yaml

from models.s2d.iemocap.probes import LinearProbe, TemporalProbe, SEEDS, seed_probe
from scripts.models.s2d.iemocap import probe_bank as bank_api
from scripts.models.s2d.iemocap import probe_reports as reports


class BaselineMismatch(ValueError):
    pass


def resolve_path(path):
    path = Path(path)
    return (PROJECT_ROOT / path).resolve() if not path.is_absolute() else path.resolve()


def validate_config(config):
    if (not isinstance(config, dict) or set(config) != {
            "protocol_version", "enabled", "seeds", "normalization", "train"}
            or config["protocol_version"] != "s2d_q1_probe_v1" or type(config["enabled"]) is not bool
            or config["seeds"] != list(SEEDS)
            or config["normalization"] not in {"none", "train_channel_standardize"}):
        raise ValueError("Invalid Q1 protocol: fixed five seeds, global options only, no split options")
    train = config["train"]
    if not isinstance(train, dict) or set(train) != {
            "learning_rate", "weight_decay", "max_epochs", "patience", "batch_size", "cpu_threads"}:
        raise ValueError("Only one shared training protocol is allowed; no feature-specific tuning")
    for name in ("max_epochs", "patience", "batch_size", "cpu_threads"):
        if type(train[name]) is not int or train[name] < 1:
            raise ValueError(f"Positive integer required: {name}")
    for name in ("learning_rate", "weight_decay"):
        if type(train[name]) not in (int, float) or not math.isfinite(train[name]) or train[name] < 0:
            raise ValueError(f"Invalid {name}")
    if train["learning_rate"] == 0:
        raise ValueError("learning_rate must be positive")
    return copy.deepcopy(config)


def load_config(path):
    return validate_config(yaml.safe_load(resolve_path(path).read_text(encoding="utf-8-sig")))


def protocol_record(config, *, preflight_only):
    return {**config, "preflight_only": preflight_only, "fit_split": "train", "selection_split": "dev",
            "selection_metric": "weighted_f1", "selection_rule": "strict improvement; earliest tie",
            "std_ddof": 1, "singleton_std": 0.0, "loss": "unweighted cross_entropy", "optimizer": "AdamW",
            "linear": {"features": list(bank_api.FEATURES), "pooling": "mean_t", "layer": [768, 6]},
            "tcn": {"features": ["d0", "b"], "projection": [768, 128], "residual_blocks": 2,
                    "convolutions_per_block": 1, "kernel_size": 3, "dilations": [1, 2], "dropout": 0.1,
                    "activation": "GELU", "padding": "left only", "pooling": "mean_t",
                    "purpose": "offline diagnostic temporal probe; B already has non-causal TMA"},
            "normalization_fit": "per representation, Train utterance/slot population moments; epsilon=1e-6",
            "determinism": "fixed seed/init/shuffle/dropout; deterministic kernels required; same-stack only",
            "deployment_boundary": "No causal deployment claim for B, whole-utterance pooling or pixel crops"}


def baseline_gate(report, provenance):
    observed = report["weighted_f1"]
    expected = bank_api.EXPECTED_DEV_WEIGHTED_F1
    difference = observed - expected
    passed = (provenance["selected_epoch"] == bank_api.EXPECTED_EPOCH
              and math.isfinite(observed) and abs(difference) <= bank_api.BASELINE_TOLERANCE)
    return {"selected_epoch": provenance["selected_epoch"], "expected_weighted_f1": expected,
            "observed_weighted_f1": observed, "difference": difference,
            "absolute_difference": abs(difference), "tolerance": bank_api.BASELINE_TOLERANCE,
            "PROBE_TRAINING_ALLOWED": "YES" if passed else "NO", "status": "PASS" if passed else "FAIL"}


@torch.no_grad()
def evaluate(model, features, labels, batch_size, device):
    model.eval()
    logits, loss_sum = [], 0.0
    for start in range(0, len(labels), batch_size):
        value = model(features[start:start + batch_size].to(device))
        loss = F.cross_entropy(value, labels[start:start + batch_size].to(device), reduction="sum")
        if not bool(torch.isfinite(value).all()) or not bool(torch.isfinite(loss)):
            raise FloatingPointError("Non-finite probe evaluation")
        logits.append(value.cpu())
        loss_sum += float(loss)
    logits = torch.cat(logits)
    report = reports.classification_report(labels, logits.argmax(-1))
    report["loss"] = loss_sum / len(labels)
    return report, logits.softmax(-1)


def fit_probe(train_x, train_y, dev_x, dev_y, *, kind, seed, config, device):
    """Only new tiny classifiers receive gradients. Inputs stay frozen on CPU."""
    if kind not in {"linear", "tcn"}:
        raise ValueError("Unknown probe type")
    if any(x.requires_grad for x in (train_x, dev_x)):
        raise ValueError("Probe inputs must be detached frozen bank tensors")
    seed_probe(seed)
    model = (LinearProbe() if kind == "linear" else TemporalProbe()).to(device)
    train = config["train"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=train["learning_rate"], weight_decay=train["weight_decay"])
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(TensorDataset(train_x, train_y), batch_size=train["batch_size"],
                        shuffle=True, drop_last=False, num_workers=0, generator=generator)
    best_score, best_epoch, best_state, stale = -math.inf, None, None, 0
    history = []
    for epoch in range(1, train["max_epochs"] + 1):
        model.train()
        loss_sum = 0.0
        for features, labels in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(features.to(device)), labels.to(device))
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("Non-finite probe training loss")
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach()) * len(labels)
        train_report, _ = evaluate(model, train_x, train_y, train["batch_size"], device)
        dev_report, _ = evaluate(model, dev_x, dev_y, train["batch_size"], device)
        history.append(dict(epoch=epoch, optimization_loss=loss_sum / len(train_y),
                            train_loss=train_report["loss"], dev_loss=dev_report["loss"],
                            **{f"{split}_{metric}": report[metric] for split, report in (
                                ("train", train_report), ("dev", dev_report)) for metric in reports.METRICS}))
        if dev_report["weighted_f1"] > best_score:
            best_score, best_epoch, stale = dev_report["weighted_f1"], epoch, 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            stale += 1
        if stale >= train["patience"]:
            break
    model.load_state_dict(best_state, strict=True)
    best_report, probabilities = evaluate(model, dev_x, dev_y, train["batch_size"], device)
    if best_report["weighted_f1"] != best_score:
        raise RuntimeError("Best probe checkpoint did not reproduce selected Dev score")
    best_report.update(best_epoch=best_epoch, stopped_epoch=epoch,
                       stop_reason="early_stopping" if stale >= train["patience"] else "max_epochs")
    return best_state, best_report, probabilities, history


def run_probes(feature_bank, output_dir, config, *, device="cpu", preflight_only=False):
    config = validate_config(config)  # Unknown split/config options fail before any bank IO.
    if device not in {"cpu", "cuda"}:
        raise ValueError("device must be cpu or cuda")
    output_dir, feature_bank = resolve_path(output_dir), resolve_path(feature_bank)
    if output_dir.exists():
        raise FileExistsError(f"Refusing existing output directory: {output_dir}")
    if feature_bank == output_dir or feature_bank in output_dir.parents or output_dir in feature_bank.parents:
        raise ValueError("Probe outputs and the frozen feature bank must be disjoint")
    preflight_only = preflight_only or not config["enabled"]
    before_threads = torch.get_num_threads()
    torch.set_num_threads(config["train"]["cpu_threads"])
    try:
        bank = bank_api.FeatureBank(feature_bank)
        output_dir.mkdir(parents=True, exist_ok=False)
        provenance_dir = output_dir / "00_provenance"
        provenance_dir.mkdir()
        git_head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT,
                                  check=True, capture_output=True, text=True).stdout.strip()
        reports.atomic_json(provenance_dir / "provenance.json", {
            "created_by": "scripts/models/s2d/iemocap/run_probes.py", "git_head": git_head,
            "python": platform.python_version(), "torch": str(torch.__version__), "sklearn": sklearn.__version__,
            "device": device, "seeds": list(SEEDS), "feature_bank_only": True,
            "source_run_opened": False, "test_accessed": False, "s2d_trained_or_modified": False,
            "code_identities": {path: bank_api.identity(PROJECT_ROOT / path) for path in (
                "models/s2d/iemocap/probes.py", "scripts/models/s2d/iemocap/run_probes.py",
                "scripts/models/s2d/iemocap/probe_bank.py", "scripts/models/s2d/iemocap/probe_reports.py",
                "utils/metrics.py")},
            "mode": "preflight_only" if preflight_only else "all_30_probes"})
        reports.atomic_json(provenance_dir / "feature_bank_reference.json", {
            "path": str(feature_bank), "metadata_identities": bank.metadata_identities,
            "shard_identities": bank.shard_identities, "provenance": bank.provenance,
            "coverage": bank.coverage})
        reports.atomic_json(provenance_dir / "protocol.json", protocol_record(config, preflight_only=preflight_only))
        reference_dir = output_dir / "01_reference"
        reference_dir.mkdir()
        logits = bank.values("dev", "logits_v")
        reference = reports.classification_report(bank.labels["dev"], logits.argmax(-1))
        gate = baseline_gate(reference, bank.provenance)
        reports.atomic_json(reference_dir / "original_s2d_dev_metrics.json", reference)
        reports.write_evaluation(reference_dir, bank.ids["dev"], bank.labels["dev"], logits.softmax(-1),
                                 reference, reference=True)
        reports.atomic_json(reference_dir / "baseline_reproduction_gate.json", gate)
        if gate["status"] != "PASS":
            raise BaselineMismatch(f"PROBE_TRAINING_ALLOWED=NO; Dev Weighted-F1 difference={gate['difference']:.17g}; "
                                   f"expected={gate['expected_weighted_f1']:.17g}, observed={gate['observed_weighted_f1']:.17g}")
        print(f"BASELINE_REPRODUCTION_GATE=PASS; Dev W-F1={reference['weighted_f1']:.17g}", flush=True)

        rows, per_class, histories, evaluations = [], [], [], {}

        def record(model, feature, kind, seed, report):
            rows.append(dict(model=model, feature=feature, probe_type=kind, seed=seed,
                             best_epoch=report.get("best_epoch", bank_api.EXPECTED_EPOCH if kind == "reference" else ""),
                             **{key: report[key] for key in reports.METRICS}))
            per_class.extend(dict(model=model, feature=feature, probe_type=kind, seed=seed, **r)
                             for r in report["per_class"])
            evaluations[f"{model}_seed{seed}" if seed != "" else model] = report

        record("OriginalS2D", "h_v", "reference", "", reference)
        majority = int(torch.bincount(bank.labels["train"], minlength=6).argmax())
        dummy_prob = torch.zeros(len(bank.ids["dev"]), 6)
        dummy_prob[:, majority] = 1
        dummy = reports.classification_report(bank.labels["dev"], dummy_prob.argmax(-1))
        dummy_dir = output_dir / "02_dummy"
        reports.write_evaluation(dummy_dir, bank.ids["dev"], bank.labels["dev"], dummy_prob, dummy)
        reports.atomic_json(dummy_dir / "metrics.json", dummy)
        reports.atomic_json(dummy_dir / "config.json", {"strategy": "most_frequent", "fit_split": "train",
                                                        "class_index": majority, "tie_break": "lowest index",
                                                        "train_counts": torch.bincount(bank.labels["train"], minlength=6).tolist()})
        record("MostFrequent", "none", "dummy", "", dummy)
        if not preflight_only:
            if device == "cuda" and not torch.cuda.is_available():
                raise RuntimeError("Requested CUDA is unavailable; no CPU fallback")
            normalization_dir = provenance_dir / "normalization"
            normalization_dir.mkdir()
            # Same protocol and Train transform for each representation/type.
            for feature, field in bank_api.FEATURES.items():
                stats = (bank.normalization(feature) if config["normalization"] != "none" else {
                    "feature": feature, "field": field, "fit_split": "none", "mean": torch.zeros(768), "std": torch.ones(768)})
                reports.atomic_json(normalization_dir / f"{feature}.json", {
                    **{k: v for k, v in stats.items() if k not in {"mean", "std"}},
                    "mean": stats["mean"].tolist(), "std": stats["std"].tolist(),
                    "train_index_identity": bank.metadata_identities["train/index.json"]})
                for kind in (("linear", "tcn") if feature in {"d0", "b"} else ("linear",)):
                    values = {split: (bank.values(split, field, pooled=kind == "linear") - stats["mean"]) / stats["std"]
                              for split in bank_api.SPLITS}
                    model_name = f"{'L' if kind == 'linear' else 'T'}-{feature.upper()}"
                    for seed in SEEDS:
                        destination = output_dir / ("03_linear" if kind == "linear" else "04_tcn") / feature / f"seed{seed}"
                        destination.mkdir(parents=True)
                        reports.atomic_json(destination / "config.json", {
                            "model": model_name, "feature": feature, "probe_type": kind, "seed": seed,
                            "protocol": config, "normalization": f"00_provenance/normalization/{feature}.json",
                            "normalization_identity": bank_api.identity(normalization_dir / f"{feature}.json"),
                            "selection_split": "dev", "selection_metric": "weighted_f1"})
                        state, best, probabilities, history = fit_probe(
                            values["train"], bank.labels["train"], values["dev"], bank.labels["dev"],
                            kind=kind, seed=seed, config=config, device=device)
                        reports.atomic_write(destination / "best_model.pt", lambda stream: torch.save(state, stream))
                        reports.atomic_json(destination / "metrics_best.json", best)
                        reports.atomic_csv(destination / "metrics_history.csv", history, history[0].keys())
                        reports.write_evaluation(destination, bank.ids["dev"], bank.labels["dev"], probabilities, best)
                        record(model_name, feature, kind, seed, best)
                        histories.extend(dict(model=model_name, feature=feature, seed=seed, **row) for row in history)
                        print(f"{model_name} seed={seed} best_epoch={best['best_epoch']} Dev W-F1={best['weighted_f1']:.6f}", flush=True)
                    del values
        if not preflight_only and len(rows) != 32:
            raise RuntimeError("Incomplete five-seed probe matrix")
        reports.write_summaries(output_dir / "05_summary", rows, per_class, histories, evaluations)
        bank.assert_unchanged()
        # Hash the finished archive before publishing its LAST completion marker.
        artifacts = {p.relative_to(output_dir).as_posix(): bank_api.identity(p)
                     for p in sorted(output_dir.rglob("*")) if p.is_file()}
        reports.atomic_json(output_dir / "complete.json", {
            "schema_version": "s2d_q1_probe_archive_v1", "status": "complete", "mode": "preflight_only" if preflight_only else "all_30_probes",
            "PROBE_TRAINING_ALLOWED": "YES", "trainable_runs": len(rows) - 2, "test_accessed": False,
            "artifacts": artifacts})
        return gate
    finally:
        torch.set_num_threads(before_threads)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-bank", required=True, help="Explicit frozen bank with exactly Train and Dev")
    parser.add_argument("--output-dir", required=True, help="New outputs/<date>/s2d_q1_probe/analysis/<id> directory")
    parser.add_argument("--config", default="configs/s2d/iemocap/q1_probe.yaml")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--preflight-only", action="store_true", help="Reproduce baseline and dummy; fit no classifier")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    run_probes(args.feature_bank, args.output_dir, load_config(args.config),
               device=args.device, preflight_only=args.preflight_only)


if __name__ == "__main__":
    main()

"""Score the fixed CE tail average on a frozen fold or compare four existing G2 exports."""

import argparse
import csv
from math import isclose
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import torch

from models.multidag_cl.ta_v0 import LABEL_VOCAB
from scripts.analysis.models.multidag_cl.group_isolated_comparison import (
    paired_report, parse_prediction, population, read_export,
)
from scripts.analysis.models.multidag_cl.pairwise_fastkill_metrics import BOOTSTRAP_RESAMPLES, BOOTSTRAP_SEED
from scripts.analysis.models.multidag_cl.post_ph.internal_protocol import write_rows
from scripts.models.multidag_cl.group_isolated import ROOT, build_runtime
from scripts.runtime.multidag_cl_group_isolated.evaluation import summarize
from scripts.runtime.multidag_cl_group_isolated.fixed_tail_average import (
    AVERAGING_EPOCHS, authenticated_run, load_fixed_tail, source_inventory,
)
from scripts.runtime.multidag_cl_group_isolated.protocol import (
    CONFIG_DIR, ENDPOINT, G1_GROUP, GROUP, HOLD, PARAMETERS, SEED, SESSIONS,
    describe, fold_config, load_fold, read_frozen, read_json, validate_hold,
)
from scripts.runtime.multidag_cl_group_isolated.provenance import source_hashes
from scripts.runtime.multidag_cl_paper_reimplementation.checkpoint import save_checkpoint_atomic
from scripts.runtime.multidag_cl_ta_v0.formal import sha256_file, write_json_atomic
from utils.output_paths import resolve_experiment_date, resolve_output_paths


G2_GROUP = "g2_fixed_tail_average_ct_multidag"
G2_SCHEMA = "ct_multidag_g2_fixed_ce_tail_v1"
DEFAULT_CE_ROOT = Path("outputs/20261126") / GROUP
AVERAGED_CHECKPOINT = Path("checkpoints/ce_equal_average_e21_e24_e27_e30.pt")
G2_PREDICTIONS = Path("reports/held_out_predictions.csv")
CE30, CE24, LS30, AVG = "ce_e30", "ce_e24", "ls_e30", "ce_avg"
ARMS = (CE30, CE24, LS30, AVG)
RELIABILITY = ("nll", "brier", "ece10")


def g2_run_id(session):
    return f"ct_multidag_g2_avg_{session.lower()}_s{SEED}"


def source_root(root, group):
    if root.name != group or root.resolve().name != group:
        raise ValueError(f"expected a canonical {group} experiment root")
    return root


def require_new_destination(destination, *sources):
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    if any(destination.resolve().is_relative_to(source.resolve()) for source in sources):
        raise ValueError("G2 destination overlaps a read-only source")


def average_identity(manifest, digest, session, sources):
    return {
        "schema": G2_SCHEMA, "held_out_session": session, "seed": SEED, "run_id": g2_run_id(session),
        "source_run_identity": manifest["run_identity"], "source_manifest_sha256": digest, "sources": sources,
        "averaging_epochs": list(AVERAGING_EPOCHS), "averaging_rule": "elementwise_equal_mean_fp32",
        "parameter_count": PARAMETERS,
    }


def export_scoring(directory, prefix, predictions, expected, *, filename=None):
    summary, rows, bins, detection, risk = summarize(predictions, expected)
    confidence = summary["auxiliary"]["high_confidence"]
    confidence["coverage"] = confidence["support"] / len(rows)
    confidence["error_rate"] = confidence["error_support"] / confidence["support"] if confidence["support"] else None
    confidence["error_share_all"] = confidence["error_support"] / len(rows)
    write_rows(directory / (filename or f"{prefix}_predictions.csv"), rows, tuple(rows[0]))
    write_rows(directory / f"{prefix}_high_confidence_errors.csv",
               (r for r in rows if r["is_error"] and r["confidence"] >= confidence["threshold"]), tuple(rows[0]))
    write_rows(directory / f"{prefix}_ece10_bins.csv", bins, tuple(bins[0]))
    write_rows(directory / f"{prefix}_error_detection.csv", detection,
               ("curve", "fpr", "tpr", "precision", "recall", "threshold"))
    write_rows(directory / f"{prefix}_risk_coverage.csv", risk, tuple(risk[0]))
    return summary, risk


def evaluate(ce_root, destination, session, projection_dir, *, device):
    ce_root = source_root(ce_root, GROUP)
    require_new_destination(destination, ce_root, projection_dir)
    config = fold_config(session)
    source = ce_root / "runs" / config["run_id"]
    frozen, raw = read_frozen(ROOT)
    manifest, manifest_sha = authenticated_run(source, config, frozen, raw)
    views, entries, data = load_fold(config, frozen, projection_dir)
    if data != manifest["data_identity"]:
        raise ValueError("certified projection and CE manifest disagree on fold identity")
    runtime, _ = build_runtime(raw, {"train": views["fit"], "dev": views["held_out"]}, device)

    started = perf_counter()
    state, sources = load_fixed_tail(source, manifest, runtime.model)
    if device == "cuda":
        torch.cuda.synchronize()
    averaging_seconds = perf_counter() - started
    started = perf_counter()
    with torch.inference_mode():
        scored = runtime.evaluate(split="dev", include_predictions=True, prediction_details=True)
    if device == "cuda":
        torch.cuda.synchronize()
    inference_seconds = perf_counter() - started
    if runtime.optimizer_steps or runtime.optimizer.state or any(p.grad is not None for p in runtime.model.parameters()):
        raise ValueError("G2 evaluation performed an optimizer operation or accumulated gradients")
    if sha256_file(source / "manifests" / "run_manifest.json") != manifest_sha:
        raise ValueError("CE run manifest changed during G2 evaluation")

    identity = average_identity(manifest, manifest_sha, session, sources)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".g2_fold_", dir=destination.parent) as temporary:
        stage = Path(temporary)
        (stage / "reports").mkdir()
        (stage / "manifests").mkdir()
        save_checkpoint_atomic(stage / AVERAGED_CHECKPOINT, {"identity": identity, "model_state_dict": state})
        summary, _ = export_scoring(stage / "reports", "held_out", scored["predictions"], entries["held_out"],
                                    filename=G2_PREDICTIONS.name)
        write_json_atomic(stage / "reports" / "held_out_metrics.json", summary)
        config_files = (ROOT / CONFIG_DIR / f"{session.lower()}_seed100.yaml", ROOT / CONFIG_DIR / "fold_identities.json",
                        ROOT / raw["frontend_source"]["config_path"])
        write_json_atomic(stage / "manifests" / "evaluation_manifest.json", {
            **HOLD, "TEST_TOUCHED": "NO", "identity": identity, "data_identity": data,
            "new_train_runs": 0, "optimizer_steps_performed": 0, "scored_populations": ["held_out"],
            "calibration": None, "device": device, "torch_version": str(torch.__version__),
            "timing_seconds": {"checkpoint_read_average_strict_load": averaging_seconds,
                               "held_out_inference_and_prediction_collection": inference_seconds},
            "scoring_source_sha256": source_hashes(ROOT, config_files),
            "files_sha256": {p.relative_to(stage).as_posix(): sha256_file(p)
                             for p in sorted(stage.rglob("*")) if p.is_file()},
        })
        require_new_destination(destination, ce_root, projection_dir)
        stage.rename(destination)
    return destination


def read_average(run, ce_run, ce_manifest, ce_sha, session, expected):
    path = run / "manifests" / "evaluation_manifest.json"
    manifest = read_json(path)
    validate_hold(manifest)
    identity = average_identity(ce_manifest, ce_sha, session, source_inventory(ce_run, ce_manifest))
    if (manifest["identity"] != identity or manifest["data_identity"] != ce_manifest["data_identity"]
            or manifest["calibration"] is not None or manifest["new_train_runs"] != 0
            or manifest["optimizer_steps_performed"] != 0 or manifest["scored_populations"] != ["held_out"]
            or manifest["TEST_TOUCHED"] != "NO"):
        raise ValueError(f"G2 export does not match the frozen CE run for {session}")
    prediction_path = run / G2_PREDICTIONS
    if sha256_file(prediction_path) != manifest["files_sha256"][G2_PREDICTIONS.as_posix()]:
        raise ValueError(f"G2 prediction SHA-256 mismatch for {session}")
    with prediction_path.open(encoding="utf-8", newline="") as stream:
        rows = [parse_prediction(row) for row in csv.DictReader(stream)]
    if population(rows) != expected:
        raise ValueError(f"G2 output differs from canonical held-out IDs/order/gold for {session}")
    return rows, {"evaluation_manifest_sha256": sha256_file(path), "files_sha256": manifest["files_sha256"],
                  "timing_seconds": manifest["timing_seconds"], "device": manifest["device"]}


def read_fold(ce_root, g1_root, g2_root, session, frozen, raw):
    evidence = {}
    exports = {}
    for arm, root, g1 in ((CE30, ce_root, False), (LS30, g1_root, True)):
        config = fold_config(session, g1=g1)
        run = root / "runs" / config["run_id"]
        manifest, digest = authenticated_run(run, config, frozen, raw)
        endpoint, trajectory, exported_run, tape, evidence[arm] = read_export(root, config, frozen, raw)
        if exported_run != manifest or evidence[arm]["run_manifest_sha256"] != digest:
            raise ValueError(f"inconsistent authenticated {arm} manifest for {session}")
        exports[arm] = (endpoint, trajectory[24], manifest, tape, digest)
    ce30, ce24, ce, ce_tape, digest = exports[CE30]
    ls30, _, ls, ls_tape, _ = exports[LS30]
    if ce["data_identity"] != ls["data_identity"] or population(ce30) != population(ls30):
        raise ValueError(f"CE and LS held-out populations differ for {session}")
    evidence["ce_ls_fit_match"] = {"initialization": ce["initialization"] == ls["initialization"],
                                   "batch_tape": ce_tape == ls_tape}
    ce_run = ce_root / "runs" / ce["run_id"]
    avg, evidence[AVG] = read_average(g2_root / "runs" / g2_run_id(session), ce_run, ce, digest, session, population(ce30))
    return dict(zip(ARMS, (ce30, ce24, ls30, avg))), evidence


def report_scope(directory, arms, expected):
    directory.mkdir()
    metrics, risks = {}, {}
    for arm, rows in arms.items():
        metrics[arm], risks[arm] = export_scoring(directory, arm, rows, expected)
    comparisons = {}
    for control in (CE30, CE24, LS30):
        report, pairs = paired_report(arms[control], arms[AVG])
        del report["ctrl"], report["g1"]
        before, after = (metrics[key]["auxiliary"] for key in (control, AVG))
        report["reliability_delta"] = {key: after[key] - before[key] for key in RELIABILITY}
        accepted = before["high_confidence"]["support"]
        report["risk_at_control_high_confidence_coverage"] = (
            {"accepted": accepted, "coverage": accepted / len(expected),
             control: risks[control][accepted - 1]["tie_expected_risk"],
             AVG: risks[AVG][accepted - 1]["tie_expected_risk"]} if accepted else None
        )
        report["per_class_r_d_net"] = {}
        for label_id, label in enumerate(LABEL_VOCAB):
            selected = [row for row in pairs if row["true_label_id"] == label_id]
            recovered = sum(row["recovered"] for row in selected)
            damaged = sum(row["damaged"] for row in selected)
            report["per_class_r_d_net"][label] = {"support": len(selected), "r": recovered,
                                                   "d": damaged, "net": recovered - damaged}
        for pair in pairs:
            pair["ce_avg_predicted_label_id"] = pair.pop("g1_predicted_label_id")
        write_rows(directory / f"{AVG}_vs_{control}_paired_utterances.csv", pairs, tuple(pairs[0]))
        comparisons[control] = report
    return {"arms": metrics, "avg_vs": comparisons}


def decision(summary):
    def at_least(delta, threshold):
        return delta >= threshold or isclose(delta, threshold, rel_tol=0.0, abs_tol=1e-10)

    pooled = summary["pooled_oof"]
    primary = pooled["avg_vs"][CE30]
    delta = primary["delta_pp"]
    folds = {session: summary[session]["avg_vs"][CE30]["delta_pp"]["weighted_f1"] for session in SESSIONS}
    e24_delta = pooled["avg_vs"][CE24]["delta_pp"]["weighted_f1"]
    passes = {"pooled_wf1_at_least_0_50pp": at_least(delta["weighted_f1"], 0.50),
              "at_least_three_positive_folds": sum(value > 0 for value in folds.values()) >= 3,
              "no_fold_below_minus_1pp": at_least(min(folds.values()), -1.00),
              "macro_f1_at_least_minus_0_30pp": at_least(delta["macro_f1"], -0.30),
              "uar_at_least_minus_0_30pp": at_least(delta["uar"], -0.30),
              "vs_ce24_wf1_at_least_minus_0_30pp": at_least(e24_delta, -0.30)}
    nll_delta = primary["reliability_delta"]["nll"]
    if all(passes.values()):
        status = "DEVELOPMENT_GO"
    elif delta["weighted_f1"] <= 0.0 and not at_least(-nll_delta, 0.03):
        status = "STOP"
    else:
        status = "REVIEW_SIGNAL"
    class_flags = [old["label"] for old, new in zip(pooled["arms"][CE30]["per_class"],
                                                     pooled["arms"][AVG]["per_class"])
                   if old["support"] >= 100 and at_least(old["recall"] - new["recall"], 0.02)]
    ls_delta = pooled["avg_vs"][LS30]
    return {"status": status, "classification_gates": passes, "fold_wf1_delta_pp": folds,
            "trajectory_or_stopping_signal": delta["weighted_f1"] > 0 and at_least(0.30, abs(e24_delta)),
            "class_recall_review": class_flags, "ce_e30_reliability_delta": primary["reliability_delta"],
            "ls_e30_gap_pp": ls_delta["delta_pp"], "ls_e30_reliability_delta": ls_delta["reliability_delta"],
            "final_strong_baseline_selected": False}


def compare(ce_root, g1_root, g2_root, destination):
    ce_root, g1_root = source_root(ce_root, GROUP), source_root(g1_root, G1_GROUP)
    require_new_destination(destination, ce_root, g1_root, g2_root / "runs")
    frozen, raw = read_frozen(ROOT)
    evidence, summary = {}, {}
    pooled = {arm: [] for arm in ARMS}
    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".g2_comparison_", dir=destination.parent) as temporary:
        stage = Path(temporary)
        for session in SESSIONS:
            arms, evidence[session] = read_fold(ce_root, g1_root, g2_root, session, frozen, raw)
            expected = population(arms[CE30])
            if describe(expected) != frozen["folds"][session]["held_out"]:
                raise ValueError(f"incorrect frozen held-out population for {session}")
            summary[session] = report_scope(stage / session, arms, expected)
            for arm, rows in arms.items():
                pooled[arm].extend(rows)
        expected = population(pooled[CE30])
        if len({uid for uid, _, _ in expected}) != len(expected):
            raise ValueError("pooled OOF contains duplicate utterance IDs")
        summary["pooled_oof"] = report_scope(stage / "pooled_oof", pooled, expected)
        write_json_atomic(stage / "summary.json", summary)
        write_json_atomic(stage / "decision.json", decision(summary))
        timings = {session: evidence[session][AVG]["timing_seconds"] for session in SESSIONS}
        write_json_atomic(stage / "comparison_manifest.json", {
            **HOLD, "TEST_TOUCHED": "NO", "new_train_runs": 0, "optimizer_steps_performed": 0,
            "parameter_count": PARAMETERS, "averaging_epochs": list(AVERAGING_EPOCHS),
            "per_fold_seconds": timings,
            "total_seconds": {key: sum(row[key] for row in timings.values()) for key in timings[SESSIONS[0]]},
            "bootstrap": {"resamples": BOOTSTRAP_RESAMPLES, "seed": BOOTSTRAP_SEED,
                          "strata": "session", "unit": "dialogue",
                          "metrics": ["weighted_f1", "accuracy"]},
            "inputs": evidence, "output_sha256": {p.relative_to(stage).as_posix(): sha256_file(p)
                                                   for p in sorted(stage.rglob("*")) if p.is_file()},
        })
        require_new_destination(destination, ce_root, g1_root, g2_root / "runs")
        stage.rename(destination)
    return destination


def main(argv=None):
    cli = argparse.ArgumentParser(description=__doc__)
    commands = cli.add_subparsers(dest="command", required=True)
    score = commands.add_parser("evaluate", help="score one new fixed-tail average")
    score.add_argument("--session", choices=SESSIONS, required=True)
    score.add_argument("--projection-dir", type=Path, required=True)
    score.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    report = commands.add_parser("compare", help="read existing predictions and report four-fold OOF comparisons")
    report.add_argument("--g1-root", type=Path, required=True)
    report.add_argument("--batch-id", required=True)
    for sub in (score, report):
        sub.add_argument("--ce-root", type=Path, default=DEFAULT_CE_ROOT)
        sub.add_argument("--experiment-date", required=True)
    args = cli.parse_args(argv)
    paths = resolve_output_paths(experiment_date=resolve_experiment_date(args.experiment_date),
                                 experiment_group=G2_GROUP, batch_id=getattr(args, "batch_id", None),
                                 run_id=g2_run_id(args.session) if args.command == "evaluate" else None)
    source_root(paths.experiment_root, G2_GROUP)
    if args.command == "evaluate":
        print(evaluate(args.ce_root, paths.run_root, args.session, args.projection_dir, device=args.device))
    else:
        print(compare(args.ce_root, args.g1_root, paths.experiment_root, paths.analysis_root))


if __name__ == "__main__":
    main()

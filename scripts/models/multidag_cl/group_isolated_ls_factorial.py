"""Evaluate the fixed G1 LS tail once per fold and report the G4 factorial."""

import argparse
from pathlib import Path
from pickle import UnpicklingError
from tempfile import TemporaryDirectory
from time import perf_counter

import torch

from scripts.analysis.models.multidag_cl.group_isolated_comparison import parse_prediction, population
from scripts.models.multidag_cl.group_isolated import ROOT, build_runtime
from scripts.models.multidag_cl.group_isolated_tail_average import (
    CE30, LS30, G2_GROUP, export_scoring, require_new_destination, source_root,
)
from scripts.runtime.multidag_cl_fusion.runtime import training_rng_identity
from scripts.runtime.multidag_cl_group_isolated import fixed_ls_factorial as g4
from scripts.runtime.multidag_cl_group_isolated.budget import SCHEMA as G3_SCHEMA, batch_tape, sync
from scripts.runtime.multidag_cl_group_isolated.budget_history import historical_imports, match_tape
from scripts.runtime.multidag_cl_group_isolated.fixed_tail_average import load_fixed_tail, source_inventory
from scripts.runtime.multidag_cl_group_isolated.protocol import (
    G1_GROUP, GROUP, HOLD, SEED, SESSIONS, fold_config, load_fold, read_frozen, validate_hold,
)
from scripts.runtime.multidag_cl_group_isolated.provenance import source_hashes
from scripts.runtime.multidag_cl_paper_reimplementation.checkpoint import save_checkpoint_atomic
from scripts.runtime.multidag_cl_ta_v0.formal import sha256_file, write_json_atomic
from utils.output_paths import resolve_experiment_date, resolve_output_paths


DEFAULT_LS_ROOT = Path("outputs/20261126") / G1_GROUP
AVERAGED_CHECKPOINT = Path("checkpoints/ls_equal_average_e21_e24_e27_e30.pt")
PREDICTIONS = Path("reports/held_out_predictions.csv")
FOLD_MANIFEST = Path("manifests/evaluation_manifest.json")
ZERO_UPDATES = {**HOLD, "TEST_TOUCHED": "NO", "new_train_runs": 0, "optimizer_steps_performed": 0}


def fold_identity(ls, session):
    return {"schema": g4.SCHEMA, "held_out_session": session, "seed": SEED,
            "source_run_identity": ls.manifest["run_identity"], "source_manifest_sha256": ls.digest,
            "sources": source_inventory(ls.run, ls.manifest), "fit_objective": ls.manifest["fit_objective"],
            "averaging_epochs": list(g4.AVERAGING_EPOCHS),
            "averaging_rule": "elementwise_equal_mean_fp32", "parameter_count": ls.manifest["parameter_count"]}


def prepare_fold(ls, evidence, session, frozen, raw, projection_dir, device):
    manifest = ls.manifest
    views, entries, data = load_fold(fold_config(session, g1=True), frozen, projection_dir)
    if data != manifest["data_identity"] or entries["held_out"] != population(ls.predictions):
        raise ValueError(f"certified projection differs from the historical fold: {session}")
    runtime, initialization = build_runtime(raw, {"train": views["fit"], "dev": views["held_out"]}, device)
    rng = training_rng_identity("cpu")
    if (initialization != manifest["initialization"]
            or any(manifest["training_initial_rng"][key] != value for key, value in rng.items())):
        raise ValueError(f"canonical initialization or CPU RNG differs from G1: {session}")
    tape = batch_tape(runtime, entries["fit"], manifest["steps_per_epoch"], manifest["optimizer_steps"])
    if match_tape(ls.run, manifest, tape) != evidence[LS30]["fit_updates_sha256"]:
        raise ValueError(f"G1 fit update log changed during qualification: {session}")

    start = perf_counter()
    state, sources = load_fixed_tail(ls.run, manifest, runtime.model, g1=True)
    sync(device)
    identity = fold_identity(ls, session)
    if sources != identity["sources"] or sha256_file(ls.run / "manifests/run_manifest.json") != ls.digest:
        raise ValueError(f"G1 checkpoint inventory or manifest changed during qualification: {session}")
    return runtime, state, entries["held_out"], identity, perf_counter() - start


def evaluate(ce_root, ls_root, g2_root, destination, projection_dir, frozen, raw, *, device):
    require_new_destination(destination, ce_root, ls_root, g2_root, projection_dir)
    reference = historical_imports()
    controls = {session: g4.read_controls(ce_root, ls_root, g2_root, session, frozen, raw, reference)
                for session in SESSIONS}
    prepared = {}
    for session, (ls, _, evidence) in controls.items():
        prepared[session] = prepare_fold(ls, evidence, session, frozen, raw, projection_dir, device)

    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".g4_eval_", dir=destination.parent) as temporary:
        stage = Path(temporary)
        for session, (runtime, state, expected, identity, averaging_seconds) in prepared.items():
            start = perf_counter()
            with torch.inference_mode():
                result = runtime.evaluate(split="dev", include_predictions=True, prediction_details=True)
            sync(device)
            seconds = perf_counter() - start
            if (runtime.optimizer_steps or runtime.optimizer.state
                    or any(param.grad is not None for param in runtime.model.parameters())):
                raise ValueError(f"G4 evaluation performed an optimizer operation or accumulated gradients: {session}")
            fold = stage / session
            (fold / "reports").mkdir(parents=True)
            (fold / "manifests").mkdir()
            save_checkpoint_atomic(fold / AVERAGED_CHECKPOINT, {"identity": identity, "model_state_dict": state})
            metrics, _ = export_scoring(fold / "reports", "held_out", result["predictions"], expected,
                                        filename=PREDICTIONS.name)
            write_json_atomic(fold / "reports/held_out_metrics.json", metrics)
            write_json_atomic(fold / FOLD_MANIFEST, {
                **ZERO_UPDATES, "identity": identity, "scored_populations": ["held_out"], "calibration": None,
                "device": device, "torch_version": str(torch.__version__),
                "timing_seconds": {"checkpoint_read_average_strict_load": averaging_seconds,
                                   "held_out_inference_and_prediction_collection": seconds},
                "files_sha256": {p.relative_to(fold).as_posix(): sha256_file(p)
                                 for p in sorted(fold.rglob("*")) if p.is_file()},
            })
        write_json_atomic(stage / "evaluation_manifest.json", {
            **ZERO_UPDATES, "schema": g4.SCHEMA,
            "source_checkpoint_reads": sum(len(identity["sources"]) for _, _, _, identity, _ in prepared.values()),
            "inference_calls": len(prepared), "config_sha256": sha256_file(ROOT / g4.CONFIG),
            "scoring_source_sha256": source_hashes(ROOT, (ROOT / g4.CONFIG,)),
            "folds": {session: {"manifest_sha256": sha256_file(stage / session / FOLD_MANIFEST),
                                "controls": controls[session][2]} for session in SESSIONS},
        })
        require_new_destination(destination, ce_root, ls_root, g2_root, projection_dir)
        stage.rename(destination)
    return destination


def require_record(manifest, expected, source):
    validate_hold(manifest)
    if any(key not in manifest or manifest[key] != value for key, value in {**ZERO_UPDATES, **expected}.items()):
        raise ValueError(f"G4 evaluation manifest differs: {source}")


def read_g3_reference(directory, controls):
    """Import G3's budget arms for description after checking the common E30 endpoints."""
    manifest_path = directory / "comparison_manifest.json"
    manifest, manifest_sha = g4.read_json_sha(manifest_path)
    validate_hold(manifest)
    positions = manifest["outer_cost_and_positions"]
    locked = dict(zip(SESSIONS, (54, 72, 72, 48)))
    if manifest["schema"] != G3_SCHEMA or set(positions) != set(locked):
        raise ValueError("G3 descriptive reference does not contain the formal four-fold budget comparison")
    hashes = manifest["output_sha256"]
    report = directory / "summary.json"
    summary, summary_sha = g4.read_json_sha(report)
    if summary_sha != hashes["summary.json"]:
        raise ValueError("G3 descriptive summary SHA-256 mismatch")
    for session, updates in locked.items():
        if positions[session]["selected_position"]["optimizer_steps"] != updates:
            raise ValueError(f"G3 descriptive reference uses different locked updates: {session}")
        for arm in (CE30, LS30):
            key = f"{session}/{arm}_predictions.csv"
            rows, _ = g4.read_csv(directory / key, hashes[key])
            if [parse_prediction(row) for row in rows] != controls[session][1][arm]:
                raise ValueError(f"G3 E30 logits do not match the original {arm} export: {session}")
    return {"comparison_manifest_sha256": manifest_sha, "summary_sha256": summary_sha,
            "locked_optimizer_steps": locked,
            "scopes": {scope: {arm: summary[scope]["arms"][arm] for arm in ("ce_u_star", "ls_u_star")}
                       for scope in (*SESSIONS, "pooled_oof")}}


def compare(ce_root, ls_root, g2_root, evaluation, destination, frozen, raw, *, g3_analysis=None):
    read_only = (ce_root, ls_root, g2_root, evaluation) + ((g3_analysis,) if g3_analysis else ())
    require_new_destination(destination, *read_only)
    batch_path = evaluation / "evaluation_manifest.json"
    batch, batch_sha = g4.read_json_sha(batch_path)
    require_record(batch, {"schema": g4.SCHEMA, "config_sha256": sha256_file(ROOT / g4.CONFIG),
                           "source_checkpoint_reads": len(SESSIONS) * len(g4.AVERAGING_EPOCHS),
                           "inference_calls": len(SESSIONS)}, batch_path)
    if set(batch["folds"]) != set(SESSIONS):
        raise ValueError("G4 requires exactly four completed LS AVG folds")
    reference = historical_imports()
    controls = {session: g4.read_controls(ce_root, ls_root, g2_root, session, frozen, raw, reference)
                for session in SESSIONS}
    descriptive = read_g3_reference(g3_analysis, controls) if g3_analysis else None

    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".g4_compare_", dir=destination.parent) as temporary:
        stage = Path(temporary)
        pooled, summary, timings = {arm: [] for arm in g4.ARMS}, {}, {}
        for session, (ls, arms, evidence) in controls.items():
            fold = evaluation / session
            path = fold / FOLD_MANIFEST
            manifest, fold_sha = g4.read_json_sha(path)
            if batch["folds"][session] != {"manifest_sha256": fold_sha, "controls": evidence}:
                raise ValueError(f"G4 fold or control provenance changed after evaluation: {session}")
            require_record(manifest, {"identity": fold_identity(ls, session), "scored_populations": ["held_out"],
                                      "calibration": None}, path)
            rows, _ = g4.read_csv(fold / PREDICTIONS, manifest["files_sha256"][PREDICTIONS.as_posix()])
            arms[g4.LS_AVG] = [parse_prediction(row) for row in rows]
            summary[session] = g4.report_scope(stage / session, arms, population(arms[CE30]))
            timings[session] = manifest["timing_seconds"]
            for arm, predictions in arms.items():
                pooled[arm].extend(predictions)
        expected = population(pooled[CE30])
        if len({uid for uid, _, _ in expected}) != len(expected):
            raise ValueError("G4 pooled OOF contains duplicate utterance IDs")
        summary["pooled_oof"] = g4.report_scope(stage / "pooled_oof", pooled, expected)
        write_json_atomic(stage / "summary.json", summary)
        write_json_atomic(stage / "decision.json", g4.decision(summary))
        if descriptive is not None:
            write_json_atomic(stage / "g3_descriptive_reference.json", descriptive)
        write_json_atomic(stage / "comparison_manifest.json", {
            **ZERO_UPDATES, "schema": g4.SCHEMA, "evaluation_manifest_sha256": batch_sha,
            "primary_contrast": g4.PRIMARY, "arms": list(g4.ARMS), "bootstrap": g4.BOOTSTRAP,
            "bootstrap_draws_sha256": {scope: report["bootstrap_draws_sha256"] for scope, report in summary.items()},
            "per_fold_timing_seconds": timings,
            "output_sha256": {p.relative_to(stage).as_posix(): sha256_file(p)
                              for p in sorted(stage.rglob("*")) if p.is_file()},
        })
        require_new_destination(destination, *read_only)
        stage.rename(destination)
    return destination


def main(argv=None):
    cli = argparse.ArgumentParser(description=__doc__)
    commands = cli.add_subparsers(dest="command", required=True)
    evaluate_args = commands.add_parser("evaluate", help="authenticate all 16 checkpoints, then score four LS averages")
    evaluate_args.add_argument("--projection-dir", type=Path, required=True)
    evaluate_args.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    evaluate_args.add_argument("--execute", action="store_true", help="release the reviewed held-out evaluations")
    compare_args = commands.add_parser("compare", help="report the factorial from existing predictions")
    compare_args.add_argument("--g3-analysis", type=Path, help="include a verified G3 budget reference")
    for command in (evaluate_args, compare_args):
        command.add_argument("--ce-root", type=Path, required=True)
        command.add_argument("--ls-root", type=Path, default=DEFAULT_LS_ROOT)
        command.add_argument("--g2-root", type=Path, required=True)
        command.add_argument("--experiment-date", required=True)
        command.add_argument("--batch-id", required=True)
    args = cli.parse_args(argv)
    if args.command == "evaluate" and not args.execute:
        cli.error("G4 evaluation requires --execute")
    try:
        config = g4.recipe()
        paths = resolve_output_paths(output_base=config["output"]["root"], experiment_group=g4.GROUP,
                                     experiment_date=resolve_experiment_date(args.experiment_date),
                                     batch_id=args.batch_id, run_id=f"{args.batch_id}_ls_avg")
        roots = tuple(source_root(path, group) for path, group in
                      ((args.ce_root, GROUP), (args.ls_root, G1_GROUP), (args.g2_root, G2_GROUP)))
        source_root(paths.experiment_root, g4.GROUP)
        frozen, raw = read_frozen(ROOT)
        if args.command == "evaluate":
            print(evaluate(*roots, paths.run_root, args.projection_dir, frozen, raw, device=args.device))
        else:
            print(compare(*roots, paths.run_root, paths.analysis_root, frozen, raw, g3_analysis=args.g3_analysis))
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, EOFError, UnpicklingError) as error:
        cli.exit(1, f"G4 STOP: {error}\n")


if __name__ == "__main__":
    main()

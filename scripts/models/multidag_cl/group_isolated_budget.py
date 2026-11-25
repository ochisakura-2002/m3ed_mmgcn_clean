"""G3 Protocol B fit-side optimizer budget selection, paired refit and OOF reporting."""

import argparse
import csv
from hashlib import sha256
import json
import os
from pathlib import Path
from pickle import UnpicklingError
import re
from tempfile import TemporaryDirectory
from time import perf_counter

import torch

from scripts.analysis.models.multidag_cl.group_isolated_comparison import paired_report, parse_prediction, population
from scripts.analysis.models.multidag_cl.post_ph.internal_protocol import write_rows
from scripts.models.multidag_cl.group_isolated import ROOT, build_runtime, steps_per_epoch
from scripts.models.multidag_cl.group_isolated_tail_average import (
    AVG, CE30, LS30, RELIABILITY, export_scoring, g2_run_id, read_average, require_new_destination,
)
from scripts.runtime.multidag_cl_group_isolated import budget as g3
from scripts.runtime.multidag_cl_group_isolated.budget_history import historical_imports, inspect_history, verify_anchor
from scripts.runtime.multidag_cl_group_isolated.fixed_tail_average import authenticated_run
from scripts.runtime.multidag_cl_group_isolated.protocol import (
    CONFIG_DIR, ENDPOINT, G1_CONFIG_DIR, G1_EPSILON, SESSIONS, describe, fold_config,
    load_fold, read_frozen, validate_hold,
)
from scripts.runtime.multidag_cl_group_isolated.provenance import runtime_provenance, source_hashes
from scripts.runtime.multidag_cl_ta_v0.formal import json_identity, sha256_file, write_json_atomic
from scripts.runtime.multidag_cl_ta_v0.training import seed_everything
from utils.output_paths import resolve_experiment_date
from models.multidag_cl.ta_v0 import LABEL_VOCAB


CE_U, LS_U = "ce_u_star", "ls_u_star"
ARMS = (CE30, LS30, CE_U, LS_U)
COMPARISONS = ((CE_U, LS_U), (CE30, CE_U), (LS30, LS_U), (CE30, LS30),
               (CE30, AVG), (CE_U, AVG), (LS_U, AVG), (LS30, AVG))


def source_configs(raw):
    return (ROOT / CONFIG_DIR / "fold_identities.json", ROOT / raw["frontend_source"]["config_path"],
            *(ROOT / directory / f"{s.lower()}_seed100.yaml" for directory in (CONFIG_DIR, G1_CONFIG_DIR)
              for s in SESSIONS))


def capture_source(raw, device):
    seed_everything(device)
    return runtime_provenance(ROOT, device, source_configs(raw))


def check_sources(raw, source):
    after = source_hashes(ROOT, source_configs(raw))
    if any(after.get(path) != digest for path, digest in source["files_sha256"].items()):
        raise ValueError("G3 source files changed during this phase")
    return {path: digest for path, digest in after.items() if path not in source["files_sha256"]}


def inner_score(stage, spec, path, manifest, runtime, validation, rows):
    outer, held = spec["outer"], spec["validation"]
    directory = stage / "selection" / outer / held
    directory.mkdir(parents=True)
    scores = []
    for step in spec["candidates"]:
        record = manifest["checkpoints"][str(step)]
        g3.sync(runtime.device)
        start = perf_counter()
        state = g3.load_state(path, manifest["identity"], record)
        predictions = g3.score_state(runtime, state, validation, rows)
        g3.sync(runtime.device)
        scores.append({**record, "valid_targets": len(rows), "hard_nll": g3.hard_nll(predictions),
                       "score_wall_seconds": perf_counter() - start})
    table = {**g3.HOLD_RECORD, **spec, "fit_population": manifest["identity"]["population"],
             "validation_population": describe(rows), "projection": manifest["identity"]["projection"],
             "trajectory": path.relative_to(stage).as_posix(),
             "trajectory_sha256": sha256_file(path / "trajectory.json"),
             "scores": scores}
    write_json_atomic(directory / "scores.json", table)


def lock_tables(directory, frozen, raw, outer_data):
    selections = {}
    for outer in SESSIONS:
        tables, inputs = [], {}
        for validation in SESSIONS:
            if validation == outer:
                continue
            name = f"selection/{outer}/{validation}/scores.json"
            path = directory / name
            payload = path.read_bytes()
            table = json.loads(payload)
            validate_hold(table)
            if (table["validation"] != validation or table["TEST_TOUCHED"] != "NO"
                    or table["projection"] != outer_data[outer]["projection"]):
                raise ValueError("G3 inner score table is in the wrong fold or projection")
            tables.append(table)
            inputs[name] = sha256(payload).hexdigest()
        selections[outer] = {**g3.select(outer, tables, frozen), "score_files_sha256": inputs}
    return {"recipe_sha256": json_identity(raw), "outer_fit_data_sha256": json_identity(outer_data),
            "selections": selections}


def run_inner(destination, projection_dir, frozen, raw, *, device, independent, source, synthetic=False):
    require_new_destination(destination, projection_dir)
    phase_start = perf_counter()
    views, entries, outer_data = {}, {}, {}
    for outer in SESSIONS:
        subsets, rows, outer_data[outer] = load_fold(fold_config(outer), frozen, projection_dir, fit_only=True)
        views[outer], entries[outer] = subsets["fit"], rows["fit"]
    specs = g3.plan(frozen, raw)
    members = []
    for spec in specs:
        outer, validation = spec["outer"], spec["validation"]
        sub, rows = g3.subset(views[outer], entries[outer], spec["fit"], "train")
        val, val_rows = g3.subset(views[outer], entries[outer], [validation], "dev")
        expected = frozen["folds"][validation]["held_out"]
        if describe(val_rows) != expected or set(describe(rows)["actors"]) & set(expected["actors"]):
            raise ValueError("G3 inner validation population or actor isolation changed")
        prepared = g3.prepare(raw, sub, rows, outer_data[outer]["projection"], source, max(spec["candidates"]),
                              device=device, synthetic=synthetic)
        members.append({"spec": spec, "fit_view": sub, "fit_rows": rows, "validation_view": val,
                        "validation_rows": val_rows, "identity": prepared.identity, "tape": prepared.tape})
        del prepared
    groups, reuse = g3.qualify_shared(members) if not independent else ([[m] for m in members], "NOT_SUPPORTED")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".g3_inner_", dir=destination.parent) as temporary:
        stage = Path(temporary)
        runs, scoring = {}, []
        for group in groups:
            fit = group[0]
            spec = fit["spec"]
            key = "_".join(spec["fit"] if len(group) > 1 else (spec["outer"], spec["validation"], *spec["fit"]))
            path = stage / "trajectories" / key
            checkpoints = sorted({step for member in group for step in member["spec"]["candidates"]})
            prepared = g3.prepare(raw, fit["fit_view"], fit["fit_rows"], fit["identity"]["projection"], source,
                                  checkpoints[-1], device=device, synthetic=synthetic)
            if (prepared.identity != fit["identity"] or any(
                prepared.tape[:len(member["tape"])] != member["tape"] for member in group
            )):
                raise ValueError("G3 preflight fit identity changed before training")
            manifest = g3.train_fixed(path, prepared, checkpoints)
            write_json_atomic(path / "trajectory.json", manifest)
            runs[key] = {"updates": manifest["updates"], "fit_wall_seconds": manifest["fit_wall_seconds"],
                         "update_wall_seconds": manifest["update_wall_seconds"],
                         "trajectory_sha256": sha256_file(path / "trajectory.json"),
                         "consumers": [[member["spec"]["outer"], member["spec"]["validation"]] for member in group]}
            scoring.extend((member, path, manifest) for member in group)
            del prepared
        # Validation cannot feed state, RNG, or results into any training trajectory.
        for member, path, manifest in scoring:
            runtime, _ = build_runtime(raw, {"train": member["fit_view"]}, device, synthetic=synthetic)
            inner_score(stage, member["spec"], path, manifest, runtime, member["validation_view"],
                        member["validation_rows"])
        lock = {**g3.HOLD_RECORD, "schema": g3.SCHEMA, "synthetic_only": synthetic, "source": source,
                "inner_reuse": reuse, "trajectories": runs, "training_runs": len(runs),
                "training_updates": sum(run["updates"] for run in runs.values()),
                "phase_wall_seconds": perf_counter() - phase_start,
                "outer_data": outer_data, **lock_tables(stage, frozen, raw, outer_data)}
        lock["late_imports_sha256"] = check_sources(raw, source)
        write_json_atomic(stage / "budget_lock.json", lock)
        require_new_destination(destination, projection_dir)
        stage.rename(destination)
    return lock


def read_lock(directory, digest, frozen, raw, *, synthetic=False):
    payload = (directory / "budget_lock.json").read_bytes()
    if sha256(payload).hexdigest() != digest:
        raise ValueError("G3 budget lock SHA-256 differs from the released selection")
    lock = json.loads(payload)
    validate_hold(lock)
    if (lock["schema"] != g3.SCHEMA or lock["synthetic_only"] is not synthetic or lock["TEST_TOUCHED"] != "NO"
            or set(lock["outer_data"]) != set(SESSIONS)):
        raise ValueError("G3 requires a complete four-fold lock in the same execution mode")
    for key, value in lock_tables(directory, frozen, raw, lock["outer_data"]).items():
        if lock[key] != value:
            raise ValueError("G3 locked selection differs from the independently scored tables")
    return lock


def paired_fit(stage, outer, selected, fit, rows, data, frozen, raw, roots, source, reference, *, device,
               synthetic=False):
    max_updates = ENDPOINT * steps_per_epoch(describe(rows), raw)

    def prepare(is_ls, stop):
        return g3.prepare(raw, fit, rows, data["projection"], source, stop, device=device, synthetic=synthetic,
                          smoothing=G1_EPSILON if is_ls else 0.0)

    snapshots, evidence, planned, weights = {}, {}, {}, {}
    reason = None
    mode = "fresh_paired_refit"
    cost = []
    try:
        if not roots:
            raise ValueError("historical CE and LS roots not supplied")
        for is_ls, root in zip((False, True), roots):
            runtime = prepare(is_ls, max_updates)
            planned[is_ls] = runtime.identity, runtime.tape
            evidence[is_ls] = inspect_history(root, outer, is_ls, frozen, raw, runtime, source, reference, selected)
        mode = "verified_historical_checkpoints" if evidence[False]["selected"] is not None else "verified_prefix_replay"
        if mode == "verified_prefix_replay":
            for is_ls in (False, True):
                runtime = prepare(is_ls, selected)
                identity, tape = planned[is_ls]
                if runtime.identity != identity or runtime.tape != tape[:selected]:
                    raise ValueError("G3 outer prefix differs from its reviewed fit identity")
                directory = stage / outer / ("ls_prefix" if is_ls else "ce_prefix")
                g3.sync(device)
                attempt_start = perf_counter()
                try:
                    history = evidence[is_ls]
                    record = g3.train_fixed(directory, runtime, [selected],
                                            verify_epoch=lambda epoch, model: verify_anchor(history, epoch, model))
                finally:
                    g3.sync(device)
                    cost.append({"arm": "LS" if is_ls else "CE", "kind": "prefix",
                                 "updates": runtime.runtime.optimizer_steps,
                                 "fit_wall_seconds": perf_counter() - attempt_start})
                write_json_atomic(directory / "trajectory.json", record)
                weights[str(is_ls)] = {"prefix": record["checkpoints"][str(selected)],
                                       "trajectory_sha256": sha256_file(directory / "trajectory.json")}
                evidence[is_ls]["selected"] = g3.load_state(directory, record["identity"],
                                                            record["checkpoints"][str(selected)])
        for is_ls in (False, True):
            snapshots[is_ls] = (evidence[is_ls]["e30"], evidence[is_ls]["selected"])
    except (OSError, EOFError, UnpicklingError, KeyError, TypeError, ValueError) as error:
        mode, reason = "fresh_paired_refit", str(error)
    if mode == "fresh_paired_refit":
        snapshots.clear()
        for is_ls in (False, True):
            runtime = prepare(is_ls, max_updates)
            planned[is_ls] = runtime.identity, runtime.tape
            directory = stage / outer / ("ls_refit" if is_ls else "ce_refit")
            record = g3.train_fixed(directory, runtime, {selected, max_updates})
            write_json_atomic(directory / "trajectory.json", record)
            weights[str(is_ls)] = {"checkpoints": record["checkpoints"],
                                   "trajectory_sha256": sha256_file(directory / "trajectory.json")}
            cost.append({"arm": "LS" if is_ls else "CE", "kind": "refit", "updates": record["updates"],
                         "fit_wall_seconds": record["fit_wall_seconds"], "update_wall_seconds": record["update_wall_seconds"]})
            snapshots[is_ls] = tuple(g3.load_state(directory, record["identity"], record["checkpoints"][str(step)])
                                     for step in (max_updates, selected))
    pair = [{k: v for k, v in identity.items() if k != "label_smoothing"}
            for identity, _ in planned.values()]
    if pair[0] != pair[1] or planned[False][1] != planned[True][1]:
        raise ValueError("G3 outer CE and LS differ beyond the objective")
    identity, tape = planned[False]
    return snapshots, {**g3.HOLD_RECORD, "outer": outer, "mode": mode, "fallback_reason": reason,
                       "fit_identity_sha256": {str(k): json_identity(v[0]) for k, v in planned.items()},
                       "fit_population": identity["population"], "initialization_sha256": identity["initial_state_sha256"],
                       "initial_rng": identity["initial_rng"], "optimizer": identity["optimizer"], "cost": cost,
                       "selected_position": g3.position(tape, selected, identity["steps_per_epoch"]),
                       "e30_position": g3.position(tape, max_updates, identity["steps_per_epoch"]),
                       "history": {str(k): v["evidence"] for k, v in evidence.items()}, "new_weights": weights}


def run_outer(destination, projection_dir, lock, lock_sha, frozen, raw, roots, *, source, device, synthetic=False):
    require_new_destination(destination, projection_dir, *(roots or ()))
    phase_start = perf_counter()
    reference = historical_imports() if roots else {}
    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".g3_outer_", dir=destination.parent) as temporary:
        stage = Path(temporary)
        fitted = {}
        for outer in SESSIONS:
            views, rows, data = load_fold(fold_config(outer), frozen, projection_dir, fit_only=True)
            if data != lock["outer_data"][outer]:
                raise ValueError("outer fit differs from the reviewed G3 budget lock")
            selected = lock["selections"][outer]["selected_updates"]
            fitted[outer] = paired_fit(stage, outer, selected, views["fit"], rows["fit"], data, frozen, raw,
                                       roots, source, reference, device=device, synthetic=synthetic)
        summary = {}
        for outer in SESSIONS:
            snapshots, record = fitted[outer]
            views, rows, data = load_fold(fold_config(outer), frozen, projection_dir)
            if data != lock["outer_data"][outer]:
                raise ValueError("outer scoring population differs from the released fit identity")
            directory = stage / outer / "held_out"
            directory.mkdir(parents=True)
            runtime, _ = build_runtime(raw, {"train": views["fit"]}, device, synthetic=synthetic)
            timing, files = {}, {}
            for is_ls, keys in ((False, (CE30, CE_U)), (True, (LS30, LS_U))):
                previous = None
                for key, state in zip(keys, snapshots[is_ls]):
                    g3.sync(device)
                    tick = perf_counter()
                    predictions = (previous if key in (CE_U, LS_U) and
                                   record["selected_position"] == record["e30_position"]
                                   else g3.score_state(runtime, state, views["held_out"], rows["held_out"]))
                    g3.sync(device)
                    timing[key] = {"inference_calls": int(predictions is not previous),
                                   "wall_seconds": perf_counter() - tick if predictions is not previous else 0.0}
                    path = directory / f"{key}.csv"
                    fields = {"true_label": "true_label_id", "predicted_label": "predicted_label_id"}
                    predictions_csv = [
                        {**{fields.get(k, k): r[k] for k in ("utterance_id", "dialogue_id", *fields)},
                         **{f"logit_{label}": value for label, value in zip(LABEL_VOCAB, r["logits"])}}
                        for r in predictions
                    ]
                    write_rows(path, predictions_csv, tuple(predictions_csv[0]))
                    files[key] = sha256_file(path)
                    previous = predictions
            summary[outer] = {**record, "scoring": timing, "prediction_sha256": files}
        write_json_atomic(stage / "outer_manifest.json", {
            **g3.HOLD_RECORD, "schema": g3.SCHEMA, "synthetic_only": synthetic, "device": device,
            "budget_lock_sha256": lock_sha, "source": source, "late_imports_sha256": check_sources(raw, source),
            "folds": summary, "phase_wall_seconds": perf_counter() - phase_start,
        })
        require_new_destination(destination, projection_dir, *(roots or ()))
        stage.rename(destination)
    return summary


def scope_report(directory, arms, expected):
    directory.mkdir()
    metrics, risks = {}, {}
    for arm, rows in arms.items():
        metrics[arm], risks[arm] = export_scoring(directory, arm, rows, expected)
    comparisons = {}
    for before, after in COMPARISONS:
        report, details = paired_report(arms[before], arms[after])
        del report["ctrl"], report["g1"]
        a, b = (metrics[key]["auxiliary"] for key in (before, after))
        report["reliability_delta"] = {key: b[key] - a[key] for key in RELIABILITY}
        accepted = a["high_confidence"]["support"]
        report["risk_at_control_high_confidence_coverage"] = ({
            "accepted": accepted, "coverage": accepted / len(expected),
            before: risks[before][accepted - 1]["tie_expected_risk"],
            after: risks[after][accepted - 1]["tie_expected_risk"],
        } if accepted else None)
        report["per_class_r_d_net"] = {}
        for index, label in enumerate(LABEL_VOCAB):
            rows = [r for r in details if r["true_label_id"] == index]
            r, d = (sum(row[key] for row in rows) for key in ("recovered", "damaged"))
            report["per_class_r_d_net"][label] = {"support": len(rows), "r": r, "d": d, "net": r - d}
        for row in details:
            row["control_predicted_label_id"] = row.pop("ctrl_predicted_label_id")
            row["candidate_predicted_label_id"] = row.pop("g1_predicted_label_id")
        key = f"{after}_vs_{before}"
        write_rows(directory / f"{key}_paired_utterances.csv", details, tuple(details[0]))
        comparisons[key] = report
    return {"arms": metrics, "comparisons": comparisons}


def compare(destination, outer_run, ce_root, g2_root, lock, lock_sha, frozen, raw, *, synthetic=False):
    require_new_destination(destination, outer_run, ce_root, g2_root)
    path = outer_run / "outer_manifest.json"
    payload = path.read_bytes()
    outer_manifest = json.loads(payload)
    validate_hold(outer_manifest)
    if (outer_manifest["schema"] != g3.SCHEMA or outer_manifest["budget_lock_sha256"] != lock_sha
            or outer_manifest["synthetic_only"] is not synthetic or outer_manifest["TEST_TOUCHED"] != "NO"
            or set(outer_manifest["folds"]) != set(SESSIONS)):
        raise ValueError("G3 comparison requires four completed folds from the released lock")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".g3_compare_", dir=destination.parent) as temporary:
        stage = Path(temporary)
        pooled = {arm: [] for arm in (*ARMS, AVG)}
        summary, g2_evidence = {}, {}
        for outer in SESSIONS:
            manifest = outer_manifest["folds"][outer]
            if (manifest["outer"] != outer or manifest["selected_position"]["optimizer_steps"]
                    != lock["selections"][outer]["selected_updates"]):
                raise ValueError("G3 outer result differs from the released selected update")
            arms = {}
            for key in ARMS:
                path = outer_run / outer / "held_out" / f"{key}.csv"
                if sha256_file(path) != manifest["prediction_sha256"][key]:
                    raise ValueError("G3 score export SHA-256 mismatch")
                with path.open(encoding="utf-8", newline="") as stream:
                    arms[key] = [parse_prediction(row) for row in csv.DictReader(stream)]
            expected = population(arms[CE30])
            if describe(expected) != frozen["folds"][outer]["held_out"]:
                raise ValueError("G3 OOF population differs from the frozen fold")
            config = fold_config(outer)
            run = ce_root / "runs" / config["run_id"]
            ce_manifest, digest = authenticated_run(run, config, frozen, raw)
            arms[AVG], g2_evidence[outer] = read_average(g2_root / "runs" / g2_run_id(outer), run,
                                                         ce_manifest, digest, outer, expected)
            summary[outer] = scope_report(stage / outer, arms, expected)
            for key, rows in arms.items():
                pooled[key].extend(rows)
        summary["pooled_oof"] = scope_report(stage / "pooled_oof", pooled, population(pooled[CE30]))
        write_json_atomic(stage / "summary.json", summary)
        costs = [cost for fold in outer_manifest["folds"].values() for cost in fold["cost"]]
        write_json_atomic(stage / "comparison_manifest.json", {
            **g3.HOLD_RECORD, "schema": g3.SCHEMA, "budget_lock_sha256": lock_sha,
            "outer_manifest_sha256": sha256(payload).hexdigest(), "outer_cost_and_positions": outer_manifest["folds"],
            "g2_descriptive_sources": g2_evidence, "primary_contrast": f"{LS_U}_vs_{CE_U}",
            "shared_u_star_selection_objective": "CE_inner_hard_label_nll",
            "inner_reuse": lock["inner_reuse"], "new_training_runs": lock["training_runs"] + len(costs),
            "optimizer_updates_performed": lock["training_updates"] + sum(cost["updates"] for cost in costs),
            "inner_phase_wall_seconds": lock["phase_wall_seconds"],
            "outer_phase_wall_seconds": outer_manifest["phase_wall_seconds"],
            "output_sha256": {p.relative_to(stage).as_posix(): sha256_file(p) for p in stage.rglob("*") if p.is_file()},
        })
        require_new_destination(destination, outer_run, ce_root, g2_root)
        stage.rename(destination)
    return destination


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("plan")
    for command in ("inner", "outer", "compare"):
        sub = commands.add_parser(command)
        sub.add_argument("--experiment-date", required=True)
        sub.add_argument("--batch-id", required=True)
        if command != "compare":
            sub.add_argument("--projection-dir", type=Path, required=True)
            sub.add_argument("--execute", action="store_true")
        if command == "inner":
            sub.add_argument("--independent", action="store_true")
        else:
            sub.add_argument("--lock-sha256", required=True)
            sub.add_argument("--ce-root", type=Path, required=command == "compare")
        if command == "outer":
            sub.add_argument("--ls-root", type=Path)
            sub.add_argument("--deterministic", action="store_true")
        if command == "compare":
            sub.add_argument("--g2-root", type=Path, required=True)
    args = parser.parse_args(argv)
    frozen, raw = read_frozen(ROOT)
    if args.command == "plan":
        print(json.dumps(g3.plan(frozen, raw), indent=2))
        return
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", args.batch_id):
        parser.error("invalid G3 batch identifier")
    root = ROOT / "outputs" / resolve_experiment_date(args.experiment_date) / g3.GROUP
    inner = root / "runs" / f"{args.batch_id}_inner"
    outer = root / "runs" / f"{args.batch_id}_outer"
    if args.command != "inner":
        lock = read_lock(inner, args.lock_sha256, frozen, raw)
    if args.command == "compare":
        print(compare(root / "analysis" / "batches" / args.batch_id, outer, args.ce_root, args.g2_root,
                      lock, args.lock_sha256, frozen, raw))
        return
    if not args.execute or os.name == "nt" or not torch.cuda.is_available():
        parser.error("formal G3 training requires an explicit release and Ubuntu CUDA")
    deterministic = not args.independent if args.command == "inner" else args.deterministic
    if deterministic:
        if os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in (":4096:8", ":16:8"):
            parser.error("deterministic G3 CUDA requires CUBLAS_WORKSPACE_CONFIG=:4096:8 or :16:8")
        torch.use_deterministic_algorithms(True)
    source = capture_source(raw, "cuda")
    if args.command == "inner":
        run_inner(inner, args.projection_dir, frozen, raw, device="cuda", independent=args.independent, source=source)
        print(sha256_file(inner / "budget_lock.json"))
    else:
        if (args.ce_root is None) != (args.ls_root is None):
            parser.error("historical reuse requires both --ce-root and --ls-root")
        roots = (args.ce_root, args.ls_root) if args.ce_root else None
        run_outer(outer, args.projection_dir, lock, args.lock_sha256, frozen, raw, roots, source=source, device="cuda")
        print(outer)


if __name__ == "__main__":
    main()

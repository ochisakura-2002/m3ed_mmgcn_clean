"""Prepare A0/B0, or run the three frozen seed100 Train/Dev arms on Ubuntu CUDA."""

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from scripts.analysis.models.multidag_cl.export_fastkill_bundle import export_bundle
from scripts.analysis.models.multidag_cl.fastkill_geometry import export_b0, export_geometry
from scripts.runtime.multidag_cl_fastkill.prefix_diagnostic import analyze, from_dataset, from_manifest
from scripts.runtime.multidag_cl_fastkill.runtime import build_runtime, prototype_rows
from scripts.runtime.multidag_cl_fastkill.spec import (
    A1, ARMS, B1, GROUP, HOLD, SEED, baseline, load_configs, require_frozen_baseline,
)
from scripts.runtime.multidag_cl_paper_reimplementation.checkpoint import load_checkpoint
from scripts.runtime.multidag_cl_ta_v0.formal import git_identity, json_identity, write_json_atomic
from scripts.runtime.multidag_cl_tav_full_audit_v1.confirm import test_hold
from scripts.runtime.multidag_cl_tav_full_audit_v1.formal import execute
from scripts.runtime.multidag_cl_tav_full_audit_v1.runtime import load_train_dev
from scripts.runtime.multidag_cl_tav_full_audit_v1.spec import write_csv
from utils.output_paths import resolve_experiment_date, resolve_output_paths


def job_plan(paths, configs):
    raw = baseline()
    jobs = {}
    for arm in ARMS:
        run_id = f"{arm}_seed{SEED}"
        jobs[arm] = {
            "experiment_id": run_id, "run_dir": (paths.runs_root / run_id).as_posix(),
            "config_sha256": json_identity(raw), "objective_config": configs[arm], "status": "PENDING",
        }
    return jobs


def export_objective(runtime, run, result):
    arm = runtime.arm
    if arm not in (A1, B1):
        return
    destination = run / "reports" / "fastkill"
    destination.mkdir(exist_ok=False)
    if arm == A1:
        write_csv(destination / "alpha_distribution.csv", runtime.alpha_history)
        write_csv(destination / "per_class_effective_loss_weight.csv", runtime.class_exposure)
        return
    write_csv(destination / "train_loss_components.csv", runtime.epoch_objectives)
    write_csv(destination / "prototype_norm_by_epoch.csv", runtime.prototype_norms)
    write_csv(destination / "prototype_cosine_by_epoch.csv", runtime.prototype_cosines)
    payload = load_checkpoint(run / "checkpoints/best_model.pt", torch.device("cpu"))
    runtime.validate_checkpoint(payload)
    prototypes = payload["training_only_class_prototypes"]
    norms, cosines = prototype_rows(prototypes, result["best_epoch"])
    write_csv(destination / "best_prototype_norm.csv", norms)
    write_csv(destination / "best_prototype_cosine.csv", cosines)
    np.savez_compressed(destination / "best_prototypes.npz", prototypes=prototypes.numpy())
    export_geometry(runtime, destination / "geometry", SEED)
    write_json_atomic(destination / "geometry_manifest.json", {
        **HOLD, "seed": SEED, "checkpoint_sha256": result["best_checkpoint_sha256"],
        "best_epoch": result["best_epoch"], "centroid_reference_split": "train",
        "optimizer_steps_performed": 0,
    })


def run_batch(paths, configs, load_data, git, *, device, archive, synthetic_epochs=None):
    synthetic = synthetic_epochs is not None
    if (synthetic and device != "cpu") or (not synthetic and (os.name == "nt" or device != "cuda")):
        raise ValueError("Fast-Kill training requires explicit Ubuntu CUDA or CPU synthetic checks")
    jobs = job_plan(paths, configs)
    locations = (paths.manifest_root, paths.launcher_log_root, paths.report_root, paths.analysis_root,
                 archive, *(job["run_dir"] for job in jobs.values()))
    for location in locations:
        if os.path.lexists(location):
            raise FileExistsError(location)
    paths.manifest_root.mkdir(parents=True, exist_ok=False)
    manifest = paths.manifest_root / "fastkill_batch_manifest.json"
    a0_directory = paths.report_root / "A0"
    state = {**HOLD, **git, "schema": "ct_multidag_fastkill_batch_v1", "status": "RUNNING",
             "batch_id": paths.manifest_root.name, "experiment_date": paths.experiment_date,
             "experiment_group": GROUP, "device": device, "synthetic_only": synthetic, "jobs": jobs,
             "a0_directory": a0_directory.as_posix()}
    current = None
    try:
        write_json_atomic(manifest, state)
        paths.launcher_log_root.mkdir(parents=True, exist_ok=False)
        with (paths.launcher_log_root / "master.log").open("x", encoding="utf-8") as master:
            def progress(message, *, flush=False):
                print(message, flush=flush)
                print(message, file=master, flush=flush)

            datasets, data_identity, data_paths = load_data()
            state.update(data_identity=data_identity, data_paths=data_paths)
            try:
                prefix, difficulties = analyze(from_dataset(datasets["train"]), a0_directory, synthetic=synthetic)
                if prefix["PREFIX_SUFFIX_INVARIANCE"] != "PASS":
                    raise ValueError("A1 blocked by suffix invariance")
            except ValueError:
                state["A_ROUTE"] = "INVALID"
                raise
            progress("PREFIX_SUFFIX_INVARIANCE = PASS", flush=True)
            reference, pairing, initialization = None, None, None
            for arm, job in jobs.items():
                current = job
                job["status"] = "RUNNING"
                if arm == A1:
                    state["A_ROUTE"] = "INVALID"
                write_json_atomic(manifest, state)
                runtime, reference = build_runtime(
                    configs[arm], datasets, device=device, reference=reference, pairing=pairing,
                    prefix_difficulties=difficulties if arm == A1 else None, formal_train_dev=not synthetic,
                    synthetic_only=synthetic, data_identity=data_identity, git_head=git["git_head"])
                if initialization is not None and runtime.initialization != initialization:
                    raise ValueError(f"{arm} failed to match CTRL initialization or RNG")
                progress(f"{arm} RUNNING", flush=True)
                result = execute(runtime, baseline(), job["run_dir"], data_paths=data_paths, git=git,
                                 synthetic_epochs=synthetic_epochs, progress=progress, experiment=configs[arm])
                test_hold(result)
                if pairing is None:
                    pairing, initialization = result["batch_pairing"], result["shared_initialization"]
                if result["batch_pairing"] != pairing:
                    raise ValueError(f"{arm} does not match the CTRL batch tape")
                export_objective(runtime, Path(job["run_dir"]), result)
                job.update(status="PASS", result=result)
                current = None
                write_json_atomic(manifest, state)
            state.update(export_bundle(paths.analysis_root, archive, state))
            state["status"] = "PASS"
            progress("ANALYSIS_BUNDLE = PASS\nPLOT_READY = YES", flush=True)
    except BaseException as error:
        if current is not None:
            current["status"] = "FAIL"
        state.update(status="FAIL", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        write_json_atomic(manifest, state)
    return state


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--config-only", action="store_true", help="print the data-free plan (default)")
    mode.add_argument("--a0", action="store_true", help="run Train-only manifest-based diagnostics")
    mode.add_argument("--b0", action="store_true", help="export zero-update frozen F0 geometry from five checkpoints")
    mode.add_argument("--execute", action="store_true", help="run CTRL, B1 and A1 sequentially on Ubuntu CUDA")
    for name in ("asset-path", "model-manifest-path", "provenance-manifest-path", "frozen-runs",
                 "batch-id", "experiment-date"):
        parser.add_argument("--" + name)
    parser.add_argument("--output-root", default="outputs")
    parser.add_argument("--diagnostic-device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args(argv)
    if args.execute and (not sys.platform.startswith("linux") or not torch.cuda.is_available()):
        parser.error("--execute requires Ubuntu CUDA; Windows supports zero-update diagnostics and config-only")
    if args.a0 or args.b0 or args.execute:
        if not all((args.batch_id, args.model_manifest_path, args.provenance_manifest_path)):
            parser.error("supply --batch-id, --model-manifest-path and --provenance-manifest-path")
    if (args.b0 or args.execute) and not args.asset_path:
        parser.error("--b0 and --execute require --asset-path")
    if args.b0 and not args.frozen_runs:
        parser.error("--b0 requires --frozen-runs pointing to the five finished F0 run directories' parent")

    paths = resolve_output_paths(output_base=args.output_root, experiment_date=resolve_experiment_date(args.experiment_date),
                                 experiment_group=GROUP, batch_id=args.batch_id)
    configs = load_configs()
    require_frozen_baseline()

    def load_data():
        return load_train_dev(args.asset_path, args.model_manifest_path, args.provenance_manifest_path)

    if args.a0:
        result, _ = analyze(from_manifest(args.model_manifest_path, args.provenance_manifest_path),
                            paths.analysis_root / "A0")
    elif args.b0:
        datasets, identity, _ = load_data()
        output = paths.analysis_root / "B0"
        export_b0(datasets, identity, args.frozen_runs, output, device=args.diagnostic_device)
        result = {**HOLD, "B0_DIAGNOSTIC": "PASS", "output": output.as_posix()}
    elif args.execute:
        archive = Path("tmp") / f"{GROUP}_{paths.experiment_date}_{paths.manifest_root.name}.tar.gz"
        state = run_batch(paths, configs, load_data, git_identity(), device="cuda", archive=archive)
        result = {**HOLD, **{key: state[key] for key in ("status", "archive", "archive_sha256", "A_ROUTE", "B_ROUTE")}}
    else:
        result = {**HOLD, "status": "PLAN_ONLY", "experiment_date": paths.experiment_date,
                  "expected_formal_steps_per_arm": 210, "expected_formal_train_targets_per_arm": 144900,
                  "jobs": job_plan(paths, configs)}
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Preflight or execute the frozen 40-identity TAV Train/Dev confirmation."""

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from models.multidag_cl.tav_full_audit_v1 import TAVFullAudit
from scripts.runtime.multidag_cl_ta_v0.formal import git_identity, json_identity, sha256_file, write_json_atomic
from scripts.runtime.multidag_cl_ta_v0.training import seed_everything
from scripts.runtime.multidag_cl_tav_full_audit_v1.confirm import (
    CONFIRM_GROUP, MATRIX_PATH, SCHEMA, checked_run, frozen_model_source, historical_jobs, load_design, reuse_candidate,
)
from scripts.runtime.multidag_cl_tav_full_audit_v1.formal import execute, run_path
from scripts.runtime.multidag_cl_tav_full_audit_v1.runtime import AuditRuntime, load_train_dev
from utils.output_paths import resolve_experiment_date, resolve_output_paths


def plan(*, output_root="outputs", experiment_date=None):
    matrix, configs, sources = load_design()
    date = resolve_experiment_date(experiment_date)
    jobs = [{**row, "planned_run_dir": run_path(configs[row["experiment_id"]], output_root=output_root,
                                               experiment_date=date).as_posix()} for row in matrix]
    if len({job["planned_run_dir"] for job in jobs}) != 40 or len(sources) != 8:
        raise ValueError("confirmatory run path or source config collision")
    return date, jobs, configs, sources


def resolve_jobs(jobs, configs, sources, data_identity, historical_runs):
    records = []
    for job in jobs:
        raw = configs[job["experiment_id"]]
        run_dir = Path(job["planned_run_dir"])
        record = {**job, "run_dir": run_dir.as_posix(), "status": "pending", "reuse_status": "not_applicable"}
        if os.path.lexists(run_dir):
            checked_run(run_dir, raw, data_identity)
            record.update(status="completed", reuse_status="existing_confirm_run")
        elif job["source_seed100_run"]:
            candidate = historical_runs.get(job["source_seed100_run"])
            if candidate is None:
                record.update(reuse_status="rerun_seed100", reuse_reason="historical batch not supplied")
            else:
                matches, reason = reuse_candidate(candidate, sources[job["config_path"]], raw, data_identity)
                record.update(source_seed100_run_dir=candidate.as_posix(), reuse_reason=reason,
                              reuse_status="reused_seed100" if matches else "rerun_seed100")
                if matches:
                    record.update(run_dir=candidate.as_posix(), status="completed")
        records.append(record)
    return records


def run(*, asset_path, model_manifest_path, provenance_manifest_path, batch_id, full_audit_batch_manifest=None,
        output_root="outputs", experiment_date=None, device="cuda"):
    selected = torch.device(device)
    if (os.name == "nt" or selected.type != "cuda" or not torch.cuda.is_available()
            or selected.index is not None and selected.index >= torch.cuda.device_count()):
        raise ValueError("formal confirmatory execution requires a remote CUDA environment")
    device = str(selected)
    date, jobs, configs, sources = plan(output_root=output_root, experiment_date=experiment_date)
    if not frozen_model_source():
        raise ValueError("frozen Full Audit source lineage cannot be verified in this checkout")
    paths = resolve_output_paths(output_base=output_root, experiment_date=date, experiment_group=CONFIRM_GROUP,
                                 batch_id=batch_id)
    state_path = paths.manifest_root / "batch_manifest.json"
    if paths.manifest_root.exists() or paths.launcher_log_root.exists():
        raise FileExistsError(f"batch ID already reserved: {batch_id}")
    git = git_identity()
    historical_runs, historical_reference = {}, None
    if full_audit_batch_manifest:
        historical_path = Path(full_audit_batch_manifest)
        historical_reference = {"path": historical_path.as_posix(), "sha256": sha256_file(historical_path)}
        _, historical_runs = historical_jobs(historical_path)
        if sha256_file(historical_path) != historical_reference["sha256"]:
            raise ValueError("historical batch manifest changed during preflight")
    datasets, data_identity, data_paths = load_train_dev(asset_path, model_manifest_path, provenance_manifest_path)
    records = resolve_jobs(jobs, configs, sources, data_identity, historical_runs)
    paths.manifest_root.mkdir(parents=True, exist_ok=False)
    paths.launcher_log_root.mkdir(parents=True, exist_ok=False)
    state = {"schema": SCHEMA, "experiment_date": date, "batch_id": batch_id, "device": device, **git,
             "matrix_path": MATRIX_PATH.as_posix(),
             "matrix_sha256": json_identity([{key: value for key, value in job.items() if key != "planned_run_dir"}
                                             for job in jobs]),
             "data_identity": data_identity, "data_paths": data_paths, "historical_batch_manifest": historical_reference,
             "training_jobs": records, "status": "running", "new_runs": 0,
             "reused_seed100": sum(row["reuse_status"] == "reused_seed100" for row in records),
             "test_loader_built": 0, "test_evaluation_count": 0, "test_used_for_selection": False}
    current = None
    try:
        write_json_atomic(state_path, state)
        for record in records:
            if record["status"] == "completed":
                continue
            current = record
            raw = configs[record["experiment_id"]]
            seed_everything(device, raw["training"]["seed"])
            runtime = AuditRuntime(TAVFullAudit(raw), datasets.__getitem__, device=device, formal_train_dev=True,
                                   data_identity=data_identity, git_head=git["git_head"])
            execute(runtime, raw, record["run_dir"], data_paths=data_paths, git=git, export_dev_evidence=True)
            checked_run(record["run_dir"], raw, data_identity)
            record.update(status="completed", executed_in_batch=True)
            state["new_runs"] += 1
            current = None
            write_json_atomic(state_path, state)
            del runtime
        state["status"] = "completed"
    except BaseException as error:
        if current is not None:
            current["status"] = "failed"
        state.update(status="failed", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        write_json_atomic(state_path, state)
        write_json_atomic(paths.launcher_log_root / "launcher_result.json", state)
    return state


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--execute", action="store_true", help="launch the formal remote CUDA batch")
    mode.add_argument("--config-only", action="store_true", help="validate configuration and print the plan (default)")
    parser.add_argument("--full-audit-batch-manifest", help="explicit historical batch for seed100 reuse")
    for flag in ("asset-path", "model-manifest-path", "provenance-manifest-path", "batch-id", "experiment-date", "device"):
        parser.add_argument("--" + flag, default="cuda" if flag == "device" else None)
    parser.add_argument("--output-root", default="outputs")
    args = parser.parse_args(argv)
    options = vars(args).copy()
    options.pop("config_only")
    if options.pop("execute"):
        if not all((args.asset_path, args.model_manifest_path, args.provenance_manifest_path, args.batch_id)):
            parser.error("--execute requires the three asset/manifest paths and --batch-id")
        result = run(**options)
    else:
        date, jobs, _, _ = plan(output_root=args.output_root, experiment_date=args.experiment_date)
        result = {"status": "PLAN_ONLY", "experiment_date": date, "expected_identities": len(jobs), "jobs": jobs,
                  "test_evaluation_count": 0, "test_loader_built": 0, "test_used_for_selection": False}
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

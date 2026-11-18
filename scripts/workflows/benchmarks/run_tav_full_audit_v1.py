"""Fixed-order TAV audit; default is a no-data, no-write plan."""

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import yaml

from models.multidag_cl.tav_full_audit_v1 import TAVFullAudit
from scripts.runtime.multidag_cl_ta_v0.formal import git_identity, json_identity, write_json_atomic
from scripts.runtime.multidag_cl_ta_v0.training import seed_everything
from scripts.runtime.multidag_cl_tav_full_audit_v1.spec import CONFIG_ROOT, load_matrices, validate_config
from scripts.runtime.multidag_cl_tav_full_audit_v1.formal import completed_manifest, execute, run_path
from scripts.runtime.multidag_cl_tav_full_audit_v1.runtime import AuditRuntime, load_train_dev
from scripts.runtime.multidag_cl_tav_full_audit_v1.evaluation import evaluate_masks
from utils.output_paths import resolve_experiment_date, resolve_output_paths


def plan(*, matrix_directory=ROOT / CONFIG_ROOT, output_root=None, experiment_date=None):
    training, evaluation = load_matrices(matrix_directory)
    date = resolve_experiment_date(experiment_date)
    jobs = []
    for row in training:
        raw = validate_config(yaml.safe_load((ROOT / row["config_path"]).read_text(encoding="utf-8")))
        jobs.append({**row, "run_dir": run_path(raw, output_root=output_root, experiment_date=date).as_posix()})
    if len({row["run_dir"] for row in jobs}) != 24:
        raise ValueError("fixed run collision within matrix")
    return date, jobs, evaluation


def run(*, asset_path, model_manifest_path, provenance_manifest_path, batch_id, matrix_directory=ROOT / CONFIG_ROOT,
        output_root=None, experiment_date=None, device="cuda"):
    if os.name == "nt":
        raise ValueError("formal matrix is remote-only; Windows permits plan and synthetic checks")
    if device != "cuda" or not torch.cuda.is_available():
        raise ValueError("formal matrix requires CUDA")
    date, jobs, evaluations = plan(matrix_directory=matrix_directory, output_root=output_root, experiment_date=experiment_date)
    datasets, data_identity, data_paths = load_train_dev(asset_path, model_manifest_path, provenance_manifest_path)
    raw_by_id = {row["experiment_id"]: validate_config(yaml.safe_load((ROOT / row["config_path"]).read_text(encoding="utf-8")))
                 for row in jobs}
    # Preflight every collision, including later phases, before the first epoch.
    for row in jobs:
        if Path(row["run_dir"]).exists():
            completed_manifest(row["run_dir"], raw_by_id[row["experiment_id"]], data_identity)
    paths = resolve_output_paths(output_base=output_root or "outputs", experiment_date=date,
                                 experiment_group="tav_full_audit_v1", batch_id=batch_id)
    paths.manifest_root.mkdir(parents=True, exist_ok=False)
    paths.analysis_root.mkdir(parents=True, exist_ok=False)
    paths.launcher_log_root.mkdir(parents=True, exist_ok=False)
    state = {"schema": "tav_full_audit_batch_v1", "batch_id": batch_id, "experiment_date": date,
             "matrix_identity": json_identity({"training": jobs, "evaluation": evaluations}),
             "training_jobs": jobs, "evaluation_jobs": evaluations, "data_identity": data_identity,
             "status": "running", "completed": [], "skipped": [], "evaluated": [],
             "test_loader_built": 0, "test_evaluation_count": 0}
    state_path = paths.manifest_root / "batch_manifest.json"
    git = git_identity()
    try:
        write_json_atomic(state_path, state)
        for row in jobs:
            raw = raw_by_id[row["experiment_id"]]
            if Path(row["run_dir"]).exists():
                completed_manifest(row["run_dir"], raw, data_identity)
                state["skipped"].append(row["experiment_id"])
            else:
                seed_everything(device)
                runtime = AuditRuntime(TAVFullAudit(raw), datasets.__getitem__, device=device,
                                       formal_train_dev=True, data_identity=data_identity, git_head=git["git_head"])
                execute(runtime, raw, row["run_dir"], data_paths=data_paths, git=git)
                state["completed"].append(row["experiment_id"])
                del runtime
            write_json_atomic(state_path, state)
        for row in jobs[:8]:
            raw = raw_by_id[row["experiment_id"]]
            manifest = completed_manifest(row["run_dir"], raw, data_identity)
            seed_everything(device)
            runtime = AuditRuntime(TAVFullAudit(raw), datasets.__getitem__, device=device,
                                   formal_train_dev=True, data_identity=data_identity, git_head=git["git_head"])
            masks = [entry for entry in evaluations if entry["source_experiment_id"] == row["experiment_id"]]
            evaluate_masks(runtime, row["run_dir"], masks, paths.analysis_root / "missing_modality",
                           expected_checkpoint_sha256=manifest["best_checkpoint_sha256"])
            state["evaluated"].extend(entry["evaluation_id"] for entry in masks)
            write_json_atomic(state_path, state)
            del runtime
        state["status"] = "completed"
    except Exception as error:
        state.update(status="failed", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        write_json_atomic(state_path, state)
        write_json_atomic(paths.launcher_log_root / "launcher_result.json", state)
    return state


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix-directory", default=str(ROOT / CONFIG_ROOT))
    parser.add_argument("--execute", action="store_true")
    for name in ("asset-path", "model-manifest-path", "provenance-manifest-path", "output-root", "experiment-date", "batch-id"):
        parser.add_argument("--" + name)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args(argv)
    if args.execute:
        if not all((args.asset_path, args.model_manifest_path, args.provenance_manifest_path, args.batch_id)):
            parser.error("execute requires all asset/manifest paths and an explicit --batch-id")
        result = run(**{key: value for key, value in vars(args).items() if key != "execute"})
    else:
        if any((args.asset_path, args.model_manifest_path, args.provenance_manifest_path, args.batch_id)):
            parser.error("data/batch options require --execute")
        date, jobs, evaluations = plan(matrix_directory=args.matrix_directory, output_root=args.output_root,
                                      experiment_date=args.experiment_date)
        result = {"status": "PLAN_ONLY", "experiment_date": date, "training_count": len(jobs),
                  "dev_evaluation_count": len(evaluations), "jobs": jobs, "test_evaluation_count": 0}
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

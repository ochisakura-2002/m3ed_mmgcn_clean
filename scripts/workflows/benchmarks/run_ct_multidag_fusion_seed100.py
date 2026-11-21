"""Validate or run the four-arm CT-MultiDAG fusion Train/Dev screen."""

import argparse
from functools import partial
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from scripts.runtime.multidag_cl_fusion.runtime import paired_runtime
from scripts.runtime.multidag_cl_fusion.spec import GROUP, load_screen_configs
from scripts.runtime.multidag_cl_ta_v0.formal import git_identity, json_identity, write_json_atomic
from scripts.runtime.multidag_cl_tav_full_audit_v1.confirm import frozen_model_source, test_hold
from scripts.runtime.multidag_cl_tav_full_audit_v1.formal import execute, run_path
from scripts.runtime.multidag_cl_tav_full_audit_v1.runtime import load_train_dev
from scripts.runtime.multidag_cl_tav_full_audit_v1.spec import write_csv
from utils.output_paths import resolve_experiment_date, resolve_output_paths


def plan(*, output_root, experiment_date):
    date = resolve_experiment_date(experiment_date)
    configs = load_screen_configs()
    jobs = [
        {"variant": arm, "experiment_id": raw["experiment_id"], "config_path": path.as_posix(),
         "config_sha256": json_identity(raw),
         "run_dir": run_path(raw, output_root=output_root, experiment_date=date).as_posix()}
        for arm, (path, raw) in configs.items()
    ]
    return date, jobs, configs


def write_reports(directory, jobs):
    summary, classes, mechanisms = [], [], []
    metrics_map = {"Acc": "accuracy", "WF1": "weighted_f1", "MacroF1": "macro_f1", "UAR": "uar"}
    baseline = jobs[0]["result"]["best_dev_metrics"]
    for job in jobs:
        arm, run_dir, result = job["variant"], Path(job["run_dir"]), job["result"]
        report = json.loads((run_dir / "reports" / "fusion_best_dev.json").read_text(encoding="utf-8"))
        if result["best_dev_metrics"] != report["metrics"]:
            raise ValueError(f"{arm} selected Dev report differs from its manifest")
        metrics = report["metrics"]
        summary.append({"variant": arm, "best_epoch": result["best_epoch"],
                        **{short: metrics[key] for short, key in metrics_map.items()},
                        "params": result["parameter_count"],
                        "delta_vs_F0": metrics["weighted_f1"] - baseline["weighted_f1"]})
        router = report["fusion_mechanism"].get("router_by_class", {})
        for row in report["per_class"]:
            classes.append({"variant": arm, **row,
                            **{f"p_{m}": router.get(row["label"], {}).get(f"p_{m}") for m in "AVT"}})
        if arm != "F0":
            for epoch in range(1, result["last_completed_epoch"] + 1):
                path = run_dir / "logs" / f"fusion_mechanism_epoch_{epoch:02d}.json"
                for entity, stats in json.loads(path.read_text(encoding="utf-8")).items():
                    mechanisms.extend({"variant": arm, "epoch": epoch, "entity": entity, "metric": metric,
                                       "value": value} for metric, value in stats.items())

    reports = {"summary": summary, "per_class": classes, "mechanism_stats": mechanisms}
    for name, rows in reports.items():
        write_csv(directory / f"fusion_seed100_{name}.csv", rows)


def execute_screen(*, date, jobs, configs, paths, load_data, git, device, synthetic_epochs=None):
    if synthetic_epochs is not None and device != "cpu":
        raise ValueError("synthetic screen must use CPU")
    jobs = [dict(job, status="PENDING") for job in jobs]
    paths.manifest_root.mkdir(parents=True, exist_ok=False)
    manifest_path = paths.manifest_root / "fusion_seed100_run_manifest.json"
    state = {"schema": "ct_multidag_fusion_seed100_batch_v1", "status": "RUNNING", "batch_id": paths.manifest_root.name,
             "experiment_date": date, "experiment_group": GROUP, "device": device, **git,
             "training_jobs": jobs,
             "test_loader_built": 0, "test_evaluation_count": 0, "test_labels_inspected": 0,
             "test_used_for_selection": False, "synthetic_only": synthetic_epochs is not None}
    current = None
    log_ready = False
    try:
        write_json_atomic(manifest_path, state)
        paths.launcher_log_root.mkdir(parents=True, exist_ok=False)
        log_ready = True
        for job in jobs:
            if os.path.lexists(job["run_dir"]):
                current = job
                raise FileExistsError(job["run_dir"])
        datasets, data_identity, data_paths = load_data()
        state.update(data_identity=data_identity, data_paths=data_paths)
        reference, f0_pairing = None, None
        for job in jobs:
            current = job
            job["status"] = "RUNNING"
            write_json_atomic(manifest_path, state)
            _, raw = configs[job["variant"]]
            runtime, reference = paired_runtime(
                raw, datasets, device=device, reference=reference, f0_pairing=f0_pairing,
                formal_train_dev=synthetic_epochs is None, synthetic_only=synthetic_epochs is not None,
                data_identity=data_identity, git_head=git["git_head"])
            result = execute(runtime, raw, job["run_dir"], data_paths=data_paths, git=git,
                             synthetic_epochs=synthetic_epochs)
            test_hold(result)
            if f0_pairing is None:
                f0_pairing = result["batch_pairing"]
                state["shared_initial_state_sha256"] = result["shared_initialization"]["shared_initial_state_sha256"]
                state["f0_batch_pairing"] = f0_pairing
            if (result["shared_initialization"]["shared_parameter_max_abs_diff"] != 0
                    or result["shared_initialization"]["shared_initial_state_sha256"] != state["shared_initial_state_sha256"]
                    or result["batch_pairing"] != f0_pairing):
                raise ValueError(f"{job['variant']} initialization or batch pairing differs from F0")
            job.update(status="PASS", result=result)
            current = None
            del runtime
            write_json_atomic(manifest_path, state)
        paths.report_root.mkdir(parents=True, exist_ok=False)
        write_reports(paths.report_root, jobs)
        state.update(status="PASS", paired_shared_initialization=True, paired_train_dev_batches=True,
                     reports=paths.report_root.as_posix())
    except BaseException as error:
        if current is not None:
            current["status"] = "FAIL"
        state.update(status="FAIL", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        write_json_atomic(manifest_path, state)
        if log_ready:
            write_json_atomic(paths.launcher_log_root / "launcher_result.json",
                              {"status": state["status"], "manifest": manifest_path.as_posix(),
                               "error": state.get("error")})
    return state


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--execute", action="store_true", help="launch formal remote CUDA training")
    mode.add_argument("--config-only", action="store_true", help="print the data-free execution plan (default)")
    for name in ("asset-path", "model-manifest-path", "provenance-manifest-path", "batch-id", "experiment-date"):
        parser.add_argument("--" + name)
    parser.add_argument("--output-root", default="outputs")
    args = parser.parse_args(argv)
    date, jobs, configs = plan(output_root=args.output_root, experiment_date=args.experiment_date)
    if args.execute:
        if not all((args.asset_path, args.model_manifest_path, args.provenance_manifest_path, args.batch_id)):
            parser.error("--execute requires the three asset/manifest paths and --batch-id")
        if os.name == "nt" or not torch.cuda.is_available():
            parser.error("--execute requires a remote CUDA environment")
        if not frozen_model_source():
            raise ValueError("frozen TAV source lineage cannot be verified")
        paths = resolve_output_paths(output_base=args.output_root, experiment_date=date, experiment_group=GROUP,
                                     batch_id=args.batch_id)
        load_data = partial(load_train_dev, args.asset_path, args.model_manifest_path, args.provenance_manifest_path)
        state = execute_screen(date=date, jobs=jobs, configs=configs, paths=paths,
                               load_data=load_data, git=git_identity(), device="cuda")
        result = {"status": state["status"], "reports": state["reports"],
                  "manifest": (paths.manifest_root / "fusion_seed100_run_manifest.json").as_posix(),
                  "test_loader_built": 0, "test_evaluation_count": 0}
    else:
        result = {"status": "PLAN_ONLY", "experiment_date": date, "jobs": jobs,
                  "test_loader_built": 0, "test_evaluation_count": 0}
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

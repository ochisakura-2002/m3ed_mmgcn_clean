"""Summarize 40 validated selected-Dev runs and the eight frozen paired comparisons."""

import argparse
import json
import math
from pathlib import Path
from statistics import mean, median, stdev
import sys

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analysis.models.multidag_cl.summarize_tav_full_audit_v1 import read_json, write_csv
from scripts.runtime.multidag_cl_ta_v0.formal import json_identity, sha256_file
from scripts.runtime.multidag_cl_tav_full_audit_v1.confirm import (
    CONFIRM_GROUP, CONFIRM_SEEDS, MATRIX_PATH, SCHEMA, checked_run, compatible_historical_source,
    configurations, historical_jobs, load_design, semantic_sha, test_hold,
)
from scripts.runtime.multidag_cl_tav_full_audit_v1.formal import run_path
from utils.output_paths import resolve_output_paths

SCORES = ("Accuracy", "Weighted-F1", "Macro-F1", "UAR")
PAIRS = (
    ("track_tav_cloff", "original_tav_cloff"),
    ("track_tav_cl7", "track_tav_cloff"),
    ("dcn_tav_cloff", "original_tav_cloff"),
    ("dcn_tav_cl7", "original_tav_cloff"),
    ("dcn_tav_cl7", "dcn_tav_cloff"),
    ("original_tav_cl7", "original_tav_cloff"),
    ("original_ta_active_cloff", "original_tav_cloff"),
    ("original_ta_active_cl7", "original_tav_cl7"),
)
TABLE_NAMES = ("TAV_MULTISEED_SEEDWISE_RESULTS.csv", "TAV_MULTISEED_SUMMARY.csv", "TAV_MULTISEED_PAIRED_DELTAS.csv")
# Student-t 0.975 quantile for the frozen five matched seeds (df=4).
T_CRITICAL = 2.7764451051977987


def collect(batch_manifest):
    batch_path = Path(batch_manifest)
    batch = read_json(batch_path)
    test_hold(batch)
    matrix, configs, sources = load_design()
    jobs = batch["training_jobs"]
    if (batch.get("schema") != SCHEMA or batch.get("status") != "completed"
            or batch.get("matrix_path") != MATRIX_PATH.as_posix() or batch.get("matrix_sha256") != json_identity(matrix)
            or batch.get("test_used_for_selection") is not False or len(jobs) != len(matrix)):
        raise ValueError("confirmatory batch is incomplete or has a different identity")
    historical = batch["historical_batch_manifest"]
    historical_runs = {}
    if historical:
        if sha256_file(historical["path"]) != historical["sha256"]:
            raise ValueError("historical batch manifest SHA256 mismatch")
        _, historical_runs = historical_jobs(historical["path"])
    output_root = batch_path.resolve().parents[5]
    layout = resolve_output_paths(output_base=output_root, experiment_date=batch["experiment_date"],
                                  experiment_group=CONFIRM_GROUP, batch_id=batch["batch_id"])
    if batch_path.resolve() != layout.manifest_root / "batch_manifest.json":
        raise ValueError("confirmatory batch manifest path mismatch")
    counts = {"reused_seed100": 0, "new_runs": 0}
    seedwise = []
    for job, row in zip(jobs, matrix):
        if ({key: job.get(key) for key in row.keys() - {"reuse_status"}} !=
                {key: row[key] for key in row.keys() - {"reuse_status"}} or job.get("status") != "completed"):
            raise ValueError("completed run matrix identity mismatch")
        raw = configs[row["experiment_id"]]
        planned_dir = run_path(raw, output_root=output_root, experiment_date=batch["experiment_date"])
        run_dir = Path(job["run_dir"])
        reuse_status = job["reuse_status"]
        if Path(job["planned_run_dir"]).resolve() != planned_dir.resolve():
            raise ValueError("confirmatory planned run path mismatch")
        if reuse_status == "reused_seed100":
            historical_dir = historical_runs.get(row["source_seed100_run"])
            if (not historical_dir or run_dir.resolve() != historical_dir.resolve() or job.get("executed_in_batch")):
                raise ValueError("reused seed100 source is not the recorded historical run")
            raw = sources[row["config_path"]]
            if semantic_sha(raw) != row["config_semantic_sha256"]:
                raise ValueError("reused seed100 semantic identity mismatch")
            counts["reused_seed100"] += 1
        else:
            if run_dir.resolve() != planned_dir.resolve():
                raise ValueError("run points outside its frozen confirmatory path")
            if reuse_status == "existing_confirm_run" and not job.get("executed_in_batch"):
                pass
            elif (job.get("executed_in_batch") is True and reuse_status ==
                  ("rerun_seed100" if row["source_seed100_run"] else "not_applicable")):
                counts["new_runs"] += 1
            else:
                raise ValueError("invalid confirmatory run execution ledger")
        manifest, metrics = checked_run(run_dir, raw, batch["data_identity"])
        if reuse_status == "reused_seed100" and not compatible_historical_source(manifest):
            raise ValueError("reused seed100 source lineage is unverifiable")
        seedwise.append({"configuration": row["configuration"], "seed": int(row["seed"]), **metrics,
                         "best_epoch": manifest["best_epoch"], "parameter_count": manifest["parameter_count"],
                         "checkpoint_sha": manifest["best_checkpoint_sha256"], "experiment_id": row["experiment_id"],
                         "source_run_id": manifest["run_id"], "reuse_status": reuse_status})
    if any(batch.get(key) != value for key, value in counts.items()):
        raise ValueError("confirmatory batch totals disagree with run ledger")
    return batch, seedwise


def descriptive(values):
    return {"n": len(values), "mean": mean(values), "SD": stdev(values), "median": median(values),
            "min": min(values), "max": max(values)}


def tables(seedwise):
    configs = configurations()
    by_seed = {(row["configuration"], int(row["seed"])): row for row in seedwise}
    if len(seedwise) != 40 or set(by_seed) != {(config, seed) for config in configs for seed in CONFIRM_SEEDS}:
        raise ValueError("statistics require the full matched 8-config × 5-seed design")
    summary = []
    for config in configs:
        for metric in (*SCORES, "best_epoch"):
            stats = descriptive([by_seed[config, seed][metric] for seed in CONFIRM_SEEDS])
            summary.append({"configuration": config, "metric": metric, **stats,
                            "mean ± SD": f"{stats['mean']:.6f} ± {stats['SD']:.6f}"})
    paired = []
    for lhs, rhs in PAIRS:
        for metric in SCORES:
            deltas = [100 * (by_seed[lhs, seed][metric] - by_seed[rhs, seed][metric]) for seed in CONFIRM_SEEDS]
            stats = descriptive(deltas)
            radius = T_CRITICAL * stats["SD"] / math.sqrt(stats["n"])
            paired.append({"comparison": f"{lhs} - {rhs}", "metric": metric, "unit": "percentage_points",
                           **{f"delta_s{seed}": delta for seed, delta in zip(CONFIRM_SEEDS, deltas)}, **stats,
                           "wins": sum(delta > 0 for delta in deltas), "ties": deltas.count(0.0),
                           "losses": sum(delta < 0 for delta in deltas),
                           "ci95_low": stats["mean"] - radius, "ci95_high": stats["mean"] + radius,
                           "ci_method": "paired_student_t_df4", "analysis": "DESCRIPTIVE"})
    return dict(zip(TABLE_NAMES, (seedwise, summary, paired)))


def summarize(batch_manifest, *, output_root="outputs", batch_id=None):
    batch, seedwise = collect(batch_manifest)
    results = tables(seedwise)
    paths = resolve_output_paths(output_base=output_root, experiment_date=batch["experiment_date"],
                                 experiment_group=CONFIRM_GROUP, batch_id=batch_id or batch["batch_id"])
    paths.analysis_root.mkdir(parents=True, exist_ok=False)
    for name, rows in results.items():
        write_csv(paths.analysis_root / name, rows)
    return {"analysis_dir": paths.analysis_root.as_posix(), "rows": {name: len(rows) for name, rows in results.items()},
            "complete_identities": len(seedwise), "reused_seed100": batch["reused_seed100"],
            "new_runs": batch["new_runs"], "test_evaluation_count": 0}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-manifest", required=True)
    parser.add_argument("--output-root", default="outputs")
    parser.add_argument("--batch-id")
    args = parser.parse_args(argv)
    print(json.dumps(summarize(**vars(args)), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

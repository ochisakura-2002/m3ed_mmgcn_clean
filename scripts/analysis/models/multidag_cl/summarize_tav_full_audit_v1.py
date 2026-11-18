"""Read only explicit best-Dev artifacts from one frozen TAV batch manifest."""

import argparse
import csv
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.runtime.multidag_cl_ta_v0.formal import sha256_file
from scripts.runtime.multidag_cl_tav_full_audit_v1.spec import design, evaluation_design
from utils.output_paths import resolve_output_paths

METRICS = {"Accuracy": "accuracy", "Weighted-F1": "weighted_f1", "Macro-F1": "macro_f1", "UAR": "uar", "loss": "loss"}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def metric_columns(metrics):
    result = {column: float(metrics[key]) for column, key in METRICS.items()}
    if not all(math.isfinite(value) for value in result.values()):
        raise ValueError("nonfinite Dev result")
    return result


def isolation(value):
    if value.get("test_loader_built") != 0 or value.get("test_evaluation_count") != 0:
        raise ValueError("Test isolation counter mismatch")


def collect(batch_manifest):
    batch_manifest = Path(batch_manifest)
    batch = read_json(batch_manifest)
    isolation(batch)
    expected = design()
    jobs = batch["training_jobs"]
    if len(jobs) != 24 or any({key: job[key] for key in row} != row for job, row in zip(jobs, expected)):
        raise ValueError("batch training matrix identity mismatch")
    if batch["evaluation_jobs"] != evaluation_design():
        raise ValueError("batch evaluation matrix identity mismatch")
    training, manifests = [], {}
    for job in jobs:
        record = {**job, **{column: "" for column in METRICS}, "best_epoch": "", "parameter_count": "", "status": "pending"}
        path = Path(job["run_dir"]) / "manifests" / "run_manifest.json"
        if path.exists():
            manifest = read_json(path)
            isolation(manifest)
            if manifest["config_sha256"] != job["config_sha256"] or manifest["data_identity"] != batch["data_identity"]:
                raise ValueError("run/config/data identity mismatch")
            record.update(status=manifest["status"], best_epoch=manifest.get("best_epoch", ""),
                          parameter_count=manifest["parameter_count"])
            if manifest["status"] == "completed":
                checkpoint = Path(job["run_dir"]) / "checkpoints" / "best_model.pt"
                if manifest.get("synthetic_only") is not False or manifest.get("last_completed_epoch") != 30:
                    raise ValueError("synthetic/partial run cannot be a formal result")
                if sha256_file(checkpoint) != manifest["best_checkpoint_sha256"]:
                    raise ValueError("best checkpoint hash mismatch")
                metrics = read_json(Path(job["run_dir"]) / "reports" / "validation_best_metrics.json")
                if metrics != manifest["best_dev_metrics"]:
                    raise ValueError("best-Dev report/manifest mismatch")
                record.update(metric_columns(metrics))
            manifests[job["experiment_id"]] = manifest
        training.append(record)
    # Exact batch-owned location only. Never search predictions, checkpoint
    # payloads, arbitrary run files, Test labels or Test metrics.
    group_root = batch_manifest.parents[3]
    evaluation_root = group_root / "analysis" / "batches" / batch["batch_id"] / "missing_modality"
    missing = []
    for row in evaluation_design():
        record = {**row, **{column: "" for column in METRICS}, "best_epoch": "", "parameter_count": "", "checkpoint_sha256": ""}
        path = evaluation_root / (row["evaluation_id"] + ".json")
        if path.exists():
            result = read_json(path)
            isolation(result)
            if (any(result[key] != value for key, value in row.items() if key != "status")
                    or result.get("status") != "completed" or result.get("optimizer_steps_performed") != 0):
                raise ValueError("Dev mask identity/status mismatch")
            source = manifests[row["source_experiment_id"]]
            if (source["status"] != "completed" or result["checkpoint_sha256"] != source["best_checkpoint_sha256"]
                    or result["data_identity"] != batch["data_identity"]):
                raise ValueError("Dev mask checkpoint/data mismatch")
            record.update(metric_columns(result["metrics"]), status="completed", best_epoch=result["best_epoch"],
                          parameter_count=result["parameter_count"], checkpoint_sha256=result["checkpoint_sha256"])
        missing.append(record)
    return batch, training, missing


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def summarize(batch_manifest, *, output_root="outputs", batch_id=None):
    batch, training, missing = collect(batch_manifest)
    paths = resolve_output_paths(output_base=output_root, experiment_date=batch["experiment_date"],
                                 experiment_group="tav_full_audit_v1", batch_id=batch_id or batch["batch_id"] + "_summary")
    paths.analysis_root.mkdir(parents=True, exist_ok=False)
    tables = {
        "TAV_FULL_AUDIT_RESULTS.csv": training,
        "TAV_GRAPHCORE_RESULTS.csv": [row for row in training if row["modalities"] == "TAV" and row["context_mode"] == "past_all_causal"],
        "TAV_CL_RESULTS.csv": [row for row in training if row["context_mode"] == "past_all_causal"],
        "TAV_TRAIN_MODALITY_RESULTS.csv": [row for row in training if row["graph_core"] == "original" and row["context_mode"] == "past_all_causal"],
        "TAV_CONTEXT_RESULTS.csv": [row for row in training if row["graph_core"] == "original" and row["curriculum"] == "CL_OFF" and row["modalities"] == "TAV"],
        "TAV_MISSING_MODALITY_RESULTS.csv": missing,
    }
    for name, rows in tables.items():
        write_csv(paths.analysis_root / name, rows)
    return {"analysis_dir": paths.analysis_root.as_posix(), "rows": {name: len(rows) for name, rows in tables.items()},
            "test_evaluation_count": 0}


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

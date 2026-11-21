"""Frozen confirmatory matrix and read-only run validation."""

import csv
import math
from pathlib import Path
import re
import subprocess

import yaml

from scripts.analysis.models.multidag_cl.summarize_tav_full_audit_v1 import metric_columns, read_json
from scripts.runtime.multidag_cl_ta_v0.formal import json_identity, sha256_file
from .spec import (
    CONFIRM_AXES, CONFIRM_GROUP, CONFIRM_SEEDS, CURRICULA, ROOT, confirm_config, design, make_config,
    read_csv, validate_config,
)

MATRIX_PATH = Path("configs/multidag_cl") / CONFIRM_GROUP / "TAV_MULTISEED_CONFIRM_MATRIX.csv"
SCHEMA = "tav_multiseed_confirm_batch_v1"
AUDIT_REFERENCE = "f78171e8c799541ec4fefc1660cd10d281ad5f2f"
SOURCE_SCOPE = (
    "models/multidag_cl", "models/common", "datasets/iemocap/tav_full_audit_dataset.py",
    "scripts/runtime/multidag_cl_paper_reimplementation", "scripts/runtime/multidag_cl_ta_v0",
    "scripts/runtime/multidag_cl_tav_full_audit_v1", "utils/metrics.py",
)
EXECUTION_EDITS = (
    ":(exclude)scripts/runtime/multidag_cl_ta_v0/training.py",
    ":(exclude)scripts/runtime/multidag_cl_tav_full_audit_v1/spec.py",
    ":(exclude)scripts/runtime/multidag_cl_tav_full_audit_v1/runtime.py",
    ":(exclude)scripts/runtime/multidag_cl_tav_full_audit_v1/formal.py",
)


def semantic_sha(raw):
    return json_identity({key: value for key, value in raw.items() if key not in ("output", "experiment_id")})


def architecture_sha(raw):
    return json_identity({key: raw[key] for key in ("architecture_lineage", "frontend_source", "graph_core")})


def configurations():
    return [f"{core}_{'ta_active' if modalities == 'TA' else 'tav'}_{'cloff' if cl == 'CL_OFF' else 'cl7'}"
            for core, modalities in CONFIRM_AXES for cl in CURRICULA]


def matrix_design():
    sources = {row["experiment_id"]: row for row in design()}
    axes = [(core, modalities, cl) for core, modalities in CONFIRM_AXES for cl in CURRICULA]
    rows = []
    for seed in CONFIRM_SEEDS:
        for configuration, (core, modalities, cl) in zip(configurations(), axes):
            source = make_config(core, modalities, cl)
            raw = confirm_config(source, seed)
            historical = source["experiment_id"]
            rows.append({"experiment_id": raw["experiment_id"], "comparison_id": f"paired_s{seed}",
                         "configuration": configuration, "model": raw["model_name"], "graph_core": core,
                         "active_modalities": modalities, "curriculum": cl, "seed": str(seed),
                         "epochs": "30", "config_path": sources[historical]["config_path"],
                         "source_seed100_run": historical if seed == 100 else "",
                         "reuse_status": "candidate" if seed == 100 else "not_applicable",
                         "output_group": CONFIRM_GROUP, "config_sha256": json_identity(raw),
                         "config_semantic_sha256": semantic_sha(raw), "architecture_sha256": architecture_sha(raw)})
    return rows


def load_design():
    rows = read_csv(ROOT / MATRIX_PATH)
    if rows != matrix_design() or len({row["experiment_id"] for row in rows}) != 40:
        raise ValueError("matrix does not match frozen 8-config × 5-seed design")
    sources, configs = {}, {}
    for row in rows:
        path = row["config_path"]
        if path not in sources:
            sources[path] = validate_config(yaml.safe_load((ROOT / path).read_text(encoding="utf-8")))
        raw = validate_config(confirm_config(sources[path], int(row["seed"])))
        if json_identity(raw) != row["config_sha256"]:
            raise ValueError(f"source YAML mismatch for {row['experiment_id']}")
        configs[row["experiment_id"]] = raw
    return rows, configs, sources


def test_hold(record):
    counters = (record.get("test_loader_built"), record.get("test_evaluation_count"), record.get("test_labels_inspected", 0))
    if (any(type(value) is not int or value != 0 for value in counters)
            or record.get("test_used_for_selection", False) is not False):
        raise ValueError("Test HOLD counters or selection flag mismatch")


def checked_epochs(run_dir, best_epoch):
    with (Path(run_dir) / "logs/epoch_metrics.tsv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    if [int(row["epoch"]) for row in rows] != list(range(1, 31)):
        raise ValueError(f"expected 30 completed epoch log rows: {run_dir}")
    selection = [(float(row["dev_weighted_f1"]), -float(row["dev_loss"]), -int(row["epoch"])) for row in rows]
    if not all(math.isfinite(value) for row in selection for value in row):
        raise ValueError(f"nonfinite epoch selector metrics: {run_dir}")
    if type(best_epoch) is not int or best_epoch != -max(selection)[2]:
        raise ValueError(f"best epoch disagrees with frozen Dev selector: {run_dir}")
    return rows[best_epoch - 1]


def checked_completion(run_dir, manifest):
    test_hold(manifest)
    if (manifest.get("status") != "completed" or manifest.get("training_finished") is not True
            or manifest.get("last_completed_epoch") != 30 or manifest.get("synthetic_only") is not False
            or manifest.get("best_checkpoint_reloaded") is not True):
        raise ValueError(f"failed, incomplete or synthetic run: {run_dir}")


def checked_run(run_dir, raw, data_identity, manifest=None):
    run_dir = Path(run_dir)
    if manifest is None:
        manifest = read_json(run_dir / "manifests/run_manifest.json")
    checked_completion(run_dir, manifest)
    resolved = yaml.safe_load((run_dir / "resolved_config.yaml").read_text(encoding="utf-8"))
    expected = {"run_id": raw["experiment_id"], "experiment_id": raw["experiment_id"], "schema": raw["schema"],
                "model_name": raw["model_name"], "architecture_lineage": raw["architecture_lineage"],
                "architecture_identity": architecture_sha(raw), "config_sha256": json_identity(raw),
                "data_identity": data_identity, "seed": raw["training"]["seed"],
                "active_modalities": raw["active_modalities"], "graph_core": raw["graph_core"],
                "context": raw["context_mode"], "curriculum": raw["curriculum"], "training_config": raw["training"]}
    if (any(manifest.get(key) != value for key, value in expected.items()) or resolved["audit_config"] != raw
            or resolved["data_identity"] != data_identity):
        raise ValueError(f"run architecture/config/data identity mismatch: {run_dir}")
    if raw["output"]["experiment_group"] == CONFIRM_GROUP and manifest.get("test_used_for_selection") is not False:
        raise ValueError(f"missing Test selection flag: {run_dir}")
    best = checked_epochs(run_dir, manifest["best_epoch"])
    metrics = read_json(run_dir / "reports/validation_best_metrics.json")
    scores = metric_columns(metrics)
    if (metrics != manifest["best_dev_metrics"] or metrics.get("valid_target_count") != 980
            or any(not 0 <= scores[key] <= 1 for key in scores.keys() - {"loss"})
            or any(not math.isclose(metrics[key], float(best[f"dev_{key}"]), abs_tol=1e-8)
                   for key in ("accuracy", "weighted_f1", "macro_f1", "uar"))
            or not math.isclose(metrics["loss"], float(best["dev_loss"]), rel_tol=1e-5, abs_tol=1e-6)
            or type(manifest.get("parameter_count")) is not int or manifest["parameter_count"] <= 0):
        raise ValueError(f"selected Dev report disagrees with formal run evidence: {run_dir}")
    if manifest["best_checkpoint_sha256"] != sha256_file(run_dir / "checkpoints/best_model.pt"):
        raise ValueError(f"selected checkpoint SHA256 mismatch: {run_dir}")
    return manifest, scores


def historical_jobs(batch_path):
    batch = read_json(batch_path)
    test_hold(batch)
    expected = design()
    jobs = batch["training_jobs"]
    if (batch.get("schema") != "tav_full_audit_batch_v1" or batch.get("status") != "completed"
            or len(jobs) != len(expected)
            or any({key: job.get(key) for key in row} != row for job, row in zip(jobs, expected))):
        raise ValueError("historical Full Audit batch is incomplete or has a different training matrix")
    return batch, {job["experiment_id"]: Path(job["run_dir"]) for job in jobs}


def source_unchanged(*args):
    return subprocess.run(["git", "diff", "--no-ext-diff", "--quiet", *args], cwd=ROOT,
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0


def frozen_model_source():
    # The screening backbone is separate and never imported by the historical model.
    return source_unchanged(AUDIT_REFERENCE, "--", *SOURCE_SCOPE, *EXECUTION_EDITS,
                            ":(exclude)models/multidag_cl/fusion/**",
                            ":(exclude)scripts/runtime/multidag_cl_tav_full_audit_v1/confirm.py")


def compatible_historical_source(manifest):
    """Historical manifests identify source by Git commit and clean-worktree flag."""
    head = manifest.get("git_head", "")
    return (manifest.get("git_dirty") is False and isinstance(head, str) and re.fullmatch(r"[0-9a-f]{40}", head)
            and source_unchanged(head, AUDIT_REFERENCE, "--", *SOURCE_SCOPE))


def reuse_candidate(run_dir, source, target, data_identity):
    manifest = read_json(Path(run_dir) / "manifests/run_manifest.json")
    # Failed/partial runs or a Test violation stop the batch. A complete run
    # without exact reusable identity or a readable intact checkpoint can be rerun.
    checked_completion(run_dir, manifest)
    try:
        if semantic_sha(source) != semantic_sha(target) or not compatible_historical_source(manifest):
            raise ValueError("historical code/config identity cannot be proved")
        checked_run(run_dir, source, data_identity, manifest)
    except (OSError, ValueError, KeyError, TypeError, yaml.YAMLError) as error:
        return False, str(error)
    return True, "matched_frozen_seed100"

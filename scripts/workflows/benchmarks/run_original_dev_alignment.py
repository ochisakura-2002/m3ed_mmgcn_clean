"""Run Stage A (CE identity) or Stage B (LS fit, fixed endpoints and reporting)."""

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import torch

from scripts.analysis.models.multidag_cl.export_fusion_topology_bundle import MANIFEST_NAME, pack_verified
from scripts.analysis.models.multidag_cl.group_isolated_comparison import population as prediction_population
from scripts.analysis.models.multidag_cl.original_dev_analysis import INTERPRETATION, replay, stage_tables, write_results
from scripts.runtime.multidag_cl_fusion.original_dev_alignment import (
    CONFIG, GROUP, PROTOCOL, REFERENCE_FILES, ROOT, STEPS, AlignmentRuntime, epoch_rows, load_inputs, preflight,
    read_config, read_reference, read_state, select_budget,
)
from scripts.runtime.multidag_cl_group_isolated.fixed_tail_average import validate_state
from scripts.runtime.multidag_cl_group_isolated.protocol import CONFIG_DIR, ENDPOINT, HOLD, read_frozen, read_json, validate_hold
from scripts.runtime.multidag_cl_group_isolated.provenance import runtime_provenance
from scripts.runtime.multidag_cl_paper_reimplementation.checkpoint import load_checkpoint
from scripts.runtime.multidag_cl_ta_v0.formal import git_identity, json_identity, sha256_file, write_json_atomic
from scripts.runtime.multidag_cl_tav_full_audit_v1.confirm import SOURCE_SCOPE, checked_completion, checked_epochs
from scripts.runtime.multidag_cl_tav_full_audit_v1.formal import execute
from utils.output_paths import resolve_output_paths


RUN_EVIDENCE = ("resolved_config.yaml", "manifests/run_manifest.json", "manifests/alignment_snapshots.json",
                "manifests/curriculum_bucket_manifest.tsv", "logs/epoch_metrics.tsv",
                "reports/validation_best_dev_evidence.json")
PLOT_SCRIPT = ROOT / "scripts/analysis/models/multidag_cl/plot_original_dev_alignment.py"
AUDITED_SOURCE = "aa4fa63d75cc4651e18d7f27315a6c4a624e4783"


def check_frozen_source():
    scope = (*SOURCE_SCOPE, "scripts/runtime/multidag_cl_fusion", "datasets/iemocap/ta_v0_dataset.py",
             ":(exclude)scripts/runtime/multidag_cl_fusion/original_dev_alignment.py")
    if subprocess.run(["git", "diff", "--quiet", AUDITED_SOURCE, "--", *scope], cwd=ROOT).returncode != 0:
        raise ValueError("F0 source differs from the audited G4 freeze checkout")


def arm_run(paths, arm):
    suffix = paths.manifest_root.name
    return paths.runs_root / f"{arm.lower()}_seed100_{suffix}"


def provenance():
    sources = (CONFIG, PLOT_SCRIPT, ROOT / CONFIG_DIR / "fold_identities.json")
    result = runtime_provenance(ROOT, "cuda", sources)
    driver = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                            capture_output=True, text=True, check=False) if shutil.which("nvidia-smi") else None
    result["environment"].update(
        gpu_driver=driver.stdout.splitlines() if driver and driver.returncode == 0 else "UNKNOWN",
        torch_build=torch.__config__.show(), gpu_name=torch.cuda.get_device_name(),
    )
    return result


def make_runtime(raw, datasets, run, config, arm, lock, historical, git, *, expected, historical_epochs=None,
                 synthetic=False):
    experiment = {**HOLD, "protocol": PROTOCOL, "arm": arm, "run_id": run.name, "config_sha256": json_identity(config),
                  "train_label_smoothing": config["train_label_smoothing"][arm],
                  "budget_updates": lock.get("selected_updates"), "budget_sha256": json_identity(lock)}
    runtime = AlignmentRuntime(
        raw, datasets, run, experiment,
        options={"device": "cpu" if synthetic else "cuda", "formal_train_dev": not synthetic,
                 "synthetic_only": synthetic, "data_identity": historical["data_identity"], "git_head": git["git_head"]},
        pairing=expected["batch_pairing"], historical_epochs=historical_epochs,
    )
    return runtime


def check_runtime(runtime, reference):
    mismatches = {
        key: {"historical": value, "actual": runtime.initialization.get(key)}
        for key, value in reference["shared_initialization"].items()
        if runtime.initialization.get(key) != value
    }
    if mismatches or runtime.curriculum.manifest.membership_sha256 != reference["curriculum_membership_sha256"]:
        raise ValueError(f"initialization/RNG or Train curriculum membership differs: {mismatches}")
    if runtime.pairing["population"] != reference["batch_pairing"]["population"] or runtime.steps_per_epoch != STEPS:
        raise ValueError("Train/Dev population or steps/epoch differs from F0")


def fit(runtime, raw, git, projection, *, synthetic_epochs=None):
    run = runtime.run_dir
    try:
        execute(runtime, raw, run, git=git, synthetic_epochs=synthetic_epochs,
                experiment=runtime.experiment, data_paths={"train_dev_projection": projection["sha256"]})
    finally:
        if runtime.snapshots:
            write_json_atomic(run / "manifests/alignment_snapshots.json", runtime.snapshots)


def checked_arm(runtime, *, synthetic_epochs=None):
    run = runtime.run_dir
    manifest = read_json(run / "manifests/run_manifest.json")
    evidence = read_json(run / "reports/validation_best_dev_evidence.json")
    snapshots = read_json(run / "manifests/alignment_snapshots.json")
    validate_hold(manifest, legacy=True)
    validate_hold(evidence, legacy=True)
    epochs = ENDPOINT if synthetic_epochs is None else synthetic_epochs
    if synthetic_epochs is None:
        checked_completion(run, manifest)
        checked_epochs(run, manifest["best_epoch"])
    experiment = runtime.experiment
    expected = {
        f"epoch_{epoch:02d}": (epoch, epoch * runtime.steps_per_epoch, True) for epoch in range(1, epochs + 1)
    }
    update = experiment["budget_updates"]
    if update is not None:
        epoch, position = divmod(update - 1, runtime.steps_per_epoch)
        expected["u_star"] = (epoch + 1, update, position + 1 == runtime.steps_per_epoch)
    if (manifest["run_id"] != run.name or manifest["experiment"] != experiment
            or manifest["config_sha256"] != json_identity(runtime.model.audit_config)
            or manifest["data_identity"] != runtime.data_identity or manifest["last_completed_epoch"] != epochs
            or manifest["optimizer_steps"] != epochs * runtime.steps_per_epoch
            or manifest["batch_pairing"] != runtime.f0_pairing
            or manifest["shared_initialization"] != runtime.initialization
            or manifest["best_dev_metrics"] != evidence["metrics"] or evidence["split"] != "dev"
            or manifest["best_checkpoint_sha256"] != evidence["checkpoint_sha256"]
            or manifest["best_checkpoint_sha256"] != sha256_file(run / "checkpoints/best_model.pt")
            or evidence["optimizer_steps_performed"] != 0 or set(snapshots) != set(expected)):
        raise ValueError("run or checkpoint inventory differs from the D0/D1 protocol")
    for name, record in snapshots.items():
        if tuple(record[key] for key in ("epoch", "optimizer_steps", "epoch_complete")) != expected[name]:
            raise ValueError(f"wrong epoch or optimizer position: {name}")
        read_state(run, name, record, experiment, runtime.model)
    return {"run": manifest, "best_evidence": evidence, "experiment": experiment, "snapshots": snapshots}


def numerical_audit(run, historical_epochs, historical_evidence, record):
    rows = epoch_rows(run)
    mismatch = []
    for actual, expected in zip(rows, historical_epochs):
        for key, value in expected.items():
            if (value != actual[key] if key == "is_best" else float(value) != float(actual[key])):
                item = {"epoch": int(expected["epoch"]), "field": key, "f0": value, "d0": actual[key]}
                if key != "is_best":
                    item["delta"] = float(actual[key]) - float(value)
                mismatch.append(item)
    best = record["best_evidence"]
    expected_predictions, predictions = historical_evidence["predictions"], best["predictions"]
    same_population = prediction_population(expected_predictions) == prediction_population(predictions)
    exact = {}
    for key in ("logits", "probabilities", "predicted_label"):
        a, b = (np.asarray([row[key] for row in items]) for items in (expected_predictions, predictions))
        if a.shape != b.shape:
            exact[key] = {"exact": False, "historical_shape": list(a.shape), "actual_shape": list(b.shape)}
        else:
            exact[key] = {"exact": np.array_equal(a, b), "different_elements": int((a != b).sum()),
                          "max_abs_delta": float(np.abs(a - b).max())}
    correct = sum(row["true_label"] == row["predicted_label"] for row in predictions)
    comparison = {
        "best_epoch": record["run"]["best_epoch"], "best_dev": best["metrics"],
        "best_correct": correct, "e30_weighted_f1": float(rows[-1]["dev_weighted_f1"]),
        "epoch_log_differences": mismatch, "population_exact": same_population,
        "prediction_comparisons": exact, "historical_best_dev": historical_evidence["metrics"],
        "historical_e30_weighted_f1": float(historical_epochs[-1]["dev_weighted_f1"]),
    }
    comparison["numerical_identity"] = (len(rows) == len(historical_epochs) == 30 and not mismatch and same_population
        and all(item["exact"] for item in exact.values()) and correct == 693 and comparison["best_epoch"] == 21
        and all(best[k] == historical_evidence[k] for k in ("metrics", "per_class", "confusion_matrix")))
    return comparison


def compare_historical_weights(path, runtime, historical):
    if not path.is_file():
        return "UNKNOWN_HISTORICAL_CHECKPOINT_BYTES_UNAVAILABLE"
    if sha256_file(path) != historical["best_checkpoint_sha256"]:
        raise ValueError("historical F0 selected checkpoint SHA-256 mismatch")
    checkpoint = load_checkpoint(path, torch.device("cpu"))
    runtime.validate_checkpoint(checkpoint)
    state = checkpoint["model_state_dict"]
    current = runtime.model.state_dict()
    validate_state(state, current)
    if any(not torch.equal(value.cpu(), state[key]) for key, value in current.items() if isinstance(value, torch.Tensor)):
        raise ValueError("actual D0 selected checkpoint tensors differ from historical F0")
    return "MATCH"


def ship(directory, archive, status):
    files = {p.relative_to(directory).as_posix(): sha256_file(p) for p in sorted(directory.rglob("*")) if p.is_file()}
    forbidden = {".pt", ".pth", ".pkl", ".npy", ".npz", ".ckpt", ".pickle"}
    if any(Path(name).suffix in forbidden or "checkpoints" in Path(name).parts for name in files):
        raise ValueError("weights or feature data in lightweight package")
    (directory / "SHA256SUMS.txt").write_text("".join(f"{sha}  {name}\n" for name, sha in files.items()),
                                              encoding="utf-8")
    files["SHA256SUMS.txt"] = sha256_file(directory / "SHA256SUMS.txt")
    write_json_atomic(directory / MANIFEST_NAME, {**HOLD, "protocol": PROTOCOL, "status": status, "files": files})
    return pack_verified(directory, archive, files)


def copy_artifacts(run, destination):
    for relative in RUN_EVIDENCE:
        path = run / relative
        if path.is_file():
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)


def require_fresh(*paths):
    for path in paths:
        if os.path.lexists(path):
            raise FileExistsError(path)


def prepare_stage(args, frozen, raw):
    historical, evidence, epochs = read_reference(args.reference_run, frozen, raw)
    datasets, entries, projection, summaries = load_inputs(args.projection_dir, frozen, historical, evidence)
    return historical, evidence, epochs, datasets, entries, projection, summaries


def stage_a(args, paths, frozen, raw, config):
    run = arm_run(paths, "D0")
    stage = paths.review_root
    archive = stage.parent / f"{paths.manifest_root.name}_stage_a.tar.gz"
    require_fresh(run, archive, stage, paths.manifest_root)
    paths.manifest_root.mkdir(parents=True, exist_ok=False)
    stage.mkdir(parents=True, exist_ok=False)
    receipt = paths.manifest_root / "stage_a.json"
    state = {**HOLD, "protocol": PROTOCOL, "run_id": run.name, "status": "RUNNING",
             "f0_reference_sha256": REFERENCE_FILES, "historical_environment": "UNKNOWN",
             "audited_source_base": AUDITED_SOURCE, "upstream_uncertified_provenance": "UNKNOWN",
             "config_sha256": json_identity(config)}
    write_json_atomic(receipt, state)
    try:
        historical, evidence, epochs, datasets, entries, projection, summaries = prepare_stage(args, frozen, raw)
        lock = select_budget(config, frozen, raw, args.historical_ce_root, train=datasets["train"])
        lock_file = paths.manifest_root / "budget_lock.json"
        write_json_atomic(lock_file, lock)
        shutil.copyfile(lock_file, stage / lock_file.name)
        state.update(budget_sha256=sha256_file(lock_file), d3_status=lock["status"],
                     projection=projection, populations=summaries)
        git = git_identity()
        runtime = make_runtime(raw, datasets, run, config, "D0", lock, historical, git, expected=historical,
                               historical_epochs=epochs)
        state.update(runtime=provenance(), initialization=runtime.initialization,
                     curriculum_membership_sha256=runtime.curriculum.manifest.membership_sha256,
                     optimizer={"type": type(runtime.optimizer).__name__,
                                "parameter_groups": runtime.optimizer.state_dict()["param_groups"]})
        write_json_atomic(receipt, state)
        check_runtime(runtime, historical)
        state["preflight"] = preflight(runtime, historical["batch_pairing"], entries)
        write_json_atomic(receipt, state)
        fit(runtime, raw, git, projection)
        record = checked_arm(runtime)
        state["numerical_audit"] = numerical_audit(run, epochs, evidence, record)
        if not state["numerical_audit"]["numerical_identity"]:
            raise ValueError("D0 and archived F0 do not have exact numerical identity")
        state["historical_best_weight_tensors"] = compare_historical_weights(
            args.reference_run / "checkpoints/best_model.pt", runtime, historical)
        state.update(status="PASS", run_evidence_sha256={name: sha256_file(run / name) for name in RUN_EVIDENCE})
    except BaseException as error:
        state.update(status="FAIL", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        write_json_atomic(receipt, state)
        shutil.copyfile(receipt, stage / receipt.name)
        copy_artifacts(run, stage / "d0")
        archive_sha = ship(stage, archive, state["status"])
        print(json.dumps({"stage": "A", "status": state["status"], "receipt": str(receipt),
                          "receipt_sha256": sha256_file(receipt), "archive": str(archive),
                          "archive_sha256": archive_sha}, ensure_ascii=False), flush=True)


def checked_approval(paths, approved_sha, config):
    receipt = paths.manifest_root / "stage_a.json"
    if sha256_file(receipt) != approved_sha:
        raise ValueError("Stage A receipt SHA-256 does not match the reviewed SHA-256")
    state = read_json(receipt)
    validate_hold(state)
    if (state["status"] != "PASS" or state["config_sha256"] != json_identity(config)
            or not state["numerical_audit"]["numerical_identity"]):
        raise ValueError("Stage A numerical identity gate has not passed")
    run = arm_run(paths, "D0")
    if state["run_id"] != run.name or set(state["run_evidence_sha256"]) != set(RUN_EVIDENCE):
        raise ValueError("reviewed D0 artifact inventory mismatch")
    if any(sha256_file(run / name) != digest for name, digest in state["run_evidence_sha256"].items()):
        raise ValueError("reviewed D0 run evidence changed after approval")
    lock_file = paths.manifest_root / "budget_lock.json"
    if sha256_file(lock_file) != state["budget_sha256"]:
        raise ValueError("pre-training D3 budget lock changed after approval")
    lock = read_json(lock_file)
    validate_hold(lock)
    if (lock["status"] != state["d3_status"] or lock["candidate_updates"] != config["budget_candidate_updates"]
            or (lock["status"] == "LOCKED" and lock["selected_updates"] not in lock["candidate_updates"])):
        raise ValueError("reviewed budget recipe mismatch")
    return state, lock


def stage_b(args, paths, frozen, raw, config):
    from scripts.analysis.models.multidag_cl.plot_original_dev_alignment import render, select_font

    approved, lock = checked_approval(paths, args.approved_stage_a_sha256, config)
    d0, d1 = (arm_run(paths, arm) for arm in ("D0", "D1"))
    destination = paths.analysis_root
    archive = destination.parent / f"{paths.manifest_root.name}_stage_b.tar.gz"
    require_fresh(d1, destination, archive)
    select_font(args.figure_font)
    destination.mkdir(parents=True, exist_ok=False)
    status_file = paths.manifest_root / "stage_b.json"
    with status_file.open("x", encoding="utf-8") as stream:
        stream.write("{}\n")
    state = {**HOLD, "protocol": PROTOCOL, "status": "RUNNING",
             "stage_a_sha256": args.approved_stage_a_sha256, "d3_status": lock["status"]}
    write_json_atomic(status_file, state)
    try:
        historical, _, _, datasets, entries, projection, _ = prepare_stage(args, frozen, raw)
        if projection != approved["projection"]:
            raise ValueError("Stage B projection differs from reviewed Stage A")
        git = git_identity()
        baseline = make_runtime(raw, datasets, d0, config, "D0", lock, historical, git, expected=historical)
        check_runtime(baseline, historical)
        record = checked_arm(baseline)
        training_reference = record["run"]
        ls = make_runtime(raw, datasets, d1, config, "D1", lock, historical, git, expected=training_reference)
        check_runtime(ls, training_reference)
        state["runtime"] = provenance()
        for key in ("files_sha256", "environment"):
            if state["runtime"][key] != approved["runtime"][key]:
                raise ValueError(f"Stage B {key} differs from the reviewed Stage A")
        state["preflight"] = preflight(ls, training_reference["batch_pairing"], entries)
        write_json_atomic(status_file, state)
        fit(ls, raw, git, projection)
        ls_record = checked_arm(ls)
        del ls
        ce, ce_sources, ce_learning = replay(baseline, d0, record, entries, config["average_epochs"])
        del baseline
        ls = make_runtime(raw, datasets, d1, config, "D1", lock, historical, git, expected=training_reference)
        smoothed, ls_sources, ls_learning = replay(ls, d1, ls_record, entries, config["average_epochs"])
        tables = destination / "tables"
        tables.mkdir()
        scores, intervals = stage_tables(tables, {**ce, **smoothed}, {**ce_sources, **ls_sources},
                                         ce_learning + ls_learning, entries["dev"], lock)
        state["render"] = render(tables, destination / "figures", font=args.figure_font)
        write_results(destination / "RESULTS.md", scores, intervals, lock)
        shutil.copyfile(PLOT_SCRIPT, destination / PLOT_SCRIPT.name)
        for name in ("stage_a.json", "budget_lock.json"):
            shutil.copyfile(paths.manifest_root / name, destination / name)
        copy_artifacts(d0, destination / "d0")
        copy_artifacts(d1, destination / "d1")
        state.update(status="PASS", endpoint_count=len(ce) + len(smoothed), scope=INTERPRETATION)
        write_json_atomic(destination / status_file.name, state)
        state.update(archive=str(archive), archive_sha256=ship(destination, archive, state["status"]))
    except BaseException as error:
        state.update(status="FAIL", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        write_json_atomic(status_file, state)
    print(json.dumps({k: state[k] for k in ("status", "endpoint_count", "d3_status", "archive", "archive_sha256")},
                     ensure_ascii=False), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("plan", "stage-a", "stage-b"))
    for name in ("reference-run", "projection-dir", "historical-ce-root", "figure-font"):
        parser.add_argument("--" + name, type=Path)
    for name in ("approved-stage-a-sha256", "experiment-date", "batch-id"):
        parser.add_argument("--" + name)
    args = parser.parse_args(argv)
    frozen, raw = read_frozen(ROOT)
    config = read_config(frozen)
    if args.stage == "plan":
        result = {"config": config, "f0_config_sha256": json_identity(raw), "steps_per_epoch": STEPS,
                  "stage_a": "D0 CE", "stage_b": "D1 LS + D2/D3/D4 fixed endpoints, analysis, figures and archive"}
        if args.reference_run is not None:
            historical, _, _ = read_reference(args.reference_run, frozen, raw)
            result["historical_f0_reference"] = {"status": "VERIFIED", "best_epoch": historical["best_epoch"],
                                                  **historical["best_dev_metrics"]}
        print(json.dumps(result, indent=2))
        return 0
    if not all((args.reference_run, args.projection_dir, args.experiment_date, args.batch_id)):
        parser.error("Stage A/B require --reference-run, --projection-dir, --experiment-date and --batch-id")
    if args.stage == "stage-b" and not args.approved_stage_a_sha256:
        parser.error("Stage B requires --approved-stage-a-sha256 from the reviewed receipt")
    if os.name == "nt" or not torch.cuda.is_available():
        parser.error("formal Stage A/B require the Ubuntu CUDA runtime")
    subprocess.run([sys.executable, str(ROOT / "scripts/dev/check_workstream.py"), "multidag"], cwd=ROOT, check=True)
    check_frozen_source()
    paths = resolve_output_paths(output_base=config["output"]["root"], experiment_date=args.experiment_date,
                                 experiment_group=GROUP, batch_id=args.batch_id)
    action = stage_a if args.stage == "stage-a" else stage_b
    action(args, paths, frozen, raw, config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

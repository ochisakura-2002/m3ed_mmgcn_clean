"""Run the registered seed100 PIL screen with D0 authentication and analysis."""

# Load the D training dependency closure before importing analysis-only packages.
from scripts.workflows.benchmarks import run_original_dev_alignment as d

import argparse
import csv
import io
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tarfile
import traceback
from uuid import uuid4

import torch
import yaml

from models.multidag_cl.ta_v0 import LABEL_VOCAB
from scripts.analysis.models.multidag_cl.pairwise_preflight_inputs import ASSET, MODEL_MANIFEST, PROJECTION_MANIFEST
from scripts.analysis.models.multidag_cl.pil_screen_analysis import export_analysis, plot_tables
from scripts.runtime.multidag_cl_pil_screen import (
    ARMS, AUX_WEIGHT, CONTROL, InvalidRun, PILRuntime, build_target_tape,
    device_parity, read_completed_arm, train_arm, validate_target_tape,
)
from scripts.runtime.multidag_cl_group_isolated.evaluation import validate_predictions
from scripts.runtime.multidag_cl_group_isolated.protocol import ENDPOINT, HOLD, read_frozen, read_json
from scripts.runtime.multidag_cl_ta_v0.formal import json_identity, sha256_file, write_json_atomic
from scripts.runtime.multidag_cl_ta_v0.training import classification_report
from scripts.workflows.benchmarks.original_dev_stage_b import SOURCE_POLICY, batch_lock, canonical, certify_sources
from utils.output_paths import resolve_output_paths


ROOT = d.ROOT
CONFIG = "configs/multidag_cl/pil_screen/seed100.yaml"
TASK = "PIL_SCREEN_001"
GROUP = "pil_screen_001"
D0_HEAD = "25271aea2d9c13aed1fccac4849aa86c3b68822c"
D0_BATCH = "d_alignment_seed100_132745"
D0_ROOT = Path("outputs/20261127/original_dev_alignment")
F0_RUN = Path("outputs/20261123/fusion_final_paired_5seed_confirmation/runs/tav_track_fusion_final_f0_s100")
STAGE_A_SHA = "0b0938bddb9cc99a7075896af0fa90b87ea068cfbbe5c3336d48c51d726235e6"
STAGE_B_SHA = "6b36f5ba0dd521355a48d3894770d31a76aa0d34dbd17ff7d0f1d0f9c20ba016"
SOURCE_POLICY_SHA = "68668add5ff0cc6bd15bbbdf56e7b42b25f8c1956d19fbf43842c4a4c1bbec11"
PROJECTION_SHA = {ASSET: "a9f23d0fb2b59abf156222a93e29896a203cf4948908ec3bbbb99cf37744c466",
                  MODEL_MANIFEST: "9a474e94539154ba4ed72a17b6815c07b9e7b5c12605d71a5e63900744366c43"}
ISOLATED_SOURCES = {CONFIG, "models/experimental/multidag_cl_pil_screen.py", "scripts/runtime/multidag_cl_pil_screen.py",
                    "scripts/workflows/benchmarks/run_pil_screen.py",
                    "scripts/analysis/models/multidag_cl/pil_screen_analysis.py"}
PIL_CLOSURE = ISOLATED_SOURCES | {"models/experimental/__init__.py"}


def git_bytes(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT)


def recipe():
    config = yaml.safe_load((ROOT / CONFIG).read_bytes())
    expected = {"task": TASK, "protocol": d.PROTOCOL, "seed": 100, "arms": list(ARMS),
                "shared_control": "certify_d0_else_one_off", "aux_weight": AUX_WEIGHT,
                "null_seed": 10007, "test": "HOLD",
                "output": {"root": "outputs", "experiment_date": None, "experiment_group": GROUP}}
    if canonical(config) != canonical(expected):
        raise InvalidRun("PIL YAML differs from the registered screen")
    return config


def source_environment(stage):
    actual = d.provenance()
    actual["files_sha256"].update({name: sha256_file(ROOT / name) for name in ISOLATED_SOURCES})
    old = stage["runtime"]["files_sha256"]
    current = actual["files_sha256"]
    if sha256_file(ROOT / SOURCE_POLICY) != SOURCE_POLICY_SHA:
        raise InvalidRun("reviewed D source policy SHA differs")
    compatibility = certify_sources(ROOT, STAGE_A_SHA, old, {k: v for k, v in current.items()
                                                             if k not in PIL_CLOSURE})
    d.check_environment(stage["runtime"]["environment"], actual["environment"],
                        diagnostics=actual["process_diagnostics"])
    if "process_diagnostics" in stage["runtime"]:
        d.check_environment(stage["runtime"]["process_diagnostics"], actual["process_diagnostics"])
    gpu = torch.cuda.get_device_properties(torch.cuda.current_device())
    if (platform.python_version() != "3.9.25" or torch.__version__ != "2.6.0+cu118"
            or "V100" not in gpu.name or not 14 * 2**30 < gpu.total_memory <= 17 * 2**30):
        raise InvalidRun("expected the reviewed Python 3.9.25 / Torch 2.6.0+cu118 / V100 16GB environment")
    tracked = set(git_bytes("ls-files", "--", *sorted(PIL_CLOSURE)).decode().splitlines())
    if tracked != PIL_CLOSURE or git_bytes("diff", "--binary", "HEAD", "--", *sorted(current), SOURCE_POLICY.as_posix()):
        raise InvalidRun("training sources must be tracked and identical to the synchronized HEAD")
    return actual, compatibility


def read_history_archive(path, dev_entries, logs, best):
    if sha256_file(path) != STAGE_B_SHA:
        raise InvalidRun("D Stage B bundle SHA differs from the final reviewed archive")
    with tarfile.open(path, "r:gz") as bundle:
        def rows(name):
            member = bundle.getmember(f"tables/{name}.csv")
            if not member.isfile():
                raise InvalidRun("expected a regular D trajectory table")
            with bundle.extractfile(member) as stream:
                return list(csv.DictReader(io.TextIOWrapper(stream, encoding="utf-8")))

        learning = [row for row in rows("trajectory") if row["arm"] == "CE"]
        evidence = [row for row in rows("predictions") if row["arm"] == "CE E30"]
    grid = {(int(row["epoch"]), row["population"]): row for row in learning}
    if len(learning) != 60 or set(grid) != {(i, role) for i in range(1, 31) for role in ("Fit", "Dev")}:
        raise InvalidRun("D0 fixed-weight learning curves are incomplete")
    history = []
    for i, log in enumerate(logs, 1):
        fit, dev = (grid[i, role] for role in ("Fit", "Dev"))
        if any(float(dev[key]) != float(log[f"dev_{key}"]) for key in d.CLASSIFICATION):
            raise InvalidRun("D0 archived trajectory disagrees with the certified Dev log")
        if (float(dev["formal_hard_ce"]) != float(log["dev_loss"])
                or float(fit["train_update_loss"]) != float(log["train_loss"])
                or (int(fit["valid_target_count"]), int(dev["valid_target_count"])) != (4830, 980)):
            raise InvalidRun("D0 archived hard CE, objective, or population disagrees")
        history.append({"epoch": i, "fit": {**{k: float(fit[k]) for k in d.CLASSIFICATION},
                                            "loss": float(fit["formal_hard_ce"]), "valid_target_count": 4830},
                        "dev": {**{k: float(dev[k]) for k in d.CLASSIFICATION},
                                "loss": float(dev["formal_hard_ce"]), "valid_target_count": 980},
                        "steps": [], "historical_train_update_loss": float(log["train_loss"])})
    predictions = [{"utterance_id": row["utterance_id"], "dialogue_id": row["dialogue_id"],
                    "true_label": int(row["true_label_id"]), "predicted_label": int(row["predicted_label_id"]),
                    "logits": [float(row[f"logit_{k}"]) for k in LABEL_VOCAB],
                    "probabilities": [float(row[f"prob_{k}"]) for k in LABEL_VOCAB]} for row in evidence]
    validate_predictions(predictions, dev_entries)
    report = classification_report([r["true_label"] for r in predictions], [r["predicted_label"] for r in predictions],
                                   loss=history[-1]["dev"]["loss"])
    if report["metrics"] != history[-1]["dev"]:
        raise InvalidRun("archived D0 E30 predictions and trajectory disagree")
    return {"Best": best, "E30": {**report, "predictions": predictions}, "history": history}


def certify_d0(d_config, stage, historical, f0_best, f0_logs, entries):
    paths = resolve_output_paths(output_base="outputs", experiment_date="20261127",
                                 experiment_group=d.GROUP, batch_id=D0_BATCH)
    reviewed, lock = d.checked_approval(paths, STAGE_A_SHA, d_config)
    run = d.arm_run(paths, "D0")
    manifest = read_json(run / "manifests/run_manifest.json")
    best = read_json(run / "reports/validation_best_dev_evidence.json")
    rows = d.epoch_rows(run)
    resolved = yaml.safe_load((run / "resolved_config.yaml").read_bytes())
    d.checked_completion(run, manifest)
    d.checked_epochs(run, 21)
    d.validate_hold(best, legacy=True)
    identity_fields = ("config_sha256", "data_identity", "architecture_identity", "shared_initialization",
                       "curriculum_membership_sha256", "batch_pairing", "training_config", "active_modalities")
    if (reviewed != stage or manifest["git_head"] != D0_HEAD or manifest["run_id"] != run.name
            or manifest["optimizer_steps"] != 210 or manifest["best_epoch"] != 21
            or sha256_file(run / "checkpoints/best_model.pt") != manifest["best_checkpoint_sha256"]
            or manifest["best_checkpoint_sha256"] != best["checkpoint_sha256"]
            or resolved["experiment"] != manifest["experiment"]
            or manifest["experiment"]["protocol"] != d.PROTOCOL or manifest["experiment"]["arm"] != "D0"
            or manifest["experiment"]["train_label_smoothing"] != 0.0
            or manifest["experiment"]["budget_sha256"] != json_identity(lock)
            or any(manifest[key] != historical[key] for key in identity_fields)):
        raise InvalidRun("D0 does not match the certified original F0 identity")
    validate_predictions(best["predictions"], entries["dev"])
    audit = d.numerical_audit(run, f0_logs, f0_best, {"run": manifest, "best_evidence": best})
    if not audit["numerical_identity"] or best["metrics"] != manifest["best_dev_metrics"]:
        raise InvalidRun("D0 lacks exact E21 logits or original epoch trajectory")
    archive = D0_ROOT / "analysis/batches" / f"{D0_BATCH}_stage_b.tar.gz"
    result = read_history_archive(archive, entries["dev"], rows, best)
    return {**result, "best_epoch": 21, "parameter_count": manifest["parameter_count"],
            "experiment": {"arm": "D0", "run_id": run.name}, "checkpoints": {
                str(run / "checkpoints/best_model.pt"): manifest["best_checkpoint_sha256"]}}


def verify_shared_control(result, f0_logs, f0_best):
    """Apply the exact numerical identity standard registered in D Stage A."""
    for record, row in zip(result["history"], f0_logs):
        steps = record["steps"]
        count = sum(step["valid_targets"] for step in steps)
        actual = {"train_loss": sum(step["objective"] * step["valid_targets"] for step in steps) / count,
                  "gradient_norm": max(step["gradient_norm"] for step in steps),
                  "train_targets": count, "optimizer_steps": steps[-1]["step"],
                  **{f"dev_{key}": record["dev"][key] for key in ("loss", *d.CLASSIFICATION)}}
        if any(value != float(row[key]) for key, value in actual.items()):
            raise InvalidRun(f"REVIEW: shared CTRL exceeds exact D tolerance at E{record['epoch']:02d}")
    for key in ("metrics", "per_class", "confusion_matrix", "predictions"):
        if result["Best"][key] != f0_best[key]:
            raise InvalidRun(f"REVIEW: shared CTRL Best {key} differs from F0 E21")
    if result["best_epoch"] != 21 or len(result["history"]) != len(f0_logs) or len(f0_logs) != ENDPOINT:
        raise InvalidRun("REVIEW: shared CTRL selection or budget differs")


def setup(args):
    config = recipe()
    frozen, raw = read_frozen(ROOT)
    d.check_frozen_source()
    stage_path = D0_ROOT / "manifests/batches" / D0_BATCH / "stage_a.json"
    if sha256_file(stage_path) != STAGE_A_SHA:
        raise InvalidRun("cannot authenticate the historical environment from the reviewed D Stage A receipt")
    stage = read_json(stage_path)
    d.validate_hold(stage)
    if stage["status"] != "PASS" or not stage["numerical_audit"]["numerical_identity"]:
        raise InvalidRun("historical Stage A numerical certificate is invalid")
    historical, f0_best, f0_logs = d.read_reference(F0_RUN, frozen, raw)
    projection_receipt = read_json(args.projection_dir / PROJECTION_MANIFEST)
    if projection_receipt["sha256"] != PROJECTION_SHA:
        raise InvalidRun("PIL accepts only the registered split-only projection bytes")
    datasets, entries, projection, summary = d.load_inputs(args.projection_dir, frozen, historical, f0_best)
    if canonical(projection) != canonical(stage["projection"]):
        raise InvalidRun("current Train/Dev identity differs from the reviewed D projection")
    targets = build_target_tape(datasets["train"], entries["train"])
    pairing = historical["batch_pairing"]
    pairing_sha = json_identity(pairing)

    def create(arm, batch_id, experiment_date):
        experiment = {**HOLD, "task": TASK, "arm": arm, "protocol": d.PROTOCOL, "batch_id": batch_id,
                      "experiment_date": experiment_date, "config_sha256": json_identity(config),
                      "original_f0_config_sha256": json_identity(raw), "target_tape_sha256": json_identity(targets)
                      if arm.startswith("C07") else None, "pairing_sha256": pairing_sha}
        instance = PILRuntime(raw, datasets, arm, experiment=experiment, targets=targets, pairing=pairing,
                              device="cuda", formal_train_dev=True, data_identity=historical["data_identity"],
                              git_head=d.git_identity()["git_head"])
        d.check_runtime(instance, historical)
        return instance

    control = create(CONTROL, args.batch_id, args.experiment_date)
    provenance, sources = source_environment(stage)
    sampler_tape = control.audit_sampler(pairing)
    candidates = [create(arm, args.batch_id, args.experiment_date) for arm in ARMS[:2]]
    parity = device_parity(control, candidates)
    del candidates
    del control
    blockers = {}
    if not targets["changed_count"]:
        blockers.update({arm: "fixed C07-NULL changes zero Train targets" for arm in ARMS[2:]})
    else:
        for arm in ARMS[2:]:
            create(arm, args.batch_id, args.experiment_date)
    # Probe the existing rendering stack in a separate process before using any training budget.
    subprocess.run([sys.executable, "-B", "-c",
                    "import io, matplotlib; matplotlib.use('Agg'); from matplotlib import pyplot as p; "
                    "f=p.figure(); f.savefig(io.BytesIO(), format='png'); f.savefig(io.BytesIO(), format='svg')"],
                   cwd=ROOT, check=True)
    try:
        d0 = certify_d0(d.read_config(frozen), stage, historical, f0_best, f0_logs, entries)
        baseline = {"arm": "D0", "status": "CERTIFIED_REUSE",
                    "run_id": d0["experiment"]["run_id"], "stage_a_sha256": STAGE_A_SHA, "stage_b_sha256": STAGE_B_SHA,
                    "checkpoints": d0["checkpoints"], "best_epoch": d0["best_epoch"]}
    except (OSError, ValueError, KeyError, TypeError, tarfile.TarError) as error:
        d0 = None
        baseline = {"arm": CONTROL, "status": "ONE_SHARED_OFF_REQUIRED",
                    "d0_reuse_blocker": f"{type(error).__name__}: {error}"}
    identity = {**HOLD, "task": TASK, "config": config, "f0_config": raw, "projection": projection,
                "projection_provenance_sha256": sha256_file(args.projection_dir / PROJECTION_MANIFEST),
                "population": summary, "initial_pairing_sha256": pairing_sha, "runtime": provenance,
                "sampler_tape": sampler_tape,
                "d_source_compatibility": sources, "historical_f0_evidence_sha256": d.REFERENCE_FILES,
                "synthetic_cuda_parity": parity,
                "baseline": baseline, "blocked_arms": blockers, "target_tape_sha256": json_identity(targets)}
    return create, datasets, entries, targets, d0, identity, f0_logs, f0_best


class Log:
    def __init__(self, path):
        self.path = path
        self.console = sys.stdout

    def write(self, value):
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(value)
        return self.console.write(value)

    def flush(self):
        self.console.flush()


def publish_analysis(state, paths, state_path):
    publication = state["publication"]
    staging = paths.analysis_root.parent / publication["staging_name"]
    staged_archive = staging.with_name(staging.name + ".tar.gz")
    archive = paths.analysis_root.parent / f"MERC_PIL_SCREEN_001_{state['batch_id']}_analysis.tar.gz"
    final = paths.analysis_root
    folder = final if final.exists() else staging
    manifest_file = folder / d.MANIFEST_NAME
    if sha256_file(manifest_file) != publication["bundle_manifest_sha256"]:
        raise InvalidRun("sealed PIL analysis manifest changed")
    inventory = {p.relative_to(folder).as_posix(): sha256_file(p) for p in folder.rglob("*")
                 if p.is_file() and p != manifest_file}
    if inventory != read_json(manifest_file)["files"]:
        raise InvalidRun("sealed PIL analysis file inventory changed")
    package = archive if archive.exists() else staged_archive
    if sha256_file(package) != publication["archive_sha256"]:
        raise InvalidRun("sealed PIL archive changed")
    if not archive.exists():
        os.link(staged_archive, archive)
    if not final.exists():
        staging.rename(final)
    if staged_archive.exists():
        if not staged_archive.samefile(archive):
            raise InvalidRun("published and staged archive differ")
        staged_archive.unlink()
    state.update(phase="complete", analysis_export="COMPLETE", analysis_bundle="PASS", plot_ready="YES",
                 archive=str(archive.resolve()), archive_bytes=archive.stat().st_size,
                 archive_sha256=publication["archive_sha256"])
    write_json_atomic(state_path, state)
    print(canonical({key: state[key] for key in ("archive", "archive_bytes", "archive_sha256")}))
    print("ANALYSIS_EXPORT=COMPLETE; ANALYSIS_BUNDLE=PASS; PLOT_READY=YES", flush=True)


def run_batch(args, paths):
    state_path = paths.manifest_root / "state.json"
    if args.action == "run":
        if any(os.path.lexists(path) for path in (paths.manifest_root, paths.analysis_root, paths.launcher_log_root)):
            raise InvalidRun("batch ID already exists")
        paths.manifest_root.mkdir(parents=True, exist_ok=False)
    elif not state_path.is_file():
        raise InvalidRun("resume needs an existing PIL batch receipt")
    with batch_lock(paths.manifest_root / ".lock"):
        old_stdout = sys.stdout
        paths.launcher_log_root.mkdir(parents=True, exist_ok=True)
        state = read_json(state_path) if state_path.is_file() else {
            **HOLD, "task": TASK, "batch_id": args.batch_id, "experiment_date": args.experiment_date,
            "arms": {}, "attempts": [], "phase": "preflight",
        }
        if state["phase"] in ("invalid", "complete"):
            raise InvalidRun("batch is terminal; retain its recorded evidence")
        if args.action == "resume-analysis" and state["phase"] not in ("analysis", "publishing"):
            raise InvalidRun("analysis recovery requires a batch that reached its analysis phase")
        state["attempts"].append({"action": args.action, "phase": state["phase"]})
        write_json_atomic(state_path, state)
        sys.stdout = Log(paths.launcher_log_root / "execution.log")
        try:
            if state["phase"] == "publishing":
                publish_analysis(state, paths, state_path)
                return
            create, datasets, entries, targets, d0, identity, f0_logs, f0_best = setup(args)
            if state.get("identity") is not None and canonical(state["identity"]) != canonical(identity):
                raise InvalidRun("resume provenance or baseline disposition changed")
            state["identity"] = identity
            tape_path = paths.manifest_root / "c07_train_targets.json"
            if not tape_path.exists():
                write_json_atomic(tape_path, targets)
            if targets["changed_count"]:
                validate_target_tape(read_json(tape_path), targets, identity["target_tape_sha256"])
            elif read_json(tape_path) != targets:
                raise InvalidRun("blocked C07 tape changed")
            state["phase"] = "training" if args.action != "resume-analysis" else "analysis"
            write_json_atomic(state_path, state)
            results = {"D0": d0} if d0 is not None else {}
            for arm in (*(() if d0 is not None else (CONTROL,)), *ARMS):
                run_id = f"{arm.lower().replace('-', '_')}_seed100_{args.batch_id}"
                if arm in identity["blocked_arms"]:
                    state["arms"][arm] = {"status": "INVALID", "reason": identity["blocked_arms"][arm]}
                    write_json_atomic(state_path, state)
                    continue
                directory = paths.runs_root / run_id
                record = state["arms"].get(arm)
                runtime = create(arm, args.batch_id, args.experiment_date)
                if directory.exists() and (record is None or record["run_id"] != run_id):
                    raise InvalidRun(f"unowned run directory collision: {directory}")
                if (directory / "result.json").is_file():
                    result = read_completed_arm(runtime, directory)
                elif args.action == "resume-analysis":
                    raise InvalidRun(f"analysis recovery requires a completed {arm}")
                else:
                    state["arms"][arm] = {"status": "RUNNING", "run_id": run_id}
                    write_json_atomic(state_path, state)
                    result = train_arm(runtime, directory, resume=directory.exists())
                del runtime
                if arm == CONTROL:
                    verify_shared_control(result, f0_logs, f0_best)
                digest = sha256_file(directory / "result.json")
                if record and record.get("result_sha256", digest) != digest:
                    raise InvalidRun(f"sealed result changed for {arm}")
                results[arm] = result
                state["arms"][arm] = {"status": "PASS", "run_id": run_id, "result_sha256": digest,
                                      "checkpoint_root": (directory / "checkpoints").as_posix(),
                                      "best_epoch": result["best_epoch"], "checkpoint_sha256": result["checkpoints"]}
                write_json_atomic(state_path, state)
            state["phase"] = "analysis"
            write_json_atomic(state_path, state)
            staging = paths.analysis_root.parent / f".{args.batch_id}.analysis-{uuid4().hex}"
            staging.mkdir(parents=True, exist_ok=False)
            state["attempts"][-1]["analysis_staging"] = staging.as_posix()
            write_json_atomic(state_path, state)
            for filename in ("state.json", "c07_train_targets.json"):
                shutil.copyfile(paths.manifest_root / filename, staging / filename)
            shutil.copyfile(paths.launcher_log_root / "execution.log", staging / "execution.log")
            shutil.copyfile(ROOT / CONFIG, staging / "seed100.yaml")
            history_receipt = D0_ROOT / "manifests/batches" / D0_BATCH / "stage_a.json"
            shutil.copyfile(history_receipt, staging / "reviewed_d_stage_a.json")
            if sha256_file(staging / "reviewed_d_stage_a.json") != STAGE_A_SHA:
                raise InvalidRun("reviewed Stage A receipt changed during collection")
            for arm, result in results.items():
                if arm == "D0":
                    source_run = D0_ROOT / "runs" / result["experiment"]["run_id"]
                    evidence_dir = staging / "runtime_evidence" / "D0"
                    d.copy_artifacts(source_run, evidence_dir)
                    if any(sha256_file(evidence_dir / name) != digest
                           for name, digest in read_json(history_receipt)["run_evidence_sha256"].items()):
                        raise InvalidRun("D0 evidence changed during collection")
                else:
                    record = state["arms"][arm]
                    source = paths.runs_root / record["run_id"] / "result.json"
                    destination = staging / "runtime_evidence" / arm / "result.json"
                    destination.parent.mkdir(parents=True)
                    shutil.copyfile(source, destination)
                    if sha256_file(destination) != record["result_sha256"]:
                        raise InvalidRun(f"{arm} evidence changed during collection")
            (staging / "implementation.diff").write_bytes(
                git_bytes("diff", "--binary", "3f0436612e06816bb07402afb250c5f50a825721", "HEAD", "--", *sorted(ISOLATED_SOURCES)))
            script = ROOT / "scripts/analysis/models/multidag_cl/pil_screen_analysis.py"
            shutil.copyfile(script, staging / script.name)
            state["candidate_verdicts"] = export_analysis(staging, results, entries["dev"], datasets["dev"], state)
            write_json_atomic(staging / "state.json", state)
            plot_tables(staging)
            staging_archive = staging.with_name(staging.name + ".tar.gz")
            digest = d.ship(staging, staging_archive, "PASS")
            state.update(phase="publishing", publication={"staging_name": staging.name, "archive_sha256": digest,
                "bundle_manifest_sha256": sha256_file(staging / d.MANIFEST_NAME)})
            write_json_atomic(state_path, state)
            publish_analysis(state, paths, state_path)
        except BaseException as error:
            if state["phase"] in ("preflight", "training") and not isinstance(error, (OSError, KeyboardInterrupt)):
                state["phase"] = "invalid"
            if isinstance(error, InvalidRun):
                state["phase"] = "invalid"
            state["attempts"][-1]["failure"] = {"phase": state["phase"], "type": type(error).__name__,
                                                "error": str(error), "traceback": traceback.format_exc()}
            print(f"PIL_SCREEN_001 BLOCKED: {type(error).__name__}: {error}", flush=True)
            write_json_atomic(state_path, state)
            raise
        finally:
            sys.stdout = old_stdout


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("plan", "preflight", "run", "resume-training", "resume-analysis"))
    parser.add_argument("--projection-dir", type=Path)
    parser.add_argument("--batch-id")
    parser.add_argument("--experiment-date")
    args = parser.parse_args(argv)
    if args.action == "plan":
        print(canonical({"config": recipe(), "f0_config": read_frozen(ROOT)[1], "steps_per_arm": 210}))
        return
    if not all((args.projection_dir, args.batch_id, args.experiment_date)):
        parser.error("formal actions require --projection-dir, --batch-id and --experiment-date")
    if os.name != "posix" or not torch.cuda.is_available():
        parser.error("formal PIL requires the authorized Ubuntu CUDA environment")
    subprocess.run([sys.executable, str(ROOT / "scripts/dev/check_workstream.py"), "multidag"], cwd=ROOT, check=True)
    paths = resolve_output_paths(output_base="outputs", experiment_group=GROUP, batch_id=args.batch_id,
                                 experiment_date=args.experiment_date)
    if args.action == "preflight":
        _, _, _, _, _, identity, _, _ = setup(args)
        print(canonical(identity), flush=True)
        print("PIL_PREFLIGHT=PASS" if not identity["blocked_arms"] else "PIL_PREFLIGHT=PARTIAL", flush=True)
    else:
        run_batch(args, paths)


if __name__ == "__main__":
    main()

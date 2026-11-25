"""Validate, train or evaluate one frozen CT-MultiDAG Protocol B fold."""

import argparse
import csv
import json
import os
from pathlib import Path
from time import perf_counter

import torch
import yaml

from scripts.runtime.multidag_cl_fusion.runtime import paired_model, training_rng_identity
from scripts.runtime.multidag_cl_group_isolated.evaluation import checkpoint_path, replay
from scripts.runtime.multidag_cl_group_isolated.protocol import (
    CONFIG_DIR, ENDPOINT, FIRST_FOLD, G1_CONFIG_DIR, G1_SCHEMA, HOLD, PARAMETERS, SESSIONS, describe, load_config,
    load_fold, read_frozen, read_json, validate_hold,
)
from scripts.runtime.multidag_cl_group_isolated.provenance import runtime_provenance, source_hashes
from scripts.runtime.multidag_cl_paper_reimplementation.checkpoint import save_checkpoint_atomic
from scripts.runtime.multidag_cl_ta_v0.formal import git_identity, json_identity, sha256_file, write_json_atomic
from scripts.runtime.multidag_cl_ta_v0.training import cpu_snapshot
from scripts.runtime.multidag_cl_tav_full_audit_v1.runtime import AuditRuntime
from utils.output_paths import resolve_experiment_date, resolve_output_paths


ROOT = Path(__file__).resolve().parents[3]
TRAIN_LOG_FIELDS = ("epoch", "train_update_loss", "max_gradient_norm", "optimizer_steps", "batch_tape_sha256")


def run_identity(config, raw, data):
    return {
        "schema": config["schema"], "run_id": config["run_id"], "config_sha256": json_identity(config),
        "f0_recipe_sha256": json_identity(raw), "data_identity_sha256": json_identity(data),
    }


def build_runtime(raw, datasets, device, *, synthetic=False, train_label_smoothing=0.0):
    if synthetic and device != "cpu":
        raise ValueError("synthetic runtime is CPU-only")
    model, _, initialization = paired_model(raw, device=device)
    if sum(p.numel() for p in model.parameters()) != PARAMETERS:
        raise ValueError("formal F0 parameter count changed")
    runtime = AuditRuntime(model, datasets.__getitem__, device=device,
                           formal_train_dev=not synthetic, synthetic_only=synthetic,
                           train_label_smoothing=train_label_smoothing)
    return runtime, initialization


def steps_per_epoch(fit, raw):
    batch_size = raw["training"]["batch_size"]
    return (fit["dialogues"] + batch_size - 1) // batch_size


def train(run_dir, config, raw, data, fit, fit_entries, *, device, git, synthetic=False, source_configs=()):
    # This function receives only fit feature tensors. AuditRuntime owns the
    # optimizer, masked loss, gradient clipping and seeded CL_OFF batch loader.
    is_g1 = config["schema"] == G1_SCHEMA
    smoothing = config.get("label_smoothing", 0.0)
    if is_g1 and not (source_configs or synthetic):
        raise ValueError("G1 requires a runtime source inventory")
    run_dir.mkdir(parents=True, exist_ok=False)
    manifest_path = run_dir / "manifests" / "run_manifest.json"
    manifest_path.parent.mkdir()
    (run_dir / "logs").mkdir()
    summary = describe(fit_entries)
    steps = steps_per_epoch(summary, raw)
    identity = run_identity(config, raw, data)
    manifest = {
        **HOLD, "TEST_TOUCHED": "NO", "run_id": config["run_id"], "run_identity": identity,
        "status": "training", "seed": config["seed"], "endpoint_epoch": ENDPOINT,
        "checkpoint_selection": "fixed_epoch", "last_completed_epoch": 0, "steps_per_epoch": steps,
        "optimizer_steps": 0, "checkpoint_sha256": {}, "synthetic_only": synthetic, "git": git,
        "data_identity": data, "f0_recipe": raw, "parameter_count": PARAMETERS,
        "device": device, "torch_version": str(torch.__version__), "cuda_version": torch.version.cuda,
    }
    write_json_atomic(manifest_path, manifest)
    try:
        runtime, initialization = build_runtime(raw, {"train": fit}, device, synthetic=synthetic,
                                                train_label_smoothing=smoothing)
        manifest["initialization"] = initialization
        if is_g1:
            manifest.update(fit_objective={"name": "uniform_label_smoothed_ce", "label_smoothing": smoothing},
                            evaluation_objective="hard_label_nll",
                            training_initial_rng=training_rng_identity(device))
        if source_configs:
            manifest["runtime_provenance"] = runtime_provenance(ROOT, device, source_configs)
        runtime.curriculum.export_manifest(run_dir / "manifests" / "fit_curriculum_bucket_manifest.tsv")
        if is_g1:
            write_json_atomic(manifest_path, manifest)
        (run_dir / "resolved_config.yaml").write_text(
            yaml.safe_dump({"protocol_b": config, "f0_recipe": raw, "data_identity": data}, sort_keys=False),
            encoding="utf-8",
        )
        expected = {}
        for uid, did, _ in fit_entries:
            expected.setdefault(did, []).append(uid)
        expected = {did: tuple(ids) for did, ids in expected.items()}
        with (run_dir / "logs" / "fit_updates.csv").open("x", encoding="utf-8", newline="") as stream:
            fields = (*TRAIN_LOG_FIELDS, "fit_epoch_wall_seconds") if is_g1 else TRAIN_LOG_FIELDS
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            fit_start = perf_counter()
            for epoch in range(1, ENDPOINT + 1):
                epoch_start = perf_counter()
                total, targets, norm = 0.0, 0, 0.0
                seen, tape = set(), []
                for batch in runtime.loader("train", epoch=epoch):
                    order = list(zip(batch["dialogue_ids"], batch["utterance_ids"]))
                    for did, ids in order:
                        if did in seen or expected.get(did) != ids:
                            raise ValueError("G0 fit batch contains a duplicate, incomplete or out-of-fold dialogue")
                        seen.add(did)
                    tape.append(json_identity(order))
                    update = runtime.train_step(batch)
                    count = update["valid_target_count"]
                    total += update["loss"] * count
                    targets += count
                    norm = max(norm, update["gradient_norm"])
                if seen != expected.keys() or targets != summary["utterances"] or runtime.optimizer_steps != epoch * steps:
                    raise ValueError("CL_OFF fit coverage or optimizer step count changed")
                path = checkpoint_path(run_dir, epoch)
                save_checkpoint_atomic(path, {
                    "identity": identity, "epoch": epoch, "optimizer_steps": runtime.optimizer_steps,
                    "model_state_dict": cpu_snapshot(runtime.model.state_dict()),
                })
                log = {"epoch": epoch, "train_update_loss": total / targets, "max_gradient_norm": norm,
                       "optimizer_steps": runtime.optimizer_steps, "batch_tape_sha256": json_identity(tape)}
                if is_g1:
                    log["fit_epoch_wall_seconds"] = perf_counter() - epoch_start
                writer.writerow(log)
                stream.flush()
                manifest["checkpoint_sha256"][str(epoch)] = sha256_file(path)
                manifest.update(last_completed_epoch=epoch, optimizer_steps=runtime.optimizer_steps)
                write_json_atomic(manifest_path, manifest)
                if not synthetic:
                    print(json.dumps(log), flush=True)
            if is_g1:
                manifest["fit_wall_seconds"] = perf_counter() - fit_start
        if source_configs:
            hashes = manifest["runtime_provenance"]["files_sha256"]
            after = source_hashes(ROOT, source_configs)
            if any(after.get(path) != digest for path, digest in hashes.items()):
                raise ValueError("runtime source changed during fit")
            manifest["runtime_provenance"]["late_imports_sha256"] = {
                path: digest for path, digest in after.items() if path not in hashes
            }
        manifest["status"] = "completed"
        write_json_atomic(manifest_path, manifest)
    except BaseException as error:
        manifest.update(status="failed", error_type=type(error).__name__)
        write_json_atomic(manifest_path, manifest)
        raise
    return manifest


def evaluate(run_dir, config, raw, data, views, entries, *, device, synthetic=False):
    manifest = read_json(run_dir / "manifests" / "run_manifest.json")
    validate_hold(manifest)
    epochs = set(map(str, range(1, ENDPOINT + 1)))
    steps = steps_per_epoch(describe(entries["fit"]), raw)
    expected = {
        "status": "completed", "last_completed_epoch": ENDPOINT, "endpoint_epoch": ENDPOINT,
        "checkpoint_selection": "fixed_epoch", "synthetic_only": synthetic, "seed": config["seed"],
        "run_identity": run_identity(config, raw, data), "run_id": config["run_id"],
        "steps_per_epoch": steps, "optimizer_steps": ENDPOINT * steps, "f0_recipe": raw,
        "data_identity": data, "parameter_count": PARAMETERS,
    }
    if (any(manifest.get(key) != value for key, value in expected.items())
            or set(manifest["checkpoint_sha256"]) != epochs):
        raise ValueError("Protocol B scoring requires the completed exact fold and its fixed epoch-30 checkpoint")
    if config["schema"] == G1_SCHEMA and not synthetic:
        inventory = manifest["runtime_provenance"]
        hashes = {**inventory["files_sha256"], **inventory["late_imports_sha256"]}
        if not hashes or any(sha256_file(ROOT / path) != digest for path, digest in hashes.items()):
            raise ValueError("G1 evaluation source differs from the fit source inventory")
    runtime, _ = build_runtime(raw, {"train": views["fit"], "dev": views["held_out"]}, device, synthetic=synthetic)
    return replay(run_dir, runtime, views, entries, manifest)


def output_path(config, experiment_date):
    output = config["output"]
    return resolve_output_paths(output_base=output["root"], experiment_date=resolve_experiment_date(experiment_date),
                                experiment_group=output["experiment_group"], run_id=config["run_id"]).run_root


def parser():
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)
    for command in ("validate", "train", "evaluate"):
        sub = commands.add_parser(command)
        sub.add_argument("--config", type=Path, required=True)
        sub.add_argument("--projection-dir", type=Path, required=command != "validate")
        if command != "validate":
            sub.add_argument("--experiment-date", required=True)
        if command == "train":
            sub.add_argument("--execute", action="store_true", help="start the single reviewed Ubuntu CUDA run")
            sub.add_argument("--release-fold", choices=SESSIONS, help="explicit later release for a deferred fold")
        if command == "evaluate":
            sub.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    return root


def main(argv=None):
    cli = parser()
    args = cli.parse_args(argv)
    config = load_config(args.config)
    frozen, raw = read_frozen(ROOT)
    session = config["held_out_session"]
    is_g1 = config["schema"] == G1_SCHEMA
    source_configs = ()
    if args.command == "train":
        if not args.execute or ((is_g1 or session != FIRST_FOLD) and args.release_fold != session):
            cli.error("single-fold training requires --execute and a matching --release-fold for deferred folds")
        if os.name == "nt" or not torch.cuda.is_available():
            cli.error("real Protocol B training requires the Ubuntu CUDA environment")
        if is_g1:
            canonical_config = ROOT / G1_CONFIG_DIR / f"{session.lower()}_seed100.yaml"
            if args.config.resolve() != canonical_config:
                cli.error("G1 training requires the canonical fold YAML")
            source_configs = (canonical_config, ROOT / CONFIG_DIR / "fold_identities.json",
                              ROOT / raw["frontend_source"]["config_path"])
    if args.command == "validate":
        if args.projection_dir:
            load_fold(config, frozen, args.projection_dir)
        print(json.dumps({
            "GROUP_ISOLATED_PROTOCOL": "PASS", "SPEAKER_LEAKAGE": 0, "DIALOGUE_LEAKAGE": 0,
            "TEST_TOUCHED": "NO", "held_out_session": session, "fit": frozen["folds"][session]["fit"],
            "held_out": frozen["folds"][session]["held_out"],
            "projection": "VERIFIED" if args.projection_dir else "METADATA_ONLY",
            "checkpoint_epoch": ENDPOINT,
            **({"label_smoothing": config["label_smoothing"]} if is_g1 else {}),
        }, indent=2))
        return

    run_dir = output_path(config, args.experiment_date)
    if args.command == "train" and (run_dir.exists() or run_dir.is_symlink()):
        raise FileExistsError(run_dir)
    views, entries, data = load_fold(config, frozen, args.projection_dir, fit_only=args.command == "train")
    if args.command == "train":
        train(run_dir, config, raw, data, views["fit"], entries["fit"], device="cuda",
              git=git_identity(), source_configs=source_configs)
        print(run_dir)
    else:
        print(evaluate(run_dir, config, raw, data, views, entries, device=args.device))


if __name__ == "__main__":
    main()

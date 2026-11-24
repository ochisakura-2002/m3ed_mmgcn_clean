"""Run the preregistered seed100 H1 Fast-Kill on certified Train/Dev only."""

import argparse
from functools import partial
from pathlib import Path
import pickle
import sys
from zipfile import BadZipFile

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

import numpy as np
import torch
import yaml

from models.multidag_cl.fusion.base import DirectConcatFusion
from models.multidag_cl.history_attenuation import FrozenHistoryModel, PARAMETER_COUNT
from scripts.analysis.models.multidag_cl.failure_audit_inference import frozen_model, membership
from scripts.analysis.models.multidag_cl.history_fastkill_metrics import EdgeDistributions, analyze, decision
from scripts.analysis.models.multidag_cl.history_fastkill_provenance import RECOVERY_SHA256, read_recovery
from scripts.analysis.models.multidag_cl.history_fastkill_runtime import (
    ADAM, EPOCHS, check_frozen, collect_dev, freeze_snapshot, lambda_values, optimizer_step_gate, train_fixed, verify_causal,
)
from scripts.analysis.models.multidag_cl.pairwise_preflight_inputs import (
    HOLD, PROJECTION_MANIFEST, SEED, TestHoldViolation, load_projected_train_dev, read_f0, validate_f0_dev,
    validate_hold, validate_population,
)
from scripts.analysis.models.multidag_cl.run_pairwise_preflight import bundle
from scripts.dev.check_workstream import check_workstream
from scripts.runtime.multidag_cl_ta_v0.formal import git_identity, sha256_file, write_json_atomic
from scripts.runtime.multidag_cl_ta_v0.training import seed_everything
from scripts.runtime.multidag_cl_tav_full_audit_v1.spec import write_csv
from utils.output_paths import resolve_experiment_date, resolve_output_paths


GROUP = "history_fastkill_seed100"
ROOT = Path(__file__).resolve().parents[4]
CONFIG = ROOT / "configs/multidag_cl/tav_full_audit_v1/history_fastkill_seed100.yaml"
SUMMARY = "HISTORY_FASTKILL_SUMMARY.csv"
REPORT = "HISTORY_FASTKILL_REPORT.md"
MANIFEST = "run_manifest.json"
CHECKPOINT = "history_attenuation_epoch20.pt"


def read_enabled(config):
    raw = yaml.safe_load(config.read_text(encoding="utf-8"))
    if (not isinstance(raw, dict) or set(raw) != {"history_attenuation_enabled"}
            or type(raw["history_attenuation_enabled"]) is not bool):
        raise ValueError("expected only the boolean history_attenuation_enabled YAML switch")
    return raw["history_attenuation_enabled"]


def write_report(output, manifest, summary, tables):
    f0, primary = manifest["f0"]["epoch"], manifest["primary_epoch"]
    lines = ["# CT-MultiDAG seed100 H1 Fast-Kill", "",
             f'**{manifest["FASTKILL"]}** · CTRL epoch {f0} · H1 epoch {primary}', "",
             "| Dev 指标 | CTRL (%) | H1 (%) | Δ (pp) |", "| --- | ---: | ---: | ---: |"]
    lines.extend(f'| {row["metric"]} | {row["ctrl_pct"]:.6f} | {row["h1_pct"]:.6f} | {row["delta_pp"]:+.6f} |'
                 for row in tables["DEV_METRICS.csv"])
    lines.extend(["", "| Dev 子组 | support | F0 errors | H1 errors | R | D | NET |",
                  "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"])
    for row in ({"group": "FULL_DEV", **summary}, *tables["H_TOP25_RECOVERY.csv"],
                *tables["SHIFT_STABLE_RECOVERY.csv"], *tables["UH_SUBGROUP_RECOVERY.csv"]):
        lines.append(f'| {row["group"]} | {row["support"]} | {row["ctrl_errors"]} | {row["primary_errors"]} | '
                     f'{row["r"]} | {row["d"]} | {row["net"]:+d} |')
    lines.extend(["", f'最终 lambda_same 为 {summary["lambda_same"]:.8f}，lambda_diff 为 {summary["lambda_diff"]:.8f}。',
                  "", "衰减统计描述 H1 实际施加的边系数。Calibration 和配对对话 bootstrap 用于解释。",
                  "唯一 PRIMARY 评分使用 epoch 20。Test HOLD。", ""])
    (output / REPORT).write_text("\n".join(lines), encoding="utf-8")


def execute(args, output, archive, *, device, git, experiment_date):
    if archive.exists():
        raise FileExistsError("existing H1 result archive cannot be overwritten")
    output.mkdir(parents=True, exist_ok=False)
    manifest = {**git, **HOLD, "schema": "ct_multidag_history_fastkill_v1", "experiment_group": GROUP,
                "experiment_date": experiment_date, "batch_id": args.batch_id, "device": device, "seed": SEED,
                "TEST_TOUCHED": "NO", "H_SUBGROUP_PROVENANCE": "FAIL", "F0_IDENTITY": "UNVERIFIED",
                "F0_BYPASS_IDENTITY": "UNVERIFIED", "LAMBDA_ZERO_IDENTITY": "UNVERIFIED",
                "formal_train_runs": 0, "optimizer_steps": 0, "primary_dev_evaluation_count": 0,
                "dev_used_for_model_selection": False, "trainable_parameter_count": PARAMETER_COUNT}
    stage = "RECOVERY_ARTIFACT"
    try:
        recovery = read_recovery(args.recovery_dir, args.recovery_archive)
        stage = "PARAMETERIZATION"
        manifest["synthetic_optimizer_gate"] = optimizer_step_gate(device)
        stage = "F0_AND_TRAIN_DEV"
        source = read_f0(args.frozen_runs)
        certificate = args.train_dev_package / PROJECTION_MANIFEST
        certificate_sha = sha256_file(certificate)
        datasets, projection = load_projected_train_dev(args.train_dev_package, source)
        if sha256_file(certificate) != certificate_sha:
            raise ValueError("Train/Dev projection certificate changed while authenticating")
        populations = validate_population(datasets, source)
        recovery.verify_source(source, projection, populations["dev"])
        manifest.update(f0=source.identity, train_dev_projection=projection,
                        projection_manifest_sha256=certificate_sha, population=populations,
                        f0_config_sha256=source.manifest["config_sha256"],
                        dev_batch_tape_sha256=source.manifest["batch_pairing"]["dev"]["batch_sha256"],
                        recovery={"batch_id": recovery.manifest["batch_id"], "archive_sha256": RECOVERY_SHA256,
                                  "manifest_sha256": recovery.manifest_sha256,
                                  "outputs_sha256": recovery.manifest["outputs_sha256"]})
        curriculum = membership(datasets["train"])
        if curriculum != source.manifest["curriculum_membership_sha256"]:
            raise ValueError("Train membership differs from the frozen F0")
        seed_everything(device, SEED)
        f0 = frozen_model(source, curriculum, device, check_test_hold=partial(validate_hold, legacy=True))
        if not isinstance(f0.fusion, DirectConcatFusion):
            raise ValueError("expected original DirectConcat F0")
        model = FrozenHistoryModel(f0).to(device).eval()
        snapshot = freeze_snapshot(model)
        check_frozen(model)
        batch_size = source.config["training"]["batch_size"]
        manifest["training"] = {"batch_size": batch_size, "epochs": EPOCHS, "optimizer": "Adam", **ADAM,
                                "loss": "masked_cross_entropy", "train_shuffle_seed": SEED, "dev_shuffle": False,
                                "history_attenuation_enabled": True}
        stage = "PRETRAIN_CAUSAL_INVARIANCE"
        manifest["causal_invariance"] = verify_causal(model, device)
        stage = "F0_REPLAY_AND_IDENTITY"
        values = lambda_values(model.attenuation)
        if any(values.values()):
            raise ValueError("H1 lambdas do not start at zero")
        ctrl = collect_dev(model, datasets["dev"], source=source, device=device, batch_size=batch_size, identity=True)
        validate_f0_dev(source, ctrl)
        if tuple(ctrl.logits.argmax(axis=1)) != recovery.predictions:
            raise ValueError("live F0 predictions differ from the recovered subgroup evidence")
        check_frozen(model, snapshot)
        manifest.update(F0_IDENTITY="PASS", F0_BYPASS_IDENTITY="PASS", LAMBDA_ZERO_IDENTITY="PASS",
                        H_SUBGROUP_PROVENANCE="PASS",
                        initial_lambdas=values, CAUSAL_INVARIANCE="PASS",
                        f0_dev_logits_max_abs_diff=float(np.max(np.abs(ctrl.logits - source.logits))))
        print("F0_IDENTITY = PASS\nlambda_same = 0\nlambda_diff = 0", flush=True)

        stage = "TRAIN_EPOCH20"
        validate_hold(manifest)
        manifest["formal_train_runs"] = 1
        history = train_fixed(model, datasets["train"], device=device, batch_size=batch_size)
        if [row["epoch"] for row in history] != list(range(1, EPOCHS + 1)):
            raise ValueError("H1 did not reach the unique epoch-20 endpoint")
        manifest.update(primary_epoch=EPOCHS, optimizer_steps=sum(row["optimizer_steps"] for row in history))
        check_frozen(model, snapshot)
        if sha256_file(source.checkpoint) != source.identity["checkpoint_sha256"]:
            raise ValueError("frozen F0 checkpoint changed during H1")
        manifest.update(F0_PARAMETERS_UNCHANGED="PASS", F0_GRADIENTS_ABSENT="PASS")

        stage = "EPOCH20_DEV"
        edges = EdgeDistributions()
        primary = collect_dev(model, datasets["dev"], source=source, device=device, batch_size=batch_size, observe=edges)
        check_frozen(model, snapshot)
        manifest["primary_dev_evaluation_count"] = 1
        validate_hold(manifest)
        summary, tables = analyze(ctrl, primary, recovery.groups)
        outcome = decision(summary)
        summary.update(lambda_values(model.attenuation))
        tables.update(edges.tables(primary.population))
        tables["TRAIN_HISTORY.csv"] = history
        tables["LAMBDA_HISTORY.csv"] = [{"epoch": 0, "raw_lambda_same": 0.0, "raw_lambda_diff": 0.0, **values},
                                        *[{key: row[key] for key in ("epoch", "raw_lambda_same", "raw_lambda_diff",
                                                                     "lambda_same", "lambda_diff")} for row in history]]

        stage = "ARTIFACTS"
        torch.save({"epoch": EPOCHS, "seed": SEED, "f0_checkpoint_sha256": source.identity["checkpoint_sha256"],
                    "state_dict": {name: value.detach().cpu() for name, value in model.attenuation.state_dict().items()}},
                   output / CHECKPOINT)
        manifest.update(outcome, status="completed",
                        h1_checkpoint={"path": CHECKPOINT, "sha256": sha256_file(output / CHECKPOINT)})
        write_csv(output / SUMMARY, [{"seed": SEED, "epoch": EPOCHS, "fastkill": outcome["FASTKILL"], **summary}])
        for name, rows in tables.items():
            write_csv(output / name, rows)
        write_report(output, manifest, summary, tables)
        names = (SUMMARY, *tables, REPORT)
        manifest["outputs_sha256"] = {name: sha256_file(output / name) for name in names}
        write_json_atomic(output / MANIFEST, manifest)
        digest = bundle(output, archive, (*names, MANIFEST))
    except (OSError, TypeError, ValueError, KeyError, IndexError, RuntimeError, EOFError, BadZipFile,
            pickle.UnpicklingError) as error:
        if isinstance(error, TestHoldViolation):
            manifest.update(TEST_TOUCHED="YES", rejected_input_hold_evidence=error.evidence)
        if stage in ("F0_AND_TRAIN_DEV", "F0_REPLAY_AND_IDENTITY"):
            manifest["F0_IDENTITY"] = "FAIL"
        manifest.update(FASTKILL="REVIEW_INVALID", status="invalid", failed_stage=stage,
                        reason=str(error), five_seed_eligible=False)
        write_json_atomic(output / MANIFEST, manifest)
        (output / REPORT).write_text(f"# CT-MultiDAG seed100 H1 Fast-Kill\n\nREVIEW_INVALID · {stage}\n\n{error}\n",
                                     encoding="utf-8")
        print(f"FASTKILL = REVIEW_INVALID\nSTAGE = {stage}\nREASON = {error}", flush=True)
        return 2

    print(f'FASTKILL = {outcome["FASTKILL"]}\nH_SUBGROUP_PROVENANCE = PASS\nPRIMARY_EPOCH = {EPOCHS}'
          f'\nΔW-F1 (pp) = {summary["delta_weighted_f1_pp"]:.8f}\nFULL_NET = {summary["net"]}'
          f'\nH_TOP25_NET = {summary["h_top25_net"]}\nSHIFT_NET = {summary["shift_net"]}\nTEST_TOUCHED = NO'
          f'\nHISTORY_FASTKILL_ARCHIVE = {archive}\nHISTORY_FASTKILL_ARCHIVE_SHA256 = {digest}', flush=True)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--frozen-runs", type=Path, required=True)
    parser.add_argument("--train-dev-package", type=Path, required=True)
    parser.add_argument("--recovery-dir", type=Path, required=True, help="HISTORY_TOP25_RECOVERY_DIR")
    parser.add_argument("--recovery-archive", type=Path, help="frozen ZIP, defaults to sibling of --recovery-dir")
    parser.add_argument("--batch-id", required=True)
    parser.add_argument("--experiment-date")
    parser.add_argument("--output-root", default="outputs")
    args = parser.parse_args(argv)
    errors = check_workstream("multidag")
    if errors:
        parser.error("\n".join(errors))
    if not sys.platform.startswith("linux") or not torch.cuda.is_available():
        parser.error("formal H1 Fast-Kill requires Ubuntu CUDA")
    try:
        if not read_enabled(args.config):
            parser.error("the formal H1 arm requires history_attenuation_enabled: true")
    except (OSError, ValueError, yaml.YAMLError) as error:
        parser.error(str(error))
    if args.recovery_archive is None:
        args.recovery_archive = args.recovery_dir.with_name(args.recovery_dir.name + ".zip")
    date = resolve_experiment_date(args.experiment_date)
    paths = resolve_output_paths(output_base=args.output_root, experiment_date=date,
                                 experiment_group=GROUP, batch_id=args.batch_id)
    output = paths.analysis_root
    return execute(args, output, output.with_name(output.name + ".zip"), device="cuda", git=git_identity(),
                   experiment_date=date)


if __name__ == "__main__":
    raise SystemExit(main())

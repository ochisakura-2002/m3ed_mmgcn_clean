"""Run the frozen seed100 F0 top-2 oracle on Train/Dev only."""

import argparse
import os
from pathlib import Path
import pickle
import sys
from tempfile import NamedTemporaryFile
from zipfile import ZIP_DEFLATED, ZipFile

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

import numpy as np
import torch
import yaml

from scripts.analysis.models.multidag_cl.pairwise_preflight_inputs import (
    CLASS_ORDER, HOLD, Q4_SUPPORT, SEED, TRAIN_DEV_COUNTS, TestHoldViolation, audit_subgroups, forward_f0,
    load_projected_train_dev, read_f0, validate_hold, validate_population,
)
from scripts.analysis.models.multidag_cl.pairwise_preflight_metrics import (
    MARGIN_LIMIT, METRICS, MIN_RECOVERABLE, MIN_WF1_GAIN_PP, analyze, decision, scores,
)
from scripts.dev.check_workstream import check_workstream
from scripts.runtime.multidag_cl_ta_v0.formal import git_identity, json_identity, sha256_file, write_json_atomic
from scripts.runtime.multidag_cl_tav_full_audit_v1.spec import write_csv
from utils.output_paths import resolve_experiment_date, resolve_output_paths


GROUP = "pairwise_preflight_seed100"
SUMMARY = "PAIRWISE_PREFLIGHT_SUMMARY.csv"
DECISION = "PAIRWISE_PREFLIGHT_DECISION.json"
REPORT = "PAIRWISE_PREFLIGHT_REPORT.md"
IDENTITY = "resolved_identity.json"
HOLD_STATUS = "test_hold_status.json"
MANIFEST = "run_manifest.json"
THRESHOLDS = {"minimum_recoverable_f": MIN_RECOVERABLE, "minimum_oracle_delta_wf1_pp": MIN_WF1_GAIN_PP}


def print_status(status):
    for name, value in status.items():
        print(f"{name} = {value if value is not None else 'NOT_AVAILABLE'}", flush=True)


def bundle(output, archive, names):
    if len(names) != len(set(names)) or any(Path(name).name != name or Path(name).suffix not in (".csv", ".json", ".md")
                                           for name in names):
        raise ValueError("preflight archive can contain only the enumerated lightweight reports")
    with NamedTemporaryFile(dir=archive.parent, suffix=".zip", delete=False) as stream:
        temporary = Path(stream.name)
    try:
        with ZipFile(temporary, "w", compression=ZIP_DEFLATED) as result:
            for name in names:
                result.write(output / name, arcname=name)
        with ZipFile(temporary) as result:
            if (result.namelist() != list(names) or result.testzip() is not None
                    or any(result.read(name) != (output / name).read_bytes() for name in names)):
                raise ValueError("preflight report archive failed integrity checks")
        digest = sha256_file(temporary)
        os.link(temporary, archive)
        return digest
    finally:
        temporary.unlink()


def report(output, summary, outcome):
    rows = ["# CT-MultiDAG seed100 零训练 Preflight", "",
            "该 Oracle 将固定 gate 内真类恰为第二名的 F0 错误改为第二名。"
            "指标是前二约束下的理想上界，未测量复判头的可学习性。",
            "", f'预注册判定 **{outcome["ZERO_TRAINING_PREFLIGHT"]}**。'
            f'符合条件的错误有 {summary["recoverable_f"]} 条，W-F1 上界差为 {summary["oracle_delta_weighted_f1_pp"]:+.6f} pp。',
            "", "| Dev 指标 | F0 (%) | Oracle (%) | Δ (pp) |", "| --- | ---: | ---: | ---: |"]
    for key in METRICS:
        rows.append(f'| {key} | {100 * summary[f"f0_{key}"]:.6f} | {100 * summary[f"oracle_{key}"]:.6f} | '
                    f'{summary[f"oracle_delta_{key}_pp"]:+.6f} |')
    rows += ["", f'Train 阈值 {summary["train_mad_threshold"]:.12g}，gate 覆盖 '
             f'{summary["n_g_train"]}/{summary["n_train_total"]}。Dev gate 覆盖 '
             f'{summary["n_g_dev"]}/{summary["n_dev"]}。冻结 Q4 内可恢复 {summary["q4_f_count"]} 条错误。',
             "", f'Q4 来源 `{outcome["Q4_SOURCE"]}`。SHIFT 分层 `{outcome["SHIFT_SUBGROUP_STATUS"]}`，'
             f'U/H 分层 `{outcome["H_SUBGROUP_STATUS"]}`。',
             "", "比例列以 0–1 表示，差值以百分点表示。分母为零的比率留空。Test 保持 HOLD。", ""]
    (output / REPORT).write_text("\n".join(rows), encoding="utf-8")


def execute(args, output, archive, *, device, git, experiment_date):
    if archive.exists():
        raise FileExistsError("existing preflight archive cannot be overwritten")
    output.mkdir(parents=True, exist_ok=False)
    manifest = {**git, "schema": "ct_multidag_pairwise_preflight_v1", "experiment_date": experiment_date,
                "experiment_group": GROUP, "batch_id": args.batch_id, "mode": "frozen_forward",
                "device": device, "seed": SEED, "formal_train_runs": 0, "optimizer_steps_performed": 0}
    resolved = {"status": "REVIEW_INVALID_IDENTITY"}
    try:
        source = read_f0(args.frozen_runs)
        datasets, projection = load_projected_train_dev(args.train_dev_package, source)
        populations = validate_population(datasets, source)
        q4, subgroups, trace_status = audit_subgroups(source, args.round1_dir)
        if q4.shape != (TRAIN_DEV_COUNTS["dev"]["utterances"],) or int(q4.sum()) != Q4_SUPPORT:
            raise ValueError("frozen seed100 Dev MAD Q4 has an unexpected population")
        train, dev = forward_f0(source, datasets, device)
        train_scores, dev_scores = scores(train.logits, train.labels), scores(dev.logits, dev.labels)
        if len(train_scores.labels) != TRAIN_DEV_COUNTS["train"]["utterances"]:
            raise ValueError("incomplete frozen Train forward")
        validate_hold(HOLD)
        resolved = {"status": "PASS", "run_id": source.config["experiment_id"], "checkpoint": source.identity,
                    "config_sha256": source.manifest["config_sha256"],
                    "architecture_identity": source.manifest["architecture_identity"],
                    "context_mode": source.config["context_mode"], "fusion": source.config["fusion"],
                    "graph_core": source.config["graph_core"], "class_order": CLASS_ORDER,
                    "data_identity": source.manifest["data_identity"], "train_dev_projection": projection,
                    "populations": populations, "dev_gold_population_sha256": json_identity(dev.population),
                    "archived_dev_logits_sha256": json_identity(source.logits.tolist()),
                    "dev_logits_max_abs_diff": float(np.max(np.abs(dev.logits - source.logits))),
                    "resolved_audit_config": source.config}
    except (OSError, ValueError, KeyError, TypeError, IndexError, RuntimeError, pickle.UnpicklingError,
            EOFError, yaml.YAMLError) as error:
        hold_violation = isinstance(error, TestHoldViolation)
        preflight = "INVALID" if hold_violation else "REVIEW_INVALID_IDENTITY"
        hold = {**HOLD, "TEST_TOUCHED": "YES" if hold_violation else "NO"}
        if hold_violation:
            hold["rejected_input_hold_evidence"] = error.evidence
        status = {"SEED": SEED, "F0_IDENTITY": "UNVERIFIED" if hold_violation else "FAIL", "PREFLIGHT": preflight,
                  "ZERO_TRAINING_PREFLIGHT": "INVALID", "PAIRWISE_HEAD_IMPLEMENTATION": "NOT_AUTHORIZED",
                  "TEST_TOUCHED": hold["TEST_TOUCHED"], "HANDOFF_READY": "NO"}
        failure = {**status, "thresholds": THRESHOLDS, "reason": str(error)}
        write_json_atomic(output / DECISION, failure)
        write_json_atomic(output / IDENTITY, {**resolved, "status": preflight, "reason": str(error)})
        write_json_atomic(output / HOLD_STATUS, hold)
        write_json_atomic(output / MANIFEST, {**manifest, **status, "status": "invalid"})
        (output / REPORT).write_text(f"# CT-MultiDAG seed100 零训练 Preflight\n\n{preflight}\n\n{error}\n", encoding="utf-8")
        print_status(status)
        print(f"REASON = {error}", flush=True)
        return 2

    summary, tables = analyze(train_scores, dev_scores, q4, subgroups)
    outcome = decision(summary["recoverable_f"], summary["oracle_delta_weighted_f1_pp"])
    status = {"SEED": SEED, "F0_IDENTITY": "PASS", "TRAIN_GATE_SUPPORT": summary["n_g_train"],
              "DEV_GATE_SUPPORT": summary["n_g_dev"], "F0_DEV_ERRORS": summary["f0_dev_errors"],
              "TRUE_TOP2_RATE_IN_GATE": summary["true_top2_rate"], "RECOVERABLE_F": summary["recoverable_f"],
              "ORACLE_DELTA_WF1_PP": summary["oracle_delta_weighted_f1_pp"], **outcome,
              "TEST_TOUCHED": "NO", "H_SUBGROUP_STATUS": trace_status["H_SUBGROUP_STATUS"], "HANDOFF_READY": "YES"}
    evidence = {**status, **trace_status, "thresholds": THRESHOLDS, "top2_logit_margin_upper_exclusive": MARGIN_LIMIT,
                "train_mad_threshold": summary["train_mad_threshold"]}
    write_csv(output / SUMMARY, [{"seed": SEED, **summary}])
    for name, rows in tables.items():
        write_csv(output / name, rows)
    write_json_atomic(output / DECISION, evidence)
    write_json_atomic(output / IDENTITY, resolved)
    write_json_atomic(output / HOLD_STATUS, {**HOLD, "TEST_TOUCHED": "NO"})
    report(output, summary, evidence)
    names = (SUMMARY, *tables, DECISION, IDENTITY, HOLD_STATUS, REPORT)
    write_json_atomic(output / MANIFEST, {**manifest, **evidence, **HOLD, "status": "completed",
                                        "score_units": "fraction", "delta_units": "percentage_points",
                                        "outputs_sha256": {name: sha256_file(output / name) for name in names}})
    digest = bundle(output, archive, (*names, MANIFEST))
    print_status(status)
    print(f"PREFLIGHT_ARCHIVE = {archive}\nPREFLIGHT_ARCHIVE_SHA256 = {digest}")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-runs", type=Path, required=True)
    parser.add_argument("--train-dev-package", type=Path, required=True)
    parser.add_argument("--round1-dir", type=Path)
    parser.add_argument("--batch-id", required=True)
    parser.add_argument("--experiment-date")
    parser.add_argument("--output-root", default="outputs")
    args = parser.parse_args(argv)
    errors = check_workstream("multidag")
    if errors:
        print("WORKSTREAM_CHECK = FAIL", flush=True)
        parser.error("\n".join(errors))
    print("WORKSTREAM_CHECK = PASS", flush=True)
    if not sys.platform.startswith("linux") or not torch.cuda.is_available():
        parser.error("formal frozen preflight requires the Ubuntu CUDA workspace")
    date = resolve_experiment_date(args.experiment_date)
    paths = resolve_output_paths(output_base=args.output_root, experiment_date=date, experiment_group=GROUP,
                                 batch_id=args.batch_id)
    output = paths.analysis_root
    archive = output.parent / f"{output.name}.zip"
    return execute(args, output, archive, device="cuda", git=git_identity(), experiment_date=date)


if __name__ == "__main__":
    raise SystemExit(main())

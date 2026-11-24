"""Run only the preregistered seed100 F0 + epoch-20 Pairwise Head Fast-Kill."""

import argparse
import pickle
from pathlib import Path
import sys
import time

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

import numpy as np
import torch
import yaml

from models.multidag_cl.pairwise_refinement import PARAMETER_COUNT
from scripts.analysis.models.multidag_cl.pairwise_fastkill_metrics import (
    GO_LIMITS, RECOVERABLE_F, STOP_LIMITS, analyze, decision,
)
from scripts.analysis.models.multidag_cl.pairwise_fastkill_runtime import (
    BATCH_SIZE, EPOCHS, TRAINING, cache_identity, collect_causal_probes, infer, supervised_rows, sync, train_head,
    verify_causal_probes, verify_writeback,
)
from scripts.analysis.models.multidag_cl.pairwise_preflight_inputs import (
    CLASS_ORDER, F0_DEV_CORRECT, F0_DEV_WF1_PERCENT, HOLD, PROJECTION_MANIFEST, Q4_SUPPORT, SEED, TRAIN_DEV_COUNTS,
    TestHoldViolation, audit_subgroups, forward_f0, load_projected_train_dev, read_f0, validate_hold, validate_population,
)
from scripts.analysis.models.multidag_cl.pairwise_preflight_metrics import MARGIN_LIMIT, gate, oracle, scores, train_threshold
from scripts.analysis.models.multidag_cl.run_pairwise_preflight import bundle, print_status
from scripts.dev.check_workstream import check_workstream
from scripts.runtime.multidag_cl_ta_v0.formal import git_identity, json_identity, sha256_file, write_json_atomic
from scripts.runtime.multidag_cl_ta_v0.training import classification_report
from scripts.runtime.multidag_cl_tav_full_audit_v1.spec import write_csv
from utils.output_paths import resolve_experiment_date, resolve_output_paths


GROUP = "pairwise_fastkill_seed100"
CONFIG = Path("configs/multidag_cl/tav_full_audit_v1/pairwise_fastkill_seed100.yaml")
SUMMARY = "PAIRWISE_FASTKILL_SUMMARY.csv"
DECISION = "PAIRWISE_FASTKILL_DECISION.json"
REPORT = "PAIRWISE_FASTKILL_REPORT.md"
MANIFEST = "run_manifest.json"
HOLD_STATUS = "test_hold_status.json"
CHECKPOINT = "pairwise_head_epoch20.pt"


def enabled(config):
    raw = yaml.safe_load(config.read_text(encoding="utf-8"))
    if (not isinstance(raw, dict) or set(raw) != {"pairwise_refinement"}
            or not isinstance(raw["pairwise_refinement"], dict) or set(raw["pairwise_refinement"]) != {"enabled"}
            or type(raw["pairwise_refinement"]["enabled"]) is not bool):
        raise ValueError("expected only the boolean pairwise_refinement.enabled YAML switch")
    return raw["pairwise_refinement"]["enabled"]


def verify_preflight(train, dev, train_gate, dev_gate):
    oracle_prediction, recovered = oracle(dev, dev_gate)
    base = classification_report(dev.labels, dev.top1, loss=0)["metrics"]
    upper = classification_report(dev.labels, oracle_prediction, loss=0)["metrics"]
    oracle_pp = 100 * (upper["weighted_f1"] - base["weighted_f1"])
    supervised = supervised_rows(train.labels, train.top1, train.top2, train_gate)
    n_train_gate, n_dev_gate = int(train_gate.sum()), int(dev_gate.sum())
    n_first = int((supervised & (train.labels == train.top1)).sum())
    n_second = int((supervised & (train.labels == train.top2)).sum())
    n_top2_dev = int((dev_gate & (dev.true_rank <= 2)).sum())
    if (int((~dev.wrong).sum()) != F0_DEV_CORRECT or round(100 * base["weighted_f1"], 4) != F0_DEV_WF1_PERCENT
            or (n_train_gate, n_dev_gate, int(dev.wrong.sum()), n_top2_dev, int(recovered.sum()))
            != (539, 159, 287, 140, RECOVERABLE_F)
            or not np.isclose(oracle_pp, 5.548124025672507, rtol=0, atol=1e-10)
            or n_first + n_second < 128 or min(n_first, n_second) < 16):
        raise ValueError("live Train/Dev gate or oracle differs from the frozen seed100 Preflight PASS")
    return {"train_gate_support": n_train_gate, "dev_gate_support": n_dev_gate,
            "dev_true_top2_rate_in_gate": n_top2_dev / n_dev_gate, "train_supervised_support": n_first + n_second,
            "train_y_eq_a": n_first, "train_y_eq_b": n_second,
            "train_gate_true_rank3plus": n_train_gate - n_first - n_second,
            "train_true_rank3plus": int((train.true_rank >= 3).sum()),
            "recoverable_f": RECOVERABLE_F, "oracle_delta_wf1_pp": oracle_pp}


def write_report(output, summary, tables, verdict, manifest):
    rows = ["# CT-MultiDAG seed100 Pairwise Fast-Kill", "",
            f'**{verdict["FASTKILL"]}** · CTRL 第 21 轮 · PRIMARY head 第 {EPOCHS} 轮', "",
            "| Dev 指标 | CTRL (%) | PRIMARY (%) | Δ (pp) |", "| --- | ---: | ---: | ---: |"]
    rows += [f'| {r["metric"]} | {r["ctrl_pct"]:.6f} | {r["primary_pct"]:.6f} | {r["delta_pp"]:+.6f} |'
             for r in tables["DEV_METRICS.csv"]]
    rows += ["", "| 样本集合 | 支持 | R | D | NET |", "| --- | ---: | ---: | ---: | ---: |"]
    rows += [f'| {r["population"]} | {r["support"]} | {r["r"]} | {r["d"]} | {r["net"]:+d} |'
             for r in tables["Q4_RECOVERY.csv"]]
    rows += [
        "", f'完整 Dev 正确数为 {summary["ctrl_correct"]} → {summary["primary_correct"]} / {summary["support"]}。',
        f'实际恢复占冻结理论机会的 {summary["r"]}/{RECOVERABLE_F}（{summary["realized_recovery_ratio"]:.2%}）。',
        f'Dev gate 覆盖 {summary["gate_support"]}/{summary["support"]}。',
        "", "Bootstrap 按 dialogue 配对抽样，CI 用于解释。NLL、Brier 和 ECE 为 calibration side effects。",
        "", f'Q4 来源 {manifest["Q4_SOURCE"]}，SHIFT 分层 {manifest["SHIFT_SUBGROUP_STATUS"]}，'
        f'U/H 分层 {manifest["H_SUBGROUP_STATUS"]}。',
        f'Head 工件 {CHECKPOINT}，SHA-256 {manifest["head_checkpoint"]["sha256"]}。',
        "", "Test 保持 HOLD。", "",
    ]
    (output / REPORT).write_text("\n".join(rows), encoding="utf-8")


def execute(args, output, archive, *, device, git, experiment_date):
    if archive.exists():
        raise FileExistsError("existing Fast-Kill archive cannot be overwritten")
    output.mkdir(parents=True, exist_ok=False)
    manifest = {**git, **HOLD, "schema": "ct_multidag_pairwise_fastkill_v1", "experiment_group": GROUP,
                "experiment_date": experiment_date, "batch_id": args.batch_id, "seed": SEED, "device": device,
                "training": TRAINING, "class_order": CLASS_ORDER, "head_parameter_count": PARAMETER_COUNT,
                "formal_train_runs": 0, "primary_dev_evaluation_count": 0}
    stage, identity = "F0_IDENTITY", "UNVERIFIED"
    hold = {**HOLD, "TEST_TOUCHED": "NO", "TEST_HOLD": "PASS"}
    cuda = torch.device(device).type == "cuda"
    try:
        source = read_f0(args.frozen_runs)
        stage = "TRAIN_DEV_PROJECTION"
        path = args.train_dev_package / PROJECTION_MANIFEST
        projection_sha = sha256_file(path)
        datasets, projection = load_projected_train_dev(args.train_dev_package, source)
        if sha256_file(path) != projection_sha:
            raise ValueError("Train/Dev projection certificate changed while authenticating")
        populations = validate_population(datasets, source)
        manifest.update(f0=source.identity, f0_config_sha256=source.manifest["config_sha256"],
                        architecture_identity=source.manifest["architecture_identity"], populations=populations,
                        projection=projection, projection_sha256=projection_sha)
        stage = "FROZEN_SUBGROUPS"
        q4, groups, trace = audit_subgroups(source, args.round1_dir)
        manifest.update(trace)
        if (q4.shape != (TRAIN_DEV_COUNTS["dev"]["utterances"],) or int(q4.sum()) != Q4_SUPPORT
                or trace["SHIFT_SUBGROUP_STATUS"] != "AVAILABLE"):
            raise ValueError("expected authenticated frozen Round 1 SHIFT and MAD Q4 trace")

        probes = {}

        def inspect_model(model, device):
            nonlocal stage
            stage = "CAUSAL_PROBE_INPUTS"
            probes.update(collect_causal_probes(model, device))
            stage = "F0_REPLAY"

        if cuda:
            torch.cuda.reset_peak_memory_stats(device)
        stage = "F0_REPLAY"
        train, dev = forward_f0(source, datasets, device, capture_states=True, inspect_model=inspect_model)
        identity = "PASS"
        stage = "FROZEN_PREFLIGHT"
        caches = {split: cache_identity(cache) for split, cache in (("train", train), ("dev", dev))}
        if any(caches[split]["population"] != size["utterances"] for split, size in TRAIN_DEV_COUNTS.items()):
            raise ValueError("incomplete frozen Train/Dev cache population")
        train_scores, dev_scores = scores(train.logits, train.labels), scores(dev.logits, dev.labels)
        threshold = train_threshold(train_scores.mad)
        train_gate = gate(train_scores.mad, train_scores.margin, True, threshold)
        dev_gate = gate(dev_scores.mad, dev_scores.margin, True, threshold)
        preflight = verify_preflight(train_scores, dev_scores, train_gate, dev_gate)
        manifest.update(caches=caches, preflight=preflight, train_mad_threshold=threshold,
                        margin_upper_exclusive=MARGIN_LIMIT,
                        dev_gold_population_sha256=json_identity(dev.population),
                        f0_dev_logits_max_abs_diff=float(np.max(np.abs(dev.logits - source.logits))),
                        f0_train_dataset_passes=1, f0_dev_dataset_passes=1, f0_gradients_disabled=True)
        validate_hold(hold)

        stage = "TRAIN_EPOCH20"
        f0_peak = torch.cuda.max_memory_allocated(device) if cuda else None
        if cuda:
            torch.cuda.reset_peak_memory_stats(device)
        manifest["formal_train_runs"] = 1
        head, history, training_seconds = train_head(train, train_scores, train_gate, device=device)
        write_csv(output / "TRAIN_HISTORY.csv", history)
        manifest.update(epoch=EPOCHS, optimizer_steps=EPOCHS * ((preflight["train_supervised_support"] + BATCH_SIZE - 1)
                                                               // BATCH_SIZE))
        head_peak = torch.cuda.max_memory_allocated(device) if cuda else None

        stage = "EPOCH20_INFERENCE"
        logits, h0, h4 = (torch.as_tensor(value, device=device) for value in (dev.logits, *dev.layer_states))
        valid = torch.ones(len(logits), dtype=torch.bool, device=device)
        sync(device)
        start = time.perf_counter()
        result = infer(head, logits, h0, h4, valid, threshold)
        sync(device)
        inference_seconds = time.perf_counter() - start
        verify_writeback(logits, valid, result)
        if not np.array_equal(result.selected.cpu().numpy(), dev_gate):
            raise ValueError("label-free Dev gate differs from frozen Preflight gate")

        stage = "CAUSAL_INVARIANCE"
        manifest["causal_invariance"] = verify_causal_probes(head, probes, threshold, logits[result.selected][0])
        stage = "EPOCH20_SCORING"
        validate_hold(hold)
        manifest["primary_dev_evaluation_count"] = 1
        summary, tables = analyze(dev, result, q4, groups)
        outcome = decision(summary, tables["PAIR_CONFUSIONS.csv"])
        measured_peak = max(f0_peak, torch.cuda.max_memory_allocated(device)) if cuda else None
        resources = {"training_wall_seconds": training_seconds, "peak_vram_allocated_bytes": measured_peak,
                     "f0_peak_vram_allocated_bytes": f0_peak, "head_stage_peak_vram_allocated_bytes": head_peak,
                     "head_inference_overhead_seconds": inference_seconds, "head_inference_rows": len(logits),
                     "head_inference_timing_scope": "cached_device_inputs_to_final_logits_including_gate_and_sync",
                     "train_cache_tensor_bytes": caches["train"]["tensor_bytes"]}
    except (OSError, ValueError, KeyError, TypeError, IndexError, RuntimeError, pickle.UnpicklingError,
            EOFError, yaml.YAMLError) as error:
        if isinstance(error, TestHoldViolation):
            hold.update(TEST_HOLD="FAIL", rejected_input_hold_evidence=error.evidence)
        elif stage in ("F0_IDENTITY", "F0_REPLAY"):
            identity = "FAIL"
        status = {"FASTKILL": "REVIEW_INVALID", "F0_IDENTITY": identity, "TEST_TOUCHED": "NO", "HANDOFF_READY": "NO"}
        failure = {**status, "stage": stage, "reason": str(error), "go_limits": GO_LIMITS, "stop_limits": STOP_LIMITS}
        write_json_atomic(output / DECISION, failure)
        write_json_atomic(output / HOLD_STATUS, hold)
        write_json_atomic(output / MANIFEST, {**manifest, **failure, "status": "invalid"})
        (output / REPORT).write_text(f"# CT-MultiDAG seed100 Pairwise Fast-Kill\n\nREVIEW_INVALID · {stage}\n\n{error}\n",
                                     encoding="utf-8")
        print_status({**status, "STAGE": stage, "REASON": str(error)})
        return 2

    binding = {"epoch": EPOCHS, "parameter_count": PARAMETER_COUNT, "f0_checkpoint_sha256": source.identity["checkpoint_sha256"],
               "projection_sha256": projection_sha, "train_mad_threshold": threshold, "class_order": CLASS_ORDER,
               "seed": SEED}
    torch.save({**binding, "state_dict": {name: value.detach().cpu() for name, value in head.state_dict().items()}},
               output / CHECKPOINT)
    manifest.update(head_checkpoint={**binding, "path": CHECKPOINT, "sha256": sha256_file(output / CHECKPOINT)},
                    resources=resources, bypass_identity="PASS")
    status = {**outcome, "SEED": SEED, "F0_IDENTITY": "PASS", "HEAD_EPOCH": EPOCHS, "TEST_TOUCHED": "NO"}
    verdict = {**status, "train_mad_threshold": threshold, "summary": summary}
    write_csv(output / SUMMARY, [{"seed": SEED, "fastkill": outcome["FASTKILL"], **summary, **resources}])
    for name, rows in tables.items():
        write_csv(output / name, rows)
    write_json_atomic(output / DECISION, verdict)
    write_json_atomic(output / HOLD_STATUS, hold)
    write_report(output, summary, tables, verdict, manifest)
    names = (SUMMARY, *tables, "TRAIN_HISTORY.csv", DECISION, HOLD_STATUS, REPORT)
    write_json_atomic(output / MANIFEST, {**manifest, **status, "status": "completed",
                                        "outputs_sha256": {name: sha256_file(output / name) for name in names}})
    digest = bundle(output, archive, (*names, MANIFEST))
    print_status({"FASTKILL": outcome["FASTKILL"], "F0_IDENTITY": "PASS", "HEAD_EPOCH": EPOCHS,
                  "DELTA_WF1_PP": summary["delta_weighted_f1_pp"], "NET": summary["net"], "Q4_NET": summary["q4_net"],
                  "TEST_TOUCHED": "NO", "HANDOFF_READY": "YES",
                  "FASTKILL_ARCHIVE": archive, "FASTKILL_ARCHIVE_SHA256": digest})
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--frozen-runs", type=Path, required=True)
    parser.add_argument("--train-dev-package", type=Path, required=True)
    parser.add_argument("--round1-dir", type=Path, required=True)
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
        parser.error("formal Fast-Kill requires the Ubuntu CUDA workspace")
    try:
        if not enabled(args.config):
            parser.error("the formal two-arm Fast-Kill requires pairwise_refinement.enabled: true")
    except (OSError, ValueError, yaml.YAMLError) as error:
        parser.error(str(error))
    date = resolve_experiment_date(args.experiment_date)
    paths = resolve_output_paths(output_base=args.output_root, experiment_date=date,
                                 experiment_group=GROUP, batch_id=args.batch_id)
    output = paths.analysis_root
    return execute(args, output, output.parent / f"{output.name}.zip", device="cuda",
                   git=git_identity(), experiment_date=date)


if __name__ == "__main__":
    raise SystemExit(main())

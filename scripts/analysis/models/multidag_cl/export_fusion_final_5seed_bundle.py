"""Validate and archive the complete paired final Fusion Dev evidence."""

from decimal import Decimal
from pathlib import Path
import shutil
from statistics import mean, median, stdev

import yaml

from models.multidag_cl.ta_v0 import LABEL_VOCAB
from scripts.analysis.models.multidag_cl.export_fusion_topology_bundle import (
    EDGE_METRICS, HOLD, MANIFEST_NAME, METRICS, check_hold, pack_verified, read_json, read_run, write_run_tables,
)
from scripts.runtime.multidag_cl_fusion.spec import (
    FINAL_ARMS, FINAL_CONFIG, FINAL_PROTOCOL, ROOT, TOPOLOGY_EDGES, final_seed_source, make_final_config,
)
from scripts.runtime.multidag_cl_ta_v0.formal import sha256_file, write_json_atomic
from scripts.runtime.multidag_cl_tav_full_audit_v1.spec import CONFIRM_SEEDS, write_csv
from utils.output_paths import validate_experiment_date


SCORE_METRICS = tuple(metric for metric in METRICS if metric != "loss")
MIN_STABLE_WINS = 4
WF1_THRESHOLD_PP = Decimal("0.20")


def archive_path(date):
    date = validate_experiment_date(date)
    return Path("tmp") / f"CT_MultiDAG_Fusion三臂五随机种子最终确认_分析数据包_{date}.tar.gz"


def describe(values):
    return {"n": len(values), "mean": mean(values), "std": stdev(values), "median": median(values),
            "min": min(values), "max": max(values)}


def paired_statistics(summary):
    expected = {(seed, arm) for seed in CONFIRM_SEEDS for arm in FINAL_ARMS}
    runs = {(row["seed"], row["variant"]): row for row in summary}
    if set(runs) != expected or len(summary) != len(expected):
        raise ValueError("statistics require all 15 distinct final Fusion runs")
    multi_seed, deltas, paired, decisions = [], [], [], {}
    for arm in FINAL_ARMS:
        for metric in METRICS:
            values = [runs[(seed, arm)][metric] for seed in CONFIRM_SEEDS]
            multi_seed.append({"variant": arm, "split": "dev", "metric": metric, **describe(values)})
        if arm == "F0":
            continue
        by_metric = {}
        for metric in SCORE_METRICS:
            differences = []
            for seed in CONFIRM_SEEDS:
                baseline, candidate = runs[(seed, "F0")][metric], runs[(seed, arm)][metric]
                delta = (Decimal(str(candidate)) - Decimal(str(baseline))) * 100
                differences.append(delta)
                deltas.append({"variant": arm, "baseline_variant": "F0", "seed": seed, "split": "dev",
                               "metric": metric, "baseline_metric": baseline, "candidate_metric": candidate,
                               "paired_delta": float(delta), "paired_delta_unit": "pp"})
            wins = sum(value > 0 for value in differences)
            ties = differences.count(Decimal(0))
            losses = len(differences) - wins - ties
            by_metric[metric] = (mean(differences), wins, ties, losses, differences)
            paired.append({"variant": arm, "baseline_variant": "F0", "split": "dev", "metric": metric,
                           **describe([float(value) for value in differences]),
                           "wins": wins, "ties": ties, "losses": losses, "unit": "pp"})
        wf1, wins, ties, losses, _ = by_metric["weighted_f1"]
        macro, uar = by_metric["macro_f1"], by_metric["uar"]
        joint_wins = sum(m > 0 and u > 0 for m, u in zip(macro[-1], uar[-1]))
        keep = wf1 >= WF1_THRESHOLD_PP and wins + ties >= MIN_STABLE_WINS and min(macro[0], uar[0]) >= 0
        trade_off = (-WF1_THRESHOLD_PP < wf1 < WF1_THRESHOLD_PP and min(macro[0], uar[0]) > 0
                     and joint_wins >= MIN_STABLE_WINS)
        stop = wf1 <= 0 and wins < MIN_STABLE_WINS
        decisions[arm] = {"mean_delta_weighted_f1_pp": float(wf1), "wins": wins, "ties": ties, "losses": losses,
                          "mean_delta_macro_f1_pp": float(macro[0]), "mean_delta_uar_pp": float(uar[0]),
                          "joint_macro_f1_uar_wins": joint_wins, "strong_keep": keep, "trade_off": trade_off,
                          "stop": stop}
    if any(item["strong_keep"] for item in decisions.values()):
        route = "KEEP"
    elif all(item["stop"] for item in decisions.values()):
        route = "STOP"
    elif any(item["trade_off"] for item in decisions.values()):
        route = "TRADE_OFF"
    else:
        route = "NO_RULE_MATCH"
    return multi_seed, deltas, paired, {"route": route, "candidates": decisions}


def mechanism_summary(rows, seeds):
    selected = {(row["variant"], row["edge"], row["seed"]): row
                for row in rows if row["stage"] == "best_checkpoint_dev"}
    expected = {(arm, edge, seed) for arm in FINAL_ARMS[1:] for edge in TOPOLOGY_EDGES[arm] for seed in seeds}
    if selected.keys() != expected:
        raise ValueError("selected mechanism evidence is incomplete")
    return [{"variant": arm, "split": "dev", "stage": "best_checkpoint_dev", "edge": edge, "metric": metric,
             **describe([selected[(arm, edge, seed)][metric] for seed in seeds])}
            for arm in FINAL_ARMS[1:] for edge in TOPOLOGY_EDGES[arm] for metric in EDGE_METRICS]


def verify_pairing(job, reference, *, device):
    arm, seed, result = job["variant"], job["seed"], job["result"]
    init = result["shared_initialization"]
    rng_fields = {"python", "numpy", "torch_cpu"}
    if device == "cuda":
        rng_fields.add("torch_cuda")
    expected_edges = {edge: reference["canonical_h4_edge_initial_state_sha256"][edge] for edge in TOPOLOGY_EDGES[arm]}
    if (reference["shared_backbone_init_max_abs_diff"] != 0 or reference["common_edge_init_max_abs_diff"] != 0
            or init["shared_parameter_max_abs_diff"] != 0 or init["common_edge_init_max_abs_diff"] != 0
            or reference["initializations"][arm] != init
            or init["shared_initial_state_sha256"] != reference["shared_initial_state_sha256"]
            or init["edge_initial_state_sha256"] != expected_edges
            or init["training_cpu_rng_sha256"] != reference["training_cpu_rng_sha256"]
            or init["training_rng_identity"] != reference["training_rng_identity"]
            or set(init["training_rng_identity"]) != rng_fields
            or result["batch_pairing"] != reference["f0_batch_pairing"]):
        raise ValueError(f"initialization, training RNG or batch pairing mismatch for seed {seed} {arm}")


def write_report(path, summary, paired, decision, state):
    lines = ["# CT-MultiDAG Fusion 三臂五随机种子最终确认", "",
             f"批次 `{state['batch_id']}` · Git `{state['git_head']}` · Test HOLD。",
             f"正式 seed 来源为冻结 `track_tav_cloff` 确认矩阵。Seeds 为 {', '.join(map(str, state['seed_source']['seeds']))}。",
             "评分按相同 seed 配对，每个 seed 等权。差值单位为 pp，标准差为样本标准差。", ""]
    if state["synthetic_only"]:
        lines.extend(["本包使用合成数据验证工作流，研究判定留待正式 Dev 结果。", ""])
    lines.extend(["## Dev 最佳 checkpoint", "",
                  "| Seed | Variant | Epoch | WF1 (%) | Macro-F1 (%) | UAR (%) | Accuracy (%) | Dev loss | 参数量 |",
                  "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"])
    for row in summary:
        scores = " | ".join(f"{100 * row[metric]:.4f}" for metric in SCORE_METRICS)
        lines.append(f"| {row['seed']} | {row['variant']} | {row['best_epoch']} | {scores} | "
                     f"{row['loss']:.6f} | {row['total_parameter_count']:,} |")
    lines.extend(["", "## 相对同 seed F0 的配对差值（pp）", "",
                  "| Variant | 指标 | Mean | SD | Median | Min | Max | W / T / L |",
                  "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |"])
    for row in paired:
        stats = " | ".join(f"{row[key]:+.4f}" for key in ("mean", "std", "median", "min", "max"))
        lines.append(f"| {row['variant']} | {row['metric']} | {stats} | {row['wins']} / {row['ties']} / {row['losses']} |")
    lines.extend(["", "## 判定", "",
                  "稳定类别平衡改善要求 Macro-F1 和 UAR 均值均为正，且至少四个相同 seed 上两项均严格上升。",
                  "稳定 WF1 wins 指至少四个 seed 上严格胜过 F0。两个候选都满足 STOP 时冻结 Fusion 路线。", ""])
    for arm, item in decision["candidates"].items():
        statuses = [label for key, label in (("strong_keep", "STRONG / KEEP"), ("trade_off", "TRADE-OFF（类别平衡改善）"),
                                             ("stop", "STOP")) if item[key]]
        lines.append(f"- {arm} · {', '.join(statuses) if statuses else 'NO_RULE_MATCH'}")
    lines.extend(["", f"Fusion 路线判定 `{decision['route']}`。", "",
                  "`fusion_mechanism_multiseed_summary.csv` 汇总各边在五个最佳 Dev checkpoint 上的门值、"
                  "message/receiver 和 applied residual/receiver。", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def export_final_bundle(directory, archive, master, state):
    check_hold(state)
    seed_source = final_seed_source()
    seeds = seed_source["seeds"]
    jobs = state["training_jobs"]
    pairings = state["pairing_by_seed"]
    if (state["protocol"] != FINAL_PROTOCOL or state["seed_source"] != seed_source
            or set(pairings) != {str(seed) for seed in seeds}
            or [(job["seed"], job["variant"]) for job in jobs] != [(seed, arm) for seed in seeds for arm in FINAL_ARMS]):
        raise ValueError("final Fusion bundle requires the exact historical five-seed paired design")
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "configs").mkdir()
    shutil.copyfile(ROOT / FINAL_CONFIG, directory / "configs/confirmation.yaml")
    summary, checkpoints, classes, mechanisms = [], [], [], []
    population_ref = None
    for job in jobs:
        arm, seed, run_dir = job["variant"], job["seed"], Path(job["run_dir"])
        raw = make_final_config(arm, seed)
        resolved_path = run_dir / "resolved_config.yaml"
        resolved = yaml.safe_load(resolved_path.read_text(encoding="utf-8"))
        if (job["experiment_id"] != raw["experiment_id"] or job["result"]["seed"] != seed
                or read_json(run_dir / "manifests/run_manifest.json") != job["result"]
                or resolved["audit_config"] != raw or resolved["data_identity"] != state["data_identity"]):
            raise ValueError(f"on-disk run/config identity mismatch for seed {seed} {arm}")
        verify_pairing(job, pairings[str(seed)], device=state["device"])
        row, checkpoint, history, predictions, population, evidence, stats = read_run(job, state, raw=raw)
        if population_ref is not None and population != population_ref:
            raise ValueError("Dev prediction population/order/labels differ across runs")
        population_ref = population
        config = f"configs/{row['run_id']}.yaml"
        shutil.copyfile(resolved_path, directory / config)
        summary.append({**row, "resolved_config": config, "topology_source_config": job["topology_source_config"]})
        checkpoints.append({**checkpoint, "resolved_config": config, "git_head": state["git_head"]})
        classes.extend({"variant": arm, "seed": seed, "split": "dev", **item} for item in evidence["per_class"])
        mechanisms.extend(stats)
        write_run_tables(directory, row, history, predictions, evidence)

    multi_seed, deltas, paired, decision = paired_statistics(summary)
    for name, rows in (("run_summary", summary), ("multi_seed_summary", multi_seed), ("paired_seed_deltas", deltas),
                       ("paired_seed_summary", paired), ("per_class_metrics", classes),
                       ("fusion_mechanism_stats", mechanisms),
                       ("fusion_mechanism_multiseed_summary", mechanism_summary(mechanisms, seeds))):
        write_csv(directory / f"{name}.csv", rows)
    labels = list(LABEL_VOCAB)
    write_json_atomic(directory / "label_vocab.json", {"itos": labels, "stoi": dict(zip(labels, range(len(labels))))})
    write_json_atomic(directory / "checkpoint_identity.json", {"selector": raw["checkpoint"]["selector"],
                                                               "checkpoints": checkpoints})
    write_json_atomic(directory / "pairing_audit.json", pairings)
    write_json_atomic(directory / "test_hold_status.json", {**HOLD, "test_labels_inspected": state["test_labels_inspected"],
                                                           "test_used_for_selection": state["test_used_for_selection"]})
    write_json_atomic(directory / "fusion_decision.json", {"synthetic_only": state["synthetic_only"], **decision})
    write_report(directory / "experiment_report.md", summary, paired, decision, state)
    master.flush()
    shutil.copyfile(master.name, directory / "master.log")
    inventory = {path.relative_to(directory).as_posix(): sha256_file(path)
                 for path in sorted(directory.rglob("*")) if path.is_file()}
    fields = ("protocol", "batch_id", "experiment_date", "experiment_group", "git_head", "device", "synthetic_only",
              "seed_source", "data_identity")
    manifest = {"schema": "ct_multidag_fusion_final_paired_analysis_bundle_v1", "ANALYSIS_BUNDLE": "PASS",
                "PLOT_READY": "YES", **{key: state[key] for key in fields}, "git_dirty": state.get("git_dirty"),
                "score_units": "fraction", "paired_delta_units": "percentage_points", "std_ddof": 1,
                "split": "dev", "confusion_normalization": "true_label_row", "variants": list(FINAL_ARMS),
                "run_count": len(summary), "test_policy": "HOLD", "files_sha256": inventory}
    manifest_path = directory / MANIFEST_NAME
    write_json_atomic(manifest_path, manifest)
    try:
        digest = pack_verified(directory, archive, inventory)
    except BaseException:
        manifest_path.unlink()
        raise
    return {"ANALYSIS_BUNDLE": "PASS", "PLOT_READY": "YES", "archive": archive.as_posix(),
            "archive_sha256": digest, "decision": decision["route"]}

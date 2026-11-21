"""Verify selected Dev evidence and package the topology screen for plotting."""

import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tarfile
from tempfile import NamedTemporaryFile

from models.multidag_cl.ta_v0 import LABEL_VOCAB
from scripts.runtime.multidag_cl_fusion.spec import (
    EDGE_PARAMETER_COUNT, F0_PARAMETER_COUNT, TOPOLOGY_EDGES, make_topology_config,
)
from scripts.runtime.multidag_cl_ta_v0.formal import json_identity, sha256_file, write_json_atomic
from scripts.runtime.multidag_cl_ta_v0.training import classification_report
from scripts.runtime.multidag_cl_tav_full_audit_v1.confirm import test_hold
from scripts.runtime.multidag_cl_tav_full_audit_v1.spec import write_csv


ARCHIVE_NAME = "CT_MultiDAG_Fusion非对称拓扑筛查_分析数据包_20261121.tar.gz"
METRICS = ("weighted_f1", "macro_f1", "uar", "accuracy", "loss")
EDGE_METRICS = (
    "gate_mean", "gate_std", "gate_q05", "gate_q25", "gate_q50", "gate_q75", "gate_q95",
    "message_l2_mean", "receiver_l2_mean", "message_receiver_ratio_mean",
    "applied_edge_residual_receiver_ratio_mean", "applied_receiver_residual_receiver_ratio_mean",
)
HOLD = {"test_policy": "HOLD", "test_loader_built": 0, "test_evaluation_count": 0,
        "test_checkpoint_selection_count": 0}
SECONDARY_DROP_PP = -0.20
MANIFEST_NAME = "analysis_bundle_manifest.json"


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def check_hold(record):
    test_hold(record)
    if type(record.get("test_checkpoint_selection_count")) is not int or record["test_checkpoint_selection_count"] != 0:
        raise ValueError("Test HOLD checkpoint selection counter mismatch")


def read_run(job, state):
    arm, run_dir, result = job["variant"], Path(job["run_dir"]), job["result"]
    raw = make_topology_config(arm)
    synthetic = state["synthetic_only"]
    expected_epochs = result["last_completed_epoch"] if synthetic else raw["training"]["max_epochs"]
    edges = TOPOLOGY_EDGES[arm]
    check_hold(result)
    if (job["status"] != "PASS" or result["status"] != ("synthetic_completed" if synthetic else "completed")
            or result["synthetic_only"] is not synthetic or not result["training_finished"]
            or not result["best_checkpoint_reloaded"] or result["last_completed_epoch"] != expected_epochs
            or result["config_sha256"] != job["config_sha256"] or result["config_sha256"] != json_identity(raw)
            or result["data_identity"] != state["data_identity"] or result["git_head"] != state["git_head"]
            or result["active_edges"] != list(edges)
            or result["parameter_count"] != result["total_parameter_count"]
            or result["total_parameter_count"] != F0_PARAMETER_COUNT + len(edges) * EDGE_PARAMETER_COUNT
            or result["fusion_parameter_count"] != len(edges) * EDGE_PARAMETER_COUNT
            or result["edge_parameter_count"] != dict.fromkeys(edges, EDGE_PARAMETER_COUNT)):
        raise ValueError(f"{arm} run manifest does not match the frozen screen")

    checkpoint = run_dir / "checkpoints/best_model.pt"
    checkpoint_sha = sha256_file(checkpoint)
    evidence = read_json(run_dir / "reports/validation_best_dev_evidence.json")
    check_hold(evidence)
    if (evidence["split"] != "dev" or evidence["optimizer_steps_performed"] != 0
            or evidence["export_mode"] != "final_selected_checkpoint_reload"
            or checkpoint_sha != evidence["checkpoint_sha256"] or checkpoint_sha != result["best_checkpoint_sha256"]
            or evidence["metrics"] != result["best_dev_metrics"]):
        raise ValueError(f"{arm} Dev prediction evidence does not identify the selected checkpoint")

    with (run_dir / "logs/epoch_metrics.tsv").open(encoding="utf-8", newline="") as stream:
        history = list(csv.DictReader(stream, delimiter="\t"))
    if [int(row["epoch"]) for row in history] != list(range(1, expected_epochs + 1)):
        raise ValueError(f"{arm} epoch history is incomplete")
    if any(not math.isfinite(float(row[key])) for row in history for key in row if key != "is_best"):
        raise ValueError(f"{arm} epoch history contains nonfinite values")
    best = max(history, key=lambda row: (float(row["dev_weighted_f1"]), -float(row["dev_loss"]), -int(row["epoch"])))
    if (int(best["epoch"]) != result["best_epoch"]
            or any(not math.isclose(float(best[f"dev_{key}"]), evidence["metrics"][key],
                                    rel_tol=1e-6, abs_tol=1e-8) for key in METRICS)):
        raise ValueError(f"{arm} selected report differs from the frozen Dev checkpoint selector")

    identity = {"variant": arm, "seed": raw["training"]["seed"], "split": "dev"}
    history = [{**identity, **row, "is_selected_checkpoint": int(row["epoch"]) == result["best_epoch"]}
               for row in history]
    pairing = result["batch_pairing"]["population"]["dev"]
    if not synthetic and any(pairing[key] != count for key, count in raw["dataset"]["population"]["dev"].items()):
        raise ValueError(f"{arm} Dev population does not match the frozen split")
    dev_batches = result["batch_pairing"]["dev"]
    predictions, population = prediction_rows(evidence, identity, pairing, dev_batches["dialogue_order_sha256"])
    mechanisms = mechanism_rows(run_dir, identity, edges, evidence, expected_epochs, result["best_epoch"])
    summary = {
        **identity, "run_id": job["experiment_id"], "best_epoch": result["best_epoch"],
        "epochs_completed": expected_epochs, **{key: evidence["metrics"][key] for key in METRICS},
        "dev_dialogues": pairing["dialogues"], "dev_utterances": pairing["utterances"],
        "active_edges": ";".join(edges), "edge_parameter_count": EDGE_PARAMETER_COUNT if edges else 0,
        "fusion_parameter_count": result["fusion_parameter_count"], "total_parameter_count": result["total_parameter_count"],
        "checkpoint_sha256": checkpoint_sha, "config_sha256": job["config_sha256"],
    }
    checkpoint_record = {
        **identity, "run_id": job["experiment_id"], "relative_to_run": "checkpoints/best_model.pt",
        "best_epoch": result["best_epoch"], "sha256": checkpoint_sha, "config_sha256": job["config_sha256"],
        "architecture_sha256": result["architecture_identity"], "dev_population_sha256": pairing["sha256"],
        "dev_batch_sequence_sha256": dev_batches["batch_sequence_sha256"],
    }
    return summary, checkpoint_record, history, predictions, population, evidence, mechanisms


def prediction_rows(evidence, identity, pairing, order_sha256):
    rows, population, targets, predictions = [], [], [], []
    utterances_by_dialogue = {}
    for item in evidence["predictions"]:
        uid, did = item["utterance_id"], item["dialogue_id"]
        target, prediction = item["true_label"], item["predicted_label"]
        logits, probabilities = item["logits"], item["probabilities"]
        if (not uid or not did or not 0 <= target < len(LABEL_VOCAB) or not 0 <= prediction < len(LABEL_VOCAB)
                or len(logits) != len(LABEL_VOCAB) or len(probabilities) != len(LABEL_VOCAB)
                or any(not math.isfinite(value) for value in logits + probabilities)
                or prediction != max(range(len(logits)), key=logits.__getitem__)):
            raise ValueError("invalid selected Dev prediction")
        maximum = max(logits)
        exp = [math.exp(value - maximum) for value in logits]
        denominator = sum(exp)
        if any(not math.isclose(prob, value / denominator, rel_tol=1e-5, abs_tol=1e-7)
               for prob, value in zip(probabilities, exp)):
            raise ValueError("Dev probabilities differ from logits")
        population.append((uid, did, target))
        utterances_by_dialogue.setdefault(did, []).append(uid)
        targets.append(target)
        predictions.append(prediction)
        rows.append({"utterance_id": uid, "dialogue_id": did, **identity, "true_label": LABEL_VOCAB[target],
                     "pred_label": LABEL_VOCAB[prediction], "true_label_id": target, "pred_label_id": prediction,
                     **{f"logit_{label}": value for label, value in zip(LABEL_VOCAB, logits)},
                     **{f"prob_{label}": value for label, value in zip(LABEL_VOCAB, probabilities)}})
    if (len(rows) != pairing["utterances"] or len({uid for uid, _, _ in population}) != len(rows)
            or len(utterances_by_dialogue) != pairing["dialogues"]
            or json_identity(list(utterances_by_dialogue.items())) != order_sha256):
        raise ValueError("Dev prediction population mismatch")
    recomputed = classification_report(targets, predictions, loss=evidence["metrics"]["loss"])
    if any(evidence[key] != value for key, value in recomputed.items()):
        raise ValueError("selected Dev metrics do not reproduce from exported predictions")
    return rows, population


def mechanism_rows(run_dir, identity, edges, evidence, epochs, best_epoch):
    if not edges:
        if evidence["fusion_mechanism"]:
            raise ValueError("F0 unexpectedly recorded Fusion mechanism statistics")
        return []
    reports = [("epoch_dev", epoch, read_json(run_dir / "logs" / f"fusion_mechanism_epoch_{epoch:02d}.json"))
               for epoch in range(1, epochs + 1)]
    reports.append(("best_checkpoint_dev", best_epoch, evidence["fusion_mechanism"]))
    rows = []
    for stage, epoch, report in reports:
        if set(report) != {*edges, "activity"}:
            raise ValueError("Fusion mechanism scope differs from active topology")
        for edge in edges:
            stats = report[edge]
            if (stats["active_count"] != evidence["metrics"]["valid_target_count"]
                    or any(not math.isfinite(stats[key]) for key in EDGE_METRICS)):
                raise ValueError(f"invalid Fusion mechanism statistics for {edge}")
            rows.append({**identity, "stage": stage, "epoch": epoch, "edge": edge,
                         "active_edges": ";".join(edges), "edge_parameter_count": EDGE_PARAMETER_COUNT,
                         "total_parameter_count": F0_PARAMETER_COUNT + len(edges) * EDGE_PARAMETER_COUNT,
                         "active_count": stats["active_count"], **{key: stats[key] for key in EDGE_METRICS}})
    return rows


def confusion_rows(matrix, *, normalized):
    rows = []
    for label, counts in zip(LABEL_VOCAB, matrix):
        total = sum(counts)
        values = [count / total if total else 0.0 for count in counts] if normalized else counts
        rows.append({"true_label": label, **dict(zip(LABEL_VOCAB, values))})
    return rows


def compare(summary):
    variants = {row["variant"]: row for row in summary}
    h4, h3, h2 = tuple(TOPOLOGY_EDGES)[1:]
    pairs = [(arm, "F0") for arm in (h4, h3, h2)] + [(h3, h4), (h2, h3)]
    rows = []
    for arm, baseline in pairs:
        deltas = {f"delta_{metric}_pp": round(100 * (variants[arm][metric] - variants[baseline][metric]), 10)
                  for metric in METRICS if metric != "loss"}
        wf1 = deltas["delta_weighted_f1_pp"]
        grade = "STRONG" if wf1 >= 0.50 else "PROMISING" if wf1 >= 0.20 else "NEGATIVE" if wf1 <= -0.20 else "NEUTRAL"
        drop = min(deltas["delta_macro_f1_pp"], deltas["delta_uar_pp"]) <= SECONDARY_DROP_PP
        rows.append({"variant": arm, "reference_variant": baseline, "seed": variants[arm]["seed"], "split": "dev",
                     **deltas, "wf1_grade_vs_f0": grade if baseline == "F0" else "",
                     "macro_f1_or_uar_drop": drop,
                     "favorable_screen_vs_f0": grade in ("STRONG", "PROMISING") and not drop
                     if baseline == "F0" else ""})
    return rows


def write_report(path, summary, comparisons, classes, *, synthetic):
    lines = ["# CT-MultiDAG Fusion 非对称拓扑重训练筛查", "",
             "Test = HOLD。下表只使用各臂最佳 Dev checkpoint。单 seed 结果用于筛查。"
             "固定 checkpoint 干预结果用于提出拓扑假设。", "",
             "| Variant | Best epoch | WF1 (%) | Macro-F1 (%) | UAR (%) | Accuracy (%) | ΔWF1 vs F0 (pp) | 等级 | Macro-F1 或 UAR 下滑 | 初筛达标 | 参数 |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- | ---: |"]
    for index, row in enumerate(summary):
        pair = comparisons[index - 1] if index else {}
        metrics = " | ".join(f"{100 * row[name]:.4f}" for name in METRICS if name != "loss")
        drop = ("是" if pair["macro_f1_or_uar_drop"] else "否") if pair else "—"
        favorable = ("是" if pair["favorable_screen_vs_f0"] else "否") if pair else "—"
        lines.append(f"| {row['variant']} | {row['best_epoch']} | {metrics} | "
                     f"{pair.get('delta_weighted_f1_pp', 0.0):+.4f} | {pair.get('wf1_grade_vs_f0', '—')} | "
                     f"{drop} | {favorable} | {row['total_parameter_count']:,} |")
    lines.extend(["", "## 各类 Dev F1 与 Recall", "",
                  "| Variant | 类别 | F1 (%) | Recall (%) | Support |", "| --- | --- | ---: | ---: | ---: |"])
    lines.extend(f"| {row['variant']} | {row['label']} | {100 * row['f1']:.4f} | "
                 f"{100 * row['recall']:.4f} | {row['support']} |" for row in classes)
    lines.extend(["", "## 判读", "",
                  "任一 Macro-F1 或 UAR 下降至少 0.20 pp 时，该臂即使达到 WF1 的 STRONG 或 PROMISING 阈值，也不算筛查成功。"
                  "WF1 差的绝对值小于 0.20 pp 视为接近。"])
    h4_f0, h3_f0, h2_f0, h3_h4, h2_h3 = (row["delta_weighted_f1_pp"] for row in comparisons)
    if synthetic:
        lines.append("本报告来自合成 smoke，研究判读留待正式 Dev 结果。")
    else:
        if h3_h4 > 0 and h3_f0 > 0:
            lines.append("H3 的 WF1 高于 H4 和 F0，支持进一步验证音频与文本双向交互及文本到视觉的三边结构。")
        if abs(h3_h4) < 0.20 and h3_f0 > -0.20:
            lines.append("H3 接近 H4 且接近或优于 F0，参数更少，优先考虑 H3。")
        if h2_h3 > 0:
            lines.append("H2 的 WF1 高于 H3，优先验证音频与文本双边交互。")
        if max(h4_f0, h3_f0, h2_f0) <= -0.20:
            lines.append("三种 Fusion 的 WF1 均落后 F0 至少 0.20 pp。该 Fusion 路线进入停止评估，保留 Direct concat + TrackCore。")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def pack_verified(directory, archive, inventory):
    archive.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=archive.parent, suffix=".tar.gz", delete=False) as stream:
        temporary = Path(stream.name)
    try:
        expected = {**inventory, MANIFEST_NAME: sha256_file(directory / MANIFEST_NAME)}
        with tarfile.open(temporary, mode="w:gz") as bundle:
            for name in expected:
                bundle.add(directory / name, arcname=name, recursive=False)
        with tarfile.open(temporary, mode="r:gz") as bundle:
            members = bundle.getmembers()
            if len(members) != len(expected) or {member.name for member in members} != set(expected):
                raise ValueError("analysis archive contents differ from the bundle manifest")
            for member in members:
                if not member.isfile():
                    raise ValueError(f"expected a regular file in the analysis archive: {member.name}")
                with bundle.extractfile(member) as stream:
                    if hashlib.sha256(stream.read()).hexdigest() != expected[member.name]:
                        raise ValueError(f"analysis archive SHA-256 mismatch: {member.name}")
        digest = sha256_file(temporary)
        os.link(temporary, archive)
    finally:
        temporary.unlink()
    return digest


def export_bundle(directory, archive, master, state):
    check_hold(state)
    directory.mkdir(parents=True, exist_ok=False)
    summary, checkpoints, classes, mechanisms = [], [], [], []
    population_ref = None
    files = ["run_summary.csv", "paired_comparison.csv", "per_class_metrics.csv", "fusion_mechanism_stats.csv",
             "label_vocab.json", "checkpoint_identity.json", "test_hold_status.json", "master.log", "experiment_report.md"]
    for job in state["training_jobs"]:
        row, checkpoint, history, predictions, population, evidence, stats = read_run(job, state)
        if population_ref is not None and population != population_ref:
            raise ValueError("Dev prediction population/order/labels differ across variants")
        population_ref = population
        summary.append(row)
        checkpoints.append(checkpoint)
        classes.extend({"variant": row["variant"], "seed": row["seed"], "split": "dev", **item}
                       for item in evidence["per_class"])
        mechanisms.extend(stats)
        suffix = f"{row['variant']}_seed{row['seed']}.csv"
        for prefix, rows in (
            ("epoch_history", history), ("dev_predictions", predictions),
            ("confusion_counts", confusion_rows(evidence["confusion_matrix"], normalized=False)),
            ("confusion_normalized", confusion_rows(evidence["confusion_matrix"], normalized=True)),
        ):
            name = f"{prefix}_{suffix}"
            write_csv(directory / name, rows)
            files.append(name)
    comparisons = compare(summary)
    for name, rows in (("run_summary", summary), ("paired_comparison", comparisons),
                       ("per_class_metrics", classes), ("fusion_mechanism_stats", mechanisms)):
        write_csv(directory / f"{name}.csv", rows)
    labels = list(LABEL_VOCAB)
    write_json_atomic(directory / "label_vocab.json", {"itos": labels, "stoi": dict(zip(labels, range(len(labels))))})
    write_json_atomic(directory / "checkpoint_identity.json", {"selector": make_topology_config("F0")["checkpoint"]["selector"],
                                                               "checkpoints": checkpoints})
    write_json_atomic(directory / "test_hold_status.json", {**HOLD, "test_labels_inspected": state["test_labels_inspected"],
                                                           "test_used_for_selection": state["test_used_for_selection"]})
    write_report(directory / "experiment_report.md", summary, comparisons, classes, synthetic=state["synthetic_only"])
    master.flush()
    shutil.copyfile(master.name, directory / "master.log")
    inventory = {name: sha256_file(directory / name) for name in sorted(files)}
    manifest_fields = ("protocol", "batch_id", "experiment_date", "experiment_group", "git_head", "device", "synthetic_only",
                       "data_identity", "common_edge_init_max_abs_diff", "shared_backbone_init_max_abs_diff",
                       "canonical_h4_edge_initial_state_sha256", "shared_initial_state_sha256", "training_cpu_rng_sha256")
    manifest = {"schema": "ct_multidag_fusion_topology_analysis_bundle_v1", "ANALYSIS_BUNDLE": "PASS", "PLOT_READY": "YES",
                **{key: state[key] for key in manifest_fields}, "git_dirty": state.get("git_dirty"),
                "score_units": "fraction", "delta_units": "percentage_points", "confusion_normalization": "true_label_row",
                "secondary_metric_drop_pp": SECONDARY_DROP_PP, "split": "dev", "seed": 100,
                "variants": list(TOPOLOGY_EDGES), "training": make_topology_config("F0")["training"],
                "population": state["f0_batch_pairing"]["population"], "files_sha256": inventory}
    write_json_atomic(directory / MANIFEST_NAME, manifest)
    try:
        digest = pack_verified(directory, archive, inventory)
    except BaseException:
        (directory / MANIFEST_NAME).unlink()
        raise
    return {"ANALYSIS_BUNDLE": "PASS", "PLOT_READY": "YES", "archive": archive.as_posix(), "archive_sha256": digest}

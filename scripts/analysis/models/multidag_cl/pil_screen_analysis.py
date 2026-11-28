"""Export the registered comparisons and render the plot-ready PIL tables."""

import argparse
from collections import defaultdict
import csv
from hashlib import sha256
from pathlib import Path

import numpy as np


MATCHED = (("C01-PREV", "C01-CURRENT"), ("C07-TRUE", "C07-NULL"))
CLASSES = ("hap", "sad", "neu", "ang", "exc", "fru")
METRICS = ("accuracy", "weighted_f1", "macro_f1", "uar")
GROUPS = ("SHIFT", "STABLE", "NO_HISTORY")
GAPS = ("1", "2", "3–4", "5+")
PAIRS = (("hap", "exc"), ("ang", "fru"), ("neu", "fru"))
BOOTSTRAP_SEED, BOOTSTRAP_COUNT = 10002, 10000


def verdict(delta_f1, delta_acc):
    if delta_f1 >= 0.50 and delta_acc >= 0:
        return "GO"
    if (0.20 <= delta_f1 < 0.50
            or (delta_f1 > 0 and delta_acc < 0)
            or (delta_f1 < 0 and delta_acc > 0)):
        return "REVIEW"
    return "STOP"


def write_table(path, rows):
    fields = tuple(dict.fromkeys(key for row in rows for key in row))
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    with path.open(encoding="utf-8", newline="") as stream:
        expected = [{key: "" if row.get(key) is None else str(row[key]) for key in fields} for row in rows]
        if list(csv.DictReader(stream)) != expected:
            raise ValueError(f"CSV round-trip mismatch: {path}")


def export_analysis(directory, results, expected, dev, state):
    from scripts.analysis.models.multidag_cl.original_dev_analysis import resampled_metrics
    from scripts.analysis.models.multidag_cl.pairwise_fastkill_metrics import transitions
    from scripts.runtime.multidag_cl_pil_screen import dialogue_history
    from scripts.runtime.multidag_cl_group_isolated.evaluation import summarize
    from scripts.runtime.multidag_cl_ta_v0.formal import write_json_atomic
    from scripts.runtime.multidag_cl_ta_v0.training import classification_report

    baseline = state["identity"]["baseline"]["arm"]
    if baseline not in results or len(expected) != 980:
        raise ValueError("analysis requires the shared baseline and all 980 Dev utterances")
    dialogues, inverse = np.unique([did for _, did, _ in expected], return_inverse=True)
    if len(dialogues) != 20:
        raise ValueError("analysis requires the fixed 20 Dev dialogues")
    draws = np.random.default_rng(BOOTSTRAP_SEED).integers(len(dialogues), size=(BOOTSTRAP_COUNT, len(dialogues)))
    weights = np.stack([(draws == i).sum(axis=1) for i in range(len(dialogues))], axis=1)
    metadata = [row for index in range(len(dev)) for row in dialogue_history(dev[index])]
    if [(row["utterance_id"], row["dialogue_id"], row["true_label"]) for row in metadata] != expected:
        raise ValueError("analysis speaker/position metadata differs from the frozen Dev order")

    observations, endpoints, tables = {}, {}, defaultdict(list)
    y = np.array([label for _, _, label in expected])
    for arm, result in results.items():
        history = result["history"]
        if [row["epoch"] for row in history] != list(range(1, 31)):
            raise ValueError(f"incomplete trajectory for {arm}")
        for row in history:
            epoch = row["epoch"]
            steps = row["steps"]
            count = sum(s["valid_targets"] for s in steps)
            objective = (sum(s["valid_targets"] * s["objective"] for s in steps) / count
                         if steps else row["historical_train_update_loss"])
            for population, key in (("Fit", "fit"), ("Dev", "dev")):
                tables["trajectory"].append({"arm": arm, "epoch": epoch, "population": population, **row[key],
                                              "is_best_epoch": epoch == result["best_epoch"]})
            c07 = arm.startswith("C07")
            aux = sum(s["valid_targets"] * s["aux_bce_per_valid"] for s in steps) / count if c07 else None
            tables["training_objective"].append({
                "arm": arm, "epoch": epoch, "actual_update_objective": objective,
                "update_hard_ce": sum(s["valid_targets"] * s["hard_ce"] for s in steps) / count if steps else objective,
                "aux_bce_per_valid": aux, "eligible": sum(s["eligible"] for s in steps) if c07 else None,
                "eligible_fraction": sum(s["eligible"] for s in steps) / count if c07 else None,
                "aux_positive": sum(s["aux_positive"] for s in steps) if c07 else None,
                "clip_count": sum(s["clipped"] for s in steps) if steps else None,
                "epoch_seconds": row.get("seconds"), "gpu_peak_bytes": row.get("gpu_peak_bytes"),
            })
            tables["optimizer_steps"].extend({"arm": arm, "epoch": epoch, **s} for s in steps)
        for endpoint in ("Best", "E30"):
            official = result[endpoint]
            summary, rows, reliability, *_ = summarize(official["predictions"], expected)
            if any(summary["metrics"][k] != official["metrics"][k] for k in (*METRICS, "valid_target_count")):
                raise ValueError(f"{arm} {endpoint} predictions do not reproduce classification")
            if history[(result["best_epoch"] if endpoint == "Best" else 30) - 1]["dev"] != official["metrics"]:
                raise ValueError(f"{arm} {endpoint} metrics disagree with the learning trajectory")
            scores = {"arm": arm, "endpoint": endpoint, **official["metrics"],
                      "epoch": result["best_epoch"] if endpoint == "Best" else 30,
                      **{k: summary["auxiliary"][k] for k in ("nll", "brier", "ece10")},
                      "correct": sum(not r["is_error"] for r in rows)}
            tables["overall_metrics"].append(scores)
            endpoints[arm, endpoint] = scores
            for row, turn, raw in zip(rows, metadata, official["predictions"]):
                previous = turn["previous"]
                gap = turn["position"] - previous[0] if previous else None
                group = "NO_HISTORY" if previous is None else "SHIFT" if previous[2] != turn["true_label"] else "STABLE"
                if row["shift_group"] != group or row["target_index"] != turn["position"]:
                    raise ValueError("speaker/turn identity differs from the original Dev analysis")
                probabilities = np.asarray(raw["probabilities"])
                stable = np.asarray([row[f"prob_{label}"] for label in CLASSES])
                if probabilities.shape != stable.shape or not np.allclose(probabilities, stable, atol=1e-7, rtol=1e-6):
                    raise ValueError("exported Dev probabilities disagree with the logits")
                logits = raw["logits"]
                label = turn["true_label"]
                row.update(speaker_id_int=turn["speaker"], predecessor_exists=previous is not None,
                           predecessor_id=previous[1] if previous else None, position=turn["position"],
                           gap=gap, gap_group="NO_HISTORY" if gap is None else str(gap) if gap <= 2 else GAPS[2 if gap <= 4 else 3],
                           true_class_margin=logits[label] - max(score for j, score in enumerate(logits) if label != j))
                row.update({f"runtime_prob_{name}": prob for name, prob in zip(CLASSES, probabilities)})
            observations[arm, endpoint] = rows
            for name, records in (("predictions", rows), ("per_class", summary["per_class"]), ("reliability", reliability)):
                tables[name].extend({"arm": arm, "endpoint": endpoint, **row} for row in records)
            matrix = summary["confusion_matrix"]
            for i, counts in enumerate(matrix):
                for j, count in enumerate(counts):
                    tables["confusion"].append({"arm": arm, "endpoint": endpoint, "true": CLASSES[i],
                                                "predicted": CLASSES[j], "count": count,
                                                "row_fraction": count / sum(counts) if sum(counts) else None})
            for left, right in PAIRS:
                for true, predicted in ((left, right), (right, left)):
                    counts = matrix[CLASSES.index(true)]
                    tables["directional_confusion"].append({"arm": arm, "endpoint": endpoint, "true": true,
                                                            "predicted": predicted, "support": sum(counts),
                                                            "count": counts[CLASSES.index(predicted)]})
        tables["resources"].append({
            "arm": arm, "train_parameters": result["parameter_count"],
            "deploy_parameters": result["parameter_count"] - (1501 if arm.startswith("C07") else 0),
            "estimated_added_inference_dense_macs_per_padded_position": 60000 if arm.startswith("C01") else 0,
            "estimated_added_inference_fp32_flops_per_padded_position": 120000 if arm.startswith("C01") else 0,
            "estimated_added_inference_selector": "O(B*T^2) batched; O(1) speaker-cache lookup online"
            if arm.startswith("C01") else "none",
            "measured_train_update_seconds": sum(s["update_seconds"] for row in history for s in row["steps"])
            if history[0]["steps"] else None,
            "measured_total_epoch_seconds": sum(r["seconds"] for r in history) if history[0]["steps"] else None,
            "measured_peak_allocated_gpu_bytes": max(row.get("gpu_peak_bytes") or 0 for row in history) or None,
        })
    comparisons = [(arm, baseline) for arm in results if arm != baseline]
    comparisons.extend((arm, match) for arm, match in MATCHED if arm in results and match in results)
    sampled = {arm: resampled_metrics(observations[arm, "Best"], inverse, weights) for arm in results}
    base = observations[baseline, "Best"]
    group_masks = {"ALL": np.ones(len(y), dtype=bool)}
    group_masks.update({g: np.array([r["shift_group"] == g for r in base]) for g in GROUPS})
    group_masks["HAS_HISTORY"] = ~group_masks["NO_HISTORY"]
    group_masks.update({f"gap_{g}": np.array([r["gap_group"] == g for r in base]) for g in GAPS})
    group_masks.update({name: y == i for i, name in enumerate(CLASSES)})
    for arm in results:
        rows = observations[arm, "Best"]
        guess = np.array([r["predicted_label_id"] for r in rows])
        margin, loss = (np.array([r[key] for r in rows]) for key in ("true_class_margin", "nll"))
        for group, selected in group_masks.items():
            n = int(selected.sum())
            tables["subgroups"].append({"arm": arm, "group": group, "support": n,
                **(classification_report(y[selected], guess[selected], loss=float(loss[selected].mean()))["metrics"] if n else {}),
                "mean_true_class_margin": float(margin[selected].mean()) if n else None,
                "interpretation": "descriptive only" if n < 30 else "descriptive"})
    for arm, ref in comparisons:
        for metric in ("accuracy", "weighted_f1"):
            delta = 100 * (endpoints[arm, "Best"][metric] - endpoints[ref, "Best"][metric])
            lower, upper = np.quantile(100 * (sampled[arm][metric] - sampled[ref][metric]), (0.025, 0.975))
            tables["paired_bootstrap"].append({"arm": arm, "control": ref, "metric": metric,
                                              "delta_pp": delta, "lower_95_pp": lower, "upper_95_pp": upper})
        after, before = (observations[x, "Best"] for x in (arm, ref))
        wrong, previous_wrong = (np.array([r["is_error"] for r in obs]) for obs in (after, before))
        gain = np.array([a["true_class_margin"] - b["true_class_margin"] for a, b in zip(after, before)])
        for group, selected in group_masks.items():
            tables["r_d_net"].append({"arm": arm, "control": ref, "group": group,
                **transitions(previous_wrong, wrong, selected),
                "mean_margin_change": float(gain[selected].mean()) if selected.any() else None})
    delta = {(r["arm"], r["control"], r["metric"]): r["delta_pp"] for r in tables["paired_bootstrap"]}
    for arm, match in MATCHED:
        if arm not in results or match not in results:
            reason = state["arms"].get(arm, {}).get("reason", "缺少有效配对臂")
            tables["verdicts"].append({"method": arm[:3], "classification": "INVALID", "mechanism": "INVALID",
                                       "reason": reason})
            continue
        f1, acc = (delta[arm, baseline, metric] for metric in ("weighted_f1", "accuracy"))
        matched_f1, matched_acc = (delta[arm, match, metric] for metric in ("weighted_f1", "accuracy"))
        matched_verdict = verdict(delta[match, baseline, "weighted_f1"], delta[match, baseline, "accuracy"])
        decision = verdict(f1, acc)
        # Use the registered 0.20 pp review floor to flag a small matched difference.
        mechanism = "ELIGIBLE_FOR_CONFIRMATION" if decision == "GO" and matched_f1 >= .20 and matched_acc >= 0 else "NOT_ESTABLISHED"
        tables["verdicts"].append({"method": arm[:3], "classification": decision, "mechanism": mechanism,
                                  "weighted_f1_pp_vs_f0": f1, "accuracy_pp_vs_f0": acc,
                                  "weighted_f1_pp_vs_matched": matched_f1, "accuracy_pp_vs_matched": matched_acc,
                                  "matched_arm": match, "matched_arm_classification": matched_verdict,
                                  "priority": decision == "GO" and f1 >= 1})
    destination = directory / "tables"
    destination.mkdir(exist_ok=False)
    for name, rows in tables.items():
        if rows:
            write_table(destination / f"{name}.csv", rows)
    write_json_atomic(directory / "analysis_protocol.json", {"base": baseline, "seed": BOOTSTRAP_SEED,
        "resamples": BOOTSTRAP_COUNT, "dialogues": dialogues.tolist(),
        "draws_sha256": sha256(draws.astype("<i8").tobytes()).hexdigest(), "gap_bins": list(GAPS),
        "subgroup_descriptive_flag_count_below": 30, "matched_small_difference_floor_pp": .20})
    report = ["# PIL_SCREEN_001 原协议 Dev 分类筛选", "", f"共享对照为 {baseline}。", "",
              "| 方法 | 性能裁决 | 机制裁决 | 匹配臂性能裁决 |",
              "|---|---|---|---|"]
    report += [f"| {r['method']} | {r['classification']} | {r['mechanism']} | {r.get('matched_arm_classification', '—')} |"
               for r in tables["verdicts"]]
    report += ["", "| 运行 | 端点 | Epoch | 正确 / 总数 | ACC % | W-F1 % | Macro-F1 % | UAR % | NLL | ECE10 |",
               "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in tables["overall_metrics"]:
        report.append(f"| {row['arm']} | {row['endpoint']} | {row['epoch']} | {row['correct']}/{row['valid_target_count']} | "
                      + " | ".join(f"{100 * row[k]:.6f}" for k in METRICS)
                      + f" | {row['nll']:.6f} | {row['ece10']:.6f} |")
    report += ["", "## Best 配对差", "", "| 候选 | 对照 | 指标 | 差值 pp | 95% 区间 pp |",
               "|---|---|---|---:|---:|"]
    for row in tables["paired_bootstrap"]:
        report.append(f"| {row['arm']} | {row['control']} | {row['metric']} | {row['delta_pp']:+.6f} | "
                      f"[{row['lower_95_pp']:+.6f}, {row['upper_95_pp']:+.6f}] |")
    report += ["", "Fit 每轮用固定轮末权重在完整 Train 人口计算。更新目标在 batch 更新时计算，C07 的辅助 BCE 另列。",
               "Dev 有 20 段对话，来自同一会话的两名演员，并已用于此前开发。配对区间以完整对话重采样，"
               "描述当前 Dev 的变动，外推到新演员、新会话和训练种子需要独立证据。",
               "匹配比较低于 0.20 pp 列为微小差值。GO 仅表示可申请固定方案确认。Test 保持 HOLD。"]
    for row in tables["verdicts"]:
        if row["classification"] == "INVALID":
            report.append(f"- {row['method']} 阻断记录见批次回执。{row['reason']}")
        if row.get("matched_arm_classification") == "GO":
            report.append(f"- {row['matched_arm']} 达到分类 GO 门槛，可保留为独立性能信号。")
    (directory / "RESULTS.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    source_update = ["# MERC 项目源增量 · PIL_SCREEN_001", "",
                     f"批次为 {state.get('batch_id', 'unknown')}。原协议为 ORIGINAL_CT_MULTIDAG_DEV，seed100，共享对照为 {baseline}。", "",
                     "| 候选 | 性能裁决 | 机制处置 |",
                     "|---|---|---|"]
    source_update.extend(f"| {r['method']} | {r['classification']} | {r['mechanism']} |" for r in tables["verdicts"])
    source_update += ["", "筛选使用固定已暴露 Dev 的 Best 对比。完整结果及配对差见 RESULTS.md。"
                       "下一阶段由用户审阅裁决后另行授权。Test 保持 HOLD。"]
    (directory / "PROJECT_SOURCE_UPDATE.md").write_text("\n".join(source_update) + "\n", encoding="utf-8")
    return tables["verdicts"]


def plot_tables(directory):
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    source = directory / "tables"
    output = directory / "figures"
    output.mkdir(exist_ok=True)
    index = []

    def read(name):
        with (source / f"{name}.csv").open(encoding="utf-8", newline="") as stream:
            return list(csv.DictReader(stream))

    def save(fig, name, label):
        fig.tight_layout()
        for suffix in ("png", "svg"):
            path = output / f"{name}.{suffix}"
            fig.savefig(path, dpi=180)
            if not path.stat().st_size:
                raise OSError(f"empty plot: {path}")
        plt.close(fig)
        index.append(f"- `{name}.png` / `{name}.svg` · {label}")

    curves, objective = read("trajectory"), read("training_objective")
    arms = tuple(dict.fromkeys(row["arm"] for row in curves))
    colors = dict(zip(arms, plt.get_cmap("tab10").colors))
    for name, key, label, scale in (("01_accuracy", "accuracy", "Accuracy (%)", 100),
                                   ("02_weighted_f1", "weighted_f1", "Weighted F1 (%)", 100),
                                   ("03_hard_label_loss", "loss", "Hard-label CE", 1),
                                   ("04_training_objective_loss", "actual_update_objective", "Training update objective", 1)):
        fig, ax = plt.subplots(figsize=(9, 4.8))
        for arm in arms:
            for population in (("Update",) if key == "actual_update_objective" else ("Fit", "Dev")):
                chosen = [r for r in (objective if population == "Update" else curves)
                          if r["arm"] == arm and r.get("population", "Update") == population]
                ax.plot([int(r["epoch"]) for r in chosen], [scale * float(r[key]) for r in chosen],
                        label=f"{arm} {population}", color=colors[arm], linestyle="--" if population == "Fit" else "-")
        ax.set(xlabel="Epoch", ylabel=label)
        ax.grid(alpha=.2)
        ax.legend(ncol=2, fontsize=8)
        save(fig, name, label)
    for name, table, level, metric, label in (
        ("05_overall", "overall_metrics", "arm", "weighted_f1", "Best and E30 weighted F1 (%)"),
        ("06_classwise", "per_class", "label", "f1", "Best classwise F1 (%)"),
        ("07_history_subgroups", "subgroups", "group", "weighted_f1", "Best weighted F1 by history group (%)"),
    ):
        rows = read(table)
        levels = arms if level == "arm" else CLASSES if level == "label" else GROUPS
        series = ("Best", "E30") if level == "arm" else arms
        fig, ax = plt.subplots(figsize=(9, 4.6))
        width = .8 / len(series)
        for i, key in enumerate(series):
            selected = {r[level]: float(r[metric]) * 100 for r in rows if r.get("endpoint", "Best") == (key if level == "arm" else "Best")
                        and (level == "arm" or r["arm"] == key) and r[metric]}
            ax.bar(np.arange(len(levels)) + (i - (len(series) - 1) / 2) * width,
                   [selected.get(x, np.nan) for x in levels], width, label=key)
        ax.set(xticks=np.arange(len(levels)), xticklabels=levels, ylabel=label)
        ax.legend(fontsize=8)
        save(fig, name, label)
    confusion = read("confusion")
    fig, axes = plt.subplots(1, len(arms), figsize=(3.15 * len(arms), 3.2), squeeze=False)
    for ax, arm in zip(axes[0], arms):
        values = {(r["true"], r["predicted"]): float(r["row_fraction"] or 0)
                  for r in confusion if r["arm"] == arm and r["endpoint"] == "Best"}
        matrix = np.array([[values[x, y] for y in CLASSES] for x in CLASSES])
        ax.imshow(matrix, cmap="Blues", vmin=0, vmax=1)
        for i, j in np.ndindex(matrix.shape):
            ax.text(j, i, f"{matrix[i,j]:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if matrix[i, j] > .55 else "black")
        ax.set(title=arm, xticks=range(6), yticks=range(6), xticklabels=CLASSES, yticklabels=CLASSES,
               xlabel="Prediction", ylabel="True")
    save(fig, "08_confusion", "Best confusion, normalized by true class")
    paired = [r for r in read("paired_bootstrap") if r["metric"] == "weighted_f1"]
    fig, ax = plt.subplots(figsize=(9, 4.6))
    for i, row in enumerate(paired):
        ax.hlines(i, float(row["lower_95_pp"]), float(row["upper_95_pp"]), color="tab:blue")
        ax.plot(float(row["delta_pp"]), i, "o", color="tab:blue")
    ax.set(yticks=range(len(paired)), yticklabels=[f"{r['arm']} − {r['control']}" for r in paired],
           xlabel="Best weighted F1 difference (pp)")
    ax.axvline(0, color="black", linewidth=.7)
    save(fig, "09_paired_bootstrap", "Best weighted F1 paired dialogue bootstrap")
    aux = [r for r in objective if r["aux_bce_per_valid"]]
    if aux:
        fig, ax = plt.subplots(figsize=(9, 4.6))
        for arm in dict.fromkeys(r["arm"] for r in aux):
            rows = [r for r in aux if r["arm"] == arm]
            ax.plot([int(r["epoch"]) for r in rows], [float(r["aux_bce_per_valid"]) for r in rows], label=arm)
        ax.set(xlabel="Epoch", ylabel="Auxiliary BCE / valid utterances")
        ax.legend()
        save(fig, "10_auxiliary_bce", "C07 auxiliary objective")
    (directory / "FIGURE_INDEX.md").write_text("# Figure index\n\n" + "\n".join(index) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    plot_tables(parser.parse_args().directory)

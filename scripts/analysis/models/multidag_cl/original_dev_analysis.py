"""Replay frozen checkpoints and export fixed-Dev comparison tables."""

from collections import defaultdict
from hashlib import sha256

import numpy as np
import torch
from sklearn.metrics import precision_recall_curve, roc_curve

from datasets.iemocap.tav_full_audit_dataset import collate
from models.multidag_cl.ta_v0 import LABEL_VOCAB
from scripts.analysis.models.multidag_cl.failure_audit_metrics import GROUPS, logit_probabilities
from scripts.analysis.models.multidag_cl.group_isolated_comparison import curve_row
from scripts.analysis.models.multidag_cl.pairwise_fastkill_metrics import (
    BOOTSTRAP_RESAMPLES, BOOTSTRAP_SEED, transitions,
)
from scripts.analysis.models.multidag_cl.post_ph.internal_protocol import write_rows
from scripts.runtime.multidag_cl_fusion.original_dev_alignment import (
    CLASSIFICATION, epoch_rows, read_state,
)
from scripts.runtime.multidag_cl_fusion.runtime import training_rng_identity
from scripts.runtime.multidag_cl_group_isolated.evaluation import summarize, validate_predictions
from scripts.runtime.multidag_cl_group_isolated.fixed_tail_average import equal_average, parameter_schema
from scripts.runtime.multidag_cl_group_isolated.protocol import ENDPOINT, SEED
from scripts.runtime.multidag_cl_ta_v0.formal import sha256_file, write_json_atomic


ENDPOINTS = ("CE Best", "CE E30", "LS Best", "LS E30", "CE AVG", "LS AVG", "CE U*", "LS U*")
PAIRS = (("LS Best", "CE Best"), ("LS E30", "CE E30"), ("CE AVG", "CE Best"),
         ("CE AVG", "CE E30"), ("LS AVG", "LS Best"), ("LS AVG", "LS E30"), ("LS U*", "CE U*"))
INTERACTION = "交互项"
METRICS = (*CLASSIFICATION, "nll", "brier", "ece10")
INTERPRETATION = ("原 Dev 仅 20 段对话、2 位演员、1 个 Session。配对区间以该 Dev 内的整段对话为抽样单位，"
                  "外推范围不包括新 Session 或训练随机性。")


class FitView:
    """Present Train examples for read-only inference with the Dev evaluator."""

    def __init__(self, train):
        self.train = train

    def __len__(self):
        return len(self.train)

    def __getitem__(self, index):
        return {**self.train[index], "split": "dev"}


def replay(runtime, run, record, entries, average_epochs):
    manifest, snapshots, experiment = (record[key] for key in ("run", "snapshots", "experiment"))
    best = manifest["best_epoch"]
    reference = parameter_schema(runtime.model)
    fit = torch.utils.data.DataLoader(FitView(runtime.dataset("train")), batch_size=16, collate_fn=collate,
                                     generator=torch.Generator().manual_seed(SEED), num_workers=0)
    if runtime.optimizer_steps or runtime.epoch:
        raise ValueError("read-only replay requires a fresh runtime")
    rng = training_rng_identity(runtime.device)
    tag = "CE" if experiment["arm"] == "D0" else "LS"
    dev_endpoints, sources, learning = {}, {}, []

    def load(name):
        return read_state(run, name, snapshots[name], experiment, runtime.model)

    def collect(endpoint, dev, names):
        key = f"{tag} {endpoint}"
        dev_endpoints[key] = dev["predictions"]
        sources[key] = {"sources": [snapshots[n] for n in names], "arm": experiment["arm"],
                        "run_id": run.name, "formal_metrics": dev["metrics"]}

    for row in epoch_rows(run):
        epoch = int(row["epoch"])
        name = f"epoch_{epoch:02d}"
        runtime.model.load_state_dict(load(name), strict=True)
        dev = runtime.evaluate(include_predictions=True)
        if any(dev["metrics"][key] != float(row[f"dev_{key}"]) for key in (*CLASSIFICATION, "loss")):
            raise ValueError(f"{tag} E{epoch} replay differs from formal Dev metrics")
        if epoch == best:
            if dev["predictions"] != record["best_evidence"]["predictions"]:
                raise ValueError(f"{tag} Best replay differs from selected checkpoint predictions")
            collect("Best", dev, [name])
        if epoch == ENDPOINT:
            collect("E30", dev, [name])
        for role, result in (("Dev", dev), ("Fit", runtime.evaluate(fit, include_predictions=True))):
            predictions = result["predictions"]
            validate_predictions(predictions, entries["dev" if role == "Dev" else "train"])
            values = curve_row(tag, epoch, predictions)
            confidence = logit_probabilities([r["logits"] for r in predictions]).max(axis=1)
            correct = np.asarray([r["true_label"] == r["predicted_label"] for r in predictions])
            high = confidence >= 0.90
            learning.append({**values, "population": role, "mean_confidence": float(confidence.mean()),
                             "mean_confidence_correct": float(confidence[correct].mean()) if correct.any() else None,
                             "mean_confidence_wrong": float(confidence[~correct].mean()) if (~correct).any() else None,
                             "high_confidence_coverage": float(high.mean()),
                             "high_confidence_error_share": float((high & ~correct).mean()),
                             "formal_hard_ce": result["metrics"]["loss"],
                             "train_update_loss": float(row["train_loss"]),
                             "max_preclip_gradient_norm": float(row["gradient_norm"]),
                             "optimizer_steps": int(row["optimizer_steps"])})
    names = [f"epoch_{e:02d}" for e in average_epochs]
    runtime.model.load_state_dict(equal_average((load(name) for name in names), reference), strict=True)
    collect("AVG", runtime.evaluate(include_predictions=True), names)
    if "u_star" in snapshots:
        runtime.model.load_state_dict(load("u_star"), strict=True)
        collect("U*", runtime.evaluate(include_predictions=True), ["u_star"])
    if (runtime.optimizer_steps or runtime.epoch or runtime.selector.best is not None
            or runtime.optimizer.state_dict()["state"] or rng != training_rng_identity(runtime.device)):
        raise ValueError("replay altered training state")
    return dev_endpoints, sources, learning


def bootstrap_draws(expected):
    dialogues, inverse = np.unique([did for _, did, _ in expected], return_inverse=True)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = rng.integers(len(dialogues), size=(BOOTSTRAP_RESAMPLES, len(dialogues)))
    weights = np.stack([(draws == i).sum(axis=1) for i in range(len(dialogues))], axis=1)
    return inverse, weights, draws


def resampled_metrics(rows, inverse, weights):
    y = np.array([r["true_label_id"] for r in rows])
    p = np.array([r["predicted_label_id"] for r in rows])
    c, d = len(LABEL_VOCAB), weights.shape[1]
    cells = inverse * c**2 + y * c + p
    counts = np.bincount(cells, minlength=d * c**2).reshape(d, -1)
    matrices = (weights @ counts).reshape(-1, c, c)
    true, predicted = matrices.sum(2), matrices.sum(1)
    tp = matrices.diagonal(axis1=1, axis2=2)
    total, denominator = true.sum(1), true + predicted
    f1 = np.divide(2 * tp, denominator, out=np.zeros_like(tp, dtype=float), where=denominator > 0)
    recall = np.divide(tp, true, out=np.zeros_like(tp, dtype=float), where=true > 0)
    scores = {"accuracy": tp.sum(1) / total, "weighted_f1": (f1 * true).sum(1) / total,
              "macro_f1": f1.mean(1), "uar": recall.mean(1)}
    for metric in ("nll", "brier"):
        sums = np.bincount(inverse, weights=[r[metric] for r in rows], minlength=d)
        scores[metric] = weights @ sums / total
    confidence = np.array([r["confidence"] for r in rows])
    bins = np.minimum((confidence * 10).astype(int), 9)
    gaps = np.bincount(inverse * 10 + bins, weights=(p == y) - confidence, minlength=d * 10)
    scores["ece10"] = np.abs(weights @ gaps.reshape(d, 10)).sum(1) / total
    return scores


def stage_tables(directory, endpoints, sources, learning, expected, budget):
    required = set(ENDPOINTS if budget["status"] == "LOCKED" else ENDPOINTS[:-2])
    if set(endpoints) != required or set(sources) != required:
        raise ValueError("endpoint inventory disagrees with the locked design")
    tables = defaultdict(list)
    inverse, weights, draws = bootstrap_draws(expected)
    point, resampled, errors, observations = {}, {}, {}, {}
    y = np.array([target for _, _, target in expected])
    for arm in ENDPOINTS:
        if arm not in endpoints:
            tables["overall_metrics"].append({"arm": arm, "status": "UNAVAILABLE"})
            continue
        summary, rows, bins, roc, risk = summarize(endpoints[arm], expected)
        # Official classification fields and zero-update evaluator population must agree exactly.
        official = sources[arm]["formal_metrics"]
        if any(summary["metrics"][key] != official[key] for key in (*CLASSIFICATION, "valid_target_count")):
            raise ValueError(f"{arm} exported classification differs from the F0 evaluator")
        auxiliary = summary["auxiliary"]
        point[arm] = {**{key: official[key] for key in CLASSIFICATION},
                      **{key: auxiliary[key] for key in ("nll", "brier", "ece10")}}
        resampled[arm] = resampled_metrics(rows, inverse, weights)
        errors[arm] = np.array([r["is_error"] for r in rows])
        observations[arm] = rows
        tables["overall_metrics"].append({
            "arm": arm, "status": "SCORED", **point[arm], "official_hard_ce": official["loss"],
            "valid_targets": len(rows), "correct": int((~errors[arm]).sum()),
            "mean_confidence": float(np.mean([r["confidence"] for r in rows])),
            "aurc": auxiliary["risk_coverage"]["aurc"],
            "tie_expected_aurc": auxiliary["risk_coverage"]["tie_expected_aurc"],
            "error_auroc": auxiliary["error_detection"]["auroc"], "error_auprc": auxiliary["error_detection"]["auprc"],
        })
        for table, data in (("predictions", rows), ("reliability", bins), ("roc_pr", roc), ("risk_coverage", risk),
                            ("per_class", summary["per_class"])):
            tables[table].extend({"arm": arm, **r} for r in data)
        for i, counts in enumerate(summary["confusion_matrix"]):
            for j, count in enumerate(counts):
                tables["confusion"].append({"arm": arm, "gold": LABEL_VOCAB[i], "prediction": LABEL_VOCAB[j],
                                            "count": count, "true_class_fraction": count / sum(counts) if sum(counts) else None})
        for group, group_metrics in summary["shift_groups"].items():
            tables["shift"].append({"arm": arm, "group": group, **group_metrics["metrics"],
                                    "brier": group_metrics.get("brier"), "ece10": group_metrics.get("ece10")})
        for i, label in enumerate(LABEL_VOCAB):
            positive = y == i
            if not positive.any() or positive.all():
                continue
            p = [r[f"prob_{label}"] for r in rows]
            fpr, tpr, _ = roc_curve(positive, p)
            precision, recall, _ = precision_recall_curve(positive, p)
            for kind, xs, ys in (("ROC", fpr, tpr), ("PR", recall, precision)):
                tables["class_roc_pr"].extend({"arm": arm, "label": label, "curve": kind, "x": x, "y": value}
                                              for x, value in zip(xs, ys))

    def contrast(label, coefficients):
        for metric in METRICS:
            scale = 100 if metric in CLASSIFICATION else 1
            draws_metric = sum(weight * resampled[a][metric] for a, weight in coefficients.items())
            low, high = np.quantile(draws_metric * scale, (0.025, 0.975))
            tables["paired_bootstrap"].append({
                "contrast": label, "metric": metric, "units": "pp" if scale == 100 else "raw",
                "delta": scale * sum(weight * point[a][metric] for a, weight in coefficients.items()),
                "lower_95": low, "upper_95": high,
            })

    for candidate, control in PAIRS:
        if candidate not in endpoints:
            continue
        label = f"{candidate} − {control}"
        contrast(label, {candidate: 1, control: -1})
        group_ids = np.array([r["shift_group"] for r in observations[control]])
        subgroup = {"ALL": np.ones(len(y), dtype=bool), **{g: group_ids == g for g in GROUPS},
                    **{label: y == i for i, label in enumerate(LABEL_VOCAB)}}
        for group, selected in subgroup.items():
            counts = transitions(errors[control], errors[candidate], selected)
            tables["r_d_net"].append({"contrast": label, "group": group, **counts})

    contrast(INTERACTION, {"LS AVG": 1, "LS E30": -1, "CE AVG": -1, "CE E30": 1})
    tables["factorial"] = [{"objective": a, "checkpoint": b, **point[f"{a} {b}"]}
                           for a in ("CE", "LS") for b in ("E30", "AVG")]
    tables["trajectory"] = learning
    by_trajectory = {(r["arm"], r["epoch"], r["population"]): r for r in learning}
    for (arm, epoch, role), fit in by_trajectory.items():
        if role == "Fit":
            dev = by_trajectory[(arm, epoch, "Dev")]
            tables["generalization_gap"].append({"arm": arm, "epoch": epoch,
                "fit_minus_dev_weighted_f1": fit["weighted_f1"] - dev["weighted_f1"],
                "dev_minus_fit_hard_nll": dev["hard_label_nll"] - fit["hard_label_nll"]})
    if budget["status"] == "LOCKED":
        tables["budget_selection"] = [*budget["scores"], *({"session": "MEAN", "updates": row["updates"],
            "hard_nll": row["hard_nll"]} for row in budget["means"])]
    for name, rows in tables.items():
        if rows:
            fields = tuple(dict.fromkeys(key for row in rows for key in row))
            write_rows(directory / f"{name}.csv", rows, fields)
    write_json_atomic(directory / "analysis_protocol.json", {
        "comparison_population": {"utterances": len(expected), "dialogues": weights.shape[1]},
        "bootstrap": {"seed": BOOTSTRAP_SEED, "resamples": BOOTSTRAP_RESAMPLES, "unit": "whole_dialogue",
                      "draws_sha256": sha256(draws.astype("<i8").tobytes()).hexdigest()},
        "scope": INTERPRETATION, "sources": sources, "budget": budget,
        "evidence_sha256": {p.name: sha256_file(p) for p in sorted(directory.glob("*.csv"))},
    })
    return tables["overall_metrics"], tables["paired_bootstrap"]


def write_results(path, scores, intervals, budget):
    sections = ["# CT-MultiDAG 原 Dev 对齐结果", "", INTERPRETATION, "",
                "| 端点 | 正确数 | Acc % | W-F1 % | Macro-F1 % | UAR % | NLL | Brier | ECE10 |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in scores:
        arm = row["arm"]
        if row["status"] == "UNAVAILABLE":
            sections.append(f"| {arm} | 缺少预算资产 | | | | | | | |")
            continue
        values = [f"{100 * row[k]:.4f}" for k in CLASSIFICATION]
        values.extend(f"{row[k]:.6f}" for k in ("nll", "brier", "ece10"))
        sections.append(f"| {arm} | {row['correct']}/{row['valid_targets']} | " + " | ".join(values) + " |")
    sections += ["", "| 配对对比 | W-F1 差（pp） | 95% 区间（pp） |",
                 "| --- | ---: | ---: |"]
    for row in intervals:
        if row["metric"] == "weighted_f1":
            sections.append(f"| {row['contrast']} | {row['delta']:+.4f} | "
                            f"[{row['lower_95']:+.4f}, {row['upper_95']:+.4f}] |")
    if budget["status"] == "UNAVAILABLE":
        sections += ["", f"D3 状态为 UNAVAILABLE。读取历史证据时记录 `{budget['reason']}`。"]
    sections += ["", "完整配对区间、逐类指标、混淆矩阵和逐话语预测见 `tables/`。",
                 "[PNG 与 SVG 图目录](figures/FIGURE_INDEX.md)"]
    path.write_text("\n".join(sections) + "\n", encoding="utf-8")

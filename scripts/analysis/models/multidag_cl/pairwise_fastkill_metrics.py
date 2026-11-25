"""Fixed Dev scoring and seed100 Fast-Kill decision rules."""

import numpy as np

from scripts.runtime.multidag_cl_ta_v0.training import classification_report

from .failure_audit_metrics import GROUPS, PAIRS, logit_probabilities
from .pairwise_preflight_inputs import CLASS_ORDER, H_GROUPS
from .pairwise_preflight_metrics import CLASS_PAIRS, METRICS, count, rate


RECOVERABLE_F = 52
BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_SEED = 7100
ECE_BINS = 10
STOP_LIMITS = {"weighted_f1_pp_min": 0.20, "net_min_exclusive": 0, "q4_net_min_exclusive": 0,
               "macro_f1_pp_min": -1.00, "uar_pp_min": -1.00, "pair_net_min_exclusive": -4}
GO_LIMITS = {"weighted_f1_pp_min": 0.50, "net_min": 4, "q4_net_min": 3, "macro_f1_pp_min": -0.30,
             "uar_pp_min": -0.30, "pair_net_min": -1, "directional_new_errors_max": 2}


def transitions(ctrl_wrong, primary_wrong, selected):
    recovered = count(selected & ctrl_wrong & ~primary_wrong)
    damaged = count(selected & ~ctrl_wrong & primary_wrong)
    return {"support": count(selected), "ctrl_errors": count(selected & ctrl_wrong),
            "primary_errors": count(selected & primary_wrong), "r": recovered,
            "d": damaged, "net": recovered - damaged}


def pair_confusions(labels, ctrl, primary):
    rows = []
    for left, right in CLASS_PAIRS:
        directional = []
        for true, false in ((left, right), (right, left)):
            gold = labels == true
            before = gold & (ctrl == false)
            after = gold & (primary == false)
            support = count(gold)
            baseline, updated = count(before), count(after)
            directional.append({"pair": f"{CLASS_ORDER[left]}/{CLASS_ORDER[right]}",
                                "true_class": CLASS_ORDER[true], "predicted_class": CLASS_ORDER[false],
                                "ctrl_count": baseline, "primary_count": updated, "difference": updated - baseline,
                                "true_class_denominator": support, "ctrl_directional_error_rate": rate(baseline, support),
                                "primary_directional_error_rate": rate(updated, support),
                                "direction_r": count(before & (primary == true)),
                                "direction_d": count(after & (ctrl == true))})
        pair_net = sum(row["direction_r"] - row["direction_d"] for row in directional)
        rows.extend({**row, "pair_net": pair_net} for row in directional)
    return rows


def decision(summary, pairs):
    focused_pairs = {"/".join(pair) for pair in PAIRS}
    focused = [row for row in pairs if row["pair"] in focused_pairs]
    if len(focused) != 2 * len(PAIRS):
        raise ValueError("incomplete prespecified directional confusion rows")
    wf1, macro, uar = (summary[f"delta_{metric}_pp"] for metric in ("weighted_f1", "macro_f1", "uar"))
    net, q4 = summary["net"], summary["q4_net"]
    min_pair_net = min(row["pair_net"] for row in focused)
    max_directional_increase = max(row["difference"] for row in focused)
    stop = {"weighted_f1": wf1 < STOP_LIMITS["weighted_f1_pp_min"],
            "net": net <= STOP_LIMITS["net_min_exclusive"],
            "q4_net": q4 <= STOP_LIMITS["q4_net_min_exclusive"],
            "macro_f1": macro < STOP_LIMITS["macro_f1_pp_min"],
            "uar": uar < STOP_LIMITS["uar_pp_min"],
            "pair_net": min_pair_net <= STOP_LIMITS["pair_net_min_exclusive"]}
    go = {"weighted_f1": wf1 >= GO_LIMITS["weighted_f1_pp_min"],
          "net": net >= GO_LIMITS["net_min"], "q4_net": q4 >= GO_LIMITS["q4_net_min"],
          "macro_f1": macro >= GO_LIMITS["macro_f1_pp_min"], "uar": uar >= GO_LIMITS["uar_pp_min"],
          "pair_net": min_pair_net >= GO_LIMITS["pair_net_min"],
          "directional_confusion": max_directional_increase <= GO_LIMITS["directional_new_errors_max"]}
    if any(stop.values()):
        status = "STOP"
    elif all(go.values()):
        status = "GO"
    else:
        status = "REVIEW_SIGNAL"
    return {"FASTKILL": status, "stop_conditions_triggered": stop, "go_conditions_met": go,
            "stop_limits": STOP_LIMITS, "go_limits": GO_LIMITS}


def calibration(logits, labels):
    z = logits.astype(np.float64, copy=False)
    p = logit_probabilities(z)
    idx = np.arange(len(labels))
    # Shift first to avoid underflowing the gold probability in NLL.
    shifted = z - z.max(axis=1, keepdims=True)
    nll = np.mean(np.log(np.exp(shifted).sum(axis=1)) - shifted[idx, labels])
    brier = np.mean(np.square(p - np.eye(len(CLASS_ORDER))[labels]).sum(axis=1))
    confidence = p.max(axis=1)
    correct = z.argmax(axis=1) == labels
    bin_index = np.minimum((confidence * ECE_BINS).astype(int), ECE_BINS - 1)
    ece = np.abs(np.bincount(bin_index, weights=correct - confidence, minlength=ECE_BINS)).sum() / len(labels)
    return {"nll": float(nll), "brier": float(brier), "ece": float(ece)}


def bootstrap(labels, ctrl, primary, dialogue_ids, *, strata=None):
    classes = len(CLASS_ORDER)
    unique, first, dialogue = np.unique(dialogue_ids, return_index=True, return_inverse=True)
    size = len(unique)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    if strata is None:
        draws = rng.integers(size, size=(BOOTSTRAP_RESAMPLES, size))
    else:
        strata = np.asarray(strata)
        if strata.shape != dialogue.shape or not np.array_equal(strata, strata[first][dialogue]):
            raise ValueError("each bootstrap dialogue must belong to exactly one stratum")
        dialogue_strata = strata[first]
        groups = [np.flatnonzero(dialogue_strata == group) for group in np.unique(dialogue_strata)]
        draws = np.concatenate([rng.choice(group, size=(BOOTSTRAP_RESAMPLES, len(group))) for group in groups], axis=1)
    scores = []
    for predictions in (ctrl, primary):
        flat = dialogue * classes**2 + labels * classes + predictions
        counts = np.bincount(flat, minlength=size * classes**2).reshape(size, classes, classes)
        matrices = counts[draws].sum(axis=1)
        true, predicted = matrices.sum(axis=2), matrices.sum(axis=1)
        tp = matrices.diagonal(axis1=1, axis2=2)
        denominator = true + predicted
        f1 = np.divide(2 * tp, denominator, out=np.zeros_like(tp, dtype=float), where=denominator != 0)
        total = true.sum(axis=1)
        scores.append(np.stack(((f1 * true).sum(axis=1) / total, tp.sum(axis=1) / total)))
    return [{"metric": name, "lower_95_pp": float(lower), "upper_95_pp": float(upper),
             "resamples": BOOTSTRAP_RESAMPLES, "seed": BOOTSTRAP_SEED, "unit": "dialogue", "dialogues": size}
            for name, (lower, upper) in zip(("weighted_f1", "accuracy"),
                                           np.quantile(100 * (scores[1] - scores[0]), [0.025, 0.975], axis=1).T)]


def subgroup_rows(names, groups, labels, ctrl_wrong, primary_wrong, *, by_class=False):
    rows = []
    for name in names:
        group = groups[name]
        if group is None:
            rows.append({"group": name, "status": "NOT_AVAILABLE", "true_class": "ALL",
                         **dict.fromkeys(("support", "ctrl_errors", "primary_errors", "r", "d", "net"))})
            continue
        rows.append({"group": name, "status": "AVAILABLE", "true_class": "ALL",
                     **transitions(ctrl_wrong, primary_wrong, group)})
        if by_class:
            rows.extend({"group": name, "status": "AVAILABLE", "true_class": label,
                         **transitions(ctrl_wrong, primary_wrong, group & (labels == i))}
                        for i, label in enumerate(CLASS_ORDER))
    return rows


def delta_distribution(selected, a, b, delta):
    rows = []
    for left, right in CLASS_PAIRS:
        pair = selected & (np.minimum(a, b) == left) & (np.maximum(a, b) == right)
        oriented = np.where(a[pair] == left, delta[pair], -delta[pair])
        rows.append({"pair": f"{CLASS_ORDER[left]}/{CLASS_ORDER[right]}", "gate_support": count(pair),
                     "mean_oriented_delta": float(oriented.mean()) if oriented.size else None,
                     "median_oriented_delta": float(np.median(oriented)) if oriented.size else None,
                     "positive_count": count(oriented > 0), "negative_count": count(oriented < 0),
                     "zero_count": count(oriented == 0)})
    return rows


def analyze(dev, result, q4, groups):
    labels = dev.labels
    final = result.output.logits.cpu().numpy()
    a, b, mad, margin, selected, delta, corrected, written = (
        value.cpu().numpy() for value in (result.top1, result.top2, result.mad, result.margin, result.selected,
                                         result.output.delta, result.output.corrected_margin, result.output.written)
    )
    primary = final.argmax(axis=1)
    ctrl_wrong, primary_wrong = a != labels, primary != labels
    full = transitions(ctrl_wrong, primary_wrong, np.ones(len(labels), dtype=bool))
    q4_recovery = transitions(ctrl_wrong, primary_wrong, q4)
    ctrl_report = classification_report(labels, a, loss=0)
    primary_report = classification_report(labels, primary, loss=0)
    metrics = []
    summary = {**full, **{f"q4_{key}": value for key, value in q4_recovery.items()},
               "realized_recovery_ratio": full["r"] / RECOVERABLE_F,
               "oracle_recoverable_f": RECOVERABLE_F, "gate_support": count(selected),
               "gate_coverage": float(selected.mean()), "writeback_count": count(written),
               "prediction_flips": count(a != primary),
               "ctrl_correct": count(~ctrl_wrong), "primary_correct": count(~primary_wrong)}
    for metric in METRICS:
        ctrl, updated = (report["metrics"][metric] for report in (ctrl_report, primary_report))
        delta_pp = 100 * (updated - ctrl)
        metrics.append({"metric": metric, "ctrl_pct": 100 * ctrl, "primary_pct": 100 * updated, "delta_pp": delta_pp})
        summary.update({f"ctrl_{metric}_pct": 100 * ctrl, f"primary_{metric}_pct": 100 * updated,
                        f"delta_{metric}_pp": delta_pp})
    classes = [{"true_class": ctrl["label"], "support": ctrl["support"],
                **{f"{name}_{key}": row[key] for name, row in (("ctrl", ctrl), ("primary", updated))
                   for key in ("precision", "recall", "f1")},
                **{f"delta_{key}_pp": 100 * (updated[key] - ctrl[key]) for key in ("precision", "recall", "f1")},
                **transitions(ctrl_wrong, primary_wrong, labels == ctrl["label_id"])}
               for ctrl, updated in zip(ctrl_report["per_class"], primary_report["per_class"])]
    samples = [{"utterance_id": uid, "dialogue_id": did, "gold": CLASS_ORDER[gold],
                "f0_top1": CLASS_ORDER[a[i]], "f0_top2": CLASS_ORDER[b[i]],
                "primary_prediction": CLASS_ORDER[primary[i]], "mad": mad[i], "margin": margin[i],
                "gate": bool(selected[i]), "delta": delta[i], "corrected_margin": corrected[i],
                "writeback": bool(written[i]), "q4": bool(q4[i]),
                "recovered": bool(ctrl_wrong[i] and not primary_wrong[i]),
                "damaged": bool(not ctrl_wrong[i] and primary_wrong[i])}
               for i, (uid, did, gold) in enumerate(dev.population)]
    ctrl_cal, primary_cal = calibration(dev.logits, labels), calibration(final, labels)
    return summary, {
        "DEV_METRICS.csv": metrics, "DEV_SAMPLE_TRANSITIONS.csv": samples,
        "Q4_RECOVERY.csv": [{"population": "FULL_DEV", **full}, {"population": "FROZEN_MAD_Q4", **q4_recovery}],
        "CLASS_METRICS.csv": classes, "PAIR_CONFUSIONS.csv": pair_confusions(labels, a, primary),
        "SHIFT_STABLE_RECOVERY.csv": subgroup_rows(GROUPS, groups, labels, ctrl_wrong, primary_wrong, by_class=True),
        "UH_SUBGROUP_RECOVERY.csv": subgroup_rows(H_GROUPS, groups, labels, ctrl_wrong, primary_wrong),
        "PAIR_DELTA_DISTRIBUTION.csv": delta_distribution(selected, a, b, delta),
        "CALIBRATION_METRICS.csv": [{"metric": key, "ctrl": value, "primary": primary_cal[key],
                                     "delta": primary_cal[key] - value} for key, value in ctrl_cal.items()],
        "BOOTSTRAP_CI.csv": bootstrap(labels, a, primary, [did for _, did, _ in dev.population]),
    }

"""Paired Dev reporting and the preregistered H1 decision."""

from collections import Counter

import numpy as np

from scripts.runtime.multidag_cl_ta_v0.training import classification_report

from .failure_audit_metrics import GROUPS
from .pairwise_fastkill_metrics import bootstrap, calibration, pair_confusions, subgroup_rows, transitions
from .pairwise_preflight_inputs import CLASS_ORDER, H_GROUPS
from .pairwise_preflight_metrics import METRICS


def decision(summary):
    wf1, macro, uar = (summary[f"delta_{metric}_pp"] for metric in ("weighted_f1", "macro_f1", "uar"))
    full, h_top25, shift = (summary[key] for key in ("net", "h_top25_net", "shift_net"))
    stop = {"weighted_f1": wf1 < 0.20, "full_net": full <= 0, "h_top25_net": h_top25 <= 0,
            "macro_f1": macro < -1.00, "uar": uar < -1.00}
    go = {"weighted_f1": wf1 >= 0.50, "full_net": full >= 4, "h_top25_net": h_top25 >= 3,
          "shift_net": shift >= 0, "macro_f1": macro >= -0.30, "uar": uar >= -0.30}
    status = "STOP" if any(stop.values()) else "GO" if all(go.values()) else "REVIEW_SIGNAL"
    return {"FASTKILL": status, "stop_conditions_triggered": stop, "go_conditions_met": go,
            "five_seed_eligible": status == "GO"}


def distribution(values):
    fields = ("min", "p10", "p25", "median", "p75", "p90", "max")
    if not len(values):
        return {"count": 0, "mean": None, "std": None, **dict.fromkeys(fields)}
    quantiles = np.quantile(values, [0, 0.10, 0.25, 0.50, 0.75, 0.90, 1])
    return {"count": len(values), "mean": float(values.mean()), "std": float(values.std()),
            **dict(zip(fields, quantiles.tolist()))}


class EdgeDistributions:
    def __init__(self):
        self.batches = []

    def __call__(self, same, factor, cosine, distance):
        arrays = tuple(value.cpu().numpy() for value in (same, factor, cosine, distance))
        same, factor, cosine, distance = arrays
        if (same.dtype != np.bool_ or any(value.shape != same.shape for value in arrays)
                or not all(np.isfinite(value).all() for value in arrays)
                or not ((factor >= 0.5) & (factor <= 1)).all()
                or not ((cosine >= -1) & (cosine <= 1)).all()
                or not ((distance >= 0) & (distance <= 1)).all()):
            raise ValueError("invalid observed H1 edge attenuation or similarity")
        self.batches.append(arrays)

    def tables(self, population):
        same, factor, cosine, distance = (np.concatenate(column) for column in zip(*self.batches))
        counts = Counter(did for _, did, _ in population)
        if len(same) != sum(size * (size - 1) // 2 for size in counts.values()):
            raise ValueError("attenuation observation does not cover each legal Dev edge exactly once")
        groups = {"overall": np.ones(len(same), dtype=bool), "same_speaker": same, "different_speaker": ~same}
        result = {}
        for name, measures in (("ATTENUATION_DISTRIBUTION.csv", {"attenuation": factor}),
                               ("SIMILARITY_DISTRIBUTION.csv", {"cosine": cosine, "distance": distance})):
            result[name] = [{"relation": relation, "measure": measure, **distribution(values[mask].astype(np.float64))}
                            for relation, mask in groups.items() for measure, values in measures.items()]
        return result


def analyze(ctrl, primary, groups):
    if (ctrl.population != primary.population or not np.isfinite(ctrl.logits).all()
            or not np.isfinite(primary.logits).all()):
        raise ValueError("Dev comparison requires identical populations and finite logits")
    labels = ctrl.labels
    before, after = ctrl.logits.argmax(axis=1), primary.logits.argmax(axis=1)
    wrong_before, wrong_after = before != labels, after != labels
    full = transitions(wrong_before, wrong_after, np.ones(len(labels), dtype=bool))
    h = transitions(wrong_before, wrong_after, groups["H_TOP25"])
    report_before = classification_report(labels, before, loss=0)
    report_after = classification_report(labels, after, loss=0)
    shift_rows = subgroup_rows(GROUPS, groups, labels, wrong_before, wrong_after)
    status = np.select((wrong_before & ~wrong_after, ~wrong_before & wrong_after,
                        wrong_before & wrong_after & (before != after), ~wrong_before & ~wrong_after),
                       ("recovered", "damaged", "wrong_to_another_wrong", "unchanged_correct"),
                       default="unchanged_wrong")
    summary = {**full, **{f"h_top25_{key}": value for key, value in h.items()},
               "shift_net": next(row["net"] for row in shift_rows if row["group"] == "SHIFT"),
               **{name: int((status == name).sum())
                  for name in ("wrong_to_another_wrong", "unchanged_correct", "unchanged_wrong")}}
    metrics = []
    for metric in METRICS:
        old, new = (report["metrics"][metric] for report in (report_before, report_after))
        delta = 100 * (new - old)
        metrics.append({"metric": metric, "ctrl_pct": 100 * old, "h1_pct": 100 * new, "delta_pp": delta})
        summary.update({f"ctrl_{metric}_pct": 100 * old, f"h1_{metric}_pct": 100 * new, f"delta_{metric}_pp": delta})
    classes = [{"arm": arm, **row} for arm, report in (("CTRL", report_before), ("H1", report_after))
               for row in report["per_class"]]
    confusion = [{"arm": arm, "true_class": label, **dict(zip(CLASS_ORDER, counts))}
                 for arm, report in (("CTRL", report_before), ("H1", report_after))
                 for label, counts in zip(CLASS_ORDER, report["confusion_matrix"])]
    samples = [{"utterance_id": uid, "dialogue_id": did, "gold": CLASS_ORDER[gold],
                "f0_prediction": CLASS_ORDER[before[index]], "h1_prediction": CLASS_ORDER[after[index]],
                "transition": status[index], "h_top25": bool(groups["H_TOP25"][index]),
                "shift_group": next(name for name in GROUPS if groups[name][index]),
                "uh_subgroup": next(name for name in H_GROUPS if groups[name][index])}
               for index, (uid, did, gold) in enumerate(ctrl.population)]
    old_cal, new_cal = calibration(ctrl.logits, labels), calibration(primary.logits, labels)
    return summary, {
        "DEV_METRICS.csv": metrics,
        "DEV_SAMPLE_TRANSITIONS.csv": samples,
        "H_TOP25_RECOVERY.csv": [{"group": "H_TOP25", **h}],
        "SHIFT_STABLE_RECOVERY.csv": shift_rows,
        "UH_SUBGROUP_RECOVERY.csv": subgroup_rows(H_GROUPS, groups, labels, wrong_before, wrong_after),
        "CLASS_METRICS.csv": classes,
        "CONFUSION_MATRIX.csv": confusion,
        "DIRECTED_CONFUSION_CHANGES.csv": pair_confusions(labels, before, after),
        "CALIBRATION_METRICS.csv": [{"metric": "ece_10" if name == "ece" else name, "ctrl": old, "h1": new_cal[name],
                                     "delta": new_cal[name] - old} for name, old in old_cal.items()],
        "BOOTSTRAP_CI.csv": bootstrap(labels, before, after, [did for _, did, _ in ctrl.population]),
    }

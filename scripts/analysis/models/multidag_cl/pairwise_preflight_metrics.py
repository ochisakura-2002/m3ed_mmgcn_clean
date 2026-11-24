"""Fixed seed100 gate and top-2 oracle for frozen logits."""

from dataclasses import dataclass
from itertools import combinations

import numpy as np

from models.multidag_cl.ta_v0 import LABEL_VOCAB
from scripts.runtime.multidag_cl_ta_v0.training import classification_report

from .failure_audit_metrics import details


MIN_RECOVERABLE = 8
MIN_WF1_GAIN_PP = 0.75
MARGIN_LIMIT = 1.0
METRICS = ("accuracy", "weighted_f1", "macro_f1", "uar")
CLASS_PAIRS = tuple(combinations(range(len(LABEL_VOCAB)), 2))


@dataclass(frozen=True)
class Scores:
    labels: np.ndarray
    top1: np.ndarray
    top2: np.ndarray
    mad: np.ndarray
    margin: np.ndarray
    true_rank: np.ndarray

    @property
    def wrong(self):
        return self.labels != self.top1


def scores(logits, labels):
    z = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(labels)
    if (labels.ndim != 1 or not labels.size or labels.dtype.kind not in "iu"
            or np.any((labels < 0) | (labels >= len(LABEL_VOCAB)))):
        raise ValueError("expected nonempty valid six-class targets")
    data = details(z, labels)
    ranked = np.argsort(-z, axis=1, kind="stable")
    top1, top2 = ranked[:, 0], ranked[:, 1]
    row = np.arange(len(labels))
    return Scores(labels, top1, top2, data["mad"], z[row, top1] - z[row, top2],
                  (ranked == labels[:, None]).argmax(axis=1) + 1)


def train_threshold(mad):
    """Use the 1-based ceil(0.75*N) order statistic without interpolation."""
    values = np.sort(mad)
    return float(values[(3 * len(values) + 3) // 4 - 1])


def gate(mad, margin, valid, threshold):
    return valid & (mad >= threshold) & (margin > 0) & (margin < MARGIN_LIMIT)


def oracle(scores, selected):
    recoverable = scores.wrong & selected & (scores.labels == scores.top2)
    return np.where(recoverable, scores.top2, scores.top1), recoverable


def decision(recoverable, delta_wf1_pp):
    passed = recoverable >= MIN_RECOVERABLE and delta_wf1_pp >= MIN_WF1_GAIN_PP
    return {"ZERO_TRAINING_PREFLIGHT": "PASS" if passed else "STOP",
            "PAIRWISE_HEAD_IMPLEMENTATION": "ELIGIBLE" if passed else "NOT_AUTHORIZED"}


def count(mask):
    return int(np.count_nonzero(mask))


def rate(numerator, denominator):
    return numerator / denominator if denominator else None


def rank_rows(population, data, selected):
    support = count(selected)
    return [{"population": population, "rank": rank, "count": n, "rate": rate(n, support)}
            for rank in range(1, len(LABEL_VOCAB) + 1)
            for n in (count(selected & (data.true_rank == rank)),)]


def composition(group, data, selected):
    support = count(selected)
    wrong = selected & data.wrong
    return {"group": group, "support": support, "f0_correct": count(selected & ~data.wrong),
            "f0_wrong": count(wrong), "f0_accuracy": rate(count(selected & ~data.wrong), support),
            "f0_error_rate": rate(count(wrong), support),
            **{f"{name}_{stat}": float(reducer(values[selected])) if support else None
               for name, values in (("mad", data.mad), ("margin", data.margin))
               for stat, reducer in (("mean", np.mean), ("median", np.median))},
            "true_top2_rate": rate(count(selected & (data.true_rank <= 2)), support),
            "wrong_true_rank2_rate": rate(count(wrong & (data.true_rank == 2)), count(wrong)),
            "wrong_true_rank3plus_rate": rate(count(wrong & (data.true_rank >= 3)), count(wrong))}


def train_support(data, selected):
    first, second = data.labels == data.top1, data.labels == data.top2

    def support(rows):
        active = rows & selected
        return {"support": count(rows), "gate_support": count(active),
                "supervised_support": count(active & (first | second)),
                "y_eq_a": count(active & first), "y_eq_b": count(active & second)}

    pairs, directions = [], []
    for a, b in CLASS_PAIRS:
        ab = (data.top1 == a) & (data.top2 == b)
        ba = (data.top1 == b) & (data.top2 == a)
        pair = f"{LABEL_VOCAB[a]}/{LABEL_VOCAB[b]}"
        pairs.append({"pair": pair, **support(ab | ba)})
        for left, right, rows in ((a, b, ab), (b, a, ba)):
            directions.append({"pair": pair, "top1": LABEL_VOCAB[left], "top2": LABEL_VOCAB[right], **support(rows)})
    classes = [{"true_label": label, **support(data.labels == index)} for index, label in enumerate(LABEL_VOCAB)]
    return pairs, directions, classes


def oracle_tables(data, selected, predicted, recovered):
    comparison = {name: classification_report(data.labels, pred, loss=0)
                  for name, pred in (("f0", data.top1), ("oracle", predicted))}
    metrics = {f"{name}_{key}": report["metrics"][key] for name, report in comparison.items() for key in METRICS}
    for key in METRICS:
        metrics[f"oracle_delta_{key}_pp"] = 100 * (metrics[f"oracle_{key}"] - metrics[f"f0_{key}"])
    classes = []
    for index, label in enumerate(LABEL_VOCAB):
        gold = data.labels == index
        support = count(gold)
        f0_correct = count(gold & ~data.wrong)
        oracle_correct = count(gold & (predicted == data.labels))
        classes.append({"true_label": label, "support": support, "f0_correct": f0_correct,
                        "f0_recall": rate(f0_correct, support), "oracle_correct": oracle_correct,
                        "oracle_recall": rate(oracle_correct, support), "recovered_count": count(gold & recovered)})
    pairs = []
    for a, b in CLASS_PAIRS:
        pair_gate = selected & (np.minimum(data.top1, data.top2) == a) & (np.maximum(data.top1, data.top2) == b)
        repaired = count(pair_gate & recovered)
        pairs.append({"pair": f"{LABEL_VOCAB[a]}/{LABEL_VOCAB[b]}", "gate_support": count(pair_gate),
                      "wrong_support": count(pair_gate & data.wrong), "true_rank2_recoverable": repaired,
                      "oracle_recovered": repaired})
    return metrics, classes, pairs


def split_summary(split, data, selected):
    row = {"split": split, "support": len(data.labels), "gate_support": count(selected),
           "gate_rate": rate(count(selected), len(data.labels))}
    for name, values in (("mad", data.mad), ("margin", data.margin)):
        row.update({f"{name}_mean": float(np.mean(values)), f"{name}_std": float(np.std(values))})
        row.update({f"{name}_{key}": float(value) for key, value in zip(
            ("min", "q25", "median", "q75", "max"), np.quantile(values, [0, .25, .5, .75, 1]))})
    return row


def analyze(train, dev, q4, subgroups):
    threshold = train_threshold(train.mad)
    all_train = np.ones(len(train.labels), dtype=bool)
    all_dev = np.ones(len(dev.labels), dtype=bool)
    train_gate = gate(train.mad, train.margin, all_train, threshold)
    dev_gate = gate(dev.mad, dev.margin, all_dev, threshold)
    supervised = train_gate & (train.true_rank <= 2)
    predicted, recovered = oracle(dev, dev_gate)
    metrics, per_class, per_pair = oracle_tables(dev, dev_gate, predicted, recovered)
    train_pairs, train_directions, train_classes = train_support(train, train_gate)
    population_ranks = (
        ("ALL_TRAIN", train, all_train), ("GATE_TRAIN", train, train_gate), ("SUPPORT_TRAIN", train, supervised),
        ("ALL_DEV", dev, all_dev), ("GATE_DEV", dev, dev_gate), ("GATE_F0_WRONG", dev, dev_gate & dev.wrong),
        ("MAD_Q4", dev, q4),
    )
    composition_rows = [composition("G", dev, dev_gate), composition("NOT_G", dev, ~dev_gate)]
    groups = []
    for name, mask in subgroups.items():
        groups.append({"group": name, "status": "AVAILABLE" if mask is not None else "NOT_AVAILABLE",
                       "support": count(mask) if mask is not None else None,
                       "gate_support": count(mask & dev_gate) if mask is not None else None,
                       "f0_errors": count(mask & dev.wrong) if mask is not None else None,
                       "f_count": count(mask & recovered) if mask is not None else None,
                       "oracle_recovery_rate": rate(count(mask & recovered), count(mask & dev.wrong))
                       if mask is not None else None})
    summary = {"train_mad_threshold": threshold, "n_train_total": len(train.labels), "n_g_train": count(train_gate),
               "gate_rate_train": float(train_gate.mean()), "n_s_train": count(supervised),
               "n_y_eq_a": count(train_gate & (train.true_rank == 1)),
               "n_y_eq_b": count(train_gate & (train.true_rank == 2)),
               "n_dev": len(dev.labels), "n_g_dev": count(dev_gate), "gate_rate_dev": float(dev_gate.mean()),
               "f0_dev_correct": count(~dev.wrong), "f0_dev_errors": count(dev.wrong),
               **{key: composition_rows[0][key] for key in ("true_top2_rate", "wrong_true_rank2_rate",
                                                            "wrong_true_rank3plus_rate")},
               "recoverable_f": count(recovered), "f_over_f0_errors": rate(count(recovered), count(dev.wrong)),
               "f_over_dev": rate(count(recovered), len(dev.labels)), "q4_support": count(q4),
               "q4_f0_errors": count(q4 & dev.wrong), "q4_gate_support": count(q4 & dev_gate),
               "q4_f_count": count(q4 & recovered), "q4_oracle_net_recovery": count(q4 & recovered), **metrics}
    return summary, {
        "TRUE_RANK_DISTRIBUTION.csv": [row for name, data, mask in population_ranks for row in rank_rows(name, data, mask)],
        "GATE_COMPOSITION.csv": composition_rows, "ORACLE_PER_CLASS.csv": per_class,
        "ORACLE_PAIR_ANALYSIS.csv": per_pair, "ORACLE_FAILURE_SUBGROUPS.csv": groups,
        "TRAIN_DEV_GATE_SHIFT.csv": [split_summary("train", train, train_gate), split_summary("dev", dev, dev_gate)],
        "TRAIN_PAIR_SUPPORT.csv": train_pairs, "TRAIN_PAIR_DIRECTION_SUPPORT.csv": train_directions,
        "TRAIN_CLASS_SUPPORT.csv": train_classes,
    }

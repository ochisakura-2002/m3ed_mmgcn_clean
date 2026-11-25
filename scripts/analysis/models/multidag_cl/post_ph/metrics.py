"""Metric definitions shared by the Dev-only post-PH audit."""

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve

from utils.metrics import compute_classification_metrics


COVERAGE_PCT = (100, 95, 90, 80, 70, 60, 50)
CONFIDENCE_THRESHOLDS = (0.70, 0.80, 0.90)


def entropy(probabilities):
    return -np.sum(probabilities * np.log(np.maximum(probabilities, np.finfo(float).tiny)), axis=-1)


def prediction_mad(probabilities):
    classes = probabilities.shape[-1]
    return np.clip(1 - np.abs(probabilities - 1 / classes).sum(axis=-1) / (2 * (1 - 1 / classes)), 0, 1)


def probability_details(probabilities, labels):
    classes = probabilities.shape[1]
    top = np.sort(probabilities, axis=1)[:, -2:]
    chosen = probabilities[np.arange(len(labels)), labels]
    return dict(confidence=top[:, 1], entropy=entropy(probabilities), mad=prediction_mad(probabilities),
                margin=top[:, 1] - top[:, 0], nll=-np.log(np.maximum(chosen, np.finfo(float).tiny)),
                brier=np.square(probabilities - np.eye(classes)[labels]).sum(axis=1))


def reliability(probabilities, outcome, bins, *, adaptive=False):
    n = len(probabilities)
    if not n:
        return None, []
    if adaptive:
        assignments = np.array_split(np.argsort(probabilities, kind="stable"), min(bins, n))
    else:
        bin_id = np.minimum(np.floor(probabilities * bins).astype(int), bins - 1)
        assignments = [np.flatnonzero(bin_id == index) for index in range(bins)]
    rows = []
    for index, indices in enumerate(assignments):
        support = len(indices)
        confidence = float(probabilities[indices].mean()) if support else None
        accuracy = float(outcome[indices].mean()) if support else None
        gap = abs(confidence - accuracy) if support else None
        rows.append(dict(
            bin=index + 1, support=support, mean_probability=confidence, observed_rate=accuracy,
            absolute_gap=gap, ece_contribution=support / n * (gap or 0),
            lower=float(probabilities[indices].min()) if adaptive else index / bins,
            upper=float(probabilities[indices].max()) if adaptive else (index + 1) / bins,
        ))
    return sum(row["ece_contribution"] for row in rows), rows


def classification(labels, predictions, classes):
    if not len(labels):
        return dict(accuracy=None, weighted_f1=None, macro_f1=None, uar=None, error_rate=None)
    metrics = compute_classification_metrics(labels, predictions, labels=list(range(classes)))
    accuracy = metrics.pop("acc")
    return dict(accuracy=accuracy, error_rate=1 - accuracy, **metrics)


def seed_stability(predictions, probabilities, labels):
    votes = np.eye(probabilities.shape[-1])[predictions].mean(axis=0)
    return dict(
        correct_count=(predictions == labels).sum(axis=0),
        prediction_agreement=votes.max(axis=1),
        majority_class=votes.argmax(axis=1),
        majority_tie=(votes == votes.max(axis=1, keepdims=True)).sum(axis=1) > 1,
        prediction_entropy=entropy(votes),
        confidence_variance=np.var(probabilities.max(axis=-1), axis=0),
        seed_disagreement=1 - votes.max(axis=1),
        seed_predictive_entropy=entropy(probabilities.mean(axis=0)),
    )


def score_ranking(score):
    """Ascending risk, with exact ties in frozen population order."""
    return np.argsort(score, kind="stable")


def high_risk_set(score, fraction=0.25):
    k = max(1, int(len(score) * fraction))
    mask = np.zeros(len(score), dtype=bool)
    mask[score_ranking(score)[-k:]] = True
    return mask


def error_detection(wrong, risk):
    positives = int(wrong.sum())
    if positives in (0, len(wrong)):
        return dict(auroc=None, auprc=None), []
    fpr, tpr, roc_thresholds = roc_curve(wrong, risk, drop_intermediate=False)
    precision, recall, pr_thresholds = precision_recall_curve(wrong, risk)
    curve = [dict(curve="ROC", fpr=fp, tpr=tp, precision=None, recall=None,
                  threshold=t if np.isfinite(t) else None) for fp, tp, t in zip(fpr, tpr, roc_thresholds)]
    curve.extend(dict(curve="PR", fpr=None, tpr=None, precision=p, recall=r, threshold=t)
                 for p, r, t in zip(precision, recall, [*pr_thresholds, None]))
    return dict(auroc=float(roc_auc_score(wrong, risk)), auprc=float(average_precision_score(wrong, risk))), curve


def selective_curve(labels, predictions, score, classes):
    order = score_ranking(score)
    sorted_score = score[order]
    wrong = (labels != predictions)[order]
    counts = np.arange(1, len(order) + 1)
    exact_errors = wrong.cumsum()
    risk = exact_errors / counts
    oracle = np.maximum(0, counts - (len(order) - exact_errors[-1])) / counts
    boundaries = np.r_[0, np.flatnonzero(np.diff(sorted_score)) + 1, len(order)]
    expected_errors = np.empty(len(order))
    tie_sizes = np.empty(len(order), dtype=int)
    for start, end in zip(boundaries, boundaries[1:]):
        preceding = exact_errors[start - 1] if start else 0
        expected_errors[start:end] = preceding + wrong[start:end].mean() * np.arange(1, end - start + 1)
        tie_sizes[start:end] = end - start
    tie_expected = expected_errors / counts
    raw = [dict(accepted=int(k), coverage=float(k / len(order)), risk=float(r), oracle_risk=float(q),
                tie_expected_risk=float(e), boundary_tie_support=int(t))
           for k, r, q, e, t in zip(counts, risk, oracle, tie_expected, tie_sizes)]
    fixed = []
    for pct in COVERAGE_PCT:
        k = max(1, len(order) * pct // 100)
        selected = order[:k]
        fixed.append(dict(target_coverage=pct / 100, accepted=k, population=len(order),
                          coverage=k / len(order), boundary_tie_support=int(tie_sizes[k - 1]),
                          tie_expected_error_rate=float(tie_expected[k - 1]),
                          **classification(labels[selected], predictions[selected], classes)))
    return dict(aurc=float(risk.mean()), excess_aurc=float((risk - oracle).mean()),
                tie_expected_aurc=float(tie_expected.mean()), tie_expected_excess_aurc=float((tie_expected - oracle).mean()),
                oracle_aurc=float(oracle.mean())), raw, fixed

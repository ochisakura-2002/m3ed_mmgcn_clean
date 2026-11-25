"""Run diagnostics A-I on the frozen CT-MultiDAG five-seed Dev archive."""

import argparse
from collections import Counter
import csv
from itertools import combinations
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from .dev_inputs import CLASSES, CONTROL, SEEDS, UH_GROUPS, digest, read_dev, read_enrichment
from .metrics import (
    CONFIDENCE_THRESHOLDS, error_detection, high_risk_set, probability_details, reliability,
    score_ranking, seed_stability, selective_curve,
)


def write_table(directory, filename, rows):
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with (directory / filename).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def average(values):
    return float(values.mean()) if len(values) else None


def calibration(probabilities, prediction, labels, seed):
    info = probability_details(probabilities, labels)
    correct = prediction == labels
    bins, metrics = [], dict(seed=seed, support=len(labels), nll=average(info["nll"]),
                            brier=average(info["brier"]), accuracy=average(correct),
                            mean_confidence=average(info["confidence"]))
    for name, count, adaptive in (("ece10", 10, False), ("ece15", 15, False), ("adaptive_ece15", 15, True)):
        metrics[name], rows = reliability(info["confidence"], correct, count, adaptive=adaptive)
        bins.extend(dict(seed=seed, metric=name, class_name=None, **row) for row in rows)
    per_class = []
    for i, name in enumerate(CLASSES):
        value, rows = reliability(probabilities[:, i], labels == i, 15)
        per_class.append(value)
        metrics[f"ece15_{name}"] = value
        bins.extend(dict(seed=seed, metric="classwise_ece15", class_name=name, **row) for row in rows)
    metrics["classwise_ece15"] = float(np.mean(per_class))
    return metrics, bins


def group_map(dev, enrichment):
    metadata = dev.metadata
    groups = {"population": {"ALL": np.ones(len(metadata), dtype=bool)},
              "emotion": {label: dev.labels == i for i, label in enumerate(CLASSES)}}
    for kind, key in (("shift", "shift_group"), ("speaker", "speaker_id"), ("session", "session"),
                      ("dialogue", "dialogue"), ("dialogue_type", "dialogue_type")):
        values = np.asarray(dev.dialogues if key == "dialogue" else [row[key] for row in metadata])
        groups[kind] = {value: values == value for value in sorted(set(values))}
    if enrichment is None:
        groups["uh"] = {name: None for name in ("U_TOP25", "H_TOP25", *UH_GROUPS)}
    else:
        names = np.array([row["UH_group"] for row in enrichment])
        groups["uh"] = {key: np.array([row[key] for row in enrichment]) for key in ("U_TOP25", "H_TOP25")}
        groups["uh"].update({name: names == name for name in UH_GROUPS})
    return groups


def high_confidence(wrong, confidence, mask, cutoff):
    if mask is None:
        return dict(support=None, error_count=None, error_rate=None)
    selected = mask & (confidence >= cutoff)
    return dict(support=int(selected.sum()), error_count=int(wrong[selected].sum()),
                error_rate=average(wrong[selected]))


def subgroup_report(groups, wrong, details):
    subgroups, high_errors = [], []
    for kind, masks in groups.items():
        for name, mask in masks.items():
            base = dict(group_type=kind, group=name, available=mask is not None)
            high = [dict(base, min_confidence=cutoff, **high_confidence(wrong, details["confidence"], mask, cutoff))
                    for cutoff in CONFIDENCE_THRESHOLDS]
            high_errors.extend(high)
            row = dict(base, support=None, accuracy=None, error_rate=None, ece10=None,
                       **{f"mean_{key}": None for key in details})
            if mask is not None:
                row.update(support=int(mask.sum()), accuracy=average(~wrong[mask]), error_rate=average(wrong[mask]),
                           ece10=reliability(details["confidence"][mask], ~wrong[mask], 10)[0],
                           **{f"mean_{key}": average(values[mask]) for key, values in details.items()})
            for cutoff, item in zip(CONFIDENCE_THRESHOLDS, high):
                row[f"high_confidence_error_rate_at_{cutoff:.2f}"] = item["error_rate"]
            subgroups.append(row)
    return subgroups, high_errors


def stability_report(groups, stability, details, wrong, history):
    counts = stability["correct_count"]
    summaries = []
    for kind, masks in groups.items():
        for name, mask in masks.items():
            if mask is None:
                continue
            for k in range(len(SEEDS) + 1):
                selected = mask & (counts == k)
                summaries.append(dict(
                    group_type=kind, group=name, correct_count=k, support=int(selected.sum()),
                    group_support=int(mask.sum()), group_fraction=average(counts[mask] == k),
                    seed100_error=average(wrong[selected]), mean_mad=average(details["mad"][selected]),
                    mean_vote_agreement=average(stability["prediction_agreement"][selected]),
                    mean_history_sensitivity=average(history[selected]) if history is not None else None,
                ))
    return summaries


def risk_redundancy(risks, wrong):
    masks = {name: high_risk_set(score) for name, score in risks.items()}
    boundary_ties = {name: int((score == score[masks[name]].min()).sum()) for name, score in risks.items()}
    correlation, overlap, quartiles, conditional = [], [], [], []
    for left, right in combinations(risks, 2):
        # Spearman with midranks so discrete votes keep their ties.
        x, y = risks[left], risks[right]
        rho = float(spearmanr(x, y).statistic) if np.ptp(x) > 0 and np.ptp(y) > 0 else None
        correlation.append(dict(left=left, right=right, support=len(wrong), spearman=rho))
        a, b = masks[left], masks[right]
        intersection, union = int((a & b).sum()), int((a | b).sum())
        overlap.append(dict(left=left, right=right, left_support=int(a.sum()), right_support=int(b.sum()),
                            left_boundary_tie_support=boundary_ties[left], right_boundary_tie_support=boundary_ties[right],
                            intersection=intersection, union=union, jaccard=intersection / union))
        for base, modifier in ((left, right), (right, left)):
            for within_top in (False, True):
                baseline = masks[base] == within_top
                with_modifier = baseline & masks[modifier]
                without_modifier = baseline & ~masks[modifier]
                with_error = average(wrong[with_modifier])
                without_error = average(wrong[without_modifier])
                conditional.append(dict(
                    conditioned_on=base, conditioned_on_top25=within_top, compared_score=modifier,
                    compared_top25_support=int(with_modifier.sum()), compared_other_support=int(without_modifier.sum()),
                    compared_top25_error_rate=with_error, compared_other_error_rate=without_error,
                    error_enrichment=(with_error - without_error) if None not in (with_error, without_error) else None,
                ))
    for name, score in risks.items():
        for quartile, indices in enumerate(np.array_split(score_ranking(score), 4), 1):
            quartiles.append(dict(score=name, quartile=quartile, support=len(indices),
                                  error_count=int(wrong[indices].sum()), error_rate=average(wrong[indices]),
                                  min_risk=float(score[indices].min()), max_risk=float(score[indices].max())))
    return masks, correlation, overlap, quartiles, conditional


def failure_archetypes(dev, wrong, details, stability, risks, masks, history, pred):
    shift = np.array([row["shift_group"] for row in dev.metadata])
    count = stability["correct_count"]
    high_mad = masks["mad"]
    low_mad = high_risk_set(-risks["mad"])
    cohorts = {
        "all_five_seeds_wrong": count == 0,
        "confidence_at_least_0.90": details["confidence"] >= .90,
        "high_mad_shift": high_mad & (shift == "SHIFT"),
        "high_mad_stable": high_mad & (shift == "STABLE"),
        "low_mad_all_five_seeds_wrong": low_mad & (count == 0),
        "seed_predictions_disagree": stability["seed_disagreement"] > 0,
    }
    if history is not None:
        cohorts["high_history_sensitivity_low_mad"] = masks["history_sensitivity"] & low_mad
    error_total = int(wrong.sum())
    report = [dict(
        kind="cohort", archetype=name, support=int(mask.sum()), error_count=int(wrong[mask].sum()),
        error_rate=average(wrong[mask]), share_of_seed100_errors=int(wrong[mask].sum()) / error_total if error_total else None,
        mean_confidence=average(details["confidence"][mask]), mean_mad=average(details["mad"][mask]),
        shift_fraction=average(shift[mask] == "SHIFT"), all_five_wrong_fraction=average(count[mask] == 0),
        mean_vote_agreement=average(stability["prediction_agreement"][mask]),
        mean_history_sensitivity=average(history[mask]) if history is not None else None,
    ) for name, mask in cohorts.items()]
    pairs = Counter(zip(dev.labels[wrong], pred[wrong]))
    for (gold, predicted), errors in pairs.most_common():
        gold_support = int((dev.labels == gold).sum())
        report.append(dict(kind="directed_confusion", archetype=f"{CLASSES[gold]}->{CLASSES[predicted]}",
                           support=gold_support, error_count=errors, error_rate=errors / gold_support,
                           share_of_seed100_errors=errors / error_total))
    return sorted(report, key=lambda row: (row["kind"], -row["error_count"], row["archetype"]))


def per_sample(dev, enrichment, details, stability, wrong, pred):
    rows = []
    for i, uid in enumerate(dev.utterances):
        seed_features = {name: values[i].item() for name, values in stability.items()}
        seed_features["majority_class"] = CLASSES[seed_features["majority_class"]]
        seed_features["correct_fraction"] = seed_features["correct_count"] / len(SEEDS)
        seed_features.update({f"prediction_seed{seed}": CLASSES[dev.predictions[j, i]] for j, seed in enumerate(SEEDS)})
        gold, predicted = CLASSES[dev.labels[i]], CLASSES[pred[i]]
        row = dict(utterance_id=uid, dialogue_id=dev.dialogues[i], gold=gold, F0_prediction=predicted,
                   correct=not bool(wrong[i]), class_pair=f"{gold}->{predicted}" if wrong[i] else None)
        rows.append(dict(row, **dev.metadata[i], **seed_features))
    missing = dict(U_TOP25=None, H_TOP25=None, UH_group=None, history_sensitivity=None, P_transition=None,
                   H1_transition=None)
    failure = [dict(row, **{name: values[i].item() for name, values in details.items()},
                    **(enrichment[i] if enrichment is not None else missing)) for i, row in enumerate(rows)]
    stability_rows = [dict(row, mad=details["mad"][i].item(), history_sensitivity=failure[i]["history_sensitivity"])
                      for i, row in enumerate(rows)]
    return stability_rows, failure


def run(predictions_dir, output_dir, *, round1=None, enrichment_path=None):
    dev = read_dev(predictions_dir, round1)
    enrichment = read_enrichment(enrichment_path, dev) if enrichment_path is not None else None
    index = SEEDS.index(CONTROL)
    pred = dev.predictions[index]
    wrong = pred != dev.labels
    details = probability_details(dev.probabilities[index], dev.labels)
    stability = seed_stability(dev.predictions, dev.probabilities, dev.labels)
    history = np.array([row["history_sensitivity"] for row in enrichment]) if enrichment is not None else None
    risks = dict(max_softmax_risk=1 - details["confidence"], entropy=details["entropy"], mad=details["mad"],
                 negative_margin=-details["margin"], seed_disagreement=stability["seed_disagreement"],
                 seed_predictive_entropy=stability["seed_predictive_entropy"])
    if history is not None:
        risks["history_sensitivity"] = history

    reliability_metrics, bins = [], []
    for i, seed in enumerate(SEEDS):
        row, raw = calibration(dev.probabilities[i], dev.predictions[i], dev.labels, seed)
        reliability_metrics.append(row)
        bins.extend(raw)
    detection, detection_curve, selective, selective_curve_rows, selective_summary = [], [], [], [], []
    for name, score in risks.items():
        meta = dict(score=name, target_seed=CONTROL)
        row, raw = error_detection(wrong, score)
        detection.append(dict(meta, support=len(wrong), errors=int(wrong.sum()), prevalence=average(wrong), **row))
        detection_curve.extend(dict(meta, **point) for point in raw)
        summary, raw, fixed = selective_curve(dev.labels, pred, score, len(CLASSES))
        selective.extend(dict(meta, **row) for row in fixed)
        selective_curve_rows.extend(dict(meta, **point) for point in raw)
        selective_summary.append(dict(meta, **summary))

    groups = group_map(dev, enrichment)
    subgroup, confidence = subgroup_report(groups, wrong, details)
    masks, correlation, overlap, quartiles, conditional = risk_redundancy(risks, wrong)
    seed_samples, failure = per_sample(dev, enrichment, details, stability, wrong, pred)
    tables = {
        "RELIABILITY_METRICS.csv": reliability_metrics, "RELIABILITY_BINS.csv": bins,
        "ERROR_DETECTION_METRICS.csv": detection, "ERROR_DETECTION_CURVES.csv": detection_curve,
        "RISK_COVERAGE_CURVES.csv": selective_curve_rows, "SELECTIVE_METRICS.csv": selective,
        "SELECTIVE_AURC.csv": selective_summary,
        "HIGH_CONFIDENCE_ERRORS.csv": confidence, "SUBGROUP_METRICS.csv": subgroup,
        "SEED_STABILITY_PER_SAMPLE.csv": seed_samples,
        "SEED_STABILITY_GROUPS.csv": stability_report(groups, stability, details, wrong, history),
        "RISK_SIGNAL_CORRELATION.csv": correlation, "RISK_SIGNAL_OVERLAP.csv": overlap,
        "RISK_SIGNAL_QUARTILES.csv": quartiles, "RISK_SIGNAL_CONDITIONAL_ENRICHMENT.csv": conditional,
        "FAILURE_SAMPLE_TABLE.csv": failure,
        "FAILURE_ARCHETYPE_SUMMARY.csv": failure_archetypes(dev, wrong, details, stability, risks, masks, history, pred),
    }
    for filename, kind in (("SPEAKER_METRICS.csv", "speaker"), ("SESSION_METRICS.csv", "session"),
                           ("DIALOGUE_METRICS.csv", "dialogue")):
        tables[filename] = [row for row in subgroup if row["group_type"] == kind]
    output_dir.mkdir(parents=True, exist_ok=False)
    for filename, rows in tables.items():
        write_table(output_dir, filename, rows)

    result = dict(
        schema="ct_multidag_post_ph_dev_audit_v1", split="dev", seeds=SEEDS, target_seed=CONTROL,
        dev_utterances=len(dev.utterances), dev_dialogues=len(set(dev.dialogues)), dev_sessions=sorted(groups["session"]),
        TEST_TOUCHED="NO", FORMAL_TRAIN_RUNS=0, source=dev.source,
        definitions=dict(
            brier="Sum squared probability errors over classes, then average over utterances",
            ece="Mean absolute confidence minus correctness within bins, weighted by bin support",
            adaptive_ece="15 equal-frequency bins in stable confidence order",
            classwise_ece="Unweighted mean of six one-vs-rest 15 fixed-width ECEs",
            auprc="Average precision, with an error as the positive class",
            selective="Ascending risk. Frozen population order breaks exact score ties for accepted sets. Coverage count is floor(target times population). The tie-expected estimates additionally average over random tie breaks.",
            aurc="Mean risk over all nonempty retained sets. E-AURC subtracts the oracle that retains correct predictions first.",
            f1_uar="Macro-F1 and UAR use the fixed six-class vocabulary, with zero for absent classes",
            top25="Highest-scoring fourth, ties in stable population order, independent of archived formal U/H membership",
            seed_disagreement="One minus the largest hard-vote share",
            seed_predictive_entropy="Entropy of mean softmax probabilities over the five seeds in nats",
            prediction_entropy="Entropy of hard-vote fractions in nats",
            shift_group="Gold-based transition from previous same-speaker gold label in the same dialogue",
            small_groups="Descriptive only. Analysts should inspect subgroup support before interpretation.",
        ),
        enrichment_sha256=digest(enrichment_path) if enrichment_path is not None else None,
        transition_rows={name: sum(row[name] is not None for row in failure) for name in ("P_transition", "H1_transition")},
        output_sha256={filename: digest(output_dir / filename) for filename in tables},
    )
    (output_dir / "DEV_AUDIT_MANIFEST.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions-dir", type=Path, required=True, help="Formal five-seed Dev analysis bundle")
    parser.add_argument("--round1", type=Path, help="Verified Round 1 audit for metadata cross-check")
    parser.add_argument("--enrichment", type=Path, help="Fully aligned Dev-only U/H enrichment CSV")
    parser.add_argument("--output-dir", type=Path, required=True, help="New directory for audit output")
    args = parser.parse_args(argv)
    result = run(args.predictions_dir, args.output_dir, round1=args.round1, enrichment_path=args.enrichment)
    print(f"DEV_AUDIT = COMPLETE ({result['dev_utterances']} utterances, {result['dev_dialogues']} dialogues)")
    print("TEST_TOUCHED = NO")


if __name__ == "__main__":
    main()

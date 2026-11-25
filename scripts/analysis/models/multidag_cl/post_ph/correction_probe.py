"""Diagnose top-two correction direction and Train–Dev shift for frozen seed100 F0."""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np
from scipy.stats import ks_2samp
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, balanced_accuracy_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from models.multidag_cl.ta_v0 import LABEL_VOCAB
from scripts.analysis.models.multidag_cl.failure_audit_metrics import details
from scripts.analysis.models.multidag_cl.pairwise_preflight_inputs import (
    F0_IDENTITY, FrozenSplit, forward_f0, load_projected_train_dev, read_f0, validate_population,
)
from scripts.analysis.models.multidag_cl.pairwise_preflight_metrics import scores
from scripts.runtime.multidag_cl_ta_v0.formal import sha256_file

from .internal_protocol import metadata, protocol_a, read_rows, write_rows


RISK_FEATURES = ("confidence", "normalized_entropy", "mad", "top2_margin")
VIEW_NAMES = ("A_RISK", "B_RISK_METADATA", "C_H0", "D_H4", "E_H0_H4")
METRIC_FIELDS = ("evaluation", "fold", "feature_view", "eligible", "flip_count", "flip_prevalence",
                 "auroc", "auprc", "balanced_accuracy", "accuracy", "tp", "fp", "eligible_net_corrections")


def prepare_frozen(split, partition):
    if partition not in ("train", "dev"):
        raise ValueError("only Train and Dev are allowed")
    if len({uid for uid, _, _ in split.population}) != len(split.population):
        raise ValueError("duplicated frozen sample")
    data = scores(split.logits, split.labels)
    per_sample = details(split.logits, split.labels)
    positions = defaultdict(int)
    rows, covariates = [], []
    for uid, did, gold in split.population:
        positions[did] += 1
        meta = metadata(uid, did)
        rows.append({"split": partition, "utterance_id": uid, "dialogue_id": did, "position": positions[did],
                     "gold": LABEL_VOCAB[gold], **meta})
        covariates.append((np.log1p(positions[did]), meta["session"], meta["speaker"], meta["dialogue_type"]))
    risk = np.column_stack([per_sample[key] for key in RISK_FEATURES])
    views = {"A_RISK": risk, "B_RISK_METADATA": np.column_stack((risk, np.asarray(covariates, dtype=object)))}
    if split.layer_states is not None:
        h0, h4 = split.layer_states
        if any(arr.shape != (len(rows), 300) or not np.isfinite(arr).all() for arr in (h0, h4)):
            raise ValueError("both h0 and h4 must be finite 300-wide states aligned to frozen samples")
        views.update(C_H0=h0, D_H4=h4, E_H0_H4=np.column_stack((h0, h4)))
    return rows, data, views, per_sample


def require_isolation(train_rows, dev_rows):
    if any(not rows or any(row["split"] != label for row in rows)
           for rows, label in ((train_rows, "train"), (dev_rows, "dev"))):
        raise ValueError("correction diagnostics require distinct Train and Dev inputs")
    for key in ("utterance_id", "dialogue_id"):
        if {row[key] for row in train_rows} & {row[key] for row in dev_rows}:
            raise ValueError(f"correction Train/Dev leakage on {key}")


def model_for(view):
    if view == "B_RISK_METADATA":
        transform = ColumnTransformer(
            [("numeric", StandardScaler(), list(range(5))),
             ("descriptors", OneHotEncoder(handle_unknown="ignore"), [5, 6, 7])])
    else:
        transform = StandardScaler()
    return make_pipeline(transform, LogisticRegression(C=1.0, class_weight="balanced", max_iter=3000))


def scoring(y, p):
    y = np.asarray(y, dtype=bool)
    p = np.asarray(p, dtype=float)
    if not len(y) or len(y) != len(p) or not np.isfinite(p).all():
        raise ValueError("invalid correction scoring arrays")
    pred = p >= 0.5
    n_pos = int(y.sum())
    tp = int(np.count_nonzero(pred & y))
    fp = int(np.count_nonzero(pred & ~y))
    two_classes = 0 < n_pos < len(y)
    return {
        "eligible": len(y), "flip_count": n_pos, "flip_prevalence": n_pos / len(y),
        "auroc": float(roc_auc_score(y, p)) if two_classes else None,
        "auprc": float(average_precision_score(y, p)) if two_classes else None,
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)) if two_classes else None,
        "accuracy": float(accuracy_score(y, pred)), "tp": tp, "fp": fp,
        "eligible_net_corrections": tp - fp,
    }


def fit_probes(train, dev, assignment, *, views=VIEW_NAMES):
    """Fit fixed Train-fold models, then fit once on Train and score each view once on Dev."""
    train_rows, train_scores, train_views, _ = train
    dev_rows, dev_scores, dev_views, _ = dev
    require_isolation(train_rows, dev_rows)
    if set(assignment) != {row["dialogue_id"] for row in train_rows}:
        raise ValueError("probe folds must cover the exact Train dialogue population")
    fold = np.array([assignment[row["dialogue_id"]] for row in train_rows])
    if len(np.unique(fold)) < 2:
        raise ValueError("grouped correction probes require at least two Train folds")
    eligible_train = train_scores.true_rank <= 2
    eligible_dev = dev_scores.true_rank <= 2
    train_target = train_scores.true_rank == 2
    dev_target = dev_scores.true_rank == 2
    output = []
    for name in views:
        train_matrix, dev_matrix = train_views[name], dev_views[name]
        oof = np.full(len(train_rows), np.nan)
        for name_fold in sorted(set(fold)):
            fit = eligible_train & (fold != name_fold)
            val = eligible_train & (fold == name_fold)
            if len(np.unique(train_target[fit])) != 2 or not np.any(val):
                raise ValueError("Train fold lacks correction directions or eligible validation utterances")
            probe = model_for(name).fit(train_matrix[fit], train_target[fit])
            p = probe.predict_proba(train_matrix[val])[:, 1]
            oof[val] = p
            output.append({"evaluation": "train_grouped_cv", "fold": name_fold, "feature_view": name,
                           **scoring(train_target[val], p)})
        output.append({"evaluation": "train_oof_pooled", "fold": "", "feature_view": name,
                       **scoring(train_target[eligible_train], oof[eligible_train])})
        final = model_for(name).fit(train_matrix[eligible_train], train_target[eligible_train])
        p = final.predict_proba(dev_matrix[eligible_dev])[:, 1]
        output.append({"evaluation": "dev_fixed_once", "fold": "", "feature_view": name,
                       **scoring(dev_target[eligible_dev], p)})
    return output


def rank_support(data, *, split_name):
    n = len(data.labels)
    result = []
    for name, mask in [("ALL", np.ones(n, dtype=bool)),
                       *((label, data.labels == index) for index, label in enumerate(LABEL_VOCAB))]:
        ranks = data.true_rank[mask]
        keep = int(np.count_nonzero(ranks == 1))
        flip = int(np.count_nonzero(ranks == 2))
        result.append({"split": split_name, "gold_group": name, "total": len(ranks), "keep_top1": keep,
                       "flip_top2": flip, "outside_top2": len(ranks) - keep - flip,
                       "eligible": keep + flip, "flip_prevalence_eligible": flip / (keep + flip) if keep + flip else None})
    return result


def archived_label_shift(protocol_audit, dev_support):
    audit = json.loads(protocol_audit.read_text(encoding="utf-8"))
    folds = audit["protocol_a"]
    train = Counter()
    for fold in folds:
        train.update(fold["validation"]["class_counts"])
    total_train = sum(train.values())
    total_dev = dev_support[0]["total"]
    if (audit["f0_identity"] != F0_IDENTITY or audit["TEST_TOUCHED"] != "NO"
            or audit["INTERNAL_DEV_PROTOCOL"] != "READY" or total_train != audit["train_tape"]["utterances"]
            or total_dev != audit["dev_counts"]["utterances"] or set(train) != set(LABEL_VOCAB)):
        raise ValueError("expected label counts from the authenticated internal protocol audit")
    return [
        {"gold": row["gold_group"], "train_count": train[row["gold_group"]], "dev_count": row["total"],
         "train_share": train[row["gold_group"]] / total_train, "dev_share": row["total"] / total_dev,
         "dev_minus_train_pp": 100 * (row["total"] / total_dev - train[row["gold_group"]] / total_train)}
        for row in dev_support[1:]
    ]


def shift_rows(train, dev):
    require_isolation(train[0], dev[0])
    result = []
    counts = [Counter(row["gold"] for row in item[0]) for item in (train, dev)]
    for label in LABEL_VOCAB:
        for name, item, count in zip(("train_in_sample", "dev"), (train, dev), counts):
            result.append({"feature": f"gold_{label}", "population": name, "n": len(item[0]),
                           "count": count[label], "mean": count[label] / len(item[0])})
    score_summary = {
        key: (train[3][key], dev[3][key]) for key in RISK_FEATURES
    }
    score_summary["confidence_risk"] = tuple(1 - item[3]["confidence"] for item in (train, dev))
    score_summary["negative_top2_probability_margin"] = tuple(-item[3]["top2_margin"] for item in (train, dev))
    for key, values in score_summary.items():
        distance = float(ks_2samp(*values).statistic)
        for name, sample in zip(("train_in_sample", "dev"), values):
            q10, median, q90 = np.quantile(sample, [0.1, 0.5, 0.9])
            result.append({"feature": key, "population": name, "n": len(sample), "mean": float(np.mean(sample)),
                           "q10": q10, "median": median, "q90": q90, "train_dev_ks_distance": distance})
    return result


def archive_support(directory):
    manifest = json.loads((directory / "analysis_bundle_manifest.json").read_text(encoding="utf-8"))
    filename = "dev_predictions_F0_seed100.csv"
    path = directory / filename
    if (manifest["test_policy"] != "HOLD" or manifest["split"] != "dev"
            or manifest["synthetic_only"] is not False or manifest["ANALYSIS_BUNDLE"] != "PASS"
            or manifest["git_head"] != F0_IDENTITY["git_head"]
            or sha256_file(path) != manifest["files_sha256"][filename]):
        raise ValueError("expected verified archived seed100 F0 Dev predictions")
    rows = list(read_rows(path))
    if len(rows) != 980 or any(row["split"] != "dev" or row["variant"] != "F0" or row["seed"] != "100"
                               or LABEL_VOCAB[int(row["true_label_id"])] != row["true_label"] for row in rows):
        raise ValueError("archived seed100 population does not match frozen F0")
    logits = np.array([[float(row[f"logit_{label}"]) for label in LABEL_VOCAB] for row in rows])
    population = tuple((row["utterance_id"], row["dialogue_id"], int(row["true_label_id"])) for row in rows)
    _, data, _, _ = prepare_frozen(FrozenSplit(logits, population), "dev")
    if (any(LABEL_VOCAB[pred] != row["pred_label"] or pred != int(row["pred_label_id"])
            for pred, row in zip(data.top1, rows)) or int((~data.wrong).sum()) != 693):
        raise ValueError("archived logits and initial predictions are inconsistent")
    return rank_support(data, split_name="archived_dev")


def store_support(directory, rows):
    write_rows(directory / "CORRECTION_DIRECTION_SUPPORT.csv", rows, tuple(rows[0]))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="mode", required=True)
    archive = commands.add_parser("archived-dev", help="compute top-two label support from verified Dev logits")
    archive.add_argument("--analysis-dir", type=Path, required=True)
    archive.add_argument("--train-protocol-audit", type=Path, help="include metadata-only Train versus Dev label shift")
    frozen = commands.add_parser("frozen", help="run authenticated frozen Train/Dev and fixed linear probes")
    frozen.add_argument("--frozen-runs", type=Path, required=True)
    frozen.add_argument("--train-dev-projection", type=Path, required=True)
    frozen.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    for command in (archive, frozen):
        command.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    output = args.output_dir
    if args.mode == "archived-dev":
        support = archive_support(args.analysis_dir)
        label_shift = archived_label_shift(args.train_protocol_audit, support) if args.train_protocol_audit else None
        output.mkdir(parents=True, exist_ok=True)
        store_support(output, support)
        if label_shift:
            write_rows(output / "TRAIN_DEV_LABEL_SHIFT.csv", label_shift, tuple(label_shift[0]))
        print("CORRECTION_DIRECTION_AUDIT = PARTIAL")
        return

    import torch

    if (output / "CORRECTION_DIRECTION_PROBE.csv").exists():
        raise FileExistsError("the fixed Dev probe results already exist in this directory")
    source = read_f0(args.frozen_runs)
    datasets, projection = load_projected_train_dev(args.train_dev_projection, source)
    population = validate_population(datasets, source)
    train_split, dev_split = forward_f0(source, datasets, torch.device(args.device), capture_states=True)
    train, dev = prepare_frozen(train_split, "train"), prepare_frozen(dev_split, "dev")
    assignment, fold_selection = protocol_a(train[0])
    probe_rows = fit_probes(train, dev, assignment)
    distribution = shift_rows(train, dev)
    support = [*rank_support(train[1], split_name="train_in_sample"), *rank_support(dev[1], split_name="dev")]
    output.mkdir(parents=True, exist_ok=True)
    store_support(output, support)
    write_rows(output / "CORRECTION_DIRECTION_PROBE.csv", probe_rows, METRIC_FIELDS)
    write_rows(output / "TRAIN_DEV_SHIFT.csv", distribution,
               ("feature", "population", "n", "count", "mean", "q10", "median", "q90", "train_dev_ks_distance"))
    states_file = output / "DEV_FROZEN_STATES.npz"
    np.savez_compressed(states_file,
                        utterance_id=[uid for uid, _, _ in dev_split.population],
                        dialogue_id=[did for _, did, _ in dev_split.population],
                        gold_label_id=dev_split.labels, h0=dev_split.layer_states[0], h4=dev_split.layer_states[1],
                        logits=dev_split.logits, checkpoint_sha256=source.identity["checkpoint_sha256"])
    evidence = {
        "CORRECTION_DIRECTION_AUDIT": "COMPLETE", "TRAIN_DEV_SHIFT_AUDIT": "COMPLETE",
        "TEST_TOUCHED": "NO", "TRAIN_PREDICTIONS": "IN_SAMPLE", "FORMAL_TRAIN_RUNS": 0,
        "DIAGNOSTIC_ONLY": True, "checkpoint": source.identity, "projection": projection,
        "population": population, "fold_selection": fold_selection, "dev_state_file_sha256": sha256_file(states_file),
        "probe": {"model": "LogisticRegression", "C": 1.0, "class_weight": "balanced", "threshold": 0.5,
                  "views_fixed": list(VIEW_NAMES), "dev_validation_per_view": 1},
    }
    (output / "CORRECTION_DIRECTION_MANIFEST.json").write_text(
        json.dumps(evidence, sort_keys=True, indent=2), encoding="utf-8")
    print("CORRECTION_DIRECTION_AUDIT = COMPLETE")
    print("TRAIN_DEV_SHIFT_AUDIT = COMPLETE")


if __name__ == "__main__":
    main()

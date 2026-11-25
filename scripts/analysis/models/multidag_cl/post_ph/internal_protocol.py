"""Build grouped internal development folds from authenticated F0 Train metadata."""

import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import re

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

from models.multidag_cl.ta_v0 import LABEL_VOCAB
from scripts.analysis.models.multidag_cl.pairwise_preflight_inputs import F0_IDENTITY, validate_hold
from scripts.runtime.multidag_cl_ta_v0.formal import json_identity, sha256_file


FOLD_COUNT = 5
FOLD_SEED = 20261124
FOLD_CANDIDATES = 32
MAX_BALANCING_SWAPS = 40
TRAIN_POPULATION = {"dialogues": 100, "utterances": 4830}
IDENTITY = re.compile(r"(Ses0[1-5])[FM]_(impro\d+[ab]?|script\d+_\d+[ab]?)")
FIELDS = ("split", "utterance_id", "dialogue_id", "position", "gold", "session",
          "speaker", "speaker_id", "dialogue_type", "validation_fold")


def read_rows(path, *, delimiter=","):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        yield from csv.DictReader(stream, delimiter=delimiter)


def write_rows(path, rows, fields):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def metadata(utterance, dialogue):
    match = IDENTITY.fullmatch(dialogue)
    turn = re.fullmatch(re.escape(dialogue) + r"_([FM])\d+", utterance)
    if match is None or turn is None:
        raise ValueError("invalid canonical utterance or dialogue ID")
    session = match[1]
    speaker = turn[1]
    return {
        "session": session, "speaker": speaker, "speaker_id": session + speaker,
        "dialogue_type": "improvised" if match[2].startswith("impro") else "scripted",
    }


def validate_train(rows):
    if not rows or any(row["split"] != "train" for row in rows):
        raise ValueError("internal development requires Train-only rows")
    if len({row["utterance_id"] for row in rows}) != len(rows):
        raise ValueError("duplicate Train utterance")
    if set(row["gold"] for row in rows) != set(LABEL_VOCAB):
        raise ValueError("invalid or incomplete gold vocabulary")
    groups = defaultdict(list)
    for row in rows:
        groups[row["dialogue_id"]].append(row)
    for turns in groups.values():
        if sorted(row["position"] for row in turns) != list(range(1, len(turns) + 1)):
            raise ValueError("incomplete or duplicate Train turn positions")
    return groups


def canonical_train(path, expected_sha):
    if sha256_file(path) != expected_sha:
        raise ValueError("canonical metadata does not match frozen F0")
    train, dev_uids, dev_dids = {}, set(), set()
    for row in read_rows(path):
        if row["split"] == "train":
            uid = row["utterance_id"]
            if uid in train:
                raise ValueError("duplicate canonical Train utterance")
            train[uid] = row
        elif row["split"] == "dev":
            dev_uids.add(row["utterance_id"])
            dev_dids.add(row["dialogue_id"])
    if sha256_file(path) != expected_sha:
        raise ValueError("canonical metadata changed during validation")
    if set(train) & dev_uids or {row["dialogue_id"] for row in train.values()} & dev_dids:
        raise ValueError("canonical Train/Dev utterance or dialogue overlap")
    return train, {"dialogues": len(dev_dids), "utterances": len(dev_uids)}


def authenticate_train(run, a0, model_manifest):
    f0 = json.loads((run / "manifests/run_manifest.json").read_text(encoding="utf-8"))
    evidence = json.loads((a0 / "a0_manifest.json").read_text(encoding="utf-8"))
    validate_hold(f0, legacy=True)
    validate_hold(evidence, legacy=True)
    archived_identity = {"seed": f0["seed"], "epoch": f0["best_epoch"], "git_head": f0["git_head"],
                         "checkpoint_sha256": f0["best_checkpoint_sha256"]}
    if (archived_identity != F0_IDENTITY or evidence["split"] != "train"
            or evidence["synthetic_only"] is not False or evidence["A0_DIAGNOSTIC"] != "PASS"
            or evidence["optimizer_steps_performed"] != 0
            or any(evidence["train_" + key] != count for key, count in TRAIN_POPULATION.items())):
        raise ValueError("expected historical F0 and its formal Train A0 evidence")
    train, dev_counts = canonical_train(model_manifest, f0["data_identity"]["sha256"]["model_manifest"])
    tape = f0["batch_pairing"]["population"]
    if any(tape["dev"][key] != count for key, count in dev_counts.items()):
        raise ValueError("formal Dev metadata counts differ from frozen F0")

    normalized = []
    for row in read_rows(a0 / "train_utterance_difficulty.csv"):
        normalized.append({**row, **{key: int(row[key]) for key in ("dialogue_length", "position", "speaker")},
                           **{key: float(row[key]) for key in
                              ("full_dialogue_dmf", "prefix_equal_shift", "prefix_wes", "weighted_shift")}})
    if json_identity(normalized) != evidence["train_scoring_input_sha256"]:
        raise ValueError("Train A0 table does not match its manifest")
    rows = []
    for row in normalized:
        uid, did = row["utterance_id"], row["dialogue_id"]
        canonical = train.get(uid)
        meta = metadata(uid, did)
        if (canonical is None or did != canonical["dialogue_id"] or meta["speaker"] != canonical["speaker"]
                or row["speaker"] != ("F", "M").index(meta["speaker"])
                or row["label"] != canonical["benchmark_label"]
                or LABEL_VOCAB[int(canonical["benchmark_label_id"])] != row["label"]):
            raise ValueError("A0 and canonical Train metadata disagree")
        rows.append({"split": "train", "utterance_id": uid, "dialogue_id": did,
                     "position": row["position"], "gold": row["label"], **meta})
    groups = validate_train(rows)
    if (set(train) != {row["utterance_id"] for row in rows}
            or {"dialogues": len(groups), "utterances": len(rows)} != TRAIN_POPULATION):
        raise ValueError("missing or unexpected Train population")
    curriculum = sorted(read_rows(run / "manifests/curriculum_bucket_manifest.tsv", delimiter="\t"),
                        key=lambda row: int(row["original_train_index"]))
    dialogues = [row["dialogue_id"] for row in curriculum]
    if ([int(row["original_train_index"]) for row in curriculum] != list(range(len(groups)))
            or len(dialogues) != len(groups) or set(dialogues) != set(groups)):
        raise ValueError("F0 Train dialogue order does not cover the authenticated dataset")
    tape_body = []
    for did in dialogues:
        ordered = [row["utterance_id"] for row in sorted(groups[did], key=lambda row: row["position"])]
        canonical_order = sorted(ordered, key=lambda uid: int(train[uid]["canonical_dialogue_order"]))
        if canonical_order != ordered:
            raise ValueError("A0 Train turn order differs from canonical metadata")
        tape_body.append((did, ordered))
    if tape["train"] != {**TRAIN_POPULATION, "sha256": json_identity(tape_body)}:
        raise ValueError("Train utterance membership/order differs from frozen F0 tape")
    return rows, {"f0_identity": archived_identity, "train_tape": tape["train"], "dev_counts": dev_counts,
                  "canonical_manifest_sha256": f0["data_identity"]["sha256"]["model_manifest"],
                  "a0_train_scoring_input_sha256": evidence["train_scoring_input_sha256"]}


def balance_key(counts, total):
    """Compare worst class-prevalence and validation-size deviations."""
    fold_sizes = counts.sum(axis=-1)
    n = total.sum()
    drift = np.max(np.abs(counts / fold_sizes[..., None] - total / n), axis=(-2, -1))
    size_drift = np.max(np.abs(fold_sizes / n - 1 / FOLD_COUNT), axis=-1)
    return np.maximum(drift, size_drift), drift, size_drift


def balance_dialogue_swaps(group_fold, group_counts, total):
    """Improve the Train-only balance objective by swapping whole dialogues."""
    counts = np.zeros((FOLD_COUNT, len(LABEL_VOCAB)), dtype=int)
    np.add.at(counts, group_fold, group_counts)
    left, right = np.triu_indices(len(group_fold), 1)
    delta = group_counts[right] - group_counts[left]
    choice = np.arange(len(left))
    for step in range(MAX_BALANCING_SWAPS):
        src, dst = group_fold[left], group_fold[right]
        alternatives = np.broadcast_to(counts, (len(left), *counts.shape)).copy()
        alternatives[choice, src] += delta
        alternatives[choice, dst] -= delta
        valid = (src != dst) & np.all((0 < alternatives) & (alternatives < total), axis=(1, 2))
        indices = np.flatnonzero(valid)
        if not len(indices):
            return counts, step
        keys = balance_key(alternatives[indices], total)
        best = int(np.lexsort(tuple(reversed(keys)))[0])
        if tuple(key[best] for key in keys) >= balance_key(counts, total):
            return counts, step
        a, b = left[indices[best]], right[indices[best]]
        group_fold[a], group_fold[b] = group_fold[b], group_fold[a]
        counts = alternatives[indices[best]]
    return counts, MAX_BALANCING_SWAPS


def protocol_a(rows):
    groups = validate_train(rows)
    ordered = sorted(rows, key=lambda row: (row["dialogue_id"], row["position"]))
    labels = np.array([LABEL_VOCAB.index(row["gold"]) for row in ordered])
    dialogue = np.array([row["dialogue_id"] for row in ordered])
    total = np.bincount(labels, minlength=len(LABEL_VOCAB))
    contenders = []
    for seed in range(FOLD_SEED, FOLD_SEED + FOLD_CANDIDATES):
        splitter = StratifiedGroupKFold(n_splits=FOLD_COUNT, shuffle=True, random_state=seed)
        validation = [valid for _, valid in splitter.split(labels, labels, dialogue)]
        counts = np.array([np.bincount(labels[valid], minlength=len(LABEL_VOCAB)) for valid in validation])
        if np.any(counts == 0) or np.any(counts == total):
            continue
        contenders.append((balance_key(counts, total), seed, validation))
    if not contenders:
        raise ValueError("no five-fold grouped assignment has all six labels on both sides")
    _, seed, validation = min(contenders, key=lambda item: item[:2])
    unique, inverse = np.unique(dialogue, return_inverse=True)
    group_counts = np.zeros((len(groups), len(LABEL_VOCAB)), dtype=int)
    np.add.at(group_counts, (inverse, labels), 1)
    group_fold = np.full(len(groups), -1, dtype=int)
    for fold, valid in enumerate(validation):
        group_fold[inverse[valid]] = fold
    if np.any(group_fold == -1):
        raise ValueError("internal development assignment is incomplete")
    counts, swaps = balance_dialogue_swaps(group_fold, group_counts, total)
    _, drift, size_drift = balance_key(counts, total)
    assignment = {str(did): f"A{fold + 1}" for did, fold in zip(unique, group_fold)}
    return assignment, {"random_state": seed, "whole_dialogue_swaps": swaps,
                        "max_class_prevalence_delta": float(drift), "max_validation_size_delta": float(size_drift)}


def protocol_b(rows):
    validate_train(rows)
    return {row["dialogue_id"]: row["session"] for row in rows}


def audit_folds(rows, assignment, *, session_holdout=False):
    groups = validate_train(rows)
    if set(assignment) != set(groups) or len(set(assignment.values())) < 2:
        raise ValueError("fold assignment must cover exact Train dialogue set")
    result = []
    for fold in sorted(set(assignment.values())):
        fit = [row for row in rows if assignment[row["dialogue_id"]] != fold]
        valid = [row for row in rows if assignment[row["dialogue_id"]] == fold]
        summary = {}
        for name, part in (("fit", fit), ("validation", valid)):
            summary[name] = {"utterances": len(part), "dialogues": len({row["dialogue_id"] for row in part}),
                             "class_counts": dict(Counter(row["gold"] for row in part)),
                             "sessions": sorted({row["session"] for row in part}),
                             "speaker_ids": sorted({row["speaker_id"] for row in part}),
                             "dialogue_type_counts": dict(Counter(row["dialogue_type"] for row in part))}
        if set(summary["fit"]["class_counts"]) != set(LABEL_VOCAB) or set(summary["validation"]["class_counts"]) != set(LABEL_VOCAB):
            raise ValueError("a fold side lacks a label")
        if session_holdout and (set(summary["fit"]["sessions"]) & set(summary["validation"]["sessions"])
                                or set(summary["fit"]["speaker_ids"]) & set(summary["validation"]["speaker_ids"])):
            raise ValueError("session holdout leaked across fold")
        result.append({"fold": fold, **summary})
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--f0-run", type=Path, required=True)
    parser.add_argument("--a0", type=Path, required=True)
    parser.add_argument("--model-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    rows, sources = authenticate_train(args.f0_run, args.a0, args.model_manifest)
    assignment_a, a_details = protocol_a(rows)
    assignment_b = protocol_b(rows)
    audit = {
        "INTERNAL_DEV_PROTOCOL": "READY", "TEST_TOUCHED": "NO", "formal_fold_training_runs": 0,
        **sources, "protocol_a_selection": a_details,
        "protocol_a": audit_folds(rows, assignment_a),
        "protocol_b": audit_folds(rows, assignment_b, session_holdout=True),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, assignment in (("A", assignment_a), ("B", assignment_b)):
        write_rows(args.output_dir / f"INTERNAL_DEV_PROTOCOL_{name}.csv",
                   ({**row, "validation_fold": assignment[row["dialogue_id"]]} for row in rows), FIELDS)
    (args.output_dir / "INTERNAL_DEV_PROTOCOL_AUDIT.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8")
    print("INTERNAL_DEV_PROTOCOL = READY")


if __name__ == "__main__":
    main()

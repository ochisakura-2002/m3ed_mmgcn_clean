"""Authenticated Dev-only inputs for the frozen five-seed F0 audit."""

import csv
from collections import Counter
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re

import numpy as np


SEEDS = (42, 100, 2024, 2026, 3407)
CONTROL = 100
CLASSES = ("hap", "sad", "neu", "ang", "exc", "fru")
DEV_SIZE = 980
DEV_DIALOGUES = 20
ARCHITECTURE = "3672bf3feeac2bee27f7c81963c5c77027cab9919b2605ed0ee443fe48091a59"
SOURCE_HEAD = "85590fbf97d0c5aad0c5547142a98fc20e342dc4"
CONTROL_CHECKPOINT = "997e7db663e3a3c2b8cb9ea34957e8fdbda909593ae7ad8360bc14ef1e3022ac"
UID = re.compile(r"((Ses0[1-5])[FM]_(impro\d+[ab]?|script\d+_\d+[ab]?))_([FM])\d+")
UH_GROUPS = ("U_ONLY", "H_ONLY", "U_INTERSECT_H", "NEITHER")
HOLD_COUNTERS = (
    "test_loader_built", "test_labels_inspected", "test_evaluation_count", "test_checkpoint_selection_count",
)


@dataclass(frozen=True)
class DevEvidence:
    utterances: tuple
    dialogues: tuple
    labels: np.ndarray
    predictions: np.ndarray
    probabilities: np.ndarray
    metadata: list
    source: dict


def read_json(path):
    with path.open(encoding="utf-8-sig") as handle:
        return json.load(handle)


def read_rows(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_digest(path, hashes):
    actual = digest(path)
    if hashes.get(path.name) != actual:
        raise ValueError(f"Archived digest mismatch for {path.name}")
    return actual


def hold(document):
    if document.get("test_policy") != "HOLD" or any(document.get(key) != 0 for key in HOLD_COUNTERS):
        raise ValueError("Dev evidence lacks zero-counter Test HOLD provenance")
    if document.get("test_used_for_selection", False) is not False:
        raise ValueError("Test participated in selection")


def index_rows(rows, required_uids=None):
    indexed = {}
    for row in rows:
        uid = row.get("utterance_id")
        if not uid or uid in indexed:
            raise ValueError(f"Missing or duplicate utterance ID {uid!r}")
        if row.get("split", "dev") != "dev":
            raise ValueError("Only Dev rows can be read by this audit")
        indexed[uid] = row
    if not indexed or (required_uids is not None and set(indexed) != set(required_uids)):
        raise ValueError("Missing or extra utterance IDs in paired evidence")
    return indexed


def read_seed(path, seed):
    rows = read_rows(path)
    index = index_rows(rows)
    if len(rows) != DEV_SIZE or len({row["dialogue_id"] for row in rows}) != DEV_DIALOGUES:
        raise ValueError(f"Incorrect Dev population in {path.name}")
    if any(row.get("split") != "dev" or row["variant"] != "F0" or int(row["seed"]) != seed for row in rows):
        raise ValueError(f"Incorrect split, variant or seed in {path.name}")
    identity = tuple((uid, row["dialogue_id"], row["true_label"]) for uid, row in index.items())
    labels = np.array([CLASSES.index(row["true_label"]) for row in rows])
    predictions = np.array([CLASSES.index(row["pred_label"]) for row in rows])
    if any(int(row["true_label_id"]) != y or int(row["pred_label_id"]) != pred
           for row, y, pred in zip(rows, labels, predictions)):
        raise ValueError(f"Label vocabulary mismatch in {path.name}")
    probabilities = np.array([[float(row[f"prob_{name}"]) for name in CLASSES] for row in rows])
    logits = np.array([[float(row[f"logit_{name}"]) for name in CLASSES] for row in rows])
    if (not np.isfinite(logits).all() or not np.isfinite(probabilities).all()
            or (probabilities < 0).any() or (probabilities > 1).any()
            or not np.allclose(probabilities.sum(axis=1), 1, rtol=0, atol=1e-6)):
        raise ValueError(f"Invalid predictive distribution in {path.name}")
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    softmax = np.exp(logits - logits.max(axis=1, keepdims=True))
    softmax /= softmax.sum(axis=1, keepdims=True)
    if (not np.allclose(probabilities, softmax, rtol=2e-6, atol=1e-8)
            or not np.array_equal(logits.argmax(axis=1), predictions)):
        raise ValueError(f"Exported probabilities or predictions disagree with logits in {path.name}")
    return identity, labels, predictions, probabilities


def sample_metadata(identity):
    previous = {}
    positions = Counter()
    metadata = []
    for uid, dialogue, label in identity:
        match = UID.fullmatch(uid)
        if match is None or match[1] != dialogue:
            raise ValueError(f"Invalid IEMOCAP utterance/dialogue identity {uid!r}")
        _, session, dialogue_type, speaker = match.groups()
        key = (dialogue, speaker)
        prior = previous.get(key)
        group = "NO_HISTORY" if prior is None else ("STABLE" if prior == label else "SHIFT")
        metadata.append(dict(
            speaker=speaker, speaker_id=f"{session}{speaker}", session=session,
            dialogue_type="improvised" if dialogue_type.startswith("impro") else "scripted",
            target_index=positions[dialogue], shift_group=group, previous_same_speaker_label=prior,
        ))
        previous[key] = label
        positions[dialogue] += 1
    return metadata


def verify_round1(directory, identity, metadata, predictions, probabilities):
    from .metrics import prediction_mad

    manifest_path = directory / "failure_audit_manifest.json"
    manifest = read_json(manifest_path)
    hold(manifest)
    if manifest.get("TEST_TOUCHED") != "NO":
        raise ValueError("Round 1 Test status is not NO")
    path = directory / "AMBIGUITY_AUDIT.csv"
    sha = verify_digest(path, manifest["outputs_sha256"])
    rows = read_rows(path)
    if {int(row["seed"]) for row in rows} != set(SEEDS):
        raise ValueError("Round 1 does not contain the frozen seeds")
    keys = [row[0] for row in identity]
    for seed_index, seed in enumerate(SEEDS):
        index = index_rows([row for row in rows if int(row["seed"]) == seed], keys)
        mads = prediction_mad(probabilities[seed_index])
        for i, (uid, dialogue, label) in enumerate(identity):
            row, meta = index[uid], metadata[i]
            predicted = CLASSES[predictions[seed_index, i]]
            if (row["dialogue_id"] != dialogue or row["true_label"] != label or row["predicted_label"] != predicted
                    or row["correct"] != str(predicted == label) or int(row["target_index"]) != meta["target_index"]
                    or row["shift_group"] != meta["shift_group"] or row["speaker"] != meta["speaker"]
                    or row["previous_same_speaker_label"] != (meta["previous_same_speaker_label"] or "")
                    or not np.isclose(float(row["mad"]), mads[i], rtol=0, atol=2e-6)):
                raise ValueError(f"Round 1 disagrees with frozen F0 or metadata at seed {seed}, ID {uid}")
    return {"manifest_sha256": digest(manifest_path), "ambiguity_sha256": sha, "git_head": manifest["git_head"]}


def read_dev(directory, round1=None):
    manifest_path = directory / "analysis_bundle_manifest.json"
    manifest = read_json(manifest_path)
    if (manifest.get("split") != "dev" or manifest.get("test_policy") != "HOLD"
            or manifest.get("git_head") != SOURCE_HEAD or manifest.get("synthetic_only") is not False):
        raise ValueError("Unexpected formal five-seed Dev archive provenance")
    hashes = manifest["files_sha256"]
    hold_path = directory / "test_hold_status.json"
    verify_digest(hold_path, hashes)
    hold(read_json(hold_path))
    vocab = directory / "label_vocab.json"
    verify_digest(vocab, hashes)
    if read_json(vocab) != {"itos": list(CLASSES), "stoi": dict(zip(CLASSES, range(len(CLASSES))))}:
        raise ValueError("Unexpected formal class vocabulary")
    checkpoint_path = directory / "checkpoint_identity.json"
    verify_digest(checkpoint_path, hashes)
    checkpoints = [row for row in read_json(checkpoint_path)["checkpoints"] if row["variant"] == "F0"]
    if len(checkpoints) != len(SEEDS) or {row["seed"] for row in checkpoints} != set(SEEDS):
        raise ValueError("Frozen F0 checkpoint identities are incomplete or duplicated")
    checkpoints = {row["seed"]: row for row in checkpoints}
    if (any(row["split"] != "dev" or row["architecture_sha256"] != ARCHITECTURE
            or row["git_head"] != SOURCE_HEAD or row["run_id"] != f"tav_track_fusion_final_f0_s{seed}"
            for seed, row in checkpoints.items()) or checkpoints[CONTROL]["sha256"] != CONTROL_CHECKPOINT
            or checkpoints[CONTROL]["best_epoch"] != 21
            or len({row["dev_population_sha256"] for row in checkpoints.values()}) != 1):
        raise ValueError("Frozen F0 architecture, Run ID, Dev identity or checkpoint provenance differs")

    identity = None
    predictions, probabilities, prediction_hashes = [], [], {}
    for seed in SEEDS:
        path = directory / f"dev_predictions_F0_seed{seed}.csv"
        prediction_hashes[path.name] = verify_digest(path, hashes)
        actual, labels, prediction, probability = read_seed(path, seed)
        if identity is not None and actual != identity:
            raise ValueError(f"Dev utterance, dialogue, gold or population order differs at seed {seed}")
        identity = actual
        predictions.append(prediction)
        probabilities.append(probability)
    predictions, probabilities = np.stack(predictions), np.stack(probabilities)
    metadata = sample_metadata(identity)
    source = dict(manifest_sha256=digest(manifest_path), git_head=SOURCE_HEAD, batch_id=manifest["batch_id"],
                  predictions_sha256=prediction_hashes, checkpoint_identity=checkpoints)
    if round1 is not None:
        source["round1"] = verify_round1(round1, identity, metadata, predictions, probabilities)
    utterances, dialogues, _ = zip(*identity)
    return DevEvidence(utterances, dialogues, labels, predictions, probabilities, metadata, source)


def _boolean(value):
    values = {"true": True, "false": False, "1": True, "0": False}
    if value.lower() not in values:
        raise ValueError(f"Invalid U/H membership {value!r}")
    return values[value.lower()]


def read_enrichment(path, dev):
    if not isinstance(path, Path) or path.name.lower().startswith("test_"):
        raise ValueError("An explicit Dev enrichment file is required")
    indexed = index_rows(read_rows(path), dev.utterances)
    rows = []
    for i, uid in enumerate(dev.utterances):
        row = indexed[uid]
        if (row["dialogue_id"] != dev.dialogues[i] or row["gold"] != CLASSES[dev.labels[i]]
                or row["F0_prediction"] != CLASSES[dev.predictions[SEEDS.index(CONTROL), i]]):
            raise ValueError(f"Enrichment population disagrees with F0 at {uid}")
        u, h = (_boolean(row[key]) for key in ("U_TOP25", "H_TOP25"))
        group = {(True, False): "U_ONLY", (False, True): "H_ONLY",
                 (True, True): "U_INTERSECT_H", (False, False): "NEITHER"}[u, h]
        sensitivity = float(row["history_sensitivity"])
        if not np.isfinite(sensitivity) or sensitivity < 0:
            raise ValueError(f"Invalid history sensitivity at {uid}")
        rows.append(dict(U_TOP25=u, H_TOP25=h, UH_group=group, history_sensitivity=sensitivity,
                         P_transition=row.get("P_transition") or None, H1_transition=row.get("H1_transition") or None))
    if any(sum(row[name] for row in rows) != len(rows) // 4 for name in ("U_TOP25", "H_TOP25")):
        raise ValueError("U/H top quartiles do not have the frozen coverage")
    for field in ("P_transition", "H1_transition"):
        support = sum(row[field] is not None for row in rows)
        if support not in (0, len(rows)):
            raise ValueError(f"Expected complete or absent coverage for {field}")
    return rows

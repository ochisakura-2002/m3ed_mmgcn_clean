"""Zero-update Train-only comparison of the three difficulty definitions."""

from collections import defaultdict
import csv
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

from datasets.iemocap.tav_full_audit_dataset import COUNTS, IDENTITY_FIELDS, MANIFEST_FIELDS, SCHEMA, identity_sha
from models.multidag_cl.paper_reimplementation.curriculum import DialogueDifficultyScorer
from models.multidag_cl.ta_v0 import LABEL_VOCAB
from scripts.runtime.multidag_cl_ta_v0.formal import json_identity, sha256_file, write_json_atomic
from scripts.runtime.multidag_cl_tav_full_audit_v1.spec import write_csv
from .objectives import alpha_values, prefix_scores
from .spec import HOLD, UNIFORM_EPOCH

SCORES = ("full_dialogue_dmf", "prefix_equal_shift", "prefix_wes")


@dataclass(frozen=True)
class TrainDialogue:
    dialogue_id: str
    utterance_ids: tuple
    labels: tuple
    speakers: tuple


def from_dataset(dataset):
    if dataset.split != "train":
        raise ValueError("A0 and A1 require a Train dataset")
    for index in range(len(dataset)):
        item = dataset[index]
        if item["split"] != "train":
            raise ValueError("A0 and A1 require Train items")
        yield TrainDialogue(item["dialogue_id"], tuple(item["utterance_ids"]),
                            tuple(item["labels"].tolist()), tuple(item["speaker_ids_int"].tolist()))


def from_manifest(manifest_path, provenance_path):
    provenance = json.loads(Path(provenance_path).read_text(encoding="utf-8"))
    manifest_sha = sha256_file(manifest_path)
    if (provenance.get("schema") != SCHEMA or provenance.get("status") != "complete"
            or provenance.get("model_manifest_sha256") != manifest_sha
            or provenance.get("split_counts") != COUNTS or provenance.get("test_labels_inspected") != 0):
        raise ValueError("A0 requires a certified canonical label manifest")
    rows, by_dialogue = [], defaultdict(list)
    with Path(manifest_path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not set(MANIFEST_FIELDS) <= set(reader.fieldnames or ()):
            raise ValueError("missing TAV manifest fields")
        for row in reader:
            rows.append({key: row[key] for key in IDENTITY_FIELDS})
            if row["split"] == "train":
                by_dialogue[row["dialogue_id"]].append(row)
    if (identity_sha(rows) != provenance.get("row_order_identity_sha256")
            or sha256_file(manifest_path) != manifest_sha):
        raise ValueError("canonical manifest identity mismatch")
    if {"dialogues": len(by_dialogue), "utterances": sum(map(len, by_dialogue.values()))} != COUNTS["train"]:
        raise ValueError("A0 Train population differs from frozen F0")
    for dialogue_id, dialogue in by_dialogue.items():
        ordered = sorted(dialogue, key=lambda row: int(row["canonical_dialogue_order"]))
        labels = tuple(int(row["benchmark_label_id"]) for row in ordered)
        if any(not 0 <= label < len(LABEL_VOCAB) or row["benchmark_label"] != LABEL_VOCAB[label]
               for label, row in zip(labels, ordered)):
            raise ValueError("A0 Train label vocabulary mismatch")
        yield TrainDialogue(dialogue_id, tuple(row["utterance_id"] for row in ordered), labels,
                            tuple({"F": 0, "M": 1}[row["speaker"]] for row in ordered))


def require_suffix_invariance(dialogue, original):
    labels, speakers = dialogue.labels, dialogue.speakers
    for end in range(1, len(labels)):
        mutated = labels[:end] + tuple((label + 1) % len(LABEL_VOCAB) for label in labels[end:])
        changed = prefix_scores(mutated, speakers)[:end]
        if changed != original[:end]:
            raise ValueError("PREFIX_SUFFIX_INVARIANCE = FAIL: prefix difficulty")
        for epoch in range(1, UNIFORM_EPOCH + 1):
            if alpha_values([entry[1] for entry in changed], epoch) != alpha_values(
                    [entry[1] for entry in original[:end]], epoch):
                raise ValueError("PREFIX_SUFFIX_INVARIANCE = FAIL: training weight")
    return len(labels) - 1


def correlation(x, y):
    return float(np.corrcoef(x, y)[0, 1]) if len(x) >= 2 and np.ptp(x) and np.ptp(y) else None


def summarize_scores(rows, unit):
    length = [row["dialogue_length"] for row in rows]
    for score in SCORES:
        values = [row[score] for row in rows]
        yield {"unit": unit, "score": score, "n": len(values), "mean": float(np.mean(values)),
               "min": min(values), "max": max(values), "median": float(np.median(values)),
               "q05": float(np.quantile(values, 0.05)), "q95": float(np.quantile(values, 0.95)),
               "length_pearson": correlation(values, length),
               "length_spearman": correlation(rankdata(values), rankdata(length))}


def exposure_rows(labels, alpha, epoch):
    labels, alpha = np.asarray(labels), np.asarray(alpha, dtype=np.float64)
    for index, label in enumerate(LABEL_VOCAB):
        weights = alpha[labels == index]
        yield {"epoch": epoch, "label": label, "target_count": len(weights),
               "effective_loss_weight": float(weights.sum()),
               "effective_exposure_vs_ctrl": float(weights.mean()) if len(weights) else None}


def analyze(dialogues, destination, *, synthetic=False):
    dialogues = sorted(dialogues, key=lambda dialogue: dialogue.dialogue_id)
    if not dialogues:
        raise ValueError("A0 requires nonempty Train dialogues")
    dialogue_rows, targets, weights, trials = [], [], {}, 0
    for dialogue in dialogues:
        did, uids, labels, speakers = (dialogue.dialogue_id, dialogue.utterance_ids,
                                        dialogue.labels, dialogue.speakers)
        if did in weights or len(uids) != len(labels):
            raise ValueError("invalid A0 dialogue identity")
        scores = prefix_scores(labels, speakers)
        trials += require_suffix_invariance(dialogue, scores)
        full = DialogueDifficultyScorer.score(labels, speakers)
        weights[did] = tuple(row[1] for row in scores)
        dialogue_rows.append({"dialogue_id": did, "dialogue_length": len(uids), "full_dialogue_dmf": full,
                              "prefix_equal_shift": scores[-1][0], "prefix_wes": scores[-1][1]})
        for t, (uid, label, speaker, (equal, wes, shift)) in enumerate(zip(uids, labels, speakers, scores), 1):
            targets.append({"dialogue_id": did, "dialogue_length": len(uids), "position": t, "utterance_id": uid,
                            "label": LABEL_VOCAB[label], "speaker": speaker, "full_dialogue_dmf": full,
                            "prefix_equal_shift": equal, "prefix_wes": wes, "weighted_shift": shift})
    if trials < 1 or len({row["utterance_id"] for row in targets}) != len(targets):
        raise ValueError("A0 requires suffix trials and unique Train utterances")
    labels = [LABEL_VOCAB.index(row["label"]) for row in targets]
    composition = []
    for score in SCORES:
        bounds = np.quantile([row[score] for row in targets], [0.2, 0.4, 0.6, 0.8])
        counts = defaultdict(int)
        for row in targets:
            band = 1 + int(np.searchsorted(bounds, row[score], side="left"))
            counts[band, row["label"], row["speaker"]] += 1
        for (band, label, speaker), count in sorted(counts.items()):
            composition.append({"score": score, "quintile": band, "label": label, "speaker": speaker, "count": count})
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    write_csv(destination / "train_dialogue_difficulty.csv", dialogue_rows)
    write_csv(destination / "train_utterance_difficulty.csv", targets)
    write_csv(destination / "difficulty_distribution.csv", [
        *summarize_scores(dialogue_rows, "dialogue_terminal"), *summarize_scores(targets, "utterance")])
    write_csv(destination / "class_speaker_by_score_quintile.csv", composition)
    write_csv(destination / "early_effective_class_exposure.csv", [
        {"score": score, **row} for score in SCORES for epoch in range(1, UNIFORM_EPOCH + 1)
        for row in exposure_rows(labels, alpha_values([item[score] for item in targets], epoch), epoch)])
    result = {**HOLD, "split": "train", "synthetic_only": synthetic, "optimizer_steps_performed": 0,
              "A0_DIAGNOSTIC": "PASS", "PREFIX_SUFFIX_INVARIANCE": "PASS", "suffix_cut_trials": trials,
              "OOF_DIFFICULTY_VALIDATION": "NOT_AVAILABLE", "train_dialogues": len(dialogues),
              "train_utterances": len(targets), "train_scoring_input_sha256": json_identity(targets)}
    write_json_atomic(destination / "a0_manifest.json", result)
    return result, weights

"""Canonical TAV metadata and Train/Dev views. Test targets remain opaque."""

import csv
import hashlib
import json
from pathlib import Path
import pickle

import numpy as np
import torch
from torch.utils.data import Dataset

from datasets.iemocap.ta_v0_dataset import UID, integer, require
from models.multidag_cl.ta_v0 import LABEL_VOCAB

SCHEMA = "iemocap_asym_tav_t1024_v1"
VERSION = "asymmetric_merc_v1"
COUNTS = {"train": {"utterances": 4830, "dialogues": 100},
          "dev": {"utterances": 980, "dialogues": 20},
          "test": {"utterances": 1623, "dialogues": 31}}
DIMS = {"text_features": 1024, "audio_features": 1582, "visual_features": 342}
IDENTITY_FIELDS = ("utterance_id", "dialogue_id", "split", "speaker", "canonical_dialogue_order",
                   "asset_dialogue_id", "asset_row_index")
MANIFEST_FIELDS = (*IDENTITY_FIELDS, "benchmark_label", "benchmark_label_id", "audio_sha256")


def require_split(split):
    require(split in ("train", "dev"), "Train/Dev only; Test factory forbidden")


def fingerprint(array):
    value = np.asarray(array)
    require(value.shape == (1582,) and np.isfinite(value).all(), "audio fingerprint requires finite A1582")
    # Historical JSON-derived arrays can be float64; compare the exact float32
    # model-input bytes used by the canonical asset, with explicit endianness.
    return hashlib.sha256(np.asarray(value, dtype="<f4").tobytes(order="C")).hexdigest()


def identity_sha(rows):
    records = [{key: str(row[key]) for key in IDENTITY_FIELDS} for row in rows]
    return hashlib.sha256(json.dumps(records, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def read_manifest(path):
    result = []
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        require(set(MANIFEST_FIELDS) <= set(reader.fieldnames or ()), "missing TAV manifest fields")
        for row in reader:
            selected = {key: row[key] for key in IDENTITY_FIELDS}
            # Deliberately do not index either Test label column.
            selected.update(audio_sha256=row["audio_sha256"], benchmark_label="", benchmark_label_id="")
            if row["split"] in ("train", "dev"):
                selected.update(benchmark_label=row["benchmark_label"], benchmark_label_id=row["benchmark_label_id"])
            result.append(selected)
    return result


def validate_metadata(asset, rows, *, expected_counts=COUNTS):
    require(asset.get("schema") == SCHEMA and asset.get("dataset_version") == VERSION, "wrong TAV schema/version")
    require(asset["label_vocab"]["itos"] == list(LABEL_VOCAB), "current label vocabulary mismatch")
    require(asset["feature_spec"] == {key: {"dimension": dim, "dtype": "float32"} for key, dim in DIMS.items()},
            "T/A/V input dimensions must be 1024/1582/342")
    splits = asset["splits"]
    require(set(splits) == set(COUNTS), "explicit canonical split metadata required")
    flat = [did for split in COUNTS for did in splits[split]]
    require(len(flat) == len(set(flat)) and set(flat) == set(asset["dialogues"]), "cross-split dialogue conflict")
    membership = {did: split for split in COUNTS for did in splits[split]}
    by_uid, ordered, ranks = {}, {}, set()
    for row in rows:
        uid = row["utterance_id"]
        match = UID.fullmatch(uid)
        require(match is not None and match[1] == row["dialogue_id"] and match[2] == row["speaker"], "ID/speaker conflict")
        require(uid not in by_uid, "duplicate utterance ID")
        order = integer(row["canonical_dialogue_order"], "canonical_dialogue_order")
        rank = (row["dialogue_id"], order)
        require(order > 0 and rank not in ranks, "duplicate/invalid canonical order")
        ranks.add(rank)
        by_uid[uid] = row
    seen, model_dids, counts = set(), set(), {}
    for split, dids in splits.items():
        n = 0
        for did in dids:
            dialogue = asset["dialogues"][did]
            ids = dialogue["utterance_ids"]
            require(len(ids) > 0 and len(dialogue["speakers"]) == len(ids), "identity lengths mismatch")
            values = []
            for index, uid in enumerate(ids):
                require(uid not in seen and uid in by_uid, "missing/duplicate exact ID")
                seen.add(uid)
                row = by_uid[uid]
                require(row["asset_dialogue_id"] == did and integer(row["asset_row_index"], "asset_row_index") == index
                        and row["split"] == membership[did] and row["speaker"] == dialogue["speakers"][index],
                        "asset row/speaker/split conflict")
                values.append((integer(row["canonical_dialogue_order"], "canonical_dialogue_order"), index, uid))
            require(len({by_uid[uid]["dialogue_id"] for uid in ids}) == 1, "mixed canonical dialogue")
            model_did = by_uid[ids[0]]["dialogue_id"]
            require(model_did not in model_dids, "canonical dialogue maps to multiple asset dialogues")
            model_dids.add(model_did)
            ordered[did] = tuple(sorted(values))
            n += len(ids)
        counts[split] = {"utterances": n, "dialogues": len(dids)}
    require(seen == set(by_uid), "asset/manifest ID sets differ")
    require(counts == expected_counts, f"canonical counts mismatch: {counts}")
    return by_uid, ordered


class TAVAuditDataset(Dataset):
    def __init__(self, asset, rows, *, split, expected_counts=COUNTS):
        require_split(split)
        by_uid, ordered = validate_metadata(asset, rows, expected_counts=expected_counts)
        self.split, self.asset, self.rows, self.ordered = split, asset, by_uid, ordered
        self.dialogue_ids = tuple(asset["splits"][split])
        for did in self.dialogue_ids:
            dialogue = asset["dialogues"][did]
            ids = dialogue["utterance_ids"]
            for key, dim in DIMS.items():
                arr = dialogue[key]
                require(isinstance(arr, np.ndarray) and arr.dtype == np.float32
                        and arr.shape == (len(ids), dim) and np.isfinite(arr).all(), f"invalid {key}")
            require(len(dialogue["benchmark_label_ids"]) == len(ids)
                    and len(dialogue["benchmark_labels"]) == len(ids), "Train/Dev target lengths")
            for index, uid in enumerate(ids):
                row, label = by_uid[uid], dialogue["benchmark_label_ids"][index]
                require(type(label) is int and 0 <= label < 6, "invalid Train/Dev target")
                require(integer(row["benchmark_label_id"], "benchmark_label_id") == label
                        and row["benchmark_label"] == dialogue["benchmark_labels"][index] == LABEL_VOCAB[label],
                        "current Train/Dev label identity mismatch")
                require(fingerprint(dialogue["audio_features"][index]) == row["audio_sha256"], "A1582 identity mismatch")

    @classmethod
    def from_files(cls, asset_path, manifest_path, *, split):
        require_split(split)
        with Path(asset_path).open("rb") as stream:
            asset = pickle.load(stream)
        return cls(asset, read_manifest(manifest_path), split=split)

    def __len__(self):
        return len(self.dialogue_ids)

    def __getitem__(self, index):
        did = self.dialogue_ids[index]
        dialogue = self.asset["dialogues"][did]
        order = self.ordered[did]
        indices = [value[1] for value in order]
        return {"dialogue_id": self.rows[order[0][2]]["dialogue_id"],
                "utterance_ids": tuple(value[2] for value in order),
                **{key: torch.from_numpy(dialogue[key][indices].copy()) for key in DIMS},
                "speaker_ids_int": torch.tensor([{"F": 0, "M": 1}[dialogue["speakers"][i]] for i in indices]),
                "labels": torch.tensor([dialogue["benchmark_label_ids"][i] for i in indices]), "split": self.split}


def collate(samples):
    require(bool(samples) and len({s["split"] for s in samples}) == 1, "empty/mixed split batch")
    require_split(samples[0]["split"])
    lengths = torch.tensor([len(s["utterance_ids"]) for s in samples])
    b, t = len(samples), int(lengths.max())
    batch = {key: torch.zeros(b, t, dim) for key, dim in DIMS.items()}
    batch.update(speaker_ids_int=torch.full((b, t), -1, dtype=torch.int64),
                 labels=torch.full((b, t), -100, dtype=torch.int64), lengths=lengths,
                 attention_mask=torch.arange(t)[None] < lengths[:, None],
                 utterance_ids=tuple(s["utterance_ids"] for s in samples),
                 dialogue_ids=tuple(s["dialogue_id"] for s in samples), split=samples[0]["split"])
    for index, sample in enumerate(samples):
        for key in (*DIMS, "speaker_ids_int", "labels"):
            batch[key][index, :int(lengths[index])] = sample[key]
    return batch

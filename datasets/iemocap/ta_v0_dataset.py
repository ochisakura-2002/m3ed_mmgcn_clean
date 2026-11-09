"""Read-only canonical-order view of the asymmetric_merc_v1 TA package.

Only trusted local PKLs are supported. The separate authoritative canonical CSV
may be a superset of the six-class TA universe. It is joined by exact ID, never
by row position or timestamp. No media/visual columns are consumed.
"""

import csv
from pathlib import Path
import pickle
import re

import numpy as np
import torch
from torch.utils.data import Dataset

from models.multidag_cl.ta_v0.config import LABEL_VOCAB

SCHEMA = "iemocap_asym_ta_v1"
VERSION = "asymmetric_merc_v1"
SPLITS = ("train", "dev", "test")
UID = re.compile(r"(Ses0[1-5][FM]_(?:impro\d+[ab]?|script\d+_\d+[ab]?))_([FM])\d+")
BRIDGE_FIELDS = (
    "dataset_version", "utterance_id", "dialogue_id", "ta_dialogue_id",
    "speaker", "split", "benchmark_label", "benchmark_label_id",
    "ta_row_index", "canonical_dialogue_order",
)
CANONICAL_FIELDS = ("utterance_id", "dialogue_id", "speaker", "global_order")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def integer(value, name):
    require(type(value) is int or isinstance(value, str) and value.isdecimal(),
            f"{name} must be an exact nonnegative integer")
    result = int(value)
    require(result >= 0, f"{name} must be nonnegative")
    return result


def unique_rows(rows, fields, name):
    result = {}
    for row in rows:
        require(all(k in row and row[k] is not None and row[k] != "" for k in fields),
                f"{name}: missing metadata")
        uid = row["utterance_id"]
        require(isinstance(uid, str) and uid not in result, f"{name}: duplicate/invalid ID {uid}")
        result[uid] = row
    require(bool(result), f"{name}: empty metadata")
    return result


def read_columns(path, fields, optional_fields=()):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        require(reader.fieldnames is not None and set(fields) <= set(reader.fieldnames),
                f"missing columns in {path}")
        # Other CSV columns are never propagated to the training input.
        selected = (*fields, *(k for k in optional_fields if k in reader.fieldnames))
        return [{k: row[k] for k in selected} for row in reader]


class TAV0Dataset(Dataset):
    def __init__(self, asset, bridge_rows, canonical_rows, *, split):
        require(split in ("train", "dev"), "development allows only train/dev; Test loader forbidden")
        require(isinstance(asset, dict) and asset.get("schema") == SCHEMA, "wrong TA schema")
        require(set(asset) == {"schema", "dataset_version", "feature_spec", "label_vocab", "dialogues", "splits"},
                "unexpected asset fields")
        require(asset["dataset_version"] == VERSION, "wrong package version")
        require(set(asset["feature_spec"]) == {"text", "audio_ta"}, "TA-only feature schema required")
        vocab = asset["label_vocab"]
        require(vocab.get("itos") == list(LABEL_VOCAB), "source label order mismatch; explicit remap required")
        if "stoi" in vocab:
            require(vocab["stoi"] == {s: i for i, s in enumerate(LABEL_VOCAB)}, "source label mapping mismatch")
        splits = asset["splits"]
        require(set(splits) == set(SPLITS), "explicit train/dev/test membership required")
        flat = [d for s in SPLITS for d in splits[s]]
        require(len(set(flat)) == len(flat) and set(flat) == set(asset["dialogues"]),
                "duplicate/cross-split/missing dialogue membership")
        membership = {d: s for s in SPLITS for d in splits[s]}
        bridge = unique_rows(bridge_rows, BRIDGE_FIELDS, "bridge")
        canonical = unique_rows(canonical_rows, CANONICAL_FIELDS, "canonical")
        ranks = {}
        for uid, row in canonical.items():
            match = UID.fullmatch(uid)
            require(match is not None and match[1] == row["dialogue_id"] and match[2] == row["speaker"],
                    f"canonical identity conflict: {uid}")
            order = integer(row["global_order"], "global_order")
            key = (row["dialogue_id"], order)
            require(order > 0 and key not in ranks, f"canonical order conflict: {uid}")
            ranks[key] = uid

        self.split = split
        self._asset = asset
        self._ordered = {}
        seen = set()
        fields = {"utterance_ids", "speakers", "benchmark_labels", "benchmark_label_ids",
                  "sentences", "text_features", "audio_features"}
        for did, dialogue in asset["dialogues"].items():
            require(set(dialogue) == fields, f"unexpected TA dialogue fields: {did}")
            ids = dialogue["utterance_ids"]
            require(len(ids) > 0, f"empty dialogue: {did}")
            for key in ("speakers", "benchmark_labels", "benchmark_label_ids", "sentences"):
                require(len(dialogue[key]) == len(ids), f"metadata lengths differ: {did}")
            ordered = []
            for index, uid in enumerate(ids):
                require(uid not in seen and uid in bridge and uid in canonical,
                        f"duplicate/missing exact ID: {uid}")
                seen.add(uid)
                row, authority = bridge[uid], canonical[uid]
                match = UID.fullmatch(uid)
                speaker = dialogue["speakers"][index]
                require(match is not None and match[1] == did and match[2] == speaker,
                        f"asset identity conflict: {uid}")
                require(row["dataset_version"] == VERSION
                        and row["dialogue_id"] == row["ta_dialogue_id"] == authority["dialogue_id"] == did
                        and row["speaker"] == authority["speaker"] == speaker
                        and row["split"] == membership[did], f"bridge identity/split conflict: {uid}")
                require(integer(row["ta_row_index"], "ta_row_index") == index,
                        f"wrong TA row mapping: {uid}")
                order = integer(authority["global_order"], "global_order")
                require(integer(row["canonical_dialogue_order"], "canonical_dialogue_order") == order,
                        f"bridge/canonical order conflict: {uid}")
                label_id = dialogue["benchmark_label_ids"][index]
                require(type(label_id) is int and 0 <= label_id < 6, f"invalid source label: {uid}")
                label = LABEL_VOCAB[label_id]
                require(dialogue["benchmark_labels"][index] == row["benchmark_label"] == label
                        and integer(row["benchmark_label_id"], "benchmark_label_id") == label_id,
                        f"label mapping conflict: {uid}")
                if authority.get("split") not in (None, ""):
                    require(authority["split"] == membership[did], f"canonical split conflict: {uid}")
                ordered.append((order, index, uid))
            self._ordered[did] = tuple(sorted(ordered))
            if membership[did] == split:
                for key, dim in (("text_features", 768), ("audio_features", 1582)):
                    arr = dialogue[key]
                    require(isinstance(arr, np.ndarray) and arr.shape == (len(ids), dim)
                            and arr.dtype == np.float32 and np.isfinite(arr).all(),
                            f"wrong feature shape/dtype/finite: {did}/{key}")
        require(seen == set(bridge), "bridge/TA ID sets differ")
        self.dialogue_ids = tuple(splits[split])
        require(bool(self.dialogue_ids), f"empty {split} split")

    @classmethod
    def from_files(cls, asset_path, bridge_path, canonical_path, *, split):
        require(split in ("train", "dev"), "development allows only train/dev; Test loader forbidden")
        # Explicit caller paths only; no search, writeback, or source-array reorder.
        bridge = read_columns(bridge_path, BRIDGE_FIELDS)
        canonical = read_columns(canonical_path, CANONICAL_FIELDS, optional_fields=("split",))
        with Path(asset_path).open("rb") as stream:
            asset = pickle.load(stream)
        return cls(asset, bridge, canonical, split=split)

    def __len__(self):
        return len(self.dialogue_ids)

    def __getitem__(self, index):
        did = self.dialogue_ids[index]
        dialogue = self._asset["dialogues"][did]
        ordered = self._ordered[did]
        indices = [r[1] for r in ordered]
        speakers = [dialogue["speakers"][i] for i in indices]
        # Advanced indexing copies the selected rows; the source array is untouched.
        return {
            "dialogue_id": did, "utterance_ids": tuple(r[2] for r in ordered),
            "text_features": torch.from_numpy(dialogue["text_features"][indices].copy()),
            "audio_features": torch.from_numpy(dialogue["audio_features"][indices].copy()),
            "speaker_ids_int": torch.tensor([{"F": 0, "M": 1}[s] for s in speakers]),
            "labels": torch.tensor([dialogue["benchmark_label_ids"][i] for i in indices]),
            "canonical_order": tuple(r[0] for r in ordered),
            "ta_row_indices": tuple(indices), "split": self.split,
        }


def ta_v0_collate(samples):
    require(bool(samples), "empty batch")
    require(len({s["split"] for s in samples}) == 1, "mixed split batch")
    lengths = torch.tensor([len(s["utterance_ids"]) for s in samples], dtype=torch.int64)
    b, length = len(samples), int(lengths.max())
    batch = {
        "text_features": torch.zeros(b, length, 768),
        "audio_features": torch.zeros(b, length, 1582),
        "speaker_ids_int": torch.full((b, length), -1, dtype=torch.int64),
        "labels": torch.full((b, length), -100, dtype=torch.int64),
        "lengths": lengths,
        "attention_mask": torch.arange(length)[None, :] < lengths[:, None],
        "utterance_ids": tuple(s["utterance_ids"] for s in samples),
        "dialogue_ids": tuple(s["dialogue_id"] for s in samples),
        "split": samples[0]["split"],
    }
    for i, sample in enumerate(samples):
        n = int(lengths[i])
        for name in ("text_features", "audio_features", "speaker_ids_int", "labels"):
            batch[name][i, :n] = sample[name]
    return batch

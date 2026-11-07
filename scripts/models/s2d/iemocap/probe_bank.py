"""Explicit Train/Dev bank reader. No source-run, video, checkpoint or Test IO."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re

import torch


SCHEMA = "s2d_uniform16_feature_bank_v1"
CONTRACT = "s2d_q1_a0_c0_d0_b_v1"
SPLITS = ("train", "dev")
LABELS = ("hap", "sad", "neu", "ang", "exc", "fru")
FEATURES = {"a0": "a0_rgb_mean", "c0": "c0_mcp_static_mean",
            "d0": "d0_static_residual_mean", "b": "s2d_contextual_slots"}
SHAPES = {**{field: (16, 768) for field in FEATURES.values()},
          "h_v": (768,), "logits_v": (6,)}
EXPECTED_EPOCH = 7
EXPECTED_DEV_WEIGHTED_F1 = 0.34355913724532466
BASELINE_TOLERANCE = 1e-10


def read_json(path):
    result = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(result, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return result


def identity(path):
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {"sha256": digest.hexdigest(), "size_bytes": path.stat().st_size}


def require_splits(values):
    if not isinstance(values, (list, tuple, dict)) or set(values) != set(SPLITS) or len(values) != 2:
        raise ValueError("Q1 requires exactly train/dev; other splits are forbidden")


def direct_file(directory, name):
    path = directory / name
    if path.is_symlink() or path.resolve().parent != directory:
        raise ValueError("Bank metadata/shards must be direct, non-symlink children")
    return path


class FeatureBank:
    """Validate metadata first, then all explicit shards before any fitting.

    No glob, recursive discovery or fallback paths. Metadata references outside
    the direct split directory (including symlinks) are refused before opening.
    Features are validated one shard at a time, then one representation is
    materialized at a time by the runner.
    """

    def __init__(self, root):
        self.root = Path(root).resolve()
        self.provenance = read_json(direct_file(self.root, "provenance.json"))
        complete = read_json(direct_file(self.root, "complete.json"))
        coverage = read_json(direct_file(self.root, "coverage.json"))
        for obj in (self.provenance, complete, coverage):
            if obj.get("schema_version") != SCHEMA:
                raise ValueError("Unsupported feature-bank schema_version")
            require_splits(obj.get("splits"))
        if (not isinstance(self.provenance["splits"], list) or not isinstance(complete["splits"], list)
                or not isinstance(coverage["splits"], dict)):
            raise ValueError("Invalid train/dev split metadata containers")
        p = self.provenance
        if (complete.get("status") != "complete" or p.get("feature_contract_version") != CONTRACT
                or p.get("dtype") != "float32" or p.get("num_frames") != 16
                or p.get("sampling_policy") != "Uniform16" or p.get("shuffle") is not False
                or p.get("feature_fields") != {k: list(v) for k, v in SHAPES.items()}
                or p.get("label_vocab", {}).get("index_to_label") != list(LABELS)
                or p.get("label_vocab", {}).get("label_to_index") != {v: i for i, v in enumerate(LABELS)}
                or p.get("label_vocab", {}).get("target") != "benchmark_label"
                or type(p.get("selected_epoch")) is not int or p["selected_epoch"] != EXPECTED_EPOCH
                or not isinstance(p.get("selected_dev_weighted_f1"), (float, int))
                or not math.isfinite(p["selected_dev_weighted_f1"])
                or abs(p["selected_dev_weighted_f1"] - EXPECTED_DEV_WEIGHTED_F1) > BASELINE_TOLERANCE):
            raise ValueError("Frozen bank provenance / epoch7 reference contract mismatch")
        self.indices, self.ids, self.labels, self.shard_identities = {}, {}, {}, {}
        self.metadata_identities = {name: identity(self.root / name) for name in (
            "provenance.json", "coverage.json", "complete.json")}
        all_ids = set()
        for split in SPLITS:
            directory = self.root / split
            if directory.resolve().parent != self.root or directory.is_symlink():
                raise ValueError("Split path must be a direct, non-symlink bank child")
            index = read_json(direct_file(directory, "index.json"))
            self.metadata_identities[f"{split}/index.json"] = identity(directory / "index.json")
            if index.get("schema_version") != SCHEMA or index.get("split") != split:
                raise ValueError("Invalid split index schema")
            shards, rows = index.get("shards"), index.get("rows")
            if (not isinstance(shards, list) or not isinstance(rows, list)
                    or type(index.get("row_count")) is not int or index["row_count"] < 1
                    or len(rows) != index["row_count"] or not shards):
                raise ValueError("Non-empty, consistent split index required")
            expected_rows, split_ids, label_parts = [], [], []
            for part, shard in enumerate(shards):
                if not isinstance(shard, dict):
                    raise ValueError("Invalid shard metadata")
                filename = shard.get("filename")
                if (not isinstance(filename, str) or re.fullmatch(r"part-[0-9]{5}\.pt", filename) is None
                        or filename != f"part-{part:05d}.pt"
                        or type(shard.get("row_count")) is not int or shard["row_count"] < 1):
                    raise ValueError("Only explicit canonical train/dev shard filenames allowed")
                count = shard["row_count"]
                offset = len(expected_rows)
                for row_no in range(count):
                    if offset + row_no >= len(rows) or not isinstance(rows[offset + row_no], dict):
                        raise ValueError("Missing shard index row")
                    uid = rows[offset + row_no].get("utterance_id")
                    if not isinstance(uid, str) or not uid.strip() or uid in all_ids:
                        raise ValueError("Utterance IDs must be non-empty and globally unique")
                    all_ids.add(uid)
                    split_ids.append(uid)
                    expected_rows.append({"global_row": offset + row_no, "utterance_id": uid,
                                          "shard": filename, "row_in_shard": row_no})
            if rows != expected_rows:
                raise ValueError("Index order/row mapping mismatch")
            row = coverage["splits"][split]
            if not isinstance(row, dict):
                raise ValueError("Invalid split coverage schema")
            exclusions = row.get("exclusions", {})
            if (not isinstance(exclusions, dict) or set(exclusions) != {"v_unavailable", "cached_face_unusable"}
                    or any(not isinstance(ids, list) for ids in exclusions.values())):
                raise ValueError("Invalid exclusion schema")
            excluded = [uid for ids in exclusions.values() for uid in ids]
            if (any(not isinstance(uid, str) or not uid for uid in excluded)
                    or len(excluded) != len(set(excluded)) or all_ids.intersection(excluded)
                    or row.get("exported_count") != len(rows) or row.get("face_cache_usable") != len(rows)
                    or row.get("unknown_exclusion_count") != 0
                    or row.get("excluded_count") != len(excluded)
                    or row.get("manifest_total") != len(rows) + len(excluded)
                    or row.get("v_available") != len(rows) + len(exclusions["cached_face_unusable"])):
                raise ValueError("Coverage/index/exclusion accounting mismatch")
            all_ids.update(excluded)
            self.indices[split], self.ids[split] = index, split_ids
            for shard in shards:
                path = self._shard_path(split, shard["filename"])
                self.shard_identities[f"{split}/{shard['filename']}"] = identity(path)
                payload = self._read_shard(split, shard, check_identity=False)
                label_parts.append(payload["labels"])
            self.labels[split] = torch.cat(label_parts)
        self.coverage = coverage

    def _shard_path(self, split, filename):
        if split not in SPLITS:
            raise ValueError("Only train/dev allowed")
        path = self.root / split / filename
        if path.resolve().parent != self.root / split or path.is_symlink():
            raise ValueError("Shard must remain a direct, non-symlink split child")
        return path

    def _read_shard(self, split, shard, *, check_identity=True):
        path = self._shard_path(split, shard["filename"])
        if check_identity and identity(path) != self.shard_identities[f"{split}/{path.name}"]:
            raise ValueError("Frozen feature bank changed during probing")
        payload = torch.load(path, map_location="cpu", weights_only=True)
        keys = {"schema_version", "split", "utterance_ids", "labels", "frame_indices", *SHAPES}
        if (not isinstance(payload, dict) or set(payload) != keys
                or payload.get("schema_version") != SCHEMA or payload.get("split") != split):
            raise ValueError("Invalid train/dev shard schema")
        n = shard["row_count"]
        expected_ids = [r["utterance_id"] for r in self.indices[split]["rows"]
                        if r["shard"] == path.name]
        if payload["utterance_ids"] != expected_ids:
            raise ValueError("Shard IDs disagree with index")
        for key, shape in {**SHAPES, "labels": (), "frame_indices": (16,)}.items():
            value = payload[key]
            dtype = torch.int64 if key in {"labels", "frame_indices"} else torch.float32
            if (not isinstance(value, torch.Tensor) or value.dtype != dtype
                    or tuple(value.shape) != (n, *shape) or not bool(torch.isfinite(value).all())):
                raise ValueError(f"Invalid feature shape/dtype/finiteness: {key}")
        labels, frames = payload["labels"], payload["frame_indices"]
        if (bool(((labels < 0) | (labels >= 6)).any()) or bool((frames < 0).any())
                or bool((frames[:, 1:] < frames[:, :-1]).any())):
            raise ValueError("Invalid labels or ordered frame indices")
        if not torch.allclose(payload[FEATURES["b"]].mean(1), payload["h_v"], rtol=1e-5, atol=1e-6):
            raise ValueError("mean_t(B) != h_v")
        return payload

    def values(self, split, field, *, pooled=False):
        if split not in SPLITS or field not in SHAPES:
            raise ValueError("Only explicit train/dev feature fields allowed")
        parts = []
        for shard in self.indices[split]["shards"]:
            value = self._read_shard(split, shard)[field]
            parts.append(value.mean(1) if pooled else value)
        return torch.cat(parts)

    def normalization(self, feature):
        """Per-channel population moments fitted over Train utterances AND slots.

        The same affine transform is used before TCN and after linear pooling;
        mean pooling commutes with this transform. Never reads Dev here.
        """
        field = FEATURES[feature]
        mean, m2, count = torch.zeros(768, dtype=torch.float64), torch.zeros(768, dtype=torch.float64), 0
        for shard in self.indices["train"]["shards"]:
            values = self._read_shard("train", shard)[field].double().reshape(-1, 768)
            variance, part_mean = torch.var_mean(values, dim=0, unbiased=False)
            part_count = len(values)
            delta = part_mean - mean
            merged_count = count + part_count
            m2 += variance * part_count + delta.square() * (count * part_count / merged_count)
            mean += delta * (part_count / merged_count)
            count = merged_count
        std = (m2 / count).clamp_min(0).sqrt().clamp_min(1e-6)
        return {"feature": feature, "field": field, "fit_split": "train", "axes": ["utterance", "slot"],
                "count": count, "epsilon": 1e-6, "ddof": 0,
                "mean": mean.float(), "std": std.float()}

    def assert_unchanged(self):
        for name, expected in {**self.metadata_identities, **self.shard_identities}.items():
            if identity(self.root / name) != expected:
                raise ValueError("Frozen feature bank changed during probing")

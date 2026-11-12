"""Strict TA/S2D-B readers and exact-ID cache contract. No source expert IO."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

import numpy as np
import torch

from scripts.analysis.models.asymmetric_merc.fusion_report import write_json
from scripts.analysis.models.asymmetric_merc.weighted_fusion import (
    LABEL_NAMES, Predictions, exact_join, load_predictions,
)
from utils.output_paths import resolve_output_paths
from utils.run_metadata import compute_file_sha256

ROOT = Path(__file__).resolve().parents[3]
BANK_SCHEMA = "s2d_uniform16_feature_bank_v1"
CACHE_SCHEMA = "asymmetric_merc_frozen_tav_v1"
V_SOURCE = "S2D-B mean contextual slots"
V_PRED_SOURCE = "Q2-1 frozen Linear-B seed100 predictions (caller-declared)"
SPLITS = ("train", "dev")
EPSILON = 1e-6
EXPECTED = {"train": (4830, 4818, 12), "dev": (980, 978, 2)}


def require_split(split):
    if split not in SPLITS:
        raise ValueError("Only train/dev allowed; Test is forbidden")


def guard_input(path):
    """Reject explicitly Test-named paths, including resolved symlink targets, before IO."""
    path = Path(path)
    for value in (path, path.resolve()):
        if any(re.search(r"(^|[^a-z0-9])test([^a-z0-9]|$)", part.lower()) for part in value.parts):
            raise ValueError("Test input path is forbidden")
    return path


def direct_file(directory, filename):
    directory = guard_input(directory).resolve()
    if not isinstance(filename, str) or Path(filename).name != filename:
        raise ValueError("Only direct metadata/shard filenames allowed")
    path = guard_input(directory / filename)
    if path.is_symlink() or path.resolve().parent != directory:
        raise ValueError("Metadata/shards must be direct non-symlink children")
    return path


def read_json(path):
    value = json.loads(guard_input(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Expected JSON object")
    return value


def identity(path):
    path = guard_input(path)
    return {"path": path.as_posix(), "sha256": compute_file_sha256(path)}


def assert_unchanged(records):
    for record in records:
        if identity(record["path"])["sha256"] != record["sha256"]:
            raise ValueError("Frozen input changed while reading")


def ids_array(values, name):
    values = np.asarray(values)
    if values.ndim != 1 or values.dtype.kind not in "US":
        raise ValueError(f"{name}: expected a string vector (NPZ requires no pickle)")
    result = values.astype(str)
    if any(not item.strip() for item in result) or len(set(result)) != len(result):
        raise ValueError(f"{name}: empty or duplicate IDs")
    return result


def numeric(values, shape, name, *, labels=False):
    value = np.asarray(values)
    if value.shape != shape or value.dtype.kind not in "fiu" or not np.isfinite(value).all():
        raise ValueError(f"{name}: invalid shape, numeric dtype or finite values")
    if labels and (value.dtype.kind not in "iu" or ((value < 0) | (value >= 6)).any()):
        raise ValueError(f"{name}: labels must be integers 0..5")
    converted = value.astype(np.int64 if labels else np.float32)
    if not np.isfinite(converted).all():
        raise ValueError(f"{name}: numeric overflow on conversion")
    return converted


def probabilities(values, n, name):
    value = numeric(values, (n, 6), name)
    if (value < 0).any() or not np.allclose(value.sum(1), 1, atol=1e-5, rtol=0):
        raise ValueError(f"{name}: invalid probability simplex")
    return value


def validate_common(data):
    ids = ids_array(data["utterance_id"], "utterance_id")
    n = len(ids)
    if not n:
        raise ValueError("Empty common split")
    dialogs = np.asarray(data["dialogue_id"])
    if dialogs.shape != (n,) or dialogs.dtype.kind not in "US" or any(not x.strip() for x in dialogs):
        raise ValueError("Invalid dialogue_id vector")
    result = {"utterance_id": ids, "dialogue_id": dialogs.astype(str),
              "label": numeric(data["label"], (n,), "label", labels=True),
              "h_ta": numeric(data["h_ta"], (n, 1500), "h_ta"),
              "ta_logits": numeric(data["ta_logits"], (n, 6), "ta_logits"),
              "ta_prob": probabilities(data["ta_prob"], n, "ta_prob")}
    logits = result["ta_logits"].astype(np.float64)
    exp = np.exp(logits - logits.max(1, keepdims=True))
    if not np.allclose(exp / exp.sum(1, keepdims=True), result["ta_prob"], atol=1e-5, rtol=0):
        raise ValueError("TA logits/probabilities disagree")
    if "h_v" in data:
        result["h_v"] = numeric(data["h_v"], (n, 768), "h_v")
    if ("v_probe_pred" in data) != ("v_probe_prob" in data):
        raise ValueError("Incomplete Linear-B predictions")
    if "v_probe_pred" in data:
        result["v_probe_pred"] = numeric(data["v_probe_pred"], (n,), "v_probe_pred", labels=True)
        result["v_probe_prob"] = probabilities(data["v_probe_prob"], n, "v_probe_prob")
        if not np.array_equal(result["v_probe_pred"], result["v_probe_prob"].argmax(1)):
            raise ValueError("Linear-B probabilities/predictions disagree")
    return result


def load_ta(path, split):
    require_split(split)
    path = guard_input(path)
    before = identity(path)
    with np.load(path, allow_pickle=False) as archive:
        if any(re.search(r"(^|_)test(_|$)", key.lower()) for key in archive.files):
            raise ValueError("Test fields are forbidden")
        evidence = "caller_declared_" + split
        if "split" in archive.files:
            aliases = {"train"} if split == "train" else {"dev", "val", "validation"}
            declared = np.asarray(archive["split"]).reshape(-1).tolist()
            if not declared or not set(declared) <= aliases:
                raise ValueError("TA split metadata disagrees; Test is forbidden")
            evidence = "validated_split_metadata"
        mapping = {"ta_logits": "logits_ta", "ta_prob": "prob_ta"}
        values = {key: archive[mapping.get(key, key)] for key in (
            "utterance_id", "dialogue_id", "label", "h_ta", "ta_logits", "ta_prob")}
    result = validate_common(values)
    assert_unchanged([before])
    return result, {**before, "split_evidence": evidence}


def _two_splits(value):
    if not isinstance(value, (list, dict)) or len(value) != 2 or set(value) != set(SPLITS):
        raise ValueError("Bank must declare exactly train/dev; Test is forbidden")


def load_visual_bank(root):
    """Read explicit Q1 bank metadata first, then one shard at a time; never glob."""
    root = guard_input(root).resolve()
    records = []

    def metadata(directory, filename):
        path = direct_file(directory, filename)
        records.append(identity(path))
        return read_json(path)

    provenance, coverage, complete = [metadata(root, name) for name in (
        "provenance.json", "coverage.json", "complete.json")]
    for obj in (provenance, coverage, complete):
        if obj.get("schema_version") != BANK_SCHEMA:
            raise ValueError("Unsupported bank schema")
        _two_splits(obj.get("splits"))
    vocab = provenance.get("label_vocab", {})
    if (complete.get("status") != "complete" or provenance.get("selected_epoch") != 7
            or provenance.get("feature_contract_version") != "s2d_q1_a0_c0_d0_b_v1"
            or provenance.get("num_frames") != 16 or provenance.get("dtype") != "float32"
            or provenance.get("sampling_policy") != "Uniform16" or provenance.get("shuffle") is not False
            or vocab.get("index_to_label") != list(LABEL_NAMES)
            or vocab.get("label_to_index") != {v: k for k, v in enumerate(LABEL_NAMES)}
            or vocab.get("target") != "benchmark_label"):
        raise ValueError("Frozen Q1 bank provenance contract mismatch")
    indices, exclusions, all_ids = {}, {}, set()
    # Validate BOTH split indices and exclusion accounting before opening any shard.
    for split in SPLITS:
        directory = root / split
        if directory.is_symlink() or directory.resolve().parent != root:
            raise ValueError("Invalid split directory")
        index = metadata(directory, "index.json")
        if index.get("schema_version") != BANK_SCHEMA or index.get("split") != split:
            raise ValueError("Invalid index split/schema; Test is forbidden")
        shards, rows = index.get("shards"), index.get("rows")
        if not isinstance(shards, list) or not shards or not isinstance(rows, list) or not rows:
            raise ValueError("Non-empty shard/index lists required")
        expected, offset = [], 0
        for k, shard in enumerate(shards):
            filename, count = shard.get("filename"), shard.get("row_count")
            if filename != f"part-{k:05d}.pt" or type(count) is not int or count < 1:
                raise ValueError("Invalid explicit shard filename/count")
            direct_file(directory, filename)
            for i in range(count):
                if offset + i >= len(rows):
                    raise ValueError("Missing index rows")
                uid = rows[offset + i].get("utterance_id")
                expected.append({"global_row": offset + i, "utterance_id": uid,
                                 "shard": filename, "row_in_shard": i})
            offset += count
        if rows != expected or index.get("row_count") != len(rows):
            raise ValueError("Index row mapping/count mismatch")
        ids = ids_array([row["utterance_id"] for row in rows], f"{split} bank IDs")
        cov = coverage["splits"][split]
        exc = cov.get("exclusions")
        if (not isinstance(exc, dict) or set(exc) != {"v_unavailable", "cached_face_unusable"}
                or any(not isinstance(value, list) for value in exc.values())):
            raise ValueError("Invalid known exclusion categories")
        excluded = [uid for values in exc.values() for uid in values]
        if excluded:
            ids_array(excluded, "excluded IDs")
        universe = set(ids) | set(excluded)
        if (set(ids) & set(excluded) or all_ids & universe
                or cov.get("unknown_exclusion_count") != 0
                or cov.get("manifest_total") != len(ids) + len(excluded)
                or cov.get("exported_count") != len(ids) or cov.get("face_cache_usable") != len(ids)
                or cov.get("excluded_count") != len(excluded)
                or cov.get("v_available") != len(ids) + len(exc["cached_face_unusable"])):
            raise ValueError("Unknown V exclusion or coverage/cross-split accounting mismatch")
        all_ids.update(universe)
        indices[split], exclusions[split] = index, excluded
    result = {}
    for split in SPLITS:
        ids, labels, features = [], [], []
        for shard in indices[split]["shards"]:
            path = direct_file(root / split, shard["filename"])
            records.append(identity(path))
            payload = torch.load(path, map_location="cpu", weights_only=True)
            if payload.get("schema_version") != BANK_SCHEMA or payload.get("split") != split:
                raise ValueError("Invalid shard split/schema; Test is forbidden")
            expected_ids = [row["utterance_id"] for row in indices[split]["rows"]
                            if row["shard"] == shard["filename"]]
            if payload.get("utterance_ids") != expected_ids:
                raise ValueError("Shard/index ID mismatch")
            n = shard["row_count"]
            for key, shape, dtype in (("labels", (n,), torch.int64),
                                      ("s2d_contextual_slots", (n, 16, 768), torch.float32)):
                value = payload[key]
                if (not isinstance(value, torch.Tensor) or value.dtype != dtype
                        or tuple(value.shape) != shape or not bool(torch.isfinite(value).all())):
                    raise ValueError(f"Invalid bank tensor: {key}")
            ids.extend(expected_ids)
            labels.append(payload["labels"].numpy())
            # Sole V source: do not use original logits_v or A0/C0/D0.
            features.append(payload["s2d_contextual_slots"].mean(dim=1).numpy())
        n = len(ids)
        result[split] = {"utterance_id": ids_array(ids, "bank IDs"),
                         "label": numeric(np.concatenate(labels), (n,), "V labels", labels=True),
                         "h_v": numeric(np.concatenate(features), (n, 768), "mean(B)")}
    assert_unchanged(records)
    return result, exclusions, records


def join_features(ta, visual, exclusions):
    ti = {uid: k for k, uid in enumerate(ids_array(ta["utterance_id"], "TA IDs"))}
    vi = {uid: k for k, uid in enumerate(ids_array(visual["utterance_id"], "V IDs"))}
    common = sorted(ti.keys() & vi.keys())
    ta_only, v_only = sorted(ti.keys() - vi.keys()), sorted(vi.keys() - ti.keys())
    if not common or v_only or set(ta_only) != set(exclusions):
        raise ValueError("Exact-ID join: empty overlap, V-only IDs or unknown V exclusion")
    t = np.array([ti[uid] for uid in common])
    v = np.array([vi[uid] for uid in common])
    if not np.array_equal(ta["label"][t], visual["label"][v]):
        raise ValueError("Exact-ID label mismatch; no correction allowed")
    result = {key: value[t] for key, value in ta.items()}
    result["h_v"] = visual["h_v"][v]
    return result, {"key": "utterance_id", "order": "lexicographic_id", "ta_rows": len(ti),
                    "v_rows": len(vi), "common_rows": len(common), "ta_only_count": len(ta_only),
                    "ta_only_ids": ta_only, "v_only_ids": v_only, "duplicate_ids": 0,
                    "label_mismatch": 0, "unknown_v_exclusion": 0}


def attach_probe(dev, path):
    guard_input(path)
    visual = load_predictions(path)  # Frozen Q2-2 reader: numeric six-class Dev CSV/TSV.
    ta = Predictions(tuple(dev["utterance_id"]), dev["label"], dev["ta_prob"].argmax(1),
                     dev["ta_prob"], {})
    joined = exact_join(ta, visual)
    if set(visual.ids) != set(ta.ids):
        raise ValueError("Linear-B Dev predictions must cover exactly Dev common IDs")
    positions = {uid: i for i, uid in enumerate(joined.ids)}
    order = np.array([positions[uid] for uid in dev["utterance_id"]])
    result = {**dev, "v_probe_pred": joined.v_predicted[order], "v_probe_prob": joined.v_probabilities[order]}
    return result, {**visual.manifest, "source": V_PRED_SOURCE, "exact_join": joined.audit}


def id_digest(ids):
    return hashlib.sha256(json.dumps(list(ids), ensure_ascii=False).encode("utf-8")).hexdigest()


def fit_normalization(train):
    result = {}
    for field, tag in (("h_ta", "ta"), ("h_v", "v")):
        values = np.asarray(train[field], dtype=np.float64)
        result[f"mu_{tag}"] = values.mean(axis=0)
        result[f"sigma_{tag}"] = np.maximum(values.std(axis=0, ddof=0), EPSILON)
    return result


def standardize(data, normalization):
    return tuple(((data[field] - normalization[f"mu_{tag}"]) / normalization[f"sigma_{tag}"])
                 .astype(np.float32) for field, tag in (("h_ta", "ta"), ("h_v", "v")))


def reserve_output(path, kind):
    path = Path(path).resolve()
    try:
        parts = path.relative_to(ROOT / "outputs").parts
    except ValueError:
        parts = None
    if parts is not None:
        if kind == "run":
            if len(parts) != 4 or parts[2] != "runs":
                raise ValueError("Run output must be outputs/<date>/<group>/runs/<run_id>")
            expected = resolve_output_paths(output_base=ROOT / "outputs", experiment_date=parts[0],
                                            experiment_group=parts[1], run_id=parts[3]).run_root
        else:
            if len(parts) != 6 or parts[2:4] != ("analysis", "batches") or parts[5] != "frozen_tav_cache":
                raise ValueError("Cache output must be outputs/<date>/<group>/analysis/batches/<id>/frozen_tav_cache")
            expected = resolve_output_paths(output_base=ROOT / "outputs", experiment_date=parts[0],
                                            experiment_group=parts[1], batch_id=parts[4]).analysis_root / parts[5]
        if path != expected:
            raise ValueError("Non-canonical output path")
    path.mkdir(parents=True, exist_ok=False)
    return path


def build_cache(ta_train_npz, ta_dev_npz, s2d_bank_dir, output_dir, *,
                v_quadrant_predictions=None, synthetic=False):
    # Reject all supplied Test paths before reading even the first Train file.
    for path in (ta_train_npz, ta_dev_npz, s2d_bank_dir, v_quadrant_predictions):
        if path is not None:
            guard_input(path)
    ta, records = {}, []
    for split, path in zip(SPLITS, (ta_train_npz, ta_dev_npz)):
        ta[split], record = load_ta(path, split)
        records.append(record)
    if set(ta["train"]["utterance_id"]) & set(ta["dev"]["utterance_id"]):
        raise ValueError("Train/Dev utterance ID overlap")
    if set(ta["train"]["dialogue_id"]) & set(ta["dev"]["dialogue_id"]):
        raise ValueError("Train/Dev dialogue ID overlap")
    visual, exclusions, bank_records = load_visual_bank(s2d_bank_dir)
    records.extend(bank_records)
    common, audits = {}, {}
    for split in SPLITS:
        common[split], audits[split] = join_features(ta[split], visual[split], exclusions[split])
        row = audits[split]
        if not synthetic and (row["ta_rows"], row["common_rows"], row["ta_only_count"]) != EXPECTED[split]:
            raise ValueError(f"Frozen {split} coverage differs from required {EXPECTED[split]}")
    probe = None
    if v_quadrant_predictions is not None:
        common["dev"], probe = attach_probe(common["dev"], v_quadrant_predictions)
        records.append({"path": probe["path"], "sha256": probe["sha256"]})
    normalization = fit_normalization(common["train"])
    assert_unchanged(records)
    output = reserve_output(output_dir, "cache")
    for split in SPLITS:
        np.savez(output / f"{split}_common.npz", **common[split], split=np.array(split))
    np.savez(output / "normalization.npz", **normalization)
    manifest = {"schema_version": CACHE_SCHEMA, "status": "complete", "synthetic": synthetic,
                "splits": list(SPLITS), "label_names": list(LABEL_NAMES),
                "source_ta": {"model": "TA-v0", "seed": 100, "best_epoch": 18, "dim": 1500,
                              "identity_evidence": "caller-declared; NPZ alone cannot verify source checkpoint"},
                "source_v": V_SOURCE, "v_quadrant_predictions": probe, "joins": audits,
                "normalization": {"fit_split": "train", "fit_subset": "common_v_valid", "ddof": 0,
                                  "epsilon": EPSILON, "denominator": "max(population_std, epsilon)",
                                  "fit_count": len(common["train"]["label"]),
                                  "fit_ids_sha256": id_digest(common["train"]["utterance_id"])},
                "inputs": records, "test_access": False,
                "artifacts": {name: compute_file_sha256(output / name) for name in (
                    "train_common.npz", "dev_common.npz", "normalization.npz")}}
    # The complete manifest is published last; partial caches cannot be trained.
    write_json(output / "cache_manifest.json", manifest)
    return manifest


def load_cache(root, *, synthetic=False):
    root = guard_input(root).resolve()
    manifest_path = direct_file(root, "cache_manifest.json")
    manifest_identity = identity(manifest_path)
    manifest = read_json(manifest_path)
    ta_source = manifest.get("source_ta", {})
    if (manifest.get("schema_version") != CACHE_SCHEMA or manifest.get("status") != "complete"
            or manifest.get("synthetic") is not synthetic or manifest.get("splits") != list(SPLITS)
            or manifest.get("label_names") != list(LABEL_NAMES) or manifest.get("test_access") is not False
            or manifest.get("source_v") != V_SOURCE
            or any(ta_source.get(key) != expected for key, expected in (
                ("model", "TA-v0"), ("seed", 100), ("best_epoch", 18), ("dim", 1500)))):
        raise ValueError("Cache schema/split/source/evidence-mode mismatch; Test is forbidden")
    expected_names = {"train_common.npz", "dev_common.npz", "normalization.npz"}
    if set(manifest.get("artifacts", {})) != expected_names:
        raise ValueError("Cache must contain exactly Train/Dev/normalization artifacts")
    records = [manifest_identity]
    for name in sorted(expected_names):
        record = identity(direct_file(root, name))
        if record["sha256"] != manifest["artifacts"][name]:
            raise ValueError("Cache artifact SHA256 mismatch")
        records.append(record)
    common = {}
    for split in SPLITS:
        with np.load(root / f"{split}_common.npz", allow_pickle=False) as archive:
            required = {"utterance_id", "dialogue_id", "label", "h_ta", "h_v", "ta_logits", "ta_prob", "split"}
            optional = {"v_probe_pred", "v_probe_prob"} if split == "dev" else set()
            if not required <= set(archive.files) or set(archive.files) - required - optional:
                raise ValueError("Unexpected cache fields; Test or extra split data is forbidden")
            if archive["split"].shape != () or archive["split"].item() != split:
                raise ValueError("Cache split mismatch; Test is forbidden")
            common[split] = validate_common(dict(archive))
        data = common[split]
        if "h_v" not in data or list(data["utterance_id"]) != sorted(data["utterance_id"]):
            raise ValueError("Cache requires sorted common V-valid features")
        audit = manifest["joins"][split]
        if (audit["common_rows"] != len(data["label"]) or audit["v_rows"] != len(data["label"])
                or audit["ta_rows"] != len(data["label"]) + audit["ta_only_count"]
                or audit["ta_only_count"] != len(audit["ta_only_ids"])
                or any(audit[key] != 0 for key in ("duplicate_ids", "label_mismatch", "unknown_v_exclusion"))):
            raise ValueError("Cache join audit mismatch")
        if not synthetic and (audit["ta_rows"], audit["common_rows"], audit["ta_only_count"]) != EXPECTED[split]:
            raise ValueError("Frozen real-data coverage mismatch")
    if set(common["train"]["utterance_id"]) & set(common["dev"]["utterance_id"]):
        raise ValueError("Cache Train/Dev overlap")
    if set(common["train"]["dialogue_id"]) & set(common["dev"]["dialogue_id"]):
        raise ValueError("Cache Train/Dev dialogue overlap")
    with np.load(root / "normalization.npz", allow_pickle=False) as archive:
        normalization = dict(archive)
    expected_stats = fit_normalization(common["train"])
    if set(normalization) != set(expected_stats) or any(
            not np.array_equal(normalization[key], value) for key, value in expected_stats.items()):
        raise ValueError("Normalization must equal Train-common-only statistics")
    contract = manifest["normalization"]
    if (contract.get("fit_split") != "train" or contract.get("fit_subset") != "common_v_valid"
            or contract.get("ddof") != 0 or contract.get("epsilon") != EPSILON
            or contract.get("fit_count") != len(common["train"]["label"])
            or contract.get("fit_ids_sha256") != id_digest(common["train"]["utterance_id"])):
        raise ValueError("Invalid Train-only normalization contract")
    if ("v_probe_pred" in common["dev"]) != (manifest.get("v_quadrant_predictions") is not None):
        raise ValueError("Linear-B provenance mismatch")
    probe = manifest.get("v_quadrant_predictions")
    if probe is not None and probe.get("source") != V_PRED_SOURCE:
        raise ValueError("Quadrant source must be frozen Linear-B seed100")
    assert_unchanged(records)
    return common, normalization, manifest, records

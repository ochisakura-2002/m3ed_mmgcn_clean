"""Export an exact A1582 bridge; canonical identity/splits remain authoritative.

Only explicitly supplied trusted PKLs are supported. Historical item 0 provides
row counts/locations and item 4 provides audio. No targets or V/T arrays are read.
"""

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import pickle
import sys
import uuid

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from datasets.iemocap.ta_v0_dataset import (
    SCHEMA, UID, integer, read_columns, require, unique_rows,
)
from datasets.iemocap.tav_full_audit_dataset import COUNTS, VERSION, fingerprint
from scripts.runtime.multidag_cl_ta_v0.formal import git_identity, sha256_file

IDENTITY_FIELDS = ("utterance_id", "dialogue_id", "ta_dialogue_id", "ta_row_index", "split", "speaker")
BRIDGE_FIELDS = ("utterance_id", "historical_dialogue_id", "historical_row_index", "audio_sha256",
                 "canonical_split", "historical_split")
ALGORITHM = "SHA256(contiguous little-endian float32 A1582 model-input bytes)"
SOURCE_FILES = {
    "exporter": Path(__file__),
    "builder": ROOT / "scripts/data/build/multidag_cl/build_tav_t1024_v1.py",
    "fingerprint": ROOT / "datasets/iemocap/tav_full_audit_dataset.py",
    "canonical_helpers": ROOT / "datasets/iemocap/ta_v0_dataset.py",
    "provenance_helpers": ROOT / "scripts/runtime/multidag_cl_ta_v0/formal.py",
}


def split_membership(splits, dialogue_ids, name):
    require(set(splits) == set(COUNTS), f"{name}: explicit train/dev/test membership required")
    membership = {}
    for split in COUNTS:
        require(isinstance(splits[split], (list, tuple)), f"{name}: split IDs must be a list")
        for did in splits[split]:
            require(isinstance(did, str) and bool(did) and did not in membership,
                    f"{name}: duplicate/cross-split dialogue")
            membership[did] = split
    require(set(membership) == set(dialogue_ids), f"{name}: split/dialogue ID sets differ")
    return membership


def canonical_records(current, model_rows, order_rows, *, expected_counts):
    """Validate only non-target metadata and A1582, including Test identity/audio."""
    require(current.get("schema") == SCHEMA and current.get("dataset_version") == VERSION,
            "explicit canonical TA package schema/version required")
    require(current["feature_spec"]["text"]["dimension"] == 1024
            and current["feature_spec"]["audio_ta"]["dimension"] == 1582,
            "canonical T1024/A1582 metadata required")
    model = unique_rows(model_rows, IDENTITY_FIELDS, "model manifest")
    orders = unique_rows(order_rows, (*IDENTITY_FIELDS, "canonical_dialogue_order"), "order manifest")
    total = sum(value["utterances"] for value in expected_counts.values())
    require(len(model) == len(orders) == total and set(model) == set(orders),
            "canonical manifest counts/ID sets differ from required universe")
    membership = split_membership(current["splits"], current["dialogues"], "canonical")
    ranks = set()
    for uid, row in model.items():
        order = orders[uid]
        for key in IDENTITY_FIELDS:
            left, right = row[key], order[key]
            if key == "ta_row_index":
                left, right = integer(left, key), integer(right, key)
            require(left == right, f"canonical manifest {key} conflict: {uid}")
        match = UID.fullmatch(uid)
        require(match is not None and match[1] == row["dialogue_id"] and match[2] == row["speaker"],
                f"canonical ID/speaker conflict: {uid}")
        rank = (row["dialogue_id"], integer(order["canonical_dialogue_order"], "canonical_dialogue_order"))
        require(rank[1] > 0 and rank not in ranks, "duplicate/invalid canonical order")
        ranks.add(rank)
    seen, model_dialogues, records = set(), set(), {}
    counts = {split: {"utterances": 0, "dialogues": 0} for split in COUNTS}
    for did, dialogue in current["dialogues"].items():
        ids, audio = dialogue["utterance_ids"], dialogue["audio_features"]
        require(len(ids) > 0 and len(dialogue["speakers"]) == len(ids), "canonical identity lengths mismatch")
        require(isinstance(audio, np.ndarray) and audio.dtype == np.float32
                and audio.shape == (len(ids), 1582) and np.isfinite(audio).all(),
                "canonical audio requires finite float32 A1582")
        split = membership[did]
        group = set()
        for index, uid in enumerate(ids):
            require(uid not in seen and uid in model, "duplicate/unmatched canonical utterance ID")
            seen.add(uid)
            row = model[uid]
            require(row["ta_dialogue_id"] == did and integer(row["ta_row_index"], "ta_row_index") == index
                    and row["split"] == split and row["speaker"] == dialogue["speakers"][index],
                    "canonical asset/manifest identity/row/split conflict")
            group.add(row["dialogue_id"])
            records[uid] = {"utterance_id": uid, "canonical_split": split, "audio_sha256": fingerprint(audio[index])}
        require(len(group) == 1 and not (group & model_dialogues), "canonical dialogue mapping is not one-to-one")
        model_dialogues.update(group)
        counts[split]["utterances"] += len(ids)
        counts[split]["dialogues"] += 1
    require(seen == set(model), "canonical asset/manifest ID sets differ")
    require(counts == expected_counts, f"canonical split counts mismatch: {counts}")
    # Match the existing builder: model-manifest order, independent of PKL dict
    # insertion order, historical row order, and order-manifest CSV row order.
    return [records[row["utterance_id"]] for row in model_rows], counts


def historical_records(historical, historical_splits):
    """Never index historical targets, speakers, text, visual, or embedded splits."""
    require(isinstance(historical, (tuple, list)) and len(historical) == 9, "historical nine-item PKL required")
    ids, arrays = historical[0], historical[4]
    require(isinstance(ids, dict) and isinstance(arrays, dict) and set(ids) == set(arrays),
            "historical identity/audio ID sets differ")
    membership = split_membership(
        {split: historical_splits[f"{split}_dialogue_ids"] for split in COUNTS}, ids, "historical")
    records = []
    counts = {split: {"utterances": 0, "dialogues": 0} for split in COUNTS}
    for did, row_ids in ids.items():
        audio = np.asarray(arrays[did])
        require(len(row_ids) > 0 and audio.shape == (len(row_ids), 1582)
                and audio.dtype.kind == "f" and audio.dtype.itemsize in (4, 8) and np.isfinite(audio).all(),
                "historical audio requires finite float32/float64 A1582")
        # Guard float64 -> float32 overflow without redefining fingerprint bytes.
        with np.errstate(over="ignore", invalid="ignore"):
            require(np.isfinite(np.asarray(audio, dtype="<f4")).all(), "nonfinite historical float32 model-input audio")
        for index in range(len(row_ids)):
            records.append({"historical_dialogue_id": did, "historical_row_index": index,
                            "historical_split": membership[did], "audio_sha256": fingerprint(audio[index])})
        counts[membership[did]]["utterances"] += len(row_ids)
        counts[membership[did]]["dialogues"] += 1
    return records, counts


def match_records(current, historical):
    """Audit the entire bijection before accepting any bridge row."""
    by_hash = defaultdict(list)
    for row in historical:
        by_hash[row["audio_sha256"]].append(row)
    current_hashes = Counter(row["audio_sha256"] for row in current)
    qc = {"canonical_utterances": len(current), "historical_utterances": len(historical),
          "historical_matched": 0, "unmatched": 0, "ambiguous": 0,
          "duplicate_current_fingerprint": sum(n - 1 for n in current_hashes.values()),
          "duplicate_historical_fingerprint": sum(len(rows) - 1 for rows in by_hash.values()),
          "multiple_historical_rows_for_one_current_utterance": 0, "reused_historical_rows": 0,
          "duplicate_historical_location": 0, "unused_historical_rows": 0, "split_conflict": 0}
    locations = [(row["historical_dialogue_id"], row["historical_row_index"]) for row in historical]
    qc["duplicate_historical_location"] = len(locations) - len(set(locations))
    used, bridge = set(), []
    for row in current:
        candidates = by_hash.get(row["audio_sha256"], ())
        if not candidates:
            qc["unmatched"] += 1
            continue
        if len(candidates) != 1:
            qc["ambiguous"] += 1
            qc["multiple_historical_rows_for_one_current_utterance"] += 1
            continue
        link = candidates[0]
        location = (link["historical_dialogue_id"], link["historical_row_index"])
        if location in used:
            qc["reused_historical_rows"] += 1
        used.add(location)
        qc["historical_matched"] += 1
        qc["split_conflict"] += int(row["canonical_split"] != link["historical_split"])
        bridge.append(dict(row, **{key: link[key] for key in
                                  ("historical_dialogue_id", "historical_row_index", "historical_split")}))
    qc["unused_historical_rows"] = len(set(locations) - used)
    failures = {key: value for key, value in qc.items()
                if key not in ("canonical_utterances", "historical_utterances", "historical_matched") and value}
    require(not failures and len(current) == len(historical) == len(used) == qc["historical_matched"],
            "audio bridge FAIL CLOSED: " + json.dumps(qc, sort_keys=True))
    return bridge, qc


def align(current, model_rows, order_rows, historical, historical_splits, *, expected_counts=COUNTS):
    """In-memory count override is for small synthetic fixtures only; no CLI override."""
    canonical, canonical_counts = canonical_records(current, model_rows, order_rows, expected_counts=expected_counts)
    history, history_counts = historical_records(historical, historical_splits)
    bridge, qc = match_records(canonical, history)
    require(history_counts == expected_counts, f"historical split counts mismatch: {history_counts}")
    qc.update(canonical_dialogues=sum(value["dialogues"] for value in canonical_counts.values()),
              historical_dialogues=sum(value["dialogues"] for value in history_counts.values()),
              canonical_split_counts=canonical_counts, historical_split_counts=history_counts)
    return bridge, qc


def csv_bytes(rows):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=BRIDGE_FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def publish(output, audit_path, bridge_data, audit_data):
    """Exclusive atomic publication on one filesystem; CSV is published last.

    Validation failures never reach here. On an ordinary I/O failure remove only
    our own newly published links. A process crash can leave an audit without CSV;
    it cannot publish a partial CSV. Existing destinations are never overwritten.
    """
    output.parent.mkdir(parents=True, exist_ok=True)
    staged, published = [], []
    try:
        for target, data in ((audit_path, audit_data), (output, bridge_data)):
            temporary = output.parent / (".audio_bridge_" + uuid.uuid4().hex + ".tmp")
            with temporary.open("xb") as stream:
                staged.append((temporary, target))
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
        for temporary, target in staged:
            os.link(temporary, target)  # Atomic and exclusive; no overwrite race.
            published.append((temporary, target))
    except BaseException:
        for temporary, target in reversed(published):
            if target.exists() and target.samefile(temporary):
                target.unlink()
        raise
    finally:
        for temporary, _ in staged:
            temporary.unlink()


def build(*, canonical_asset, model_manifest, order_manifest, historical_asset, historical_split_manifest, output):
    inputs = {"canonical_asset": Path(canonical_asset), "model_manifest": Path(model_manifest),
              "order_manifest": Path(order_manifest), "historical_asset": Path(historical_asset),
              "historical_split_manifest": Path(historical_split_manifest)}
    output = Path(output)
    require(output.suffix == ".csv", "output must be an explicit .csv path")
    audit_path = output.with_suffix(".audit.json")
    require(not ({output.resolve(), audit_path.resolve()} & {path.resolve() for path in inputs.values()}),
            "output would overwrite an input")
    if any(os.path.lexists(path) for path in (output, audit_path)):
        raise FileExistsError("bridge/audit destination must be new; no overwrite")
    hashes = {key: sha256_file(path) for key, path in inputs.items()}
    code_hashes = {key: sha256_file(path) for key, path in SOURCE_FILES.items()}
    splits = json.loads(inputs["historical_split_manifest"].read_text(encoding="utf-8-sig"))
    require(splits["project_pkl_sha256"] == hashes["historical_asset"], "historical asset/split SHA256 mismatch")
    with inputs["canonical_asset"].open("rb") as stream:
        current = pickle.load(stream)
    with inputs["historical_asset"].open("rb") as stream:
        historical = pickle.load(stream)
    rows, qc = align(current, read_columns(inputs["model_manifest"], IDENTITY_FIELDS),
                     read_columns(inputs["order_manifest"], (*IDENTITY_FIELDS, "canonical_dialogue_order")),
                     historical, splits)
    data = csv_bytes(rows)
    audit = {"schema": "audio_fingerprint_bridge_v1", "status": "complete", "alignment": qc,
             "fingerprint_algorithm": ALGORITHM, "historical_row_index_base": 0,
             "split_authority": "current canonical asset, cross-checked against both current manifests",
             "historical_split_role": "cross-split conflict audit only",
             "output_order": "model manifest CSV row order", "bridge_rows": len(rows),
             "sources": {key: {"filename": path.name, "sha256": hashes[key]} for key, path in inputs.items()},
             "source_code": {key: {"path": path.relative_to(ROOT).as_posix(), "sha256": code_hashes[key]}
                             for key, path in SOURCE_FILES.items()},
             "builder_sha256": code_hashes["builder"], "exporter_sha256": code_hashes["exporter"],
             "bridge_sha256": hashlib.sha256(data).hexdigest(), "bridge_filename": output.name,
             "timestamp_utc": datetime.now(timezone.utc).isoformat(), **git_identity(),
             "historical_labels_inspected": 0, "test_labels_inspected": 0,
             "test_loader_built": 0, "test_evaluation_count": 0}
    audit_data = (json.dumps(audit, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    require(hashes == {key: sha256_file(path) for key, path in inputs.items()}, "source changed while exporting")
    require(code_hashes == {key: sha256_file(path) for key, path in SOURCE_FILES.items()},
            "source code changed while exporting")
    publish(output, audit_path, data, audit_data)
    return audit


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("canonical-asset", "model-manifest", "order-manifest", "historical-asset",
                 "historical-split-manifest", "output"):
        parser.add_argument("--" + name, required=True)
    result = build(**vars(parser.parse_args(argv)))
    print(json.dumps({"status": result["status"], "alignment": result["alignment"],
                      "bridge_sha256": result["bridge_sha256"], "test_evaluation_count": 0}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

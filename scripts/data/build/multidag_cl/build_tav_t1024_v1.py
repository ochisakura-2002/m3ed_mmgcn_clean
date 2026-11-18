"""Attach historical V342 by an explicit, revalidated A1582 fingerprint bridge."""

import argparse
import csv
import json
from pathlib import Path
import pickle
import sys

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from datasets.iemocap.ta_v0_dataset import integer, require
from datasets.iemocap.tav_full_audit_dataset import (
    COUNTS, DIMS, MANIFEST_FIELDS, SCHEMA, VERSION, TAVAuditDataset, fingerprint, identity_sha, validate_metadata,
)
from scripts.runtime.multidag_cl_ta_v0.formal import sha256_file

ASSET_NAME = "iemocap_asym_tav_t1024_v1.pkl"
MODEL_NAME = "model_input_manifest_tav_t1024_v1.csv"
PROVENANCE_NAME = "tav_t1024_provenance_manifest_v1.json"


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def align_visual(current, model_rows, order_rows, historical, historical_splits, bridge,
                 *, expected_counts=COUNTS):
    """No historical target access. Current Test target objects pass through untouched.

    historical is the existing nine-item PKL; only indices 0, 4 and 5 are
    consulted. Its separate split manifest is conflict evidence only.
    """
    total = sum(value["utterances"] for value in expected_counts.values())
    require(current.get("schema") == "iemocap_asym_ta_v1" and current.get("dataset_version") == VERSION,
            "explicit current canonical TA T1024 package required")
    require(current["feature_spec"]["text"]["dimension"] == 1024, "current text must be T1024")
    require(isinstance(historical, (tuple, list)) and len(historical) == 9, "historical nine-item asset required")
    require(len(model_rows) == len(order_rows) == len(bridge) == total, "matched row count must equal canonical universe")
    model = {row["utterance_id"]: row for row in model_rows}
    orders = {row["utterance_id"]: row for row in order_rows}
    bridges = {row["utterance_id"]: row for row in bridge}
    require(len(model) == len(orders) == len(bridges) == total and set(model) == set(orders) == set(bridges),
            "duplicate/unmatched canonical bridge IDs")
    history_membership = {}
    for split in COUNTS:
        for did in historical_splits[f"{split}_dialogue_ids"]:
            require(did not in history_membership, "historical split overlap")
            history_membership[did] = split
    require(set(history_membership) == set(historical[0]) == set(historical[4]) == set(historical[5]),
            "historical feature/split ID sets differ")
    h_fingerprints, h_locations = {}, set()
    for did, ids in historical[0].items():
        audio, visual = np.asarray(historical[4][did]), np.asarray(historical[5][did])
        require(audio.shape == (len(ids), 1582) and visual.shape == (len(ids), 342)
                and np.isfinite(audio).all() and np.isfinite(visual).all(), "historical A/V dimension/finite failure")
        for index in range(len(ids)):
            digest = fingerprint(audio[index])
            require(digest not in h_fingerprints, "ambiguous historical A1582 fingerprint")
            h_fingerprints[digest] = (did, index)
            h_locations.add((did, index))
    require(len(h_locations) == total, "historical universe must exactly match canonical universe")
    source_membership = {}
    for split, dids in current["splits"].items():
        for did in dids:
            require(did not in source_membership, "current split overlap")
            source_membership[did] = split
    require(set(source_membership) == set(current["dialogues"]), "current split/dialogue set mismatch")
    dialogues, rows, seen_audio, used_locations, seen_ids = {}, [], set(), set(), set()
    for did, dialogue in current["dialogues"].items():
        ids = dialogue["utterance_ids"]
        split = source_membership[did]
        for key, dim in (("text_features", 1024), ("audio_features", 1582)):
            array = dialogue[key]
            require(isinstance(array, np.ndarray) and array.dtype == np.float32
                    and array.shape == (len(ids), dim) and np.isfinite(array).all(), "current T/A dimensions/dtype")
        visuals = []
        for index, uid in enumerate(ids):
            require(uid in model and uid not in seen_ids, "duplicate/unmatched source ID")
            seen_ids.add(uid)
            row, order, link = model[uid], orders[uid], bridges[uid]
            require(row["ta_dialogue_id"] == order["ta_dialogue_id"] == did
                    and integer(row["ta_row_index"], "ta_row_index") == integer(order["ta_row_index"], "ta_row_index") == index
                    and row["dialogue_id"] == order["dialogue_id"]
                    and row["split"] == order["split"] == split
                    and row["speaker"] == order["speaker"] == dialogue["speakers"][index], "canonical identity/row/split conflict")
            digest = fingerprint(dialogue["audio_features"][index])
            require(digest not in seen_audio, "ambiguous current A1582 fingerprint")
            seen_audio.add(digest)
            location = (link["historical_dialogue_id"], integer(link["historical_row_index"], "historical_row_index"))
            require(link["audio_sha256"] == digest and h_fingerprints.get(digest) == location,
                    "unmatched/incorrect fingerprint bridge")
            require(location not in used_locations, "ambiguous reused historical row")
            used_locations.add(location)
            require(history_membership[location[0]] == split, "split_conflict: historical/current membership")
            visual = np.asarray(historical[5][location[0]][location[1]], dtype=np.float32)
            require(np.isfinite(visual).all(), "nonfinite V342 after float32 conversion")
            visuals.append(visual)
            target = {"utterance_id": uid, "dialogue_id": row["dialogue_id"], "split": split,
                      "speaker": row["speaker"], "canonical_dialogue_order": order["canonical_dialogue_order"],
                      "asset_dialogue_id": did, "asset_row_index": index,
                      "benchmark_label": "", "benchmark_label_id": "", "audio_sha256": digest}
            if split in ("train", "dev"):
                target.update(benchmark_label=dialogue["benchmark_labels"][index],
                              benchmark_label_id=dialogue["benchmark_label_ids"][index])
                require(str(target["benchmark_label_id"]) == str(row["benchmark_label_id"]) == str(order["benchmark_label_id"])
                        and target["benchmark_label"] == row["benchmark_label"] == order["benchmark_label"],
                        "current Train/Dev label provenance conflict")
            rows.append(target)
        # Shallow metadata copy is intentional: Test targets are opaque objects,
        # neither indexed nor compared, and come exclusively from current data.
        dialogues[did] = dict(dialogue, visual_features=np.stack(visuals))
    require(seen_ids == set(model) and used_locations == h_locations, "unmatched mapping rows")
    # Preserve the current manifest's row order, independently of PKL dict order.
    by_uid = {row["utterance_id"]: row for row in rows}
    rows = [by_uid[row["utterance_id"]] for row in model_rows]
    asset = dict(current, schema=SCHEMA, dialogues=dialogues,
                 feature_spec={key: {"dimension": dim, "dtype": "float32"} for key, dim in DIMS.items()})
    validate_metadata(asset, rows, expected_counts=expected_counts)
    for split in ("train", "dev"):
        TAVAuditDataset(asset, rows, split=split, expected_counts=expected_counts)
    return asset, rows, {"matched": total, "unmatched": 0, "ambiguous": 0, "split_conflict": 0}


def build(*, canonical_asset, model_manifest, order_manifest, historical_asset, historical_split_manifest,
          fingerprint_bridge, destination):
    inputs = {"canonical_asset": Path(canonical_asset), "model_manifest": Path(model_manifest),
              "order_manifest": Path(order_manifest), "historical_asset": Path(historical_asset),
              "historical_split_manifest": Path(historical_split_manifest), "fingerprint_bridge": Path(fingerprint_bridge)}
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError("asset destination must be a new directory")
    hashes = {key: sha256_file(path) for key, path in inputs.items()}
    with inputs["canonical_asset"].open("rb") as stream:
        current = pickle.load(stream)
    with inputs["historical_asset"].open("rb") as stream:
        historical = pickle.load(stream)
    splits = json.loads(inputs["historical_split_manifest"].read_text(encoding="utf-8"))
    require(splits["project_pkl_sha256"] == hashes["historical_asset"], "historical asset/split SHA256 mismatch")
    asset, rows, qc = align_visual(current, read_csv(inputs["model_manifest"]), read_csv(inputs["order_manifest"]),
                                   historical, splits, read_csv(inputs["fingerprint_bridge"]))
    require(hashes == {key: sha256_file(path) for key, path in inputs.items()}, "source changed while building")
    destination.mkdir(parents=True, exist_ok=False)
    try:
        with (destination / ASSET_NAME).open("xb") as stream:
            pickle.dump(asset, stream, protocol=pickle.HIGHEST_PROTOCOL)
        with (destination / MODEL_NAME).open("x", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=MANIFEST_FIELDS, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        provenance = {"schema": SCHEMA, "status": "complete", "split_counts": COUNTS, "alignment": qc,
                      "feature_dimensions": {"text": 1024, "audio": 1582, "visual": 342},
                      "sources": {key: {"filename": path.name, "sha256": hashes[key]} for key, path in inputs.items()},
                      "mapping_method": "explicit_bridge_revalidated_unique_float32_little_endian_A1582_sha256",
                      "label_provenance": "current_canonical_asset_only; Test target objects copied opaquely",
                      "split_provenance": "current_canonical_asset_and_current_manifests_only",
                      "row_order_identity_sha256": identity_sha(rows),
                      "canonical_order": "current provenance canonical_dialogue_order; no renumbering",
                      "test_labels_inspected": 0, "test_loader_built": 0, "test_evaluation_count": 0,
                      "asset_sha256": sha256_file(destination / ASSET_NAME),
                      "model_manifest_sha256": sha256_file(destination / MODEL_NAME),
                      "builder_sha256": sha256_file(Path(__file__))}
        with (destination / PROVENANCE_NAME).open("x", encoding="utf-8") as stream:
            json.dump(provenance, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
    except Exception:
        # Never erase partial evidence; it cannot be mistaken for a complete
        # asset because the complete provenance sidecar is written last.
        raise
    return provenance


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("canonical-asset", "model-manifest", "order-manifest", "historical-asset",
                 "historical-split-manifest", "fingerprint-bridge", "destination"):
        parser.add_argument("--" + name, required=True)
    result = build(**vars(parser.parse_args(argv)))
    print(json.dumps({"status": result["status"], "alignment": result["alignment"], "test_evaluation_count": 0}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Build a certified Train/Dev-only projection of a trusted canonical TAV package."""

import argparse
import csv
import os
from pathlib import Path
import pickle
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from datasets.iemocap.tav_full_audit_dataset import COUNTS, DIMS, IDENTITY_FIELDS, MANIFEST_FIELDS, SCHEMA, identity_sha
from scripts.analysis.models.multidag_cl.pairwise_preflight_inputs import (
    ASSET, FEATURE_DIMENSIONS, HOLD, MODEL_MANIFEST, PROJECTION_MANIFEST, PROJECTION_SCHEMA, SPLITS,
    load_projected_train_dev, read_json, validate_hold,
)
from scripts.dev.check_workstream import check_workstream
from scripts.runtime.multidag_cl_ta_v0.formal import sha256_file, write_json_atomic


ASSET_METADATA = ("schema", "dataset_version", "feature_spec", "label_vocab")
DIALOGUE_FIELDS = ("utterance_ids", "speakers", "benchmark_labels", "benchmark_label_ids", *DIMS)
PACKAGE_FILES = (ASSET, MODEL_MANIFEST, PROJECTION_MANIFEST)


def read_model_manifest(path):
    """Retain source column/row order. For Test retain only identity-hash metadata."""
    selected, identity_rows = [], []
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream, strict=True)
        fields = next(reader, None)
        if not fields or len(set(fields)) != len(fields) or not set(MANIFEST_FIELDS) <= set(fields):
            raise ValueError("missing or duplicate canonical model manifest columns")
        positions = {key: fields.index(key) for key in IDENTITY_FIELDS}
        for values in reader:
            if not values:
                continue
            if len(values) != len(fields):
                raise ValueError("canonical manifest row has a different column count from its header")
            identity = {key: values[index] for key, index in positions.items()}
            split = identity["split"]
            if split not in COUNTS:
                raise ValueError("unknown canonical manifest split")
            identity_rows.append(identity)
            if split in SPLITS:
                selected.append(dict(zip(fields, values)))
    return fields, selected, identity_sha(identity_rows)


def canonical_identity(provenance, hashes, row_order_sha):
    """Assemble the same data_identity saved by the canonical TAV runtime."""
    validate_hold(provenance, legacy=True)
    alignment = {"matched": sum(count["utterances"] for count in COUNTS.values()),
                 "unmatched": 0, "ambiguous": 0, "split_conflict": 0}
    expected = {"schema": SCHEMA, "status": "complete", "split_counts": COUNTS, "alignment": alignment,
                "feature_dimensions": FEATURE_DIMENSIONS, "row_order_identity_sha256": row_order_sha,
                "asset_sha256": hashes["asset"], "model_manifest_sha256": hashes["model_manifest"],
                "test_labels_inspected": 0}
    if any(provenance.get(key) != value for key, value in expected.items()):
        raise ValueError("canonical TAV source provenance, row order or SHA-256 mismatch")
    return {"schema": SCHEMA, "sha256": hashes, "row_order_identity_sha256": row_order_sha,
            "split_counts": COUNTS, "feature_dimensions": FEATURE_DIMENSIONS}


def project_asset(asset):
    splits, dialogues = asset["splits"], asset["dialogues"]
    if set(splits) != set(COUNTS):
        raise ValueError("canonical pickle must contain the frozen split keys")
    all_ids = [did for ids in splits.values() for did in ids]
    if len(all_ids) != len(set(all_ids)) or set(all_ids) != set(dialogues):
        raise ValueError("canonical split membership overlaps or differs from the dialogue keys")
    if any(len(splits[split]) != COUNTS[split]["dialogues"] for split in SPLITS):
        raise ValueError("canonical Train/Dev dialogue count mismatch")
    projected_splits = {split: ids for split, ids in splits.items() if split in SPLITS}
    keep = {did for ids in projected_splits.values() for did in ids}
    return {**{key: asset[key] for key in ASSET_METADATA}, "splits": projected_splits,
            "dialogues": {did: {field: dialogues[did][field] for field in DIALOGUE_FIELDS}
                          for did in dialogues if did in keep}}


def publish(staging, destination):
    """Reserve a new directory and publish the certification sidecar last."""
    destination.mkdir(exist_ok=False)
    published = []
    try:
        for name in PACKAGE_FILES:
            target = destination / name
            os.link(staging / name, target)
            published.append(target)
    except BaseException:
        for path in reversed(published):
            path.unlink()
        destination.rmdir()
        raise


def build(*, asset_path, model_manifest_path, provenance_manifest_path, output_dir):
    output = Path(output_dir)
    if output.exists() or output.is_symlink():
        raise FileExistsError("projection output directory already exists")
    output = output.resolve()
    paths = {"asset": Path(asset_path).resolve(strict=True),
             "model_manifest": Path(model_manifest_path).resolve(strict=True),
             "provenance_manifest": Path(provenance_manifest_path).resolve(strict=True)}
    if any(path.parent in (output, *output.parents) for path in paths.values()):
        raise ValueError("projection output must be outside the canonical source directories")

    before = {key: sha256_file(path) for key, path in paths.items()}
    fields, rows, row_order_sha = read_model_manifest(paths["model_manifest"])
    source = canonical_identity(read_json(paths["provenance_manifest"]), before, row_order_sha)
    # A monolithic pickle is deserialized once. Test dialogue values are never
    # fetched by project_asset. Only the split membership and dialogue keys are used.
    with paths["asset"].open("rb") as stream:
        projected = project_asset(pickle.load(stream))

    split_counts = {split: {"dialogues": len(projected["splits"][split]),
                            "utterances": sum(row["split"] == split for row in rows)} for split in SPLITS}
    if split_counts != {split: COUNTS[split] for split in SPLITS}:
        raise ValueError("canonical Train/Dev population count mismatch")

    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".train_dev_projection_", dir=output.parent) as temporary:
        stage = Path(temporary)
        with (stage / ASSET).open("xb") as stream:
            pickle.dump(projected, stream, protocol=pickle.HIGHEST_PROTOCOL)
        with (stage / MODEL_MANIFEST).open("x", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        provenance = {**HOLD, "schema": PROJECTION_SCHEMA, "splits": list(SPLITS), "split_counts": split_counts,
                      "source_data_identity": source, "row_order_identity_sha256": identity_sha(rows),
                      "sha256": {name: sha256_file(stage / name) for name in (ASSET, MODEL_MANIFEST)}}
        write_json_atomic(stage / PROJECTION_MANIFEST, provenance)
        # Exercise the actual consumer. It checks only Train/Dev payloads and
        # validates raw row indices, labels, speakers, fingerprints and shapes.
        load_projected_train_dev(stage, SimpleNamespace(manifest={"data_identity": source}))
        if before != {key: sha256_file(path) for key, path in paths.items()}:
            raise ValueError("canonical source SHA-256 changed during projection")
        publish(stage, output)
    return provenance


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("asset-path", "model-manifest-path", "provenance-manifest-path", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args(argv)
    errors = check_workstream("multidag")
    if errors:
        print("WORKSTREAM_CHECK = FAIL", flush=True)
        parser.error("\n".join(errors))
    print("WORKSTREAM_CHECK = PASS", flush=True)
    if not sys.platform.startswith("linux"):
        parser.error("formal data projection must run on Ubuntu; use synthetic unit tests on Windows")
    try:
        result = build(**vars(args))
    except FileExistsError as error:
        print(f"BUILD_STATUS = REFUSE\nREASON = {error}", file=sys.stderr)
        return 2
    except (OSError, ValueError, KeyError, TypeError, csv.Error, pickle.PickleError, EOFError) as error:
        print(f"BUILD_STATUS = INVALID\nREASON = {error}", file=sys.stderr)
        return 2
    print("BUILD_STATUS = PASS")
    for split, counts in result["split_counts"].items():
        print(f"{split.upper()} = {counts['dialogues']} dialogues / {counts['utterances']} utterances")
    print("SOURCE_IMMUTABILITY = PASS\nPAIRWISE_PREFLIGHT_VALIDATOR_COMPATIBLE = YES\nTEST_TOUCHED = NO")
    print(f"TAV_TRAIN_DEV_PACKAGE = {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

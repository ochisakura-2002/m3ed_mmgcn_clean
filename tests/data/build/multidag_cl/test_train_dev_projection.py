import csv
from pathlib import Path
import pickle
from types import SimpleNamespace

import numpy as np
import pytest

from datasets.iemocap.tav_full_audit_dataset import DIMS, MANIFEST_FIELDS, SCHEMA, VERSION, fingerprint, identity_sha
from scripts.analysis.models.multidag_cl import pairwise_preflight_inputs as preflight
from scripts.data.build.multidag_cl import build_train_dev_projection as builder
from scripts.data.build.multidag_cl.build_tav_t1024_v1 import ASSET_NAME, MODEL_NAME, PROVENANCE_NAME
from scripts.runtime.multidag_cl_ta_v0.formal import sha256_file, write_json_atomic


class OpaqueTestDialogue:
    def __getitem__(self, key):
        raise AssertionError(f"Test payload was accessed: {key}")

    def __iter__(self):
        raise AssertionError("Test payload was iterated")


def read_rows(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        return reader.fieldnames, list(reader)


def write_rows(path, fields, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


@pytest.fixture
def canonical(tmp_path, monkeypatch):
    directory = tmp_path / "canonical"
    directory.mkdir()
    paths = {"asset": directory / ASSET_NAME, "model_manifest": directory / MODEL_NAME,
             "provenance_manifest": directory / PROVENANCE_NAME}
    counts = {"train": {"dialogues": 2, "utterances": 4}, "dev": {"dialogues": 1, "utterances": 2},
              "test": {"dialogues": 1, "utterances": 1}}
    monkeypatch.setattr(builder, "COUNTS", counts)
    monkeypatch.setattr(preflight, "COUNTS", counts)
    monkeypatch.setattr(preflight, "TRAIN_DEV_COUNTS", {split: counts[split] for split in preflight.SPLITS})

    train_a, train_b, dev, held = "Ses01F_impro01", "Ses02F_impro01", "Ses04F_impro01", "Ses05F_impro01"
    splits = {"train": [train_b, train_a], "dev": [dev], "test": [held]}
    dialogues = {}
    rows = []
    for i, (did, split) in enumerate(((train_a, "train"), (held, "test"), (dev, "dev"), (train_b, "train"))):
        if split == "test":
            dialogues[did] = OpaqueTestDialogue()
            rows.append({"utterance_id": f"{did}_F000", "dialogue_id": did, "split": split, "speaker": "F",
                         "canonical_dialogue_order": "1", "asset_dialogue_id": did, "asset_row_index": "0",
                         "benchmark_label": "PRIVATE_TEST_TARGET", "benchmark_label_id": "PRIVATE_TEST_ID",
                         "audio_sha256": "PRIVATE_TEST_FEATURE_HASH", "source_note": "PRIVATE_TEST_NOTE"})
            continue
        ids = (f"{did}_F002", f"{did}_M001")
        labels = [i, i + 1]
        rng = np.random.default_rng(100 + i)
        dialogue = {"utterance_ids": ids, "speakers": ("F", "M"), "benchmark_label_ids": labels,
                    "benchmark_labels": [preflight.CLASS_ORDER[label] for label in labels],
                    **{name: rng.standard_normal((2, dim), dtype=np.float32) for name, dim in DIMS.items()}}
        dialogue["visual_features"][0, 0] = -0.0
        dialogues[did] = dialogue
        for position, uid in enumerate(ids):
            rows.append({"utterance_id": uid, "dialogue_id": did, "split": split, "speaker": dialogue["speakers"][position],
                         "canonical_dialogue_order": f"0{2 - position}", "asset_dialogue_id": did,
                         "asset_row_index": str(position), "benchmark_label": dialogue["benchmark_labels"][position],
                         "benchmark_label_id": str(labels[position]),
                         "audio_sha256": fingerprint(dialogue["audio_features"][position]),
                         "source_note": '001, "kept"\n原字段'})
    asset = {"schema": SCHEMA, "dataset_version": VERSION,
             "feature_spec": {name: {"dimension": dim, "dtype": "float32"} for name, dim in DIMS.items()},
             "label_vocab": {"itos": list(preflight.CLASS_ORDER)}, "splits": splits, "dialogues": dialogues}
    paths["asset"].write_bytes(pickle.dumps(asset, protocol=pickle.HIGHEST_PROTOCOL))
    fields = ["source_note", *MANIFEST_FIELDS]
    write_rows(paths["model_manifest"], fields, rows)
    provenance = {"schema": SCHEMA, "status": "complete", "split_counts": counts,
                  "feature_dimensions": preflight.FEATURE_DIMENSIONS,
                  "alignment": {"matched": len(rows), "unmatched": 0, "ambiguous": 0, "split_conflict": 0},
                  "row_order_identity_sha256": identity_sha(rows), "test_labels_inspected": 0,
                  "test_loader_built": 0, "test_evaluation_count": 0,
                  "asset_sha256": sha256_file(paths["asset"]), "model_manifest_sha256": sha256_file(paths["model_manifest"])}
    write_json_atomic(paths["provenance_manifest"], provenance)
    return SimpleNamespace(paths=paths, fields=fields, rows=rows, asset=asset, provenance=provenance,
                           output=tmp_path / "projection")


def build_projection(package, output=None):
    return builder.build(asset_path=package.paths["asset"], model_manifest_path=package.paths["model_manifest"],
                         provenance_manifest_path=package.paths["provenance_manifest"],
                         output_dir=package.output if output is None else output)


def test_projection_preserves_source_bytes_and_exact_train_dev_content(canonical, monkeypatch):
    before = {key: sha256_file(path) for key, path in canonical.paths.items()}
    opened_splits = []
    actual_dataset = preflight.TAVAuditDataset

    def observe_dataset(*args, split, **kwargs):
        opened_splits.append(split)
        return actual_dataset(*args, split=split, **kwargs)

    monkeypatch.setattr(preflight, "TAVAuditDataset", observe_dataset)
    certificate = build_projection(canonical)
    out = canonical.output
    assert opened_splits == ["train", "dev"]
    assert {path.name for path in out.iterdir()} == set(builder.PACKAGE_FILES)
    assert before == {key: sha256_file(path) for key, path in canonical.paths.items()}
    filtered = [row for row in canonical.rows if row["split"] in preflight.SPLITS]
    assert read_rows(out / preflight.MODEL_MANIFEST) == (canonical.fields, filtered)
    assert "PRIVATE_TEST_" not in (out / preflight.MODEL_MANIFEST).read_text(encoding="utf-8")
    projected = pickle.loads((out / preflight.ASSET).read_bytes())
    assert projected["splits"] == {split: canonical.asset["splits"][split] for split in preflight.SPLITS}
    assert list(projected["dialogues"]) == [did for did in canonical.asset["dialogues"]
                                            if did not in canonical.asset["splits"]["test"]]
    for key in builder.ASSET_METADATA:
        assert projected[key] == canonical.asset[key]
    for did, dialogue in projected["dialogues"].items():
        original = canonical.asset["dialogues"][did]
        for key in ("utterance_ids", "speakers", "benchmark_labels", "benchmark_label_ids"):
            assert dialogue[key] == original[key]
        for name in DIMS:
            before_array, after_array = original[name], dialogue[name]
            assert after_array.dtype == before_array.dtype == np.float32
            assert after_array.shape == before_array.shape
            assert after_array.tobytes() == before_array.tobytes()

    assert certificate["schema"] == preflight.PROJECTION_SCHEMA
    assert certificate["row_order_identity_sha256"] == identity_sha(filtered)
    assert certificate["split_counts"] == preflight.TRAIN_DEV_COUNTS
    assert {key: certificate[key] for key in preflight.HOLD} == preflight.HOLD
    assert certificate["sha256"] == {name: sha256_file(out / name) for name in (preflight.ASSET, preflight.MODEL_MANIFEST)}
    # Supply the canonical runtime identity independently, as a historical run
    # manifest would. Invoke the real verifier again on the published package.
    source_identity = {"schema": SCHEMA, "sha256": before, "row_order_identity_sha256": identity_sha(canonical.rows),
                       "split_counts": preflight.COUNTS, "feature_dimensions": preflight.FEATURE_DIMENSIONS}
    source = SimpleNamespace(manifest={"data_identity": source_identity})
    datasets, verified = preflight.load_projected_train_dev(out, source)
    assert verified == certificate and certificate["source_data_identity"] == source_identity
    assert tuple(datasets["train"].dialogue_ids) == tuple(canonical.asset["splits"]["train"])
    assert opened_splits == ["train", "dev", "train", "dev"]


@pytest.mark.parametrize(("key", "value"), [
    ("schema", "wrong_schema"), ("asset_sha256", "a" * 64), ("model_manifest_sha256", "b" * 64),
    ("row_order_identity_sha256", "c" * 64), ("status", "partial"),
    ("alignment", {"matched": 0}), ("test_labels_inspected", 1),
])
def test_rejects_unverified_source_before_unpickling(canonical, monkeypatch, key, value):
    write_json_atomic(canonical.paths["provenance_manifest"], {**canonical.provenance, key: value})

    def forbidden(*_args, **_kwargs):
        pytest.fail("unverified canonical pickle was deserialized")

    monkeypatch.setattr(builder.pickle, "load", forbidden)
    with pytest.raises(ValueError, match="provenance|Test HOLD"):
        build_projection(canonical)
    assert not canonical.output.exists()


def test_rejects_wrong_train_dev_count_even_with_matching_source_hashes(canonical):
    rows = canonical.rows[1:]
    write_rows(canonical.paths["model_manifest"], canonical.fields, rows)
    write_json_atomic(canonical.paths["provenance_manifest"], {
        **canonical.provenance, "row_order_identity_sha256": identity_sha(rows),
        "model_manifest_sha256": sha256_file(canonical.paths["model_manifest"]),
    })
    with pytest.raises(ValueError, match="population count mismatch"):
        build_projection(canonical)
    assert not canonical.output.exists()


@pytest.mark.parametrize("fault", ["shape", "label", "speaker", "row_index"])
def test_consumer_validation_blocks_inconsistent_payloads(canonical, fault):
    did = canonical.asset["splits"]["train"][0]
    dialogue = canonical.asset["dialogues"][did]
    if fault == "shape":
        dialogue["visual_features"] = dialogue["visual_features"][:, :-1]
    elif fault == "label":
        dialogue["benchmark_label_ids"][0] = 5
    elif fault == "speaker":
        dialogue["speakers"] = ("M", "M")
    else:
        row = next(row for row in canonical.rows if row["asset_dialogue_id"] == did)
        row["asset_row_index"] = "1"
        write_rows(canonical.paths["model_manifest"], canonical.fields, canonical.rows)
    canonical.paths["asset"].write_bytes(pickle.dumps(canonical.asset, protocol=pickle.HIGHEST_PROTOCOL))
    write_json_atomic(canonical.paths["provenance_manifest"], {
        **canonical.provenance, "asset_sha256": sha256_file(canonical.paths["asset"]),
        "model_manifest_sha256": sha256_file(canonical.paths["model_manifest"]),
        "row_order_identity_sha256": identity_sha(canonical.rows),
    })
    with pytest.raises(ValueError, match="invalid visual_features|label identity mismatch|row/speaker/split conflict"):
        build_projection(canonical)
    assert not canonical.output.exists()


def test_rejects_source_directory_and_refuses_existing_output(canonical):
    source_child = canonical.paths["asset"].parent / "projection"
    with pytest.raises(ValueError, match="outside the canonical source directories"):
        build_projection(canonical, source_child)
    assert not source_child.exists()
    build_projection(canonical)
    checksums = {path.name: sha256_file(path) for path in canonical.output.iterdir()}
    with pytest.raises(FileExistsError, match="already exists"):
        build_projection(canonical)
    assert {path.name: sha256_file(path) for path in canonical.output.iterdir()} == checksums


def test_rejects_concurrent_source_change_before_publication(canonical, monkeypatch):
    verify = builder.load_projected_train_dev

    def change_source_after_validation(*args):
        result = verify(*args)
        with canonical.paths["provenance_manifest"].open("a", encoding="utf-8") as stream:
            stream.write("\n")
        return result

    monkeypatch.setattr(builder, "load_projected_train_dev", change_source_after_validation)
    with pytest.raises(ValueError, match="source SHA-256 changed"):
        build_projection(canonical)
    assert not canonical.output.exists()


def test_publication_failure_does_not_leave_partial_package(canonical, monkeypatch):
    link = builder.os.link

    def fail_second_link(source, target):
        if Path(target).name == preflight.MODEL_MANIFEST:
            raise OSError("simulated publication failure")
        link(source, target)

    monkeypatch.setattr(builder.os, "link", fail_second_link)
    with pytest.raises(OSError, match="simulated publication failure"):
        build_projection(canonical)
    assert not canonical.output.exists()

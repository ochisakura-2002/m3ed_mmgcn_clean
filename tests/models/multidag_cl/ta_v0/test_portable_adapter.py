"""Synthetic portable manifests; no real assets or Test loader are used."""

from copy import deepcopy
import csv
import inspect
from pathlib import Path
import pickle

import numpy as np
import pytest
import yaml

from datasets.iemocap.ta_v0_dataset import (
    JOIN_FIELDS, MODEL_MANIFEST_FIELDS, PACKAGE_ROWS, PROVENANCE_MANIFEST_FIELDS,
    TAV0Dataset,
)
from scripts.models.multidag_cl.ta_v0.train import (
    DEFAULT_CONFIG, legacy_synthetic_package, synthetic_package, validate_config,
)


def load(package, *, split="train"):
    return TAV0Dataset(*package, split=split, expected_rows=len(package[1]))


def write_package(root, package):
    asset, model, provenance = package
    asset_path = root / "ta" / "iemocap_asym_ta_v1.pkl"
    asset_path.parent.mkdir()
    asset_path.write_bytes(pickle.dumps(asset))
    metadata = root / "metadata"
    metadata.mkdir()
    paths = []
    for name, rows, fields in (
        ("model_input_manifest_v1.csv", model, MODEL_MANIFEST_FIELDS),
        ("iemocap_asym_manifest_v1.csv", provenance, PROVENANCE_MANIFEST_FIELDS),
    ):
        path = metadata / name
        with path.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=(*fields, "video_clip_path"))
            writer.writeheader()
            writer.writerows({**{k: row[k] for k in fields}, "video_clip_path": "MUST_NOT_OPEN"}
                             for row in rows)
        paths.append(path)
    return asset_path, *paths


def test_exact_join_ignores_physical_csv_order_and_fetches_original_ta_rows():
    asset, model, provenance = package = synthetic_package()
    original = deepcopy(asset)
    model.reverse()
    provenance[:] = provenance[3:] + provenance[:3]
    dataset = load(package)
    item = dataset[0]
    assert item["canonical_order"] == (1, 2, 3)
    assert item["ta_row_indices"] == (2, 1, 0)
    dialogue = original["dialogues"][item["dialogue_id"]]
    assert tuple(dialogue["utterance_ids"]) != item["utterance_ids"]
    for position, source in enumerate(item["ta_row_indices"]):
        assert item["utterance_ids"][position] == dialogue["utterance_ids"][source]
        for field in ("text_features", "audio_features"):
            np.testing.assert_array_equal(item[field][position], dialogue[field][source])
    for did, dialogue in asset["dialogues"].items():
        for field in ("text_features", "audio_features"):
            np.testing.assert_array_equal(dialogue[field], original["dialogues"][did][field])


def test_noncontiguous_provenance_order_is_the_only_order_authority():
    asset, model, provenance = package = synthetic_package()
    # Storage/TA row order, dialogue_order, model-side order and global_order all
    # disagree with the authoritative provenance order.
    for row in model:
        row["canonical_dialogue_order"] = row["global_order"] = row["ta_row_index"] + 1
    for row in provenance:
        row["dialogue_order"] = row["global_order"] = row["ta_row_index"] + 1
        row["canonical_dialogue_order"] = {1: 5, 2: 16, 3: 23}[row["canonical_dialogue_order"]]
    item = load(package)[0]
    assert item["canonical_order"] == (5, 16, 23)
    assert item["ta_row_indices"] == (2, 1, 0)
    np.testing.assert_array_equal(item["audio_features"],
                                  asset["dialogues"][item["dialogue_id"]]["audio_features"][[2, 1, 0]])


@pytest.mark.parametrize("include_dialogue_bridge", [False, True])
def test_portable_file_contract_needs_neither_global_order_nor_dialogue_bridge(tmp_path, include_dialogue_bridge):
    package = synthetic_package()
    paths = write_package(tmp_path, package)
    if include_dialogue_bridge:
        # Deliberately unrelated 151-row dialogue-level provenance, never read.
        (tmp_path / "metadata" / "dialogue_split_bridge_v1.csv").write_text(
            "surrogate_dialogue_id,real_dialogue_id,split\n" + "unused,unused,train\n" * 151,
            encoding="utf-8",
        )
    before = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    dataset = TAV0Dataset.from_files(
        asset_path=paths[0], model_manifest_path=paths[1], provenance_manifest_path=paths[2],
        split="dev", expected_rows=len(package[1]),
    )
    assert dataset[0]["canonical_order"] == (1, 2, 3)
    assert dataset[0]["ta_row_indices"] == (2, 1, 0)
    assert {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()} == before
    assert "global_order" not in MODEL_MANIFEST_FIELDS + PROVENANCE_MANIFEST_FIELDS
    assert not any("visual" in key or "path" in key for key in dataset[0])


def test_manifest_id_sets_must_match_even_when_counts_match():
    package = synthetic_package()
    package[2][0]["utterance_id"] = "Ses01F_impro01_M099"
    with pytest.raises(ValueError, match="model/provenance ID sets differ"):
        load(package)


@pytest.mark.parametrize("side", [1, 2], ids=["model", "provenance"])
@pytest.mark.parametrize("field,value", [
    ("dialogue_id", "Ses01F_impro99"), ("split", "dev"), ("speaker", "M"),
    ("benchmark_label", "fru"), ("benchmark_label_id", 5),
    ("ta_dialogue_id", "wrong-ta-dialogue"), ("ta_row_index", 1),
])
def test_every_manifest_cross_check_fails_on_mismatch(side, field, value):
    package = synthetic_package()
    package[side][0][field] = value
    with pytest.raises(ValueError, match=f"model/provenance {field} conflict"):
        load(package)


@pytest.mark.parametrize("case", ["duplicate", "missing", "extra"])
@pytest.mark.parametrize("side", [1, 2], ids=["model", "provenance"])
def test_join_rejects_duplicate_missing_and_extra_manifest_ids(side, case):
    package = synthetic_package()
    rows = package[side]
    if case == "duplicate":
        rows.append(dict(rows[0]))
    elif case == "missing":
        rows.pop()
    else:
        rows.append(dict(rows[0], utterance_id="Ses01F_impro01_M099"))
    with pytest.raises(ValueError):
        TAV0Dataset(*package, split="train", expected_rows=10)


@pytest.mark.parametrize("field,value", [
    ("split", "dev"), ("ta_row_index", 1), ("ta_dialogue_id", "wrong-ta-dialogue"),
    ("benchmark_label_id", 5), ("benchmark_label", "fru"),
])
def test_agreeing_manifests_cannot_override_asset_identity_or_targets(field, value):
    package = synthetic_package()
    for rows in package[1:]:
        rows[0][field] = value
    with pytest.raises(ValueError):
        load(package)


def test_duplicate_canonical_order_within_dialogue_fails():
    package = synthetic_package()
    package[2][1]["canonical_dialogue_order"] = package[2][0]["canonical_dialogue_order"]
    with pytest.raises(ValueError, match="canonical order conflict"):
        load(package)


@pytest.mark.parametrize("order", [0, -1, True, 1.5, "1.5", "", None])
def test_canonical_order_requires_exact_positive_integer(order):
    package = synthetic_package()
    package[2][0]["canonical_dialogue_order"] = order
    with pytest.raises(ValueError):
        load(package)


@pytest.mark.parametrize("side,field", [(1, "session"), (1, "ta_asset_relative_path"),
                                       (2, "canonical_dialogue_order")])
def test_portable_required_fields_cannot_be_omitted(side, field):
    package = synthetic_package()
    del package[side][0][field]
    with pytest.raises(ValueError, match="missing metadata"):
        load(package)


def test_default_row_count_is_7433_and_smaller_fixture_requires_explicit_count():
    package = synthetic_package()
    assert PACKAGE_ROWS == 7433
    with pytest.raises(ValueError, match="row counts must both equal 7433"):
        TAV0Dataset(*package, split="train")
    assert inspect.signature(TAV0Dataset.from_files).parameters["expected_rows"].default == 7433
    assert len(load(package)) == 2


@pytest.mark.parametrize("count", [True, None, 0, -1, 10.0])
def test_invalid_expected_row_count_fails(count):
    with pytest.raises(ValueError, match="invalid expected_rows"):
        TAV0Dataset(*synthetic_package(), split="train", expected_rows=count)


def test_test_split_rejected_before_file_or_manifest_access(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Test attempted a file read")

    monkeypatch.setattr(Path, "open", forbidden)
    monkeypatch.setattr(pickle, "load", forbidden)
    with pytest.raises(ValueError, match="Test loader forbidden"):
        TAV0Dataset.from_files("DO_NOT_OPEN", "DO_NOT_OPEN", "DO_NOT_OPEN", split="test")
    with pytest.raises(ValueError, match="Test loader forbidden"):
        TAV0Dataset(None, None, None, split="test")


def test_portable_runtime_never_silently_falls_back_to_legacy_schema():
    with pytest.raises(ValueError, match="missing metadata"):
        TAV0Dataset(*legacy_synthetic_package(), split="train", expected_rows=10)
    assert "bridge_path" not in inspect.signature(TAV0Dataset.from_files).parameters
    assert "canonical_path" not in inspect.signature(TAV0Dataset.from_files).parameters


def test_baseline_uses_only_unresolved_portable_path_fields():
    root = Path(__file__).resolve().parents[4]
    raw = yaml.safe_load((root / DEFAULT_CONFIG).read_text(encoding="utf-8"))
    assert set(raw["dataset"]) == {
        "package", "label_vocab", "asset_path", "model_manifest_path", "provenance_manifest_path",
    }
    for field in ("asset_path", "model_manifest_path", "provenance_manifest_path"):
        assert raw["dataset"][field] is None
    validate_config(raw)
    raw["dataset"]["bridge_path"] = raw["dataset"].pop("model_manifest_path")
    with pytest.raises(ValueError):
        validate_config(raw)


def test_full_default_universe_count_joins_synthetic_identity_metadata_only():
    asset, model, provenance = package = synthetic_package()
    did = asset["splits"]["test"][0]
    dialogue = asset["dialogues"][did]
    template = model[-1]
    for number in range(1, PACKAGE_ROWS - len(model) + 1):
        uid = f"{did}_F{number:04d}"
        row = dict(template, utterance_id=uid, ta_row_index=number)
        model.append(row)
        provenance.append({k: row[k] for k in JOIN_FIELDS} | {"canonical_dialogue_order": number + 1})
        for field, value in (("utterance_ids", uid), ("speakers", "F"), ("benchmark_labels", "hap"),
                             ("benchmark_label_ids", 0), ("sentences", "synthetic identity only")):
            dialogue[field].append(value)
    # Test features must remain unused by a Train/Dev dataset. There is no Test loader.
    dialogue["text_features"] = dialogue["audio_features"] = None
    assert len(model) == len(provenance) == PACKAGE_ROWS
    for split in ("train", "dev"):
        assert len(TAV0Dataset(*package, split=split)) == 2

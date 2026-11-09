from copy import deepcopy
import csv
import pickle

import numpy as np
import pytest

from datasets.iemocap.ta_v0_dataset import (
    BRIDGE_FIELDS, CANONICAL_FIELDS, LegacySyntheticTAV0Dataset as TAV0Dataset, ta_v0_collate,
)
from models.multidag_cl.ta_v0 import LABEL_VOCAB
from scripts.models.multidag_cl.ta_v0.train import legacy_synthetic_package as synthetic_package


def test_canonical_order_fetches_exact_original_rows_without_mutation():
    asset, bridge, canonical = package = synthetic_package()
    did = asset["splits"]["train"][0]
    dialogue = asset["dialogues"][did]
    original = deepcopy(dialogue)
    dataset = TAV0Dataset(*package, split="train")
    item = dataset[0]
    assert item["canonical_order"] == (1, 2, 3)
    assert item["ta_row_indices"] == (2, 1, 0)
    for pos, row in enumerate(item["ta_row_indices"]):
        assert item["utterance_ids"][pos] == original["utterance_ids"][row]
        np.testing.assert_array_equal(item["text_features"][pos].numpy(), original["text_features"][row])
        np.testing.assert_array_equal(item["audio_features"][pos].numpy(), original["audio_features"][row])
    for key in ("text_features", "audio_features"):
        np.testing.assert_array_equal(dialogue[key], original[key])
        assert dialogue[key].flags.writeable
    item["text_features"][0, 0] = 999
    np.testing.assert_array_equal(dialogue["text_features"], original["text_features"])
    batch = ta_v0_collate([dataset[1], dataset[0]])
    assert batch["lengths"].tolist() == [2, 3]
    assert batch["utterance_ids"] == (dataset[1]["utterance_ids"], dataset[0]["utterance_ids"])
    assert (batch["labels"][~batch["attention_mask"]] == -100).all()
    assert not any("visual" in key for key in item.keys() | batch.keys())


@pytest.mark.parametrize("case", [
    "duplicate_bridge", "missing_bridge", "missing_canonical", "duplicate_canonical",
    "duplicate_asset_id", "wrong_row", "cross_split", "wrong_split", "wrong_dialogue",
    "speaker_conflict", "order_collision", "order_conflict", "missing_order",
    "legacy_labels", "wrong_label", "wrong_stoi", "visual_schema", "extra_visual", "canonical_split",
])
def test_identity_and_schema_fail_closed(case):
    asset, bridge, canonical = package = synthetic_package()
    first = bridge[0]
    did = first["dialogue_id"]
    if case == "duplicate_bridge":
        bridge.append(dict(first))
    elif case == "missing_bridge":
        bridge.pop(0)
    elif case == "missing_canonical":
        canonical.pop(0)
    elif case == "duplicate_canonical":
        canonical.append(dict(canonical[0]))
    elif case == "duplicate_asset_id":
        asset["dialogues"][did]["utterance_ids"][1] = first["utterance_id"]
    elif case == "wrong_row":
        first["ta_row_index"] = 1
    elif case == "cross_split":
        asset["splits"]["dev"].append(did)
    elif case == "wrong_split":
        first["split"] = "dev"
    elif case == "canonical_split":
        canonical[0]["split"] = "dev"
    elif case == "wrong_dialogue":
        first["ta_dialogue_id"] = "Ses01F_impro99"
    elif case == "speaker_conflict":
        canonical[0]["speaker"] = "M"
    elif case == "order_collision":
        canonical[1]["global_order"] = canonical[0]["global_order"]
    elif case == "order_conflict":
        first["canonical_dialogue_order"] = 9
    elif case == "missing_order":
        del canonical[0]["global_order"]
    elif case == "legacy_labels":
        asset["label_vocab"]["itos"] = ["exc", "neu", "fru", "sad", "hap", "ang"]
    elif case == "wrong_label":
        first["benchmark_label_id"] = 5
    elif case == "wrong_stoi":
        asset["label_vocab"]["stoi"] = {label: 5 - i for i, label in enumerate(LABEL_VOCAB)}
    elif case == "visual_schema":
        asset["feature_spec"]["visual"] = {"dimension": 342}
    elif case == "extra_visual":
        asset["dialogues"][did]["visual_features"] = None
    with pytest.raises(ValueError):
        TAV0Dataset(*package, split="train")


def test_label_order_is_exact_and_all_classes_preserved():
    asset, bridge, canonical = package = synthetic_package()
    assert LABEL_VOCAB == ("hap", "sad", "neu", "ang", "exc", "fru")
    # Assign every class across train and dev, retaining exact source/bridge identity.
    for i, row in enumerate(bridge):
        label_id = i % 6
        row["benchmark_label_id"], row["benchmark_label"] = label_id, LABEL_VOCAB[label_id]
        d = asset["dialogues"][row["dialogue_id"]]
        index = row["ta_row_index"]
        d["benchmark_label_ids"][index] = label_id
        d["benchmark_labels"][index] = LABEL_VOCAB[label_id]
    labels = []
    for split in ("train", "dev"):
        dataset = TAV0Dataset(*package, split=split)
        labels.extend(int(v) for item in dataset for v in item["labels"])
    assert set(labels) == set(range(6))


def test_canonical_superset_is_supported():
    asset, bridge, canonical = package = synthetic_package()
    canonical.append({"utterance_id": "Ses01F_impro01_M099", "dialogue_id": "Ses01F_impro01",
                      "speaker": "M", "global_order": 5})
    assert len(TAV0Dataset(*package, split="train")) == 2


def test_file_adapter_uses_explicit_paths_and_selected_columns(tmp_path):
    asset, bridge, canonical = synthetic_package()
    path = tmp_path / "ta.pkl"
    path.write_bytes(pickle.dumps(asset))
    before = path.read_bytes()
    for name, rows, fields in (("bridge.csv", bridge, BRIDGE_FIELDS),
                               ("canonical.csv", canonical, CANONICAL_FIELDS)):
        with (tmp_path / name).open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=(*fields, "video_clip_path"))
            writer.writeheader()
            writer.writerows(dict(row, video_clip_path="MUST_NOT_OPEN") for row in rows)
    dataset = TAV0Dataset.from_files(path, tmp_path / "bridge.csv", tmp_path / "canonical.csv", split="dev")
    assert dataset[0]["canonical_order"] == (1, 2, 3)
    assert path.read_bytes() == before
    assert not (tmp_path / "MUST_NOT_OPEN").exists()
    with pytest.raises(ValueError, match="Test loader forbidden"):
        TAV0Dataset.from_files("DO_NOT_OPEN", "DO_NOT_OPEN", "DO_NOT_OPEN", split="test")

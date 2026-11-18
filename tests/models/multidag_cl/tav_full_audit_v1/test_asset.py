from copy import deepcopy
import csv
import json
import pickle

import numpy as np
import pytest

from datasets.iemocap.tav_full_audit_dataset import TAVAuditDataset, fingerprint, identity_sha
from models.multidag_cl.ta_v0 import LABEL_VOCAB
from scripts.data.build.multidag_cl.build_tav_t1024_v1 import align_visual
from scripts.data.build.multidag_cl import build_tav_t1024_v1 as builder
from scripts.runtime.multidag_cl_ta_v0.formal import sha256_file


class ForbiddenTargets:
    def __getitem__(self, key):
        raise AssertionError("Test or historical targets accessed")

    def __iter__(self):
        raise AssertionError("Test or historical targets iterated")

    def __len__(self):
        raise AssertionError("Test target length accessed")


class ForbiddenLabelRow(dict):
    def __getitem__(self, key):
        if key in ("benchmark_label", "benchmark_label_id"):
            raise AssertionError("Test manifest target accessed")
        return super().__getitem__(key)


def package():
    rng = np.random.default_rng(100)
    current = {"schema": "iemocap_asym_ta_v1", "dataset_version": "asymmetric_merc_v1",
               "feature_spec": {"text": {"dimension": 1024}, "audio_ta": {"dimension": 1582}},
               "label_vocab": {"itos": list(LABEL_VOCAB)}, "dialogues": {}, "splits": {s: [] for s in ("train", "dev", "test")}}
    history = [{}, {}, ForbiddenTargets(), {}, {}, {}, {}, [], []]
    splits = {f"{s}_dialogue_ids": [] for s in current["splits"]}
    rows, orders, bridge = [], [], []
    counts = {s: {"utterances": 0, "dialogues": 0} for s in current["splits"]}
    for number, split in enumerate(("train", "train", "dev", "test"), 1):
        did, historical_did = f"Ses01F_impro0{number}", f"surrogate_{number}"
        ids = [f"{did}_F000", f"{did}_M001"]
        audio = rng.normal(size=(2, 1582)).astype(np.float32)
        current["splits"][split].append(did)
        current["dialogues"][did] = {"utterance_ids": ids, "speakers": ["F", "M"], "sentences": ["a", "b"],
                                     "benchmark_labels": ["hap", "sad"], "benchmark_label_ids": [0, 1],
                                     "text_features": rng.normal(size=(2, 1024)).astype(np.float32), "audio_features": audio}
        if split == "test":
            current["dialogues"][did]["benchmark_labels"] = ForbiddenTargets()
            current["dialogues"][did]["benchmark_label_ids"] = ForbiddenTargets()
        history[0][historical_did] = ["untrusted_id_1", "untrusted_id_0"]
        history[4][historical_did] = audio[::-1].copy().astype(np.float64)
        history[5][historical_did] = np.stack([np.full(342, number * 10 + 1), np.full(342, number * 10)]).astype(np.float64)
        splits[f"{split}_dialogue_ids"].append(historical_did)
        counts[split]["utterances"] += 2
        counts[split]["dialogues"] += 1
        for index, uid in enumerate(ids):
            row = {"utterance_id": uid, "dialogue_id": did, "ta_dialogue_id": did, "ta_row_index": index,
                   "split": split, "speaker": ["F", "M"][index], "benchmark_label": ["hap", "sad"][index],
                   "benchmark_label_id": index, "canonical_dialogue_order": 20 - index * 10}
            rows.append(ForbiddenLabelRow(row) if split == "test" else row)
            orders.append(ForbiddenLabelRow(row) if split == "test" else row.copy())
            bridge.append({"utterance_id": uid, "historical_dialogue_id": historical_did,
                           "historical_row_index": 1 - index, "audio_sha256": fingerprint(audio[index])})
    return current, rows, orders, history, splits, bridge, counts


def aligned():
    *arguments, counts = package()
    asset, rows, qc = align_visual(*arguments, expected_counts=counts)
    return asset, rows, qc, counts


def test_visual_mapping_preserves_current_identity_targets_splits_and_order():
    *arguments, counts = package()
    current = arguments[0]
    asset, rows, qc = align_visual(*arguments, expected_counts=counts)
    assert qc == {"matched": 8, "unmatched": 0, "ambiguous": 0, "split_conflict": 0}
    assert asset["splits"] is current["splits"]
    for number, (did, value) in enumerate(asset["dialogues"].items(), 1):
        assert value["benchmark_label_ids"] is current["dialogues"][did]["benchmark_label_ids"]
        assert value["benchmark_labels"] is current["dialogues"][did]["benchmark_labels"]
        assert value["utterance_ids"] is current["dialogues"][did]["utterance_ids"]
        assert value["text_features"] is current["dialogues"][did]["text_features"]
        assert value["audio_features"] is current["dialogues"][did]["audio_features"]
        np.testing.assert_array_equal(value["visual_features"][0], np.full(342, number * 10))
        np.testing.assert_array_equal(value["visual_features"][1], np.full(342, number * 10 + 1))
    data = TAVAuditDataset(asset, rows, split="train", expected_counts=counts)
    assert data[0]["utterance_ids"][0].endswith("M001")
    assert data[0]["labels"].tolist() == [1, 0]
    assert len(identity_sha(rows)) == 64


@pytest.mark.parametrize("failure", ["unmatched", "duplicate_bridge", "wrong_fingerprint", "wrong_row", "ambiguous_audio", "split_conflict", "visual_dim", "canonical_order", "canonical_label", "current_text"])
def test_alignment_blocks_nonexact_or_conflicting_sources(failure):
    current, rows, orders, history, splits, bridge, counts = package()
    did = next(iter(current["dialogues"]))
    hdid = bridge[0]["historical_dialogue_id"]
    if failure == "unmatched":
        bridge.pop()
    elif failure == "duplicate_bridge":
        bridge[1] = bridge[0]
    elif failure == "wrong_fingerprint":
        bridge[0]["audio_sha256"] = "0" * 64
    elif failure == "wrong_row":
        bridge[0]["historical_row_index"] = 0
    elif failure == "ambiguous_audio":
        history[4][hdid][0] = history[4][hdid][1]
    elif failure == "split_conflict":
        splits["train_dialogue_ids"].remove(hdid)
        splits["dev_dialogue_ids"].append(hdid)
    elif failure == "visual_dim":
        history[5][hdid] = history[5][hdid][:, :341]
    elif failure == "canonical_order":
        orders[1]["canonical_dialogue_order"] = orders[0]["canonical_dialogue_order"]
    elif failure == "canonical_label":
        orders[0]["benchmark_label_id"] = 4
    elif failure == "current_text":
        current["dialogues"][did]["text_features"] = current["dialogues"][did]["text_features"][:, :1023]
    with pytest.raises(ValueError):
        align_visual(current, rows, orders, history, splits, bridge, expected_counts=counts)


def test_dataset_never_indexes_test_features_or_targets():
    asset, rows, _, counts = aligned()
    for did in asset["splits"]["test"]:
        for key in ("text_features", "audio_features", "visual_features"):
            asset["dialogues"][did][key] = ForbiddenTargets()
    for split in ("train", "dev"):
        dataset = TAVAuditDataset(asset, rows, split=split, expected_counts=counts)
        assert dataset[0]["visual_features"].shape[-1] == 342
    with pytest.raises(ValueError, match="Test factory"):
        TAVAuditDataset(None, None, split="test")
    with pytest.raises(ValueError, match="Test factory"):
        TAVAuditDataset.from_files("FORBIDDEN", "FORBIDDEN", split="test")


def test_builder_file_roundtrip_provenance_and_exclusive_destination(monkeypatch, artifact_dir):
    current, rows, orders, history, splits, bridge, counts = package()
    for name, value in (("canonical.pkl", current), ("history.pkl", history)):
        with (artifact_dir / name).open("xb") as stream:
            pickle.dump(value, stream)
    for name, values in (("model.csv", rows), ("order.csv", orders), ("bridge.csv", bridge)):
        with (artifact_dir / name).open("x", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(values[0]))
            writer.writeheader()
            writer.writerows(dict(row) for row in values)
    splits["project_pkl_sha256"] = sha256_file(artifact_dir / "history.pkl")
    (artifact_dir / "splits.json").write_text(json.dumps(splits))
    # Only this fixture's universe count changes; production CLI has no count override.
    actual_align = builder.align_visual
    monkeypatch.setattr(builder, "align_visual", lambda *args: actual_align(*args, expected_counts=counts))
    monkeypatch.setattr(builder, "COUNTS", counts)
    inputs = dict(canonical_asset=artifact_dir / "canonical.pkl", model_manifest=artifact_dir / "model.csv",
                  order_manifest=artifact_dir / "order.csv", historical_asset=artifact_dir / "history.pkl",
                  historical_split_manifest=artifact_dir / "splits.json", fingerprint_bridge=artifact_dir / "bridge.csv",
                  destination=artifact_dir / "tav")
    before = {key: sha256_file(path) for key, path in inputs.items() if key != "destination"}
    result = builder.build(**inputs)
    assert result["asset_sha256"] == sha256_file(artifact_dir / "tav" / builder.ASSET_NAME)
    assert result["model_manifest_sha256"] == sha256_file(artifact_dir / "tav" / builder.MODEL_NAME)
    assert result["row_order_identity_sha256"] and result["test_labels_inspected"] == 0
    assert before == {key: sha256_file(path) for key, path in inputs.items() if key != "destination"}
    with pytest.raises(FileExistsError):
        builder.build(**inputs)

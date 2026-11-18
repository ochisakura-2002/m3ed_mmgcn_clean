"""Small synthetic audio-only bijections; no real assets or Test metrics."""

import csv
import hashlib
import json
from pathlib import Path
import pickle
import shutil
import struct
import uuid

import numpy as np
import pytest

from datasets.iemocap.tav_full_audit_dataset import fingerprint
from scripts.data.build.multidag_cl import build_audio_fingerprint_bridge_v1 as exporter
from scripts.data.build.multidag_cl.build_tav_t1024_v1 import align_visual


class Forbidden:
    def __getitem__(self, key):
        raise AssertionError("target or unrelated feature accessed")

    def __iter__(self):
        raise AssertionError("target or unrelated feature iterated")

    def __len__(self):
        raise AssertionError("target length accessed")

    def __array__(self, *args, **kwargs):
        raise AssertionError("unrelated feature converted")

    def __eq__(self, other):
        raise AssertionError("target compared")


class IdentityOnlyRow(dict):
    def __getitem__(self, key):
        if key in ("benchmark_label", "benchmark_label_id"):
            raise AssertionError("manifest label accessed")
        return super().__getitem__(key)

    def get(self, key, *args):
        if key in ("benchmark_label", "benchmark_label_id"):
            raise AssertionError("manifest label accessed")
        return super().get(key, *args)


class AudioOnlyHistory(list):
    def __getitem__(self, key):
        if key not in (0, 4):
            raise AssertionError("historical non-audio field accessed")
        return super().__getitem__(key)


def package():
    rng = np.random.default_rng(192)
    current = {"schema": "iemocap_asym_ta_v1", "dataset_version": "asymmetric_merc_v1",
               "feature_spec": {"text": {"dimension": 1024}, "audio_ta": {"dimension": 1582}},
               "label_vocab": Forbidden(), "dialogues": {}, "splits": {s: [] for s in exporter.COUNTS}}
    historical = AudioOnlyHistory([{}, Forbidden(), Forbidden(), Forbidden(), {}, Forbidden(),
                                  Forbidden(), Forbidden(), Forbidden()])
    splits = {f"{s}_dialogue_ids": [] for s in exporter.COUNTS}
    model, orders, expected = [], [], []
    counts = {s: {"utterances": 0, "dialogues": 0} for s in exporter.COUNTS}
    for number, split in enumerate(("train", "train", "dev", "test"), 1):
        real, did, hdid = f"Ses01F_impro0{number}", f"ta_{number}", f"official_{number}"
        ids = [f"{real}_F000", f"{real}_M001"]
        audio = rng.normal(size=(2, 1582)).astype(np.float32)
        current["splits"][split].append(did)
        current["dialogues"][did] = {"utterance_ids": ids, "audio_features": audio, "speakers": ["F", "M"],
                                     "benchmark_labels": Forbidden(), "benchmark_label_ids": Forbidden(),
                                     "text_features": Forbidden(), "sentences": Forbidden()}
        historical[0][hdid] = ["untrusted_1", "untrusted_0"]
        historical[4][hdid] = audio[::-1].astype(np.float64)
        splits[f"{split}_dialogue_ids"].append(hdid)
        counts[split]["dialogues"] += 1
        counts[split]["utterances"] += 2
        for index, uid in enumerate(ids):
            row = {"utterance_id": uid, "dialogue_id": real, "ta_dialogue_id": did,
                   "ta_row_index": index, "split": split, "speaker": ["F", "M"][index],
                   "benchmark_label": Forbidden(), "benchmark_label_id": Forbidden()}
            model.append(IdentityOnlyRow(row))
            orders.append(IdentityOnlyRow(dict(row, canonical_dialogue_order=20 - index * 10)))
            expected.append({"utterance_id": uid, "historical_dialogue_id": hdid,
                             "historical_row_index": 1 - index, "audio_sha256": fingerprint(audio[index]),
                             "canonical_split": split, "historical_split": split})
    return current, model, orders, historical, splits, expected, counts


def aligned(values=None):
    current, model, orders, history, splits, _, counts = package() if values is None else values
    return exporter.align(current, model, orders, history, splits, expected_counts=counts)


@pytest.fixture
def artifact_dir():
    # Follow the existing TAV fixtures: Python 3.9 tempfile directory ACLs are
    # incompatible with this Windows sandbox. Allocate only our own small data.
    parent = (Path(__file__).resolve().parents[4] / "tmp").resolve()
    path = parent / ("audio_bridge_test_" + uuid.uuid4().hex)
    path.mkdir(parents=True)
    yield path
    if path.resolve() != path or path.parent != parent or not path.name.startswith("audio_bridge_test_"):
        raise AssertionError("synthetic cleanup path escaped workspace")
    shutil.rmtree(path)


def write_inputs(directory):
    current, model, orders, history, splits, expected, counts = package()
    inputs = {"canonical_asset": directory / "canonical.pkl", "model_manifest": directory / "model.csv",
              "order_manifest": directory / "order.csv", "historical_asset": directory / "historical.pkl",
              "historical_split_manifest": directory / "splits.json", "output": directory / "bridge.csv"}
    inputs["canonical_asset"].write_bytes(pickle.dumps(current))
    inputs["historical_asset"].write_bytes(pickle.dumps(history))
    for key, rows in (("model_manifest", model), ("order_manifest", orders)):
        fields = (*exporter.IDENTITY_FIELDS, *(("canonical_dialogue_order",) if key == "order_manifest" else ()))
        with inputs[key].open("x", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            writer.writerows({key: row[key] for key in fields} for row in rows)
    splits["project_pkl_sha256"] = exporter.sha256_file(inputs["historical_asset"])
    inputs["historical_split_manifest"].write_text(json.dumps(splits), encoding="utf-8")
    return inputs, expected, counts


def tiny_build(monkeypatch, inputs, counts):
    # Exercise actual CLI/build wiring while keeping synthetic data small.
    actual = exporter.align
    monkeypatch.setattr(exporter, "align", lambda *a: actual(*a, expected_counts=counts))
    return exporter.build(**inputs)


def test_perfect_one_to_one_mapping_and_frozen_counts():
    values = package()
    rows, qc = aligned(values)
    assert rows == values[-2]
    assert qc["canonical_utterances"] == qc["historical_utterances"] == qc["historical_matched"] == 8
    assert qc["canonical_dialogues"] == qc["historical_dialogues"] == 4
    assert qc["canonical_split_counts"] == qc["historical_split_counts"] == values[-1]
    assert all(qc[key] == 0 for key in ("unmatched", "ambiguous", "duplicate_current_fingerprint",
                                      "duplicate_historical_fingerprint", "reused_historical_rows", "split_conflict"))
    assert exporter.COUNTS == {"train": {"utterances": 4830, "dialogues": 100},
                               "dev": {"utterances": 980, "dialogues": 20},
                               "test": {"utterances": 1623, "dialogues": 31}}
    with pytest.raises(ValueError, match="required universe"):
        exporter.align(*values[:5])


def test_reuses_fingerprint_and_float64_rounding_matches_float32():
    assert exporter.fingerprint is fingerprint
    values = package()
    # These are distinct float64 values that round to the same float32 inputs.
    for array in values[3][4].values():
        array[:] = np.nextafter(array, np.inf)
    assert aligned(values)[0] == values[-2]


@pytest.mark.parametrize("kind", ["historical_duplicate", "current_duplicate", "unmatched", "split_conflict"])
def test_mapping_failures_are_closed_and_counted(kind):
    values = package()
    current, _, _, history, splits = values[:5]
    if kind == "historical_duplicate":
        history[4]["official_1"][0] = history[4]["official_1"][1]
        required = {"duplicate_historical_fingerprint": 1, "ambiguous": 1,
                    "multiple_historical_rows_for_one_current_utterance": 1}
    elif kind == "current_duplicate":
        current["dialogues"]["ta_1"]["audio_features"][1] = current["dialogues"]["ta_1"]["audio_features"][0]
        required = {"duplicate_current_fingerprint": 1, "reused_historical_rows": 1}
    elif kind == "unmatched":
        current["dialogues"]["ta_1"]["audio_features"][0] += 7
        required = {"unmatched": 1}
    else:
        splits["train_dialogue_ids"].remove("official_1")
        splits["dev_dialogue_ids"].append("official_1")
        required = {"split_conflict": 2}
    with pytest.raises(ValueError, match="FAIL CLOSED") as error:
        aligned(values)
    qc = json.loads(str(error.value).split("FAIL CLOSED: ")[1])
    for key, count in required.items():
        assert qc[key] == count


def test_reused_location_rejected_even_with_distinct_fingerprints():
    values = package()
    canonical, _ = exporter.canonical_records(*values[:3], expected_counts=values[-1])
    historical, _ = exporter.historical_records(*values[3:5])
    historical[1]["historical_row_index"] = historical[0]["historical_row_index"]
    with pytest.raises(ValueError, match='"reused_historical_rows": 1'):
        exporter.match_records(canonical, historical)


def test_ambiguous_historical_candidates_never_select_first():
    values = package()
    canonical, _ = exporter.canonical_records(*values[:3], expected_counts=values[-1])
    historical, _ = exporter.historical_records(*values[3:5])
    historical.append(dict(historical[0], historical_dialogue_id="extra"))
    with pytest.raises(ValueError, match='"ambiguous": 1'):
        exporter.match_records(canonical, historical)


def test_labels_and_all_unrelated_features_are_opaque():
    # All canonical targets, both manifests' targets, and historical fields other
    # than 0/4 raise if accessed. Even Test T features cannot be read here.
    values = package()
    baseline = aligned(values)[0]
    list.__setitem__(values[3], 2, {"arbitrary": [999, -1, "changed"]})
    list.__setitem__(values[3], 5, Forbidden())
    assert aligned(values)[0] == baseline


def test_deterministic_model_manifest_order_and_sha256():
    values = package()
    original = exporter.csv_bytes(aligned(values)[0])
    values[0]["dialogues"] = dict(reversed(list(values[0]["dialogues"].items())))
    values[2].reverse()
    list.__setitem__(values[3], 0, dict(reversed(list(values[3][0].items()))))
    list.__setitem__(values[3], 4, dict(reversed(list(values[3][4].items()))))
    repeat = exporter.csv_bytes(aligned(values)[0])
    assert repeat == original
    assert hashlib.sha256(repeat).hexdigest() == hashlib.sha256(original).hexdigest()
    values[1].reverse()
    assert [row["utterance_id"] for row in aligned(values)[0]] == [row["utterance_id"] for row in values[1]]


def test_sha256_known_bytes_endianness_and_noncontiguous_input():
    values = [0.0, -0.0, 1.0, -2.5] + [0.0] * 1578
    expected = hashlib.sha256(struct.pack("<1582f", *values)).hexdigest()
    assert expected == "539fbc958d19a57c576b094790a9aa156f5e787ff7b101b18c406c57837f6bdf"
    array = np.asarray(values, dtype=np.float32)
    strided = np.zeros((1582, 2), dtype=np.float32)
    strided[:, 1] = array
    assert not strided[:, 1].flags.c_contiguous
    for variant in (array, array.astype(np.float64), array.astype(">f4"), strided[:, 1]):
        assert fingerprint(variant) == expected


@pytest.mark.parametrize("kind", ["model_duplicate", "model_missing", "order_id", "order_rank", "row",
                                  "speaker", "canonical_split", "split_overlap", "history_split_overlap",
                                  "history_missing_split", "current_dtype", "audio_dim", "nan", "overflow",
                                  "historical_count", "historical_extra", "wrong_text_metadata"])
def test_identity_split_shape_and_universe_guards(kind):
    values = package()
    current, model, orders, history, splits = values[:5]
    if kind == "model_duplicate":
        model[1] = model[0]
    elif kind == "model_missing":
        model.pop()
    elif kind == "order_id":
        orders[0]["utterance_id"] = "bad"
    elif kind == "order_rank":
        orders[0]["canonical_dialogue_order"] = orders[1]["canonical_dialogue_order"]
    elif kind == "row":
        model[0]["ta_row_index"] = orders[0]["ta_row_index"] = 1
    elif kind == "speaker":
        current["dialogues"]["ta_1"]["speakers"][0] = "M"
    elif kind == "canonical_split":
        model[0]["split"] = orders[0]["split"] = "dev"
    elif kind == "split_overlap":
        current["splits"]["test"].append("ta_1")
    elif kind == "history_split_overlap":
        splits["test_dialogue_ids"].append("official_1")
    elif kind == "history_missing_split":
        splits["train_dialogue_ids"].remove("official_1")
    elif kind == "current_dtype":
        current["dialogues"]["ta_1"]["audio_features"] = current["dialogues"]["ta_1"]["audio_features"].astype(np.float64)
    elif kind == "audio_dim":
        history[4]["official_1"] = history[4]["official_1"][:, :1581]
    elif kind == "nan":
        history[4]["official_1"][0, 0] = np.nan
    elif kind == "overflow":
        history[4]["official_1"][0, 0] = np.finfo(np.float64).max
    elif kind == "historical_count":
        history[0]["official_1"].pop()
    elif kind == "historical_extra":
        history[4]["extra"] = history[4]["official_1"]
    else:
        current["feature_spec"]["text"]["dimension"] = 768
    with pytest.raises(ValueError):
        aligned(values)


def test_emitted_bridge_is_accepted_by_existing_builder():
    values = package()
    bridge, _ = aligned(values)
    current, model, orders, guarded_history, splits, _, counts = values
    history = [guarded_history[0], Forbidden(), Forbidden(), Forbidden(), guarded_history[4], {}, Forbidden(), [], []]
    current["label_vocab"] = {"itos": ["hap", "sad", "neu", "ang", "exc", "fru"]}
    for dialogue in current["dialogues"].values():
        dialogue["text_features"] = np.zeros((2, 1024), dtype=np.float32)
    # Only the consumer needs Train/Dev labels. Its Test fields remain opaque.
    for row, order in zip(model, orders):
        if row["split"] != "test":
            for item in (row, order):
                item["benchmark_label"], item["benchmark_label_id"] = "hap", 0
            dialogue = current["dialogues"][row["ta_dialogue_id"]]
            dialogue["benchmark_labels"], dialogue["benchmark_label_ids"] = ["hap"] * 2, [0] * 2
    model = [dict(row) if row["split"] != "test" else row for row in model]
    orders = [dict(row) if row["split"] != "test" else row for row in orders]
    for did in history[0]:
        history[5][did] = np.zeros((2, 342), dtype=np.float32)
    assert align_visual(current, model, orders, history, splits, bridge, expected_counts=counts)[2]["matched"] == 8


def test_build_roundtrip_audit_source_integrity_and_exclusive_output(monkeypatch, artifact_dir):
    inputs, expected, counts = write_inputs(artifact_dir)
    before = {key: exporter.sha256_file(path) for key, path in inputs.items() if key != "output"}
    audit = tiny_build(monkeypatch, inputs, counts)
    output = inputs["output"]
    assert output.read_bytes() == exporter.csv_bytes(expected)
    assert json.loads(output.with_suffix(".audit.json").read_text(encoding="utf-8")) == audit
    assert audit["bridge_sha256"] == exporter.sha256_file(output)
    assert audit["fingerprint_algorithm"] == exporter.ALGORITHM
    assert audit["historical_row_index_base"] == 0 and audit["bridge_rows"] == 8
    assert len(audit["git_head"]) == 40 and audit["timestamp_utc"].endswith("+00:00")
    assert audit["builder_sha256"] == exporter.sha256_file(exporter.SOURCE_FILES["builder"])
    assert audit["exporter_sha256"] == exporter.sha256_file(Path(exporter.__file__))
    assert all(audit[key] == 0 for key in ("historical_labels_inspected", "test_labels_inspected",
                                         "test_loader_built", "test_evaluation_count"))
    assert before == {key: value["sha256"] for key, value in audit["sources"].items()}
    assert before == {key: exporter.sha256_file(path) for key, path in inputs.items() if key != "output"}
    with pytest.raises(FileExistsError):
        exporter.build(**inputs)
    assert output.read_bytes() == exporter.csv_bytes(expected)


@pytest.mark.parametrize("failure", ["binding", "source_mutation", "code_mutation", "default_count", "matching"])
def test_build_failures_do_not_publish_bridge_or_audit(monkeypatch, artifact_dir, failure):
    inputs, _, counts = write_inputs(artifact_dir)
    actual = exporter.align
    def align(*args):
        if failure == "source_mutation":
            with inputs["model_manifest"].open("a", encoding="utf-8") as stream:
                stream.write("\n")
        if failure == "code_mutation":
            code_copy.write_text("after", encoding="utf-8")
        if failure == "matching":
            args[0]["dialogues"]["ta_1"]["audio_features"][0] += 10
        return actual(*args, expected_counts=counts)
    if failure == "binding":
        splits = json.loads(inputs["historical_split_manifest"].read_text())
        splits["project_pkl_sha256"] = "0" * 64
        inputs["historical_split_manifest"].write_text(json.dumps(splits))
    elif failure == "code_mutation":
        code_copy = artifact_dir / "code.py"
        code_copy.write_text("before", encoding="utf-8")
        monkeypatch.setattr(exporter, "SOURCE_FILES", dict(exporter.SOURCE_FILES, synthetic_code=code_copy))
    if failure != "default_count":
        monkeypatch.setattr(exporter, "align", align)
    with pytest.raises(ValueError):
        exporter.build(**inputs)
    assert not inputs["output"].exists()
    assert not inputs["output"].with_suffix(".audit.json").exists()
    assert not list(artifact_dir.glob(".audio_bridge_*.tmp"))


def test_publication_failure_rolls_back_own_files_and_preserves_existing(monkeypatch, artifact_dir):
    output, audit = artifact_dir / "bridge.csv", artifact_dir / "bridge.audit.json"
    actual = exporter.os.link
    def broken_link(source, target):
        if target == output:
            raise OSError("synthetic publication failure")
        return actual(source, target)
    monkeypatch.setattr(exporter.os, "link", broken_link)
    with pytest.raises(OSError, match="publication failure"):
        exporter.publish(output, audit, b"bridge", b"audit")
    assert not output.exists() and not audit.exists()
    assert not list(artifact_dir.glob(".audio_bridge_*.tmp"))
    audit.write_bytes(b"existing audit")
    with pytest.raises(FileExistsError):
        exporter.publish(output, audit, b"bridge", b"new audit")
    assert audit.read_bytes() == b"existing audit" and not output.exists()


def test_cli_requires_all_six_paths_and_has_no_count_or_discovery_options(monkeypatch, artifact_dir):
    with pytest.raises(SystemExit) as error:
        exporter.main([])
    assert error.value.code == 2
    with pytest.raises(SystemExit):
        exporter.main(["--expected-count", "8"])
    inputs, _, counts = write_inputs(artifact_dir)
    actual = exporter.align
    monkeypatch.setattr(exporter, "align", lambda *a: actual(*a, expected_counts=counts))
    argv = [item for key, path in inputs.items() for item in ("--" + key.replace("_", "-"), str(path))]
    assert exporter.main(argv) == 0
    assert inputs["output"].exists()

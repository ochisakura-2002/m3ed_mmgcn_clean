from __future__ import annotations

import json
from pathlib import Path
import pickle

import numpy as np
import pytest

from datasets.iemocap.official_feature_dataset import IEMOCAPOfficialFeatureDataset
from scripts.data.build.multidag_cl.build_official_feature_json import (
    build_official_artifacts,
)
from scripts.data.build.multidag_cl.import_official_assets import (
    import_official_assets,
)
from scripts.data.build.multidag_cl.inspect_official_assets import (
    OfficialAssetsUnavailable,
    inspect_official_assets,
)
from scripts.data.build.multidag_cl.schema import (
    AUDIO_DIM,
    MODALITY_ORDER,
    OFFICIAL_FILENAMES,
    TEXT_DIM,
    VISUAL_DIM,
    atomic_write_jsonl,
)
from scripts.data.build.multidag_cl.validate_official_features import (
    validate_official_directory,
)
from scripts.models.multidag_cl.paper_reimplementation import train as runtime_cli


ROOT = Path(__file__).resolve().parents[4]
PAPER_DATA_CONFIG = (
    ROOT
    / "configs/multidag_cl/paper_reimplementation/iemocap/full_context"
    / "paper_data_reproduction/paper_data_reproduction.yaml"
)


def _records() -> list[dict[str, str]]:
    return [
        {"split": "train", "dialogue_id": "Ses01F_impro01", "utterance_id": "Ses01F_impro01_F000", "text": "a", "speaker": "F", "label": "happy"},
        {"split": "train", "dialogue_id": "Ses01F_impro01", "utterance_id": "Ses01F_impro01_M001", "text": "b", "speaker": "M", "label": "sad"},
        {"split": "dev", "dialogue_id": "Ses02M_script01", "utterance_id": "Ses02M_script01_M000", "text": "c", "speaker": "M", "label": "neutral"},
        {"split": "test", "dialogue_id": "Ses05F_impro01", "utterance_id": "Ses05F_impro01_F000", "text": "d", "speaker": "F", "label": "happy"},
    ]


def _official_assets(root: Path, *, remove_ids: bool = False) -> Path:
    records = _records()
    manifest = root / "utterances.jsonl"
    atomic_write_jsonl(manifest, records)
    feature_paths: list[Path] = []
    for modality, dimension, offset in (
        ("text", TEXT_DIM, 1.0),
        ("audio", AUDIO_DIM, 2.0),
        ("visual", VISUAL_DIM, 3.0),
    ):
        path = root / f"{modality}.jsonl"
        atomic_write_jsonl(
            path,
            (
                {
                    "utterance_id": record["utterance_id"],
                    "feature": [offset + index] * dimension,
                }
                for index, record in enumerate(records)
            ),
        )
        feature_paths.append(path)
    official_root = root / "official"
    build_official_artifacts(
        manifest_path=manifest,
        text_feature_path=feature_paths[0],
        audio_feature_path=feature_paths[1],
        visual_feature_path=feature_paths[2],
        output_dir=official_root,
        speaker_order=["F", "M"],
        label_order=["happy", "sad", "neutral"],
    )
    if remove_ids:
        for split, filename in OFFICIAL_FILENAMES.items():
            path = official_root / filename
            dialogues = json.loads(path.read_text(encoding="utf-8"))
            for dialogue in dialogues:
                for utterance in dialogue:
                    del utterance["dialogue_id"]
                    del utterance["utterance_id"]
            path.write_text(json.dumps(dialogues), encoding="utf-8")
    return official_root


def test_inspection_recovers_schema_counts_vocab_order_and_sessions(tmp_path: Path) -> None:
    official_root = _official_assets(tmp_path)
    output = tmp_path / "official_asset_manifest.json"
    manifest = inspect_official_assets(
        official_root=official_root,
        manifest_output=output,
    )

    assert output.is_file()
    assert manifest["status"] == "PASS"
    assert manifest["modality_order"] == list(MODALITY_ORDER)
    assert manifest["dimensions"] == {"text": 1024, "audio": 1582, "visual": 342}
    assert manifest["total_dimension"] == 2948
    assert manifest["all_vectors_finite"] is True
    assert manifest["splits"]["train"]["dialogue_count"] == 1
    assert manifest["splits"]["train"]["utterance_count"] == 2
    assert manifest["splits"]["dev"]["dialogue_ids"] == ["Ses02M_script01"]
    assert manifest["splits"]["test"]["session_distribution"]["counts"] == {"Ses05": 1}
    assert manifest["speaker_vocab"] == {"stoi": {"F": 0, "M": 1}, "itos": ["F", "M"]}
    assert manifest["label_vocab"]["itos"] == ["happy", "sad", "neutral"]


def test_import_preserves_exact_split_order_and_feature_values_without_ids(
    tmp_path: Path,
) -> None:
    official_root = _official_assets(tmp_path, remove_ids=True)
    output_dir = tmp_path / "imported"
    result = import_official_assets(official_root=official_root, output_dir=output_dir)

    assert result["status"] == "PASS"
    split_manifest = json.loads(
        (output_dir / "official_split_manifest.json").read_text(encoding="utf-8")
    )
    train_id = "multidag_cl::train::dialogue::000000"
    dev_id = "multidag_cl::dev::dialogue::000000"
    test_id = "multidag_cl::test::dialogue::000000"
    assert split_manifest["train_dialogue_ids"] == [train_id]
    assert split_manifest["dev_dialogue_ids"] == [dev_id]
    assert split_manifest["test_dialogue_ids"] == [test_id]
    assert split_manifest["identifier_sources"]["dev"] == [
        "generated_from_official_split_and_position"
    ]

    pkl_path = output_dir / "IEMOCAP_features_multidag_cl_official.pkl"
    with pkl_path.open("rb") as file:
        value = pickle.load(file)
    source_train = json.loads(
        (official_root / OFFICIAL_FILENAMES["train"]).read_text(encoding="utf-8")
    )
    np.testing.assert_array_equal(value[3][train_id], [item["cls"][0] for item in source_train[0]])
    np.testing.assert_array_equal(value[4][train_id], [item["cls"][1] for item in source_train[0]])
    np.testing.assert_array_equal(value[5][train_id], [item["cls"][2] for item in source_train[0]])
    assert value[7] == [train_id, dev_id]
    assert value[8] == [test_id]

    for split, expected in (("train", [train_id]), ("val", [dev_id]), ("test", [test_id])):
        dataset = IEMOCAPOfficialFeatureDataset(
            pkl_path,
            split=split,
            val_split_strategy="official_split_manifest",
            split_manifest_path=output_dir / "official_split_manifest.json",
        )
        assert dataset.keys == expected
        assert dataset.get_feature_dims() == {
            "text_feature_dim": 1024,
            "audio_feature_dim": 1582,
            "visual_feature_dim": 342,
        }

    asset_manifest = json.loads(
        (output_dir / "official_asset_manifest.json").read_text(encoding="utf-8")
    )
    assert asset_manifest["splits"]["dev"]["session_distribution"]["available"] is False
    assert asset_manifest["layer2"]["project_pkl_sha256"] == result["project_pkl_sha256"]


def test_missing_official_label_validates_and_converts_to_minus_one_without_loss(
    tmp_path: Path,
) -> None:
    official_root = _official_assets(tmp_path)
    train_path = official_root / OFFICIAL_FILENAMES["train"]
    train_data = json.loads(train_path.read_text(encoding="utf-8"))
    missing_label_utterance = train_data[0][1]
    missing_label_utterance.pop("label")
    train_path.write_text(json.dumps(train_data), encoding="utf-8")

    validation = validate_official_directory(official_root)
    assert validation["status"] == "PASS"
    assert validation["missing_label_utterance_count"] == 1
    assert validation["missing_label_utterance_count_by_split"] == {
        "train": 1,
        "dev": 0,
        "test": 0,
    }

    inspection = inspect_official_assets(official_root=official_root)
    assert inspection["status"] == "PASS"
    assert inspection["missing_label_utterance_count"] == 1
    assert inspection["splits"]["train"]["missing_label_utterance_count"] == 1

    output_dir = tmp_path / "imported_missing_label"
    result = import_official_assets(official_root=official_root, output_dir=output_dir)
    assert result["status"] == "PASS"

    pkl_path = output_dir / "IEMOCAP_features_multidag_cl_official.pkl"
    with pkl_path.open("rb") as file:
        value = pickle.load(file)
    dialogue_id = "Ses01F_impro01"
    assert value[0][dialogue_id] == [
        "Ses01F_impro01_F000",
        "Ses01F_impro01_M001",
    ]
    assert value[1][dialogue_id] == ["F", "M"]
    assert value[2][dialogue_id] == [0, -1]
    assert value[6][dialogue_id] == ["a", "b"]
    np.testing.assert_array_equal(
        value[3][dialogue_id],
        [item["cls"][0] for item in train_data[0]],
    )
    np.testing.assert_array_equal(
        value[4][dialogue_id],
        [item["cls"][1] for item in train_data[0]],
    )
    np.testing.assert_array_equal(
        value[5][dialogue_id],
        [item["cls"][2] for item in train_data[0]],
    )

    dataset = IEMOCAPOfficialFeatureDataset(
        pkl_path,
        split="train",
        val_split_strategy="official_split_manifest",
        split_manifest_path=output_dir / "official_split_manifest.json",
    )
    item = dataset[0]
    assert item["utterance_ids"] == value[0][dialogue_id]
    assert item["labels"].tolist() == [0, -1]
    assert item["speaker_ids_int"].shape[0] == 2
    assert item["text_features"].shape[0] == 2
    assert item["audio_features"].shape[0] == 2
    assert item["visual_features"].shape[0] == 2

    # Inspection/conversion must not edit the author-released JSON.
    assert json.loads(train_path.read_text(encoding="utf-8")) == train_data


def test_present_official_label_must_exist_in_vocab(tmp_path: Path) -> None:
    official_root = _official_assets(tmp_path)
    train_path = official_root / OFFICIAL_FILENAMES["train"]
    train_data = json.loads(train_path.read_text(encoding="utf-8"))
    train_data[0][0]["label"] = "not_in_official_vocab"
    train_path.write_text(json.dumps(train_data), encoding="utf-8")

    with pytest.raises(ValueError, match="absent from label_vocab"):
        validate_official_directory(official_root)


def test_missing_assets_and_paper_data_check_are_explicitly_blocked(tmp_path: Path) -> None:
    try:
        inspect_official_assets(official_root=tmp_path / "missing")
    except OfficialAssetsUnavailable as error:
        assert "missing=" in str(error)
    else:
        raise AssertionError("missing official assets must fail closed")

    result = runtime_cli.main(
        ["--mode", "check", "--config", str(PAPER_DATA_CONFIG), "--device", "cpu"]
    )
    assert result["status"] in {"BLOCKED_OFFICIAL_ASSETS", "PASS"}
    assert result["optimizer_steps"] == 0

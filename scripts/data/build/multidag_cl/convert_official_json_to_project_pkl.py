"""Convert Layer 1 MultiDAG-CL JSON features to the project nine-item PKL.

The conversion is structural only: feature lists are copied into NumPy arrays
without extraction, pooling, normalization, standardization, or resampling.
The nine-item compatibility schema cannot represent a distinct dev split, so
dev dialogues are included in ``trainVid`` and their exact membership is kept
in a required split-manifest sidecar.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.data.build.multidag_cl.schema import (
    LABEL_VOCAB_FILENAME,
    MISSING_LABEL_INDEX,
    OFFICIAL_FILENAMES,
    SPLIT_ORDER,
    atomic_write_json,
    atomic_write_pickle,
    compute_sha256,
    load_vocab,
    read_json,
    resolve_official_identities,
)
from scripts.data.build.multidag_cl.validate_official_features import (
    validate_official_directory,
)


def convert_official_directory_to_project_pkl(
    *,
    official_dir: Path,
    output_pkl: Path,
    split_manifest_output: Path,
    fail_if_exists: bool = True,
) -> dict[str, Any]:
    official_dir = Path(official_dir)
    output_pkl = Path(output_pkl)
    split_manifest_output = Path(split_manifest_output)
    if fail_if_exists:
        existing = [path for path in (output_pkl, split_manifest_output) if path.exists()]
        if existing:
            raise FileExistsError(f"Refusing to overwrite Layer 2 artifacts: {existing}.")

    validation = validate_official_directory(
        official_dir,
        require_identifiers=False,
        enforce_mmgcn_session_boundary=False,
    )
    label_vocab = load_vocab(official_dir / LABEL_VOCAB_FILENAME, name="label_vocab")

    video_ids: dict[str, list[str]] = {}
    video_speakers: dict[str, list[str]] = {}
    video_labels: dict[str, list[int]] = {}
    video_text: dict[str, np.ndarray] = {}
    video_audio: dict[str, np.ndarray] = {}
    video_visual: dict[str, np.ndarray] = {}
    video_sentence: dict[str, list[str]] = {}
    split_ids: dict[str, list[str]] = {split: [] for split in SPLIT_ORDER}
    identifier_sources: dict[str, list[str]] = {split: [] for split in SPLIT_ORDER}

    for split in SPLIT_ORDER:
        raw_data = read_json(official_dir / OFFICIAL_FILENAMES[split])
        identities = resolve_official_identities(raw_data, split=split)
        for dialogue, identity in zip(raw_data, identities):
            dialogue_id = str(identity["dialogue_id"])
            if dialogue_id in video_ids:
                raise ValueError(f"Dialogue appears in more than one split: {dialogue_id!r}.")
            split_ids[split].append(dialogue_id)
            identifier_sources[split].append(str(identity["identifier_source"]))
            video_ids[dialogue_id] = list(identity["utterance_ids"])
            video_speakers[dialogue_id] = [str(item["speaker"]) for item in dialogue]
            video_labels[dialogue_id] = [
                (
                    int(label_vocab["stoi"][str(item["label"])])
                    if "label" in item
                    else MISSING_LABEL_INDEX
                )
                for item in dialogue
            ]
            # No dtype is forced here.  JSON numeric values are only regrouped.
            video_text[dialogue_id] = np.asarray([item["cls"][0] for item in dialogue])
            video_audio[dialogue_id] = np.asarray([item["cls"][1] for item in dialogue])
            video_visual[dialogue_id] = np.asarray([item["cls"][2] for item in dialogue])
            video_sentence[dialogue_id] = [str(item["text"]) for item in dialogue]

    train_vid = split_ids["train"] + split_ids["dev"]
    test_vid = list(split_ids["test"])
    project_value = [
        video_ids,
        video_speakers,
        video_labels,
        video_text,
        video_audio,
        video_visual,
        video_sentence,
        train_vid,
        test_vid,
    ]
    atomic_write_pickle(output_pkl, project_value)
    project_pkl_sha256 = compute_sha256(output_pkl)

    split_manifest = {
        "schema": "multidag_cl_official_split_manifest_v1",
        "schema_version": 1,
        "source_format": "multidag_cl_official_json_feature",
        "project_format": "iemocap_nine_item_feature_pkl",
        "train_dialogue_ids": split_ids["train"],
        "dev_dialogue_ids": split_ids["dev"],
        "test_dialogue_ids": split_ids["test"],
        "identifier_sources": identifier_sources,
        "nine_item_trainVid_policy": "train_plus_dev",
        "project_pkl_filename": output_pkl.name,
        "project_pkl_sha256": project_pkl_sha256,
        "feature_value_transform": "NONE",
        "normalization": "NONE",
        "re_extraction": False,
    }
    atomic_write_json(split_manifest_output, split_manifest)

    return {
        "status": "PASS",
        "output_pkl": str(output_pkl),
        "split_manifest": str(split_manifest_output),
        "dialogue_count": len(video_ids),
        "utterance_count": sum(len(value) for value in video_ids.values()),
        "missing_label_utterance_count": validation[
            "missing_label_utterance_count"
        ],
        "trainVid_count": len(train_vid),
        "testVid_count": len(test_vid),
        "project_pkl_sha256": project_pkl_sha256,
        "feature_value_transform": "NONE",
        "source_validation": validation,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-dir", type=Path, required=True)
    parser.add_argument("--output-pkl", type=Path, required=True)
    parser.add_argument("--split-manifest-output", type=Path, required=True)
    parser.add_argument("--allow-overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = convert_official_directory_to_project_pkl(
        official_dir=args.official_dir,
        output_pkl=args.output_pkl,
        split_manifest_output=args.split_manifest_output,
        fail_if_exists=not args.allow_overwrite,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

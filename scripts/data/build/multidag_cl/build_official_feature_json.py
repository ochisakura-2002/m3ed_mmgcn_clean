"""Build the released MultiDAG-CL JSON/vocab interface from aligned features.

This Stage C1 builder does not extract or normalize features.  It joins one
explicit IEMOCAP utterance manifest with three independently produced feature
JSONL files, preserving manifest order and feature values.
"""

from __future__ import annotations

import argparse
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.data.build.multidag_cl.schema import (
    AUDIO_DIM,
    LABEL_VOCAB_FILENAME,
    MODALITY_ORDER,
    OFFICIAL_FILENAMES,
    SPEAKER_VOCAB_FILENAME,
    SPLIT_ORDER,
    TEXT_DIM,
    VISUAL_DIM,
    atomic_write_json,
    atomic_write_pickle,
    load_feature_records,
    load_utterance_manifest,
    read_json,
    validate_official_dialogues,
)


def _ordered_vocab(
    values: Sequence[str],
    *,
    explicit_order: Sequence[str] | None,
    name: str,
) -> dict[str, Any]:
    observed = list(dict.fromkeys(str(value) for value in values))
    if explicit_order is None:
        itos = observed
    else:
        itos = [str(value) for value in explicit_order]
        if len(itos) != len(set(itos)):
            raise ValueError(f"Explicit {name} order contains duplicates.")
        if set(itos) != set(observed):
            raise ValueError(
                f"Explicit {name} order does not exactly cover observed values: "
                f"order={itos}, observed={observed}."
            )
    return {"stoi": {token: index for index, token in enumerate(itos)}, "itos": itos}


def _load_optional_order(path: Path | None, *, name: str) -> list[str] | None:
    if path is None:
        return None
    value = read_json(path)
    if not isinstance(value, list) or not value:
        raise ValueError(f"{name} order file must be a non-empty JSON list.")
    return [str(item) for item in value]


def build_official_artifacts(
    *,
    manifest_path: Path,
    text_feature_path: Path,
    audio_feature_path: Path,
    visual_feature_path: Path,
    output_dir: Path,
    speaker_order: Sequence[str] | None = None,
    label_order: Sequence[str] | None = None,
    fail_if_exists: bool = True,
) -> dict[str, Any]:
    """Join independent modality features into official-loader-compatible files."""

    if speaker_order is None or label_order is None:
        raise ValueError(
            "Exact speaker and label vocab order is not confirmed by the released "
            "source. Provide both explicit orders; Stage C1 will not infer them."
        )
    records = load_utterance_manifest(manifest_path)
    features = {
        "text": load_feature_records(text_feature_path, expected_dim=TEXT_DIM),
        "audio": load_feature_records(audio_feature_path, expected_dim=AUDIO_DIM),
        "visual": load_feature_records(visual_feature_path, expected_dim=VISUAL_DIM),
    }
    manifest_ids = {str(record["utterance_id"]) for record in records}
    for name, feature_map in features.items():
        missing = sorted(manifest_ids - set(feature_map))
        extra = sorted(set(feature_map) - manifest_ids)
        if missing or extra:
            raise ValueError(
                f"{name} feature IDs do not exactly match the manifest; "
                f"missing={missing[:5]}, extra={extra[:5]}."
            )

    split_dialogues: dict[str, OrderedDict[str, list[dict[str, Any]]]] = {
        split: OrderedDict() for split in SPLIT_ORDER
    }
    speaker_values: list[str] = []
    label_values: list[str] = []
    for record in records:
        split = str(record["split"])
        dialogue_id = str(record["dialogue_id"])
        utterance_id = str(record["utterance_id"])
        speaker = str(record["speaker"])
        label = str(record["label"])
        speaker_values.append(speaker)
        label_values.append(label)
        utterance = {
            # The released loader ignores these two identifiers.  They make the
            # official-compatible Layer 1 losslessly convertible to Layer 2.
            "dialogue_id": dialogue_id,
            "utterance_id": utterance_id,
            "text": str(record["text"]),
            "speaker": speaker,
            "label": label,
            "cls": [
                features["text"][utterance_id],
                features["audio"][utterance_id],
                features["visual"][utterance_id],
            ],
        }
        split_dialogues[split].setdefault(dialogue_id, []).append(utterance)

    output_dir = Path(output_dir)
    destinations = [output_dir / OFFICIAL_FILENAMES[split] for split in SPLIT_ORDER]
    destinations += [
        output_dir / SPEAKER_VOCAB_FILENAME,
        output_dir / LABEL_VOCAB_FILENAME,
    ]
    if fail_if_exists:
        existing = [path for path in destinations if path.exists()]
        if existing:
            raise FileExistsError(f"Refusing to overwrite Stage C1 artifacts: {existing}.")

    split_summaries: dict[str, Any] = {}
    for split in SPLIT_ORDER:
        raw_data = list(split_dialogues[split].values())
        if not raw_data:
            raise ValueError(
                f"Manifest has no {split} dialogues; exact official files require all splits."
            )
        split_summaries[split] = validate_official_dialogues(
            raw_data,
            split=split,
            require_identifiers=True,
        )
        atomic_write_json(output_dir / OFFICIAL_FILENAMES[split], raw_data)

    speaker_vocab = _ordered_vocab(
        speaker_values, explicit_order=speaker_order, name="speaker"
    )
    label_vocab = _ordered_vocab(label_values, explicit_order=label_order, name="label")
    atomic_write_pickle(output_dir / SPEAKER_VOCAB_FILENAME, speaker_vocab)
    atomic_write_pickle(output_dir / LABEL_VOCAB_FILENAME, label_vocab)

    return {
        "status": "PASS",
        "output_dir": str(output_dir),
        "modality_order": list(MODALITY_ORDER),
        "dimensions": {"text": TEXT_DIM, "audio": AUDIO_DIM, "visual": VISUAL_DIM},
        "splits": split_summaries,
        "speaker_vocab": speaker_vocab,
        "label_vocab": label_vocab,
        "feature_value_transform": "NONE",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--text-features", type=Path, required=True)
    parser.add_argument("--audio-features", type=Path, required=True)
    parser.add_argument("--visual-features", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--speaker-order-json", type=Path, required=True)
    parser.add_argument("--label-order-json", type=Path, required=True)
    parser.add_argument("--allow-overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = build_official_artifacts(
        manifest_path=args.manifest,
        text_feature_path=args.text_features,
        audio_feature_path=args.audio_features,
        visual_feature_path=args.visual_features,
        output_dir=args.output_dir,
        speaker_order=_load_optional_order(args.speaker_order_json, name="speaker"),
        label_order=_load_optional_order(args.label_order_json, name="label"),
        fail_if_exists=not args.allow_overwrite,
    )
    print(summary)


if __name__ == "__main__":
    main()

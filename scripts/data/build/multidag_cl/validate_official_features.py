"""Validate MultiDAG-CL official-compatible JSON features and vocabularies."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.data.build.multidag_cl.schema import (
    LABEL_VOCAB_FILENAME,
    MODALITY_ORDER,
    OFFICIAL_FILENAMES,
    SPEAKER_VOCAB_FILENAME,
    SPLIT_ORDER,
    load_vocab,
    read_json,
    validate_official_dialogues,
)


def validate_official_directory(
    official_dir: Path,
    *,
    require_identifiers: bool = False,
    enforce_mmgcn_session_boundary: bool = False,
) -> dict[str, Any]:
    official_dir = Path(official_dir)
    speaker_vocab = load_vocab(
        official_dir / SPEAKER_VOCAB_FILENAME, name="speaker_vocab"
    )
    label_vocab = load_vocab(official_dir / LABEL_VOCAB_FILENAME, name="label_vocab")

    summaries: dict[str, Any] = {}
    observed_speakers: set[str] = set()
    observed_labels: set[str] = set()
    observed_utterance_ids: set[str] = set()
    for split in SPLIT_ORDER:
        raw_data = read_json(official_dir / OFFICIAL_FILENAMES[split])
        summaries[split] = validate_official_dialogues(
            raw_data,
            split=split,
            require_identifiers=require_identifiers,
            enforce_mmgcn_session_boundary=enforce_mmgcn_session_boundary,
        )
        for dialogue in raw_data:
            for utterance in dialogue:
                observed_speakers.add(str(utterance["speaker"]))
                observed_labels.add(str(utterance["label"]))
                if "utterance_id" in utterance:
                    utterance_id = str(utterance["utterance_id"])
                    if utterance_id in observed_utterance_ids:
                        raise ValueError(
                            f"Duplicate utterance_id across splits: {utterance_id!r}."
                        )
                    observed_utterance_ids.add(utterance_id)

    if set(speaker_vocab["stoi"]) != observed_speakers:
        raise ValueError(
            "speaker_vocab values do not exactly match feature files: "
            f"vocab={sorted(speaker_vocab['stoi'])}, data={sorted(observed_speakers)}."
        )
    if set(label_vocab["stoi"]) != observed_labels:
        raise ValueError(
            "label_vocab values do not exactly match feature files: "
            f"vocab={sorted(label_vocab['stoi'])}, data={sorted(observed_labels)}."
        )

    return {
        "status": "PASS",
        "official_dir": str(official_dir),
        "modality_order": list(MODALITY_ORDER),
        "splits": summaries,
        "speaker_vocab_size": len(speaker_vocab["itos"]),
        "label_vocab_size": len(label_vocab["itos"]),
        "mmgcn_session_boundary_enforced": bool(enforce_mmgcn_session_boundary),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-dir", type=Path, required=True)
    parser.add_argument("--require-identifiers", action="store_true")
    parser.add_argument("--enforce-mmgcn-session-boundary", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = validate_official_directory(
        args.official_dir,
        require_identifiers=args.require_identifiers,
        enforce_mmgcn_session_boundary=args.enforce_mmgcn_session_boundary,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

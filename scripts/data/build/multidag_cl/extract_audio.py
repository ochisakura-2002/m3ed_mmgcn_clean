"""Extract candidate utterance-level openSMILE IS10 audio features.

MultiDAG-CL directly confirms only the 1582-dimensional audio slot.  MMDAG says
it uses openSMILE following MMGCN, and MMGCN specifies IS10; that chain is
corroborating inference, not direct MultiDAG-CL evidence.  The acknowledgement
flag prevents these features from being mislabeled author-official.
"""

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
    AUDIO_DIM,
    FEATURE_RECORD_SCHEMA,
    atomic_write_json,
    atomic_write_jsonl,
    load_utterance_manifest,
    validate_vector,
)


EVIDENCE_STATUS = "INFERENCE_MMDAG_TO_MMGCN_IS10_NOT_CONFIRMED_BY_MULTIDAG_CL"


def extract_opensmile_compare_2010(
    records: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    try:
        import opensmile
    except ImportError as error:
        raise RuntimeError(
            "Audio extraction requires a separately pinned opensmile environment; "
            "Stage C1 does not install dependencies."
        ) from error

    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.ComParE_2010,
        feature_level=opensmile.FeatureLevel.Functionals,
    )
    outputs: list[dict[str, Any]] = []
    for record in records:
        if "audio_path" not in record:
            raise ValueError(
                f"Manifest utterance {record['utterance_id']!r} has no audio_path."
            )
        audio_path = Path(str(record["audio_path"]))
        if not audio_path.is_file():
            raise FileNotFoundError(f"Utterance WAV not found: {audio_path}")
        frame = smile.process_file(str(audio_path))
        if len(frame) != 1:
            raise ValueError(
                f"IS10 functionals must yield one utterance row; got {len(frame)} for "
                f"{record['utterance_id']!r}."
            )
        feature = validate_vector(
            frame.iloc[0].to_numpy().tolist(),
            expected_dim=AUDIO_DIM,
            location=str(record["utterance_id"]),
        )
        outputs.append({"utterance_id": str(record["utterance_id"]), "feature": feature})
    metadata = {
        "schema": FEATURE_RECORD_SCHEMA,
        "modality": "audio",
        "dimension": AUDIO_DIM,
        "granularity": "utterance",
        "dialogue_context_used": False,
        "feature_set": "ComParE_2010",
        "feature_level": "Functionals",
        "resampling": "NONE_BY_THIS_SCRIPT",
        "opensmile_version": str(getattr(opensmile, "__version__", "unknown")),
        "evidence_status": EVIDENCE_STATUS,
    }
    return outputs, metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata-output", type=Path, required=True)
    parser.add_argument(
        "--acknowledge-inferred-is10-spec",
        action="store_true",
        help="Required because IS10 is not directly stated by MultiDAG-CL.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.acknowledge_inferred_is10_spec:
        raise RuntimeError(
            "Official MultiDAG-CL audio extractor/config is UNKNOWN.  IS10 is only an "
            "upstream inference; pass --acknowledge-inferred-is10-spec to test it."
        )
    records = load_utterance_manifest(args.manifest)
    features, metadata = extract_opensmile_compare_2010(records)
    atomic_write_jsonl(args.output, features)
    atomic_write_json(args.metadata_output, metadata)
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

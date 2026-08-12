"""Inspect the author-released MultiDAG-CL feature assets without changing them."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.data.build.multidag_cl.schema import (
    LABEL_VOCAB_FILENAME,
    MODALITY_DIMS,
    MODALITY_ORDER,
    OFFICIAL_FILENAMES,
    SPEAKER_VOCAB_FILENAME,
    SPLIT_ORDER,
    atomic_write_json,
    compute_sha256,
    load_vocab,
    parse_session_id,
    read_json,
    resolve_official_identities,
)
from scripts.data.build.multidag_cl.validate_official_features import (
    validate_official_directory,
)


ASSET_MANIFEST_SCHEMA = "multidag_cl_official_asset_manifest_v1"


class OfficialAssetsUnavailable(FileNotFoundError):
    """The explicit official root does not contain all five released assets."""


def required_official_paths(official_root: Path) -> dict[str, Path]:
    root = Path(official_root)
    paths = {OFFICIAL_FILENAMES[split]: root / OFFICIAL_FILENAMES[split] for split in SPLIT_ORDER}
    paths[SPEAKER_VOCAB_FILENAME] = root / SPEAKER_VOCAB_FILENAME
    paths[LABEL_VOCAB_FILENAME] = root / LABEL_VOCAB_FILENAME
    return paths


def assert_official_assets_available(official_root: Path) -> dict[str, Path]:
    paths = required_official_paths(official_root)
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise OfficialAssetsUnavailable(
            f"Official MultiDAG-CL assets are unavailable under {Path(official_root)}; "
            f"missing={missing}."
        )
    return paths


def _session_distribution(dialogues: list[dict[str, Any]]) -> dict[str, Any]:
    if any(
        item["identifier_source"] != "embedded_official_json_fields"
        for item in dialogues
    ):
        return {
            "available": False,
            "reason": "official JSON does not expose original dialogue IDs",
            "counts": None,
        }
    sessions: list[str] = []
    for item in dialogues:
        try:
            sessions.append(parse_session_id(item["dialogue_id"]))
        except ValueError:
            return {
                "available": False,
                "reason": "embedded dialogue IDs are not parseable IEMOCAP Session IDs",
                "counts": None,
            }
    return {
        "available": True,
        "reason": None,
        "counts": dict(sorted(Counter(sessions).items())),
    }


def inspect_official_assets(
    *,
    official_root: Path,
    manifest_output: Path | None = None,
    fail_if_exists: bool = True,
) -> dict[str, Any]:
    """Validate and inventory an explicit official asset directory."""

    official_root = Path(official_root)
    paths = assert_official_assets_available(official_root)
    if manifest_output is not None and fail_if_exists and Path(manifest_output).exists():
        raise FileExistsError(f"Refusing to overwrite manifest: {manifest_output}")

    validation = validate_official_directory(
        official_root,
        require_identifiers=False,
        enforce_mmgcn_session_boundary=False,
    )
    speaker_vocab = load_vocab(
        official_root / SPEAKER_VOCAB_FILENAME, name="speaker_vocab"
    )
    label_vocab = load_vocab(
        official_root / LABEL_VOCAB_FILENAME, name="label_vocab"
    )

    split_manifests: dict[str, Any] = {}
    all_dialogue_ids: set[str] = set()
    all_utterance_ids: set[str] = set()
    for split in SPLIT_ORDER:
        raw_data = read_json(official_root / OFFICIAL_FILENAMES[split])
        dialogues = resolve_official_identities(raw_data, split=split)
        dialogue_ids = [item["dialogue_id"] for item in dialogues]
        utterance_ids = [
            utterance_id
            for item in dialogues
            for utterance_id in item["utterance_ids"]
        ]
        overlap = all_dialogue_ids.intersection(dialogue_ids)
        if overlap:
            raise ValueError(f"Dialogue IDs overlap across official splits: {sorted(overlap)}")
        utterance_overlap = all_utterance_ids.intersection(utterance_ids)
        if utterance_overlap:
            raise ValueError(
                f"Utterance IDs overlap across official splits: {sorted(utterance_overlap)}"
            )
        all_dialogue_ids.update(dialogue_ids)
        all_utterance_ids.update(utterance_ids)
        split_manifests[split] = {
            "dialogue_count": validation["splits"][split]["dialogue_count"],
            "utterance_count": validation["splits"][split]["utterance_count"],
            "dialogue_ids": dialogue_ids,
            "dialogue_order": dialogues,
            "session_distribution": _session_distribution(dialogues),
        }

    manifest = {
        "schema": ASSET_MANIFEST_SCHEMA,
        "status": "PASS",
        "source": "author_released_pre_extracted_multidag_cl_assets",
        "official_root": str(official_root),
        "source_files": {
            name: {"sha256": compute_sha256(path), "size_bytes": path.stat().st_size}
            for name, path in paths.items()
        },
        "modality_order": list(MODALITY_ORDER),
        "dimensions": dict(zip(MODALITY_ORDER, MODALITY_DIMS)),
        "total_dimension": sum(MODALITY_DIMS),
        "all_vectors_finite": True,
        "splits": split_manifests,
        "speaker_vocab": speaker_vocab,
        "label_vocab": label_vocab,
        "identifier_policy": {
            "embedded": "preserved verbatim when both fields are present",
            "missing": "surrogate IDs generated only from official split and source position",
            "session_inference_from_surrogate_ids": False,
        },
        "feature_value_transform": "NONE",
        "normalization": "NONE",
        "re_extraction": False,
    }
    if manifest_output is not None:
        atomic_write_json(Path(manifest_output), manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-root", type=Path, required=True)
    parser.add_argument(
        "--manifest-output",
        type=Path,
        default=Path("official_asset_manifest.json"),
    )
    parser.add_argument("--allow-overwrite", action="store_true")
    return parser.parse_args()


def main() -> dict[str, Any]:
    args = parse_args()
    try:
        result = inspect_official_assets(
            official_root=args.official_root,
            manifest_output=args.manifest_output,
            fail_if_exists=not args.allow_overwrite,
        )
    except OfficialAssetsUnavailable as error:
        result = {
            "status": "BLOCKED_OFFICIAL_ASSETS",
            "official_root": str(args.official_root),
            "manifest_created": False,
            "error": str(error),
        }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


if __name__ == "__main__":
    main()

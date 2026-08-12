"""Run an explicit visual backend against the Stage C1 utterance contract.

The official MultiDAG-CL materials do not identify the DenseNet variant,
checkpoint, face detector/alignment, sampled frames, layer, or temporal pooling.
Accordingly this file supplies an independent, fail-closed 342-D backend
interface instead of inventing an extractor.
"""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
import sys
from typing import Any, Callable

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.data.build.multidag_cl.schema import (
    FEATURE_RECORD_SCHEMA,
    VISUAL_DIM,
    atomic_write_json,
    atomic_write_jsonl,
    load_utterance_manifest,
    validate_vector,
)


EVIDENCE_STATUS = "UNKNOWN_MULTIDAG_CL_VISUAL_EXTRACTION_SPEC"


def load_backend(specification: str) -> Callable[[dict[str, Any]], Any]:
    if ":" not in specification:
        raise ValueError("Visual backend must use module.path:function syntax.")
    module_name, function_name = specification.rsplit(":", maxsplit=1)
    function = getattr(importlib.import_module(module_name), function_name)
    if not callable(function):
        raise TypeError(f"Visual backend is not callable: {specification}")
    return function


def extract_with_backend(
    records: list[dict[str, Any]],
    *,
    backend: Callable[[dict[str, Any]], Any],
    backend_specification: str,
    checkpoint_provenance: str,
    preprocessing_specification: str,
    frame_pooling: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    outputs: list[dict[str, Any]] = []
    for record in records:
        for key in ("video_path", "start_time", "end_time"):
            if key not in record:
                raise ValueError(
                    f"Manifest utterance {record['utterance_id']!r} lacks {key}."
                )
        feature = validate_vector(
            list(backend(dict(record))),
            expected_dim=VISUAL_DIM,
            location=str(record["utterance_id"]),
        )
        outputs.append({"utterance_id": str(record["utterance_id"]), "feature": feature})
    metadata = {
        "schema": FEATURE_RECORD_SCHEMA,
        "modality": "visual",
        "dimension": VISUAL_DIM,
        "output_granularity": "utterance",
        "dialogue_context_used": False,
        "backend": backend_specification,
        "checkpoint_provenance": checkpoint_provenance,
        "preprocessing": preprocessing_specification,
        "frame_pooling": frame_pooling,
        "evidence_status": EVIDENCE_STATUS,
    }
    return outputs, metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--backend", required=True)
    parser.add_argument("--checkpoint-provenance", required=True)
    parser.add_argument("--preprocessing-specification", required=True)
    parser.add_argument("--frame-pooling", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata-output", type=Path, required=True)
    parser.add_argument(
        "--acknowledge-unknown-official-visual-spec",
        action="store_true",
        help="Required: this backend is a candidate, not author-official evidence.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.acknowledge_unknown_official_visual_spec:
        raise RuntimeError(
            "Official MultiDAG-CL visual preprocessing/pooling is UNKNOWN; pass "
            "--acknowledge-unknown-official-visual-spec only for a documented candidate."
        )
    records = load_utterance_manifest(args.manifest)
    backend = load_backend(args.backend)
    features, metadata = extract_with_backend(
        records,
        backend=backend,
        backend_specification=args.backend,
        checkpoint_provenance=args.checkpoint_provenance,
        preprocessing_specification=args.preprocessing_specification,
        frame_pooling=args.frame_pooling,
    )
    atomic_write_jsonl(args.output, features)
    atomic_write_json(args.metadata_output, metadata)
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

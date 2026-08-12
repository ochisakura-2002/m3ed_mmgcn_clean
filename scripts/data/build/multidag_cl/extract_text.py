"""Extract independent utterance text features for the Stage C1 interface.

Important provenance boundary: the MultiDAG-CL release confirms a RoBERTa-named
1024-dimensional artifact, but does not release its feature-extraction script or
fine-tuned checkpoint.  DAG-ERC documents a fine-tuned RoBERTa-Large pooler
pipeline.  This implementation therefore requires an explicit local checkpoint
and acknowledgement; its output is not author-official until that checkpoint is
identified and verified.
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
    FEATURE_RECORD_SCHEMA,
    TEXT_DIM,
    atomic_write_json,
    atomic_write_jsonl,
    load_utterance_manifest,
    validate_vector,
)


EVIDENCE_STATUS = "INFERENCE_FROM_DAG_ERC_LINEAGE_CHECKPOINT_UNKNOWN"


def extract_with_local_roberta_pooler(
    records: list[dict[str, Any]],
    *,
    model_dir: Path,
    batch_size: int,
    max_length: int,
    device: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    try:
        import torch
        import transformers
        from transformers import AutoModel, AutoTokenizer
    except ImportError as error:
        raise RuntimeError(
            "Text extraction requires the separately managed PyTorch/Transformers "
            "feature environment."
        ) from error

    model_dir = Path(model_dir)
    if not model_dir.is_dir():
        raise FileNotFoundError(f"Local RoBERTa checkpoint directory not found: {model_dir}")
    if batch_size <= 0 or max_length <= 0:
        raise ValueError("batch_size and max_length must be positive.")
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
    model = AutoModel.from_pretrained(str(model_dir), local_files_only=True)
    if str(getattr(model.config, "model_type", "")).lower() != "roberta":
        raise ValueError("Stage C1 text checkpoint must identify model_type=roberta.")
    if int(getattr(model.config, "hidden_size", -1)) != TEXT_DIM:
        raise ValueError(f"RoBERTa checkpoint hidden_size must be {TEXT_DIM}.")

    torch_device = torch.device(device)
    model.to(torch_device)
    model.eval()
    outputs: list[dict[str, Any]] = []
    with torch.no_grad():
        for start in range(0, len(records), batch_size):
            batch = records[start : start + batch_size]
            encoded = tokenizer(
                [str(record["text"]) for record in batch],
                add_special_tokens=True,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(torch_device) for key, value in encoded.items()}
            model_output = model(**encoded)
            pooled = getattr(model_output, "pooler_output", None)
            if pooled is None:
                raise ValueError(
                    "Checkpoint returned no pooler_output; Stage C1 does not silently "
                    "substitute mean pooling or first-token hidden state."
                )
            array = pooled.detach().to(device="cpu", dtype=torch.float32).numpy()
            if array.shape != (len(batch), TEXT_DIM):
                raise ValueError(f"Unexpected text feature shape: {array.shape}.")
            for record, feature in zip(batch, array):
                values = validate_vector(
                    feature.tolist(),
                    expected_dim=TEXT_DIM,
                    location=str(record["utterance_id"]),
                )
                outputs.append(
                    {"utterance_id": str(record["utterance_id"]), "feature": values}
                )

    metadata = {
        "schema": FEATURE_RECORD_SCHEMA,
        "modality": "text",
        "dimension": TEXT_DIM,
        "granularity": "utterance",
        "dialogue_context_used": False,
        "pooling": "model_pooler_output",
        "max_length": int(max_length),
        "checkpoint_local_path": str(model_dir),
        "model_type": str(model.config.model_type),
        "transformers_version": str(transformers.__version__),
        "evidence_status": EVIDENCE_STATUS,
    }
    return outputs, metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-provenance", required=True)
    parser.add_argument("--max-length", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata-output", type=Path, required=True)
    parser.add_argument(
        "--acknowledge-unresolved-official-checkpoint",
        action="store_true",
        help="Required: output remains inferred until the official checkpoint is verified.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.acknowledge_unresolved_official_checkpoint:
        raise RuntimeError(
            "Official MultiDAG-CL text checkpoint/preprocessing is unresolved; pass "
            "--acknowledge-unresolved-official-checkpoint only for an explicitly "
            "provenanced candidate extraction."
        )
    records = load_utterance_manifest(args.manifest)
    features, metadata = extract_with_local_roberta_pooler(
        records,
        model_dir=args.model_dir,
        batch_size=args.batch_size,
        max_length=args.max_length,
        device=args.device,
    )
    metadata["checkpoint_provenance"] = str(args.checkpoint_provenance)
    atomic_write_jsonl(args.output, features)
    atomic_write_json(args.metadata_output, metadata)
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

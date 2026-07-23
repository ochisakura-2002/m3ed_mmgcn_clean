"""Build the immutable IEMOCAP Clean RoBERTa v1 text-feature PKL.

The transformer is loaded exclusively from ``--model-dir`` with offline mode
enabled.  Only item 3 (``videoText``) of the legacy nine-item PKL is replaced.
"""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import pickle
import platform
import random
import sys
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
import uuid

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.run_metadata import compute_file_sha256  # noqa: E402


MODEL_ID = "FacebookAI/roberta-base"
MODEL_REVISION = "c8b8a37ce3afa8b16a98ff5d0016c157a16ef432"
FEATURE_SET_NAME = "iemocap_clean_roberta_base_utterance_mean_v1"
TEXT_FEATURE_DIM = 768
SCHEMA_VERSION = 1
CAUSALITY_SCOPE = "utterance_only_no_dialogue_context"
LABEL_USAGE = "labels_not_used_for_feature_extraction"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-pkl", required=True)
    parser.add_argument("--expected-input-sha256", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--model-revision", default=MODEL_REVISION)
    parser.add_argument("--output-pkl", required=True)
    parser.add_argument("--pooling", choices=("mean",), default="mean")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--metadata-output", default=None)
    parser.add_argument("--sha256-output", default=None)
    parser.add_argument(
        "--fail-if-output-exists",
        action="store_true",
        default=True,
        help="Refuse an existing output (enabled by default for immutable v1).",
    )
    return parser.parse_args()


def set_deterministic_seed(seed: int) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def decode_sentence(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def load_nine_item_pkl(path: Path) -> list[Any] | tuple[Any, ...]:
    with Path(path).open("rb") as file:
        try:
            value = pickle.load(file, encoding="latin1")
        except TypeError:
            file.seek(0)
            value = pickle.load(file)
    validate_nine_item_structure(value)
    return value


def validate_nine_item_structure(value: Any) -> None:
    if not isinstance(value, (list, tuple)) or len(value) != 9:
        raise ValueError("IEMOCAP feature PKL must be a list/tuple of exactly nine items.")
    names = (
        "videoIDs",
        "videoSpeakers",
        "videoLabels",
        "videoText",
        "videoAudio",
        "videoVisual",
        "videoSentence",
    )
    mappings = value[:7]
    for name, mapping in zip(names, mappings):
        if not isinstance(mapping, Mapping):
            raise TypeError(f"{name} must be a dialogue mapping.")
    dialogue_ids = set(mappings[0])
    if not dialogue_ids:
        raise ValueError("IEMOCAP feature PKL contains no dialogues.")
    for name, mapping in zip(names[1:], mappings[1:]):
        if set(mapping) != dialogue_ids:
            raise ValueError(f"{name} dialogue IDs do not match videoIDs.")
    for dialogue_id in mappings[0]:
        expected = len(mappings[0][dialogue_id])
        if expected <= 0:
            raise ValueError(f"Dialogue {dialogue_id!r} is empty.")
        for name, mapping in zip(names[1:], mappings[1:]):
            if len(mapping[dialogue_id]) != expected:
                raise ValueError(
                    f"Dialogue {dialogue_id!r} length mismatch: "
                    f"videoIDs={expected}, {name}={len(mapping[dialogue_id])}."
                )
    if not isinstance(value[7], (list, tuple, set)) or not isinstance(
        value[8], (list, tuple, set)
    ):
        raise TypeError("trainVid and testVid must be list/tuple/set collections.")


def verify_sha256(path: Path, expected: str) -> str:
    actual = compute_file_sha256(Path(path)).lower()
    expected_normalized = str(expected).strip().lower()
    if actual != expected_normalized:
        raise ValueError(
            f"Input PKL SHA256 mismatch: expected={expected_normalized}, actual={actual}."
        )
    return actual


def mean_pool_without_padding_or_special(
    hidden_state: torch.Tensor,
    attention_mask: torch.Tensor,
    special_tokens_mask: torch.Tensor,
) -> torch.Tensor:
    """Mean-pool only non-padding, non-special token positions."""

    if hidden_state.ndim != 3:
        raise ValueError("hidden_state must have shape [B, T, D].")
    if attention_mask.shape != hidden_state.shape[:2]:
        raise ValueError("attention_mask shape must match hidden_state [B, T].")
    if special_tokens_mask.shape != hidden_state.shape[:2]:
        raise ValueError("special_tokens_mask shape must match hidden_state [B, T].")
    valid = attention_mask.to(dtype=torch.bool) & ~special_tokens_mask.to(dtype=torch.bool)
    counts = valid.sum(dim=1)
    if torch.any(counts == 0):
        bad = torch.nonzero(counts == 0, as_tuple=False).flatten().tolist()
        raise ValueError(f"Mean pooling found zero valid tokens for batch rows {bad}.")
    weights = valid.to(dtype=hidden_state.dtype).unsqueeze(-1)
    return (hidden_state * weights).sum(dim=1) / counts.to(hidden_state.dtype).unsqueeze(-1)


def _input_lengths(tokenizer: Any, texts: Sequence[str]) -> List[int]:
    encoded = tokenizer(
        list(texts),
        add_special_tokens=True,
        truncation=False,
        padding=False,
    )
    input_ids = encoded["input_ids"]
    if isinstance(input_ids, torch.Tensor):
        if input_ids.ndim != 2:
            raise ValueError("Unexpected tokenizer input_ids shape for truncation audit.")
        return [int(row.ne(getattr(tokenizer, "pad_token_id", -1)).sum()) for row in input_ids]
    return [len(row) for row in input_ids]


def extract_text_features(
    texts: Sequence[str],
    *,
    tokenizer: Any,
    model: Any,
    device: torch.device,
    batch_size: int,
    max_length: int,
) -> Tuple[np.ndarray, int]:
    """Encode independent utterances and return float32 pooled features."""

    if int(batch_size) <= 0 or int(max_length) <= 0:
        raise ValueError("batch_size and max_length must be positive.")
    model.eval()
    feature_batches: List[np.ndarray] = []
    truncated_count = 0
    with torch.no_grad():
        for start in range(0, len(texts), int(batch_size)):
            batch_texts = list(texts[start : start + int(batch_size)])
            truncated_count += sum(
                length > int(max_length) for length in _input_lengths(tokenizer, batch_texts)
            )
            encoded = tokenizer(
                batch_texts,
                add_special_tokens=True,
                padding=True,
                truncation=True,
                max_length=int(max_length),
                return_special_tokens_mask=True,
                return_tensors="pt",
            )
            if "special_tokens_mask" not in encoded:
                raise ValueError("Tokenizer did not return special_tokens_mask.")
            special_mask = encoded["special_tokens_mask"].to(device)
            model_inputs = {
                key: value.to(device)
                for key, value in encoded.items()
                if key != "special_tokens_mask"
            }
            output = model(**model_inputs)
            hidden = (
                output.last_hidden_state
                if hasattr(output, "last_hidden_state")
                else output[0]
            )
            pooled = mean_pool_without_padding_or_special(
                hidden,
                model_inputs["attention_mask"],
                special_mask,
            )
            array = pooled.detach().to(device="cpu", dtype=torch.float32).numpy()
            if array.ndim != 2 or array.shape[1] != TEXT_FEATURE_DIM:
                raise ValueError(
                    f"RoBERTa text output must have dimension {TEXT_FEATURE_DIM}; "
                    f"got {array.shape}."
                )
            if not np.isfinite(array).all():
                raise ValueError("RoBERTa text output contains NaN or Inf.")
            feature_batches.append(array)
    if not feature_batches:
        raise ValueError("No utterances were supplied for feature extraction.")
    return np.concatenate(feature_batches, axis=0).astype(np.float32, copy=False), truncated_count


def _deep_equal(left: Any, right: Any) -> bool:
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        left_array = np.asarray(left)
        right_array = np.asarray(right)
        return (
            left_array.shape == right_array.shape
            and left_array.dtype == right_array.dtype
            and np.array_equal(left_array, right_array, equal_nan=True)
        )
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return set(left) == set(right) and all(_deep_equal(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return type(left) is type(right) and len(left) == len(right) and all(
            _deep_equal(a, b) for a, b in zip(left, right)
        )
    if isinstance(left, set) and isinstance(right, set):
        return left == right
    try:
        result = left == right
        return bool(np.all(result)) if isinstance(result, np.ndarray) else bool(result)
    except Exception:
        return False


def self_check_output(source: Sequence[Any], candidate: Sequence[Any]) -> None:
    validate_nine_item_structure(candidate)
    for index in (0, 1, 2, 4, 5, 6, 7, 8):
        if not _deep_equal(source[index], candidate[index]):
            raise ValueError(f"Output self-check failed: top-level item {index} changed.")
    for dialogue_id in source[0]:
        text = np.asarray(candidate[3][dialogue_id])
        if text.dtype != np.float32 or text.ndim != 2 or text.shape[1] != TEXT_FEATURE_DIM:
            raise ValueError(
                f"Output text feature contract failed for {dialogue_id!r}: "
                f"shape={text.shape}, dtype={text.dtype}."
            )
        if len(text) != len(source[0][dialogue_id]) or not np.isfinite(text).all():
            raise ValueError(f"Output text features invalid for {dialogue_id!r}.")


def _atomic_pickle(value: Any, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("wb") as file:
            pickle.dump(value, file, protocol=pickle.HIGHEST_PROTOCOL)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_text(text: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as file:
            file.write(text)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def build_clean_feature_pkl(
    *,
    input_pkl: Path,
    expected_input_sha256: str,
    output_pkl: Path,
    tokenizer: Any,
    model: Any,
    model_local_path: Path,
    model_id: str = MODEL_ID,
    model_revision: str = MODEL_REVISION,
    pooling: str = "mean",
    batch_size: int = 32,
    max_length: int = 512,
    device: str = "cpu",
    seed: int = 42,
    metadata_output: Optional[Path] = None,
    sha256_output: Optional[Path] = None,
    fail_if_output_exists: bool = True,
    transformers_version: str = "unknown",
) -> Dict[str, Any]:
    """Build and atomically publish one offline clean-feature artifact."""

    input_pkl = Path(input_pkl)
    output_pkl = Path(output_pkl)
    metadata_output = metadata_output or output_pkl.with_suffix(output_pkl.suffix + ".metadata.json")
    sha256_output = sha256_output or output_pkl.with_suffix(output_pkl.suffix + ".sha256")
    if pooling != "mean":
        raise ValueError("Clean RoBERTa v1 supports pooling=mean only.")
    if model_id != MODEL_ID or model_revision != MODEL_REVISION:
        raise ValueError("Clean RoBERTa v1 model_id/revision are immutable protocol fields.")
    for destination in (output_pkl, Path(metadata_output), Path(sha256_output)):
        if fail_if_output_exists and destination.exists():
            raise FileExistsError(f"Refusing to overwrite existing v1 artifact: {destination}")

    source_sha = verify_sha256(input_pkl, expected_input_sha256)
    source = load_nine_item_pkl(input_pkl)
    dialogue_order = list(source[0].keys())
    sentences: List[str] = []
    boundaries: Dict[Any, Tuple[int, int]] = {}
    empty_count = 0
    unk_token = str(getattr(tokenizer, "unk_token", "") or "<unk>")
    for dialogue_id in dialogue_order:
        start = len(sentences)
        for raw_sentence in source[6][dialogue_id]:
            sentence = decode_sentence(raw_sentence)
            if not sentence.strip():
                sentence = unk_token
                empty_count += 1
            sentences.append(sentence)
        boundaries[dialogue_id] = (start, len(sentences))

    set_deterministic_seed(seed)
    used_device = torch.device(device)
    features, truncated_count = extract_text_features(
        sentences,
        tokenizer=tokenizer,
        model=model,
        device=used_device,
        batch_size=batch_size,
        max_length=max_length,
    )
    video_text = {
        dialogue_id: features[start:end].copy()
        for dialogue_id, (start, end) in boundaries.items()
    }
    copied = list(copy.deepcopy(source))
    copied[3] = video_text
    candidate: list[Any] | tuple[Any, ...] = tuple(copied) if isinstance(source, tuple) else copied
    self_check_output(source, candidate)
    _atomic_pickle(candidate, output_pkl)
    reloaded = load_nine_item_pkl(output_pkl)
    self_check_output(source, reloaded)
    output_sha = compute_file_sha256(output_pkl)

    first_id = dialogue_order[0]
    metadata: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "feature_set_name": FEATURE_SET_NAME,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_pkl_path": str(input_pkl),
        "source_pkl_sha256": source_sha,
        "output_pkl_sha256": output_sha,
        "model_id": model_id,
        "model_revision": model_revision,
        "model_local_path": str(model_local_path),
        "transformers_version": str(transformers_version),
        "torch_version": str(torch.__version__),
        "python_version": platform.python_version(),
        "pooling": pooling,
        "max_length": int(max_length),
        "batch_size": int(batch_size),
        "dtype": "float32",
        "text_feature_dim": TEXT_FEATURE_DIM,
        "audio_feature_dim": int(np.asarray(source[4][first_id]).shape[-1]),
        "visual_feature_dim": int(np.asarray(source[5][first_id]).shape[-1]),
        "dialogue_count": len(dialogue_order),
        "utterance_count": len(sentences),
        "empty_sentence_count": int(empty_count),
        "truncated_sentence_count": int(truncated_count),
        "device_used": str(used_device),
        "seed": int(seed),
        "causality_scope": CAUSALITY_SCOPE,
        "label_usage": LABEL_USAGE,
    }
    _atomic_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", Path(metadata_output))
    _atomic_text(f"{output_sha}  {output_pkl.name}\n", Path(sha256_output))
    return metadata


def load_local_transformer(model_dir: Path, model_revision: str) -> tuple[Any, Any, str]:
    """Load tokenizer/model with both library and process offline safeguards."""

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    model_dir = Path(model_dir)
    if not model_dir.is_dir():
        raise FileNotFoundError(f"Local RoBERTa model directory not found: {model_dir}")
    try:
        import transformers
        from transformers import AutoModel, AutoTokenizer
    except ImportError as error:
        raise RuntimeError(
            "transformers is required; install the pinned feature environment without "
            "downloading model weights at runtime."
        ) from error
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_dir), revision=model_revision, local_files_only=True
    )
    model = AutoModel.from_pretrained(
        str(model_dir), revision=model_revision, local_files_only=True
    )
    return tokenizer, model, str(transformers.__version__)


def main() -> None:
    args = parse_args()
    if args.model_id != MODEL_ID or args.model_revision != MODEL_REVISION:
        raise ValueError("This v1 builder only accepts the pinned model ID and revision.")
    set_deterministic_seed(args.seed)
    tokenizer, model, transformers_version = load_local_transformer(
        Path(args.model_dir), args.model_revision
    )
    requested_device = str(args.device)
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"Requested device {requested_device!r}, but CUDA is unavailable.")
    model = model.to(torch.device(requested_device))
    metadata = build_clean_feature_pkl(
        input_pkl=Path(args.input_pkl),
        expected_input_sha256=args.expected_input_sha256,
        output_pkl=Path(args.output_pkl),
        tokenizer=tokenizer,
        model=model,
        model_local_path=Path(args.model_dir),
        model_id=args.model_id,
        model_revision=args.model_revision,
        pooling=args.pooling,
        batch_size=args.batch_size,
        max_length=args.max_length,
        device=requested_device,
        seed=args.seed,
        metadata_output=None if args.metadata_output is None else Path(args.metadata_output),
        sha256_output=None if args.sha256_output is None else Path(args.sha256_output),
        fail_if_output_exists=args.fail_if_output_exists,
        transformers_version=transformers_version,
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

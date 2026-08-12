"""Shared, fail-closed contracts for MultiDAG-CL Stage C1 features.

The released loader concatenates ``cls[0] + cls[1] + cls[2]``.  The released
README identifies those positions as text, audio, and visual with dimensions
1024, 1582, and 342.  This module freezes only that confirmed interface; it
does not claim that the original extractor checkpoints or preprocessing are
known.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
import hashlib
import json
import math
from numbers import Real
from pathlib import Path
import pickle
import re
from typing import Any


TEXT_DIM = 1024
AUDIO_DIM = 1582
VISUAL_DIM = 342
MODALITY_ORDER = ("text", "audio", "visual")
MODALITY_DIMS = (TEXT_DIM, AUDIO_DIM, VISUAL_DIM)
SPLIT_ORDER = ("train", "dev", "test")
OFFICIAL_FILENAMES = {
    split: f"{split}_data_roberta_mm.json.feature" for split in SPLIT_ORDER
}
SPEAKER_VOCAB_FILENAME = "speaker_vocab.pkl"
LABEL_VOCAB_FILENAME = "label_vocab.pkl"
OFFICIAL_ASSET_MANIFEST_FILENAME = "official_asset_manifest.json"
OFFICIAL_SPLIT_MANIFEST_FILENAME = "official_split_manifest.json"
PROJECT_PKL_FILENAME = "IEMOCAP_features_multidag_cl_official.pkl"
FEATURE_RECORD_SCHEMA = "multidag_cl_modality_feature_jsonl_v1"
UTTERANCE_MANIFEST_SCHEMA = "multidag_cl_iemocap_utterance_manifest_v1"

_SESSION_RE = re.compile(r"^(Ses0[1-5])", flags=re.IGNORECASE)


def read_json(path: Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as file:
        return json.load(file)


def compute_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: Path, value: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as file:
        json.dump(value, file, ensure_ascii=False, indent=2)
        file.write("\n")
    temporary.replace(destination)


def atomic_write_jsonl(path: Path, records: Iterable[Mapping[str, Any]]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as file:
        for record in records:
            file.write(json.dumps(dict(record), ensure_ascii=False) + "\n")
    temporary.replace(destination)


def atomic_write_pickle(path: Path, value: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    with temporary.open("wb") as file:
        pickle.dump(value, file, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(destination)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"{path}:{line_number} must contain a JSON object.")
            records.append(value)
    if not records:
        raise ValueError(f"JSONL file contains no records: {path}")
    return records


def load_utterance_manifest(path: Path) -> list[dict[str, Any]]:
    records = load_jsonl(path)
    required = {
        "split",
        "dialogue_id",
        "utterance_id",
        "text",
        "speaker",
        "label",
    }
    seen_utterances: set[str] = set()
    closed_dialogues: set[tuple[str, str]] = set()
    current_dialogue: tuple[str, str] | None = None
    for index, record in enumerate(records):
        missing = required - set(record)
        if missing:
            raise ValueError(f"Manifest record {index} is missing {sorted(missing)}.")
        split = str(record["split"]).lower()
        if split not in SPLIT_ORDER:
            raise ValueError(
                f"Manifest record {index} has split={split!r}; expected {SPLIT_ORDER}."
            )
        record["split"] = split
        dialogue_id = str(record["dialogue_id"])
        utterance_id = str(record["utterance_id"])
        if utterance_id in seen_utterances:
            raise ValueError(f"Duplicate utterance_id in manifest: {utterance_id!r}.")
        seen_utterances.add(utterance_id)
        dialogue_key = (split, dialogue_id)
        if dialogue_key != current_dialogue:
            if dialogue_key in closed_dialogues:
                raise ValueError(
                    "Manifest dialogue records must be contiguous; repeated block for "
                    f"{dialogue_key}."
                )
            if current_dialogue is not None:
                closed_dialogues.add(current_dialogue)
            current_dialogue = dialogue_key
    return records


def load_feature_records(path: Path, *, expected_dim: int) -> dict[str, list[float]]:
    feature_map: dict[str, list[float]] = {}
    for index, record in enumerate(load_jsonl(path)):
        if set(("utterance_id", "feature")) - set(record):
            raise ValueError(
                f"Feature record {index} in {path} requires utterance_id and feature."
            )
        utterance_id = str(record["utterance_id"])
        if utterance_id in feature_map:
            raise ValueError(f"Duplicate feature for utterance_id={utterance_id!r}.")
        feature = validate_vector(
            record["feature"],
            expected_dim=expected_dim,
            location=f"{path}:{index}:{utterance_id}",
        )
        feature_map[utterance_id] = feature
    return feature_map


def validate_vector(value: Any, *, expected_dim: int, location: str) -> list[float]:
    if not isinstance(value, list):
        raise TypeError(f"{location} feature must be a JSON list.")
    if len(value) != int(expected_dim):
        raise ValueError(
            f"{location} feature dimension must be {expected_dim}; got {len(value)}."
        )
    normalized: list[float] = []
    for component_index, component in enumerate(value):
        if isinstance(component, bool) or not isinstance(component, Real):
            raise TypeError(
                f"{location}[{component_index}] must be a finite real number."
            )
        number = float(component)
        if not math.isfinite(number):
            raise ValueError(f"{location}[{component_index}] is NaN or Inf.")
        normalized.append(number)
    return normalized


def parse_session_id(dialogue_id: str) -> str:
    match = _SESSION_RE.match(str(dialogue_id))
    if match is None:
        raise ValueError(f"Cannot parse IEMOCAP session from {dialogue_id!r}.")
    return match.group(1).capitalize()


def resolve_official_identities(
    raw_data: Any,
    *,
    split: str,
) -> list[dict[str, Any]]:
    """Resolve embedded IDs or explicit position-based surrogate IDs.

    The released loader does not require identifiers.  Surrogate IDs preserve
    exact split membership and source order but deliberately carry no IEMOCAP
    Session claim.
    """

    split = str(split).lower()
    if split not in SPLIT_ORDER:
        raise ValueError(f"Unsupported split={split!r}.")
    if not isinstance(raw_data, list):
        raise TypeError(f"{split} feature data must be a dialogue list.")

    resolved: list[dict[str, Any]] = []
    for dialogue_index, dialogue in enumerate(raw_data):
        if not isinstance(dialogue, list) or not dialogue:
            raise ValueError(f"{split} dialogue {dialogue_index} must be non-empty.")
        dialogue_presence = ["dialogue_id" in item for item in dialogue]
        utterance_presence = ["utterance_id" in item for item in dialogue]
        if any(dialogue_presence) != all(dialogue_presence):
            raise ValueError(
                f"{split} dialogue {dialogue_index} partially defines dialogue_id."
            )
        if any(utterance_presence) != all(utterance_presence):
            raise ValueError(
                f"{split} dialogue {dialogue_index} partially defines utterance_id."
            )
        if all(dialogue_presence) != all(utterance_presence):
            raise ValueError(
                f"{split} dialogue {dialogue_index} must define both identifier fields "
                "or neither."
            )

        if all(dialogue_presence):
            dialogue_ids = [str(item["dialogue_id"]) for item in dialogue]
            if len(set(dialogue_ids)) != 1:
                raise ValueError(
                    f"{split} dialogue {dialogue_index} changes dialogue_id within dialogue."
                )
            dialogue_id = dialogue_ids[0]
            utterance_ids = [str(item["utterance_id"]) for item in dialogue]
            if len(set(utterance_ids)) != len(utterance_ids):
                raise ValueError(
                    f"{split} dialogue {dialogue_index} has duplicate utterance_id values."
                )
            source = "embedded_official_json_fields"
        else:
            dialogue_id = f"multidag_cl::{split}::dialogue::{dialogue_index:06d}"
            utterance_ids = [
                f"{dialogue_id}::utterance::{utterance_index:06d}"
                for utterance_index in range(len(dialogue))
            ]
            source = "generated_from_official_split_and_position"

        resolved.append(
            {
                "dialogue_index": dialogue_index,
                "dialogue_id": dialogue_id,
                "utterance_ids": utterance_ids,
                "identifier_source": source,
            }
        )
    return resolved


def validate_official_dialogues(
    raw_data: Any,
    *,
    split: str,
    expected_dims: Sequence[int] = MODALITY_DIMS,
    require_identifiers: bool = False,
    enforce_mmgcn_session_boundary: bool = False,
) -> dict[str, Any]:
    """Validate one released-loader-compatible ``*.json.feature`` value."""

    split = str(split).lower()
    if split not in SPLIT_ORDER:
        raise ValueError(f"Unsupported split={split!r}.")
    if not isinstance(raw_data, list) or not raw_data:
        raise ValueError(f"{split} feature data must be a non-empty dialogue list.")
    if len(expected_dims) != 3:
        raise ValueError("expected_dims must contain text, audio, visual dimensions.")

    dialogue_ids: list[str] = []
    utterance_ids: list[str] = []
    speakers: set[str] = set()
    labels: set[str] = set()
    utterance_count = 0
    for dialogue_index, dialogue in enumerate(raw_data):
        if not isinstance(dialogue, list) or not dialogue:
            raise ValueError(f"{split} dialogue {dialogue_index} must be non-empty.")
        observed_dialogue_id: str | None = None
        for utterance_index, utterance in enumerate(dialogue):
            location = f"{split}[{dialogue_index}][{utterance_index}]"
            if not isinstance(utterance, Mapping):
                raise TypeError(f"{location} must be an utterance object.")
            required = {"text", "speaker", "label", "cls"}
            if require_identifiers:
                required |= {"dialogue_id", "utterance_id"}
            missing = required - set(utterance)
            if missing:
                raise ValueError(f"{location} is missing {sorted(missing)}.")
            cls = utterance["cls"]
            if not isinstance(cls, list) or len(cls) != 3:
                raise ValueError(f"{location}.cls must contain exactly three modalities.")
            for modality_index, (name, expected_dim) in enumerate(
                zip(MODALITY_ORDER, expected_dims)
            ):
                validate_vector(
                    cls[modality_index],
                    expected_dim=int(expected_dim),
                    location=f"{location}.cls[{modality_index}] {name}",
                )
            speakers.add(str(utterance["speaker"]))
            labels.add(str(utterance["label"]))
            if "dialogue_id" in utterance:
                dialogue_id = str(utterance["dialogue_id"])
                if observed_dialogue_id is None:
                    observed_dialogue_id = dialogue_id
                elif dialogue_id != observed_dialogue_id:
                    raise ValueError(f"{location} changes dialogue_id within a dialogue.")
            if "utterance_id" in utterance:
                utterance_ids.append(str(utterance["utterance_id"]))
            utterance_count += 1
        if observed_dialogue_id is not None:
            dialogue_ids.append(observed_dialogue_id)

    if len(set(dialogue_ids)) != len(dialogue_ids):
        raise ValueError(f"{split} contains duplicate dialogue identifiers.")
    if len(set(utterance_ids)) != len(utterance_ids):
        raise ValueError(f"{split} contains duplicate utterance identifiers.")
    if enforce_mmgcn_session_boundary:
        if not dialogue_ids:
            raise ValueError("Session-boundary validation requires embedded dialogue_id fields.")
        allowed = {"Ses05"} if split == "test" else {"Ses01", "Ses02", "Ses03", "Ses04"}
        invalid = [value for value in dialogue_ids if parse_session_id(value) not in allowed]
        if invalid:
            raise ValueError(
                f"{split} violates the MMGCN Ses01-Ses04/Ses05 boundary: {invalid}."
            )

    return {
        "split": split,
        "dialogue_count": len(raw_data),
        "utterance_count": utterance_count,
        "speaker_values": sorted(speakers),
        "label_values": sorted(labels),
        "dimensions": dict(zip(MODALITY_ORDER, map(int, expected_dims))),
        "modality_order": list(MODALITY_ORDER),
        "identifiers_present": len(dialogue_ids) == len(raw_data),
    }


def validate_vocab(value: Any, *, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(("stoi", "itos")) - set(value):
        raise ValueError(f"{name} must be a mapping with stoi and itos.")
    stoi = value["stoi"]
    itos = value["itos"]
    if not isinstance(stoi, Mapping) or not isinstance(itos, list):
        raise TypeError(f"{name}.stoi must be a mapping and itos must be a list.")
    expected = {str(token): index for index, token in enumerate(itos)}
    normalized = {str(token): int(index) for token, index in stoi.items()}
    if normalized != expected:
        raise ValueError(f"{name} stoi/itos are not exact inverses.")
    return {"stoi": normalized, "itos": [str(token) for token in itos]}


def load_vocab(path: Path, *, name: str) -> dict[str, Any]:
    with Path(path).open("rb") as file:
        value = pickle.load(file)
    return validate_vocab(value, name=name)

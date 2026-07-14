"""IEMOCAP feature-set registry and checksum validation helpers."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Mapping, Optional

import yaml

from utils.run_metadata import compute_file_sha256


UNPINNED_SHA256 = "TO_BE_FILLED_AFTER_GENERATION"
DEFAULT_REGISTRY_PATH = Path("configs/data/iemocap_feature_sets.yaml")
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping.")
    return value


def _resolve(path_text: str, project_root: Path) -> Path:
    path = Path(str(path_text)).expanduser()
    return path if path.is_absolute() else Path(project_root) / path


def load_iemocap_feature_registry(
    project_root: Path,
    registry_path: Path = DEFAULT_REGISTRY_PATH,
) -> Mapping[str, Any]:
    """Load the versioned feature registry without touching feature data."""

    resolved = _resolve(str(registry_path), Path(project_root))
    if not resolved.is_file():
        raise FileNotFoundError(f"IEMOCAP feature registry not found: {resolved}")
    with resolved.open("r", encoding="utf-8") as file:
        registry = yaml.safe_load(file)
    return _mapping(registry, "IEMOCAP feature registry")


def _configured_text_dim(config: Mapping[str, Any]) -> Optional[int]:
    model = _mapping(config.get("model", {}), "model")
    dataset = _mapping(config.get("dataset", {}), "dataset")
    value = model.get("text_feature_dim", model.get("text_dim"))
    if value is None:
        value = dataset.get("text_feature_dim")
    return None if value is None else int(value)


def validate_iemocap_feature_config(
    config: Mapping[str, Any],
    project_root: Path,
    *,
    require_file: bool = True,
    verify_checksum: bool = True,
    registry_path: Path = DEFAULT_REGISTRY_PATH,
) -> Optional[str]:
    """Validate an IEMOCAP feature pin and return the verified checksum.

    A placeholder digest is accepted only when the config explicitly marks
    itself as a smoke run.  Even then, an actual training invocation still
    requires the PKL to exist; only checksum pinning is relaxed.
    """

    dataset = _mapping(config.get("dataset", {}), "dataset")
    if str(dataset.get("name", "")).strip().upper() != "IEMOCAP":
        return None

    path_text = str(dataset.get("feature_pkl_path", "")).strip()
    if not path_text:
        raise ValueError("dataset.feature_pkl_path is required for IEMOCAP.")

    expected = str(dataset.get("feature_sha256", "")).strip()
    allow_unpinned = dataset.get("allow_unpinned_feature_for_smoke") is True
    if expected == UNPINNED_SHA256:
        if not allow_unpinned:
            raise ValueError(
                "Refusing unpinned IEMOCAP features: dataset.feature_sha256 is "
                f"still {UNPINNED_SHA256!r}. Only an explicit "
                "allow_unpinned_feature_for_smoke: true smoke config may proceed."
            )
    elif not _SHA256_PATTERN.fullmatch(expected.lower()):
        raise ValueError("dataset.feature_sha256 must be a lowercase 64-character SHA256.")

    feature_set_name = str(dataset.get("feature_set_name", "")).strip()
    if feature_set_name:
        registry = load_iemocap_feature_registry(project_root, registry_path)
        registry_key = feature_set_name if feature_set_name in registry else None
        if registry_key is None:
            matches = [
                str(key)
                for key, value in registry.items()
                if isinstance(value, Mapping)
                and str(value.get("feature_set_name", "")).strip() == feature_set_name
            ]
            if len(matches) == 1:
                registry_key = matches[0]
        if registry_key is None:
            raise ValueError(f"Unknown IEMOCAP feature_set_name: {feature_set_name!r}.")
        entry = _mapping(registry[registry_key], f"registry.{registry_key}")
        if Path(str(entry.get("path", ""))).as_posix() != Path(path_text).as_posix():
            raise ValueError(
                f"dataset.feature_pkl_path does not match registry.{registry_key}.path."
            )
        configured_dim = _configured_text_dim(config)
        if configured_dim is not None and int(entry.get("text_dim", -1)) != configured_dim:
            raise ValueError(
                f"Configured text dimension {configured_dim} does not match "
                f"registry.{registry_key}.text_dim={entry.get('text_dim')}."
            )
        registry_sha = str(entry.get("sha256", "")).strip()
        if expected != registry_sha:
            raise ValueError(
                f"dataset.feature_sha256 does not match registry.{registry_key}.sha256."
            )

    feature_path = _resolve(path_text, Path(project_root))
    if not require_file:
        return None
    if not feature_path.is_file():
        raise FileNotFoundError(f"IEMOCAP feature PKL not found: {feature_path}")
    if expected == UNPINNED_SHA256 or not verify_checksum:
        return None

    actual = compute_file_sha256(feature_path)
    if actual.lower() != expected.lower():
        raise ValueError(
            "IEMOCAP feature SHA256 mismatch: "
            f"configured={expected.lower()}, actual={actual.lower()}."
        )
    return actual.lower()


__all__ = [
    "DEFAULT_REGISTRY_PATH",
    "UNPINNED_SHA256",
    "load_iemocap_feature_registry",
    "validate_iemocap_feature_config",
]

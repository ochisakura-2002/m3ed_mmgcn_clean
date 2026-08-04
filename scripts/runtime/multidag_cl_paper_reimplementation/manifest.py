"""Canonical run allocation, resolved config, and immutable manifest helpers."""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional

import torch
import yaml

from models.registry.paper_reimplementation import get_model_metadata
from utils.output_paths import (
    allocate_configured_run,
    configured_output_root,
    resolve_experiment_date,
)
from utils.run_metadata import compute_file_sha256

from .adapter import FeatureRegistryMetadata


@dataclass(frozen=True)
class RunPaths:
    run_id: str
    run_dir: Path
    checkpoints: Path
    logs: Path
    reports: Path
    predictions: Path
    manifests: Path
    resolved_config: Path
    run_manifest: Path


def load_config(path: Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as file:
        value = yaml.safe_load(file)
    if not isinstance(value, dict):
        raise TypeError("runtime config must be a YAML mapping")
    return value


def _project_relative(path: Path, project_root: Path) -> str:
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(Path(project_root).resolve()).as_posix()
    except ValueError:
        return Path(path).as_posix()


def _serialize_resolved_config(config: Mapping[str, Any]) -> bytes:
    return yaml.safe_dump(
        dict(config), sort_keys=False, allow_unicode=True
    ).encode("utf-8")


def _run_paths(run_dir: Path) -> RunPaths:
    run_dir = Path(run_dir)
    return RunPaths(
        run_id=run_dir.name,
        run_dir=run_dir,
        checkpoints=run_dir / "checkpoints",
        logs=run_dir / "logs",
        reports=run_dir / "reports",
        predictions=run_dir / "predictions",
        manifests=run_dir / "manifests",
        resolved_config=run_dir / "resolved_config.yaml",
        run_manifest=run_dir / "manifests" / "run_manifest.json",
    )


def prepare_run_paths(
    config: Mapping[str, Any],
    *,
    config_path: Path,
    project_root: Path,
    output_root_override: Optional[Path],
    experiment_date: Optional[str],
    experiment_group: Optional[str],
    resume_run_dir: Optional[Path] = None,
) -> tuple[dict[str, Any], RunPaths]:
    resolved = deepcopy(dict(config))
    if resume_run_dir is not None:
        run_dir = Path(resume_run_dir)
        if not run_dir.is_dir():
            raise FileNotFoundError(f"Resume run directory not found: {run_dir}")
        paths = _run_paths(run_dir)
        if not paths.resolved_config.is_file():
            raise FileNotFoundError(
                f"Resume resolved config not found: {paths.resolved_config}"
            )
        original_bytes = paths.resolved_config.read_bytes()
        original = yaml.safe_load(original_bytes.decode("utf-8"))
        if not isinstance(original, Mapping):
            raise TypeError("resume resolved config must be a YAML mapping")
        original_output = original.get("output")
        if not isinstance(original_output, Mapping):
            raise ValueError("resume resolved config has no output mapping")
        candidate_output = resolved.setdefault("output", {})
        if not isinstance(candidate_output, dict):
            raise TypeError("resume config output must be a mapping")
        original_date = str(original_output.get("experiment_date"))
        original_group = str(original_output.get("experiment_group"))
        configured_date = candidate_output.get("experiment_date")
        configured_group = candidate_output.get("experiment_group")
        if configured_date is not None and str(configured_date) != original_date:
            raise ValueError("resume config experiment_date mismatch")
        if experiment_date is not None and str(experiment_date) != original_date:
            raise ValueError("resume CLI experiment_date mismatch")
        if configured_group is not None and str(configured_group) != original_group:
            raise ValueError("resume config experiment_group mismatch")
        if experiment_group is not None and str(experiment_group) != original_group:
            raise ValueError("resume CLI experiment_group mismatch")
        for name in (
            "output_base",
            "root",
            "experiment_date",
            "experiment_group",
            "run_id",
        ):
            if name not in original_output:
                raise ValueError(f"resume resolved config output.{name} is missing")
            candidate_output[name] = deepcopy(original_output[name])
        if _serialize_resolved_config(resolved) != original_bytes:
            raise ValueError(
                "resume config mismatch: immutable resolved_config.yaml differs"
            )
        return resolved, paths

    frozen_date = resolve_experiment_date(cli_date=experiment_date, config=resolved)
    output_base = configured_output_root(resolved, override=output_root_override)
    if not output_base.is_absolute():
        output_base = Path(project_root) / output_base
    layout = allocate_configured_run(
        config=resolved,
        config_path=config_path,
        experiment_name=str(resolved.get("run_name", "multidag_cl_paper_reimplementation")),
        experiment_date=frozen_date,
        output_base=output_base,
        experiment_group=experiment_group,
        resume_run_dir=resume_run_dir,
    )
    if layout.run_root is None:
        raise RuntimeError("run allocation did not return a run root")
    run_dir = layout.run_root
    paths = _run_paths(run_dir)
    for directory in (
        paths.checkpoints,
        paths.logs,
        paths.reports,
        paths.predictions,
        paths.manifests,
    ):
        directory.mkdir(parents=True, exist_ok=True)
    output = resolved.setdefault("output", {})
    output["output_base"] = _project_relative(output_base, project_root)
    output["root"] = _project_relative(run_dir, project_root)
    output["experiment_date"] = layout.experiment_date
    output["experiment_group"] = layout.experiment_group
    output["run_id"] = paths.run_id
    with paths.resolved_config.open("xb") as file:
        file.write(_serialize_resolved_config(resolved))
    return resolved, paths


def _git_value(project_root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def build_run_manifest(
    *,
    config: Mapping[str, Any],
    paths: RunPaths,
    project_root: Path,
    feature: FeatureRegistryMetadata,
    bucket_membership_sha256: str,
    configured_bucket_count: int,
    actual_bucket_count: int,
    optimizer_audit: Mapping[str, Any],
    command_entrypoint: str,
) -> dict[str, Any]:
    metadata = get_model_metadata(config.get("registry_key"))
    dataset = config["dataset"]
    runtime = config["runtime"]
    checkpoint = config["checkpoint"]
    labels = list(dataset["label_names"])
    manifest = {
        "project_git_head": _git_value(project_root, "rev-parse", "HEAD"),
        "project_git_dirty": bool(
            _git_value(project_root, "status", "--porcelain", "--untracked-files=no")
        ),
        "command_entrypoint": command_entrypoint,
        "official_repo_url": metadata["official_repo_url"],
        "official_commit": metadata["official_commit"],
        "paper_sha256": metadata["paper_sha256"],
        "paper_title": metadata["paper_title"],
        "paper_venue": metadata["paper_venue"],
        "paper_year": metadata["paper_year"],
        "canonical_name": metadata["canonical_name"],
        "canonical_model_name": metadata["canonical_name"],
        "implementation_identity": metadata["implementation_identity"],
        "conformance_profile": metadata["conformance_profile"],
        "data_track": metadata["data_track"],
        "registry_key": config["registry_key"],
        "feature_registry": feature.registry_key,
        "feature_path": feature.feature_path,
        "feature_sha256": feature.feature_sha256,
        "feature_dimensions": feature.feature_dimensions,
        "split_protocol": dataset["split_protocol"],
        "split_membership_sha256": dataset.get("split_membership_sha256", "runtime_dataset_contract"),
        "validation_session": dataset.get("validation_session"),
        "test_session": dataset.get("test_session"),
        "seed": runtime["seed"],
        "label_mapping": {str(index): name for index, name in enumerate(labels)},
        "speaker_mapping": dataset["speaker_mapping"],
        "resolved_config_path": _project_relative(paths.resolved_config, project_root),
        "resolved_config_sha256": compute_file_sha256(paths.resolved_config),
        "optimizer": dict(runtime["optimizer"]),
        "optimizer_parameter_inventory": dict(optimizer_audit),
        "scheduler": runtime["scheduler"],
        "gradient_clipping": runtime["gradient_clipping"],
        "class_weight": runtime["class_weight"],
        "label_smoothing": runtime["label_smoothing"],
        "curriculum_profile": config["model_core"]["curriculum"],
        "bucket_count": configured_bucket_count,
        "configured_bucket_count": configured_bucket_count,
        "actual_bucket_count": actual_bucket_count,
        "bucket_membership_sha256": bucket_membership_sha256,
        "resumed_global_epoch": 1,
        "resume_history": [],
        "checkpoint_metric": checkpoint["primary_metric"],
        "checkpoint_tiebreaks": [
            checkpoint["secondary_tiebreak"],
            checkpoint["tertiary_tiebreak"],
        ],
        "test_split_used_for_selection": False,
        "test_evaluation_count": checkpoint["test_evaluation_count"],
        "dag_topology_causal": metadata["dag_topology_causal"],
        "end_to_end_causal": metadata["end_to_end_causal"],
        "context_label": metadata["context_label"],
        "model_math_deviation_list": list(config["provenance"]["model_math_deviation_list"]),
        "protocol_repair_list": list(config["provenance"]["protocol_repair_list"]),
        "engineering_adapter_list": list(config["provenance"]["engineering_adapter_list"]),
        "protocol_deviation_list": list(config["provenance"]["protocol_deviation_list"]),
        "paper_number_reproduced": False,
        "protocol_comparability": dataset["protocol_comparability"],
        "formal_experiment": runtime["formal_experiment"],
        "smoke_only": runtime["smoke_only"],
        "experiment_date": config["output"]["experiment_date"],
        "experiment_group": config["output"]["experiment_group"],
        "run_id": paths.run_id,
        "manifest_directory": _project_relative(paths.manifests, project_root),
        "deterministic_settings": {
            "seed": runtime["seed"],
            "sampler_seed_rule": "base_seed_plus_global_epoch",
        },
        "environment": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "platform": platform.platform(),
            "device": runtime["device"],
        },
    }
    validate_manifest(manifest)
    return manifest


REQUIRED_MANIFEST_FIELDS = {
    "project_git_head",
    "official_repo_url",
    "official_commit",
    "paper_sha256",
    "canonical_name",
    "implementation_identity",
    "conformance_profile",
    "data_track",
    "registry_key",
    "feature_registry",
    "feature_path",
    "feature_sha256",
    "feature_dimensions",
    "split_protocol",
    "validation_session",
    "test_session",
    "seed",
    "label_mapping",
    "speaker_mapping",
    "resolved_config_path",
    "resolved_config_sha256",
    "optimizer",
    "curriculum_profile",
    "bucket_count",
    "bucket_membership_sha256",
    "checkpoint_metric",
    "test_split_used_for_selection",
    "dag_topology_causal",
    "end_to_end_causal",
    "model_math_deviation_list",
    "protocol_repair_list",
    "engineering_adapter_list",
    "protocol_deviation_list",
    "formal_experiment",
    "smoke_only",
}


def validate_manifest(manifest: Mapping[str, Any]) -> None:
    missing = sorted(REQUIRED_MANIFEST_FIELDS - set(manifest))
    if missing:
        raise ValueError(f"run manifest missing required fields: {missing}")
    for name in (
        "paper_sha256",
        "feature_sha256",
        "resolved_config_sha256",
        "bucket_membership_sha256",
        "split_membership_sha256",
    ):
        value = str(manifest[name]).lower()
        if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
            raise ValueError(f"manifest {name} must be a SHA256")
    if manifest["test_split_used_for_selection"] is not False:
        raise ValueError("manifest must be validation-clean")
    if not isinstance(manifest.get("resume_history"), list):
        raise ValueError("manifest resume_history must be a list")


def _write_json_atomic_replace(path: Path, value: Mapping[str, Any]) -> None:
    path = Path(path)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as file:
            json.dump(dict(value), file, ensure_ascii=False, indent=2, sort_keys=True)
            file.write("\n")
            file.flush()
            os.fsync(file.fileno())
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_manifest(path: Path, manifest: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"run manifest already exists: {path}")
    with path.open("x", encoding="utf-8", newline="\n") as file:
        json.dump(dict(manifest), file, ensure_ascii=False, indent=2, sort_keys=True)
        file.write("\n")


def append_final_evaluation(path: Path, final_evaluation: Mapping[str, Any]) -> None:
    path = Path(path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if "final_evaluation" in manifest:
        raise RuntimeError("final evaluation is append-only and already present")
    manifest["final_evaluation"] = dict(final_evaluation)
    _write_json_atomic_replace(path, manifest)


RESUME_IDENTITY_FIELDS = (
    "registry_key",
    "implementation_identity",
    "conformance_profile",
    "data_track",
    "feature_sha256",
    "split_protocol",
    "split_membership_sha256",
    "bucket_membership_sha256",
    "resolved_config_sha256",
)


def _portable_checkpoint_path(
    checkpoint_path: Path, *, run_dir: Path, project_root: Path
) -> str:
    resolved = Path(checkpoint_path).resolve()
    for root in (Path(run_dir).resolve(), Path(project_root).resolve()):
        try:
            return resolved.relative_to(root).as_posix()
        except ValueError:
            continue
    raise ValueError("resume checkpoint path must be run-relative or project-relative")


def append_resume_history(
    path: Path,
    *,
    checkpoint_path: Path,
    checkpoint: Mapping[str, Any],
    run_dir: Path,
    project_root: Path,
    expected_identity: Mapping[str, Any],
    resume_timestamp: Optional[str] = None,
) -> dict[str, Any]:
    """Validate resume provenance and atomically append exactly one history row."""

    path = Path(path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    validate_manifest(manifest)
    missing_expected = sorted(set(RESUME_IDENTITY_FIELDS) - set(expected_identity))
    if missing_expected:
        raise ValueError(f"resume expected identity missing fields: {missing_expected}")
    initial_identity = {name: deepcopy(manifest.get(name)) for name in RESUME_IDENTITY_FIELDS}
    for name in RESUME_IDENTITY_FIELDS:
        if manifest.get(name) != expected_identity[name]:
            raise ValueError(f"resume manifest {name} mismatch")

    checkpoint_field_map = {
        "registry_key": "registry_key",
        "implementation_identity": "implementation_identity",
        "conformance_profile": "conformance_profile",
        "data_track": "data_track",
        "feature_sha256": "feature_sha256",
        "split_protocol": "split_protocol",
        "split_membership_sha256": "split_membership_sha256",
        "bucket_membership_sha256": "curriculum_membership_sha256",
        "resolved_config_sha256": "resolved_config_sha256",
    }
    for identity_name, checkpoint_name in checkpoint_field_map.items():
        if checkpoint.get(checkpoint_name) != expected_identity[identity_name]:
            raise ValueError(f"resume checkpoint {checkpoint_name} mismatch")
    if checkpoint.get("checkpoint_locked") is True:
        raise RuntimeError("a locked best checkpoint cannot resume training")

    completed_epoch = int(checkpoint["epoch"])
    resumed_global_epoch = completed_epoch + 1
    timestamp = resume_timestamp or datetime.now(timezone.utc).isoformat().replace(
        "+00:00", "Z"
    )
    entry = {
        "checkpoint_path": _portable_checkpoint_path(
            checkpoint_path, run_dir=run_dir, project_root=project_root
        ),
        "checkpoint_sha256": compute_file_sha256(Path(checkpoint_path)),
        "completed_epoch": completed_epoch,
        "resumed_global_epoch": resumed_global_epoch,
        "checkpoint_global_step": int(checkpoint["global_step"]),
        "resume_timestamp": timestamp,
        "curriculum_membership_sha256": checkpoint[
            "curriculum_membership_sha256"
        ],
    }
    history = manifest["resume_history"]
    history.append(entry)
    manifest["resumed_global_epoch"] = resumed_global_epoch
    if {name: manifest.get(name) for name in RESUME_IDENTITY_FIELDS} != initial_identity:
        raise RuntimeError("resume history attempted to mutate initial identity")
    validate_manifest(manifest)
    _write_json_atomic_replace(path, manifest)
    return entry


__all__ = [
    "REQUIRED_MANIFEST_FIELDS",
    "RunPaths",
    "append_final_evaluation",
    "append_resume_history",
    "build_run_manifest",
    "load_config",
    "prepare_run_paths",
    "validate_manifest",
    "write_manifest",
]

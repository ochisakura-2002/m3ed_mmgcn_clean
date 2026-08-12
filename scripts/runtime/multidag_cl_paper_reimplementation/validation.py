"""Fail-closed Stage-B3 runtime, split, feature, and output validation."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Mapping, Optional

from models.registry.paper_reimplementation import (
    REGISTRY_KEY,
    canonical_model_key,
    get_model_metadata,
)
from utils.iemocap_features import DEFAULT_REGISTRY_PATH, load_iemocap_feature_registry
from utils.run_metadata import compute_file_sha256

from models.multidag_cl.paper_reimplementation.config import (
    ConformanceProfile,
    DataTrack,
    MultiDAGCLConfig,
)
from .adapter import FeatureRegistryMetadata
from .optimizer import validate_optimizer_config


ALLOWED_MODES = {"check", "synthetic-smoke", "real-batch-smoke", "train", "evaluate"}
SMOKE_OUTPUT_ROOT = Path(
    "tmp/assistant_work/paper_reproduction_stage_b3_runtime_correction/smoke_outputs"
)
SYNTHETIC_FEATURE_SHA256 = (
    "621271096b4ca47b3e61efd51ad28bc728fbcdc3264badc367cec90546709406"
)
OFFICIAL_FEATURE_REGISTRY_KEY = "multidag_cl_official_2948_v1"
OFFICIAL_FEATURE_SHA256_SENTINEL = "FROM_OFFICIAL_ASSET_MANIFEST"
OFFICIAL_SPLIT_PROTOCOL = "multidag_cl_official_exact_train_dev_test"
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class RuntimeValidationError(ValueError):
    """One or more protocol invariants are invalid."""


class LocalAssetUnavailable(FileNotFoundError):
    """The real project-fair smoke is valid but its local feature asset is absent."""


class OfficialAssetsUnavailable(LocalAssetUnavailable):
    """The Stage-C2 paper-data config is valid but official assets are absent."""


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RuntimeValidationError(f"{name} must be a mapping")
    return value


def _resolve(path_text: str, project_root: Path) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else Path(project_root) / path


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _runtime_section(config: Mapping[str, Any]) -> Mapping[str, Any]:
    return _mapping(config.get("runtime"), "runtime")


def _configured_num_classes(config: Mapping[str, Any]) -> int:
    model_core = _mapping(config.get("model_core"), "model_core")
    data = _mapping(model_core.get("data"), "model_core.data")
    return int(data.get("num_classes", -1))


def _validate_identity(config: Mapping[str, Any], core: MultiDAGCLConfig) -> None:
    key = canonical_model_key(config.get("registry_key"))
    if key != REGISTRY_KEY:
        raise RuntimeValidationError("unknown registry key")
    metadata = get_model_metadata(key)
    configured = {
        "canonical_name": core.canonical_name,
        "implementation_identity": core.implementation_identity,
        "conformance_profile": core.conformance_profile.value,
    }
    expected = {name: metadata[name] for name in configured}
    if configured != expected:
        raise RuntimeValidationError(
            f"registry identity mismatch: configured={configured}, expected={expected}"
        )
    if core.data_track.value not in metadata.get("supported_data_tracks", []):
        raise RuntimeValidationError(
            f"unsupported data track for paper reimplementation: {core.data_track.value!r}"
        )
    if "author_official" in str(config).lower():
        raise RuntimeValidationError("author_official identity claims are forbidden")


def _validate_split(
    config: Mapping[str, Any], formal: bool, core: MultiDAGCLConfig
) -> None:
    dataset = _mapping(config.get("dataset"), "dataset")
    if str(dataset.get("name", "")).upper() == "SYNTHETIC":
        return
    if str(dataset.get("name", "")).upper() != "IEMOCAP":
        raise RuntimeValidationError("dataset.name must be IEMOCAP or SYNTHETIC")
    if core.data_track is DataTrack.PAPER_DATA:
        if dataset.get("val_split_strategy") != "official_split_manifest":
            raise RuntimeValidationError(
                "paper_data must use exact official_split_manifest membership"
            )
        if dataset.get("split_protocol") != OFFICIAL_SPLIT_PROTOCOL:
            raise RuntimeValidationError("unexpected paper-data split protocol")
        split_manifest = str(dataset.get("official_split_manifest_path", "")).strip()
        if not split_manifest:
            raise RuntimeValidationError(
                "paper_data requires dataset.official_split_manifest_path"
            )
        if dataset.get("validation_session") is not None:
            raise RuntimeValidationError(
                "paper_data must not select validation by Session"
            )
        return
    if dataset.get("val_split_strategy") != "session_holdout":
        raise RuntimeValidationError("project-fair uses the existing session_holdout split")
    validation_session = dataset.get("validation_session")
    test_session = dataset.get("test_session")
    if validation_session not in {"Ses01", "Ses02", "Ses03", "Ses04"}:
        raise RuntimeValidationError("validation_session must be one of Ses01--Ses04")
    if validation_session == test_session:
        raise RuntimeValidationError("Validation and Test sessions must differ")
    if formal and test_session != "Ses05":
        raise RuntimeValidationError("formal project-fair Test must be Ses05")
    if dataset.get("split_protocol") != "clean_roberta_session_holdout_fair_comparison":
        raise RuntimeValidationError("unexpected project-fair split protocol")


def _validate_checkpoint(config: Mapping[str, Any], formal: bool) -> None:
    checkpoint = _mapping(config.get("checkpoint"), "checkpoint")
    primary = str(checkpoint.get("primary_metric", ""))
    secondary = str(checkpoint.get("secondary_tiebreak", ""))
    tertiary = str(checkpoint.get("tertiary_tiebreak", ""))
    combined = " ".join((primary, secondary, tertiary)).lower()
    if "test" in combined:
        raise RuntimeValidationError("Test may not enter checkpoint selection or tie-breaks")
    if primary != "val_weighted_f1":
        raise RuntimeValidationError("checkpoint primary metric must be val_weighted_f1")
    if secondary != "val_loss_lower" or tertiary != "earlier_epoch":
        raise RuntimeValidationError("checkpoint tie-breaks must be val loss then earlier epoch")
    if checkpoint.get("test_split_used_for_selection") is not False:
        raise RuntimeValidationError("test_split_used_for_selection must be false")
    count = checkpoint.get("test_evaluation_count")
    if formal and count != 1:
        raise RuntimeValidationError("formal runtime must plan exactly one Test evaluation")
    if count not in (0, 1):
        raise RuntimeValidationError("test_evaluation_count must be 0 or 1")


def _validate_runtime_controls(
    config: Mapping[str, Any],
    core: MultiDAGCLConfig,
    *,
    effective_mode: str,
    project_root: Path,
) -> None:
    runtime = _runtime_section(config)
    formal = runtime.get("formal_experiment") is True
    smoke_only = runtime.get("smoke_only") is True
    if runtime.get("formal_experiment") not in (True, False):
        raise RuntimeValidationError("runtime.formal_experiment must be bool")
    if runtime.get("smoke_only") not in (True, False):
        raise RuntimeValidationError("runtime.smoke_only must be bool")
    if formal == smoke_only:
        raise RuntimeValidationError("formal_experiment and smoke_only must be opposites")
    if runtime.get("scheduler") != "none":
        raise RuntimeValidationError("scheduler must be none")
    gradient_clipping = _mapping(
        runtime.get("gradient_clipping"), "runtime.gradient_clipping"
    )
    expected_gradient_clipping = {
        "mode": "global_norm",
        "max_norm": 5.0,
        "norm_type": 2.0,
        "error_if_nonfinite": True,
    }
    configured_gradient_clipping = {
        "mode": gradient_clipping.get("mode"),
        "max_norm": float(gradient_clipping.get("max_norm", -1.0)),
        "norm_type": float(gradient_clipping.get("norm_type", -1.0)),
        "error_if_nonfinite": gradient_clipping.get("error_if_nonfinite"),
    }
    if configured_gradient_clipping != expected_gradient_clipping:
        raise RuntimeValidationError(
            "runtime.gradient_clipping must explicitly configure global norm 5.0, "
            "norm type 2.0, and non-finite failure"
        )
    if float(core.gradient_clip_norm) != configured_gradient_clipping["max_norm"]:
        raise RuntimeValidationError(
            "runtime gradient clipping must match the Stage-B1 frozen core value"
        )
    if runtime.get("class_weight") is not None:
        raise RuntimeValidationError("class_weight must be null")
    if float(runtime.get("label_smoothing", -1)) != 0.0:
        raise RuntimeValidationError("label_smoothing must be zero")
    if runtime.get("early_stopping") is not False:
        raise RuntimeValidationError("early stopping must be disabled")
    validate_optimizer_config(_mapping(runtime.get("optimizer"), "runtime.optimizer"))

    if formal:
        expected = {
            "epochs": 30,
            "batch_size": 16,
            "seed": 100,
        }
        for name, value in expected.items():
            if runtime.get(name) != value:
                raise RuntimeValidationError(
                    f"formal runtime.{name} must be {value!r}"
                )
        if core.graph_layers != 4 or core.bucket_count != 5:
            raise RuntimeValidationError("formal primary profile requires 4 DAG layers and 5 buckets")
        if core.classifier_dropout != 0.4 or core.learning_rate != 5.0e-4:
            raise RuntimeValidationError("formal dropout/LR do not match the paper protocol")
        if core.conformance_profile is not ConformanceProfile.PAPER_FORMULA_BEHAVIOR:
            raise RuntimeValidationError("source profile cannot be the default formal profile")
        if core.data_track is DataTrack.PAPER_DATA:
            assets = config.get("official_assets", {})
            if (
                not isinstance(assets, Mapping)
                or assets.get("verification") != "official_asset_manifest"
            ):
                raise RuntimeValidationError(
                    "paper_data requires runtime verification from official_asset_manifest"
                )
        experiment = _mapping(config.get("experiment"), "experiment")
        if experiment.get("context_label") != "full_context":
            raise RuntimeValidationError("primary formal profile must use full_context")
        output = _mapping(config.get("output"), "output")
        if Path(str(output.get("root"))).as_posix() != "outputs":
            raise RuntimeValidationError("formal output root must use the canonical outputs base")
    else:
        output = _mapping(config.get("output"), "output")
        if Path(str(output.get("root", ""))).as_posix() != "outputs":
            raise RuntimeValidationError(
                "smoke YAML keeps canonical output.root=outputs and uses "
                "runtime.smoke_output_root for the mandatory tmp override"
            )
        output_root = _resolve(str(runtime.get("smoke_output_root", "")), project_root)
        allowed_root = _resolve(str(SMOKE_OUTPUT_ROOT), project_root)
        if not _is_within(output_root, allowed_root):
            raise RuntimeValidationError(
                "runtime.smoke_output_root must stay under the Stage-B3 tmp root"
            )
        if _is_within(output_root, _resolve("outputs", project_root)):
            raise RuntimeValidationError("smoke output may not write to outputs")

    limits = _mapping(runtime.get("limits"), "runtime.limits")
    if effective_mode == "synthetic-smoke":
        expected_limits = {
            "epochs": 1,
            "optimizer_steps": 1,
            "train_batches": 1,
        }
        for name, value in expected_limits.items():
            if int(limits.get(name, -1)) != value:
                raise RuntimeValidationError(f"synthetic smoke {name} must equal {value}")
        if int(limits.get("validation_batches", -1)) > 2:
            raise RuntimeValidationError("synthetic smoke validation batches must be <=2")
        if int(limits.get("test_batches", -1)) > 1:
            raise RuntimeValidationError("synthetic smoke fake test batches must be <=1")
    if effective_mode == "real-batch-smoke":
        if int(limits.get("epochs", -1)) != 0:
            raise RuntimeValidationError("real-batch smoke epochs must be 0")
        if int(limits.get("optimizer_steps", -1)) != 0:
            raise RuntimeValidationError("real-batch smoke optimizer steps must be 0")
        if int(limits.get("train_batches", -1)) > 1:
            raise RuntimeValidationError("real-batch smoke train batches must be <=1")
        if int(limits.get("validation_batches", -1)) > 1:
            raise RuntimeValidationError("real-batch smoke validation batches must be <=1")
        if int(limits.get("real_test_batches", -1)) != 0:
            raise RuntimeValidationError("real-batch smoke may not access real Test")


def resolve_feature_metadata(
    config: Mapping[str, Any],
    *,
    project_root: Path,
    require_file: bool,
    verify_checksum: bool,
) -> FeatureRegistryMetadata:
    dataset = _mapping(config.get("dataset"), "dataset")
    registry_key = str(dataset.get("feature_registry", "")).strip()
    feature_path = str(dataset.get("feature_path", "")).strip()
    configured_sha = str(dataset.get("feature_sha256", "")).strip()
    dimensions = _mapping(dataset.get("feature_dimensions"), "dataset.feature_dimensions")
    configured_dims = {
        "text": int(dimensions.get("text", -1)),
        "audio": int(dimensions.get("audio", -1)),
        "visual": int(dimensions.get("visual", -1)),
    }

    if registry_key == OFFICIAL_FEATURE_REGISTRY_KEY:
        registry = load_iemocap_feature_registry(project_root, DEFAULT_REGISTRY_PATH)
        if registry_key not in registry:
            raise RuntimeValidationError(
                f"unknown feature registry key: {registry_key!r}"
            )
        entry = _mapping(registry[registry_key], f"feature_registry.{registry_key}")
        expected_dims = {
            "text": int(entry.get("text_dim", -1)),
            "audio": int(entry.get("audio_dim", -1)),
            "visual": int(entry.get("visual_dim", -1)),
        }
        if Path(feature_path).as_posix() != Path(str(entry.get("path", ""))).as_posix():
            raise RuntimeValidationError("official feature path does not match registry")
        if configured_dims != expected_dims or configured_dims != {
            "text": 1024,
            "audio": 1582,
            "visual": 342,
        }:
            raise RuntimeValidationError("official feature dimensions must be 1024/1582/342")
        if configured_sha != OFFICIAL_FEATURE_SHA256_SENTINEL:
            raise RuntimeValidationError(
                "official feature SHA must be resolved from official_asset_manifest"
            )

        asset_manifest_text = str(
            dataset.get("official_asset_manifest_path", "")
        ).strip()
        split_manifest_text = str(
            dataset.get("official_split_manifest_path", "")
        ).strip()
        registry_manifest = Path(str(entry.get("sha256_source", ""))).as_posix()
        if Path(asset_manifest_text).as_posix() != registry_manifest:
            raise RuntimeValidationError(
                "official asset manifest path does not match feature registry"
            )
        feature_file = _resolve(feature_path, project_root)
        asset_manifest_file = _resolve(asset_manifest_text, project_root)
        split_manifest_file = _resolve(split_manifest_text, project_root)
        missing = [
            path
            for path in (feature_file, asset_manifest_file, split_manifest_file)
            if not path.is_file()
        ]
        if missing:
            raise OfficialAssetsUnavailable(
                "Official MultiDAG-CL paper-data artifacts are unavailable: "
                + ", ".join(str(path) for path in missing)
            )
        with asset_manifest_file.open("r", encoding="utf-8") as file:
            asset_manifest = json.load(file)
        with split_manifest_file.open("r", encoding="utf-8") as file:
            split_manifest = json.load(file)
        if not isinstance(asset_manifest, Mapping) or asset_manifest.get("status") != "PASS":
            raise RuntimeValidationError("official asset manifest is not a PASS manifest")
        if asset_manifest.get("dimensions") != configured_dims:
            raise RuntimeValidationError("official asset manifest dimensions mismatch")
        if asset_manifest.get("all_vectors_finite") is not True:
            raise RuntimeValidationError("official asset manifest does not confirm finite vectors")
        layer2 = _mapping(asset_manifest.get("layer2"), "official_asset_manifest.layer2")
        expected_sha = str(layer2.get("project_pkl_sha256", "")).strip().lower()
        if not SHA256_PATTERN.fullmatch(expected_sha):
            raise RuntimeValidationError("official asset manifest PKL SHA256 is invalid")
        if str(split_manifest.get("project_pkl_sha256", "")).lower() != expected_sha:
            raise RuntimeValidationError("official split manifest PKL SHA256 mismatch")
        expected_split_sha = str(layer2.get("split_manifest_sha256", "")).lower()
        if not SHA256_PATTERN.fullmatch(expected_split_sha):
            raise RuntimeValidationError("official split manifest SHA256 is invalid")
        if compute_file_sha256(split_manifest_file).lower() != expected_split_sha:
            raise RuntimeValidationError("official split manifest checksum mismatch")
        if verify_checksum and compute_file_sha256(feature_file).lower() != expected_sha:
            raise RuntimeValidationError("official project PKL checksum mismatch")
        label_vocab = _mapping(asset_manifest.get("label_vocab"), "label_vocab")
        label_names = label_vocab.get("itos")
        if not isinstance(label_names, list) or len(label_names) != _configured_num_classes(config):
            raise RuntimeValidationError("official label vocab size does not match num_classes")
        if isinstance(dataset, dict):
            configured_label_names = dataset.get("label_names")
            resolved_label_names = [str(value) for value in label_names]
            if configured_label_names == "FROM_OFFICIAL_ASSET_MANIFEST":
                dataset["label_names"] = resolved_label_names
            elif configured_label_names != resolved_label_names:
                raise RuntimeValidationError(
                    "paper_data label names must be resolved from official_asset_manifest"
                )
        return FeatureRegistryMetadata(
            registry_key=registry_key,
            feature_path=feature_path,
            feature_sha256=expected_sha,
            text_dim=configured_dims["text"],
            audio_dim=configured_dims["audio"],
            visual_dim=configured_dims["visual"],
        )

    expected_sha = configured_sha.lower()
    if not SHA256_PATTERN.fullmatch(expected_sha):
        raise RuntimeValidationError("feature SHA256 is missing or invalid")

    if registry_key == "synthetic_v1":
        expected_dims = {"text": 8, "audio": 6, "visual": 5}
        if feature_path != "synthetic://multidag-cl-stage-b3-v1":
            raise RuntimeValidationError("unexpected synthetic feature identity")
        if expected_sha != SYNTHETIC_FEATURE_SHA256 or configured_dims != expected_dims:
            raise RuntimeValidationError("synthetic feature registry metadata mismatch")
        return FeatureRegistryMetadata(
            registry_key=registry_key,
            feature_path=feature_path,
            feature_sha256=expected_sha,
            text_dim=8,
            audio_dim=6,
            visual_dim=5,
        )

    registry = load_iemocap_feature_registry(project_root, DEFAULT_REGISTRY_PATH)
    if registry_key not in registry:
        raise RuntimeValidationError(f"unknown feature registry key: {registry_key!r}")
    entry = _mapping(registry[registry_key], f"feature_registry.{registry_key}")
    expected = {
        "path": Path(str(entry.get("path", ""))).as_posix(),
        "sha256": str(entry.get("sha256", "")).lower(),
        "dimensions": {
            "text": int(entry.get("text_dim", -1)),
            "audio": int(entry.get("audio_dim", -1)),
            "visual": int(entry.get("visual_dim", -1)),
        },
    }
    configured = {
        "path": Path(feature_path).as_posix(),
        "sha256": expected_sha,
        "dimensions": configured_dims,
    }
    if configured != expected:
        raise RuntimeValidationError(
            f"feature config does not match registry: configured={configured}, expected={expected}"
        )
    feature_file = _resolve(feature_path, project_root)
    if require_file and not feature_file.is_file():
        raise LocalAssetUnavailable(f"IEMOCAP feature asset is unavailable: {feature_path}")
    if require_file and verify_checksum:
        actual_sha = compute_file_sha256(feature_file).lower()
        if actual_sha != expected_sha:
            raise RuntimeValidationError(
                f"feature SHA256 mismatch: configured={expected_sha}, actual={actual_sha}"
            )
    return FeatureRegistryMetadata(
        registry_key=registry_key,
        feature_path=feature_path,
        feature_sha256=expected_sha,
        text_dim=configured_dims["text"],
        audio_dim=configured_dims["audio"],
        visual_dim=configured_dims["visual"],
    )


def validate_runtime_config(
    config: Mapping[str, Any],
    *,
    mode: Optional[str],
    project_root: Path,
    verify_checksum: bool = True,
) -> tuple[MultiDAGCLConfig, FeatureRegistryMetadata]:
    if not isinstance(config, Mapping):
        raise RuntimeValidationError("config must be a mapping")
    effective_mode = str(mode or _runtime_section(config).get("mode", "")).strip()
    if effective_mode not in ALLOWED_MODES:
        raise RuntimeValidationError(f"unknown runtime mode: {effective_mode!r}")
    core_mapping = _mapping(config.get("model_core"), "model_core")
    core = MultiDAGCLConfig.from_mapping(core_mapping)
    _validate_identity(config, core)
    formal = _runtime_section(config).get("formal_experiment") is True
    _validate_split(config, formal, core)
    _validate_checkpoint(config, formal)
    _validate_runtime_controls(
        config,
        core,
        effective_mode=effective_mode,
        project_root=project_root,
    )
    require_file = effective_mode in {"real-batch-smoke", "train"}
    feature = resolve_feature_metadata(
        config,
        project_root=project_root,
        require_file=require_file,
        verify_checksum=verify_checksum,
    )
    configured_dims = (
        core.text_feature_dim,
        core.audio_feature_dim,
        core.visual_feature_dim,
    )
    registered_dims = (feature.text_dim, feature.audio_dim, feature.visual_dim)
    if configured_dims != registered_dims:
        raise RuntimeValidationError(
            f"feature dimension mismatch: model={configured_dims}, registry={registered_dims}"
        )
    return core, feature


__all__ = [
    "ALLOWED_MODES",
    "LocalAssetUnavailable",
    "OfficialAssetsUnavailable",
    "OFFICIAL_FEATURE_REGISTRY_KEY",
    "OFFICIAL_FEATURE_SHA256_SENTINEL",
    "OFFICIAL_SPLIT_PROTOCOL",
    "RuntimeValidationError",
    "SMOKE_OUTPUT_ROOT",
    "SYNTHETIC_FEATURE_SHA256",
    "resolve_feature_metadata",
    "validate_runtime_config",
]

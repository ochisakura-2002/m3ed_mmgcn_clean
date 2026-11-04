"""Small S2D-V0 config reader; config checking never opens data or weights."""

from __future__ import annotations

import copy
import os
from pathlib import Path

import yaml

from datasets.iemocap.label_vocab import load_label_vocab
from datasets.iemocap.s2d_faces import validate_face_config


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def resolve_path(value: str | None, base: Path = PROJECT_ROOT) -> Path | None:
    if value is None or not str(value).strip():
        return None
    path = Path(os.path.expandvars(str(value))).expanduser()
    return (path if path.is_absolute() else base / path).resolve()


def load_config(path: str | Path) -> tuple[dict, object]:
    with resolve_path(str(path)).open(encoding="utf-8-sig") as stream:
        config = copy.deepcopy(yaml.safe_load(stream))
    if not isinstance(config, dict):
        raise ValueError("Expected a YAML mapping")
    validate_face_config(config["face"])
    model, train = config["model"], config["train"]
    for key in ("pretrained_checkpoint_sha256", "landmark_checkpoint_sha256"):
        digest = model.get(key)
        if digest is not None and (not isinstance(digest, str) or len(digest) != 64
                                   or any(c not in "0123456789abcdef" for c in digest)):
            raise ValueError(f"model.{key} must be a 64-character lowercase SHA256")
    trusted_legacy = model.get("landmark_trusted_legacy_pickle", False)
    if not isinstance(trusted_legacy, bool):
        raise ValueError("model.landmark_trusted_legacy_pickle must be a boolean")
    if trusted_legacy and not model.get("landmark_checkpoint_sha256"):
        raise ValueError("model.landmark_trusted_legacy_pickle=True requires landmark_checkpoint_sha256")
    if model.get("landmark_strict", True) is not True:
        raise ValueError("S2D-V0 requires landmark_strict=True")
    if (model["num_frames"], model["input_size"]) != (16, 224):
        raise ValueError("S2D-V0 requires num_frames=16 and input_size=224")
    if model["sdl"] is not False or train["loss"] != "CrossEntropy" or train["seed"] != 100:
        raise ValueError("S2D-V0 freezes SDL=OFF, CrossEntropy, seed=100")
    if config["logging"]["monitor_metric"] != "val_weighted_f1":
        raise ValueError("Select checkpoints using Validation Weighted-F1 only")
    if config["test"]["final_evaluation_only"] is not True:
        raise ValueError("Test is final evaluation only")
    if config["dataset"]["target"] != "benchmark_label":
        raise ValueError("The only V0 training target is benchmark_label")
    if train["epochs"] < 1 or train["batch_size"] < 1 or train["num_workers"] < 0 or train["learning_rate"] <= 0:
        raise ValueError("Invalid training sizes/learning rate")
    vocab = load_label_vocab(resolve_path(config["dataset"]["label_vocab_path"]))
    return config, vocab


def config_summary(config, vocab) -> dict:
    fields = {"data_root": config["dataset"].get("data_root"),
              "manifest_path": config["dataset"].get("manifest_path"),
              "face.backend": config["face"].get("backend"),
              "face.cache_path": config["face"].get("cache_path"),
              "pretrained_checkpoint": config["model"].get("pretrained_checkpoint"),
              "landmark_checkpoint": config["model"].get("landmark_checkpoint")}
    return {"status": "CONFIG_VALID_ONLY", "data_or_weights_opened": False,
            "num_classes": vocab.num_classes, "label_vocab": vocab.as_dict(),
            "TA_LABEL_INDEX_MAPPING_UNVERIFIED": "NO" if vocab.ta_mapping_verified else "YES",
            "REMOTE_REQUIRED_FIELDS": [key for key, value in fields.items() if not value],
            "FACE_TRAINING_MODE": "cached_bbox",
            "FACE_DETECTOR_DEVICE": config["face"].get("device", "cpu"),
            "REMOTE_GPU_SMOKE_REQUIRED": "YES"}

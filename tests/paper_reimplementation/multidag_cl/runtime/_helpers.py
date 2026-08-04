from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import torch
import yaml

from datasets.iemocap.official_feature_dataset import iemocap_dialogue_collate_fn
from models.multidag_cl.paper_reimplementation.config import MultiDAGCLConfig
from scripts.runtime.multidag_cl_paper_reimplementation.adapter import (
    FeatureRegistryMetadata,
)
from scripts.runtime.multidag_cl_paper_reimplementation.trainer import (
    SyntheticDialogueDataset,
)


ROOT = Path(__file__).resolve().parents[4]
CONFIG_ROOT = (
    ROOT
    / "configs/multidag_cl/paper_reimplementation/iemocap/full_context/clean_roberta_features"
)
FORMAL_CONFIG = CONFIG_ROOT / "project_fair.yaml"
SYNTHETIC_CONFIG = CONFIG_ROOT / "synthetic_smoke.yaml"
REAL_CONFIG = CONFIG_ROOT / "real_batch_smoke.yaml"


def load_config(path: Path = SYNTHETIC_CONFIG) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def core_config(config: dict | None = None) -> MultiDAGCLConfig:
    value = load_config() if config is None else config
    return MultiDAGCLConfig.from_mapping(value["model_core"])


def feature_metadata(config: dict | None = None) -> FeatureRegistryMetadata:
    value = load_config() if config is None else config
    dataset = value["dataset"]
    dims = dataset["feature_dimensions"]
    return FeatureRegistryMetadata(
        registry_key=dataset["feature_registry"],
        feature_path=dataset["feature_path"],
        feature_sha256=dataset["feature_sha256"],
        text_dim=dims["text"],
        audio_dim=dims["audio"],
        visual_dim=dims["visual"],
    )


def synthetic_dataset(split: str = "train") -> SyntheticDialogueDataset:
    return SyntheticDialogueDataset(load_config(), split)


def synthetic_batch() -> dict:
    dataset = synthetic_dataset()
    return iemocap_dialogue_collate_fn([dataset[0], dataset[1]])


def mutated_config(path: Path = SYNTHETIC_CONFIG) -> dict:
    return deepcopy(load_config(path))


def assert_tensor_values_equal(left: dict, right: dict) -> None:
    for name, value in left.items():
        if isinstance(value, torch.Tensor):
            torch.testing.assert_close(value, right[name], rtol=0, atol=0)

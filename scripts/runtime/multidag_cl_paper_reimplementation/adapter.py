"""Thin, non-mutating adapter from the project batch to the Stage-B2 contract."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch

from models.multidag_cl.paper_reimplementation.config import MultiDAGCLConfig
from models.multidag_cl.paper_reimplementation.contracts import MultiDAGBatchContract


@dataclass(frozen=True)
class FeatureRegistryMetadata:
    registry_key: str
    feature_path: str
    feature_sha256: str
    text_dim: int
    audio_dim: int
    visual_dim: int

    @property
    def feature_dimensions(self) -> dict[str, int]:
        return {
            "text": self.text_dim,
            "audio": self.audio_dim,
            "visual": self.visual_dim,
        }


class ProjectBatchAdapter:
    """Validate an already-canonical batch and return the same mapping object."""

    def __init__(
        self,
        config: MultiDAGCLConfig,
        feature_metadata: FeatureRegistryMetadata,
    ) -> None:
        if not isinstance(feature_metadata, FeatureRegistryMetadata):
            raise TypeError("feature_metadata must be FeatureRegistryMetadata")
        configured = (
            config.text_feature_dim,
            config.audio_feature_dim,
            config.visual_feature_dim,
        )
        registered = (
            feature_metadata.text_dim,
            feature_metadata.audio_dim,
            feature_metadata.visual_dim,
        )
        if configured != registered:
            raise ValueError(
                f"model feature dimensions {configured} do not match registry {registered}"
            )
        self.config = config
        self.feature_metadata = feature_metadata
        self.contract = MultiDAGBatchContract(config)

    def adapt(
        self,
        batch: Mapping[str, Any],
        *,
        split: str,
        require_labels: bool = True,
    ) -> Mapping[str, Any]:
        tensor_objects = {
            name: value
            for name, value in batch.items()
            if isinstance(value, torch.Tensor)
        }
        self.contract.validate(
            batch,
            require_labels=require_labels,
            split=split,
        )
        for name, original in tensor_objects.items():
            if batch[name] is not original:
                raise RuntimeError(f"adapter unexpectedly replaced tensor field {name}")
        return batch

    def manifest_metadata(self) -> dict[str, Any]:
        return {
            "feature_registry": self.feature_metadata.registry_key,
            "feature_path": self.feature_metadata.feature_path,
            "feature_sha256": self.feature_metadata.feature_sha256,
            "feature_dimensions": self.feature_metadata.feature_dimensions,
            "adapter": "thin_validator_no_tensor_copy",
        }


__all__ = ["FeatureRegistryMetadata", "ProjectBatchAdapter"]

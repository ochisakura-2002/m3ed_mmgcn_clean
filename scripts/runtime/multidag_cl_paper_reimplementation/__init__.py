"""Public Stage-B3 runtime surface for the independent MultiDAG-CL lineage."""

from .adapter import FeatureRegistryMetadata, ProjectBatchAdapter
from .checkpoint import (
    TestEvaluationGate,
    ValidationCandidate,
    ValidationCheckpointSelector,
    strict_reload_checkpoint,
)
from .curriculum import CurriculumRuntime
from .optimizer import build_optimizer
from .trainer import SyntheticDialogueDataset, ValidationCleanCoordinator, run_runtime
from .validation import (
    LocalAssetUnavailable,
    OfficialAssetsUnavailable,
    RuntimeValidationError,
    validate_runtime_config,
)

__all__ = [
    "CurriculumRuntime",
    "FeatureRegistryMetadata",
    "LocalAssetUnavailable",
    "OfficialAssetsUnavailable",
    "ProjectBatchAdapter",
    "RuntimeValidationError",
    "SyntheticDialogueDataset",
    "TestEvaluationGate",
    "ValidationCandidate",
    "ValidationCheckpointSelector",
    "ValidationCleanCoordinator",
    "build_optimizer",
    "run_runtime",
    "strict_reload_checkpoint",
    "validate_runtime_config",
]

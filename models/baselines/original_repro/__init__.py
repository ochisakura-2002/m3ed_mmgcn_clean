"""Paper-oriented, noncausal MERC baseline reproductions."""

from .dialoguegcn import OriginalReproDialogueGCN
from .gsmcc import ProjectPaperOrientedGSMCC
from .mmgcn import OriginalReproMMGCN
from .multidag_cl import OriginalReproMultiDAGCL
from .registry import (
    ORIGINAL_REPRO_MODEL_KEYS,
    build_original_repro_model,
    get_model_constructor_args,
    get_source_metadata,
)

__all__ = [
    "ORIGINAL_REPRO_MODEL_KEYS",
    "OriginalReproDialogueGCN",
    "ProjectPaperOrientedGSMCC",
    "OriginalReproMMGCN",
    "OriginalReproMultiDAGCL",
    "build_original_repro_model",
    "get_model_constructor_args",
    "get_source_metadata",
]

"""Compatibility wrapper for canonical paper-aligned model paths."""

from models.dialoguegcn.paper_aligned import OriginalReproDialogueGCN
from models.gsmcc.project_variant.full_context import ProjectPaperOrientedGSMCC
from models.mmgcn.paper_aligned import OriginalReproMMGCN
from models.multidag_cl.paper_aligned import OriginalReproMultiDAGCL
from models.registry.paper_aligned import (
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

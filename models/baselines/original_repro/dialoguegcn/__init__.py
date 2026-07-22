"""Compatibility wrapper for :mod:`models.dialoguegcn.paper_aligned`."""

from models.dialoguegcn.paper_aligned import (
    DialogueGCNRelationalGraphNetwork,
    OriginalReproDialogueGCN,
    build_dialoguegcn_graph,
    dialoguegcn_relation_id,
)

__all__ = [
    "DialogueGCNRelationalGraphNetwork",
    "OriginalReproDialogueGCN",
    "build_dialoguegcn_graph",
    "dialoguegcn_relation_id",
]

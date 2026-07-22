"""Compatibility gates for the model-first package layout migration."""

from __future__ import annotations

import pytest

from models.baselines.causal_baseline_registry import (
    build_new_causal_baseline as old_build_new_causal_baseline,
)
from models.baselines.dialoguegcn import CausalDialogueGCNBaseline as OldCausalDialogueGCN
from models.baselines.gsmcc import CausalGSMCCInspiredBaseline as OldCausalGSMCC
from models.baselines.mmgcn.mm_gcn import M3EDMMGCN as OldM3EDMMGCN
from models.baselines.multidag_cl import MultiDAGCLBaseline as OldMultiDAGCL
from models.baselines.original_repro import (
    OriginalReproDialogueGCN as OldPaperAlignedDialogueGCN,
)
from models.baselines.original_repro import OriginalReproMMGCN as OldPaperAlignedMMGCN
from models.baselines.original_repro import (
    OriginalReproMultiDAGCL as OldPaperAlignedMultiDAGCL,
)
from models.baselines.original_repro import (
    ProjectPaperOrientedGSMCC as OldFullContextGSMCC,
)
from models.baselines.original_repro.registry import (
    build_original_repro_model as old_build_paper_aligned_model,
)
from models.dialoguegcn.paper_aligned import OriginalReproDialogueGCN
from models.dialoguegcn.unified import CausalDialogueGCNBaseline
from models.gsmcc.project_variant.causal import CausalGSMCCInspiredBaseline
from models.gsmcc.project_variant.full_context import ProjectPaperOrientedGSMCC
from models.mmgcn.paper_aligned import OriginalReproMMGCN
from models.mmgcn.unified.mm_gcn import M3EDMMGCN
from models.multidag_cl.paper_aligned import OriginalReproMultiDAGCL
from models.multidag_cl.unified import MultiDAGCLBaseline
from models.registry.causal import build_new_causal_baseline
from models.registry.paper_aligned import build_original_repro_model


@pytest.mark.parametrize(
    ("old_symbol", "canonical_symbol"),
    [
        (OldM3EDMMGCN, M3EDMMGCN),
        (OldPaperAlignedMMGCN, OriginalReproMMGCN),
        (OldMultiDAGCL, MultiDAGCLBaseline),
        (OldPaperAlignedMultiDAGCL, OriginalReproMultiDAGCL),
        (OldCausalDialogueGCN, CausalDialogueGCNBaseline),
        (OldPaperAlignedDialogueGCN, OriginalReproDialogueGCN),
        (OldCausalGSMCC, CausalGSMCCInspiredBaseline),
        (OldFullContextGSMCC, ProjectPaperOrientedGSMCC),
        (old_build_new_causal_baseline, build_new_causal_baseline),
        (old_build_paper_aligned_model, build_original_repro_model),
    ],
)
def test_old_and_canonical_imports_share_identity(old_symbol, canonical_symbol) -> None:
    assert old_symbol is canonical_symbol


def test_mmgcn_state_dict_schema_and_strict_load_are_path_independent() -> None:
    constructor_args = {
        "text_dim": 5,
        "audio_dim": 4,
        "visual_dim": 3,
        "hidden_dim": 6,
        "num_classes": 2,
        "num_layers": 1,
        "dropout": 0.0,
    }
    old_model = OldM3EDMMGCN(**constructor_args)
    canonical_model = M3EDMMGCN(**constructor_args)

    assert tuple(old_model.state_dict()) == tuple(canonical_model.state_dict())
    canonical_model.load_state_dict(old_model.state_dict(), strict=True)

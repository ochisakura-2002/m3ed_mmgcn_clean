"""Registry and source provenance for original MERC reproductions."""

from __future__ import annotations

import inspect
from typing import Any

from .common import CAUSAL_GRADE
from .dialoguegcn import OriginalReproDialogueGCN
from .gsmcc import ProjectPaperOrientedGSMCC
from .mmgcn import OriginalReproMMGCN
from .multidag_cl import OriginalReproMultiDAGCL


MODEL_REGISTRY = {
    "original_repro_mmgcn": OriginalReproMMGCN,
    "original_repro_multidag_cl": OriginalReproMultiDAGCL,
    "project_paper_oriented_gsmcc": ProjectPaperOrientedGSMCC,
    "original_repro_dialoguegcn": OriginalReproDialogueGCN,
}
ORIGINAL_REPRO_MODEL_KEYS = tuple(MODEL_REGISTRY)

MODEL_ALIASES = {
    "mmgcn": "original_repro_mmgcn",
    "original_mmgcn": "original_repro_mmgcn",
    "multidag_cl": "original_repro_multidag_cl",
    "multidag+cl": "original_repro_multidag_cl",
    "multidag-cl": "original_repro_multidag_cl",
    "gsmcc": "project_paper_oriented_gsmcc",
    "gs-mcc": "project_paper_oriented_gsmcc",
    "dialoguegcn": "original_repro_dialoguegcn",
    "dialogue_gcn": "original_repro_dialoguegcn",
}

SOURCE_PROVENANCE = {
    "original_repro_mmgcn": {
        "paper": "https://aclanthology.org/2021.acl-long.440/",
        "repository": "https://github.com/hujingwen6666/MMGCN",
        "commit": "85732d984f70c1c84dd47c81aa97f1271397b899",
        "implementation_status": "official_code_adapted",
        "paper_reproduction_eligible": "true",
    },
    "original_repro_multidag_cl": {
        "paper": "https://aclanthology.org/2024.lrec-main.380/",
        "repository": "https://github.com/vanntc711/MultiDAG-CL",
        "commit": "59a75877065a91bf9388fbb564607fe79717fd4f",
        "implementation_status": "official_code_adapted_with_protocol_repairs",
        "paper_reproduction_eligible": "true",
    },
    "project_paper_oriented_gsmcc": {
        "paper": "https://ojs.aaai.org/index.php/AAAI/article/view/33242",
        "repository": "https://github.com/FuchenZhang/GS-MCC",
        "commit": "38c4038a7738f9bf7b3132c3e99a126e1cf1f28d",
        "implementation_status": "PROJECT_VARIANT_NOT_PAPER_REPRODUCTION",
        "paper_reproduction_eligible": "false",
    },
    "original_repro_dialoguegcn": {
        "paper": "https://aclanthology.org/D19-1015/",
        "repository": "https://github.com/declare-lab/conv-emotion",
        "commit": "6128ca20e9c736605cce7e99d5d95db0356c35f5",
        "implementation_status": "paper_equation_aligned_official_code_adapted",
        "paper_reproduction_eligible": "true",
    },
}


def canonical_model_key(key: str) -> str:
    normalized = str(key).strip().lower()
    normalized = MODEL_ALIASES.get(normalized, normalized)
    if normalized not in MODEL_REGISTRY:
        choices = ", ".join(ORIGINAL_REPRO_MODEL_KEYS)
        raise KeyError(f"unknown original-reproduction model {key!r}; expected one of: {choices}")
    return normalized


def get_model_constructor_args(config: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Return the canonical key and validated constructor arguments."""

    model_config = dict(config.get("model", config))
    key = canonical_model_key(
        model_config.pop("key", model_config.pop("name", model_config.pop("model_key", "")))
    )
    requested_grade = model_config.pop("causal_grade", CAUSAL_GRADE)
    if requested_grade != CAUSAL_GRADE:
        raise ValueError(
            f"original-MERC models must declare causal_grade={CAUSAL_GRADE!r}, "
            f"got {requested_grade!r}"
        )
    fidelity_status = model_config.pop("fidelity_status", None)
    expected_status = SOURCE_PROVENANCE[key]["implementation_status"]
    if fidelity_status is not None and fidelity_status != expected_status:
        raise ValueError(
            f"model fidelity_status must be {expected_status!r}, got {fidelity_status!r}"
        )

    constructor = MODEL_REGISTRY[key]
    signature = inspect.signature(constructor.__init__)
    accepted = set(signature.parameters) - {"self"}
    unexpected = sorted(set(model_config) - accepted)
    if unexpected:
        raise TypeError(f"unexpected model options for {key}: {', '.join(unexpected)}")
    missing = [
        name
        for name, parameter in signature.parameters.items()
        if name != "self"
        and parameter.default is inspect.Parameter.empty
        and name not in model_config
    ]
    if missing:
        raise TypeError(f"missing model options for {key}: {', '.join(missing)}")
    return key, model_config


def build_original_repro_model(config: dict[str, Any]):
    key, constructor_args = get_model_constructor_args(config)
    return MODEL_REGISTRY[key](**constructor_args)


def get_source_metadata(key: str) -> dict[str, str]:
    return dict(SOURCE_PROVENANCE[canonical_model_key(key)])


__all__ = [
    "MODEL_REGISTRY",
    "ORIGINAL_REPRO_MODEL_KEYS",
    "SOURCE_PROVENANCE",
    "build_original_repro_model",
    "canonical_model_key",
    "get_model_constructor_args",
    "get_source_metadata",
]

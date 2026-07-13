"""Exact registry for the two project causal graph baselines added in Task 2."""

from __future__ import annotations

from typing import Any, Mapping

import torch.nn as nn

from models.baselines.dialoguegcn import (
    CausalDialogueGCNBaseline,
    CausalDialogueGCNConfig,
)
from models.baselines.gsmcc import CausalGSMCCConfig, CausalGSMCCInspiredBaseline


CAUSAL_GSMCC_NAME = "causal_gsmcc_inspired"
CAUSAL_DIALOGUEGCN_NAME = "causal_dialoguegcn"

_ALIASES = {
    CAUSAL_GSMCC_NAME: CAUSAL_GSMCC_NAME,
    "causal-gsmcc-inspired": CAUSAL_GSMCC_NAME,
    "causal_gsmcc": CAUSAL_GSMCC_NAME,
    "gsmcc": CAUSAL_GSMCC_NAME,
    "gsmcc_inspired": CAUSAL_GSMCC_NAME,
    "causalgsmccinspiredbaseline": CAUSAL_GSMCC_NAME,
    CAUSAL_DIALOGUEGCN_NAME: CAUSAL_DIALOGUEGCN_NAME,
    "causal-dialoguegcn": CAUSAL_DIALOGUEGCN_NAME,
    "dialoguegcn": CAUSAL_DIALOGUEGCN_NAME,
    "causaldialoguegcn": CAUSAL_DIALOGUEGCN_NAME,
    "causaldialoguegcnbaseline": CAUSAL_DIALOGUEGCN_NAME,
}


def normalize_new_causal_model_name(name: Any) -> str:
    """Return the canonical new-baseline name or raise a clear error."""

    key = str(name).strip().lower()
    normalized = _ALIASES.get(key)
    if normalized is None:
        supported = ", ".join((CAUSAL_GSMCC_NAME, CAUSAL_DIALOGUEGCN_NAME))
        raise ValueError(
            f"Unsupported new causal baseline name {name!r}; supported models: "
            f"{supported}."
        )
    return normalized


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping.")
    return value


def get_new_causal_model_family(config: Mapping[str, Any]) -> str:
    """Return the exact internal family key used by the runtime dispatcher."""

    model = _mapping(config.get("model", {}), "model")
    name = normalize_new_causal_model_name(model.get("name", ""))
    return "gsmcc" if name == CAUSAL_GSMCC_NAME else "dialoguegcn"


def validate_new_causal_model_config(config: Mapping[str, Any]) -> str:
    """Validate the strict causal contract and model/dataset dimensions."""

    model = _mapping(config.get("model", {}), "model")
    graph = _mapping(config.get("graph", {}), "graph")
    dataset = _mapping(config.get("dataset", {}), "dataset")
    name = normalize_new_causal_model_name(model.get("name", ""))

    if str(graph.get("context_mode", "")).strip().lower() != "causal":
        raise ValueError("graph.context_mode must be 'causal'.")
    if int(graph.get("window_future", 0)) != 0:
        raise ValueError("graph.window_future must be exactly 0.")
    if bool(model.get("bidirectional", False)):
        raise ValueError("model.bidirectional=True is forbidden.")
    if str(model.get("nodal_attention", "none")).strip().lower() == "full":
        raise ValueError("full nodal attention is forbidden for causal baselines.")

    dimension_pairs = (
        ("text_dim", "text_feature_dim"),
        ("audio_dim", "audio_feature_dim"),
        ("visual_dim", "visual_feature_dim"),
    )
    for model_key, dataset_key in dimension_pairs:
        if model_key not in model:
            raise ValueError(f"model.{model_key} is required.")
        if int(model[model_key]) <= 0:
            raise ValueError(f"model.{model_key} must be positive.")
        if dataset_key in dataset and int(dataset[dataset_key]) != int(model[model_key]):
            raise ValueError(
                f"dataset.{dataset_key} and model.{model_key} must match: "
                f"{dataset[dataset_key]} != {model[model_key]}."
            )

    model_classes = int(model.get("num_classes", 0))
    if model_classes <= 0:
        raise ValueError("model.num_classes must be positive.")
    if "num_classes" in dataset and int(dataset["num_classes"]) != model_classes:
        raise ValueError("dataset.num_classes and model.num_classes must match.")

    if name == CAUSAL_DIALOGUEGCN_NAME:
        encoder = str(model.get("context_encoder_type", "causal_gru")).lower()
        if encoder not in {"causal_gru", "linear"}:
            raise ValueError("DialogueGCN context_encoder_type must be causal_gru or linear.")
    return name


def build_new_causal_baseline(config: Mapping[str, Any]) -> nn.Module:
    """Build one of the two new models without affecting legacy builders."""

    name = validate_new_causal_model_config(config)
    model = _mapping(config["model"], "model")
    graph = _mapping(config["graph"], "graph")
    loss = _mapping(config.get("loss", {}), "loss")

    common = {
        "text_dim": int(model["text_dim"]),
        "audio_dim": int(model["audio_dim"]),
        "visual_dim": int(model["visual_dim"]),
        "hidden_dim": int(model["hidden_dim"]),
        "num_classes": int(model["num_classes"]),
        "dropout": float(model.get("dropout", 0.1)),
        "window_past": (
            None if graph.get("window_past") is None else int(graph["window_past"])
        ),
        "context_mode": str(graph.get("context_mode", "causal")),
        "window_future": int(graph.get("window_future", 0)),
        "bidirectional": bool(model.get("bidirectional", False)),
        "nodal_attention": str(model.get("nodal_attention", "none")),
    }
    if name == CAUSAL_GSMCC_NAME:
        typed_config = CausalGSMCCConfig(
            **common,
            num_filter_steps=int(model.get("num_filter_steps", 2)),
            num_graph_layers=int(model.get("num_graph_layers", 1)),
            modality_encoder_type=str(model.get("modality_encoder_type", "linear")),
            fusion_type=str(model.get("fusion_type", "concat")),
            classification_weight=float(loss.get("classification_weight", 1.0)),
            consistency_weight=float(loss.get("consistency_weight", 0.0)),
            complementarity_weight=float(loss.get("complementarity_weight", 0.0)),
        )
        built: nn.Module = CausalGSMCCInspiredBaseline(typed_config)
    else:
        typed_config = CausalDialogueGCNConfig(
            **common,
            context_hidden_dim=int(model.get("context_hidden_dim", model["hidden_dim"])),
            graph_hidden_dim=int(model.get("graph_hidden_dim", model["hidden_dim"])),
            context_encoder_type=str(model.get("context_encoder_type", "causal_gru")),
            num_graph_layers=int(model.get("num_graph_layers", 1)),
            num_speakers=int(model.get("num_speakers", 2)),
            fusion_type=str(model.get("fusion_type", "concat")),
        )
        built = CausalDialogueGCNBaseline(typed_config)

    classifier = getattr(built, "classifier", None)
    if classifier is None or int(classifier.out_features) != int(model["num_classes"]):
        raise RuntimeError("Built model classifier output dimension is inconsistent.")
    for module in built.modules():
        if getattr(module, "bidirectional", False):
            raise RuntimeError("Built model contains a bidirectional recurrent module.")
    return built


__all__ = [
    "CAUSAL_DIALOGUEGCN_NAME",
    "CAUSAL_GSMCC_NAME",
    "build_new_causal_baseline",
    "get_new_causal_model_family",
    "normalize_new_causal_model_name",
    "validate_new_causal_model_config",
]

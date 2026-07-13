from __future__ import annotations

import pytest

from models.baselines.causal_baseline_registry import (
    build_new_causal_baseline,
    get_new_causal_model_family,
    normalize_new_causal_model_name,
    validate_new_causal_model_config,
)


def _config(name: str) -> dict:
    model = {
        "name": name,
        "text_dim": 4,
        "audio_dim": 3,
        "visual_dim": 2,
        "hidden_dim": 8,
        "num_classes": 4,
        "dropout": 0.0,
        "num_graph_layers": 1,
    }
    if "dialogue" in name.lower():
        model.update(
            {
                "context_hidden_dim": 8,
                "graph_hidden_dim": 8,
                "context_encoder_type": "causal_gru",
                "num_speakers": 2,
                "nodal_attention": "none",
            }
        )
    else:
        model.update(
            {
                "num_filter_steps": 2,
                "modality_encoder_type": "linear",
                "fusion_type": "concat",
            }
        )
    return {
        "dataset": {
            "name": "SYNTHETIC",
            "num_classes": 4,
            "text_feature_dim": 4,
            "audio_feature_dim": 3,
            "visual_feature_dim": 2,
        },
        "model": model,
        "graph": {"context_mode": "causal", "window_past": 2, "window_future": 0},
        "loss": {"class_weight_mode": "none"},
    }


@pytest.mark.parametrize(
    ("name", "canonical", "family"),
    [
        ("CausalGSMCCInspiredBaseline", "causal_gsmcc_inspired", "gsmcc"),
        ("dialoguegcn", "causal_dialoguegcn", "dialoguegcn"),
    ],
)
def test_registry_builds_both_models(name: str, canonical: str, family: str) -> None:
    config = _config(name)
    assert normalize_new_causal_model_name(name) == canonical
    assert validate_new_causal_model_config(config) == canonical
    assert get_new_causal_model_family(config) == family
    model = build_new_causal_baseline(config)
    assert model.classifier.out_features == 4


def test_registry_rejects_unknown_model_name() -> None:
    with pytest.raises(ValueError, match="Unsupported new causal baseline"):
        normalize_new_causal_model_name("MMGCN")


@pytest.mark.parametrize(
    ("section", "key", "value"),
    [
        ("graph", "context_mode", "full"),
        ("graph", "window_future", 1),
        ("model", "bidirectional", True),
        ("model", "nodal_attention", "full"),
    ],
)
def test_registry_rejects_noncausal_configuration(
    section: str, key: str, value: object
) -> None:
    config = _config("causal_dialoguegcn")
    config[section][key] = value
    with pytest.raises(ValueError):
        validate_new_causal_model_config(config)

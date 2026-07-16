from __future__ import annotations

from pathlib import Path

import pytest
import torch
import yaml

from models.baselines.original_repro import build_original_repro_model


def _batch() -> dict[str, torch.Tensor]:
    torch.manual_seed(7)
    return {
        "text_features": torch.randn(2, 4, 8),
        "audio_features": torch.randn(2, 4, 6),
        "visual_features": torch.randn(2, 4, 5),
        "attention_mask": torch.tensor([[1, 1, 1, 1], [1, 1, 1, 0]]),
        "lengths": torch.tensor([4, 3]),
        "speaker_ids_int": torch.tensor([[0, 1, 0, 1], [1, 0, 1, 0]]),
        "labels": torch.tensor([[0, 1, 2, 1], [2, 0, 1, -100]]),
    }


@pytest.mark.parametrize(
    ("name", "extra"),
    [
        ("original_repro_mmgcn", {"hidden_dim": 8, "graph_layers": 1}),
        (
            "original_repro_multidag_cl",
            {"hidden_dim": 8, "graph_layers": 1, "curriculum_bucket_count": 2},
        ),
        (
            "project_paper_oriented_gsmcc",
            {"hidden_dim": 8, "graph_layers": 1, "window": 1},
        ),
        (
            "original_repro_dialoguegcn",
            {
                "context_hidden_dim": 4,
                "graph_hidden_dim": 4,
                "window_past": 1,
                "window_future": 1,
            },
        ),
    ],
)
def test_original_model_forward_backward_contract(name: str, extra: dict) -> None:
    config = {
        "model": {
            "name": name,
            "causal_grade": "noncausal_offline_full_context",
            "text_feature_dim": 8,
            "audio_feature_dim": 6,
            "visual_feature_dim": 5,
            "num_classes": 3,
            "dropout": 0.0,
            **extra,
        }
    }
    if name == "original_repro_dialoguegcn":
        config["model"]["use_class_weight"] = False
    model = build_original_repro_model(config)
    output = model(**_batch())
    assert set(output) == {
        "logits",
        "loss",
        "classification_loss",
        "aux_losses",
        "features",
        "diagnostics",
    }
    assert output["logits"].shape == (2, 4, 3)
    assert torch.isfinite(output["loss"])
    assert output["diagnostics"]["causal_grade"] == "noncausal_offline_full_context"
    output["loss"].backward()
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_gsmcc_contrastive_switch_changes_loss_contract() -> None:
    base = {
        "name": "project_paper_oriented_gsmcc",
        "text_feature_dim": 8,
        "audio_feature_dim": 6,
        "visual_feature_dim": 5,
        "num_classes": 3,
        "hidden_dim": 8,
        "graph_layers": 1,
        "window": 1,
        "dropout": 0.0,
    }
    enabled = build_original_repro_model({"model": {**base, "use_contrastive_loss": True}})
    disabled = build_original_repro_model({"model": {**base, "use_contrastive_loss": False}})
    enabled_output = enabled(**_batch())
    disabled_output = disabled(**_batch())
    assert "contrastive_loss" in enabled_output["aux_losses"]
    assert disabled_output["aux_losses"] == {}
    assert torch.allclose(
        enabled_output["loss"],
        enabled_output["classification_loss"]
        + enabled_output["aux_losses"]["contrastive_loss"],
    )
    assert not torch.allclose(
        enabled_output["features"]["low_frequency"],
        enabled_output["features"]["high_frequency"],
    )


def test_mmgcn_full_context_multimodal_and_speaker_paths() -> None:
    config = {
        "model": {
            "name": "original_repro_mmgcn",
            "text_feature_dim": 8,
            "audio_feature_dim": 6,
            "visual_feature_dim": 5,
            "num_classes": 3,
            "hidden_dim": 8,
            "graph_layers": 1,
            "dropout": 0.0,
        }
    }
    model = build_original_repro_model(config).eval()
    batch = _batch()
    output = model(**batch)
    adjacency = output["diagnostics"]["adjacency"]
    valid_utterances = int(batch["attention_mask"].sum())
    assert adjacency.shape == (3 * valid_utterances, 3 * valid_utterances)
    assert torch.count_nonzero(adjacency[:valid_utterances, valid_utterances:]) > 0
    encoded_text = output["features"]["modality_encoder_output"]["text"]
    assert not torch.allclose(encoded_text[0, 0], encoded_text[0, 1])


def test_original_repro_mmgcn_defaults_and_formal_configs_keep_residual_enabled() -> None:
    model = build_original_repro_model(
        {
            "model": {
                "name": "original_repro_mmgcn",
                "text_feature_dim": 8,
                "audio_feature_dim": 6,
                "visual_feature_dim": 5,
                "hidden_dim": 8,
                "graph_layers": 1,
            }
        }
    )
    assert model.graph_net.use_residual is True

    roots = [
        Path("configs/smoke/original_repro"),
        Path("configs/experiments/original_merc/screening"),
        Path("configs/experiments/original_merc/clean_screening"),
        Path("configs/experiments/original_merc/legacy_fold_bases"),
        Path("configs/experiments/original_merc/clean_fold_bases"),
    ]
    configs = [path for root in roots for path in root.glob("mmgcn_*.yaml")]
    assert len(configs) == 6
    for path in configs:
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert config["model"]["use_residual"] is True, path


def test_dialoguegcn_attention_and_ablation_switches() -> None:
    base = {
        "name": "original_repro_dialoguegcn",
        "text_feature_dim": 8,
        "audio_feature_dim": 6,
        "visual_feature_dim": 5,
        "num_classes": 3,
        "context_hidden_dim": 4,
        "graph_hidden_dim": 4,
        "dropout": 0.0,
        "window_past": 1,
        "window_future": 1,
    }
    weighted = build_original_repro_model(
        {"model": {**base, "use_class_weight": True, "class_weight": [1.0, 2.0, 3.0]}}
    )
    ablated = build_original_repro_model(
        {"model": {**base, "use_class_weight": False, "use_nodal_attention": False}}
    )
    output = weighted(**_batch())
    edge_attention = output["diagnostics"]["edge_attention"]
    valid_rows = output["diagnostics"]["adjacency"].any(dim=-1)
    assert torch.allclose(edge_attention.sum(dim=-1)[valid_rows], torch.ones_like(edge_attention.sum(dim=-1)[valid_rows]))
    assert weighted.class_weight is not None
    assert ablated.class_weight is None
    assert not ablated.use_nodal_attention

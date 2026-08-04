from __future__ import annotations

from copy import deepcopy

import torch

from models.multidag_cl.paper_reimplementation.config import MultiDAGCLConfig


def config_mapping(
    *,
    profile: str = "paper_formula_behavior",
    graph_layers: int = 1,
    causal_text_ablation: bool = False,
    curriculum_enabled: bool = True,
) -> dict:
    source = profile == "official_source_behavior"
    return {
        "identity": {
            "canonical_name": "multidag_cl_paper_reimplementation",
            "display_name": "MultiDAG-CL Paper Reimplementation",
            "implementation_identity": "paper_reimplementation",
            "conformance_profile": profile,
        },
        "data": {
            "track": "project_fair",
            "text_feature_dim": 4,
            "audio_feature_dim": 3,
            "visual_feature_dim": 2,
            "num_classes": 3,
        },
        "encoder": {
            "profile": (
                "official_source_single_projection"
                if source
                else "paper_modality_specific"
            ),
            "modality_order": (
                ["text", "audio", "visual"]
                if source
                else ["audio", "visual", "text"]
            ),
            "modality_output_dims": (
                {"audio": 0, "visual": 0, "text": 0}
                if source
                else {"audio": 100, "visual": 100, "text": 100}
            ),
            "text_sequence_axis": "not_applicable" if source else "dialogue_utterance",
            "text_bidirectional": False if source or causal_text_ablation else True,
            "causal_text_ablation": causal_text_ablation,
            "single_projection": source,
        },
        "graph": {
            "predecessor_profile": "official_same_speaker_count_window",
            "window_past_same_speaker": 1,
            "window_future": 0,
            "allow_self_edge": False,
            "allow_future_edge": False,
            "global_nodal_attention": False,
        },
        "attention": {
            "score": "concat_linear",
            "relation_on_values": True,
            "dropout": 0.0,
        },
        "dag": {
            "hidden_dim": 300,
            "layers": graph_layers,
            "dual_gru": "swapped_input_hidden_sum",
            "layer_parameter_sharing": False,
            "representation": (
                "encoder_plus_all_dag_layers_plus_raw_features"
                if source
                else "encoder_plus_all_dag_layers"
            ),
            "raw_feature_skip": source,
        },
        "classifier": {
            "hidden_dim": 300,
            "hidden_layers": 2,
            "activation": "relu",
            "dropout": 0.4,
        },
        "curriculum": {
            "enabled": curriculum_enabled,
            "bucket_count": 12 if source else 5,
            "partition": "source_ceiling_chunks" if source else "balanced_stable_contiguous",
            "schedule": "official_one_bucket_per_epoch",
        },
        "training": {
            "epochs": 30,
            "batch_size": 16,
            "seed": 100,
            "optimizer": {
                "name": "adamw_transformers_3_5_1_compatible",
                "learning_rate": 5.0e-4,
                "betas": [0.9, 0.999],
                "eps": 1.0e-6,
                "weight_decay": 0.0,
                "bias_correction": True,
                "parameter_grouping": "all_parameters_single_group",
            },
            "gradient_clip_norm": 5.0,
            "scheduler": "none",
            "amp": False,
            "early_stopping": False,
        },
        "loss": {
            "name": "cross_entropy",
            "class_weight": None,
            "label_smoothing": 0.0,
            "ignore_index": -100,
        },
        "checkpoint": {"test_split_used_for_selection": False},
    }


def make_config(**kwargs) -> MultiDAGCLConfig:
    return MultiDAGCLConfig.from_mapping(config_mapping(**kwargs))


def make_batch() -> dict:
    torch.manual_seed(11)
    text = torch.randn(2, 4, 4)
    audio = torch.randn(2, 4, 3)
    visual = torch.randn(2, 4, 2)
    text[1, 3] = 0
    audio[1, 3] = 0
    visual[1, 3] = 0
    return {
        "dialogue_ids": ["d0", "d1"],
        "utterance_ids": [["d0_u0", "d0_u1", "d0_u2", "d0_u3"], ["d1_u0", "d1_u1", "d1_u2"]],
        "text_features": text,
        "audio_features": audio,
        "visual_features": visual,
        "labels": torch.tensor([[0, 1, 2, 1], [2, 0, 1, -100]], dtype=torch.int64),
        "speaker_ids_int": torch.tensor([[0, 1, 0, 1], [1, 0, 1, 0]], dtype=torch.int64),
        "attention_mask": torch.tensor([[1, 1, 1, 1], [1, 1, 1, 0]], dtype=torch.int64),
        "lengths": torch.tensor([4, 3], dtype=torch.int64),
    }


def cloned_batch(batch: dict) -> dict:
    copied = deepcopy(batch)
    for name, value in batch.items():
        if isinstance(value, torch.Tensor):
            copied[name] = value.clone()
    return copied

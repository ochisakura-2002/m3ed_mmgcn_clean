from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import torch
import yaml
import pytest

from models.registry.paper_aligned import build_original_repro_model
from scripts.baselines.original_merc_runtime import (
    SyntheticDialogueDataset,
    load_checkpoint,
    rebuild_model_from_checkpoint,
)
import scripts.baselines.train_original_merc_baseline as train_module
from scripts.baselines.train_original_merc_baseline import run_training


def _synthetic_config(output_root: Path) -> dict:
    return {
        "run_name": "original_dialoguegcn_synthetic_test",
        "model": {
            "name": "original_repro_dialoguegcn",
            "causal_grade": "noncausal_offline_full_context",
            "text_feature_dim": 8,
            "audio_feature_dim": 6,
            "visual_feature_dim": 5,
            "num_classes": 3,
            "context_hidden_dim": 4,
            "graph_hidden_dim": 4,
            "num_speakers": 2,
            "dropout": 0.0,
            "use_nodal_attention": True,
            "use_class_weight": False,
            "base_model": "GRU",
            "window_past": 1,
            "window_future": 1,
            "active_modalities": ["text"],
        },
        "dataset": {"name": "SYNTHETIC", "num_classes": 3, "label_list": ["a", "b", "c"]},
        "synthetic": {"split_sizes": {"train": 4, "val": 4, "test": 5}, "sequence_length": 4},
        "training": {
            "epochs": 2,
            "batch_size": 2,
            "seed": 5,
            "select_best_by": "val_weighted_f1",
            "grad_clip": 1.0,
            "num_workers": 0,
            "early_stopping_patience": 0,
            "amp": False,
            "max_train_batches": 1,
            "max_eval_batches": 1,
        },
        "optimizer": {"name": "Adam", "learning_rate": 0.001, "weight_decay": 0.0},
        "scheduler": {"name": "none"},
        "system": {"seed": 5, "device": "cpu"},
        "output": {"run_root": str(output_root)},
    }


def test_train_checkpoint_reload_and_test_artifacts(monkeypatch) -> None:
    test_root = Path("tmp") / f"pytest_original_runtime_{uuid4().hex}"
    test_root.mkdir(parents=True)
    config_path = test_root / "config.yaml"
    config = _synthetic_config(test_root / "runs")
    with config_path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(config, file, sort_keys=False)
    train_batch_caps = []
    eval_batch_caps = []
    original_train_one_epoch = train_module.train_one_epoch
    original_evaluate_model = train_module.evaluate_model

    def recording_train_one_epoch(*args, **kwargs):
        train_batch_caps.append(args[7])
        return original_train_one_epoch(*args, **kwargs)

    def recording_evaluate_model(*args, **kwargs):
        eval_batch_caps.append(args[4])
        return original_evaluate_model(*args, **kwargs)

    monkeypatch.setattr(train_module, "train_one_epoch", recording_train_one_epoch)
    monkeypatch.setattr(train_module, "evaluate_model", recording_evaluate_model)
    result = run_training(config_path)
    assert result["run_status"] == "PASS"
    assert result["numeric_status"] == "FINITE"
    assert result["checkpoint_numeric_validation"] == "passed"
    assert result["checkpoint_nonfinite_tensor_count"] == 0
    assert result["checkpoint_nonfinite_element_count"] == 0
    assert result["checkpoint_parameters_finite"] is True
    assert result["final_metrics_finite"] is True
    assert result["prediction_count_correct"] is True
    run_dir = Path(result["run_dir"])
    best_path = run_dir / "checkpoints" / "best_model.pt"
    last_path = run_dir / "checkpoints" / "last_model.pt"
    assert best_path.is_file() and last_path.is_file()
    checkpoint = load_checkpoint(best_path, torch.device("cpu"))
    assert checkpoint["test_split_used_for_selection"] is False
    rebuilt = rebuild_model_from_checkpoint(checkpoint, torch.device("cpu"))
    assert not rebuilt.training
    second_reload = rebuild_model_from_checkpoint(checkpoint, torch.device("cpu"))
    torch.manual_seed(19)
    batch = {
        "text_features": torch.randn(1, 3, 8),
        "audio_features": torch.randn(1, 3, 6),
        "visual_features": torch.randn(1, 3, 5),
        "attention_mask": torch.ones(1, 3, dtype=torch.long),
        "lengths": torch.tensor([3]),
        "speaker_ids_int": torch.tensor([[0, 1, 0]]),
        "labels": torch.tensor([[0, 1, 2]]),
    }
    with torch.no_grad():
        first_logits = rebuilt(**batch)["logits"]
        second_logits = second_reload(**batch)["logits"]
    torch.testing.assert_close(first_logits, second_logits, rtol=0, atol=0)
    evaluation = run_dir / "logs" / "evaluations" / "test_best_model"
    assert (evaluation / "metrics.csv").is_file()
    assert (evaluation / "predictions.csv").is_file()
    assert (evaluation / "confusion_matrix.csv").is_file()
    assert (evaluation / "per_class_metrics.csv").is_file()
    expected_val_count = sum(
        int(SyntheticDialogueDataset(config, "val")[index]["length"])
        for index in range(len(SyntheticDialogueDataset(config, "val")))
    )
    expected_test_count = sum(
        int(SyntheticDialogueDataset(config, "test")[index]["length"])
        for index in range(len(SyntheticDialogueDataset(config, "test")))
    )
    assert expected_val_count > config["training"]["batch_size"]
    assert expected_test_count > config["training"]["batch_size"]
    assert result["final_validation_prediction_count"] == expected_val_count
    assert result["final_test_prediction_count"] == expected_test_count
    assert result["outer_test_valid_utterance_count"] == expected_test_count
    prediction_count = len(
        (evaluation / "predictions.csv").read_text(encoding="utf-8-sig").splitlines()
    ) - 1
    assert prediction_count == expected_test_count
    assert train_batch_caps == [1, 1]
    assert eval_batch_caps == [1, 1, None, None]


def test_feature_failure_happens_before_model_initialization(monkeypatch) -> None:
    calls = {"model_builds": 0}

    def counted_build(config):
        calls["model_builds"] += 1
        return build_original_repro_model(config)

    monkeypatch.setattr(train_module, "build_original_repro_model", counted_build)
    config = _synthetic_config(Path("tmp/unused_original_runtime_output"))
    config["dataset"] = {
        "name": "IEMOCAP",
        "feature_pkl_path": f"tmp/definitely_missing_{uuid4().hex}.pkl",
        "feature_sha256": "0" * 64,
        "feature_protocol": "legacy_mmgcn_features_v1",
        "feature_cleanliness": "legacy_emotion_supervised_text_features",
        "usage": "reproduction_diagnostic",
        "experiment_track": "legacy_official_split_safe_selection",
        "protocol_comparability": "paper_adjacent_not_exact",
        "num_classes": 3,
        "label_list": ["a", "b", "c"],
        "val_split_strategy": "official_train_stratified",
        "outer_test_session": "Ses05",
        "inner_val_ratio": 0.1,
        "split_seed": 5,
    }
    config_root = Path("tmp") / f"pytest_feature_order_{uuid4().hex}"
    config_root.mkdir(parents=True)
    config_path = config_root / "missing_feature.yaml"
    with config_path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(config, file, sort_keys=False)
    with pytest.raises(FileNotFoundError):
        run_training(config_path, dry_run_only=True)
    assert calls["model_builds"] == 0

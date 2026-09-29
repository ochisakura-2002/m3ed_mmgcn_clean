from __future__ import annotations

import math
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

from datasets.iemocap.official_feature_dataset import iemocap_dialogue_collate_fn
from models.multidag_cl.paper_reimplementation.config import MultiDAGCLConfig
from models.registry.paper_reimplementation import build_paper_reimplementation_model
import scripts.models.multidag_cl.paper_reimplementation.train as cli
import scripts.runtime.multidag_cl_paper_reimplementation.diagnostic_artifacts as artifact_module
from scripts.runtime.multidag_cl_paper_reimplementation.adapter import (
    FeatureRegistryMetadata,
    ProjectBatchAdapter,
)
from scripts.runtime.multidag_cl_paper_reimplementation.checkpoint import (
    load_checkpoint,
    save_checkpoint_atomic,
)
from scripts.runtime.multidag_cl_paper_reimplementation.diagnostics import (
    CHECKPOINT_TYPES,
    PREDICTION_FIELDS,
    DiagnosticCheckpointTracker,
    annotate_checkpoint_payload,
    checkpoint_allows_test_evaluation,
    checkpoint_summary_rows,
    compute_per_class_metrics,
    compute_probability_metrics,
    named_prediction_rows,
    parameter_count_row,
    parse_diagnostic_settings,
    prediction_flip_details,
    prediction_flip_summary,
    reliability_bins,
)
from scripts.runtime.multidag_cl_paper_reimplementation.evaluation import evaluate_model
import scripts.runtime.multidag_cl_paper_reimplementation.trainer as trainer_module
import scripts.runtime.multidag_cl_paper_reimplementation.validation as validation_module


ROOT = Path(__file__).resolve().parents[4]
BASE_DIR = ROOT / (
    "configs/multidag_cl/paper_reimplementation/iemocap/full_context/"
    "paper_data_reproduction/causal_text_encoder_diagnostic/modality_ablation"
)
CONFIG_DIR = BASE_DIR.parent / "overfitting_diagnostic"
SUBSETS = {
    "tav": ("text", "audio", "visual"),
    "ta": ("text", "audio"),
    "tv": ("text", "visual"),
    "t": ("text",),
    "av": ("audio", "visual"),
    "a": ("audio",),
    "v": ("visual",),
}
LABELS = ["happy", "sad", "neutral", "angry", "excited", "frustrated"]


def _load(path: Path) -> dict:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


@pytest.mark.parametrize("key,active", SUBSETS.items())
def test_seven_overfitting_configs_are_controlled_derivations_and_unique(
    key: str,
    active: tuple[str, ...],
) -> None:
    source = _load(BASE_DIR / f"cl7_unilstm_{key}.yaml")
    derived = _load(CONFIG_DIR / f"cl7_unilstm_{key}_overfitdiag.yaml")
    settings = parse_diagnostic_settings(derived)
    assert settings.enabled
    assert settings.ece_bins == 15
    assert settings.export_dag_attention is False
    assert all(
        getattr(settings, name)
        for name in (
            "save_extended_epoch_metrics",
            "save_per_class_epoch_metrics",
            "save_checkpoint_summary",
            "save_selected_val_predictions",
            "save_prediction_flip_analysis",
            "save_calibration_summary",
            "save_reliability_bins",
            "save_parameter_count",
        )
    )
    assert derived["model_core"]["modality_ablation"]["active_modalities"] == list(active)
    assert derived["run_name"] == (
        f"multidag_cl_unilstm_{key}_overfitdiag_paper_data_cl7_seed100"
    )
    assert derived["output"]["experiment_group"] == (
        "multidag_cl_unilstm_overfitting_diagnostic_paper_data_cl7"
    )
    del derived["diagnostics"]
    derived["run_name"] = source["run_name"]
    derived["output"]["experiment_group"] = source["output"]["experiment_group"]
    assert derived == source


def test_formal_entry_check_constructs_all_seven_without_data_or_training(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed = []
    original_builder = trainer_module.build_paper_reimplementation_model

    def metadata(config, *, project_root, require_file, verify_checksum):
        del project_root, verify_checksum
        assert require_file is False
        return FeatureRegistryMetadata(
            registry_key=config["dataset"]["feature_registry"],
            feature_path=config["dataset"]["feature_path"],
            feature_sha256="0" * 64,
            text_dim=1024,
            audio_dim=1582,
            visual_dim=342,
        )

    def observe_builder(config, **kwargs):
        model = original_builder(config, **kwargs)
        observed.append(model.config.active_modalities)
        return model

    def forbidden_dataset(*args, **kwargs):
        pytest.fail("check mode accessed a dataset")

    monkeypatch.setattr(validation_module, "resolve_feature_metadata", metadata)
    monkeypatch.setattr(trainer_module, "build_paper_reimplementation_model", observe_builder)
    monkeypatch.setattr(trainer_module, "_make_dataset", forbidden_dataset)
    for key, active in SUBSETS.items():
        result = cli.main(
            [
                "--mode",
                "check",
                "--config",
                str(CONFIG_DIR / f"cl7_unilstm_{key}_overfitdiag.yaml"),
                "--device",
                "cpu",
            ]
        )
        assert result["status"] == "PASS"
        assert result["optimizer_steps"] == 0
        assert result["model_parameter_count"] == 5_975_110
        assert result["parameter_count_breakdown"]["total_parameters"] == 5_975_110
        assert "run_dir" not in result
        assert observed[-1] == active


def test_diagnostics_disabled_is_the_backward_compatible_default() -> None:
    config = _load(BASE_DIR / "cl7_unilstm_tav.yaml")
    settings = parse_diagnostic_settings(config)
    assert settings.enabled is False
    assert settings.to_mapping() == {
        "enabled": False,
        "ece_bins": 15,
        "save_extended_epoch_metrics": False,
        "save_per_class_epoch_metrics": False,
        "save_checkpoint_summary": False,
        "save_selected_val_predictions": False,
        "save_prediction_flip_analysis": False,
        "save_calibration_summary": False,
        "save_reliability_bins": False,
        "save_parameter_count": False,
        "export_dag_attention": False,
    }


def test_probability_and_per_class_metrics_match_synthetic_reference() -> None:
    logits = torch.tensor(
        [
            [3.0, 0.0, 0.0],
            [0.0, 3.0, 0.0],
            [0.0, 3.0, 0.0],
            [0.0, 0.0, 3.0],
        ]
    )
    labels = torch.tensor([0, 1, 2, 2])
    metrics = compute_probability_metrics(
        logits, labels, num_classes=3, ece_bins=5
    )
    assert metrics["accuracy"] == pytest.approx(0.75)
    assert metrics["weighted_f1"] == pytest.approx(0.75)
    assert metrics["macro_f1"] == pytest.approx((1.0 + 2 / 3 + 2 / 3) / 3)
    assert metrics["uar"] == pytest.approx((1.0 + 1.0 + 0.5) / 3)
    assert metrics["nll"] == pytest.approx(
        torch.nn.functional.cross_entropy(logits.double(), labels).item()
    )
    probabilities = torch.softmax(logits.double(), dim=1)
    one_hot = torch.nn.functional.one_hot(labels, num_classes=3).double()
    assert metrics["brier"] == pytest.approx(
        ((probabilities - one_hot) ** 2).sum(dim=1).mean().item()
    )
    assert 0.0 <= metrics["ece"] <= 1.0
    assert metrics["mean_confidence_correct"] == pytest.approx(
        probabilities.max(dim=1).values[:2].mean().item()
    )
    per_class = compute_per_class_metrics(
        labels.tolist(), logits.argmax(dim=1).tolist(), ["a", "b", "c"]
    )
    assert [row["class"] for row in per_class] == ["a", "b", "c"]
    assert [row["support"] for row in per_class] == [1, 1, 2]


def _epoch_metrics(loss: float, wf1: float, uar: float) -> dict[str, float]:
    return {
        "val_loss": loss,
        "val_accuracy": 0.5,
        "val_weighted_f1": wf1,
        "val_macro_f1": 0.4,
        "val_uar": uar,
        "val_nll": loss,
        "val_brier": 0.3,
        "val_ece": 0.2,
        "val_mean_confidence_correct": 0.7,
        "val_mean_confidence_wrong": 0.6,
    }


def test_four_checkpoint_selection_and_ties_are_deterministic() -> None:
    tracker = DiagnosticCheckpointTracker()
    tracker.update(epoch=1, metrics=_epoch_metrics(0.5, 0.6, 0.4))
    tracker.update(epoch=2, metrics=_epoch_metrics(0.4, 0.6, 0.5))
    tracker.update(epoch=3, metrics=_epoch_metrics(0.4, 0.6, 0.5))
    records = {record.checkpoint_type: record for record in tracker.ordered_records()}
    assert records["min_val_loss"].epoch == 2
    assert records["best_val_wf1"].epoch == 2
    assert records["best_val_uar"].epoch == 2
    assert records["final"].epoch == 3
    rows = checkpoint_summary_rows(tracker, "TAV")
    assert [row["checkpoint_type"] for row in rows] == list(CHECKPOINT_TYPES)
    assert set(rows[0]) >= {
        "condition",
        "checkpoint_type",
        "epoch",
        "val_loss",
        "val_weighted_f1",
        "val_uar",
        "val_ece",
    }

    checkpoint_dir = ROOT / "tmp/assistant_work/overfitdiag_checkpoint_schema_test"
    for checkpoint_type in CHECKPOINT_TYPES:
        payload = annotate_checkpoint_payload(
            {"model_state_dict": {}, "checkpoint_locked": checkpoint_type == "best_val_wf1"},
            checkpoint_type,
        )
        path = checkpoint_dir / f"{checkpoint_type}.pt"
        save_checkpoint_atomic(path, payload)
        loaded = load_checkpoint(path, torch.device("cpu"))
        assert loaded["diagnostic_checkpoint_type"] == checkpoint_type
        assert checkpoint_allows_test_evaluation(loaded) is (
            checkpoint_type == "best_val_wf1"
        )


def _prediction_result(probabilities: list[list[float]], true: list[int]) -> dict:
    rows = []
    for index, (values, true_id) in enumerate(zip(probabilities, true)):
        predicted = max(range(len(values)), key=values.__getitem__)
        rows.append(
            {
                "dialogue_id": "d1",
                "utterance_id": f"u{index}",
                "speaker_id": index % 2,
                "true_label": true_id,
                "predicted_label": predicted,
                "probabilities": values,
                "pred_confidence": values[predicted],
                "true_label_probability": values[true_id],
                "correct": predicted == true_id,
            }
        )
    return {"predictions": rows}


def test_prediction_schema_flips_and_reliability_are_complete() -> None:
    early_result = _prediction_result(
        [
            [0.8, 0.04, 0.04, 0.04, 0.04, 0.04],
            [0.6, 0.2, 0.05, 0.05, 0.05, 0.05],
            [0.1, 0.1, 0.6, 0.1, 0.05, 0.05],
            [0.6, 0.1, 0.1, 0.1, 0.05, 0.05],
        ],
        [0, 1, 2, 3],
    )
    late_result = _prediction_result(
        [
            [0.7, 0.06, 0.06, 0.06, 0.06, 0.06],
            [0.1, 0.7, 0.05, 0.05, 0.05, 0.05],
            [0.7, 0.05, 0.1, 0.05, 0.05, 0.05],
            [0.8, 0.04, 0.04, 0.04, 0.04, 0.04],
        ],
        [0, 1, 2, 3],
    )
    early = named_prediction_rows(early_result, LABELS)
    late = named_prediction_rows(late_result, LABELS)
    assert set(early[0]) == set(PREDICTION_FIELDS)
    for row in early:
        assert sum(row[f"prob_{name}"] for name in LABELS) == pytest.approx(1.0)
        assert row["pred_confidence"] == max(row[f"prob_{name}"] for name in LABELS)
        assert row["true_label_probability"] == row[f"prob_{row['true_label']}"]
    details = prediction_flip_details("TAV", "min_val_loss->final", early, late)
    assert [row["flip_type"] for row in details] == ["CC", "WC", "CW", "WW"]
    summary = prediction_flip_summary(details)
    all_rows = [row for row in summary if row["true_label"] == "ALL"]
    assert sum(row["count"] for row in all_rows) == len(details)

    probabilities = torch.tensor(
        [[row[f"prob_{name}"] for name in LABELS] for row in early]
    )
    true_ids = torch.tensor([LABELS.index(row["true_label"]) for row in early])
    bins = reliability_bins(probabilities, true_ids, ece_bins=15)
    assert sum(row["count"] for row in bins) == len(early)


def test_parameter_count_is_identical_for_all_seven_conditions() -> None:
    counts = []
    for key in SUBSETS:
        model = build_paper_reimplementation_model(
            _load(CONFIG_DIR / f"cl7_unilstm_{key}_overfitdiag.yaml")
        )
        row = parameter_count_row(model, key.upper())
        counts.append(row["total_parameters"])
        assert row["trainable_parameters"] == row["total_parameters"]
        assert (
            row["encoder_parameters"]
            + row["dag_parameters"]
            + row["classifier_parameters"]
            + row["fusion_parameters"]
            == row["total_parameters"]
        )
    assert counts == [5_975_110] * 7


def test_attention_instrumentation_does_not_change_tav_logits() -> None:
    config = _load(CONFIG_DIR / "cl7_unilstm_tav_overfitdiag.yaml")
    torch.manual_seed(20260929)
    plain = build_paper_reimplementation_model(config).eval()
    instrumented = build_paper_reimplementation_model(
        config, collect_attention_diagnostics=True
    ).eval()
    instrumented.load_state_dict(plain.state_dict(), strict=True)
    generator = torch.Generator().manual_seed(11)
    inputs = {
        "text_features": torch.randn(1, 3, 1024, generator=generator),
        "audio_features": torch.randn(1, 3, 1582, generator=generator),
        "visual_features": torch.randn(1, 3, 342, generator=generator),
        "attention_mask": torch.ones(1, 3, dtype=torch.long),
        "lengths": torch.tensor([3], dtype=torch.long),
        "speaker_ids_int": torch.tensor([[0, 1, 0]], dtype=torch.long),
    }
    with torch.no_grad():
        plain_output = plain(**inputs)
        diagnostic_output = instrumented(**inputs)
    torch.testing.assert_close(plain_output.logits, diagnostic_output.logits, rtol=0, atol=0)
    assert plain_output.diagnostics is None
    assert diagnostic_output.diagnostics is not None
    assert diagnostic_output.diagnostics.layer_diagnostics[-1].attention_weights is not None


def test_extended_evaluation_emits_probabilities_metrics_classes_and_attention() -> None:
    config = _load(CONFIG_DIR / "cl7_unilstm_tav_overfitdiag.yaml")
    core = MultiDAGCLConfig.from_mapping(config["model_core"])
    model = build_paper_reimplementation_model(
        config, collect_attention_diagnostics=True
    ).eval()
    generator = torch.Generator().manual_seed(23)
    item = {
        "dialogue_id": "dialogue_1",
        "utterance_ids": ["u0", "u1", "u2"],
        "sentences": ["a", "b", "c"],
        "text_features": torch.randn(3, 1024, generator=generator),
        "audio_features": torch.randn(3, 1582, generator=generator),
        "visual_features": torch.randn(3, 342, generator=generator),
        "labels": torch.tensor([0, 1, 2], dtype=torch.long),
        "speaker_ids_int": torch.tensor([0, 1, 0], dtype=torch.long),
        "length": 3,
    }
    batch = iemocap_dialogue_collate_fn([item])
    feature = FeatureRegistryMetadata(
        registry_key="multidag_cl_official_2948_v1",
        feature_path=config["dataset"]["feature_path"],
        feature_sha256="0" * 64,
        text_dim=1024,
        audio_dim=1582,
        visual_dim=342,
    )
    result = evaluate_model(
        model,
        [batch],
        adapter=ProjectBatchAdapter(core, feature),
        device=torch.device("cpu"),
        split="validation",
        labels=list(range(6)),
        diagnostics_enabled=True,
        ece_bins=15,
        label_names=LABELS,
        collect_attention=True,
    )
    assert result["metrics"]["prediction_count"] == 3
    assert set(result["diagnostic_metrics"]) == {
        "accuracy",
        "weighted_f1",
        "macro_f1",
        "uar",
        "nll",
        "brier",
        "ece",
        "mean_confidence_correct",
        "mean_confidence_wrong",
    }
    assert [row["class"] for row in result["per_class_metrics"]] == LABELS
    assert sum(row["support"] for row in result["per_class_metrics"]) == 3
    assert all(
        sum(row["probabilities"]) == pytest.approx(1.0)
        for row in result["predictions"]
    )
    assert result["attention_rows"]
    assert set(result["attention_rows"][0]) == {
        "dialogue_id",
        "target_utterance_id",
        "source_utterance_id",
        "layer_index",
        "attention_weight",
    }


def test_artifact_writer_emits_required_diagnostic_files(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _load(CONFIG_DIR / "cl7_unilstm_tav_overfitdiag.yaml")
    core = MultiDAGCLConfig.from_mapping(config["model_core"])
    feature = FeatureRegistryMetadata(
        registry_key=config["dataset"]["feature_registry"],
        feature_path=config["dataset"]["feature_path"],
        feature_sha256="0" * 64,
        text_dim=1024,
        audio_dim=1582,
        visual_dim=342,
    )
    model = build_paper_reimplementation_model(config)
    tracker = DiagnosticCheckpointTracker()
    for epoch in range(1, 5):
        tracker.update(
            epoch=epoch,
            metrics=_epoch_metrics(1.0 / epoch, 0.4 + epoch / 100, 0.3 + epoch / 100),
        )
    run_dir = ROOT / "tmp/assistant_work/overfitdiag_artifact_writer_test/run"
    paths = SimpleNamespace(
        run_dir=run_dir,
        checkpoints=run_dir / "checkpoints",
        reports=run_dir / "reports",
        predictions=run_dir / "predictions",
    )
    calls = []

    def fake_reload(path, **kwargs):
        calls.append((Path(path).name, kwargs["require_locked"]))
        return {}

    def fake_evaluate(*args, split, **kwargs):
        checkpoint_type = split.removeprefix("validation_")
        offset = CHECKPOINT_TYPES.index(checkpoint_type) * 0.01
        probabilities = torch.tensor(
            [
                [0.70 - offset, 0.06 + offset, 0.06, 0.06, 0.06, 0.06],
                [0.10, 0.60 - offset, 0.10 + offset, 0.05, 0.10, 0.05],
                [0.50 + offset, 0.10, 0.20 - offset, 0.05, 0.10, 0.05],
                [0.10, 0.10, 0.10, 0.50 + offset, 0.10, 0.10 - offset],
            ],
            dtype=torch.float64,
        )
        true_labels = torch.tensor([0, 1, 2, 3])
        predictions = probabilities.argmax(dim=1)
        rows = []
        for index in range(4):
            rows.append(
                {
                    "dialogue_id": "d1",
                    "utterance_id": f"u{index}",
                    "speaker_id": index % 2,
                    "true_label": int(true_labels[index]),
                    "predicted_label": int(predictions[index]),
                    "probabilities": probabilities[index].tolist(),
                    "pred_confidence": float(probabilities[index].max()),
                    "true_label_probability": float(
                        probabilities[index, true_labels[index]]
                    ),
                    "correct": bool(predictions[index] == true_labels[index]),
                }
            )
        return {
            "predictions": rows,
            "diagnostic_metrics": compute_probability_metrics(
                probabilities.log(), true_labels, num_classes=6, ece_bins=15
            ),
            "diagnostic_probabilities": probabilities,
            "diagnostic_labels": true_labels,
            "confusion_matrix": torch.eye(6, dtype=torch.int64),
        }

    monkeypatch.setattr(artifact_module, "strict_reload_checkpoint", fake_reload)
    monkeypatch.setattr(artifact_module, "evaluate_model", fake_evaluate)
    artifacts = artifact_module.export_diagnostic_artifacts(
        model=model,
        val_loader=[],
        adapter=None,
        device=torch.device("cpu"),
        label_ids=list(range(6)),
        label_names=LABELS,
        paths=paths,
        expected_checkpoint_identity={
            "registry_key": config["registry_key"],
            "model_config": core.to_mapping(),
            "feature_sha256": feature.feature_sha256,
        },
        tracker=tracker,
        condition="TAV",
        settings=parse_diagnostic_settings(config),
    )
    expected = [
        run_dir / "reports/checkpoint_summary.csv",
        run_dir / "reports/per_class_epoch_metrics.csv",
        run_dir / "reports/calibration_summary.csv",
        run_dir / "reports/reliability_bins.csv",
        run_dir / "reports/prediction_flip_details.csv",
        run_dir / "reports/prediction_flip_summary.csv",
        run_dir / "reports/ww_high_confidence_summary.csv",
        run_dir / "reports/parameter_count.csv",
        run_dir / "predictions/diagnostic/val_predictions_min_val_loss.csv",
        run_dir / "predictions/diagnostic/val_predictions_best_val_wf1.csv",
        run_dir / "predictions/diagnostic/val_predictions_best_val_uar.csv",
        run_dir / "predictions/diagnostic/val_predictions_final.csv",
    ]
    assert all(path.is_file() for path in expected if "per_class_epoch" not in path.name)
    assert all(path.as_posix() in artifacts for path in expected if "per_class_epoch" not in path.name)
    assert calls == [
        ("min_val_loss_model.pt", False),
        ("best_val_wf1_model.pt", True),
        ("best_val_uar_model.pt", False),
        ("final_model.pt", False),
    ]


@pytest.mark.parametrize(
    "section,error",
    [
        ({"enabled": False, "save_parameter_count": True}, "requires"),
        ({"enabled": True, "ece_bins": 1}, ">= 2"),
        ({"enabled": True, "unknown": True}, "unknown"),
    ],
)
def test_invalid_diagnostic_schema_fails_closed(section: dict, error: str) -> None:
    config = deepcopy(_load(BASE_DIR / "cl7_unilstm_tav.yaml"))
    config["diagnostics"] = {"overfitting_generalization": section}
    with pytest.raises((TypeError, ValueError), match=error):
        parse_diagnostic_settings(config)

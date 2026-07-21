from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

import pandas as pd
import torch
import yaml

from scripts.analyze.analyze_original_merc_results import (
    CLEAN_FIVEFOLD_TRACK,
    LEGACY_FIVEFOLD_TRACK,
    LEGACY_OFFICIAL_TRACK,
    aggregate,
    analyze,
    audit_run_numeric_validity,
    write_top2_selection,
)


MODELS = [
    "original_repro_mmgcn",
    "original_repro_multidag_cl",
    "project_paper_oriented_gsmcc",
    "original_repro_dialoguegcn",
]


def _validation_rows(tmp_path: Path) -> pd.DataFrame:
    rows = []
    for index, model in enumerate(MODELS):
        run_dir = tmp_path / model
        (run_dir / "logs").mkdir(parents=True)
        pd.DataFrame(
            {
                "epoch": [1, 2, 3],
                "train_loss": [1.0, 0.8, 0.7],
                "val_loss": [1.1, 0.9, 0.85],
                "val_weighted_f1": [0.3, 0.4, 0.45],
            }
        ).to_csv(run_dir / "logs" / "epoch_metrics.csv", index=False)
        rows.append(
            {
                "profile": "clean_screening",
                "experiment_track": CLEAN_FIVEFOLD_TRACK,
                "model_name": model,
                "weighted_f1": 0.80 - index * 0.05,
                "run_dir": str(run_dir),
                # A deliberately reversed Test-like column must be ignored.
                "test_weighted_f1": index,
            }
        )
    return pd.DataFrame(rows)


def test_top2_requires_all_clean_candidates_and_ignores_test_metrics() -> None:
    test_root = Path("tmp") / f"pytest_original_analysis_{uuid4().hex}"
    output_dir = test_root / "results"
    output_dir.mkdir(parents=True)
    rows = _validation_rows(test_root)

    write_top2_selection(output_dir, rows.iloc[:3])
    pending = yaml.safe_load((output_dir / "top2_selection.yaml").read_text(encoding="utf-8"))
    assert pending["status"] == "pending_all_four_clean_screening_results"
    assert pending["selected_models"] == []

    ranking = write_top2_selection(output_dir, rows)
    ready = yaml.safe_load((output_dir / "top2_selection.yaml").read_text(encoding="utf-8"))
    assert ready["status"] == "ready"
    assert ready["test_split_used_for_selection"] is False
    assert ready["selected_models"] == MODELS[:2]
    assert ranking.iloc[0]["model_name"] == MODELS[0]
    assert "test_weighted_f1" not in ranking.columns


def test_aggregate_keeps_protocol_tracks_separate() -> None:
    rows = []
    for track, score in (
        (LEGACY_OFFICIAL_TRACK, 0.61),
        (LEGACY_FIVEFOLD_TRACK, 0.62),
        (CLEAN_FIVEFOLD_TRACK, 0.70),
    ):
        rows.append(
            {
                "model_name": "original_repro_mmgcn",
                "experiment_track": track,
                "weighted_f1": score,
                "macro_f1": score,
                "uar": score,
                "accuracy": score,
                "paper_weighted_f1_points": 66.22,
                "paper_gap_weighted_f1_points": 0.0 if track == LEGACY_OFFICIAL_TRACK else float("nan"),
            }
        )
    summary = aggregate(pd.DataFrame(rows))
    assert set(summary["experiment_track"]) == {
        LEGACY_OFFICIAL_TRACK,
        LEGACY_FIVEFOLD_TRACK,
        CLEAN_FIVEFOLD_TRACK,
    }
    assert len(summary) == 3


def _write_numeric_audit_run(
    root: Path,
    *,
    run_name: str,
    model_name: str = "project_paper_oriented_gsmcc",
    profile: str = "clean_screening",
    empty_train_loss: bool = False,
    nan_probability: bool = False,
) -> Path:
    run_dir = root / run_name
    logs = run_dir / "logs"
    evaluations = logs / "evaluations"
    (run_dir / "checkpoints").mkdir(parents=True)
    config = {
        "profile": profile,
        "protocol_version": "original_merc_three_track_v2",
        "model": {
            "name": model_name,
            "text_feature_dim": 768,
            "fidelity_status": "PROJECT_VARIANT_NOT_PAPER_REPRODUCTION",
            "causal_grade": "noncausal_offline_full_context",
        },
        "dataset": {
            "experiment_track": CLEAN_FIVEFOLD_TRACK,
            "protocol_comparability": "fair_comparison_not_paper_reproduction",
        },
        "system": {"seed": 42},
    }
    logs.mkdir(parents=True)
    (logs / "experiment_config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    pd.DataFrame(
        {
            "epoch": [1],
            "train_loss": ["" if empty_train_loss else 1.0],
            "train_classification_loss": [0.8],
            "val_loss": [0.9],
            "val_weighted_f1": [0.5],
            "artifact_run_id": [run_name],
        }
    ).to_csv(logs / "epoch_metrics.csv", index=False)
    for split in ("val", "test"):
        destination = evaluations / f"{split}_best_model"
        destination.mkdir(parents=True)
        pd.DataFrame(
            [
                {
                    "split": split,
                    "model_name": model_name,
                    "feature_set_name": "clean",
                    "feature_protocol": "clean_roberta_v1",
                    "feature_cleanliness": "clean",
                    "usage": "fair_main_experiment",
                    "outer_test_session": "Ses05",
                    "checkpoint": "checkpoints/best_model.pt",
                    "checkpoint_epoch": 1,
                    "loss": 0.9,
                    "accuracy": 0.5,
                    "weighted_f1": 0.5,
                    "macro_f1": 0.5,
                    "uar": 0.5,
                }
            ]
        ).to_csv(destination / "metrics.csv", index=False)
        pd.DataFrame(
            [
                {
                    "confidence": 0.7,
                    "probability_0": float("nan") if nan_probability else 0.7,
                    "probability_1": 0.3,
                }
            ]
        ).to_csv(destination / "predictions.csv", index=False)
        if split == "test":
            pd.DataFrame(
                [
                    {
                        "label_id": 0,
                        "label": run_name,
                        "precision": 0.5,
                        "recall": 0.5,
                        "f1": 0.5,
                        "support": 1,
                    }
                ]
            ).to_csv(destination / "per_class_metrics.csv", index=False)
            pd.DataFrame([[1, 0], [0, 1]]).to_csv(
                destination / "confusion_matrix.csv", index=False
            )
    torch.save(
        {"model_state_dict": {"weight": torch.ones(2)}},
        run_dir / "checkpoints" / "best_model.pt",
    )
    (logs / "run_summary.json").write_text(
        json.dumps(
            {
                # Legacy pipelines could claim PASS without numerical fields.
                "run_status": "PASS",
                "checkpoint_reload": "passed",
            }
        ),
        encoding="utf-8",
    )
    return run_dir


def _write_valid_clean_screening_candidates(
    runs_root: Path,
    *,
    excluded_models: set[str] | None = None,
) -> None:
    excluded_models = excluded_models or set()
    for index, model_name in enumerate(MODELS):
        if model_name in excluded_models:
            continue
        _write_numeric_audit_run(
            runs_root,
            run_name=f"valid_{index}",
            model_name=model_name,
        )


def _analyze_test_runs(runs_root: Path, output_dir: Path) -> dict[str, object]:
    return analyze(
        runs_root,
        output_dir,
        Path("docs/baselines/original_repro/paper_targets.csv"),
    )


def test_empty_required_loss_and_nan_probability_are_invalid() -> None:
    test_root = Path("tmp") / f"pytest_original_invalid_audit_{uuid4().hex}"
    empty_loss = _write_numeric_audit_run(
        test_root, run_name="empty_loss", empty_train_loss=True
    )
    nan_probability = _write_numeric_audit_run(
        test_root, run_name="nan_probability", nan_probability=True
    )
    empty_audit = audit_run_numeric_validity(empty_loss)
    probability_audit = audit_run_numeric_validity(nan_probability)
    assert empty_audit["run_status"] == "NUMERICALLY_INVALID"
    assert "train_loss" in empty_audit["invalid_reasons"]
    assert probability_audit["run_status"] == "NUMERICALLY_INVALID"
    assert "probability_0" in probability_audit["invalid_reasons"]


def test_artifact_audit_uses_logs_evaluations_not_legacy_evaluation_path() -> None:
    test_root = Path("tmp") / f"pytest_original_artifact_path_{uuid4().hex}"
    run_dir = _write_numeric_audit_run(test_root, run_name="finite")
    wrong = run_dir / "evaluation" / "test_best_model"
    wrong.mkdir(parents=True)
    pd.DataFrame([{"loss": float("nan")}]).to_csv(wrong / "metrics.csv", index=False)
    audit = audit_run_numeric_validity(run_dir)
    assert audit["run_status"] == "PASS"
    assert audit["numeric_status"] == "FINITE"


def test_invalid_screening_candidate_blocks_top2() -> None:
    test_root = Path("tmp") / f"pytest_original_invalid_top2_{uuid4().hex}"
    output_dir = test_root / "results"
    output_dir.mkdir(parents=True)
    rows = _validation_rows(test_root)
    valid_rows = rows[rows["model_name"] != "project_paper_oriented_gsmcc"]
    ranking = write_top2_selection(
        output_dir,
        valid_rows,
        invalid_screening_models={"project_paper_oriented_gsmcc"},
    )
    selection = yaml.safe_load(
        (output_dir / "top2_selection.yaml").read_text(encoding="utf-8")
    )
    assert selection["status"] == "pending_invalid_run_repair"
    assert selection["selected_models"] == []
    assert "project_paper_oriented_gsmcc" not in set(ranking["model_name"])


def test_invalid_run_is_excluded_from_aggregate_outputs() -> None:
    test_root = Path("tmp") / f"pytest_original_invalid_aggregate_{uuid4().hex}"
    runs_root = test_root / "runs"
    _write_numeric_audit_run(runs_root, run_name="finite")
    _write_numeric_audit_run(
        runs_root,
        run_name="invalid",
        nan_probability=True,
    )
    output_dir = test_root / "analysis"
    result = analyze(
        runs_root,
        output_dir,
        Path("docs/baselines/original_repro/paper_targets.csv"),
    )
    invalid = pd.read_csv(output_dir / "invalid_runs.csv")
    detailed = pd.read_csv(output_dir / "runs_detailed.csv")
    aggregate_rows = pd.read_csv(output_dir / "aggregate_metrics.csv")
    assert result["invalid_runs"] == 1
    assert set(invalid["run_id"]) == {"invalid"}
    assert set(detailed["run_dir"]) == {str(runs_root / "finite")}
    assert aggregate_rows["runs"].tolist() == [1]


def test_unresolved_invalid_clean_screening_run_blocks_top2() -> None:
    test_root = Path("tmp") / f"pytest_original_unresolved_top2_{uuid4().hex}"
    runs_root = test_root / "runs"
    _write_valid_clean_screening_candidates(
        runs_root,
        excluded_models={"project_paper_oriented_gsmcc"},
    )
    _write_numeric_audit_run(
        runs_root,
        run_name="gsmcc_invalid",
        nan_probability=True,
    )

    output_dir = test_root / "analysis"
    _analyze_test_runs(runs_root, output_dir)

    selection = yaml.safe_load(
        (output_dir / "top2_selection.yaml").read_text(encoding="utf-8")
    )
    invalid = pd.read_csv(output_dir / "invalid_runs.csv")
    resolution = pd.read_csv(output_dir / "invalid_run_resolution.csv")
    assert selection["status"] == "pending_invalid_run_repair"
    assert selection["invalid_screening_models"] == [
        "project_paper_oriented_gsmcc"
    ]
    assert set(invalid["run_id"]) == {"gsmcc_invalid"}
    assert not bool(resolution.iloc[0]["valid_replacement_exists"])
    assert bool(resolution.iloc[0]["blocks_current_selection"])


def test_valid_replacement_resolves_historical_invalid_clean_screening_run() -> None:
    test_root = Path("tmp") / f"pytest_original_resolved_top2_{uuid4().hex}"
    runs_root = test_root / "runs"
    _write_valid_clean_screening_candidates(runs_root)
    _write_numeric_audit_run(
        runs_root,
        run_name="gsmcc_historical_invalid",
        nan_probability=True,
    )

    output_dir = test_root / "analysis"
    _analyze_test_runs(runs_root, output_dir)

    selection = yaml.safe_load(
        (output_dir / "top2_selection.yaml").read_text(encoding="utf-8")
    )
    invalid = pd.read_csv(output_dir / "invalid_runs.csv")
    resolution = pd.read_csv(output_dir / "invalid_run_resolution.csv")
    historical = resolution[resolution["run_id"] == "gsmcc_historical_invalid"].iloc[0]
    assert selection["status"] == "ready"
    assert selection["invalid_screening_models"] == []
    assert set(invalid["run_id"]) == {"gsmcc_historical_invalid"}
    assert bool(historical["valid_replacement_exists"])
    assert not bool(historical["blocks_current_selection"])


def test_repeated_invalid_clean_screening_rerun_still_blocks_top2() -> None:
    test_root = Path("tmp") / f"pytest_original_reinvalid_top2_{uuid4().hex}"
    runs_root = test_root / "runs"
    _write_valid_clean_screening_candidates(
        runs_root,
        excluded_models={"project_paper_oriented_gsmcc"},
    )
    for run_name in ("gsmcc_historical_invalid", "gsmcc_rerun_invalid"):
        _write_numeric_audit_run(
            runs_root,
            run_name=run_name,
            nan_probability=True,
        )

    output_dir = test_root / "analysis"
    _analyze_test_runs(runs_root, output_dir)

    selection = yaml.safe_load(
        (output_dir / "top2_selection.yaml").read_text(encoding="utf-8")
    )
    resolution = pd.read_csv(output_dir / "invalid_run_resolution.csv")
    assert selection["status"] == "pending_invalid_run_repair"
    assert selection["invalid_screening_models"] == [
        "project_paper_oriented_gsmcc"
    ]
    assert not resolution["valid_replacement_exists"].any()
    assert resolution["blocks_current_selection"].all()


def test_valid_run_for_another_model_does_not_resolve_invalid_model() -> None:
    test_root = Path("tmp") / f"pytest_original_cross_model_top2_{uuid4().hex}"
    runs_root = test_root / "runs"
    invalid_model = "original_repro_mmgcn"
    _write_valid_clean_screening_candidates(
        runs_root,
        excluded_models={invalid_model},
    )
    _write_numeric_audit_run(
        runs_root,
        run_name="mmgcn_invalid",
        model_name=invalid_model,
        nan_probability=True,
    )

    output_dir = test_root / "analysis"
    _analyze_test_runs(runs_root, output_dir)

    selection = yaml.safe_load(
        (output_dir / "top2_selection.yaml").read_text(encoding="utf-8")
    )
    resolution = pd.read_csv(output_dir / "invalid_run_resolution.csv")
    assert selection["status"] == "pending_invalid_run_repair"
    assert selection["invalid_screening_models"] == [invalid_model]
    assert not bool(resolution.iloc[0]["valid_replacement_exists"])
    assert bool(resolution.iloc[0]["blocks_current_selection"])


def test_extended_artifacts_are_written_only_for_valid_runs() -> None:
    test_root = Path("tmp") / f"pytest_original_extended_filter_{uuid4().hex}"
    runs_root = test_root / "runs"
    _write_numeric_audit_run(runs_root, run_name="valid")
    _write_numeric_audit_run(
        runs_root,
        run_name="invalid",
        nan_probability=True,
    )
    output_dir = test_root / "analysis"
    (output_dir / "confusion_matrices").mkdir(parents=True)
    (output_dir / "training_curves").mkdir(parents=True)
    pd.DataFrame([[99]]).to_csv(
        output_dir / "confusion_matrices" / "stale_invalid.csv", index=False
    )
    pd.DataFrame([[99]]).to_csv(
        output_dir / "training_curves" / "stale_invalid.csv", index=False
    )

    _analyze_test_runs(runs_root, output_dir)

    per_class = pd.read_csv(output_dir / "per_class_metrics.csv")
    invalid_audit = pd.read_csv(output_dir / "invalid_runs.csv")
    resolution = pd.read_csv(output_dir / "invalid_run_resolution.csv")
    assert set(per_class["run_id"]) == {"valid"}
    assert (output_dir / "confusion_matrices" / "valid.csv").is_file()
    assert not (output_dir / "confusion_matrices" / "invalid.csv").exists()
    assert not (output_dir / "confusion_matrices" / "stale_invalid.csv").exists()
    assert (output_dir / "training_curves" / "valid.csv").is_file()
    assert not (output_dir / "training_curves" / "invalid.csv").exists()
    assert not (output_dir / "training_curves" / "stale_invalid.csv").exists()
    assert set(invalid_audit["run_id"]) == {"invalid"}
    assert set(resolution["run_id"]) == {"invalid"}

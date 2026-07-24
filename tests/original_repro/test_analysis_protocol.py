from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

import pandas as pd
import torch
import yaml

from scripts.analyze.analyze_original_merc_results import (
    CLEAN_FIVEFOLD_TRACK,
    FORMAL_ORIGINAL_MERC,
    LEGACY_FIVEFOLD_TRACK,
    LEGACY_OFFICIAL_TRACK,
    SMOKE_ORIGINAL_MERC,
    aggregate,
    analyze,
    audit_run_numeric_validity,
    classify_run_scope,
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
    protocol_version: str = "original_merc_three_track_v2",
    config_source: str = (
        "configs/gsmcc/project_variant/iemocap/full_context/clean_roberta_features/gsmcc_clean.yaml"
    ),
    experiment_track: str = CLEAN_FIVEFOLD_TRACK,
    text_feature_dim: int = 768,
    val_weighted_f1: float = 0.5,
    test_weighted_f1: float = 0.5,
    epochs: int = 3,
    max_train_batches: int | None = None,
    max_eval_batches: int | None = None,
    empty_train_loss: bool = False,
    nan_probability: bool = False,
) -> Path:
    run_dir = root / run_name
    logs = run_dir / "logs"
    evaluations = logs / "evaluations"
    (run_dir / "checkpoints").mkdir(parents=True)
    config = {
        "profile": profile,
        "protocol_version": protocol_version,
        "model": {
            "name": model_name,
            "text_feature_dim": text_feature_dim,
            "fidelity_status": "PROJECT_VARIANT_NOT_PAPER_REPRODUCTION",
            "causal_grade": "noncausal_offline_full_context",
        },
        "dataset": {
            "experiment_track": experiment_track,
            "protocol_comparability": "fair_comparison_not_paper_reproduction",
        },
        "training": {
            "epochs": epochs,
            "max_train_batches": max_train_batches,
            "max_eval_batches": max_eval_batches,
        },
        "system": {"seed": 42},
    }
    logs.mkdir(parents=True)
    (logs / "experiment_config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    (run_dir / "run_metadata.json").write_text(
        json.dumps(
            {
                "protocol_version": protocol_version,
                "config_path": config_source,
            }
        ),
        encoding="utf-8",
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
                    "weighted_f1": (
                        val_weighted_f1 if split == "val" else test_weighted_f1
                    ),
                    "macro_f1": (
                        val_weighted_f1 if split == "val" else test_weighted_f1
                    ),
                    "uar": (
                        val_weighted_f1 if split == "val" else test_weighted_f1
                    ),
                },
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
                "numeric_status": "FINITE",
                "checkpoint_reload": "passed",
                "checkpoint_parameters_finite": True,
                "final_metrics_finite": True,
                "prediction_count_correct": True,
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


def test_generated_formal_fold_config_is_in_formal_scope() -> None:
    test_root = Path("tmp") / f"pytest_generated_formal_scope_{uuid4().hex}"
    run_dir = _write_numeric_audit_run(
        test_root,
        run_name="mmgcn_clean_seed13_ses01",
        profile="formal_clean_fold_base",
        config_source=(
            "tmp/original_merc_pipeline_configs/"
            "mmgcn_clean_seed13_ses01.yaml"
        ),
    )
    assert classify_run_scope(run_dir)["run_scope"] == FORMAL_ORIGINAL_MERC


def test_canonical_smoke_config_is_in_smoke_scope(tmp_path: Path) -> None:
    test_root = tmp_path / "canonical_smoke_scope"
    run_dir = _write_numeric_audit_run(
        test_root,
        run_name="mmgcn_legacy_config_path_only",
        profile="unexpected_profile",
        config_source=(
            "configs/mmgcn/paper_aligned/iemocap/full_context/"
            "legacy_mmgcn_features/smoke.yaml"
        ),
    )
    scope = classify_run_scope(run_dir)
    assert scope["run_scope"] == SMOKE_ORIGINAL_MERC
    assert scope["scope_reason"] == "smoke_config_source"


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


def test_analysis_isolates_formal_smoke_and_unrelated_history() -> None:
    test_root = Path("tmp") / f"pytest_original_scope_{uuid4().hex}"
    runs_root = test_root / "runs"
    output_dir = test_root / "analysis"
    clean_test_scores = {
        "original_repro_multidag_cl": 0.6000423456369230,
        "original_repro_mmgcn": 0.5321450366878487,
        "project_paper_oriented_gsmcc": 0.5014781805884865,
        "original_repro_dialoguegcn": 0.4995813923273042,
    }
    legacy_test_scores = {
        "original_repro_mmgcn": 0.6597636429358654,
        "original_repro_multidag_cl": 0.6046130292848565,
        "project_paper_oriented_gsmcc": 0.5782917443889111,
        "original_repro_dialoguegcn": 0.5736613720698336,
    }
    clean_val_scores = {
        "original_repro_multidag_cl": 0.5733994094479845,
        "project_paper_oriented_gsmcc": 0.5021340714075007,
        "original_repro_dialoguegcn": 0.4587602094680373,
        "original_repro_mmgcn": 0.4526791544287256,
    }
    config_stems = {
        "original_repro_mmgcn": "mmgcn",
        "original_repro_multidag_cl": "multidag_cl",
        "project_paper_oriented_gsmcc": "gsmcc",
        "original_repro_dialoguegcn": "dialoguegcn",
    }

    formal_run_ids = set()
    for model_name, test_score in clean_test_scores.items():
        run_name = f"formal_clean_{config_stems[model_name]}"
        formal_run_ids.add(run_name)
        _write_numeric_audit_run(
            runs_root,
            run_name=run_name,
            model_name=model_name,
            profile="clean_screening",
            config_source=(
                "configs/experiments/original_merc/clean_screening/"
                f"{config_stems[model_name]}_clean.yaml"
            ),
            experiment_track=CLEAN_FIVEFOLD_TRACK,
            text_feature_dim=768,
            val_weighted_f1=clean_val_scores[model_name],
            test_weighted_f1=test_score,
        )
    for model_name, test_score in legacy_test_scores.items():
        run_name = f"formal_legacy_{config_stems[model_name]}"
        formal_run_ids.add(run_name)
        _write_numeric_audit_run(
            runs_root,
            run_name=run_name,
            model_name=model_name,
            profile="screening",
            config_source=(
                "configs/experiments/original_merc/screening/"
                f"{config_stems[model_name]}_legacy.yaml"
            ),
            experiment_track=LEGACY_OFFICIAL_TRACK,
            text_feature_dim=100,
            val_weighted_f1=0.4,
            test_weighted_f1=test_score,
        )

    historical_invalid_ids = {
        "original_gsmcc_legacy_screening_20260716_151823_9ab9c9",
        "project_gsmcc_clean_screening_20260716_175031_53d594",
    }
    _write_numeric_audit_run(
        runs_root,
        run_name="original_gsmcc_legacy_screening_20260716_151823_9ab9c9",
        profile="screening",
        config_source="configs/gsmcc/project_variant/iemocap/full_context/legacy_mmgcn_features/screening.yaml",
        experiment_track=LEGACY_OFFICIAL_TRACK,
        text_feature_dim=100,
        nan_probability=True,
    )
    _write_numeric_audit_run(
        runs_root,
        run_name="project_gsmcc_clean_screening_20260716_175031_53d594",
        profile="clean_screening",
        config_source="configs/gsmcc/project_variant/iemocap/full_context/clean_roberta_features/gsmcc_clean.yaml",
        experiment_track=CLEAN_FIVEFOLD_TRACK,
        nan_probability=True,
    )

    smoke_ids = {
        "original_gsmcc_legacy_smoke_20260721_130424_628ebb",
        "original_gsmcc_clean_smoke_20260721_130452_df46ec",
    }
    gsmcc_smoke_configs = {
        "clean": (
            "configs/gsmcc/project_variant/iemocap/full_context/"
            "clean_roberta_features/smoke.yaml"
        ),
        "legacy": (
            "configs/gsmcc/project_variant/iemocap/full_context/"
            "legacy_mmgcn_features/smoke.yaml"
        ),
    }
    for run_name, text_dim in zip(sorted(smoke_ids), (768, 100)):
        feature_name = "clean" if text_dim == 768 else "legacy"
        _write_numeric_audit_run(
            runs_root,
            run_name=run_name,
            profile="smoke",
            config_source=gsmcc_smoke_configs[feature_name],
            experiment_track=(
                CLEAN_FIVEFOLD_TRACK
                if text_dim == 768
                else LEGACY_OFFICIAL_TRACK
            ),
            text_feature_dim=text_dim,
            val_weighted_f1=0.999,
            test_weighted_f1=0.999,
            epochs=2,
            max_train_batches=1,
            max_eval_batches=1,
        )

    out_of_scope_ids = {
        "causal_history",
        "old_pipeline_history",
        "unproven_history",
    }
    _write_numeric_audit_run(
        runs_root,
        run_name="causal_history",
        protocol_version="causal_baseline_v1",
        config_source="configs/mmgcn/unified/m3ed/causal_context/m3ed_features/skeleton.yaml",
        nan_probability=True,
    )
    _write_numeric_audit_run(
        runs_root,
        run_name="old_pipeline_history",
        config_source="configs/pipeline/legacy_original_pipeline.yaml",
        nan_probability=True,
    )
    (runs_root / "unproven_history").mkdir(parents=True)

    result = _analyze_test_runs(runs_root, output_dir)

    assert result["collected_runs"] == 8
    assert result["invalid_runs"] == 2
    assert result["smoke_runs"] == 2
    assert result["excluded_runs"] == 3

    invalid = pd.read_csv(output_dir / "invalid_runs.csv")
    resolution = pd.read_csv(output_dir / "invalid_run_resolution.csv")
    smoke = pd.read_csv(output_dir / "smoke_runs.csv")
    excluded = pd.read_csv(output_dir / "excluded_runs.csv")
    assert set(invalid["run_id"]) == historical_invalid_ids
    assert set(resolution["run_id"]) == historical_invalid_ids
    assert resolution["valid_replacement_exists"].all()
    assert not resolution["blocks_current_selection"].any()
    assert set(smoke["run_id"]) == smoke_ids
    assert set(smoke["feature_track"]) == {"clean", "legacy"}
    assert set(smoke["numeric_status"]) == {"FINITE"}
    assert set(smoke["run_status"]) == {"PASS"}
    assert smoke["checkpoint_parameters_finite"].all()
    assert smoke["prediction_count_correct"].all()
    assert set(excluded["run_id"]) == out_of_scope_ids
    assert not set(invalid["run_id"]) & out_of_scope_ids

    aggregate_rows = pd.read_csv(output_dir / "aggregate_metrics.csv")
    assert aggregate_rows["runs"].eq(1).all()
    assert aggregate_rows["weighted_f1_std"].eq(0.0).all()
    for track, expected_scores in (
        (CLEAN_FIVEFOLD_TRACK, clean_test_scores),
        (LEGACY_OFFICIAL_TRACK, legacy_test_scores),
    ):
        track_rows = aggregate_rows[
            aggregate_rows["experiment_track"] == track
        ].set_index("model_name")
        assert set(track_rows.index) == set(expected_scores)
        for model_name, expected_score in expected_scores.items():
            assert track_rows.loc[model_name, "weighted_f1_mean"] == expected_score

    ranking = pd.read_csv(output_dir / "protocol_separated_ranking.csv")
    clean_order = ranking[
        ranking["experiment_track"] == CLEAN_FIVEFOLD_TRACK
    ].sort_values("rank_within_experiment_track")["model_name"].tolist()
    legacy_order = ranking[
        ranking["experiment_track"] == LEGACY_OFFICIAL_TRACK
    ].sort_values("rank_within_experiment_track")["model_name"].tolist()
    assert clean_order == list(clean_test_scores)
    assert legacy_order == list(legacy_test_scores)

    selection = yaml.safe_load(
        (output_dir / "top2_selection.yaml").read_text(encoding="utf-8")
    )
    assert selection["status"] == "ready"
    assert selection["invalid_screening_models"] == []
    assert selection["selected_models"] == [
        "original_repro_multidag_cl",
        "project_paper_oriented_gsmcc",
    ]
    assert selection["test_split_used_for_selection"] is False
    selection_evidence = pd.read_csv(
        output_dir / "clean_screening_validation_selection_evidence.csv"
    ).sort_values("rank")
    assert selection_evidence["model_name"].tolist() == list(clean_val_scores)
    for row in selection_evidence.itertuples(index=False):
        assert abs(
            row.clean_validation_weighted_f1_mean
            - clean_val_scores[row.model_name]
        ) < 1.0e-12

    formal_csvs = [
        "runs_detailed.csv",
        "run_manifest.csv",
        "fold_metrics.csv",
        "aggregate_metrics.csv",
        "protocol_separated_ranking.csv",
        "training_stability.csv",
        "baseline_selection_matrix.csv",
        "paper_gap.csv",
        "per_class_metrics.csv",
        "runtime_memory.csv",
        "model_complexity.csv",
        "run_protocol_audit.csv",
        "clean_screening_validation_selection_evidence.csv",
    ]
    for filename in formal_csvs:
        assert "_smoke_" not in (output_dir / filename).read_text(encoding="utf-8")
    assert {
        path.stem for path in (output_dir / "confusion_matrices").glob("*.csv")
    } == formal_run_ids
    assert {
        path.stem for path in (output_dir / "training_curves").glob("*.csv")
    } == formal_run_ids

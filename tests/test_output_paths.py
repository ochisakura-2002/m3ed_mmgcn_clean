from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path

import pytest
import yaml

from utils.output_paths import (
    configured_output_root,
    create_unique_run_dir,
    discover_analysis_directories,
    discover_launcher_log_directories,
    discover_run_directories,
    infer_experiment_date_from_run,
    infer_experiment_group_from_run,
    is_experiment_date,
    resolve_day_output_root,
    resolve_experiment_date,
    resolve_experiment_group,
    resolve_output_paths,
    resolve_output_category,
    validate_experiment_date,
)


def test_default_uses_machine_local_date_in_compact_format() -> None:
    assert resolve_experiment_date(now=datetime(2026, 7, 16, 23, 59)) == "20260716"


def test_date_priority_cli_then_config_then_environment() -> None:
    config = {"output": {"experiment_date": "20260715"}}
    env = {"MERC_EXPERIMENT_DATE": "20260714"}
    assert resolve_experiment_date("20260716", config, env) == "20260716"
    assert resolve_experiment_date(None, config, env) == "20260715"
    assert resolve_experiment_date(None, {"output": {}}, env) == "20260714"


@pytest.mark.parametrize(
    "value",
    ["20260230", "20261301", "2026-07-16", "abc", "2026071", "202607160"],
)
def test_invalid_dates_are_rejected(value: str) -> None:
    with pytest.raises(ValueError):
        validate_experiment_date(value)


def test_leap_year_and_valid_calendar_dates() -> None:
    assert validate_experiment_date("20240229") == "20240229"
    assert validate_experiment_date("20260716") == "20260716"
    with pytest.raises(ValueError):
        validate_experiment_date("20230229")


def test_dated_root_and_category_paths() -> None:
    assert resolve_day_output_root("20260716") == Path("outputs/20260716")
    assert resolve_output_category(
        "runs",
        "20260716",
        experiment_group="formal_long32_primary_seed42",
    ) == Path(
        "outputs/20260716/formal_long32_primary_seed42/runs"
    )


def test_pipeline_date_is_frozen_across_midnight() -> None:
    calls = iter(
        [datetime(2026, 7, 16, 23, 59, 59), datetime(2026, 7, 17, 0, 0, 1)]
    )
    frozen = resolve_experiment_date(now=lambda: next(calls))
    assert resolve_day_output_root(frozen) == Path("outputs/20260716")
    assert resolve_output_category(
        "runs",
        frozen,
        experiment_group="same_day_group",
    ) == Path(
        "outputs/20260716/same_day_group/runs"
    )


def test_two_runs_on_same_day_do_not_overwrite(tmp_path: Path) -> None:
    suffixes = iter(["a1b2c3", "d4e5f6"])
    first = create_unique_run_dir(
        "same_name",
        "20260716",
        tmp_path,
        experiment_group="repeat_safe",
        now=datetime(2026, 7, 16, 14, 30, 25),
        suffix_factory=lambda: next(suffixes),
    )
    second = create_unique_run_dir(
        "same_name",
        "20260716",
        tmp_path,
        experiment_group="repeat_safe",
        now=datetime(2026, 7, 16, 14, 30, 25),
        suffix_factory=lambda: next(suffixes),
    )
    assert first != second
    assert first.is_dir() and second.is_dir()


def test_resume_keeps_original_run_directory(tmp_path: Path) -> None:
    original = tmp_path / "20260716" / "resume_group" / "runs" / "existing"
    original.mkdir(parents=True)
    resumed = create_unique_run_dir(
        "ignored",
        "20260717",
        tmp_path,
        experiment_group="resume_group",
        resume_run_dir=original,
    )
    assert resumed == original
    assert not (tmp_path / "20260717").exists()


def test_new_and_legacy_discovery_with_static_directories_excluded(
    tmp_path: Path,
) -> None:
    new_run = tmp_path / "20260716" / "new_group" / "runs" / "new-run"
    dated_legacy_run = tmp_path / "20260716" / "runs" / "dated-old-run"
    old_run = tmp_path / "runs" / "old-run"
    long_training_run = tmp_path / "long_training" / "primary" / "long-old-run"
    new_launcher_logs = (
        tmp_path
        / "20260716"
        / "new_group"
        / "logs"
        / "launcher"
        / "new-batch"
    )
    old_launcher_logs = tmp_path / "launcher_logs" / "old-batch"
    new_analysis = (
        tmp_path / "20260716" / "new_group" / "analysis" / "new-analysis"
    )
    old_analysis = tmp_path / "analysis" / "old-analysis"
    for path in (
        new_run,
        dated_legacy_run,
        old_run,
        long_training_run,
        new_launcher_logs,
        old_launcher_logs,
        new_analysis,
        old_analysis,
    ):
        path.mkdir(parents=True)
    for name in ("environment", "reference", "cache", "20260230"):
        (tmp_path / name / "runs" / "not-a-run").mkdir(parents=True)

    assert discover_run_directories(tmp_path) == [
        new_run,
        dated_legacy_run,
        old_run,
        long_training_run,
    ]
    assert discover_analysis_directories(tmp_path) == [new_analysis, old_analysis]
    assert discover_launcher_log_directories(tmp_path) == [
        new_launcher_logs,
        old_launcher_logs,
    ]
    assert not is_experiment_date("environment")
    assert not is_experiment_date("20260230")


def test_infer_date_prefers_metadata_then_new_layout(tmp_path: Path) -> None:
    run_dir = tmp_path / "20260716" / "metadata_group" / "runs" / "run-a"
    run_dir.mkdir(parents=True)
    assert infer_experiment_date_from_run(run_dir) == "20260716"
    assert infer_experiment_group_from_run(run_dir) == "metadata_group"
    (run_dir / "run_metadata.json").write_text(
        json.dumps({"experiment_date": "20260715"}), encoding="utf-8"
    )
    assert infer_experiment_date_from_run(run_dir) == "20260715"


def test_common_run_metadata_contains_resolved_date_paths(tmp_path: Path) -> None:
    from utils.io import prepare_run_environment

    config = {
        "project": {"experiment_name": "metadata-check"},
        "system": {"output_dir": str(tmp_path), "seed": 42},
        "output": {
            "root": str(tmp_path),
            "experiment_date": None,
            "experiment_group": "metadata_check",
        },
        "dataset": {"name": "M3ED"},
        "model": {"name": "SimpleMLP"},
    }
    run = prepare_run_environment(config, experiment_date="20260716")
    metadata = json.loads(
        (run["run_dir"] / "run_metadata.json").read_text(encoding="utf-8")
    )
    for key in (
        "experiment_date",
        "experiment_group",
        "experiment_root",
        "output_root",
        "day_output_root",
        "run_dir",
        "log_dir",
        "analysis_dir",
        "manifest_dir",
    ):
        assert metadata[key]
    assert metadata["experiment_date"] == "20260716"
    assert metadata["experiment_group"] == "metadata_check"
    assert run["run_dir"].parent == (
        tmp_path / "20260716" / "metadata_check" / "runs"
    )
    assert (run["run_dir"] / "resolved_config.yaml").is_file()
    assert (run["run_dir"] / "metrics").is_dir()
    assert (run["run_dir"] / "artifacts").is_dir()


def test_common_resolver_returns_all_functional_roots() -> None:
    paths = resolve_output_paths(
        output_base="outputs",
        experiment_date="20260726",
        experiment_group="formal_long32_primary_seed42",
        run_id="run_a",
        batch_id="batch_a",
    )
    root = Path("outputs/20260726/formal_long32_primary_seed42")
    assert paths.experiment_root == root
    assert paths.run_root == root / "runs" / "run_a"
    assert paths.launcher_log_root == root / "logs" / "launcher" / "batch_a"
    assert paths.manifest_root == root / "manifests" / "batches" / "batch_a"
    assert paths.review_root == root / "review" / "batches" / "batch_a"
    assert paths.report_root == root / "reports" / "batches" / "batch_a"
    assert paths.analysis_root == root / "analysis" / "batches" / "batch_a"


def test_fixed_run_id_collision_fails_without_overwrite(tmp_path: Path) -> None:
    first = create_unique_run_dir(
        "ignored",
        "20260726",
        tmp_path,
        experiment_group="collision_check",
        run_id="fixed_run",
    )
    marker = first / "keep.txt"
    marker.write_text("history", encoding="utf-8")
    with pytest.raises(FileExistsError):
        create_unique_run_dir(
            "ignored",
            "20260726",
            tmp_path,
            experiment_group="collision_check",
            run_id="fixed_run",
        )
    assert marker.read_text(encoding="utf-8") == "history"


def test_config_path_infers_stable_group_without_using_run_id() -> None:
    first = resolve_experiment_group(
        config_path=(
            "configs/mmgcn/unified/iemocap/causal_context/"
            "clean_roberta_features/val_ses01.yaml"
        )
    )
    second = resolve_experiment_group(
        config_path=(
            "configs/mmgcn/unified/iemocap/causal_context/"
            "clean_roberta_features/val_ses02.yaml"
        )
    )
    assert first == second
    assert len(first) <= 36


def test_long_config_family_group_is_stably_bounded() -> None:
    config_path = (
        "configs/gsmcc/project_variant/synthetic/causal_context/"
        "synthetic/smoke_end_to_end.yaml"
    )
    first = resolve_experiment_group(config_path=config_path)
    second = resolve_experiment_group(config_path=config_path)
    assert first == second
    assert first.startswith("gsmcc_project_variant")
    assert len(first) <= 36


def test_all_output_owning_yaml_resolves_to_canonical_outputs_layout() -> None:
    config_paths = sorted(Path("configs").rglob("*.yaml"))
    output_owning: list[Path] = []
    legacy_fragments = (
        "outputs/long_training/primary/",
        "outputs/long_training/multi_seed/",
        "outputs/launcher_logs/",
        "tmp/smoke_outputs",
    )
    for config_path in config_paths:
        text = config_path.read_text(encoding="utf-8")
        config = yaml.safe_load(text) or {}
        assert isinstance(config, dict)
        output = config.get("output", {})
        system = config.get("system", {})
        expansion = config.get("expansion", {})
        output = output if isinstance(output, dict) else {}
        system = system if isinstance(system, dict) else {}
        expansion = expansion if isinstance(expansion, dict) else {}
        owns_output = (
            any(
                key in output
                for key in ("root", "run_root", "output_base", "experiment_group")
            )
            or "output_dir" in system
            or "output_root_template" in expansion
        )
        if not owns_output:
            continue
        output_owning.append(config_path)
        assert not any(fragment in text for fragment in legacy_fragments)
        group = resolve_experiment_group(config=config, config_path=config_path)
        base = configured_output_root(config)
        assert base == Path("outputs")
        layout = resolve_output_paths(
            output_base=base,
            experiment_date="20260726",
            experiment_group=group,
            run_id="layout_audit",
        )
        assert layout.run_root == (
            Path("outputs") / "20260726" / group / "runs" / "layout_audit"
        )

    # Stage B3 adds three canonical MultiDAG-CL paper-reimplementation configs.
    assert len(config_paths) == 198
    assert len(output_owning) == 130


def test_original_pipeline_passes_one_frozen_date_to_every_job(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts.workflows.paper_aligned import run_pipeline as pipeline

    config_path = tmp_path / "train.yaml"
    config_path.write_text("model: {name: placeholder}\n", encoding="utf-8")
    manifest_path = tmp_path / "manifest.yaml"
    manifest_path.write_text(
        yaml.safe_dump(
            {
                "output": {
                    "root": str(tmp_path / "outputs"),
                    "experiment_group": "original_pipeline_test",
                },
                "stages": {
                    "smoke": [
                        {"config": str(config_path)},
                        {"config": str(config_path)},
                    ]
                },
            }
        ),
        encoding="utf-8",
    )
    seen: list[str] = []

    def fake_training(*args, **kwargs):
        seen.append(kwargs["experiment_date"])
        return {"run_id": f"run-{len(seen)}", "run_dir": str(tmp_path / "run")}

    monkeypatch.setattr(pipeline, "run_training", fake_training)
    pipeline.run_pipeline(
        manifest_path,
        "smoke",
        True,
        "cpu",
        tmp_path / "outputs",
        "20260716",
    )
    assert seen == ["20260716", "20260716"]


def test_original_pipeline_discovers_dated_top2_selection_without_legacy_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts.workflows.paper_aligned import run_pipeline as pipeline

    selection_dir = (
        tmp_path / "outputs" / "20260716" / "analysis" / "original_merc"
    )
    selection_dir.mkdir(parents=True)
    (selection_dir / "top2_selection.yaml").write_text(
        yaml.safe_dump(
            {
                "status": "ready",
                "test_split_used_for_selection": False,
                "selection_source": "clean_single_fold_screening_validation_only",
                "ordered_criteria": ["clean_screening_validation_weighted_f1"],
                "jobs": [
                    {"config": "first.yaml"},
                    {"config": "second.yaml"},
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(pipeline, "PROJECT_ROOT", tmp_path)

    jobs = pipeline.expand_jobs(
        {"stages": {"top2": []}, "top2_selection_file": None},
        "top2",
    )

    assert [job["config"] for job in jobs] == ["first.yaml", "second.yaml"]


def test_original_pipeline_marks_numeric_failure_invalid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts.workflows.paper_aligned import run_pipeline as pipeline
    from scripts.runtime.paper_aligned import NumericValidationError

    manifest_path = tmp_path / "pipeline.yaml"
    manifest_path.write_text(
        yaml.safe_dump(
            {
                "output": {
                    "root": str(tmp_path / "outputs"),
                    "experiment_date": None,
                    "experiment_group": "original_pipeline_test",
                },
                "stages": {"smoke": [{"config": "unused.yaml"}]},
            }
        ),
        encoding="utf-8",
    )

    def fail_training(*_args, **_kwargs):
        raise NumericValidationError(
            numeric_status="NONFINITE_GRADIENT",
            model_name="project_paper_oriented_gsmcc",
            epoch=1,
            batch_index=1,
            stage="gradients_after_backward",
            tensor_or_parameter="text_input.weight",
            classification_loss=1.0,
            auxiliary_losses={"contrastive_loss": 2.0},
            total_loss=3.0,
            learning_rate=1e-5,
            amp_enabled=False,
        )

    monkeypatch.setattr(pipeline, "run_training", fail_training)
    with pytest.raises(NumericValidationError):
        pipeline.run_pipeline(
            manifest_path,
            "smoke",
            True,
            "cpu",
            tmp_path / "outputs",
            "20260720",
        )
    status_paths = list(
        (
            tmp_path
            / "outputs"
            / "20260720"
            / "original_pipeline_test"
            / "manifests"
        ).rglob("run_status.json")
    )
    assert len(status_paths) == 1
    status = json.loads(status_paths[0].read_text(encoding="utf-8"))
    assert status["run_status"] == "NUMERICALLY_INVALID"
    assert status["numeric_status"] == "NONFINITE_GRADIENT"
    assert status["exit_code"] == 1

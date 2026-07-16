from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path

import pytest
import yaml

from utils.output_paths import (
    create_unique_run_dir,
    discover_analysis_directories,
    discover_run_directories,
    infer_experiment_date_from_run,
    is_experiment_date,
    resolve_day_output_root,
    resolve_experiment_date,
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
    assert resolve_output_category("runs", "20260716") == Path(
        "outputs/20260716/runs"
    )


def test_pipeline_date_is_frozen_across_midnight() -> None:
    calls = iter(
        [datetime(2026, 7, 16, 23, 59, 59), datetime(2026, 7, 17, 0, 0, 1)]
    )
    frozen = resolve_experiment_date(now=lambda: next(calls))
    assert resolve_day_output_root(frozen) == Path("outputs/20260716")
    assert resolve_output_category("runs", frozen) == Path(
        "outputs/20260716/runs"
    )


def test_two_runs_on_same_day_do_not_overwrite(tmp_path: Path) -> None:
    suffixes = iter(["a1b2c3", "d4e5f6"])
    first = create_unique_run_dir(
        "same_name",
        "20260716",
        tmp_path,
        now=datetime(2026, 7, 16, 14, 30, 25),
        suffix_factory=lambda: next(suffixes),
    )
    second = create_unique_run_dir(
        "same_name",
        "20260716",
        tmp_path,
        now=datetime(2026, 7, 16, 14, 30, 25),
        suffix_factory=lambda: next(suffixes),
    )
    assert first != second
    assert first.is_dir() and second.is_dir()


def test_resume_keeps_original_run_directory(tmp_path: Path) -> None:
    original = tmp_path / "20260716" / "runs" / "existing"
    original.mkdir(parents=True)
    resumed = create_unique_run_dir(
        "ignored",
        "20260717",
        tmp_path,
        resume_run_dir=original,
    )
    assert resumed == original
    assert not (tmp_path / "20260717").exists()


def test_new_and_legacy_discovery_with_static_directories_excluded(
    tmp_path: Path,
) -> None:
    new_run = tmp_path / "20260716" / "runs" / "new-run"
    old_run = tmp_path / "runs" / "old-run"
    new_analysis = tmp_path / "20260716" / "analysis" / "new-analysis"
    old_analysis = tmp_path / "analysis" / "old-analysis"
    for path in (new_run, old_run, new_analysis, old_analysis):
        path.mkdir(parents=True)
    for name in ("environment", "reference", "cache", "20260230"):
        (tmp_path / name / "runs" / "not-a-run").mkdir(parents=True)

    assert discover_run_directories(tmp_path) == [new_run, old_run]
    assert discover_analysis_directories(tmp_path) == [new_analysis, old_analysis]
    assert not is_experiment_date("environment")
    assert not is_experiment_date("20260230")


def test_infer_date_prefers_metadata_then_new_layout(tmp_path: Path) -> None:
    run_dir = tmp_path / "20260716" / "runs" / "run-a"
    run_dir.mkdir(parents=True)
    assert infer_experiment_date_from_run(run_dir) == "20260716"
    (run_dir / "run_metadata.json").write_text(
        json.dumps({"experiment_date": "20260715"}), encoding="utf-8"
    )
    assert infer_experiment_date_from_run(run_dir) == "20260715"


def test_common_run_metadata_contains_resolved_date_paths(tmp_path: Path) -> None:
    from utils.io import prepare_run_environment

    config = {
        "project": {"experiment_name": "metadata-check"},
        "system": {"output_dir": str(tmp_path), "seed": 42},
        "output": {"root": str(tmp_path), "experiment_date": None},
        "dataset": {"name": "M3ED"},
        "model": {"name": "SimpleMLP"},
    }
    run = prepare_run_environment(config, experiment_date="20260716")
    metadata = json.loads(
        (run["run_dir"] / "run_metadata.json").read_text(encoding="utf-8")
    )
    for key in (
        "experiment_date",
        "output_root",
        "day_output_root",
        "run_dir",
        "log_dir",
        "analysis_dir",
        "manifest_dir",
    ):
        assert metadata[key]
    assert metadata["experiment_date"] == "20260716"


def test_original_pipeline_passes_one_frozen_date_to_every_job(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts.baselines import run_original_merc_pipeline as pipeline

    config_path = tmp_path / "train.yaml"
    config_path.write_text("model: {name: placeholder}\n", encoding="utf-8")
    manifest_path = tmp_path / "manifest.yaml"
    manifest_path.write_text(
        yaml.safe_dump(
            {
                "output": {"root": str(tmp_path / "outputs")},
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
    from scripts.baselines import run_original_merc_pipeline as pipeline

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

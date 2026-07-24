from __future__ import annotations

import csv
import os
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import pytest
import yaml

from scripts.experiments.run_causal8_original8_formal import (
    BATCH_PREFIX,
    EXPERIMENT_DATE_ENV,
    MANIFEST_COLUMNS,
    STATUS_COLUMNS,
    BatchPaths,
    EpochCsvTailer,
    LauncherLogger,
    PreparedRun,
    PreflightResult,
    PublicGateError,
    RunSnapshot,
    RunSpec,
    _initial_status,
    _status_from_artifacts,
    _write_batch_state,
    build_run_plan,
    discover_current_run_dir,
    launch_batch,
    monitor_process,
    parse_args,
    prepare_runs,
    snapshot_runs,
)


def _preflight(*_: object, **__: object) -> PreflightResult:
    return PreflightResult(
        git_commit="a" * 40,
        git_state=f"commit={'a' * 40}\nstatus:\n",
        feature_records=(
            {
                "path": "legacy.pkl",
                "expected_sha256": "1" * 64,
                "actual_sha256": "1" * 64,
                "status": "PASS",
            },
            {
                "path": "clean.pkl",
                "expected_sha256": "2" * 64,
                "actual_sha256": "2" * 64,
                "status": "PASS",
            },
        ),
    )


def _result(
    prepared: PreparedRun,
    output_root: Path,
    experiment_date: str,
    *,
    status: str = "PASS",
    include_run: bool = True,
) -> dict[str, object]:
    run_id = f"new_run_{prepared.spec.index:02d}"
    run_dir = output_root / experiment_date / "runs" / run_id
    if include_run:
        run_dir.mkdir(parents=True, exist_ok=True)
    return {
        "index": prepared.spec.index,
        "family": prepared.spec.family,
        "label": prepared.spec.label,
        "config": prepared.spec.config.as_posix(),
        "started_at": "2026-07-22T23:59:59+08:00",
        "finished_at": "2026-07-23T00:00:01+08:00",
        "elapsed_seconds": 2.0,
        "exit_code": 0 if status == "PASS" else 1,
        "status": status,
        "run_id": run_id if include_run else "",
        "run_dir": str(run_dir) if include_run else "",
        "epoch_rows": 1,
        "configured_epochs": 30,
        "best_checkpoint_exists": status == "PASS",
        "last_checkpoint_exists": status == "PASS",
        "val_metrics_exists": status == "PASS",
        "test_metrics_exists": status == "PASS",
        "numeric_status": "FINITE",
        "error_message": "",
    }


def _read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file, delimiter="\t"))


def test_plan_has_causal_first_original_second_in_exact_order() -> None:
    jobs = build_run_plan()
    assert len(jobs) == 16
    assert [job.index for job in jobs] == list(range(1, 17))
    assert [job.family for job in jobs[:8]] == ["causal"] * 8
    assert [job.family for job in jobs[8:]] == ["original"] * 8
    assert [job.config.as_posix() for job in jobs] == [
        "configs/benchmarks/causal_unified/pipelines/mmgcn/legacy_mmgcn_features/val_ses01.yaml",
        "configs/benchmarks/causal_unified/pipelines/mmgcn/legacy_mmgcn_features/val_ses02.yaml",
        "configs/benchmarks/causal_unified/pipelines/mmgcn/legacy_mmgcn_features/val_ses03.yaml",
        "configs/benchmarks/causal_unified/pipelines/mmgcn/legacy_mmgcn_features/val_ses04.yaml",
        "configs/benchmarks/causal_unified/pipelines/multidag_cl/legacy_mmgcn_features/val_ses01.yaml",
        "configs/benchmarks/causal_unified/pipelines/multidag_cl/legacy_mmgcn_features/val_ses02.yaml",
        "configs/benchmarks/causal_unified/pipelines/multidag_cl/legacy_mmgcn_features/val_ses03.yaml",
        "configs/benchmarks/causal_unified/pipelines/multidag_cl/legacy_mmgcn_features/val_ses04.yaml",
        "configs/mmgcn/paper_aligned/iemocap/full_context/legacy_mmgcn_features/screening.yaml",
        "configs/multidag_cl/paper_aligned/iemocap/full_context/legacy_mmgcn_features/screening.yaml",
        "configs/dialoguegcn/paper_aligned/iemocap/full_context/legacy_mmgcn_features/screening.yaml",
        "configs/gsmcc/project_variant/iemocap/full_context/legacy_mmgcn_features/screening.yaml",
        "configs/mmgcn/paper_aligned/iemocap/full_context/clean_roberta_features/mmgcn_clean.yaml",
        "configs/multidag_cl/paper_aligned/iemocap/full_context/clean_roberta_features/multidag_cl_clean.yaml",
        "configs/dialoguegcn/paper_aligned/iemocap/full_context/clean_roberta_features/dialoguegcn_clean.yaml",
        "configs/gsmcc/project_variant/iemocap/full_context/clean_roberta_features/gsmcc_clean.yaml",
    ]
    args = parse_args([])
    assert args.start_index == 1
    assert args.end_index == 16
    assert args.continue_on_error is True
    assert args.poll_seconds == 5.0


def test_prepared_commands_freeze_date_and_leave_formal_yaml_unchanged(
    tmp_path: Path,
) -> None:
    jobs = build_run_plan()
    source_text = {
        job.config: job.config.read_text(encoding="utf-8") for job in jobs
    }
    paths = BatchPaths.create(tmp_path, "20260722", "batch")
    prepared = prepare_runs(
        jobs,
        paths,
        "20260722",
        "cuda",
        str(tmp_path),
    )
    assert len(prepared) == 16
    for item in prepared:
        assert item.command[1] == "-u"
        date_index = item.command.index("--experiment-date")
        assert item.command[date_index + 1] == "20260722"
    for item in prepared[:8]:
        assert item.spec.entrypoint.as_posix() == "scripts/workflows/run_pipeline.py"
        pipeline = yaml.safe_load(item.launch_config.read_text(encoding="utf-8"))
        train = yaml.safe_load(item.expected_config.read_text(encoding="utf-8"))
        assert pipeline["output"]["experiment_date"] == "20260722"
        assert train["system"]["device"] == "cuda"
        assert train["output"]["root"] == str(tmp_path)
    for item in prepared[8:]:
        assert item.spec.entrypoint.as_posix() == (
            "scripts/workflows/paper_aligned/train.py"
        )
        assert "--device" in item.command
        assert "--output-root" in item.command
    assert all(
        job.config.read_text(encoding="utf-8") == source_text[job.config]
        for job in jobs
    )


def test_date_stays_frozen_across_midnight_and_final_summary_is_written(
    tmp_path: Path,
) -> None:
    args = parse_args(
        [
            "--experiment-date",
            "20260722",
            "--output-root",
            str(tmp_path),
            "--start-index",
            "1",
            "--end-index",
            "2",
        ]
    )
    calls: list[tuple[int, str, str | None]] = []

    def executor(
        prepared: PreparedRun,
        _: BatchPaths,
        output_root: Path,
        experiment_date: str,
        __: float,
        ___: LauncherLogger,
    ) -> dict[str, object]:
        calls.append(
            (
                prepared.spec.index,
                experiment_date,
                os.environ.get(EXPERIMENT_DATE_ENV),
            )
        )
        return _result(prepared, output_root, experiment_date)

    summary = launch_batch(
        args,
        preflight_fn=_preflight,
        executor=executor,
        now_provider=lambda: datetime(2026, 7, 22, 23, 59, 59).astimezone(),
    )
    assert calls == [(1, "20260722", "20260722"), (2, "20260722", "20260722")]
    assert summary["experiment_date"] == "20260722"
    assert summary["counts"]["PASS"] == 2
    report = (
        tmp_path
        / "20260722"
        / "reports"
        / summary["batch_id"]
        / "final_summary.json"
    )
    assert report.is_file()
    assert "20260723" not in "\n".join(summary["run_dirs"])


def test_epoch_csv_emits_each_epoch_once(tmp_path: Path) -> None:
    path = tmp_path / "epoch_metrics.csv"
    path.write_text(
        "epoch,train_loss,val_loss,val_weighted_f1,learning_rate\n"
        "1,1.0,0.9,0.4,0.001\n",
        encoding="utf-8",
    )
    messages: list[str] = []
    tailer = EpochCsvTailer()
    assert tailer.emit_new(path, 3, "causal_mmgcn_ses03", messages.append) == 1
    assert tailer.emit_new(path, 3, "causal_mmgcn_ses03", messages.append) == 0
    with path.open("a", encoding="utf-8") as file:
        file.write("2,0.8,0.7,0.5,0.0005\n")
    assert tailer.emit_new(path, 3, "causal_mmgcn_ses03", messages.append) == 1
    assert len(messages) == 2
    assert all(message.startswith("[EPOCH][03/16]") for message in messages)
    assert "epoch=1" in messages[0]
    assert "epoch=2" in messages[1]


def test_epoch_csv_uses_actual_fields_for_different_header(tmp_path: Path) -> None:
    path = tmp_path / "epoch_metrics.csv"
    path.write_text(
        "cycle,train_objective,valid_score,lr\n1,0.8,0.6,0.0001\n",
        encoding="utf-8",
    )
    messages: list[str] = []
    tailer = EpochCsvTailer()
    tailer.emit_new(path, 5, "causal_multidag_cl_ses01", messages.append)
    assert len(messages) == 1
    assert "train_objective=0.8" in messages[0]
    assert "valid_score=0.6" in messages[0]
    assert "lr=0.0001" in messages[0]


def test_monitor_emits_heartbeat_without_waiting_for_epoch(tmp_path: Path) -> None:
    class FakeProcess:
        pid = 1234

        def __init__(self) -> None:
            self.polls = 0

        def poll(self) -> int | None:
            self.polls += 1
            return None if self.polls <= 2 else 0

    class FakeLogger:
        def __init__(self) -> None:
            self.messages: list[str] = []

        def log(self, message: str) -> None:
            self.messages.append(message)

    clock = [0.0]
    prepared = PreparedRun(
        spec=RunSpec(1, "causal", "causal_mmgcn_ses01", Path("a.yaml"), Path("x.py")),
        command=("python", "-u", "x.py"),
        launch_config=tmp_path / "pipeline.yaml",
        expected_config=tmp_path / "train.yaml",
    )
    logger = FakeLogger()
    monitor_process(
        FakeProcess(),
        prepared,
        tmp_path / "missing_runs",
        RunSnapshot(fingerprints={}),
        logger,  # type: ignore[arg-type]
        poll_seconds=5,
        heartbeat_seconds=60,
        monotonic=lambda: clock[0],
        sleep=lambda _: clock.__setitem__(0, 61.0),
        gpu_query=lambda: ("42", "1024"),
    )
    heartbeat = [message for message in logger.messages if message.startswith("[HEARTBEAT]")]
    assert len(heartbeat) == 1
    assert "process_alive=true" in heartbeat[0]
    assert "gpu_utilization=42" in heartbeat[0]
    assert "epoch_rows=0" in heartbeat[0]


def test_failed_job_continues_by_default(tmp_path: Path) -> None:
    args = parse_args(
        [
            "--experiment-date",
            "20260722",
            "--output-root",
            str(tmp_path),
            "--start-index",
            "1",
            "--end-index",
            "2",
        ]
    )
    called: list[int] = []

    def executor(
        prepared: PreparedRun,
        _: BatchPaths,
        output_root: Path,
        experiment_date: str,
        __: float,
        ___: LauncherLogger,
    ) -> dict[str, object]:
        called.append(prepared.spec.index)
        return _result(
            prepared,
            output_root,
            experiment_date,
            status="FAILED_PROCESS" if prepared.spec.index == 1 else "PASS",
        )

    summary = launch_batch(
        args,
        preflight_fn=_preflight,
        executor=executor,
        now_provider=lambda: datetime(2026, 7, 22, 12, 0, 1).astimezone(),
    )
    assert called == [1, 2]
    assert summary["status"] == "COMPLETED_WITH_FAILURES"
    assert summary["counts"]["FAILED_PROCESS"] == 1
    assert summary["counts"]["PASS"] == 1


def test_public_gate_failure_stops_before_any_job(tmp_path: Path) -> None:
    args = parse_args(
        ["--experiment-date", "20260722", "--output-root", str(tmp_path)]
    )
    called = False

    def gate(*_: object, **__: object) -> PreflightResult:
        raise PublicGateError("CUDA unavailable")

    def executor(*_: object, **__: object) -> dict[str, object]:
        nonlocal called
        called = True
        return {}

    with pytest.raises(PublicGateError, match="CUDA unavailable"):
        launch_batch(args, preflight_fn=gate, executor=executor)
    assert called is False
    assert not list(tmp_path.rglob("final_summary.json"))


def test_start_end_range_runs_only_requested_indices(tmp_path: Path) -> None:
    args = parse_args(
        [
            "--experiment-date",
            "20260722",
            "--output-root",
            str(tmp_path),
            "--start-index",
            "4",
            "--end-index",
            "5",
        ]
    )
    called: list[int] = []

    def executor(
        prepared: PreparedRun,
        _: BatchPaths,
        output_root: Path,
        experiment_date: str,
        __: float,
        ___: LauncherLogger,
    ) -> dict[str, object]:
        called.append(prepared.spec.index)
        return _result(prepared, output_root, experiment_date)

    summary = launch_batch(
        args,
        preflight_fn=_preflight,
        executor=executor,
        now_provider=lambda: datetime(2026, 7, 22, 12, 0, 2).astimezone(),
    )
    assert called == [4, 5]
    status_path = (
        tmp_path
        / "20260722"
        / "manifests"
        / summary["batch_id"]
        / "run_status.tsv"
    )
    rows = _read_tsv(status_path)
    assert [row["status"] for row in rows[3:5]] == ["PASS", "PASS"]
    assert sum(row["status"] == "NOT_STARTED" for row in rows) == 14


def _write_run_config(run_dir: Path, config: dict[str, object]) -> None:
    logs = run_dir / "logs"
    logs.mkdir(parents=True)
    (logs / "experiment_config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )


def test_manifest_discovery_uses_only_new_matching_run_and_rejects_smoke(
    tmp_path: Path,
) -> None:
    expected_config = tmp_path / "expected.yaml"
    expected = {
        "profile": "clean_screening",
        "protocol_version": "original_merc_three_track_v2",
        "run_name": "original_mmgcn_clean_screening",
        "model": {"name": "original_repro_mmgcn"},
        "dataset": {
            "feature_pkl_path": "clean.pkl",
            "feature_sha256": "abc",
        },
        "training": {"epochs": 3},
    }
    expected_config.write_text(
        yaml.safe_dump(expected, sort_keys=False), encoding="utf-8"
    )
    runs_root = tmp_path / "20260722" / "runs"
    old_run = runs_root / "old_formal_run"
    _write_run_config(old_run, expected)
    snapshot = snapshot_runs(runs_root)

    new_run = runs_root / "new_formal_run"
    _write_run_config(new_run, expected)
    smoke = dict(expected)
    smoke["profile"] = "smoke"
    smoke_run = runs_root / "original_mmgcn_clean_smoke_1"
    _write_run_config(smoke_run, smoke)

    discovered = discover_current_run_dir(runs_root, snapshot, expected_config)
    assert discovered == new_run

    paths = BatchPaths.create(tmp_path / "batch_output", "20260722", "batch")
    jobs = build_run_plan()
    statuses = [_initial_status(job, 3) for job in jobs]
    statuses[0].update(
        {
            "status": "PASS",
            "run_id": new_run.name,
            "run_dir": str(new_run),
        }
    )
    _write_batch_state(paths, statuses)
    manifest = _read_tsv(paths.manifest_dir / "run_manifest.tsv")
    assert list(manifest[0]) == MANIFEST_COLUMNS
    assert [row["run_id"] for row in manifest] == ["new_formal_run"]
    assert "smoke" not in (paths.manifest_dir / "run_dirs.txt").read_text(
        encoding="utf-8"
    )
    status_rows = _read_tsv(paths.manifest_dir / "run_status.tsv")
    assert list(status_rows[0]) == STATUS_COLUMNS


def test_missing_config_is_a_public_gate_failure(tmp_path: Path) -> None:
    jobs = build_run_plan()
    jobs[0] = replace(jobs[0], config=tmp_path / "missing.yaml")
    from scripts.experiments.run_causal8_original8_formal import run_preflight_checks

    with pytest.raises(PublicGateError, match="does not exist"):
        run_preflight_checks(
            jobs,
            tmp_path,
            "20260722",
            "cuda",
            cuda_check=lambda: True,
        )


@pytest.mark.parametrize(
    ("exit_code", "numeric_status", "artifacts", "expected"),
    [
        (0, "FINITE", True, "PASS"),
        (1, "FINITE", True, "FAILED_PROCESS"),
        (0, "NONFINITE_FORWARD", True, "FAILED_NUMERIC"),
        (0, "FINITE", False, "FAILED_ARTIFACTS"),
        (0, "UNKNOWN", True, "FAILED_ARTIFACTS"),
    ],
)
def test_status_classification_distinguishes_failure_modes(
    exit_code: int,
    numeric_status: str,
    artifacts: bool,
    expected: str,
) -> None:
    assert _status_from_artifacts(
        exit_code,
        numeric_status,
        artifacts,
        artifacts,
        artifacts,
        artifacts,
    ) == expected

"""Run the versioned Causal-8 plus Original-MERC-8 formal retrace batch."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import platform
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Mapping, TextIO

import yaml


PROJECT_ROOT = next(
    candidate
    for candidate in Path(__file__).resolve().parents
    if (candidate / "AGENTS.md").is_file() and (candidate / "scripts").is_dir()
)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.output_paths import (  # noqa: E402
    EXPERIMENT_DATE_ENV,
    resolve_experiment_date,
)


LAUNCHER_VERSION = "causal8_original8_formal_v1"
BATCH_PREFIX = "causal8_original8_formal"
TOTAL_RUNS = 16
HEARTBEAT_SECONDS = 60.0
LEGACY_FEATURE_PATH = Path(
    "third_party/MMGCN_official/IEMOCAP_features/IEMOCAP_features.pkl"
)
CLEAN_FEATURE_PATH = Path(
    "data/processed/iemocap/"
    "IEMOCAP_features_clean_roberta_base_c8b8a37_utterance_mean_v1.pkl"
)
EXPECTED_FEATURE_SHA256 = {
    LEGACY_FEATURE_PATH.as_posix(): (
        "ceb5cc9a45d1792998438e27f86a12642320aa6315546e4425316140ea503fb3"
    ),
    CLEAN_FEATURE_PATH.as_posix(): (
        "c604c557bc3fbb129ca02b2acd57166b669a89ef76ff0cea1e14f9cb206324bf"
    ),
}
STATUS_COLUMNS = [
    "index",
    "family",
    "label",
    "config",
    "started_at",
    "finished_at",
    "elapsed_seconds",
    "exit_code",
    "status",
    "run_id",
    "run_dir",
    "epoch_rows",
    "configured_epochs",
    "best_checkpoint_exists",
    "last_checkpoint_exists",
    "val_metrics_exists",
    "test_metrics_exists",
    "numeric_status",
    "error_message",
]
MANIFEST_COLUMNS = [
    "index",
    "family",
    "label",
    "config",
    "run_id",
    "run_dir",
    "status",
    "started_at",
    "finished_at",
]


class PublicGateError(RuntimeError):
    """A batch-wide prerequisite failed before formal training could start."""


class LauncherRuntimeError(RuntimeError):
    """The launcher itself could not safely continue."""


@dataclass(frozen=True)
class RunSpec:
    index: int
    family: str
    label: str
    config: Path
    entrypoint: Path


@dataclass(frozen=True)
class PreparedRun:
    spec: RunSpec
    command: tuple[str, ...]
    launch_config: Path
    expected_config: Path


@dataclass(frozen=True)
class BatchPaths:
    batch_id: str
    manifest_dir: Path
    log_dir: Path
    report_dir: Path

    @classmethod
    def create(
        cls, output_root: Path, experiment_date: str, batch_id: str
    ) -> "BatchPaths":
        day_root = output_root / experiment_date
        paths = cls(
            batch_id=batch_id,
            manifest_dir=day_root / "manifests" / batch_id,
            log_dir=day_root / "logs" / batch_id,
            report_dir=day_root / "reports" / batch_id,
        )
        for path in (paths.manifest_dir, paths.log_dir, paths.report_dir):
            path.mkdir(parents=True, exist_ok=False)
        return paths


@dataclass(frozen=True)
class PreflightResult:
    git_commit: str
    git_state: str
    feature_records: tuple[dict[str, str], ...]


@dataclass(frozen=True)
class RunSnapshot:
    fingerprints: Mapping[Path, tuple[tuple[str, int, int], ...]]


class LauncherLogger:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._file: TextIO = path.open("a", encoding="utf-8", buffering=1)

    def log(self, message: str) -> None:
        line = str(message).rstrip("\n")
        print(line, flush=True)
        self._file.write(line + "\n")
        self._file.flush()

    def close(self) -> None:
        self._file.close()

    def __enter__(self) -> "LauncherLogger":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


class BatchLock:
    def __init__(self, path: Path, batch_id: str) -> None:
        self.path = path
        self.batch_id = batch_id
        self.acquired = False

    def acquire(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            try:
                payload = json.loads(self.path.read_text(encoding="utf-8"))
                pid = int(payload.get("pid", -1))
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                pid = -1
            if _pid_is_alive(pid):
                raise PublicGateError(
                    f"Another {BATCH_PREFIX} batch is running with pid={pid}."
                )
            self.path.unlink()
        payload = {
            "batch_id": self.batch_id,
            "pid": os.getpid(),
            "created_at": _timestamp(),
        }
        try:
            descriptor = os.open(
                self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY
            )
        except FileExistsError as error:
            raise PublicGateError(
                f"Concurrent batch lock already exists: {self.path}"
            ) from error
        with os.fdopen(descriptor, "w", encoding="utf-8") as file:
            json.dump(payload, file, ensure_ascii=False, indent=2)
            file.write("\n")
        self.acquired = True

    def release(self) -> None:
        if self.acquired and self.path.exists():
            self.path.unlink()
        self.acquired = False

    def __enter__(self) -> "BatchLock":
        self.acquire()
        return self

    def __exit__(self, *_: object) -> None:
        self.release()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-date", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-root", default="outputs")
    parser.add_argument("--start-index", type=int, default=1)
    parser.add_argument("--end-index", type=int, default=TOTAL_RUNS)
    parser.add_argument(
        "--continue-on-error",
        dest="continue_on_error",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "--no-continue-on-error",
        dest="continue_on_error",
        action="store_false",
    )
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    return parser.parse_args(argv)


def build_run_plan() -> list[RunSpec]:
    causal: list[RunSpec] = []
    index = 1
    for model_dir, model_label in (
        ("mmgcn", "mmgcn"),
        ("multidag_cl", "multidag_cl"),
    ):
        for session in range(1, 5):
            causal.append(
                RunSpec(
                    index=index,
                    family="causal",
                    label=f"causal_{model_label}_ses{session:02d}",
                    config=Path(
                        "configs/benchmarks/causal_unified/pipelines/"
                        f"{model_dir}/legacy_mmgcn_features/"
                        f"val_ses{session:02d}.yaml"
                    ),
                    entrypoint=Path("scripts/workflows/run_pipeline.py"),
                )
            )
            index += 1

    original: list[RunSpec] = []
    canonical_original_configs = {
        "mmgcn": {
            "legacy": Path(
                "configs/mmgcn/paper_aligned/iemocap/full_context/"
                "legacy_mmgcn_features/screening.yaml"
            ),
            "clean": Path(
                "configs/mmgcn/paper_aligned/iemocap/full_context/"
                "clean_roberta_features/mmgcn_clean.yaml"
            ),
        },
        "multidag_cl": {
            "legacy": Path(
                "configs/multidag_cl/paper_aligned/iemocap/full_context/"
                "legacy_mmgcn_features/screening.yaml"
            ),
            "clean": Path(
                "configs/multidag_cl/paper_aligned/iemocap/full_context/"
                "clean_roberta_features/multidag_cl_clean.yaml"
            ),
        },
        "dialoguegcn": {
            "legacy": Path(
                "configs/dialoguegcn/paper_aligned/iemocap/full_context/"
                "legacy_mmgcn_features/screening.yaml"
            ),
            "clean": Path(
                "configs/dialoguegcn/paper_aligned/iemocap/full_context/"
                "clean_roberta_features/dialoguegcn_clean.yaml"
            ),
        },
        "gsmcc": {
            "legacy": Path(
                "configs/gsmcc/project_variant/iemocap/full_context/"
                "legacy_mmgcn_features/screening.yaml"
            ),
            "clean": Path(
                "configs/gsmcc/project_variant/iemocap/full_context/"
                "clean_roberta_features/gsmcc_clean.yaml"
            ),
        },
    }
    for track in ("legacy", "clean"):
        for model in ("mmgcn", "multidag_cl", "dialoguegcn", "gsmcc"):
            config = canonical_original_configs[model][track]
            original.append(
                RunSpec(
                    index=index,
                    family="original",
                    label=f"original_{model}_{track}",
                    config=config,
                    entrypoint=Path(
                        "scripts/workflows/paper_aligned/train.py"
                    ),
                )
            )
            index += 1
    return causal + original


def _resolve_project_path(path: Path | str) -> Path:
    value = Path(path)
    return value if value.is_absolute() else PROJECT_ROOT / value


def _record_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def _command_path(path: Path) -> str:
    return _record_path(path)


def _timestamp(now: datetime | None = None) -> str:
    return (now or datetime.now().astimezone()).astimezone().isoformat()


def _pid_is_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        import ctypes

        process_query_limited_information = 0x1000
        handle = ctypes.windll.kernel32.OpenProcess(  # type: ignore[attr-defined]
            process_query_limited_information, False, pid
        )
        if not handle:
            return False
        ctypes.windll.kernel32.CloseHandle(handle)  # type: ignore[attr-defined]
        return True
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise PublicGateError(f"Cannot read YAML config {path}: {error}") from error
    if not isinstance(value, dict):
        raise PublicGateError(f"YAML config is not a mapping: {path}")
    return value


def _write_yaml(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as file:
        yaml.safe_dump(dict(value), file, sort_keys=False, allow_unicode=True)


def _sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _cuda_available() -> bool:
    import torch

    return bool(torch.cuda.is_available())


def _git_capture(*args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return result.stdout.rstrip()


def _check_output_writable(output_root: Path, experiment_date: str) -> None:
    day_root = output_root / experiment_date
    try:
        day_root.mkdir(parents=True, exist_ok=True)
        probe = day_root / f".{BATCH_PREFIX}.write_probe_{os.getpid()}"
        with probe.open("x", encoding="utf-8") as file:
            file.write("writable\n")
        probe.unlink()
    except OSError as error:
        raise PublicGateError(
            f"Output directory is not writable: {day_root}: {error}"
        ) from error


def _declared_feature(config: Mapping[str, Any]) -> tuple[str, str]:
    dataset = config.get("dataset", {})
    return (
        str(dataset.get("feature_pkl_path", "")).replace("\\", "/"),
        str(dataset.get("feature_sha256", "")).lower(),
    )


def run_preflight_checks(
    jobs: list[RunSpec],
    output_root: Path,
    experiment_date: str,
    device: str,
    *,
    cuda_check: Callable[[], bool] = _cuda_available,
) -> PreflightResult:
    if len(jobs) != TOTAL_RUNS:
        raise PublicGateError(f"Expected {TOTAL_RUNS} jobs, found {len(jobs)}.")
    for job in jobs:
        for path in (job.config, job.entrypoint):
            resolved = _resolve_project_path(path)
            if not resolved.is_file():
                raise PublicGateError(f"Required file does not exist: {path}")

    if not str(device).lower().startswith("cuda") or not cuda_check():
        raise PublicGateError(
            f"CUDA is required for this formal batch; requested device={device!r}."
        )

    declared: set[tuple[str, str]] = set()
    for job in jobs:
        config = _load_yaml(_resolve_project_path(job.config))
        feature_path, feature_sha = _declared_feature(config)
        expected_sha = EXPECTED_FEATURE_SHA256.get(feature_path)
        if expected_sha is None or feature_sha != expected_sha:
            raise PublicGateError(
                f"Unexpected feature declaration in {job.config}: "
                f"path={feature_path!r}, sha256={feature_sha!r}."
            )
        declared.add((feature_path, feature_sha))
        if job.family == "causal":
            train_config_path = _causal_train_config(config)
            train_feature_path, train_feature_sha = _declared_feature(
                _load_yaml(train_config_path)
            )
            if (train_feature_path, train_feature_sha) != (
                feature_path,
                feature_sha,
            ):
                raise PublicGateError(
                    "Feature declaration differs between causal pipeline and "
                    f"train config: {job.config}"
                )
    if declared != set(EXPECTED_FEATURE_SHA256.items()):
        raise PublicGateError(
            "The 16 configs must resolve to exactly the Legacy and Clean feature sets."
        )

    feature_records = []
    for relative_path, expected_sha in EXPECTED_FEATURE_SHA256.items():
        feature_path = _resolve_project_path(relative_path)
        if not feature_path.is_file():
            raise PublicGateError(f"Feature PKL does not exist: {relative_path}")
        actual_sha = _sha256(feature_path)
        if actual_sha != expected_sha:
            raise PublicGateError(
                f"Feature SHA mismatch for {relative_path}: "
                f"expected={expected_sha}, actual={actual_sha}"
            )
        feature_records.append(
            {
                "path": relative_path,
                "expected_sha256": expected_sha,
                "actual_sha256": actual_sha,
                "status": "PASS",
            }
        )

    _check_output_writable(output_root, experiment_date)
    try:
        git_commit = _git_capture("rev-parse", "HEAD")
        git_status = _git_capture("status", "--short")
    except (OSError, subprocess.CalledProcessError) as error:
        raise PublicGateError(f"Cannot record Git state: {error}") from error
    if not git_commit:
        raise PublicGateError("Git commit is empty.")
    git_state = f"commit={git_commit}\nstatus:\n{git_status}\n"
    return PreflightResult(
        git_commit=git_commit,
        git_state=git_state,
        feature_records=tuple(feature_records),
    )


def _causal_train_config(pipeline_config: Mapping[str, Any]) -> Path:
    train = pipeline_config.get("train", {})
    value = train.get("train_config_path")
    if value is None or not str(value).strip():
        raise PublicGateError("Causal pipeline config has no train_config_path.")
    path = _resolve_project_path(str(value))
    if not path.is_file():
        raise PublicGateError(f"Causal train config does not exist: {value}")
    return path


def prepare_runs(
    jobs: list[RunSpec],
    paths: BatchPaths,
    experiment_date: str,
    device: str,
    output_root_argument: str,
) -> list[PreparedRun]:
    prepared = []
    resolved_dir = paths.manifest_dir / "resolved_configs"
    for job in jobs:
        source_config = _resolve_project_path(job.config)
        if job.family == "causal":
            pipeline_config = _load_yaml(source_config)
            source_train_config = _causal_train_config(pipeline_config)
            train_config = _load_yaml(source_train_config)
            train_config.setdefault("system", {})["device"] = device
            train_config["system"]["output_dir"] = output_root_argument
            train_config.setdefault("output", {}).update(
                {
                    "root": output_root_argument,
                    "experiment_date": experiment_date,
                }
            )
            train_destination = resolved_dir / f"{job.index:02d}_{job.label}_train.yaml"
            _write_yaml(train_destination, train_config)

            pipeline_config.setdefault("output", {}).update(
                {
                    "root": output_root_argument,
                    "experiment_date": experiment_date,
                }
            )
            pipeline_config.setdefault("train", {})["train_config_path"] = (
                _command_path(train_destination)
            )
            pipeline_destination = (
                resolved_dir / f"{job.index:02d}_{job.label}_pipeline.yaml"
            )
            _write_yaml(pipeline_destination, pipeline_config)
            command = (
                sys.executable,
                "-u",
                _command_path(job.entrypoint),
                "--config",
                _command_path(pipeline_destination),
                "--experiment-date",
                experiment_date,
            )
            prepared.append(
                PreparedRun(
                    spec=job,
                    command=command,
                    launch_config=pipeline_destination,
                    expected_config=train_destination,
                )
            )
        else:
            command = (
                sys.executable,
                "-u",
                _command_path(job.entrypoint),
                "--config",
                _command_path(source_config),
                "--device",
                device,
                "--output-root",
                output_root_argument,
                "--experiment-date",
                experiment_date,
            )
            prepared.append(
                PreparedRun(
                    spec=job,
                    command=command,
                    launch_config=source_config,
                    expected_config=source_config,
                )
            )
    return prepared


def _atomic_write_tsv(
    path: Path, rows: list[Mapping[str, Any]], columns: list[str]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file, fieldnames=columns, delimiter="\t", lineterminator="\n"
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})
    os.replace(temporary, path)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as file:
        json.dump(dict(value), file, ensure_ascii=False, indent=2)
        file.write("\n")
    os.replace(temporary, path)


def _initial_status(job: RunSpec, configured_epochs: int | None) -> dict[str, Any]:
    return {
        "index": job.index,
        "family": job.family,
        "label": job.label,
        "config": job.config.as_posix(),
        "started_at": "",
        "finished_at": "",
        "elapsed_seconds": "",
        "exit_code": "",
        "status": "NOT_STARTED",
        "run_id": "",
        "run_dir": "",
        "epoch_rows": 0,
        "configured_epochs": configured_epochs if configured_epochs is not None else "",
        "best_checkpoint_exists": False,
        "last_checkpoint_exists": False,
        "val_metrics_exists": False,
        "test_metrics_exists": False,
        "numeric_status": "UNKNOWN",
        "error_message": "",
    }


def _nested(config: Mapping[str, Any], *keys: str) -> Any:
    value: Any = config
    for key in keys:
        if not isinstance(value, Mapping):
            return None
        value = value.get(key)
    return value


def configured_epochs(config_path: Path) -> int | None:
    config = _load_yaml(config_path)
    for keys in (
        ("training", "epochs"),
        ("training", "max_epochs"),
        ("train", "max_epochs"),
        ("train", "epochs"),
    ):
        value = _nested(config, *keys)
        if value is not None:
            return int(value)
    return None


def _run_fingerprint(run_dir: Path) -> tuple[tuple[str, int, int], ...]:
    rows = []
    for relative in (
        Path("run_metadata.json"),
        Path("logs/experiment_config.yaml"),
        Path("logs/epoch_metrics.csv"),
    ):
        path = run_dir / relative
        if not path.is_file():
            continue
        try:
            stat = path.stat()
        except OSError:
            continue
        rows.append((relative.as_posix(), stat.st_mtime_ns, stat.st_size))
    return tuple(rows)


def snapshot_runs(runs_root: Path) -> RunSnapshot:
    fingerprints: dict[Path, tuple[tuple[str, int, int], ...]] = {}
    if runs_root.is_dir():
        for run_dir in runs_root.iterdir():
            if run_dir.is_dir():
                fingerprints[run_dir] = _run_fingerprint(run_dir)
    return RunSnapshot(fingerprints=fingerprints)


def _normalized_path(value: Any) -> str:
    return str(value or "").strip().replace("\\", "/").lower()


def _config_signature(config: Mapping[str, Any]) -> dict[str, str]:
    values = {
        "causal": config.get("causal"),
        "profile": config.get("profile"),
        "protocol_version": config.get("protocol_version"),
        "run_name": config.get("run_name"),
        "experiment_name": _nested(config, "project", "experiment_name"),
        "model_name": _nested(config, "model", "name"),
        "feature_pkl_path": _nested(config, "dataset", "feature_pkl_path"),
        "feature_sha256": _nested(config, "dataset", "feature_sha256"),
        "val_split_strategy": _nested(config, "dataset", "val_split_strategy"),
        "val_session_id": _nested(config, "dataset", "val_session_id"),
    }
    return {
        key: _normalized_path(value)
        for key, value in values.items()
        if value is not None and str(value).strip()
    }


def _metadata_config_matches(metadata: Mapping[str, Any], expected: Path) -> bool:
    value = metadata.get("config_path") or metadata.get("source_config_path")
    if value is None or not str(value).strip():
        return True
    candidate = Path(str(value))
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    return candidate.resolve() == expected.resolve()


def run_matches_config(run_dir: Path, expected_config: Path) -> bool:
    if "_smoke_" in run_dir.name.lower():
        return False
    config_path = run_dir / "logs" / "experiment_config.yaml"
    if not config_path.is_file():
        return False
    try:
        actual = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        expected = yaml.safe_load(expected_config.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        return False
    if not isinstance(actual, dict) or not isinstance(expected, dict):
        return False
    if str(actual.get("profile", "")).strip().lower() == "smoke":
        return False
    expected_signature = _config_signature(expected)
    actual_signature = _config_signature(actual)
    if any(
        actual_signature.get(key) != value
        for key, value in expected_signature.items()
    ):
        return False
    metadata_path = run_dir / "run_metadata.json"
    if metadata_path.is_file():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
        if not isinstance(metadata, dict) or not _metadata_config_matches(
            metadata, expected_config
        ):
            return False
    return True


def discover_current_run_dir(
    runs_root: Path,
    snapshot: RunSnapshot,
    expected_config: Path,
) -> Path | None:
    if not runs_root.is_dir():
        return None
    new_matches = []
    updated_matches = []
    for run_dir in runs_root.iterdir():
        if not run_dir.is_dir() or not run_matches_config(run_dir, expected_config):
            continue
        current = _run_fingerprint(run_dir)
        if run_dir not in snapshot.fingerprints:
            new_matches.append(run_dir)
        elif current != snapshot.fingerprints[run_dir]:
            updated_matches.append(run_dir)
    candidates = new_matches if new_matches else updated_matches
    if len(candidates) > 1:
        locations = ", ".join(sorted(_record_path(path) for path in candidates))
        raise LauncherRuntimeError(
            f"Multiple new or updated runs match the current config: {locations}"
        )
    return candidates[0] if candidates else None


class EpochCsvTailer:
    def __init__(self) -> None:
        self.seen: set[str] = set()
        self.row_count = 0

    @staticmethod
    def _complete_rows(path: Path) -> list[dict[str, str]]:
        if not path.is_file():
            return []
        try:
            text = path.read_text(encoding="utf-8-sig")
        except OSError:
            return []
        if not text:
            return []
        lines = text.splitlines(keepends=True)
        if lines and not lines[-1].endswith(("\n", "\r")):
            lines = lines[:-1]
        if len(lines) < 2:
            return []
        reader = csv.DictReader(io.StringIO("".join(lines)))
        return [
            {str(key): "" if value is None else str(value) for key, value in row.items()}
            for row in reader
            if None not in row
        ]

    @staticmethod
    def _row_key(row: Mapping[str, str], row_index: int) -> str:
        for name in ("epoch", "epoch_index", "epoch_number"):
            value = row.get(name)
            if value is not None and str(value).strip():
                return f"{name}:{value}"
        return f"row:{row_index}"

    @staticmethod
    def _display_fields(row: Mapping[str, str]) -> list[tuple[str, str]]:
        preferred = (
            "epoch",
            "train_loss",
            "val_loss",
            "val_accuracy",
            "val_weighted_f1",
            "learning_rate",
            "lr",
        )
        selected = [
            (name, row[name])
            for name in preferred
            if name in row and str(row[name]).strip()
        ]
        selected_names = {name for name, _ in selected}
        for name, value in row.items():
            normalized = name.lower()
            if name in selected_names or not str(value).strip():
                continue
            if (
                "epoch" in normalized
                or normalized.startswith("train_")
                or normalized.startswith("val_")
                or normalized.startswith("valid_")
                or normalized in {"learning_rate", "lr"}
            ):
                selected.append((name, value))
        if not selected:
            selected = [(name, value) for name, value in row.items() if value]
        return selected

    def emit_new(
        self,
        csv_path: Path,
        index: int,
        label: str,
        log: Callable[[str], None],
    ) -> int:
        rows = self._complete_rows(csv_path)
        self.row_count = max(self.row_count, len(rows))
        emitted = 0
        for row_index, row in enumerate(rows, start=1):
            key = self._row_key(row, row_index)
            if key in self.seen:
                continue
            self.seen.add(key)
            fields = " ".join(
                f"{name}={str(value).replace(chr(10), ' ')}"
                for name, value in self._display_fields(row)
            )
            log(f"[EPOCH][{index:02d}/{TOTAL_RUNS}][{label}] {fields}")
            emitted += 1
        return emitted


def query_gpu_status() -> tuple[str, str]:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        first_line = result.stdout.strip().splitlines()[0]
        utilization, memory = (value.strip() for value in first_line.split(",", 1))
        return utilization, memory
    except (OSError, subprocess.SubprocessError, IndexError, ValueError):
        return "unavailable", "unavailable"


def _elapsed_text(seconds: float) -> str:
    total = max(0, int(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, seconds_value = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds_value:02d}"


def monitor_process(
    process: Any,
    prepared: PreparedRun,
    runs_root: Path,
    snapshot: RunSnapshot,
    logger: LauncherLogger,
    *,
    poll_seconds: float,
    heartbeat_seconds: float = HEARTBEAT_SECONDS,
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
    gpu_query: Callable[[], tuple[str, str]] = query_gpu_status,
) -> tuple[Path | None, int]:
    start = monotonic()
    next_heartbeat = start + heartbeat_seconds
    run_dir: Path | None = None
    tailer = EpochCsvTailer()
    while True:
        if run_dir is None:
            run_dir = discover_current_run_dir(
                runs_root, snapshot, prepared.expected_config
            )
        if run_dir is not None:
            tailer.emit_new(
                run_dir / "logs" / "epoch_metrics.csv",
                prepared.spec.index,
                prepared.spec.label,
                logger.log,
            )
        now = monotonic()
        alive = process.poll() is None
        if now >= next_heartbeat:
            gpu_utilization, gpu_memory = gpu_query()
            logger.log(
                "[HEARTBEAT] "
                f"index={prepared.spec.index} label={prepared.spec.label} "
                f"elapsed={_elapsed_text(now - start)} process_pid={process.pid} "
                f"process_alive={str(alive).lower()} "
                f"gpu_utilization={gpu_utilization} gpu_memory={gpu_memory} "
                f"epoch_rows={tailer.row_count} "
                f"current_run_dir={_record_path(run_dir) if run_dir else ''}"
            )
            while next_heartbeat <= now:
                next_heartbeat += heartbeat_seconds
        if not alive:
            break
        sleep(poll_seconds)

    if run_dir is None:
        run_dir = discover_current_run_dir(
            runs_root, snapshot, prepared.expected_config
        )
    if run_dir is not None:
        tailer.emit_new(
            run_dir / "logs" / "epoch_metrics.csv",
            prepared.spec.index,
            prepared.spec.label,
            logger.log,
        )
    return run_dir, tailer.row_count


def _metrics_exists(run_dir: Path, split: str) -> bool:
    evaluations = run_dir / "logs" / "evaluations"
    return any(evaluations.glob(f"{split}_*/metrics.csv"))


def inspect_numeric_status(run_dir: Path | None) -> str:
    if run_dir is None:
        return "UNKNOWN"
    summary_path = run_dir / "logs" / "run_summary.json"
    if summary_path.is_file():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return "UNKNOWN"
        if isinstance(summary, dict) and summary.get("numeric_status"):
            return str(summary["numeric_status"])

    checked_numeric = False
    csv_paths = [run_dir / "logs" / "epoch_metrics.csv"]
    evaluations = run_dir / "logs" / "evaluations"
    if evaluations.is_dir():
        csv_paths.extend(evaluations.glob("*_*/metrics.csv"))
    for path in csv_paths:
        if not path.is_file():
            continue
        try:
            with path.open("r", encoding="utf-8-sig", newline="") as file:
                rows = list(csv.DictReader(file))
        except (OSError, csv.Error):
            return "UNKNOWN"
        for row in rows:
            for value in row.values():
                try:
                    numeric = float(str(value))
                except (TypeError, ValueError):
                    continue
                checked_numeric = True
                if not math.isfinite(numeric):
                    return "NONFINITE_ARTIFACT"
    return "FINITE" if checked_numeric else "UNKNOWN"


def _status_from_artifacts(
    exit_code: int,
    numeric_status: str,
    best_exists: bool,
    last_exists: bool,
    val_exists: bool,
    test_exists: bool,
) -> str:
    if numeric_status not in {"FINITE", "UNKNOWN"}:
        return "FAILED_NUMERIC"
    if exit_code != 0:
        return "FAILED_PROCESS"
    if not all((best_exists, last_exists, val_exists, test_exists)):
        return "FAILED_ARTIFACTS"
    if numeric_status != "FINITE":
        return "FAILED_ARTIFACTS"
    return "PASS"


def execute_one_run(
    prepared: PreparedRun,
    paths: BatchPaths,
    output_root: Path,
    experiment_date: str,
    poll_seconds: float,
    logger: LauncherLogger,
) -> dict[str, Any]:
    job = prepared.spec
    runs_root = output_root / experiment_date / "runs"
    snapshot = snapshot_runs(runs_root)
    marker_dir = paths.manifest_dir / "markers"
    marker_dir.mkdir(exist_ok=True)
    marker_path = marker_dir / f"{job.index:02d}_{job.label}.started"
    started_at = _timestamp()
    marker_path.write_text(started_at + "\n", encoding="utf-8")
    log_path = paths.log_dir / f"{job.index:02d}_{job.label}.log"
    logger.log(
        f"[START][{job.index:02d}/{TOTAL_RUNS}][{job.label}] "
        + " ".join(prepared.command)
    )
    start_monotonic = time.monotonic()
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env[EXPERIMENT_DATE_ENV] = experiment_date
    with log_path.open("w", encoding="utf-8", buffering=1) as child_log:
        process = subprocess.Popen(
            list(prepared.command),
            cwd=PROJECT_ROOT,
            env=env,
            stdout=child_log,
            stderr=subprocess.STDOUT,
            text=True,
        )
        try:
            run_dir, epoch_rows = monitor_process(
                process,
                prepared,
                runs_root,
                snapshot,
                logger,
                poll_seconds=poll_seconds,
            )
            exit_code = int(process.wait())
        except KeyboardInterrupt:
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            raise
    finished_at = _timestamp()
    elapsed_seconds = time.monotonic() - start_monotonic
    best_exists = bool(run_dir and (run_dir / "checkpoints/best_model.pt").is_file())
    last_exists = bool(run_dir and (run_dir / "checkpoints/last_model.pt").is_file())
    val_exists = bool(run_dir and _metrics_exists(run_dir, "val"))
    test_exists = bool(run_dir and _metrics_exists(run_dir, "test"))
    numeric_status = inspect_numeric_status(run_dir)
    status = _status_from_artifacts(
        exit_code,
        numeric_status,
        best_exists,
        last_exists,
        val_exists,
        test_exists,
    )
    logger.log(
        f"[END][{job.index:02d}/{TOTAL_RUNS}][{job.label}] "
        f"status={status} exit_code={exit_code} epoch_rows={epoch_rows}"
    )
    return {
        "index": job.index,
        "family": job.family,
        "label": job.label,
        "config": job.config.as_posix(),
        "started_at": started_at,
        "finished_at": finished_at,
        "elapsed_seconds": round(elapsed_seconds, 3),
        "exit_code": exit_code,
        "status": status,
        "run_id": run_dir.name if run_dir else "",
        "run_dir": _record_path(run_dir) if run_dir else "",
        "epoch_rows": epoch_rows,
        "configured_epochs": configured_epochs(prepared.expected_config) or "",
        "best_checkpoint_exists": best_exists,
        "last_checkpoint_exists": last_exists,
        "val_metrics_exists": val_exists,
        "test_metrics_exists": test_exists,
        "numeric_status": numeric_status,
        "error_message": "",
    }


def _write_batch_state(
    paths: BatchPaths,
    statuses: list[dict[str, Any]],
) -> None:
    ordered = sorted(statuses, key=lambda row: int(row["index"]))
    _atomic_write_tsv(paths.manifest_dir / "run_status.tsv", ordered, STATUS_COLUMNS)
    manifest_rows = [row for row in ordered if row.get("run_id") and row.get("run_dir")]
    _atomic_write_tsv(
        paths.manifest_dir / "run_manifest.tsv",
        manifest_rows,
        MANIFEST_COLUMNS,
    )
    run_dirs_path = paths.manifest_dir / "run_dirs.txt"
    run_dirs_path.write_text(
        "".join(f"{row['run_dir']}\n" for row in manifest_rows),
        encoding="utf-8",
    )


def _planned_rows(prepared: list[PreparedRun]) -> list[dict[str, Any]]:
    return [
        {
            "index": item.spec.index,
            "family": item.spec.family,
            "label": item.spec.label,
            "config": item.spec.config.as_posix(),
            "entrypoint": item.spec.entrypoint.as_posix(),
            "launch_config": _record_path(item.launch_config),
            "command": " ".join(item.command),
        }
        for item in prepared
    ]


def _write_launcher_metadata(
    paths: BatchPaths,
    args: argparse.Namespace,
    experiment_date: str,
    output_root: Path,
    preflight: PreflightResult,
    started_at: str,
) -> None:
    metadata = {
        "launcher_version": LAUNCHER_VERSION,
        "batch_id": paths.batch_id,
        "started_at": started_at,
        "experiment_date": experiment_date,
        "experiment_date_env": experiment_date,
        "device": args.device,
        "output_root": _record_path(output_root),
        "start_index": args.start_index,
        "end_index": args.end_index,
        "continue_on_error": args.continue_on_error,
        "poll_seconds": args.poll_seconds,
        "pid": os.getpid(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.executable,
        "git_commit": preflight.git_commit,
        "feature_records": list(preflight.feature_records),
    }
    _write_json(paths.manifest_dir / "launcher_metadata.json", metadata)
    (paths.manifest_dir / "git_state.txt").write_text(
        preflight.git_state, encoding="utf-8"
    )
    feature_lines = [
        "feature\texpected_sha256\tactual_sha256\tstatus"
    ] + [
        "\t".join(
            (
                row["path"],
                row["expected_sha256"],
                row["actual_sha256"],
                row["status"],
            )
        )
        for row in preflight.feature_records
    ]
    (paths.manifest_dir / "feature_sha256.txt").write_text(
        "\n".join(feature_lines) + "\n", encoding="utf-8"
    )


def _validate_arguments(args: argparse.Namespace) -> None:
    if not 1 <= args.start_index <= TOTAL_RUNS:
        raise PublicGateError("--start-index must be between 1 and 16.")
    if not 1 <= args.end_index <= TOTAL_RUNS:
        raise PublicGateError("--end-index must be between 1 and 16.")
    if args.start_index > args.end_index:
        raise PublicGateError("--start-index cannot exceed --end-index.")
    if args.poll_seconds <= 0:
        raise PublicGateError("--poll-seconds must be greater than zero.")


def launch_batch(
    args: argparse.Namespace,
    *,
    preflight_fn: Callable[..., PreflightResult] = run_preflight_checks,
    executor: Callable[..., dict[str, Any]] = execute_one_run,
    now_provider: Callable[[], datetime] = lambda: datetime.now().astimezone(),
) -> dict[str, Any]:
    _validate_arguments(args)
    experiment_date = resolve_experiment_date(cli_date=args.experiment_date)
    output_root_argument = str(args.output_root)
    output_root = _resolve_project_path(output_root_argument).resolve()
    jobs = build_run_plan()
    preflight = preflight_fn(
        jobs, output_root, experiment_date, args.device
    )
    launched_at = now_provider()
    batch_id = f"{BATCH_PREFIX}_{launched_at.strftime('%Y%m%d_%H%M%S')}"
    lock_path = output_root / f".{BATCH_PREFIX}.lock"
    old_experiment_date = os.environ.get(EXPERIMENT_DATE_ENV)
    os.environ[EXPERIMENT_DATE_ENV] = experiment_date
    paths: BatchPaths | None = None
    logger: LauncherLogger | None = None
    statuses: list[dict[str, Any]] = []
    launcher_status = "PASS"
    internal_error = ""
    try:
        with BatchLock(lock_path, batch_id):
            paths = BatchPaths.create(output_root, experiment_date, batch_id)
            logger = LauncherLogger(paths.log_dir / "launcher.log")
            prepared = prepare_runs(
                jobs,
                paths,
                experiment_date,
                args.device,
                output_root_argument,
            )
            configured_by_index = {
                item.spec.index: configured_epochs(item.expected_config)
                for item in prepared
            }
            statuses = [
                _initial_status(job, configured_by_index[job.index])
                for job in jobs
            ]
            _atomic_write_tsv(
                paths.manifest_dir / "planned_runs.tsv",
                _planned_rows(prepared),
                [
                    "index",
                    "family",
                    "label",
                    "config",
                    "entrypoint",
                    "launch_config",
                    "command",
                ],
            )
            _write_launcher_metadata(
                paths,
                args,
                experiment_date,
                output_root,
                preflight,
                _timestamp(launched_at),
            )
            _write_batch_state(paths, statuses)
            logger.log(
                f"[BATCH] id={batch_id} experiment_date={experiment_date} "
                f"range={args.start_index}-{args.end_index}"
            )
            for item in prepared:
                if not args.start_index <= item.spec.index <= args.end_index:
                    continue
                try:
                    row = executor(
                        item,
                        paths,
                        output_root,
                        experiment_date,
                        args.poll_seconds,
                        logger,
                    )
                except KeyboardInterrupt:
                    row = _initial_status(
                        item.spec, configured_by_index[item.spec.index]
                    )
                    row.update(
                        {
                            "started_at": _timestamp(),
                            "finished_at": _timestamp(),
                            "status": "INTERRUPTED",
                            "error_message": "KeyboardInterrupt",
                        }
                    )
                    statuses[item.spec.index - 1] = row
                    _write_batch_state(paths, statuses)
                    launcher_status = "INTERRUPTED"
                    raise
                except Exception as error:
                    row = _initial_status(
                        item.spec, configured_by_index[item.spec.index]
                    )
                    row.update(
                        {
                            "started_at": _timestamp(),
                            "finished_at": _timestamp(),
                            "status": "FAILED_PROCESS",
                            "error_message": f"launcher_exception:{error}",
                        }
                    )
                    statuses[item.spec.index - 1] = row
                    _write_batch_state(paths, statuses)
                    launcher_status = "FAILED_INTERNAL"
                    internal_error = str(error)
                    raise LauncherRuntimeError(
                        f"Launcher failed at job {item.spec.index}: {error}"
                    ) from error
                statuses[item.spec.index - 1] = row
                _write_batch_state(paths, statuses)
                if row["status"] != "PASS":
                    launcher_status = "COMPLETED_WITH_FAILURES"
                    if not args.continue_on_error:
                        break
    except KeyboardInterrupt:
        if launcher_status == "PASS":
            launcher_status = "INTERRUPTED"
        raise
    except Exception as error:
        if launcher_status == "PASS":
            launcher_status = "FAILED_INTERNAL"
            internal_error = str(error)
        raise
    finally:
        if paths is not None:
            counts = {
                status: sum(row.get("status") == status for row in statuses)
                for status in (
                    "PASS",
                    "FAILED_PROCESS",
                    "FAILED_ARTIFACTS",
                    "FAILED_NUMERIC",
                    "INTERRUPTED",
                    "NOT_STARTED",
                )
            }
            summary = {
                "launcher_version": LAUNCHER_VERSION,
                "batch_id": batch_id,
                "status": launcher_status,
                "experiment_date": experiment_date,
                "started_at": _timestamp(launched_at),
                "finished_at": _timestamp(),
                "selected_range": [args.start_index, args.end_index],
                "continue_on_error": args.continue_on_error,
                "counts": counts,
                "run_ids": [row["run_id"] for row in statuses if row.get("run_id")],
                "run_dirs": [row["run_dir"] for row in statuses if row.get("run_dir")],
                "internal_error": internal_error,
                "batch_manifest": _record_path(
                    paths.manifest_dir / "run_manifest.tsv"
                ),
                "run_dirs_file": _record_path(
                    paths.manifest_dir / "run_dirs.txt"
                ),
            }
            _write_json(paths.report_dir / "final_summary.json", summary)
            if logger is not None:
                logger.log(
                    f"[SUMMARY] status={launcher_status} counts={json.dumps(counts)}"
                )
                logger.close()
        if old_experiment_date is None:
            os.environ.pop(EXPERIMENT_DATE_ENV, None)
        else:
            os.environ[EXPERIMENT_DATE_ENV] = old_experiment_date
    return summary


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        summary = launch_batch(args)
    except PublicGateError as error:
        print(f"[PUBLIC_GATE_FAILED] {error}", file=sys.stderr, flush=True)
        return 2
    except KeyboardInterrupt:
        print("[INTERRUPTED] launcher interrupted", file=sys.stderr, flush=True)
        return 130
    except LauncherRuntimeError as error:
        print(f"[LAUNCHER_FAILED] {error}", file=sys.stderr, flush=True)
        return 3
    except Exception as error:
        print(f"[LAUNCHER_FAILED] {error}", file=sys.stderr, flush=True)
        return 3
    return 0 if summary["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())

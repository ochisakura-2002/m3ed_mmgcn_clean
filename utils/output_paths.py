"""Canonical experiment-output paths with read-only legacy discovery.

New artifacts use the fixed hierarchy::

    <output_base>/<YYYYMMDD>/<experiment_group>/<functional_directory>/...

The functional directories are ``runs``, ``logs``, ``manifests``, ``review``,
``reports``, and ``analysis``.  Batch-owned manifests, review, reports, and
analysis live below ``batches/<batch_id>``; launcher logs live below
``logs/launcher/<batch_id>``.

Discovery deliberately remains read-only compatible with the former dated
category layout and the pre-refactor long-training roots.  No helper in this
module writes a new run to a legacy path.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
from typing import Any, Callable, Iterable, Mapping, Optional


EXPERIMENT_DATE_ENV = "MERC_EXPERIMENT_DATE"
DATE_PATTERN = re.compile(r"^\d{8}$")
IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
STATIC_OUTPUT_DIRECTORIES = frozenset({"environment", "reference", "cache"})
FUNCTIONAL_OUTPUT_DIRECTORIES = frozenset(
    {"runs", "logs", "manifests", "review", "reports", "analysis"}
)
# Kept as an import-compatible name for existing callers and tests.
DYNAMIC_OUTPUT_CATEGORIES = FUNCTIONAL_OUTPUT_DIRECTORIES
LEGACY_DYNAMIC_OUTPUT_DIRECTORIES = frozenset(
    {*FUNCTIONAL_OUTPUT_DIRECTORIES, "smoke", "audits"}
)
RESERVED_EXPERIMENT_GROUPS = frozenset(
    {*FUNCTIONAL_OUTPUT_DIRECTORIES, *STATIC_OUTPUT_DIRECTORIES}
)
MAX_INFERRED_GROUP_LENGTH = 36


@dataclass(frozen=True)
class ExperimentOutputPaths:
    """Resolved paths for one experiment group and optional run/batch."""

    output_base: Path
    experiment_date: str
    experiment_group: str
    experiment_root: Path
    runs_root: Path
    run_root: Optional[Path]
    logs_root: Path
    launcher_log_root: Path
    manifest_root: Path
    review_root: Path
    report_root: Path
    analysis_root: Path


def validate_experiment_date(value: Any) -> str:
    """Return a canonical ``YYYYMMDD`` date or raise ``ValueError``."""

    if not isinstance(value, str) or DATE_PATTERN.fullmatch(value) is None:
        raise ValueError(
            f"Invalid experiment date {value!r}; expected exactly eight digits (YYYYMMDD)."
        )
    try:
        datetime.strptime(value, "%Y%m%d")
    except ValueError as error:
        raise ValueError(
            f"Invalid experiment date {value!r}; it is not a valid calendar date."
        ) from error
    return value


def is_experiment_date(value: Any) -> bool:
    """Return whether *value* is both date-shaped and calendar-valid."""

    try:
        validate_experiment_date(value)
    except ValueError:
        return False
    return True


def _validate_identifier(value: Any, *, label: str) -> str:
    normalized = str(value).strip()
    if not normalized or IDENTIFIER_PATTERN.fullmatch(normalized) is None:
        raise ValueError(
            f"Invalid {label} {value!r}; use letters, digits, '.', '_' or '-' "
            "without path separators."
        )
    return normalized


def validate_experiment_group(value: Any) -> str:
    """Validate one stable, path-safe experiment-group identifier."""

    normalized = _validate_identifier(value, label="experiment_group")
    if normalized in RESERVED_EXPERIMENT_GROUPS:
        raise ValueError(
            f"Invalid experiment_group {normalized!r}; it conflicts with a fixed "
            "functional output directory."
        )
    return normalized


def validate_batch_id(value: Any) -> str:
    """Validate one path-safe batch identifier."""

    return _validate_identifier(value, label="batch_id")


def _configured_experiment_date(config: Optional[Mapping[str, Any]]) -> Optional[str]:
    if config is None:
        return None
    output = config.get("output", {})
    if isinstance(output, Mapping):
        value = output.get("experiment_date")
        if value is not None and str(value).strip():
            return str(value).strip()
    return None


def _local_date(now: Optional[datetime | date | Callable[[], datetime | date]]) -> str:
    value = now() if callable(now) else now
    if value is None:
        value = datetime.now()
    if not isinstance(value, (datetime, date)):
        raise TypeError("now must be a date, datetime, callable, or None")
    return value.strftime("%Y%m%d")


def resolve_experiment_date(
    cli_date: Optional[str] = None,
    config: Optional[Mapping[str, Any]] = None,
    env: Optional[Mapping[str, str]] = None,
    now: Optional[datetime | date | Callable[[], datetime | date]] = None,
    inferred_date: Optional[str] = None,
) -> str:
    """Resolve the date by CLI, config, environment, inference, then local date."""

    candidates = (
        cli_date,
        _configured_experiment_date(config),
        (os.environ if env is None else env).get(EXPERIMENT_DATE_ENV),
    )
    for candidate in candidates:
        if candidate is not None and str(candidate).strip():
            return validate_experiment_date(str(candidate).strip())
    if inferred_date is not None and str(inferred_date).strip():
        return validate_experiment_date(str(inferred_date).strip())
    return validate_experiment_date(_local_date(now))


def sanitize_run_name(value: str) -> str:
    """Return a path-safe run-name fragment."""

    normalized = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip())
    return normalized or "experiment"


def _bounded_inferred_group(value: str) -> str:
    """Bound inferred groups for Windows paths while preserving stable identity."""

    if len(value) <= MAX_INFERRED_GROUP_LENGTH:
        return value
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:8]
    prefix = value[: MAX_INFERRED_GROUP_LENGTH - len(digest) - 1].rstrip("_.-")
    return f"{prefix}_{digest}"


def _group_from_config_path(config_path: str | Path) -> Optional[str]:
    path = Path(config_path)
    parts = list(path.parts)
    lowered = [part.lower() for part in parts]
    try:
        index = lowered.index("configs")
    except ValueError:
        return None
    relative = parts[index + 1 :]
    if len(relative) < 2:
        return None
    parent_parts = relative[:-1]
    stem = Path(relative[-1]).stem.lower()
    group = sanitize_run_name("_".join(parent_parts)).lower()
    for role in ("smoke", "quick", "debug", "audit"):
        if role in stem and not group.endswith(f"_{role}"):
            group = f"{group}_{role}"
            break
    return validate_experiment_group(_bounded_inferred_group(group))


def resolve_experiment_group(
    cli_group: Optional[str] = None,
    config: Optional[Mapping[str, Any]] = None,
    config_path: Optional[str | Path] = None,
    default: Optional[str] = None,
) -> str:
    """Resolve a stable experiment group without deriving it from a run ID.

    Priority is an explicit CLI value, ``output.experiment_group``, top-level
    ``experiment_group``, ``project.experiment_group``, canonical config parent
    path, then an explicit caller default.  The final fallback uses a stable
    configured experiment name and prefixes it with ``single_``.
    """

    candidates: list[Any] = [cli_group]
    if config is not None:
        output = config.get("output", {})
        project = config.get("project", {})
        candidates.extend(
            [
                output.get("experiment_group") if isinstance(output, Mapping) else None,
                config.get("experiment_group"),
                project.get("experiment_group") if isinstance(project, Mapping) else None,
            ]
        )
    for candidate in candidates:
        if candidate is not None and str(candidate).strip():
            return validate_experiment_group(candidate)

    if config_path is not None:
        inferred = _group_from_config_path(config_path)
        if inferred is not None:
            return inferred

    if default is not None and str(default).strip():
        return validate_experiment_group(default)

    if config is not None:
        output = config.get("output", {})
        project = config.get("project", {})
        stable_name = (
            output.get("experiment_name") if isinstance(output, Mapping) else None
        ) or (
            project.get("experiment_name") if isinstance(project, Mapping) else None
        ) or config.get("run_name")
        if stable_name is not None and str(stable_name).strip():
            return validate_experiment_group(
                f"single_{sanitize_run_name(str(stable_name)).lower()}"
            )
    raise ValueError(
        "experiment_group is required for new output writes; set "
        "output.experiment_group or use a canonical config path."
    )


def resolve_day_output_root(
    experiment_date: str,
    output_root: str | Path = "outputs",
) -> Path:
    """Return ``<output_base>/<YYYYMMDD>`` without creating it."""

    return Path(output_root) / validate_experiment_date(experiment_date)


def resolve_experiment_root(
    experiment_date: str,
    experiment_group: str,
    output_base: str | Path = "outputs",
) -> Path:
    """Return the canonical experiment root without creating it."""

    return (
        resolve_day_output_root(experiment_date, output_base)
        / validate_experiment_group(experiment_group)
    )


def resolve_output_paths(
    *,
    output_base: str | Path = "outputs",
    experiment_date: str,
    experiment_group: str,
    run_id: Optional[str] = None,
    batch_id: Optional[str] = None,
) -> ExperimentOutputPaths:
    """Resolve all canonical roots requested by experiment-output owners."""

    base = Path(output_base)
    frozen_date = validate_experiment_date(experiment_date)
    group = validate_experiment_group(experiment_group)
    experiment_root = resolve_experiment_root(frozen_date, group, base)
    safe_run_id = None if run_id is None else _validate_identifier(run_id, label="run_id")
    safe_batch_id = (
        None if batch_id is None else validate_batch_id(batch_id)
    )
    runs_root = experiment_root / "runs"
    logs_root = experiment_root / "logs"
    manifest_root = experiment_root / "manifests"
    review_root = experiment_root / "review"
    report_root = experiment_root / "reports"
    analysis_root = experiment_root / "analysis"
    launcher_log_root = logs_root / "launcher"
    if safe_batch_id is not None:
        launcher_log_root /= safe_batch_id
        manifest_root = manifest_root / "batches" / safe_batch_id
        review_root = review_root / "batches" / safe_batch_id
        report_root = report_root / "batches" / safe_batch_id
        analysis_root = analysis_root / "batches" / safe_batch_id
    return ExperimentOutputPaths(
        output_base=base,
        experiment_date=frozen_date,
        experiment_group=group,
        experiment_root=experiment_root,
        runs_root=runs_root,
        run_root=None if safe_run_id is None else runs_root / safe_run_id,
        logs_root=logs_root,
        launcher_log_root=launcher_log_root,
        manifest_root=manifest_root,
        review_root=review_root,
        report_root=report_root,
        analysis_root=analysis_root,
    )


def resolve_output_category(
    category: str,
    experiment_date: str,
    output_root: str | Path = "outputs",
    *,
    experiment_group: str,
) -> Path:
    """Return one fixed functional directory under an experiment root."""

    normalized = str(category).strip()
    if normalized not in FUNCTIONAL_OUTPUT_DIRECTORIES:
        allowed = ", ".join(sorted(FUNCTIONAL_OUTPUT_DIRECTORIES))
        raise ValueError(f"Unsupported output category {category!r}; expected one of: {allowed}")
    return resolve_experiment_root(
        experiment_date, experiment_group, output_root
    ) / normalized


def _configured_output(config: Optional[Mapping[str, Any]]) -> Mapping[str, Any]:
    if config is None:
        return {}
    output = config.get("output", {})
    return output if isinstance(output, Mapping) else {}


def configured_output_root(
    config: Optional[Mapping[str, Any]],
    override: Optional[str | Path] = None,
    default: str | Path = "outputs",
) -> Path:
    """Resolve the logical output base across current and legacy schemas.

    ``output.output_base`` is authoritative for a resolved config whose
    ``output.root`` is the exact run root.
    """

    if override is not None:
        return Path(override)
    output = _configured_output(config)
    base = output.get("output_base")
    if base is not None and str(base).strip():
        return Path(str(base))
    root = output.get("root")
    if root is not None and str(root).strip():
        return Path(str(root))
    legacy_run_root = output.get("run_root")
    if legacy_run_root is not None and str(legacy_run_root).strip():
        path = Path(str(legacy_run_root))
        return path.parent if path.name == "runs" else path
    if config is not None:
        system = config.get("system", {})
        if isinstance(system, Mapping):
            system_root = system.get("output_dir")
            if system_root is not None and str(system_root).strip():
                return Path(str(system_root))
    return Path(default)


def configured_run_id(config: Optional[Mapping[str, Any]]) -> Optional[str]:
    """Return an explicitly fixed run ID, if one is configured."""

    value = _configured_output(config).get("run_id")
    if value is None or not str(value).strip():
        return None
    return _validate_identifier(value, label="run_id")


def _normalized_comparison_path(path: str | Path) -> str:
    value = Path(path)
    try:
        value = value.resolve()
    except OSError:
        value = value.absolute()
    return os.path.normcase(str(value))


def create_unique_run_dir(
    experiment_name: str,
    experiment_date: str,
    output_root: str | Path = "outputs",
    *,
    experiment_group: str,
    run_id: Optional[str] = None,
    configured_run_root: Optional[str | Path] = None,
    now: Optional[datetime | Callable[[], datetime]] = None,
    suffix_factory: Optional[Callable[[], str]] = None,
    resume_run_dir: Optional[str | Path] = None,
) -> Path:
    """Atomically allocate a canonical run directory.

    Fixed run IDs use ``mkdir(exist_ok=False)`` and therefore fail before any
    epoch when the target already exists.  No existing run is deleted, moved,
    or silently reused.
    """

    frozen_date = validate_experiment_date(experiment_date)
    group = validate_experiment_group(experiment_group)
    if resume_run_dir is not None:
        existing = Path(resume_run_dir)
        if not existing.is_dir():
            raise FileNotFoundError(f"Resume run directory not found: {existing}")
        return existing

    if run_id is not None:
        layout = resolve_output_paths(
            output_base=output_root,
            experiment_date=frozen_date,
            experiment_group=group,
            run_id=run_id,
        )
        assert layout.run_root is not None
        if configured_run_root is not None and (
            _normalized_comparison_path(configured_run_root)
            != _normalized_comparison_path(layout.run_root)
        ):
            raise ValueError(
                "Configured output.root does not match the canonical run root: "
                f"configured={configured_run_root}, expected={layout.run_root}"
            )
        layout.run_root.mkdir(parents=True, exist_ok=False)
        return layout.run_root

    return create_unique_category_dir(
        "runs",
        experiment_name,
        frozen_date,
        output_root,
        experiment_group=group,
        now=now,
        suffix_factory=suffix_factory,
    )


def allocate_configured_run(
    *,
    config: Mapping[str, Any],
    config_path: str | Path,
    experiment_name: str,
    experiment_date: str,
    output_base: str | Path,
    experiment_group: Optional[str] = None,
    resume_run_dir: Optional[str | Path] = None,
) -> ExperimentOutputPaths:
    """Create and resolve one run using a config's canonical output contract."""

    group = resolve_experiment_group(
        cli_group=experiment_group,
        config=config,
        config_path=config_path,
    )
    fixed_run_id = configured_run_id(config)
    configured_root = (
        _configured_output(config).get("root") if fixed_run_id is not None else None
    )
    run_dir = create_unique_run_dir(
        experiment_name,
        experiment_date,
        output_base,
        experiment_group=group,
        run_id=fixed_run_id,
        configured_run_root=configured_root,
        resume_run_dir=resume_run_dir,
    )
    return resolve_output_paths(
        output_base=output_base,
        experiment_date=experiment_date,
        experiment_group=group,
        run_id=run_dir.name,
    )


def create_unique_category_dir(
    category: str,
    name: str,
    experiment_date: str,
    output_root: str | Path = "outputs",
    *,
    experiment_group: str,
    now: Optional[datetime | Callable[[], datetime]] = None,
    suffix_factory: Optional[Callable[[], str]] = None,
) -> Path:
    """Atomically allocate a unique directory in one canonical category."""

    frozen_date = validate_experiment_date(experiment_date)
    category_root = resolve_output_category(
        category,
        frozen_date,
        output_root,
        experiment_group=experiment_group,
    )
    category_root.mkdir(parents=True, exist_ok=True)
    safe_name = sanitize_run_name(name)
    make_suffix = suffix_factory or (lambda: secrets.token_hex(3))

    for _ in range(100):
        timestamp_value = now() if callable(now) else now
        if timestamp_value is None:
            timestamp_value = datetime.now()
        if not isinstance(timestamp_value, datetime):
            raise TypeError("now must be a datetime, callable, or None")
        directory_id = (
            f"{safe_name}_{frozen_date}_{timestamp_value.strftime('%H%M%S')}_"
            f"{sanitize_run_name(make_suffix())}"
        )
        destination = category_root / directory_id
        try:
            destination.mkdir(parents=False, exist_ok=False)
        except FileExistsError:
            continue
        return destination
    raise RuntimeError("Unable to allocate a unique output directory after 100 attempts")


def _deduplicate_paths(paths: Iterable[Path]) -> list[Path]:
    result: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        try:
            key = os.path.normcase(str(path.resolve()))
        except OSError:
            key = os.path.normcase(str(path.absolute()))
        if key not in seen:
            seen.add(key)
            result.append(path)
    return result


def _children(path: Path) -> list[Path]:
    if not path.is_dir():
        return []
    return sorted((child for child in path.iterdir() if child.is_dir()))


def _discover_category_directories(output_root: str | Path, category: str) -> list[Path]:
    """Discover canonical paths first, followed by supported legacy layouts."""

    root = Path(output_root)
    discovered: list[Path] = []
    if root.is_dir():
        dated_roots = sorted(
            (
                child
                for child in root.iterdir()
                if child.is_dir() and is_experiment_date(child.name)
            ),
            key=lambda item: item.name,
            reverse=True,
        )
        for dated_root in dated_roots:
            # Canonical: outputs/<date>/<group>/<category>/*
            for group_root in _children(dated_root):
                if group_root.name in LEGACY_DYNAMIC_OUTPUT_DIRECTORIES:
                    continue
                discovered.extend(_children(group_root / category))
            # Read-only compatibility: outputs/<date>/<category>/*
            discovered.extend(_children(dated_root / category))

        # Read-only compatibility: outputs/<category>/*
        discovered.extend(_children(root / category))
        if category == "runs":
            # Read-only compatibility for the pre-refactor long-training runs.
            discovered.extend(_children(root / "long_training" / "primary"))
            discovered.extend(_children(root / "long_training" / "multi_seed"))
    return _deduplicate_paths(discovered)


def discover_run_directories(output_root: str | Path = "outputs") -> list[Path]:
    """Discover canonical runs, then all explicitly supported legacy runs."""

    return _discover_category_directories(output_root, "runs")


def discover_analysis_directories(output_root: str | Path = "outputs") -> list[Path]:
    """Discover canonical analysis directories, then legacy ones."""

    return _discover_category_directories(output_root, "analysis")


def discover_launcher_log_directories(
    output_root: str | Path = "outputs",
) -> list[Path]:
    """Discover canonical launcher batches, then legacy launcher-log batches."""

    root = Path(output_root)
    discovered: list[Path] = []
    if root.is_dir():
        dated_roots = sorted(
            (
                child
                for child in root.iterdir()
                if child.is_dir() and is_experiment_date(child.name)
            ),
            key=lambda item: item.name,
            reverse=True,
        )
        for dated_root in dated_roots:
            for group_root in _children(dated_root):
                if group_root.name in LEGACY_DYNAMIC_OUTPUT_DIRECTORIES:
                    continue
                discovered.extend(
                    _children(group_root / "logs" / "launcher")
                )
            # Read-only compatibility for an intermediate dated layout.
            discovered.extend(_children(dated_root / "logs" / "launcher"))
        # Read-only compatibility: outputs/launcher_logs/<batch_id>.
        discovered.extend(_children(root / "launcher_logs"))
    return _deduplicate_paths(discovered)


def find_analysis_artifact(
    relative_path: str | Path,
    output_root: str | Path = "outputs",
) -> Path:
    """Find an analysis artifact in canonical and read-only legacy locations."""

    root = Path(output_root)
    relative = Path(relative_path)
    if relative.is_absolute():
        if not relative.exists():
            raise FileNotFoundError(f"Analysis artifact not found: {relative}")
        return relative
    candidates: list[Path] = []
    if root.is_dir():
        for dated_root in sorted(
            (
                child
                for child in root.iterdir()
                if child.is_dir() and is_experiment_date(child.name)
            ),
            key=lambda item: item.name,
            reverse=True,
        ):
            for group_root in _children(dated_root):
                if group_root.name in LEGACY_DYNAMIC_OUTPUT_DIRECTORIES:
                    continue
                candidates.append(group_root / "analysis" / relative)
            candidates.append(dated_root / "analysis" / relative)
    candidates.extend([root / "analysis" / relative, root / relative])
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Analysis artifact not found in canonical or legacy outputs: "
        f"{relative}"
    )


def find_run_directory(run_id: str, output_root: str | Path = "outputs") -> Path:
    """Resolve one run ID across canonical and read-only legacy layouts."""

    matches = [path for path in discover_run_directories(output_root) if path.name == run_id]
    if not matches:
        raise FileNotFoundError(f"Run ID not found in canonical or legacy outputs: {run_id}")
    if len(matches) > 1:
        locations = ", ".join(str(path) for path in matches)
        raise RuntimeError(f"Run ID is ambiguous across output directories: {locations}")
    return matches[0]


def infer_experiment_date_from_run(run_dir: str | Path) -> Optional[str]:
    """Infer a frozen experiment date from metadata, layout, or legacy run ID."""

    path = Path(run_dir)
    metadata_path = path / "run_metadata.json"
    if metadata_path.is_file():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            metadata = {}
        if isinstance(metadata, Mapping):
            value = metadata.get("experiment_date")
            if is_experiment_date(value):
                return str(value)

    for part in reversed(path.parts):
        if is_experiment_date(part):
            return part
    match = re.search(r"(?:^|_)(\d{8})(?:_|$)", path.name)
    if match and is_experiment_date(match.group(1)):
        return match.group(1)
    return None


def infer_experiment_group_from_run(run_dir: str | Path) -> Optional[str]:
    """Infer an experiment group from run metadata or canonical path shape."""

    path = Path(run_dir)
    metadata_path = path / "run_metadata.json"
    if metadata_path.is_file():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            metadata = {}
        if isinstance(metadata, Mapping):
            value = metadata.get("experiment_group")
            if value is not None:
                try:
                    return validate_experiment_group(value)
                except ValueError:
                    pass
    parts = path.parts
    for index, part in enumerate(parts):
        if (
            is_experiment_date(part)
            and index + 2 < len(parts)
            and parts[index + 2] == "runs"
        ):
            try:
                return validate_experiment_group(parts[index + 1])
            except ValueError:
                return None
    return None


def resolved_output_metadata(
    *,
    output_root: str | Path,
    experiment_date: str,
    experiment_group: str,
    run_id: Optional[str] = None,
    batch_id: Optional[str] = None,
) -> dict[str, str]:
    """Build canonical resolved path fields for config and run metadata."""

    layout = resolve_output_paths(
        output_base=output_root,
        experiment_date=experiment_date,
        experiment_group=experiment_group,
        run_id=run_id,
        batch_id=batch_id,
    )
    return {
        "experiment_date": layout.experiment_date,
        "experiment_group": layout.experiment_group,
        "output_base": str(layout.output_base),
        "day_output_root": str(layout.output_base / layout.experiment_date),
        "experiment_root": str(layout.experiment_root),
        "run_root": "" if layout.run_root is None else str(layout.run_root),
        "launcher_log_root": str(layout.launcher_log_root),
        "manifest_root": str(layout.manifest_root),
        "review_root": str(layout.review_root),
        "report_root": str(layout.report_root),
        "analysis_root": str(layout.analysis_root),
    }


__all__ = [
    "DATE_PATTERN",
    "DYNAMIC_OUTPUT_CATEGORIES",
    "EXPERIMENT_DATE_ENV",
    "ExperimentOutputPaths",
    "FUNCTIONAL_OUTPUT_DIRECTORIES",
    "STATIC_OUTPUT_DIRECTORIES",
    "allocate_configured_run",
    "configured_output_root",
    "configured_run_id",
    "create_unique_category_dir",
    "create_unique_run_dir",
    "discover_analysis_directories",
    "discover_launcher_log_directories",
    "discover_run_directories",
    "find_analysis_artifact",
    "find_run_directory",
    "infer_experiment_date_from_run",
    "infer_experiment_group_from_run",
    "is_experiment_date",
    "resolve_day_output_root",
    "resolve_experiment_date",
    "resolve_experiment_group",
    "resolve_experiment_root",
    "resolve_output_category",
    "resolve_output_paths",
    "resolved_output_metadata",
    "sanitize_run_name",
    "validate_batch_id",
    "validate_experiment_date",
    "validate_experiment_group",
]

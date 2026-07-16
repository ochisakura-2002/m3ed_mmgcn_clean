"""Central path policy for date-organized experiment outputs.

New runtime artifacts live below ``<output_root>/<YYYYMMDD>/<category>``.
Discovery deliberately remains backward compatible with the former
``<output_root>/<category>`` layout.
"""

from __future__ import annotations

from datetime import date, datetime
import json
import os
from pathlib import Path
import re
import secrets
from typing import Any, Callable, Iterable, Mapping, Optional


EXPERIMENT_DATE_ENV = "MERC_EXPERIMENT_DATE"
DATE_PATTERN = re.compile(r"^\d{8}$")
STATIC_OUTPUT_DIRECTORIES = frozenset({"environment", "reference", "cache"})
DYNAMIC_OUTPUT_CATEGORIES = frozenset(
    {
        "runs",
        "logs",
        "analysis",
        "reports",
        "manifests",
        "smoke",
        "audits",
        "review",
    }
)


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
    """Resolve and validate the experiment date using the required priority.

    Priority: explicit CLI value, ``output.experiment_date``,
    ``MERC_EXPERIMENT_DATE``, then the runtime machine's local date.
    """

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


def resolve_day_output_root(
    experiment_date: str,
    output_root: str | Path = "outputs",
) -> Path:
    """Return ``<output_root>/<YYYYMMDD>`` without creating it."""

    return Path(output_root) / validate_experiment_date(experiment_date)


def resolve_output_category(
    category: str,
    experiment_date: str,
    output_root: str | Path = "outputs",
) -> Path:
    """Return one dynamic category path below a validated date root."""

    normalized = str(category).strip()
    if normalized not in DYNAMIC_OUTPUT_CATEGORIES:
        allowed = ", ".join(sorted(DYNAMIC_OUTPUT_CATEGORIES))
        raise ValueError(f"Unsupported output category {category!r}; expected one of: {allowed}")
    return resolve_day_output_root(experiment_date, output_root) / normalized


def configured_output_root(
    config: Optional[Mapping[str, Any]],
    override: Optional[str | Path] = None,
    default: str | Path = "outputs",
) -> Path:
    """Resolve the logical output root across supported config schemas."""

    if override is not None:
        return Path(override)
    if config is not None:
        output = config.get("output", {})
        if isinstance(output, Mapping):
            root = output.get("root")
            if root is not None and str(root).strip():
                return Path(str(root))
            legacy_run_root = output.get("run_root")
            if legacy_run_root is not None and str(legacy_run_root).strip():
                path = Path(str(legacy_run_root))
                return path.parent if path.name == "runs" else path
        system = config.get("system", {})
        if isinstance(system, Mapping):
            root = system.get("output_dir")
            if root is not None and str(root).strip():
                return Path(str(root))
    return Path(default)


def sanitize_run_name(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip())
    return normalized or "experiment"


def create_unique_run_dir(
    experiment_name: str,
    experiment_date: str,
    output_root: str | Path = "outputs",
    *,
    now: Optional[datetime | Callable[[], datetime]] = None,
    suffix_factory: Optional[Callable[[], str]] = None,
    resume_run_dir: Optional[str | Path] = None,
) -> Path:
    """Atomically create a collision-resistant run directory.

    A supplied ``resume_run_dir`` is returned unchanged after validation, so a
    resumed run never moves merely because the current date changed.
    """

    frozen_date = validate_experiment_date(experiment_date)
    if resume_run_dir is not None:
        existing = Path(resume_run_dir)
        if not existing.is_dir():
            raise FileNotFoundError(f"Resume run directory not found: {existing}")
        return existing

    return create_unique_category_dir(
        "runs",
        experiment_name,
        frozen_date,
        output_root,
        now=now,
        suffix_factory=suffix_factory,
    )


def create_unique_category_dir(
    category: str,
    name: str,
    experiment_date: str,
    output_root: str | Path = "outputs",
    *,
    now: Optional[datetime | Callable[[], datetime]] = None,
    suffix_factory: Optional[Callable[[], str]] = None,
) -> Path:
    """Atomically allocate a unique directory within a dated category."""

    frozen_date = validate_experiment_date(experiment_date)
    category_root = resolve_output_category(category, frozen_date, output_root)
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


def _discover_category_directories(output_root: str | Path, category: str) -> list[Path]:
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
            category_root = dated_root / category
            if category_root.is_dir():
                discovered.extend(sorted((p for p in category_root.iterdir() if p.is_dir())))

        legacy_root = root / category
        if legacy_root.is_dir():
            discovered.extend(sorted((p for p in legacy_root.iterdir() if p.is_dir())))
    return _deduplicate_paths(discovered)


def discover_run_directories(output_root: str | Path = "outputs") -> list[Path]:
    """Discover new-layout runs first, then legacy-layout runs."""

    return _discover_category_directories(output_root, "runs")


def discover_analysis_directories(output_root: str | Path = "outputs") -> list[Path]:
    """Discover new-layout analysis directories first, then legacy ones."""

    return _discover_category_directories(output_root, "analysis")


def find_analysis_artifact(
    relative_path: str | Path,
    output_root: str | Path = "outputs",
) -> Path:
    """Find an analysis artifact in new dated locations, then legacy ones."""

    root = Path(output_root)
    relative = Path(relative_path)
    if relative.is_absolute():
        if not relative.exists():
            raise FileNotFoundError(f"Analysis artifact not found: {relative}")
        return relative
    dated_candidates: list[Path] = []
    if root.is_dir():
        for child in sorted(root.iterdir(), key=lambda item: item.name, reverse=True):
            if child.is_dir() and is_experiment_date(child.name):
                dated_candidates.append(child / "analysis" / relative)
    candidates = dated_candidates + [root / "analysis" / relative, root / relative]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Analysis artifact not found in date-organized or legacy outputs: "
        f"{relative}"
    )


def find_run_directory(run_id: str, output_root: str | Path = "outputs") -> Path:
    """Resolve one run ID across the new and legacy layouts."""

    matches = [path for path in discover_run_directories(output_root) if path.name == run_id]
    if not matches:
        raise FileNotFoundError(f"Run ID not found in date-organized or legacy outputs: {run_id}")
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


def resolved_output_metadata(
    *,
    output_root: str | Path,
    experiment_date: str,
    run_dir: Optional[str | Path] = None,
    pipeline_name: str = "pipeline",
) -> dict[str, str]:
    """Build canonical resolved path fields for config and run metadata."""

    root = Path(output_root)
    frozen_date = validate_experiment_date(experiment_date)
    day_root = resolve_day_output_root(frozen_date, root)
    return {
        "experiment_date": frozen_date,
        "output_root": str(root),
        "day_output_root": str(day_root),
        "run_dir": "" if run_dir is None else str(Path(run_dir)),
        "log_dir": str(resolve_output_category("logs", frozen_date, root) / sanitize_run_name(pipeline_name)),
        "analysis_dir": str(resolve_output_category("analysis", frozen_date, root)),
        "manifest_dir": str(resolve_output_category("manifests", frozen_date, root) / sanitize_run_name(pipeline_name)),
    }


__all__ = [
    "DATE_PATTERN",
    "DYNAMIC_OUTPUT_CATEGORIES",
    "EXPERIMENT_DATE_ENV",
    "STATIC_OUTPUT_DIRECTORIES",
    "configured_output_root",
    "create_unique_category_dir",
    "create_unique_run_dir",
    "discover_analysis_directories",
    "discover_run_directories",
    "find_run_directory",
    "find_analysis_artifact",
    "infer_experiment_date_from_run",
    "is_experiment_date",
    "resolve_day_output_root",
    "resolve_experiment_date",
    "resolve_output_category",
    "resolved_output_metadata",
    "sanitize_run_name",
    "validate_experiment_date",
]

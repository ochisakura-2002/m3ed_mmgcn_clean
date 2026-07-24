"""Read-only validation for one executed config migration batch."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import subprocess
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence

import yaml


MOVE_COLUMNS = (
    "old_path",
    "new_path",
    "model",
    "implementation",
    "dataset",
    "context_mode",
    "feature_set",
    "purpose",
    "scope",
    "is_smoke",
    "pre_move_sha256",
    "post_move_sha256",
    "yaml_content_changed",
    "approved_changed_keys",
    "active_reference_count",
    "test_reference_count",
    "doc_reference_count",
    "status",
    "notes",
)
BATCH3_MOVE_COLUMNS = (
    "old_path",
    "new_path",
    "model",
    "implementation",
    "dataset",
    "context_mode",
    "feature_set",
    "purpose",
    "scope",
    "is_smoke",
    "is_formal",
    "is_paper_aligned",
    "is_official",
    "pre_move_sha256",
    "post_move_sha256",
    "yaml_content_changed",
    "approved_changed_keys",
    "active_reference_count",
    "test_reference_count",
    "doc_reference_count",
    "yaml_reference_count",
    "status",
    "notes",
)
REFERENCE_COLUMNS = (
    "old_config_path",
    "source_file",
    "source_line",
    "reference_type",
    "historical_reference",
    "updated",
    "new_config_path",
    "remaining_reference_allowed",
    "notes",
)
BATCH3_REFERENCE_COLUMNS = (
    "old_config_path",
    "new_config_path",
    "source_file",
    "source_line",
    "reference_type",
    "historical_reference",
    "requires_update",
    "updated",
    "remaining_reference_allowed",
    "notes",
)
SEMANTIC_DIFF_COLUMNS = (
    "old_path",
    "new_path",
    "byte_identical",
    "semantic_identical",
    "changed_keys",
    "change_allowed",
    "status",
    "notes",
)
BATCH3_SEMANTIC_DIFF_COLUMNS = (
    "old_path",
    "new_path",
    "implementation",
    "byte_identical",
    "semantic_identical",
    "changed_keys",
    "change_allowed",
    "status",
    "notes",
)
EXPECTED_BATCH_COUNTS = {1: 16, 2: 17, 3: 17, 4: 10}
EXPECTED_PREVIEW_PATH_CHANGES = {2: 4, 3: 4, 4: 0}
EXPECTED_IMPLEMENTATION_COUNTS = {
    2: {"unified": 13, "paper_aligned": 4},
    3: {"unified": 13, "paper_aligned": 4},
    4: {"unified": 6, "paper_aligned": 4},
}
DEFAULT_TRACKED_YAML_COUNT = 183
SNAPSHOT_IDENTITY_FIELDS = {
    "model_classification": "model",
    "implementation": "implementation",
    "dataset_classification": "dataset",
    "context_mode": "context_mode",
    "feature_set": "feature_set",
    "purpose": "purpose",
}
YES_NO = {"YES", "NO"}
CONFIG_PATH_KEYS = {
    "base_config",
    "config",
    "config_path",
    "eval_config",
    "parent_config",
    "pipeline_config",
    "source_config",
    "train_config",
}
SCRIPT_PATH_KEYS = {"entrypoint", "script"}
APPROVED_PATH_KEYS = CONFIG_PATH_KEYS | SCRIPT_PATH_KEYS
GROUP_MEETING_PATHS = {
    "scripts/analyze/export_group_meeting_baseline_report.py",
    "tests/analyze/test_group_meeting_baseline_report.py",
}
WINDOWS_ABSOLUTE = re.compile(r"^[A-Za-z]:[\\/]")
TEXT_SUFFIXES = {".csv", ".json", ".md", ".py", ".yaml", ".yml"}


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or ()), [dict(row) for row in reader]


def _missing_columns(actual: Sequence[str], required: Iterable[str]) -> list[str]:
    actual_set = set(actual)
    return [column for column in required if column not in actual_set]


def _run_git(repo_root: Path, *args: str) -> list[str]:
    result = subprocess.run(
        ["git", *args],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return [line.strip().replace("\\", "/") for line in result.stdout.splitlines()]


def _tracked_yaml_paths(repo_root: Path) -> set[str]:
    return {
        path
        for path in _run_git(repo_root, "ls-files", "configs")
        if PurePosixPath(path).suffix.lower() in {".yaml", ".yml"}
    }


def _tracked_text(repo_root: Path) -> dict[str, str]:
    paths = set(_run_git(repo_root, "ls-files"))
    paths.update(
        _run_git(
            repo_root,
            "diff",
            "--cached",
            "--name-only",
            "--diff-filter=ACMR",
        )
    )
    result: dict[str, str] = {}
    for relative in sorted(paths):
        if relative in GROUP_MEETING_PATHS:
            continue
        path = repo_root / relative
        if (
            path.is_file()
            and PurePosixPath(relative).suffix.lower() in TEXT_SUFFIXES
            and PurePosixPath(relative).parts[0]
            not in {"data", "outputs", "third_party", "tmp"}
        ):
            try:
                result[relative] = path.read_text(encoding="utf-8-sig")
            except UnicodeDecodeError:
                continue
    return result


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_text_sha256(path: Path) -> str:
    # Git stores these text YAML blobs with LF while core.autocrlf may materialize
    # CRLF (or mixed EOLs) in the Windows worktree. Compare canonical Git text
    # bytes so line-ending smudging is not reported as a YAML content change.
    canonical = path.read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha256(canonical).hexdigest()


def _is_relative_repo_path(value: str) -> bool:
    if not value or "\\" in value:
        return False
    if value.startswith("/") or WINDOWS_ABSOLUTE.match(value):
        return False
    return not any(part in {".", ".."} for part in PurePosixPath(value).parts)


def _batch_number(value: str) -> int | None:
    match = re.fullmatch(r"Batch ([1-7])", value)
    return int(match.group(1)) if match else None


def _walk_scalars(value: Any, prefix: str = "") -> Iterable[tuple[str, Any]]:
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{prefix}.{key}" if prefix else str(key)
            yield from _walk_scalars(child, child_path)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _walk_scalars(child, f"{prefix}[{index}]")
    else:
        yield prefix or "<root>", value


def _value_at_path(value: Any, path: str) -> Any:
    current = value
    for key, index in re.findall(r"([^.[]+)|\[(\d+)\]", path):
        if key:
            if not isinstance(current, dict) or key not in current:
                raise KeyError(path)
            current = current[key]
        else:
            if not isinstance(current, list):
                raise KeyError(path)
            current = current[int(index)]
    return current


def _git_index_text(repo_root: Path, relative: str) -> str | None:
    result = subprocess.run(
        ["git", "show", f":{relative}"],
        cwd=repo_root,
        check=False,
        capture_output=True,
    )
    if result.returncode != 0:
        return None
    try:
        return result.stdout.decode("utf-8-sig")
    except UnicodeDecodeError:
        return None


def _git_head_bytes(repo_root: Path, relative: str) -> bytes | None:
    return _git_commit_bytes(repo_root, "HEAD", relative)


def _git_commit_bytes(
    repo_root: Path,
    commit: str,
    relative: str,
) -> bytes | None:
    result = subprocess.run(
        ["git", "show", f"{commit}:{relative}"],
        cwd=repo_root,
        check=False,
        capture_output=True,
    )
    return result.stdout if result.returncode == 0 else None


def _git_head_commit(repo_root: Path) -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _validate_path_changes(
    repo_root: Path,
    before_yaml: Any,
    after_yaml: Any,
    changed_paths: set[str],
    plan_by_old: Mapping[str, Mapping[str, str]],
    plan_by_candidate: Mapping[str, Mapping[str, str]],
    tracked: set[str],
) -> list[str]:
    """Validate each changed config/script path against its concrete target."""

    errors: list[str] = []
    for path in sorted(changed_paths):
        leaf = _path_leaf(path)
        if leaf not in APPROVED_PATH_KEYS:
            errors.append(f"{path}: key is not an approved path key")
            continue
        try:
            before_value = _value_at_path(before_yaml, path)
            after_value = _value_at_path(after_yaml, path)
        except (KeyError, IndexError, ValueError):
            errors.append(f"{path}: changed path cannot be resolved")
            continue
        if not isinstance(before_value, str) or not isinstance(after_value, str):
            errors.append(f"{path}: path values must be strings")
            continue
        if not _is_relative_repo_path(before_value):
            errors.append(f"{path}: before value is not a safe repository path")
        if not _is_relative_repo_path(after_value):
            errors.append(f"{path}: after value is not a safe repository path")
            continue

        if leaf in CONFIG_PATH_KEYS:
            if not before_value.startswith("configs/"):
                errors.append(f"{path}: before value must be under configs/")
            if not after_value.startswith("configs/"):
                errors.append(f"{path}: after value must be under configs/")
                continue
            referenced_plan = plan_by_old.get(before_value)
            if referenced_plan is None:
                errors.append(
                    f"{path}: before value is not an old path in this migration plan: "
                    f"{before_value}"
                )
                continue
            expected = referenced_plan["candidate_new_path"]
            if after_value != expected:
                errors.append(
                    f"{path}: after value does not match the planned candidate: "
                    f"{after_value} != {expected}"
                )
                continue
            target_plan = plan_by_candidate.get(after_value)
            if target_plan is None:
                errors.append(
                    f"{path}: after value is not a candidate in this migration batch"
                )
                continue
            if target_plan.get("old_path") != before_value:
                errors.append(
                    f"{path}: after value belongs to a different migration mapping"
                )
                continue
            for field in (
                "model",
                "implementation",
                "dataset",
                "context_mode",
                "feature_set",
                "purpose",
            ):
                if target_plan.get(field, "") != referenced_plan.get(field, ""):
                    errors.append(
                        f"{path}: migration identity mismatch on {field}"
                    )
            if after_value not in tracked and not (repo_root / after_value).is_file():
                errors.append(f"{path}: planned candidate does not exist: {after_value}")
        else:
            if not after_value.startswith("scripts/"):
                errors.append(f"{path}: script target must be under scripts/")
            elif not (repo_root / after_value).is_file():
                errors.append(f"{path}: canonical script target does not exist")
    return errors


def preview_batch_migration(
    repo_root: Path,
    batch: int,
    plan_path: Path,
    *,
    strict: bool = False,
    tracked_yaml_override: set[str] | None = None,
    working_text_override: Mapping[str, str] | None = None,
    index_text_override: Mapping[str, str] | None = None,
    expected_path_changes: int | None = None,
) -> tuple[dict[str, int], list[str]]:
    """Recompute preflight path changes and actual YAML state without artifacts."""

    repo_root = repo_root.resolve()
    plan_columns, plan = _read_csv(plan_path)
    errors: list[str] = []
    for column in _missing_columns(
        plan_columns,
        ("old_path", "candidate_new_path", "migration_batch"),
    ):
        errors.append(f"plan is missing column: {column}")
    if errors:
        return {}, errors

    batch_label = f"Batch {batch}"
    batch_plan = [row for row in plan if row["migration_batch"] == batch_label]
    expected_count = EXPECTED_BATCH_COUNTS.get(batch)
    if strict and expected_count is not None and len(batch_plan) != expected_count:
        errors.append(
            f"{batch_label} must contain {expected_count} rows; found {len(batch_plan)}"
        )
    plan_by_old = {row["old_path"]: row for row in batch_plan}
    plan_by_candidate = {row["candidate_new_path"]: row for row in batch_plan}
    if len(plan_by_old) != len(batch_plan):
        errors.append(f"{batch_label} plan contains duplicate old_path values")
    if len(plan_by_candidate) != len(batch_plan):
        errors.append(f"{batch_label} plan contains duplicate candidate_new_path values")

    tracked = (
        set(tracked_yaml_override)
        if tracked_yaml_override is not None
        else _tracked_yaml_paths(repo_root)
    )
    working_override = (
        dict(working_text_override) if working_text_override is not None else None
    )
    index_override = (
        dict(index_text_override) if index_text_override is not None else None
    )
    preview_changes: list[tuple[str, str, str]] = []
    real_modifications = 0
    actual_approved = 0
    actual_unapproved = 0

    for row in batch_plan:
        old_path = row["old_path"]
        candidate = row["candidate_new_path"]
        current_path = old_path if old_path in tracked else candidate
        if working_override is not None:
            working_text = working_override.get(current_path)
        else:
            disk = repo_root / current_path
            working_text = (
                disk.read_text(encoding="utf-8-sig") if disk.is_file() else None
            )
        if working_text is None:
            errors.append(f"{current_path}: current YAML is missing for preview")
            continue
        try:
            working_yaml = yaml.safe_load(working_text)
        except yaml.YAMLError as exc:
            errors.append(f"{current_path}: YAML parse failed during preview: {exc}")
            continue

        for path, value in _walk_scalars(working_yaml):
            if (
                _path_leaf(path) in CONFIG_PATH_KEYS
                and isinstance(value, str)
                and value in plan_by_old
            ):
                planned_value = plan_by_old[value]["candidate_new_path"]
                if value != planned_value:
                    preview_changes.append((current_path, path, planned_value))

        if index_override is not None:
            index_text = index_override.get(current_path)
        else:
            index_text = _git_index_text(repo_root, current_path)
        if index_text is None:
            errors.append(f"{current_path}: Git index content is unavailable")
            continue
        if working_text.encode("utf-8") == index_text.encode("utf-8"):
            continue
        real_modifications += 1
        try:
            index_yaml = yaml.safe_load(index_text)
        except yaml.YAMLError as exc:
            errors.append(f"{current_path}: Git index YAML parse failed: {exc}")
            actual_unapproved += 1
            continue
        changes = _semantic_changes(index_yaml, working_yaml)
        validation_errors = _validate_path_changes(
            repo_root,
            index_yaml,
            working_yaml,
            changes,
            plan_by_old,
            plan_by_candidate,
            tracked,
        )
        if validation_errors:
            actual_unapproved += 1
        else:
            actual_approved += 1

    metrics = {
        "PREVIEW_PATH_CHANGES_REQUIRED": len(preview_changes),
        "ACTUAL_FILE_CHANGES": real_modifications,
        "APPROVED_ACTUAL_CHANGES": actual_approved,
        "UNAPPROVED_ACTUAL_CHANGES": actual_unapproved,
        "ACTUAL_YAML_CONTENT_MODIFICATIONS": real_modifications,
        "ACTUAL_UNAPPROVED_SEMANTIC_CHANGES": actual_unapproved,
    }
    expected_path_changes = (
        expected_path_changes
        if expected_path_changes is not None
        else EXPECTED_PREVIEW_PATH_CHANGES.get(batch)
    )
    if (
        strict
        and expected_path_changes is not None
        and metrics["PREVIEW_PATH_CHANGES_REQUIRED"] != expected_path_changes
    ):
        errors.append(
            "preview path-change count mismatch: "
            f"{metrics['PREVIEW_PATH_CHANGES_REQUIRED']} != {expected_path_changes}"
        )
    if strict and metrics["ACTUAL_FILE_CHANGES"] != 0:
        errors.append("preview found real Batch YAML content modifications")
    if strict and metrics["UNAPPROVED_ACTUAL_CHANGES"] != 0:
        errors.append("preview found actual unapproved semantic changes")
    return metrics, errors


def _load_snapshot(
    path: Path,
) -> tuple[dict[str, Any], dict[str, Any], list[str]]:
    errors: list[str] = []
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        return {}, {}, [f"cannot load snapshot {path}: {exc}"]
    if not isinstance(payload, dict) or not isinstance(payload.get("entries"), list):
        errors.append(f"snapshot must contain an entries list: {path}")
        return payload if isinstance(payload, dict) else {}, {}, errors
    entries: dict[str, Any] = {}
    for index, entry in enumerate(payload["entries"], start=1):
        if not isinstance(entry, dict):
            errors.append(f"snapshot entry {index} is not a mapping: {path}")
            continue
        key = str(entry.get("old_path", ""))
        if not key:
            errors.append(f"snapshot entry {index} lacks old_path: {path}")
        elif key in entries:
            errors.append(f"snapshot duplicates old_path {key}: {path}")
        else:
            entries[key] = entry
    return payload, entries, errors


def _semantic_changes(left: Any, right: Any, prefix: str = "") -> set[str]:
    if type(left) is not type(right):
        return {prefix or "<root>"}
    if isinstance(left, dict):
        changes: set[str] = set()
        for key in set(left) | set(right):
            child = f"{prefix}.{key}" if prefix else str(key)
            if key not in left or key not in right:
                changes.add(child)
            else:
                changes.update(_semantic_changes(left[key], right[key], child))
        return changes
    if isinstance(left, list):
        changes = set()
        if len(left) != len(right):
            changes.add(prefix or "<root>")
        for index, (left_item, right_item) in enumerate(zip(left, right)):
            changes.update(
                _semantic_changes(left_item, right_item, f"{prefix}[{index}]")
            )
        return changes
    return set() if left == right else {prefix or "<root>"}


def _semantic_change_values(
    left: Any,
    right: Any,
    prefix: str = "",
) -> dict[str, tuple[Any, Any]]:
    if type(left) is not type(right):
        return {prefix or "<root>": (left, right)}
    if isinstance(left, dict):
        changes: dict[str, tuple[Any, Any]] = {}
        for key in set(left) | set(right):
            child = f"{prefix}.{key}" if prefix else str(key)
            if key not in left:
                changes[child] = (None, right[key])
            elif key not in right:
                changes[child] = (left[key], None)
            else:
                changes.update(_semantic_change_values(left[key], right[key], child))
        return changes
    if isinstance(left, list):
        changes = {}
        if len(left) != len(right):
            changes[prefix or "<root>"] = (left, right)
            return changes
        for index, (left_item, right_item) in enumerate(zip(left, right)):
            changes.update(
                _semantic_change_values(
                    left_item,
                    right_item,
                    f"{prefix}[{index}]",
                )
            )
        return changes
    return {} if left == right else {prefix or "<root>": (left, right)}


def _split_keys(value: str) -> set[str]:
    return {item.strip() for item in value.split("|") if item.strip()}


def _path_leaf(path: str) -> str:
    return path.rsplit(".", 1)[-1].split("[", 1)[0]


def _is_audit_record(path: str) -> bool:
    return path.startswith("docs/refactors/CONFIG_")


def _documented_later_reference_changes(
    repo_root: Path,
    batch: int,
    source_file: str,
) -> set[tuple[str, str]]:
    """Return active reference rewrites recorded by completed later batches."""

    changes: set[tuple[str, str]] = set()
    artifacts_dir = repo_root / "docs/refactors"
    for later_batch in range(batch + 1, 8):
        path = artifacts_dir / f"CONFIG_BATCH{later_batch}_REFERENCE_AUDIT.csv"
        if not path.is_file():
            continue
        columns, rows = _read_csv(path)
        required = {
            "old_config_path",
            "new_config_path",
            "source_file",
            "historical_reference",
            "updated",
        }
        if not required.issubset(columns):
            continue
        for row in rows:
            if (
                row["source_file"] == source_file
                and row["historical_reference"] == "NO"
                and row["updated"] == "YES"
            ):
                changes.add(
                    (row["old_config_path"], row["new_config_path"])
                )
    return changes


def _is_documented_later_reference_drift(
    repo_root: Path,
    batch: int,
    source_file: str,
    snapshot_yaml: Any,
    disk_yaml: Any,
) -> bool:
    changes = _semantic_change_values(snapshot_yaml, disk_yaml)
    if not changes:
        return False
    documented = _documented_later_reference_changes(
        repo_root,
        batch,
        source_file,
    )
    return bool(documented) and all(
        _path_leaf(path) in APPROVED_PATH_KEYS
        and isinstance(old_value, str)
        and isinstance(new_value, str)
        and (old_value, new_value) in documented
        for path, (old_value, new_value) in changes.items()
    )


def _phase4a_frozen_historical_pairs(
    repo_root: Path,
    plan: Sequence[Mapping[str, str]],
) -> set[tuple[str, str]]:
    """Resolve Phase 4A graph-declared frozen references to current source paths."""

    graph_path = repo_root / "docs/refactors/CONFIG_REFERENCE_GRAPH_PHASE4A.csv"
    if not graph_path.is_file():
        return set()
    columns, rows = _read_csv(graph_path)
    required = {
        "source_file",
        "referenced_config",
        "is_historical_doc",
        "requires_update_on_move",
    }
    if not required.issubset(columns):
        return set()
    source_candidates = {
        row["old_path"]: row["candidate_new_path"]
        for row in plan
        if row.get("old_path") and row.get("candidate_new_path")
    }
    frozen: set[tuple[str, str]] = set()
    for row in rows:
        if (
            row["is_historical_doc"] != "YES"
            or row["requires_update_on_move"] != "NO"
        ):
            continue
        source_file = row["source_file"]
        frozen.add((row["referenced_config"], source_file))
        candidate_source = source_candidates.get(source_file)
        if candidate_source:
            frozen.add((row["referenced_config"], candidate_source))
    return frozen


def _historical_reference_is_allowed(
    row: Mapping[str, str],
    frozen_pairs: set[tuple[str, str]],
) -> bool:
    source_file = row["source_file"]
    if source_file.startswith("docs/"):
        return row["reference_type"] == "historical_doc"
    if source_file.startswith("tests/dev/test_audit_config_"):
        return row["reference_type"] in {
            "dedicated_migration_audit_test",
            "test_fixture",
        }
    if source_file.startswith("tests/"):
        return (
            row["reference_type"] == "test_fixture"
            and (row["old_config_path"], source_file) in frozen_pairs
        )
    if source_file.startswith("configs/"):
        return (
            row["reference_type"] == "yaml_reference"
            and (row["old_config_path"], source_file) in frozen_pairs
        )
    return False


def audit_batch_migration(
    repo_root: Path,
    batch: int,
    plan_path: Path,
    moves_path: Path,
    *,
    strict: bool = False,
    tracked_yaml_override: set[str] | None = None,
    tracked_text_override: Mapping[str, str] | None = None,
    staged_paths_override: set[str] | None = None,
    expected_batch_count: int | None = None,
    expected_tracked_yaml_count: int | None = None,
    before_snapshot_path: Path | None = None,
    after_snapshot_path: Path | None = None,
    semantic_diff_path: Path | None = None,
    reference_audit_path: Path | None = None,
    git_head_bytes_override: Mapping[str, bytes] | None = None,
    git_head_commit_override: str | None = None,
) -> list[str]:
    """Return migration errors without changing the repository."""

    repo_root = repo_root.resolve()
    batch_label = f"Batch {batch}"
    artifacts_dir = moves_path.parent
    before_snapshot_path = before_snapshot_path or (
        artifacts_dir / f"CONFIG_BATCH{batch}_BEFORE_SNAPSHOT.json"
    )
    after_snapshot_path = after_snapshot_path or (
        artifacts_dir / f"CONFIG_BATCH{batch}_AFTER_SNAPSHOT.json"
    )
    semantic_diff_path = semantic_diff_path or (
        artifacts_dir / f"CONFIG_BATCH{batch}_SEMANTIC_DIFF.csv"
    )
    reference_audit_path = reference_audit_path or (
        artifacts_dir / f"CONFIG_BATCH{batch}_REFERENCE_AUDIT.csv"
    )

    plan_columns, plan = _read_csv(plan_path)
    move_columns, moves = _read_csv(moves_path)
    reference_columns, references = _read_csv(reference_audit_path)
    semantic_columns, semantic_rows = _read_csv(semantic_diff_path)
    errors: list[str] = []

    for column in _missing_columns(
        plan_columns, ("old_path", "candidate_new_path", "migration_batch")
    ):
        errors.append(f"plan is missing column: {column}")
    required_move_columns = (
        BATCH3_MOVE_COLUMNS if batch in {3, 4} else MOVE_COLUMNS
    )
    required_reference_columns = (
        BATCH3_REFERENCE_COLUMNS if batch in {3, 4} else REFERENCE_COLUMNS
    )
    required_semantic_columns = (
        BATCH3_SEMANTIC_DIFF_COLUMNS
        if batch in {3, 4}
        else SEMANTIC_DIFF_COLUMNS
    )
    for column in _missing_columns(move_columns, required_move_columns):
        errors.append(f"moves is missing column: {column}")
    for column in _missing_columns(reference_columns, required_reference_columns):
        errors.append(f"reference audit is missing column: {column}")
    for column in _missing_columns(semantic_columns, required_semantic_columns):
        errors.append(f"semantic diff is missing column: {column}")
    if errors:
        return errors

    batch_plan = [row for row in plan if row["migration_batch"] == batch_label]
    other_plan = [row for row in plan if row["migration_batch"] != batch_label]
    expected_batch_count = (
        expected_batch_count
        if expected_batch_count is not None
        else EXPECTED_BATCH_COUNTS.get(batch)
    )
    if expected_batch_count is not None and len(batch_plan) != expected_batch_count:
        errors.append(
            f"{batch_label} must contain {expected_batch_count} rows; "
            f"found {len(batch_plan)}"
        )
    expected_implementations = EXPECTED_IMPLEMENTATION_COUNTS.get(batch)
    if expected_implementations is not None:
        actual_implementations: dict[str, int] = defaultdict(int)
        for row in batch_plan:
            actual_implementations[row.get("implementation", "")] += 1
        if dict(actual_implementations) != expected_implementations:
            errors.append(
                f"{batch_label} implementation counts mismatch: "
                f"{dict(actual_implementations)} != {expected_implementations}"
            )
    if batch == 2 and any(row.get("model") != "mmgcn" for row in batch_plan):
        errors.append("Batch 2 must contain only MMGCN configs")
    if batch == 3 and any(
        row.get("model") != "multidag_cl" for row in batch_plan
    ):
        errors.append("Batch 3 must contain only MultiDAG configs")
    if batch == 4 and any(
        row.get("model") != "dialoguegcn" for row in batch_plan
    ):
        errors.append("Batch 4 must contain only DialogueGCN configs")
    if len(moves) != len(batch_plan):
        errors.append(
            f"moves row count {len(moves)} does not match plan {len(batch_plan)}"
        )

    plan_by_old = {row["old_path"]: row for row in batch_plan}
    plan_by_candidate = {row["candidate_new_path"]: row for row in batch_plan}
    moves_by_old = {row["old_path"]: row for row in moves}
    if len(plan_by_old) != len(batch_plan):
        errors.append(f"{batch_label} plan contains duplicate old_path values")
    if len(plan_by_candidate) != len(batch_plan):
        errors.append(f"{batch_label} plan contains duplicate candidate_new_path values")
    if len(moves_by_old) != len(moves):
        errors.append("moves contains duplicate old_path values")
    if set(plan_by_old) != set(moves_by_old):
        errors.append("moves old_path set does not exactly match the batch plan")

    tracked = (
        set(tracked_yaml_override)
        if tracked_yaml_override is not None
        else _tracked_yaml_paths(repo_root)
    )
    expected_tracked_yaml_count = (
        expected_tracked_yaml_count
        if expected_tracked_yaml_count is not None
        else DEFAULT_TRACKED_YAML_COUNT
    )
    if strict and len(tracked) != expected_tracked_yaml_count:
        errors.append(
            f"tracked YAML count must be {expected_tracked_yaml_count}; "
            f"found {len(tracked)}"
        )

    before_payload, before, snapshot_errors = _load_snapshot(before_snapshot_path)
    errors.extend(snapshot_errors)
    after_payload, after, snapshot_errors = _load_snapshot(after_snapshot_path)
    errors.extend(snapshot_errors)
    snapshot_baseline_head = (
        git_head_commit_override
        if git_head_commit_override is not None
        else str(before_payload.get("git_head", ""))
    )
    if batch in {2, 3, 4}:
        expected_snapshot_head = snapshot_baseline_head
        if not expected_snapshot_head:
            errors.append("Git HEAD commit cannot be resolved")
        expected_before_source = (
            "git_head" if batch == 2 else "working_tree_pre_move"
        )
        if before_payload.get("snapshot_source") != expected_before_source:
            errors.append(
                f"Batch {batch} before snapshot_source must be "
                f"{expected_before_source}"
            )
        if before_payload.get("git_head") != expected_snapshot_head:
            errors.append(
                f"Batch {batch} before snapshot git_head does not match "
                "its expected baseline"
            )
        if after_payload.get("snapshot_source") != "working_tree":
            errors.append(
                f"Batch {batch} after snapshot_source must be working_tree"
            )
        if after_payload.get("git_head") != expected_snapshot_head:
            errors.append(
                f"Batch {batch} after snapshot git_head does not match "
                "its expected baseline"
            )
        for label, payload in (
            ("before", before_payload),
            ("after", after_payload),
        ):
            if payload.get("migration_batch") != batch:
                errors.append(
                    f"Batch {batch} {label} snapshot migration_batch mismatch"
                )
            if payload.get("yaml_count") != len(batch_plan):
                errors.append(
                    f"Batch {batch} {label} snapshot yaml_count mismatch"
                )
    semantic_by_old = {row["old_path"]: row for row in semantic_rows}
    if len(semantic_by_old) != len(semantic_rows):
        errors.append("semantic diff contains duplicate old_path values")

    for old_path, plan_row in plan_by_old.items():
        move = moves_by_old.get(old_path)
        if move is None:
            continue
        new_path = plan_row["candidate_new_path"]
        old_disk = repo_root / old_path
        new_disk = repo_root / new_path
        label = f"{old_path} -> {new_path}"

        if move["new_path"] != new_path:
            errors.append(f"{old_path}: moves new_path disagrees with plan")
        for field in (
            "model",
            "implementation",
            "dataset",
            "context_mode",
            "feature_set",
            "purpose",
            "scope",
        ):
            if move[field] != plan_row.get(field, ""):
                errors.append(f"{old_path}: moves disagrees with plan on {field}")
        if plan_row.get("manual_review") != "NO":
            errors.append(f"{old_path}: batch row requires manual review")
        if plan_row.get("collision_status") != "CLEAR":
            errors.append(f"{old_path}: batch row has unresolved collision")
        if not _is_relative_repo_path(old_path):
            errors.append(f"{old_path}: old path is not a safe relative path")
        if not _is_relative_repo_path(new_path):
            errors.append(f"{new_path}: new path is not a safe relative path")
        if (
            "author_official" in new_path.lower()
            or move["implementation"] == "author_official"
        ):
            errors.append(f"{label}: author_official is forbidden")
        if batch in {3, 4}:
            expected_model = (
                "multidag_cl" if batch == 3 else "dialoguegcn"
            )
            expected_prefix = (
                f"configs/{expected_model}/{move['implementation']}/"
            )
            if move["model"] != expected_model:
                errors.append(
                    f"{label}: Batch {batch} model must be {expected_model}"
                )
            if move["implementation"] not in {"unified", "paper_aligned"}:
                errors.append(
                    f"{label}: Batch {batch} implementation must be unified "
                    "or paper_aligned"
                )
            elif not new_path.startswith(expected_prefix):
                errors.append(
                    f"{label}: implementation is placed under the wrong "
                    "canonical lineage"
                )
            expected_paper_flag = (
                "YES" if move["implementation"] == "paper_aligned" else "NO"
            )
            if move["is_paper_aligned"] != expected_paper_flag:
                errors.append(
                    f"{label}: is_paper_aligned does not match implementation"
                )
            if move["is_official"] != "NO":
                errors.append(
                    f"{label}: paper_aligned/Batch {batch} configs "
                    "must be non-official"
                )
            for flag in (
                "is_smoke",
                "is_formal",
                "is_paper_aligned",
                "is_official",
            ):
                if move[flag] not in YES_NO:
                    errors.append(f"{label}: {flag} must be YES or NO")
        if old_disk.exists():
            errors.append(f"{old_path}: old path still exists")
        if not new_disk.is_file():
            errors.append(f"{new_path}: new path is missing")
            continue
        if new_disk.is_symlink():
            errors.append(f"{new_path}: migrated YAML must not be a symlink")
        if new_path not in tracked:
            errors.append(f"{new_path}: new path is not tracked or in the Git index")
        try:
            parsed_disk = yaml.safe_load(new_disk.read_text(encoding="utf-8-sig"))
        except (OSError, yaml.YAMLError) as exc:
            errors.append(f"{new_path}: YAML parse failed: {exc}")
            continue
        if "author_official" in new_disk.read_text(
            encoding="utf-8-sig"
        ).lower():
            errors.append(f"{new_path}: YAML contains author_official")
        if batch in {3, 4}:
            model_section = (
                parsed_disk.get("model", {})
                if isinstance(parsed_disk, dict)
                else {}
            )
            model_name = (
                model_section.get("name")
                if isinstance(model_section, dict)
                else None
            )
            expected_model_name = (
                (
                    "MultiDAGCL"
                    if move["implementation"] == "unified"
                    else "original_repro_multidag_cl"
                )
                if batch == 3
                else (
                    "causal_dialoguegcn"
                    if move["implementation"] == "unified"
                    else "original_repro_dialoguegcn"
                )
            )
            if model_name != expected_model_name:
                errors.append(
                    f"{new_path}: model registry/consumer key mismatch: "
                    f"{model_name!r} != {expected_model_name!r}"
                )
            elif batch == 4:
                registry_root = repo_root
                if not (registry_root / "models/registry").is_dir():
                    registry_root = Path(__file__).resolve().parents[2]
                registry_source = (
                    registry_root / "models/registry/causal.py"
                    if move["implementation"] == "unified"
                    else registry_root / "models/registry/paper_aligned.py"
                )
                try:
                    registry_text = registry_source.read_text(
                        encoding="utf-8-sig"
                    )
                except OSError as exc:
                    errors.append(
                        f"{new_path}: DialogueGCN registry source unavailable: "
                        f"{exc}"
                    )
                else:
                    quoted_key = f'"{expected_model_name}"'
                    if quoted_key not in registry_text:
                        errors.append(
                            f"{new_path}: DialogueGCN registry key is missing "
                            f"from {registry_source}"
                        )

        before_entry = before.get(old_path)
        after_entry = after.get(old_path)
        semantic_row = semantic_by_old.get(old_path)
        if before_entry is None:
            errors.append(f"{old_path}: missing before snapshot entry")
            continue
        if after_entry is None:
            errors.append(f"{old_path}: missing after snapshot entry")
            continue
        if semantic_row is None:
            errors.append(f"{old_path}: missing semantic diff entry")
            continue
        if before_entry.get("new_path") != new_path:
            errors.append(f"{old_path}: before snapshot new_path mismatch")
        if after_entry.get("new_path") != new_path:
            errors.append(f"{old_path}: after snapshot new_path mismatch")

        head_bytes = (
            git_head_bytes_override.get(old_path)
            if git_head_bytes_override is not None
            else _git_commit_bytes(
                repo_root,
                snapshot_baseline_head if batch == 2 else "HEAD",
                old_path,
            )
        )
        if batch == 2:
            if head_bytes is None:
                errors.append(f"{old_path}: Git HEAD old_path content is unavailable")
                continue
            try:
                head_yaml = yaml.safe_load(head_bytes.decode("utf-8-sig"))
            except (UnicodeDecodeError, yaml.YAMLError) as exc:
                errors.append(f"{old_path}: Git HEAD YAML parse failed: {exc}")
                continue
            head_sha = hashlib.sha256(head_bytes).hexdigest()
            if before_entry.get("snapshot_source") != "git_head":
                errors.append(f"{old_path}: before entry snapshot_source must be git_head")
            if before_entry.get("old_path") != old_path:
                errors.append(f"{old_path}: before entry old_path mismatch")
            if str(before_entry.get("sha256", "")) != head_sha:
                errors.append(f"{old_path}: before snapshot SHA does not match Git HEAD")
            if before_entry.get("parsed_yaml") != head_yaml:
                errors.append(
                    f"{old_path}: before snapshot semantics do not match Git HEAD"
                )

        disk_sha = (
            _canonical_text_sha256(new_disk)
            if batch == 2
            else _sha256(new_disk)
        )
        before_sha = str(before_entry.get("sha256", ""))
        after_sha = str(after_entry.get("sha256", ""))
        documented_later_reference_drift = (
            _is_documented_later_reference_drift(
                repo_root,
                batch,
                new_path,
                after_entry.get("parsed_yaml"),
                parsed_disk,
            )
        )
        if move["pre_move_sha256"] != before_sha:
            errors.append(f"{old_path}: pre-move SHA does not match snapshot")
        if move["post_move_sha256"] != after_sha or (
            after_sha != disk_sha and not documented_later_reference_drift
        ):
            errors.append(f"{old_path}: post-move SHA does not match disk/snapshot")
        if (
            after_entry.get("parsed_yaml") != parsed_disk
            and not documented_later_reference_drift
        ):
            errors.append(f"{old_path}: after snapshot semantics do not match disk")
        if batch in {2, 3, 4}:
            for snapshot_label, snapshot_entry in (
                ("before", before_entry),
                ("after", after_entry),
            ):
                audit_fields = snapshot_entry.get("audit_fields")
                if not isinstance(audit_fields, dict):
                    errors.append(
                        f"{old_path}: {snapshot_label} snapshot audit_fields missing"
                    )
                    continue
                for audit_field, plan_field in SNAPSHOT_IDENTITY_FIELDS.items():
                    if audit_fields.get(audit_field) != plan_row.get(plan_field, ""):
                        errors.append(
                            f"{old_path}: {snapshot_label} snapshot identity "
                            f"mismatch on {plan_field}"
                        )

        changes = _semantic_changes(
            before_entry.get("parsed_yaml"), after_entry.get("parsed_yaml")
        )
        semantic_changed_keys = {_path_leaf(path) for path in changes}
        changed_keys = _split_keys(move["approved_changed_keys"])
        declared_change = move["yaml_content_changed"]
        path_change_errors: list[str] = []
        if declared_change not in YES_NO:
            errors.append(f"{old_path}: yaml_content_changed must be YES or NO")
        if declared_change == "NO":
            if before_sha != after_sha or changes:
                errors.append(f"{old_path}: undeclared YAML content change")
            if changed_keys:
                errors.append(f"{old_path}: unchanged YAML declares changed keys")
        else:
            if semantic_changed_keys != changed_keys:
                errors.append(
                    f"{old_path}: approved_changed_keys do not match semantic diff"
                )
            unapproved = sorted(
                path for path in changes if _path_leaf(path) not in APPROVED_PATH_KEYS
            )
            if unapproved:
                errors.append(
                    f"{old_path}: non-approved semantic keys changed: {unapproved}"
                )
            path_change_errors = _validate_path_changes(
                repo_root,
                before_entry.get("parsed_yaml"),
                after_entry.get("parsed_yaml"),
                changes,
                plan_by_old,
                plan_by_candidate,
                tracked,
            )
            errors.extend(f"{old_path}: {error}" for error in path_change_errors)

        byte_identical = "YES" if before_sha == after_sha else "NO"
        semantic_identical = "YES" if not changes else "NO"
        if semantic_row["new_path"] != new_path:
            errors.append(f"{old_path}: semantic diff new_path mismatch")
        if (
            batch in {3, 4}
            and semantic_row["implementation"] != move["implementation"]
        ):
            errors.append(f"{old_path}: semantic diff implementation mismatch")
        if semantic_row["byte_identical"] != byte_identical:
            errors.append(f"{old_path}: semantic diff byte result mismatch")
        if semantic_row["semantic_identical"] != semantic_identical:
            errors.append(f"{old_path}: semantic diff semantic result mismatch")
        if _split_keys(semantic_row["changed_keys"]) != semantic_changed_keys:
            errors.append(f"{old_path}: semantic diff changed_keys mismatch")
        expected_allowed = (
            not changes
            or (
                semantic_changed_keys == changed_keys
                and all(_path_leaf(path) in APPROVED_PATH_KEYS for path in changes)
                and not path_change_errors
            )
        )
        if semantic_row["change_allowed"] != ("YES" if expected_allowed else "NO"):
            errors.append(f"{old_path}: semantic diff change_allowed mismatch")
        if semantic_row["status"] != ("PASS" if expected_allowed else "FAIL"):
            errors.append(f"{old_path}: semantic diff status mismatch")

    expected_source_changes = {2: 4, 3: 4, 4: 0}
    if batch in expected_source_changes:
        declared_source_changes = sum(
            move["yaml_content_changed"] == "YES"
            and _split_keys(move["approved_changed_keys"]) == {"source_config"}
            for move in moves
        )
        expected_changes = expected_source_changes[batch]
        if declared_source_changes != expected_changes:
            errors.append(
                f"Batch {batch} must declare exactly {expected_changes} "
                "source_config-only YAML changes; "
                f"found {declared_source_changes}"
            )

    completed_batches = {batch}
    for candidate_batch in range(1, 8):
        if candidate_batch == batch:
            continue
        candidate_label = f"Batch {candidate_batch}"
        candidate_plan = [
            row for row in plan if row["migration_batch"] == candidate_label
        ]
        candidate_moves_path = (
            artifacts_dir / f"CONFIG_BATCH{candidate_batch}_MOVES.csv"
        )
        if not candidate_plan or not candidate_moves_path.is_file():
            continue
        try:
            candidate_move_columns, candidate_moves = _read_csv(
                candidate_moves_path
            )
        except OSError:
            continue
        if _missing_columns(candidate_move_columns, ("old_path", "new_path")):
            continue
        expected_pairs = {
            (row["old_path"], row["candidate_new_path"]) for row in candidate_plan
        }
        recorded_pairs = {
            (row["old_path"], row["new_path"]) for row in candidate_moves
        }
        if expected_pairs != recorded_pairs:
            continue
        if all(
            not (repo_root / old_path).exists()
            and (repo_root / new_path).is_file()
            and new_path in tracked
            for old_path, new_path in expected_pairs
        ):
            completed_batches.add(candidate_batch)

    batch_old_paths = set(plan_by_old)
    for row in other_plan:
        old_path = row["old_path"]
        if old_path in batch_old_paths:
            continue
        row_batch = _batch_number(row.get("migration_batch", ""))
        if row_batch is not None and (
            row_batch < batch or row_batch in completed_batches
        ):
            candidate = row["candidate_new_path"]
            if (repo_root / old_path).exists():
                errors.append(f"earlier-batch old YAML still exists: {old_path}")
            if not (repo_root / candidate).is_file():
                errors.append(f"earlier-batch canonical YAML is missing: {candidate}")
            if candidate not in tracked:
                errors.append(f"earlier-batch canonical YAML is not tracked: {candidate}")
        else:
            if not (repo_root / old_path).is_file():
                errors.append(
                    f"non-{batch_label} YAML was moved or is missing: {old_path}"
                )
            if old_path not in tracked:
                errors.append(f"non-{batch_label} YAML is not tracked: {old_path}")
            if (
                row.get("manual_review") == "YES"
                and not (repo_root / old_path).is_file()
            ):
                errors.append(f"manual-review YAML was moved: {old_path}")

    hash_paths: dict[str, list[str]] = defaultdict(list)
    for relative in sorted(tracked):
        path = repo_root / relative
        if path.is_file():
            hash_paths[_sha256(path)].append(relative)
    for duplicate_paths in hash_paths.values():
        if len(duplicate_paths) > 1 and any(
            path in {row["candidate_new_path"] for row in batch_plan}
            for path in duplicate_paths
        ):
            errors.append(
                "migrated YAML has a byte-identical duplicate: "
                + ", ".join(duplicate_paths)
            )

    reference_lookup = {
        (row["old_config_path"], row["source_file"]): row for row in references
    }
    frozen_historical_pairs = _phase4a_frozen_historical_pairs(repo_root, plan)
    for row in references:
        for flag in (
            "historical_reference",
            "updated",
            "remaining_reference_allowed",
        ):
            if row[flag] not in YES_NO:
                errors.append(
                    f"reference audit {row['source_file']}: {flag} must be YES or NO"
                )
        if row["historical_reference"] == "YES":
            if row["updated"] != "NO" or row["remaining_reference_allowed"] != "YES":
                errors.append(
                    f"historical reference is not correctly retained: "
                    f"{row['source_file']}:{row['source_line']}"
                )
            if not _historical_reference_is_allowed(
                row, frozen_historical_pairs
            ):
                errors.append(
                    "historical reference is outside the approved categories: "
                    f"{row['source_file']}:{row['source_line']}"
                )
        if batch in {3, 4}:
            if row["requires_update"] not in YES_NO:
                errors.append(
                    f"reference audit {row['source_file']}: "
                    "requires_update must be YES or NO"
                )
            expected_requires_update = (
                "NO" if row["historical_reference"] == "YES" else "YES"
            )
            if row["requires_update"] != expected_requires_update:
                errors.append(
                    f"reference audit {row['source_file']}: "
                    "requires_update disagrees with historical status"
                )
        if batch in {2, 3, 4} and row["updated"] == "YES":
            if (
                row["historical_reference"] != "NO"
                or row["remaining_reference_allowed"] != "NO"
            ):
                errors.append(
                    f"updated reference has invalid flags: "
                    f"{row['source_file']}:{row['source_line']}"
                )
            plan_row = plan_by_old.get(row["old_config_path"])
            if (
                plan_row is None
                or row["new_config_path"]
                != plan_row["candidate_new_path"]
            ):
                errors.append(
                    f"updated reference does not use the planned candidate: "
                    f"{row['source_file']}:{row['source_line']}"
                )

    if batch in {2, 3, 4}:
        reference_counts: dict[str, Counter[str]] = defaultdict(Counter)
        for row in references:
            if row["updated"] != "YES":
                continue
            source = row["source_file"]
            if source.startswith("tests/"):
                category = "test"
            elif source.startswith("docs/"):
                category = "doc"
            elif batch in {3, 4} and source.startswith("configs/"):
                category = "yaml"
            else:
                category = "active"
            reference_counts[row["old_config_path"]][category] += 1
        for old_path, move in moves_by_old.items():
            counts = reference_counts[old_path]
            expected_counts = {
                "active_reference_count": counts["active"],
                "test_reference_count": counts["test"],
                "doc_reference_count": counts["doc"],
            }
            if batch in {3, 4}:
                expected_counts["yaml_reference_count"] = counts["yaml"]
            for field, expected in expected_counts.items():
                try:
                    actual = int(move[field])
                except ValueError:
                    errors.append(f"{old_path}: {field} must be an integer")
                    continue
                if actual != expected:
                    errors.append(
                        f"{old_path}: {field} mismatch: {actual} != {expected}"
                    )

    tracked_text = (
        dict(tracked_text_override)
        if tracked_text_override is not None
        else _tracked_text(repo_root)
    )
    for old_path in sorted(batch_old_paths):
        windows_old = old_path.replace("/", "\\")
        for source_file, text in tracked_text.items():
            if old_path not in text and windows_old not in text:
                continue
            if _is_audit_record(source_file):
                continue
            audit_row = reference_lookup.get((old_path, source_file))
            allowed_historical = bool(
                audit_row
                and audit_row["historical_reference"] == "YES"
                and audit_row["remaining_reference_allowed"] == "YES"
                and audit_row["updated"] == "NO"
                and _historical_reference_is_allowed(
                    audit_row, frozen_historical_pairs
                )
            )
            if allowed_historical:
                continue
            if source_file.startswith("scripts/"):
                errors.append(
                    f"production code retains old config path: "
                    f"{source_file}: {old_path}"
                )
            elif source_file.startswith("tests/"):
                errors.append(
                    f"tests retain old config path: {source_file}: {old_path}"
                )
            elif source_file.startswith("configs/"):
                errors.append(
                    f"YAML retains old config path: {source_file}: {old_path}"
                )
            else:
                errors.append(
                    f"unapproved old config reference remains: "
                    f"{source_file}: {old_path}"
                )

    staged = (
        set(staged_paths_override)
        if staged_paths_override is not None
        else set(_run_git(repo_root, "diff", "--cached", "--name-only"))
    )
    protected_staged = sorted(GROUP_MEETING_PATHS & staged)
    if protected_staged:
        errors.append(
            "group-meeting files are staged: " + ", ".join(protected_staged)
        )

    return errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read-only audit for one executed config migration batch."
    )
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--batch", type=int, required=True)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--moves")
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Recompute preflight and actual YAML state without reading artifacts.",
    )
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()

    def resolve(value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else repo_root / path

    if args.preview:
        metrics, errors = preview_batch_migration(
            repo_root,
            int(args.batch),
            resolve(args.plan),
            strict=bool(args.strict),
        )
        for key in (
            "PREVIEW_PATH_CHANGES_REQUIRED",
            "ACTUAL_FILE_CHANGES",
            "APPROVED_ACTUAL_CHANGES",
            "UNAPPROVED_ACTUAL_CHANGES",
            "ACTUAL_YAML_CONTENT_MODIFICATIONS",
            "ACTUAL_UNAPPROVED_SEMANTIC_CHANGES",
        ):
            print(f"{key}={metrics.get(key, 0)}")
        if errors:
            print(
                f"CONFIG_BATCH_MIGRATION_PREVIEW=FAIL "
                f"batch={args.batch} errors={len(errors)}"
            )
            for error in errors:
                print(f"- {error}")
            return 1
        print(f"CONFIG_BATCH_MIGRATION_PREVIEW=PASS batch={args.batch}")
        return 0
    if not args.moves:
        print("CONFIG_BATCH_MIGRATION_AUDIT=FAIL moves artifact is required")
        return 2

    errors = audit_batch_migration(
        repo_root,
        int(args.batch),
        resolve(args.plan),
        resolve(args.moves),
        strict=bool(args.strict),
    )
    if errors:
        print(
            f"CONFIG_BATCH_MIGRATION_AUDIT=FAIL "
            f"batch={args.batch} errors={len(errors)}"
        )
        for error in errors:
            print(f"- {error}")
        return 1
    print(
        f"CONFIG_BATCH_MIGRATION_AUDIT=PASS "
        f"batch={args.batch} moved={EXPECTED_BATCH_COUNTS.get(args.batch, 'unknown')}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

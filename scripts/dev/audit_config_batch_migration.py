"""Read-only validation for one executed config migration batch."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import subprocess
from collections import defaultdict
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
EXPECTED_BATCH_COUNTS = {1: 16}
DEFAULT_TRACKED_YAML_COUNT = 183
YES_NO = {"YES", "NO"}
APPROVED_PATH_KEYS = {
    "base_config",
    "config",
    "config_path",
    "entrypoint",
    "eval_config",
    "parent_config",
    "pipeline_config",
    "script",
    "train_config",
}
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


def _is_relative_repo_path(value: str) -> bool:
    if not value or "\\" in value:
        return False
    if value.startswith("/") or WINDOWS_ABSOLUTE.match(value):
        return False
    return not any(part in {".", ".."} for part in PurePosixPath(value).parts)


def _load_snapshot(path: Path) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        return {}, [f"cannot load snapshot {path}: {exc}"]
    if not isinstance(payload, dict) or not isinstance(payload.get("entries"), list):
        errors.append(f"snapshot must contain an entries list: {path}")
        return {}, errors
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
    return entries, errors


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


def _split_keys(value: str) -> set[str]:
    return {item.strip() for item in value.split("|") if item.strip()}


def _path_leaf(path: str) -> str:
    return path.rsplit(".", 1)[-1].split("[", 1)[0]


def _is_audit_record(path: str) -> bool:
    return path.startswith("docs/refactors/CONFIG_")


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
    for column in _missing_columns(move_columns, MOVE_COLUMNS):
        errors.append(f"moves is missing column: {column}")
    for column in _missing_columns(reference_columns, REFERENCE_COLUMNS):
        errors.append(f"reference audit is missing column: {column}")
    for column in _missing_columns(semantic_columns, SEMANTIC_DIFF_COLUMNS):
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
    if len(moves) != len(batch_plan):
        errors.append(
            f"moves row count {len(moves)} does not match plan {len(batch_plan)}"
        )

    plan_by_old = {row["old_path"]: row for row in batch_plan}
    moves_by_old = {row["old_path"]: row for row in moves}
    if len(plan_by_old) != len(batch_plan):
        errors.append(f"{batch_label} plan contains duplicate old_path values")
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

    before, snapshot_errors = _load_snapshot(before_snapshot_path)
    errors.extend(snapshot_errors)
    after, snapshot_errors = _load_snapshot(after_snapshot_path)
    errors.extend(snapshot_errors)
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

        disk_sha = _sha256(new_disk)
        before_sha = str(before_entry.get("sha256", ""))
        after_sha = str(after_entry.get("sha256", ""))
        if move["pre_move_sha256"] != before_sha:
            errors.append(f"{old_path}: pre-move SHA does not match snapshot")
        if move["post_move_sha256"] != after_sha or after_sha != disk_sha:
            errors.append(f"{old_path}: post-move SHA does not match disk/snapshot")
        if after_entry.get("parsed_yaml") != parsed_disk:
            errors.append(f"{old_path}: after snapshot semantics do not match disk")

        changes = _semantic_changes(
            before_entry.get("parsed_yaml"), after_entry.get("parsed_yaml")
        )
        changed_keys = _split_keys(move["approved_changed_keys"])
        declared_change = move["yaml_content_changed"]
        if declared_change not in YES_NO:
            errors.append(f"{old_path}: yaml_content_changed must be YES or NO")
        if declared_change == "NO":
            if before_sha != after_sha or changes:
                errors.append(f"{old_path}: undeclared YAML content change")
            if changed_keys:
                errors.append(f"{old_path}: unchanged YAML declares changed keys")
        else:
            if changes != changed_keys:
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

        byte_identical = "YES" if before_sha == after_sha else "NO"
        semantic_identical = "YES" if not changes else "NO"
        if semantic_row["new_path"] != new_path:
            errors.append(f"{old_path}: semantic diff new_path mismatch")
        if semantic_row["byte_identical"] != byte_identical:
            errors.append(f"{old_path}: semantic diff byte result mismatch")
        if semantic_row["semantic_identical"] != semantic_identical:
            errors.append(f"{old_path}: semantic diff semantic result mismatch")
        if _split_keys(semantic_row["changed_keys"]) != changes:
            errors.append(f"{old_path}: semantic diff changed_keys mismatch")
        expected_allowed = (
            not changes
            or (
                changes == changed_keys
                and all(_path_leaf(path) in APPROVED_PATH_KEYS for path in changes)
            )
        )
        if semantic_row["change_allowed"] != ("YES" if expected_allowed else "NO"):
            errors.append(f"{old_path}: semantic diff change_allowed mismatch")
        if semantic_row["status"] != ("PASS" if expected_allowed else "FAIL"):
            errors.append(f"{old_path}: semantic diff status mismatch")

    batch_old_paths = set(plan_by_old)
    for row in other_plan:
        old_path = row["old_path"]
        if old_path in batch_old_paths:
            continue
        if not (repo_root / old_path).is_file():
            errors.append(
                f"non-{batch_label} YAML was moved or is missing: {old_path}"
            )
        if old_path not in tracked:
            errors.append(f"non-{batch_label} YAML is not tracked: {old_path}")
        if row.get("manual_review") == "YES" and not (repo_root / old_path).is_file():
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
    parser.add_argument("--moves", required=True)
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()

    def resolve(value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else repo_root / path

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

"""Read-only validation for the Phase 4A config migration inventory and plan."""

from __future__ import annotations

import argparse
import csv
import re
import subprocess
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath
from typing import Iterable, Mapping, Sequence


CLASSIFICATION_COLUMNS = (
    "old_path",
    "model",
    "implementation",
    "dataset",
    "context_mode",
    "feature_set",
    "purpose",
    "scope",
    "is_pipeline",
    "is_benchmark",
    "is_smoke",
    "is_analysis",
    "is_ablation",
    "is_modality_missing",
    "is_formal",
    "is_official",
    "is_paper_aligned",
    "is_project_variant",
    "entrypoint",
    "referenced_configs",
    "referenced_scripts",
    "feature_path",
    "output_path",
    "candidate_new_path",
    "risk_level",
    "confidence",
    "manual_review",
    "notes",
)

MAPPING_COLUMNS = (
    "old_path",
    "candidate_new_path",
    "model",
    "implementation",
    "dataset",
    "context_mode",
    "feature_set",
    "purpose",
    "scope",
    "migration_batch",
    "requires_script_update",
    "requires_test_update",
    "requires_doc_update",
    "requires_yaml_reference_update",
    "requires_entrypoint_update",
    "collision_status",
    "manual_review",
    "risk_level",
    "confidence",
    "notes",
)

ALLOWED_MODELS = {
    "",
    "mmgcn",
    "multidag_cl",
    "dialoguegcn",
    "gsmcc",
    "simple_mlp",
    "sdt",
}
ALLOWED_IMPLEMENTATIONS = {
    "",
    "unified",
    "paper_aligned",
    "project_variant",
    "experimental",
}
ALLOWED_CONTEXT_MODES = {"", "full_context", "causal_context"}
ALLOWED_FEATURE_SETS = {
    "",
    "author_features",
    "legacy_mmgcn_features",
    "clean_roberta_features",
    "m3ed_features",
    "synthetic",
}
ALLOWED_BATCHES = {f"Batch {index}" for index in range(1, 8)}
ALLOWED_RISKS = {"low", "medium", "high"}
ALLOWED_CONFIDENCE = {"low", "medium", "high"}
YES_NO = {"YES", "NO"}
FORBIDDEN_SEGMENTS = {
    "original",
    "official_repro",
    "new",
    "latest",
    "final",
    "final_v2",
    "fixed",
    "fixed2",
    "best",
}
PROTECTED_ROOTS = {"data", "outputs", "third_party", "tmp"}
SNAKE_COMPONENT = re.compile(r"^[a-z0-9][a-z0-9_.-]*$")
WINDOWS_ABSOLUTE = re.compile(r"^[A-Za-z]:[\\/]")


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or ()), [dict(row) for row in reader]


def _tracked_yaml_paths(repo_root: Path) -> set[str]:
    result = subprocess.run(
        ["git", "ls-files", "configs"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return {
        line.strip().replace("\\", "/")
        for line in result.stdout.splitlines()
        if PurePosixPath(line.strip()).suffix.lower() in {".yaml", ".yml"}
    }


def _missing_columns(actual: Sequence[str], required: Iterable[str]) -> list[str]:
    actual_set = set(actual)
    return [column for column in required if column not in actual_set]


def _path_errors(label: str, value: str, *, candidate: bool) -> list[str]:
    errors: list[str] = []
    if not value:
        return [f"{label}: path is empty"]
    if "\\" in value:
        errors.append(f"{label}: path must use '/' separators: {value}")
    if value.startswith("/") or value.startswith("\\") or WINDOWS_ABSOLUTE.match(value):
        errors.append(f"{label}: absolute path is forbidden: {value}")
    normalized = value.replace("\\", "/")
    parts = PurePosixPath(normalized).parts
    if any(part in {".", ".."} for part in parts):
        errors.append(f"{label}: relative traversal is forbidden: {value}")
    if parts and parts[0].lower() in PROTECTED_ROOTS:
        errors.append(f"{label}: protected path is forbidden: {value}")
    if candidate:
        if not normalized.startswith("configs/"):
            errors.append(f"{label}: candidate must be under configs/: {value}")
        for index, part in enumerate(parts):
            comparable = Path(part).stem.lower() if index == len(parts) - 1 else part.lower()
            if comparable in FORBIDDEN_SEGMENTS:
                errors.append(f"{label}: forbidden ambiguous segment {part!r}: {value}")
            if part != "_shared" and not SNAKE_COMPONENT.fullmatch(part):
                errors.append(f"{label}: candidate component is not lowercase snake-style: {part}")
    return errors


def _duplicates(rows: Sequence[Mapping[str, str]], key: str) -> dict[str, list[int]]:
    positions: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows, start=2):
        positions[str(row.get(key, ""))].append(index)
    return {value: lines for value, lines in positions.items() if value and len(lines) > 1}


def audit_plan(
    repo_root: Path,
    classification_path: Path,
    mapping_path: Path,
    *,
    strict: bool = False,
    tracked_yaml_override: set[str] | None = None,
) -> list[str]:
    """Return all validation errors without changing the repository."""

    repo_root = repo_root.resolve()
    classification_columns, classification = _read_csv(classification_path)
    mapping_columns, mapping = _read_csv(mapping_path)
    errors: list[str] = []

    for column in _missing_columns(classification_columns, CLASSIFICATION_COLUMNS):
        errors.append(f"classification is missing column: {column}")
    for column in _missing_columns(mapping_columns, MAPPING_COLUMNS):
        errors.append(f"mapping is missing column: {column}")
    if errors:
        return errors

    tracked = (
        set(tracked_yaml_override)
        if tracked_yaml_override is not None
        else _tracked_yaml_paths(repo_root)
    )
    classification_paths = [row["old_path"] for row in classification]
    mapping_paths = [row["old_path"] for row in mapping]

    for value, lines in _duplicates(classification, "old_path").items():
        errors.append(f"classification old_path is duplicated at rows {lines}: {value}")
    for value, lines in _duplicates(mapping, "old_path").items():
        errors.append(f"mapping old_path is duplicated at rows {lines}: {value}")

    classification_set = set(classification_paths)
    mapping_set = set(mapping_paths)
    for path in sorted(tracked - mapping_set):
        errors.append(f"tracked YAML is missing from mapping: {path}")
    for path in sorted(mapping_set - tracked):
        errors.append(f"mapping contains non-tracked YAML: {path}")
    for path in sorted(tracked - classification_set):
        errors.append(f"tracked YAML is missing from classification: {path}")
    for path in sorted(classification_set - tracked):
        errors.append(f"classification contains non-tracked YAML: {path}")

    classification_by_path = {
        row["old_path"]: row for row in classification if row.get("old_path")
    }
    mapping_by_path = {row["old_path"]: row for row in mapping if row.get("old_path")}

    classification_flags = (
        "is_pipeline",
        "is_benchmark",
        "is_smoke",
        "is_analysis",
        "is_ablation",
        "is_modality_missing",
        "is_formal",
        "is_official",
        "is_paper_aligned",
        "is_project_variant",
        "manual_review",
    )
    for row_number, row in enumerate(classification, start=2):
        label = f"classification row {row_number}"
        errors.extend(_path_errors(f"{label} old_path", row["old_path"], candidate=False))
        errors.extend(
            _path_errors(
                f"{label} candidate_new_path",
                row["candidate_new_path"],
                candidate=True,
            )
        )
        if row["model"] not in ALLOWED_MODELS:
            errors.append(f"{label}: invalid model {row['model']!r}")
        if row["implementation"] not in ALLOWED_IMPLEMENTATIONS:
            errors.append(f"{label}: invalid implementation {row['implementation']!r}")
        if row["context_mode"] not in ALLOWED_CONTEXT_MODES:
            errors.append(f"{label}: invalid context_mode {row['context_mode']!r}")
        if row["feature_set"] not in ALLOWED_FEATURE_SETS:
            errors.append(f"{label}: invalid feature_set {row['feature_set']!r}")
        if row["risk_level"] not in ALLOWED_RISKS:
            errors.append(f"{label}: invalid risk_level {row['risk_level']!r}")
        if row["confidence"] not in ALLOWED_CONFIDENCE:
            errors.append(f"{label}: invalid confidence {row['confidence']!r}")
        for flag in classification_flags:
            if row[flag] not in YES_NO:
                errors.append(f"{label}: {flag} must be YES or NO")
        if row["is_official"] == "YES":
            errors.append(f"{label}: author_official configs do not exist in Phase 4A")
        if row["implementation"] == "paper_aligned":
            if row["is_paper_aligned"] != "YES" or row["is_official"] != "NO":
                errors.append(f"{label}: paper_aligned must be non-official and flagged")
        if row["model"] == "gsmcc":
            if (
                row["implementation"] != "project_variant"
                or row["is_project_variant"] != "YES"
                or row["is_official"] != "NO"
            ):
                errors.append(f"{label}: every GS-MCC config must be project_variant")

    candidate_groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row_number, row in enumerate(mapping, start=2):
        label = f"mapping row {row_number}"
        errors.extend(_path_errors(f"{label} old_path", row["old_path"], candidate=False))
        errors.extend(
            _path_errors(
                f"{label} candidate_new_path",
                row["candidate_new_path"],
                candidate=True,
            )
        )
        if row["old_path"] == row["candidate_new_path"]:
            errors.append(f"{label}: old_path equals candidate_new_path")
        if row["model"] not in ALLOWED_MODELS:
            errors.append(f"{label}: invalid model {row['model']!r}")
        if row["implementation"] not in ALLOWED_IMPLEMENTATIONS:
            errors.append(f"{label}: invalid implementation {row['implementation']!r}")
        if row["context_mode"] not in ALLOWED_CONTEXT_MODES:
            errors.append(f"{label}: invalid context_mode {row['context_mode']!r}")
        if row["feature_set"] not in ALLOWED_FEATURE_SETS:
            errors.append(f"{label}: invalid feature_set {row['feature_set']!r}")
        if row["migration_batch"] not in ALLOWED_BATCHES:
            errors.append(f"{label}: invalid migration_batch {row['migration_batch']!r}")
        if row["risk_level"] not in ALLOWED_RISKS:
            errors.append(f"{label}: invalid risk_level {row['risk_level']!r}")
        if row["confidence"] not in ALLOWED_CONFIDENCE:
            errors.append(f"{label}: invalid confidence {row['confidence']!r}")
        if row["manual_review"] not in YES_NO:
            errors.append(f"{label}: manual_review must be YES or NO")
        for flag in (
            "requires_script_update",
            "requires_test_update",
            "requires_doc_update",
            "requires_yaml_reference_update",
            "requires_entrypoint_update",
        ):
            if row[flag] not in YES_NO:
                errors.append(f"{label}: {flag} must be YES or NO")
        if row["implementation"] == "author_official":
            errors.append(f"{label}: author_official is forbidden in Phase 4A")
        if row["model"] == "gsmcc" and row["implementation"] != "project_variant":
            errors.append(f"{label}: GS-MCC mapping must use project_variant")
        candidate_groups[row["candidate_new_path"]].append(row)

    for candidate, rows in sorted(candidate_groups.items()):
        if len(rows) < 2:
            continue
        if not all(
            row["collision_status"] == "MANUAL_REVIEW"
            and row["manual_review"] == "YES"
            for row in rows
        ):
            errors.append(
                "candidate collision is not fully marked MANUAL_REVIEW: "
                f"{candidate}"
            )

    comparable_fields = (
        "candidate_new_path",
        "model",
        "implementation",
        "dataset",
        "context_mode",
        "feature_set",
        "purpose",
        "scope",
        "manual_review",
        "risk_level",
        "confidence",
    )
    if strict:
        for old_path in sorted(set(classification_by_path) & set(mapping_by_path)):
            left = classification_by_path[old_path]
            right = mapping_by_path[old_path]
            for field in comparable_fields:
                if left[field] != right[field]:
                    errors.append(
                        f"{old_path}: classification/mapping disagree on {field}: "
                        f"{left[field]!r} != {right[field]!r}"
                    )

    for path in sorted(tracked):
        if not (repo_root / Path(path)).is_file():
            errors.append(f"tracked YAML does not exist on disk: {path}")
    return errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read-only audit for the Phase 4A config migration plan."
    )
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--classification", required=True)
    parser.add_argument("--mapping", required=True)
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    classification_path = Path(args.classification)
    mapping_path = Path(args.mapping)
    if not classification_path.is_absolute():
        classification_path = repo_root / classification_path
    if not mapping_path.is_absolute():
        mapping_path = repo_root / mapping_path
    errors = audit_plan(
        repo_root,
        classification_path,
        mapping_path,
        strict=bool(args.strict),
    )
    if errors:
        print(f"CONFIG_MIGRATION_PLAN_AUDIT=FAIL errors={len(errors)}")
        for error in errors:
            print(f"- {error}")
        return 1
    _, rows = _read_csv(mapping_path)
    collisions = Counter(row["candidate_new_path"] for row in rows)
    collision_count = sum(1 for count in collisions.values() if count > 1)
    print(
        "CONFIG_MIGRATION_PLAN_AUDIT=PASS "
        f"tracked_yaml={len(rows)} candidate_collisions={collision_count}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

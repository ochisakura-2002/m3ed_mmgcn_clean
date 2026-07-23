from __future__ import annotations

import csv
from pathlib import Path

import pytest

from scripts.dev.audit_config_migration_plan import (
    CLASSIFICATION_COLUMNS,
    MAPPING_COLUMNS,
    audit_plan,
)


OLD_A = "configs/smoke/a.yaml"
OLD_B = "configs/smoke/b.yaml"
CANDIDATE_A = (
    "configs/mmgcn/unified/synthetic/causal_context/synthetic/smoke_a.yaml"
)
CANDIDATE_B = (
    "configs/mmgcn/unified/synthetic/causal_context/synthetic/smoke_b.yaml"
)


def _write_csv(path: Path, columns: tuple[str, ...], rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _classification(old_path: str, candidate: str) -> dict[str, str]:
    row = {column: "" for column in CLASSIFICATION_COLUMNS}
    row.update(
        {
            "old_path": old_path,
            "model": "mmgcn",
            "implementation": "unified",
            "dataset": "synthetic",
            "context_mode": "causal_context",
            "feature_set": "synthetic",
            "purpose": "smoke",
            "scope": "model",
            "is_pipeline": "NO",
            "is_benchmark": "NO",
            "is_smoke": "YES",
            "is_analysis": "NO",
            "is_ablation": "NO",
            "is_modality_missing": "NO",
            "is_formal": "NO",
            "is_official": "NO",
            "is_paper_aligned": "NO",
            "is_project_variant": "NO",
            "entrypoint": "scripts/models/mmgcn/unified/train.py",
            "candidate_new_path": candidate,
            "risk_level": "low",
            "confidence": "high",
            "manual_review": "NO",
            "notes": "synthetic fixture",
        }
    )
    return row


def _mapping(old_path: str, candidate: str) -> dict[str, str]:
    row = {column: "" for column in MAPPING_COLUMNS}
    row.update(
        {
            "old_path": old_path,
            "candidate_new_path": candidate,
            "model": "mmgcn",
            "implementation": "unified",
            "dataset": "synthetic",
            "context_mode": "causal_context",
            "feature_set": "synthetic",
            "purpose": "smoke",
            "scope": "model",
            "migration_batch": "Batch 1",
            "requires_script_update": "NO",
            "requires_test_update": "NO",
            "requires_doc_update": "NO",
            "requires_yaml_reference_update": "NO",
            "requires_entrypoint_update": "NO",
            "collision_status": "CLEAR",
            "manual_review": "NO",
            "risk_level": "low",
            "confidence": "high",
            "notes": "synthetic fixture",
        }
    )
    return row


def _run(
    tmp_path: Path,
    classifications: list[dict[str, str]],
    mappings: list[dict[str, str]],
    tracked: set[str],
) -> list[str]:
    for path in tracked:
        target = tmp_path / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("model: synthetic\n", encoding="utf-8")
    classification_path = tmp_path / "classification.csv"
    mapping_path = tmp_path / "mapping.csv"
    _write_csv(classification_path, CLASSIFICATION_COLUMNS, classifications)
    _write_csv(mapping_path, MAPPING_COLUMNS, mappings)
    return audit_plan(
        tmp_path,
        classification_path,
        mapping_path,
        strict=True,
        tracked_yaml_override=tracked,
    )


def test_valid_plan_passes(tmp_path: Path) -> None:
    errors = _run(
        tmp_path,
        [_classification(OLD_A, CANDIDATE_A)],
        [_mapping(OLD_A, CANDIDATE_A)],
        {OLD_A},
    )
    assert errors == []


def test_valid_executed_move_passes(tmp_path: Path) -> None:
    errors = _run(
        tmp_path,
        [_classification(OLD_A, CANDIDATE_A)],
        [_mapping(OLD_A, CANDIDATE_A)],
        {CANDIDATE_A},
    )
    assert errors == []


def test_missing_tracked_yaml_fails(tmp_path: Path) -> None:
    errors = _run(
        tmp_path,
        [_classification(OLD_A, CANDIDATE_A)],
        [_mapping(OLD_A, CANDIDATE_A)],
        {OLD_A, OLD_B},
    )
    assert any("missing from mapping" in error for error in errors)


def test_duplicate_candidate_path_fails(tmp_path: Path) -> None:
    errors = _run(
        tmp_path,
        [
            _classification(OLD_A, CANDIDATE_A),
            _classification(OLD_B, CANDIDATE_A),
        ],
        [_mapping(OLD_A, CANDIDATE_A), _mapping(OLD_B, CANDIDATE_A)],
        {OLD_A, OLD_B},
    )
    assert any("collision is not fully marked" in error for error in errors)


def test_gsmcc_author_official_fails(tmp_path: Path) -> None:
    classification = _classification(OLD_A, CANDIDATE_A)
    mapping = _mapping(OLD_A, CANDIDATE_A)
    for row in (classification, mapping):
        row["model"] = "gsmcc"
        row["implementation"] = "author_official"
    classification["is_official"] = "YES"
    errors = _run(tmp_path, [classification], [mapping], {OLD_A})
    assert any("GS-MCC" in error or "author_official" in error for error in errors)


def test_paper_aligned_official_fails(tmp_path: Path) -> None:
    classification = _classification(OLD_A, CANDIDATE_A)
    mapping = _mapping(OLD_A, CANDIDATE_A)
    for row in (classification, mapping):
        row["implementation"] = "paper_aligned"
    classification["is_paper_aligned"] = "YES"
    classification["is_official"] = "YES"
    errors = _run(tmp_path, [classification], [mapping], {OLD_A})
    assert any("non-official" in error or "author_official" in error for error in errors)


@pytest.mark.parametrize(
    "candidate, expected",
    [
        ("C:/repo/configs/mmgcn/formal.yaml", "absolute path"),
        ("configs/mmgcn/unified/final/formal.yaml", "forbidden ambiguous"),
    ],
)
def test_invalid_candidate_path_fails(
    tmp_path: Path, candidate: str, expected: str
) -> None:
    errors = _run(
        tmp_path,
        [_classification(OLD_A, candidate)],
        [_mapping(OLD_A, candidate)],
        {OLD_A},
    )
    assert any(expected in error for error in errors)


def test_fully_marked_manual_review_collision_passes(tmp_path: Path) -> None:
    first_classification = _classification(OLD_A, CANDIDATE_A)
    second_classification = _classification(OLD_B, CANDIDATE_A)
    first_mapping = _mapping(OLD_A, CANDIDATE_A)
    second_mapping = _mapping(OLD_B, CANDIDATE_A)
    for classification in (first_classification, second_classification):
        classification["manual_review"] = "YES"
    for mapping in (first_mapping, second_mapping):
        mapping["collision_status"] = "MANUAL_REVIEW"
        mapping["manual_review"] = "YES"
    errors = _run(
        tmp_path,
        [first_classification, second_classification],
        [first_mapping, second_mapping],
        {OLD_A, OLD_B},
    )
    assert errors == []


def test_unmarked_manual_review_collision_fails(tmp_path: Path) -> None:
    first_classification = _classification(OLD_A, CANDIDATE_A)
    second_classification = _classification(OLD_B, CANDIDATE_A)
    first_mapping = _mapping(OLD_A, CANDIDATE_A)
    second_mapping = _mapping(OLD_B, CANDIDATE_A)
    first_mapping["collision_status"] = "MANUAL_REVIEW"
    first_mapping["manual_review"] = "YES"
    first_classification["manual_review"] = "YES"
    errors = _run(
        tmp_path,
        [first_classification, second_classification],
        [first_mapping, second_mapping],
        {OLD_A, OLD_B},
    )
    assert any("collision is not fully marked" in error for error in errors)

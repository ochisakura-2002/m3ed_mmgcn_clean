from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import yaml

from scripts.dev.audit_config_batch_migration import (
    MOVE_COLUMNS,
    REFERENCE_COLUMNS,
    SEMANTIC_DIFF_COLUMNS,
    audit_batch_migration,
)
from scripts.dev.audit_config_migration_plan import MAPPING_COLUMNS


OLD_A = "configs/smoke/a.yaml"
OLD_B = "configs/smoke/b.yaml"
NEW_A = "configs/mmgcn/unified/synthetic/causal_context/synthetic/a.yaml"
NEW_B = "configs/mmgcn/unified/synthetic/causal_context/synthetic/b.yaml"
LATER_OLD = "configs/later/c.yaml"
LATER_NEW = "configs/mmgcn/unified/iemocap/causal_context/synthetic/c.yaml"
MANUAL_OLD = "configs/later/manual_d.yaml"
MANUAL_NEW = "configs/manual/review/d.yaml"


@dataclass
class Fixture:
    root: Path
    plan_path: Path
    moves_path: Path
    before_path: Path
    after_path: Path
    semantic_path: Path
    reference_path: Path
    tracked: set[str]
    plan_rows: list[dict[str, str]]
    move_rows: list[dict[str, str]]
    before_entries: list[dict[str, object]]
    after_entries: list[dict[str, object]]
    semantic_rows: list[dict[str, str]]
    reference_rows: list[dict[str, str]]


def _write_csv(
    path: Path, columns: tuple[str, ...], rows: list[dict[str, str]]
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, entries: list[dict[str, object]]) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "migration_batch": 1,
                "yaml_count": len(entries),
                "entries": entries,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def _yaml_bytes(payload: dict[str, object]) -> bytes:
    return yaml.safe_dump(payload, sort_keys=True).encode("utf-8")


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _plan_row(
    old_path: str,
    new_path: str,
    *,
    batch: str,
    manual_review: str = "NO",
    collision_status: str = "CLEAR",
) -> dict[str, str]:
    row = {column: "" for column in MAPPING_COLUMNS}
    row.update(
        {
            "old_path": old_path,
            "candidate_new_path": new_path,
            "model": "mmgcn",
            "implementation": "unified",
            "dataset": "synthetic",
            "context_mode": "causal_context",
            "feature_set": "synthetic",
            "purpose": "smoke",
            "scope": "model",
            "migration_batch": batch,
            "requires_script_update": "NO",
            "requires_test_update": "NO",
            "requires_doc_update": "NO",
            "requires_yaml_reference_update": "NO",
            "requires_entrypoint_update": "NO",
            "collision_status": collision_status,
            "manual_review": manual_review,
            "risk_level": "low",
            "confidence": "high",
            "notes": "synthetic fixture",
        }
    )
    return row


def _move_row(
    plan_row: dict[str, str],
    payload: bytes,
) -> dict[str, str]:
    row = {column: "" for column in MOVE_COLUMNS}
    row.update(
        {
            "old_path": plan_row["old_path"],
            "new_path": plan_row["candidate_new_path"],
            "model": plan_row["model"],
            "implementation": plan_row["implementation"],
            "dataset": plan_row["dataset"],
            "context_mode": plan_row["context_mode"],
            "feature_set": plan_row["feature_set"],
            "purpose": plan_row["purpose"],
            "scope": plan_row["scope"],
            "is_smoke": "YES",
            "pre_move_sha256": _sha(payload),
            "post_move_sha256": _sha(payload),
            "yaml_content_changed": "NO",
            "approved_changed_keys": "",
            "active_reference_count": "0",
            "test_reference_count": "0",
            "doc_reference_count": "0",
            "status": "MOVED",
            "notes": "synthetic fixture",
        }
    )
    return row


def _snapshot_entry(
    plan_row: dict[str, str],
    payload: dict[str, object],
    raw: bytes,
) -> dict[str, object]:
    return {
        "old_path": plan_row["old_path"],
        "new_path": plan_row["candidate_new_path"],
        "sha256": _sha(raw),
        "top_level_keys": list(payload),
        "parsed_yaml": payload,
        "audit_fields": {},
    }


def _semantic_row(plan_row: dict[str, str]) -> dict[str, str]:
    row = {column: "" for column in SEMANTIC_DIFF_COLUMNS}
    row.update(
        {
            "old_path": plan_row["old_path"],
            "new_path": plan_row["candidate_new_path"],
            "byte_identical": "YES",
            "semantic_identical": "YES",
            "changed_keys": "",
            "change_allowed": "YES",
            "status": "PASS",
            "notes": "synthetic fixture",
        }
    )
    return row


def _persist(fixture: Fixture) -> None:
    _write_csv(fixture.plan_path, MAPPING_COLUMNS, fixture.plan_rows)
    _write_csv(fixture.moves_path, MOVE_COLUMNS, fixture.move_rows)
    _write_json(fixture.before_path, fixture.before_entries)
    _write_json(fixture.after_path, fixture.after_entries)
    _write_csv(
        fixture.semantic_path, SEMANTIC_DIFF_COLUMNS, fixture.semantic_rows
    )
    _write_csv(
        fixture.reference_path, REFERENCE_COLUMNS, fixture.reference_rows
    )


def _make_fixture(tmp_path: Path) -> Fixture:
    batch_plans = [
        _plan_row(OLD_A, NEW_A, batch="Batch 1"),
        _plan_row(OLD_B, NEW_B, batch="Batch 1"),
    ]
    later_plan = _plan_row(LATER_OLD, LATER_NEW, batch="Batch 2")
    manual_plan = _plan_row(
        MANUAL_OLD,
        MANUAL_NEW,
        batch="Batch 7",
        manual_review="YES",
        collision_status="MANUAL_REVIEW",
    )
    payloads = [
        {"seed": 1, "model": {"name": "fixture_a", "dropout": 0.1}},
        {"seed": 2, "model": {"name": "fixture_b", "dropout": 0.2}},
    ]
    raw_payloads = [_yaml_bytes(payload) for payload in payloads]
    for plan_row, raw in zip(batch_plans, raw_payloads):
        target = tmp_path / plan_row["candidate_new_path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
    for path, seed in ((LATER_OLD, 3), (MANUAL_OLD, 4)):
        target = tmp_path / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(_yaml_bytes({"seed": seed}))

    fixture = Fixture(
        root=tmp_path,
        plan_path=tmp_path / "plan.csv",
        moves_path=tmp_path / "CONFIG_BATCH1_MOVES.csv",
        before_path=tmp_path / "CONFIG_BATCH1_BEFORE_SNAPSHOT.json",
        after_path=tmp_path / "CONFIG_BATCH1_AFTER_SNAPSHOT.json",
        semantic_path=tmp_path / "CONFIG_BATCH1_SEMANTIC_DIFF.csv",
        reference_path=tmp_path / "CONFIG_BATCH1_REFERENCE_AUDIT.csv",
        tracked={NEW_A, NEW_B, LATER_OLD, MANUAL_OLD},
        plan_rows=[*batch_plans, later_plan, manual_plan],
        move_rows=[
            _move_row(plan_row, raw)
            for plan_row, raw in zip(batch_plans, raw_payloads)
        ],
        before_entries=[
            _snapshot_entry(plan_row, payload, raw)
            for plan_row, payload, raw in zip(
                batch_plans, payloads, raw_payloads
            )
        ],
        after_entries=[
            _snapshot_entry(plan_row, payload, raw)
            for plan_row, payload, raw in zip(
                batch_plans, payloads, raw_payloads
            )
        ],
        semantic_rows=[_semantic_row(row) for row in batch_plans],
        reference_rows=[],
    )
    _persist(fixture)
    return fixture


def _audit(
    fixture: Fixture,
    *,
    tracked_text: dict[str, str] | None = None,
    expected_total: int = 4,
) -> list[str]:
    return audit_batch_migration(
        fixture.root,
        1,
        fixture.plan_path,
        fixture.moves_path,
        strict=True,
        tracked_yaml_override=fixture.tracked,
        tracked_text_override=tracked_text or {},
        staged_paths_override=set(),
        expected_batch_count=2,
        expected_tracked_yaml_count=expected_total,
        before_snapshot_path=fixture.before_path,
        after_snapshot_path=fixture.after_path,
        semantic_diff_path=fixture.semantic_path,
        reference_audit_path=fixture.reference_path,
    )


def _change_after(
    fixture: Fixture,
    *,
    path: str,
    value: object,
    declared: bool,
) -> None:
    payload = dict(fixture.after_entries[0]["parsed_yaml"])
    current: dict[str, object] = payload
    parts = path.split(".")
    for part in parts[:-1]:
        current[part] = dict(current[part])  # type: ignore[arg-type]
        current = current[part]  # type: ignore[assignment]
    current[parts[-1]] = value
    raw = _yaml_bytes(payload)
    (fixture.root / NEW_A).write_bytes(raw)
    fixture.after_entries[0]["parsed_yaml"] = payload
    fixture.after_entries[0]["sha256"] = _sha(raw)
    fixture.move_rows[0]["post_move_sha256"] = _sha(raw)
    fixture.move_rows[0]["yaml_content_changed"] = "YES" if declared else "NO"
    fixture.move_rows[0]["approved_changed_keys"] = path if declared else ""
    fixture.semantic_rows[0].update(
        {
            "byte_identical": "NO",
            "semantic_identical": "NO",
            "changed_keys": path,
            "change_allowed": "YES" if declared else "NO",
            "status": "PASS" if declared else "FAIL",
        }
    )
    _persist(fixture)


def test_valid_batch1_fixture_passes(tmp_path: Path) -> None:
    fixture = _make_fixture(tmp_path)
    assert _audit(fixture) == []


def test_old_path_still_exists_fails(tmp_path: Path) -> None:
    fixture = _make_fixture(tmp_path)
    target = fixture.root / OLD_A
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("redirect: true\n", encoding="utf-8")
    assert any("old path still exists" in error for error in _audit(fixture))


def test_new_path_missing_fails(tmp_path: Path) -> None:
    fixture = _make_fixture(tmp_path)
    (fixture.root / NEW_A).unlink()
    assert any("new path is missing" in error for error in _audit(fixture))


def test_non_batch_yaml_moved_fails(tmp_path: Path) -> None:
    fixture = _make_fixture(tmp_path)
    (fixture.root / LATER_OLD).unlink()
    assert any("non-Batch 1 YAML" in error for error in _audit(fixture))


def test_tracked_yaml_count_change_fails(tmp_path: Path) -> None:
    fixture = _make_fixture(tmp_path)
    fixture.tracked.add("configs/unplanned/extra.yaml")
    assert any("tracked YAML count" in error for error in _audit(fixture))


def test_undeclared_content_change_fails(tmp_path: Path) -> None:
    fixture = _make_fixture(tmp_path)
    _change_after(fixture, path="config_path", value="configs/new.yaml", declared=False)
    assert any("undeclared YAML content change" in error for error in _audit(fixture))


def test_non_approved_key_change_fails(tmp_path: Path) -> None:
    fixture = _make_fixture(tmp_path)
    _change_after(fixture, path="model.dropout", value=0.9, declared=True)
    assert any("non-approved semantic keys" in error for error in _audit(fixture))


def test_active_old_reference_fails(tmp_path: Path) -> None:
    fixture = _make_fixture(tmp_path)
    errors = _audit(
        fixture,
        tracked_text={"scripts/live.py": f'CONFIG = "{OLD_A}"\n'},
    )
    assert any("production code retains old config path" in error for error in errors)


def test_historical_reference_may_remain(tmp_path: Path) -> None:
    fixture = _make_fixture(tmp_path)
    row = {column: "" for column in REFERENCE_COLUMNS}
    row.update(
        {
            "old_config_path": OLD_A,
            "source_file": "docs/history.md",
            "source_line": "12",
            "reference_type": "historical_doc",
            "historical_reference": "YES",
            "updated": "NO",
            "new_config_path": NEW_A,
            "remaining_reference_allowed": "YES",
            "notes": "frozen historical command",
        }
    )
    fixture.reference_rows.append(row)
    _persist(fixture)
    assert _audit(
        fixture, tracked_text={"docs/history.md": f"historical: {OLD_A}\n"}
    ) == []


def test_manual_review_yaml_moved_fails(tmp_path: Path) -> None:
    fixture = _make_fixture(tmp_path)
    (fixture.root / MANUAL_OLD).unlink()
    assert any("manual-review YAML was moved" in error for error in _audit(fixture))

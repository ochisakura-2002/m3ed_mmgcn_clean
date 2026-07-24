from __future__ import annotations

import copy
import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import pytest
import yaml

from scripts.dev.audit_config_batch_migration import (
    BATCH3_MOVE_COLUMNS,
    BATCH3_REFERENCE_COLUMNS,
    BATCH3_SEMANTIC_DIFF_COLUMNS,
    BATCH5_MOVE_COLUMNS,
    BATCH5_SEMANTIC_DIFF_COLUMNS,
    MOVE_COLUMNS,
    REFERENCE_COLUMNS,
    SEMANTIC_DIFF_COLUMNS,
    audit_batch_migration,
    preview_batch_migration,
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
    batch: int
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
    git_head_bytes: dict[str, bytes]
    git_head_commit: str | None


def _write_csv(
    path: Path, columns: tuple[str, ...], rows: list[dict[str, str]]
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(
    path: Path,
    entries: list[dict[str, object]],
    *,
    batch: int,
    snapshot_source: str | None = None,
    git_head: str | None = None,
) -> None:
    payload: dict[str, object] = {
        "schema_version": 1,
        "migration_batch": batch,
        "yaml_count": len(entries),
        "entries": entries,
    }
    if snapshot_source is not None:
        payload["snapshot_source"] = snapshot_source
    if git_head is not None:
        payload["git_head"] = git_head
    path.write_text(
        json.dumps(payload, indent=2) + "\n",
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
        "audit_fields": {
            "model_classification": plan_row["model"],
            "implementation": plan_row["implementation"],
            "dataset_classification": plan_row["dataset"],
            "context_mode": plan_row["context_mode"],
            "feature_set": plan_row["feature_set"],
            "purpose": plan_row["purpose"],
        },
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
    move_columns = (
        BATCH5_MOVE_COLUMNS
        if fixture.batch == 5
        else BATCH3_MOVE_COLUMNS
        if fixture.batch in {3, 4}
        else MOVE_COLUMNS
    )
    reference_columns = (
        BATCH3_REFERENCE_COLUMNS
        if fixture.batch in {3, 4, 5}
        else REFERENCE_COLUMNS
    )
    semantic_columns = (
        BATCH5_SEMANTIC_DIFF_COLUMNS
        if fixture.batch == 5
        else BATCH3_SEMANTIC_DIFF_COLUMNS
        if fixture.batch in {3, 4}
        else SEMANTIC_DIFF_COLUMNS
    )
    _write_csv(fixture.moves_path, move_columns, fixture.move_rows)
    if fixture.batch == 2:
        for entry in fixture.before_entries:
            entry["snapshot_source"] = "git_head"
        for entry in fixture.after_entries:
            entry["snapshot_source"] = "working_tree"
    elif fixture.batch in {3, 4, 5}:
        for entry in fixture.before_entries:
            entry["snapshot_source"] = "working_tree_pre_move"
        for entry in fixture.after_entries:
            entry["snapshot_source"] = "working_tree"
    _write_json(
        fixture.before_path,
        fixture.before_entries,
        batch=fixture.batch,
        snapshot_source=(
            "git_head"
            if fixture.batch == 2
            else "working_tree_pre_move"
            if fixture.batch in {3, 4, 5}
            else None
        ),
        git_head=fixture.git_head_commit,
    )
    _write_json(
        fixture.after_path,
        fixture.after_entries,
        batch=fixture.batch,
        snapshot_source=(
            "working_tree" if fixture.batch in {2, 3, 4, 5} else None
        ),
        git_head=fixture.git_head_commit,
    )
    _write_csv(fixture.semantic_path, semantic_columns, fixture.semantic_rows)
    _write_csv(fixture.reference_path, reference_columns, fixture.reference_rows)


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
        {
            "seed": 1,
            "model": {"name": "fixture_a", "dropout": 0.1},
            "provenance": {"source_config": OLD_B},
        },
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
        batch=1,
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
        git_head_bytes={},
        git_head_commit=None,
    )
    _persist(fixture)
    return fixture


def _make_batch2_fixture(tmp_path: Path) -> Fixture:
    batch_plans: list[dict[str, str]] = []
    for index in range(17):
        plan_row = _plan_row(
            f"configs/batch2/old_{index:02d}.yaml",
            f"configs/mmgcn/{'unified' if index < 13 else 'paper_aligned'}"
            f"/synthetic/causal_context/synthetic/new_{index:02d}.yaml",
            batch="Batch 2",
        )
        plan_row["implementation"] = (
            "unified" if index < 13 else "paper_aligned"
        )
        batch_plans.append(plan_row)

    earlier_plan = _plan_row(
        "configs/batch1/old.yaml",
        "configs/mmgcn/unified/synthetic/causal_context/synthetic/batch1.yaml",
        batch="Batch 1",
    )
    later_plan = _plan_row(
        "configs/batch3/old.yaml",
        "configs/mmgcn/unified/synthetic/causal_context/synthetic/batch3.yaml",
        batch="Batch 3",
    )

    before_payloads: list[dict[str, object]] = []
    after_payloads: list[dict[str, object]] = []
    for index, plan_row in enumerate(batch_plans):
        payload: dict[str, object] = {
            "seed": index,
            "model": {"name": f"fixture_{index:02d}"},
        }
        if index < 4:
            payload["provenance"] = {
                "source_config": batch_plans[index + 4]["old_path"]
            }
        before_payloads.append(payload)
        after_payload = copy.deepcopy(payload)
        if index < 4:
            after_payload["provenance"]["source_config"] = (  # type: ignore[index]
                batch_plans[index + 4]["candidate_new_path"]
            )
        after_payloads.append(after_payload)

    before_raw = [_yaml_bytes(payload) for payload in before_payloads]
    after_raw = [_yaml_bytes(payload) for payload in after_payloads]
    for plan_row, raw in zip(batch_plans, after_raw):
        target = tmp_path / plan_row["candidate_new_path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
    for plan_row in (earlier_plan, later_plan):
        target_path = (
            plan_row["candidate_new_path"]
            if plan_row is earlier_plan
            else plan_row["old_path"]
        )
        target = tmp_path / target_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(_yaml_bytes({"seed": target_path}))

    move_rows = [
        _move_row(plan_row, raw)
        for plan_row, raw in zip(batch_plans, before_raw)
    ]
    semantic_rows = [_semantic_row(plan_row) for plan_row in batch_plans]
    for index in range(4):
        move_rows[index].update(
            {
                "post_move_sha256": _sha(after_raw[index]),
                "yaml_content_changed": "YES",
                "approved_changed_keys": "source_config",
            }
        )
        semantic_rows[index].update(
            {
                "byte_identical": "NO",
                "semantic_identical": "NO",
                "changed_keys": "source_config",
                "change_allowed": "YES",
                "status": "PASS",
            }
        )

    tracked = {
        *(row["candidate_new_path"] for row in batch_plans),
        earlier_plan["candidate_new_path"],
        later_plan["old_path"],
    }
    fixture = Fixture(
        batch=2,
        root=tmp_path,
        plan_path=tmp_path / "plan.csv",
        moves_path=tmp_path / "CONFIG_BATCH2_MOVES.csv",
        before_path=tmp_path / "CONFIG_BATCH2_BEFORE_SNAPSHOT.json",
        after_path=tmp_path / "CONFIG_BATCH2_AFTER_SNAPSHOT.json",
        semantic_path=tmp_path / "CONFIG_BATCH2_SEMANTIC_DIFF.csv",
        reference_path=tmp_path / "CONFIG_BATCH2_REFERENCE_AUDIT.csv",
        tracked=tracked,
        plan_rows=[earlier_plan, *batch_plans, later_plan],
        move_rows=move_rows,
        before_entries=[
            _snapshot_entry(plan_row, payload, raw)
            for plan_row, payload, raw in zip(
                batch_plans, before_payloads, before_raw
            )
        ],
        after_entries=[
            _snapshot_entry(plan_row, payload, raw)
            for plan_row, payload, raw in zip(
                batch_plans, after_payloads, after_raw
            )
        ],
        semantic_rows=semantic_rows,
        reference_rows=[],
        git_head_bytes={
            plan_row["old_path"]: raw
            for plan_row, raw in zip(batch_plans, before_raw)
        },
        git_head_commit="0123456789abcdef0123456789abcdef01234567",
    )
    _persist(fixture)
    return fixture


def _make_batch3_fixture(tmp_path: Path) -> Fixture:
    batch_plans: list[dict[str, str]] = []
    for index in range(17):
        implementation = "unified" if index < 13 else "paper_aligned"
        context_mode = (
            "causal_context"
            if implementation == "unified"
            else "full_context"
        )
        plan_row = _plan_row(
            f"configs/batch3/old_{index:02d}.yaml",
            f"configs/multidag_cl/{implementation}/iemocap/"
            f"{context_mode}/legacy_mmgcn_features/new_{index:02d}.yaml",
            batch="Batch 3",
        )
        plan_row.update(
            {
                "model": "multidag_cl",
                "implementation": implementation,
                "dataset": "iemocap",
                "context_mode": context_mode,
                "feature_set": "legacy_mmgcn_features",
                "purpose": "formal",
            }
        )
        batch_plans.append(plan_row)

    earlier_batch1 = _plan_row(
        "configs/batch1/old.yaml",
        "configs/mmgcn/unified/synthetic/causal_context/synthetic/batch1.yaml",
        batch="Batch 1",
    )
    earlier_batch2 = _plan_row(
        "configs/batch2/old.yaml",
        "configs/mmgcn/unified/synthetic/causal_context/synthetic/batch2.yaml",
        batch="Batch 2",
    )
    later_batch4 = _plan_row(
        "configs/batch4/old.yaml",
        "configs/dialoguegcn/unified/iemocap/causal_context/"
        "legacy_mmgcn_features/batch4.yaml",
        batch="Batch 4",
    )
    manual_batch7 = _plan_row(
        "configs/batch7/manual.yaml",
        "configs/benchmarks/ablations/missing_modality/manual.yaml",
        batch="Batch 7",
        manual_review="YES",
        collision_status="MANUAL_REVIEW",
    )

    before_payloads: list[dict[str, object]] = []
    after_payloads: list[dict[str, object]] = []
    for index, plan_row in enumerate(batch_plans):
        unified = plan_row["implementation"] == "unified"
        payload: dict[str, object] = {
            "seed": index,
            "model": {
                "name": (
                    "MultiDAGCL"
                    if unified
                    else "original_repro_multidag_cl"
                ),
                "curriculum": {"enabled": True, "warmup_epochs": 2},
                "context": {
                    "window_past": 5,
                    "window_future": 0 if unified else 5,
                    "causal": unified,
                },
                "graph": {"layers": 2, "speaker_edges": True},
            },
            "training": {"epochs": 3, "batch_size": 2},
        }
        if index < 4:
            payload["provenance"] = {
                "source_config": batch_plans[index + 4]["old_path"]
            }
        before_payloads.append(payload)
        after_payload = copy.deepcopy(payload)
        if index < 4:
            provenance = after_payload["provenance"]
            assert isinstance(provenance, dict)
            provenance["source_config"] = batch_plans[index + 4][
                "candidate_new_path"
            ]
        after_payloads.append(after_payload)

    before_raw = [_yaml_bytes(payload) for payload in before_payloads]
    after_raw = [_yaml_bytes(payload) for payload in after_payloads]
    for plan_row, raw in zip(batch_plans, after_raw):
        target = tmp_path / plan_row["candidate_new_path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
    for plan_row in (
        earlier_batch1,
        earlier_batch2,
        later_batch4,
        manual_batch7,
    ):
        target_path = (
            plan_row["candidate_new_path"]
            if plan_row["migration_batch"] in {"Batch 1", "Batch 2"}
            else plan_row["old_path"]
        )
        target = tmp_path / target_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(_yaml_bytes({"seed": target_path}))

    move_rows = [
        _move_row(plan_row, raw)
        for plan_row, raw in zip(batch_plans, before_raw)
    ]
    semantic_rows = [_semantic_row(plan_row) for plan_row in batch_plans]
    for plan_row, move_row, semantic_row in zip(
        batch_plans, move_rows, semantic_rows
    ):
        move_row.update(
            {
                "is_smoke": "NO",
                "is_formal": "YES",
                "is_paper_aligned": (
                    "YES"
                    if plan_row["implementation"] == "paper_aligned"
                    else "NO"
                ),
                "is_official": "NO",
                "yaml_reference_count": "0",
            }
        )
        semantic_row["implementation"] = plan_row["implementation"]
    for index in range(4):
        move_rows[index].update(
            {
                "post_move_sha256": _sha(after_raw[index]),
                "yaml_content_changed": "YES",
                "approved_changed_keys": "source_config",
            }
        )
        semantic_rows[index].update(
            {
                "byte_identical": "NO",
                "semantic_identical": "NO",
                "changed_keys": "source_config",
                "change_allowed": "YES",
                "status": "PASS",
            }
        )

    tracked = {
        *(row["candidate_new_path"] for row in batch_plans),
        earlier_batch1["candidate_new_path"],
        earlier_batch2["candidate_new_path"],
        later_batch4["old_path"],
        manual_batch7["old_path"],
    }
    fixture = Fixture(
        batch=3,
        root=tmp_path,
        plan_path=tmp_path / "plan.csv",
        moves_path=tmp_path / "CONFIG_BATCH3_MOVES.csv",
        before_path=tmp_path / "CONFIG_BATCH3_BEFORE_SNAPSHOT.json",
        after_path=tmp_path / "CONFIG_BATCH3_AFTER_SNAPSHOT.json",
        semantic_path=tmp_path / "CONFIG_BATCH3_SEMANTIC_DIFF.csv",
        reference_path=tmp_path / "CONFIG_BATCH3_REFERENCE_AUDIT.csv",
        tracked=tracked,
        plan_rows=[
            earlier_batch1,
            earlier_batch2,
            *batch_plans,
            later_batch4,
            manual_batch7,
        ],
        move_rows=move_rows,
        before_entries=[
            _snapshot_entry(plan_row, payload, raw)
            for plan_row, payload, raw in zip(
                batch_plans, before_payloads, before_raw
            )
        ],
        after_entries=[
            _snapshot_entry(plan_row, payload, raw)
            for plan_row, payload, raw in zip(
                batch_plans, after_payloads, after_raw
            )
        ],
        semantic_rows=semantic_rows,
        reference_rows=[],
        git_head_bytes={},
        git_head_commit="0123456789abcdef0123456789abcdef01234567",
    )
    _persist(fixture)
    return fixture


def _make_batch4_fixture(tmp_path: Path) -> Fixture:
    batch_plans: list[dict[str, str]] = []
    for index in range(10):
        implementation = "unified" if index < 6 else "paper_aligned"
        context_mode = (
            "causal_context"
            if implementation == "unified"
            else "full_context"
        )
        plan_row = _plan_row(
            f"configs/batch4/old_{index:02d}.yaml",
            f"configs/dialoguegcn/{implementation}/iemocap/"
            f"{context_mode}/legacy_mmgcn_features/new_{index:02d}.yaml",
            batch="Batch 4",
        )
        plan_row.update(
            {
                "model": "dialoguegcn",
                "implementation": implementation,
                "dataset": "iemocap",
                "context_mode": context_mode,
                "feature_set": "legacy_mmgcn_features",
                "purpose": "formal",
            }
        )
        batch_plans.append(plan_row)

    earlier_rows = [
        _plan_row(
            f"configs/batch{batch}/old.yaml",
            f"configs/mmgcn/unified/synthetic/causal_context/"
            f"synthetic/batch{batch}.yaml",
            batch=f"Batch {batch}",
        )
        for batch in range(1, 4)
    ]
    later_batch5 = _plan_row(
        "configs/batch5/old.yaml",
        "configs/gsmcc/project_variant/iemocap/causal_context/"
        "legacy_mmgcn_features/batch5.yaml",
        batch="Batch 5",
    )
    manual_batch7 = _plan_row(
        "configs/batch7/manual.yaml",
        "configs/benchmarks/ablations/missing_modality/manual.yaml",
        batch="Batch 7",
        manual_review="YES",
        collision_status="MANUAL_REVIEW",
    )

    payloads: list[dict[str, object]] = []
    raw_payloads: list[bytes] = []
    for index, plan_row in enumerate(batch_plans):
        unified = plan_row["implementation"] == "unified"
        payload: dict[str, object] = {
            "causal": unified,
            "system": {"seed": index},
            "dataset": {
                "name": "IEMOCAP",
                "val_split_strategy": "official_prefix",
                "outer_test_session": "Ses05",
            },
            "model": {
                "name": (
                    "causal_dialoguegcn"
                    if unified
                    else "original_repro_dialoguegcn"
                ),
                "num_speakers": 2,
                "dropout": 0.2,
            },
            "graph": {
                "context_mode": "causal" if unified else "full_context",
                "window_past": 5,
                "window_future": 0 if unified else 5,
                "relation_count": 4,
                "speaker_count": 2,
                "edge_construction": "speaker_temporal",
            },
            "training": {"epochs": 3, "batch_size": 2},
        }
        payloads.append(payload)
        raw_payloads.append(_yaml_bytes(payload))

    for plan_row, raw in zip(batch_plans, raw_payloads):
        target = tmp_path / plan_row["candidate_new_path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
    for plan_row in [*earlier_rows, later_batch5, manual_batch7]:
        target_path = (
            plan_row["candidate_new_path"]
            if plan_row["migration_batch"] in {"Batch 1", "Batch 2", "Batch 3"}
            else plan_row["old_path"]
        )
        target = tmp_path / target_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(_yaml_bytes({"seed": target_path}))

    move_rows = [
        _move_row(plan_row, raw)
        for plan_row, raw in zip(batch_plans, raw_payloads)
    ]
    semantic_rows = [_semantic_row(plan_row) for plan_row in batch_plans]
    for plan_row, move_row, semantic_row in zip(
        batch_plans, move_rows, semantic_rows
    ):
        move_row.update(
            {
                "is_smoke": "NO",
                "is_formal": "YES",
                "is_paper_aligned": (
                    "YES"
                    if plan_row["implementation"] == "paper_aligned"
                    else "NO"
                ),
                "is_official": "NO",
                "yaml_reference_count": "0",
            }
        )
        semantic_row["implementation"] = plan_row["implementation"]

    tracked = {
        *(row["candidate_new_path"] for row in batch_plans),
        *(row["candidate_new_path"] for row in earlier_rows),
        later_batch5["old_path"],
        manual_batch7["old_path"],
    }
    fixture = Fixture(
        batch=4,
        root=tmp_path,
        plan_path=tmp_path / "plan.csv",
        moves_path=tmp_path / "CONFIG_BATCH4_MOVES.csv",
        before_path=tmp_path / "CONFIG_BATCH4_BEFORE_SNAPSHOT.json",
        after_path=tmp_path / "CONFIG_BATCH4_AFTER_SNAPSHOT.json",
        semantic_path=tmp_path / "CONFIG_BATCH4_SEMANTIC_DIFF.csv",
        reference_path=tmp_path / "CONFIG_BATCH4_REFERENCE_AUDIT.csv",
        tracked=tracked,
        plan_rows=[
            *earlier_rows,
            *batch_plans,
            later_batch5,
            manual_batch7,
        ],
        move_rows=move_rows,
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
        semantic_rows=semantic_rows,
        reference_rows=[],
        git_head_bytes={},
        git_head_commit="0123456789abcdef0123456789abcdef01234567",
    )
    _persist(fixture)
    return fixture


def _make_batch5_fixture(tmp_path: Path) -> Fixture:
    batch_plans: list[dict[str, str]] = []
    payloads: list[dict[str, object]] = []
    raw_payloads: list[bytes] = []
    for index in range(13):
        causal_context = index < 7
        context_mode = (
            "causal_context" if causal_context else "full_context"
        )
        feature_set = (
            "clean_roberta_features"
            if index in {5, 8, 11}
            else "legacy_mmgcn_features"
        )
        plan_row = _plan_row(
            f"configs/batch5/old_{index:02d}.yaml",
            f"configs/gsmcc/project_variant/iemocap/{context_mode}/"
            f"{feature_set}/new_{index:02d}.yaml",
            batch="Batch 5",
        )
        plan_row.update(
            {
                "model": "gsmcc",
                "implementation": "project_variant",
                "dataset": "iemocap",
                "context_mode": context_mode,
                "feature_set": feature_set,
                "purpose": "formal" if index not in {5, 6, 11, 12} else "smoke",
            }
        )
        batch_plans.append(plan_row)
        if causal_context:
            payload: dict[str, object] = {
                "causal": True,
                "system": {"seed": index},
                "dataset": {"name": "IEMOCAP"},
                "model": {
                    "name": "causal_gsmcc_inspired",
                    "fusion_type": "concat",
                    "active_modalities": ["text", "audio", "visual"],
                },
                "graph": {
                    "context_mode": "causal",
                    "window_past": 5,
                    "window_future": 0,
                },
                "missing_modalities": {"enabled": False},
                "training": {"epochs": 2, "batch_size": 2},
            }
        else:
            payload = {
                "system": {"seed": index},
                "dataset": {"name": "IEMOCAP"},
                "model": {
                    "name": "project_paper_oriented_gsmcc",
                    "fidelity_status": "PROJECT_VARIANT_NOT_PAPER_REPRODUCTION",
                    "causal_grade": "noncausal_offline_full_context",
                    "fusion_type": "concat",
                    "active_modalities": ["text", "audio", "visual"],
                },
                "missing_modalities": {"enabled": False},
                "training": {"epochs": 2, "batch_size": 2},
            }
        payloads.append(payload)
        raw_payloads.append(_yaml_bytes(payload))

    earlier_rows = [
        _plan_row(
            f"configs/batch{batch}/old.yaml",
            f"configs/mmgcn/unified/synthetic/causal_context/"
            f"synthetic/batch{batch}.yaml",
            batch=f"Batch {batch}",
        )
        for batch in range(1, 5)
    ]
    later_batch6 = _plan_row(
        "configs/batch6/old.yaml",
        "configs/benchmarks/causal_unified/formal/batch6.yaml",
        batch="Batch 6",
    )
    manual_batch7 = _plan_row(
        "configs/batch7/manual.yaml",
        "configs/benchmarks/ablations/missing_modality/manual.yaml",
        batch="Batch 7",
        manual_review="YES",
        collision_status="MANUAL_REVIEW",
    )

    for plan_row, raw in zip(batch_plans, raw_payloads):
        target = tmp_path / plan_row["candidate_new_path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
    for plan_row in [*earlier_rows, later_batch6, manual_batch7]:
        target_path = (
            plan_row["candidate_new_path"]
            if plan_row["migration_batch"]
            in {"Batch 1", "Batch 2", "Batch 3", "Batch 4"}
            else plan_row["old_path"]
        )
        target = tmp_path / target_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(_yaml_bytes({"seed": target_path}))

    move_rows = [
        _move_row(plan_row, raw)
        for plan_row, raw in zip(batch_plans, raw_payloads)
    ]
    semantic_rows = [_semantic_row(plan_row) for plan_row in batch_plans]
    for plan_row, move_row, semantic_row in zip(
        batch_plans, move_rows, semantic_rows
    ):
        move_row.update(
            {
                "is_smoke": (
                    "YES" if plan_row["purpose"] == "smoke" else "NO"
                ),
                "is_formal": (
                    "NO" if plan_row["purpose"] == "smoke" else "YES"
                ),
                "is_project_variant": "YES",
                "is_official": "NO",
                "yaml_reference_count": "0",
            }
        )
        semantic_row["context_mode"] = plan_row["context_mode"]

    tracked = {
        *(row["candidate_new_path"] for row in batch_plans),
        *(row["candidate_new_path"] for row in earlier_rows),
        later_batch6["old_path"],
        manual_batch7["old_path"],
    }
    fixture = Fixture(
        batch=5,
        root=tmp_path,
        plan_path=tmp_path / "plan.csv",
        moves_path=tmp_path / "CONFIG_BATCH5_MOVES.csv",
        before_path=tmp_path / "CONFIG_BATCH5_BEFORE_SNAPSHOT.json",
        after_path=tmp_path / "CONFIG_BATCH5_AFTER_SNAPSHOT.json",
        semantic_path=tmp_path / "CONFIG_BATCH5_SEMANTIC_DIFF.csv",
        reference_path=tmp_path / "CONFIG_BATCH5_REFERENCE_AUDIT.csv",
        tracked=tracked,
        plan_rows=[
            *earlier_rows,
            *batch_plans,
            later_batch6,
            manual_batch7,
        ],
        move_rows=move_rows,
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
        semantic_rows=semantic_rows,
        reference_rows=[],
        git_head_bytes={},
        git_head_commit="0123456789abcdef0123456789abcdef01234567",
    )
    _persist(fixture)
    return fixture


def _audit(
    fixture: Fixture,
    *,
    tracked_text: dict[str, str] | None = None,
    expected_total: int = 4,
) -> list[str]:
    expected_batch_count = sum(
        row["migration_batch"] == f"Batch {fixture.batch}"
        for row in fixture.plan_rows
    )
    return audit_batch_migration(
        fixture.root,
        fixture.batch,
        fixture.plan_path,
        fixture.moves_path,
        strict=True,
        tracked_yaml_override=fixture.tracked,
        tracked_text_override=tracked_text or {},
        staged_paths_override=set(),
        expected_batch_count=expected_batch_count,
        expected_tracked_yaml_count=expected_total,
        before_snapshot_path=fixture.before_path,
        after_snapshot_path=fixture.after_path,
        semantic_diff_path=fixture.semantic_path,
        reference_audit_path=fixture.reference_path,
        git_head_bytes_override=(
            fixture.git_head_bytes if fixture.batch == 2 else None
        ),
        git_head_commit_override=fixture.git_head_commit,
    )


def _change_after_values(
    fixture: Fixture,
    *,
    values: dict[str, object],
    approved_keys: set[str] | None,
    index: int = 0,
) -> None:
    payload = yaml.safe_load(
        yaml.safe_dump(
            fixture.after_entries[index]["parsed_yaml"], sort_keys=True
        )
    )
    for path, value in values.items():
        current: dict[str, object] = payload
        parts = path.split(".")
        for part in parts[:-1]:
            current = current[part]  # type: ignore[assignment]
        current[parts[-1]] = value
    raw = _yaml_bytes(payload)
    new_path = fixture.move_rows[index]["new_path"]
    (fixture.root / new_path).write_bytes(raw)
    fixture.after_entries[index]["parsed_yaml"] = payload
    fixture.after_entries[index]["sha256"] = _sha(raw)
    fixture.move_rows[index]["post_move_sha256"] = _sha(raw)
    declared = approved_keys is not None
    changed_keys = "|".join(sorted(approved_keys or set()))
    fixture.move_rows[index]["yaml_content_changed"] = (
        "YES" if declared else "NO"
    )
    fixture.move_rows[index]["approved_changed_keys"] = changed_keys
    fixture.semantic_rows[index].update(
        {
            "byte_identical": "NO",
            "semantic_identical": "NO",
            "changed_keys": changed_keys,
            "change_allowed": "YES" if declared else "NO",
            "status": "PASS" if declared else "FAIL",
        }
    )
    _persist(fixture)


def _change_after(
    fixture: Fixture,
    *,
    path: str,
    value: object,
    declared: bool,
    index: int = 0,
) -> None:
    _change_after_values(
        fixture,
        values={path: value},
        approved_keys={path.rsplit(".", 1)[-1]} if declared else None,
        index=index,
    )


def test_valid_batch1_fixture_passes(tmp_path: Path) -> None:
    fixture = _make_fixture(tmp_path)
    assert _audit(fixture) == []


def test_valid_source_config_migration_passes(tmp_path: Path) -> None:
    fixture = _make_fixture(tmp_path)
    _change_after(
        fixture,
        path="provenance.source_config",
        value=NEW_B,
        declared=True,
    )
    assert _audit(fixture) == []


def test_valid_train_config_migration_passes(tmp_path: Path) -> None:
    fixture = _make_fixture(tmp_path)
    for entries in (fixture.before_entries, fixture.after_entries):
        parsed = entries[0]["parsed_yaml"]
        assert isinstance(parsed, dict)
        provenance = parsed["provenance"]
        assert isinstance(provenance, dict)
        provenance["train_config"] = OLD_B
        raw = _yaml_bytes(parsed)
        entries[0]["sha256"] = _sha(raw)
    before_raw = _yaml_bytes(
        fixture.before_entries[0]["parsed_yaml"]  # type: ignore[arg-type]
    )
    after_raw = _yaml_bytes(
        fixture.after_entries[0]["parsed_yaml"]  # type: ignore[arg-type]
    )
    fixture.move_rows[0]["pre_move_sha256"] = _sha(before_raw)
    fixture.move_rows[0]["post_move_sha256"] = _sha(after_raw)
    (fixture.root / NEW_A).write_bytes(after_raw)
    _persist(fixture)

    _change_after(
        fixture,
        path="provenance.train_config",
        value=NEW_B,
        declared=True,
    )
    assert _audit(fixture) == []


def test_unrelated_source_config_target_fails(tmp_path: Path) -> None:
    fixture = _make_fixture(tmp_path)
    _change_after(
        fixture,
        path="provenance.source_config",
        value="configs/unrelated/train.yaml",
        declared=True,
    )
    assert any(
        "does not match the planned candidate" in error
        for error in _audit(fixture)
    )


def test_wrong_in_batch_source_config_target_fails(tmp_path: Path) -> None:
    fixture = _make_fixture(tmp_path)
    _change_after(
        fixture,
        path="provenance.source_config",
        value=NEW_A,
        declared=True,
    )
    assert any(
        "does not match the planned candidate" in error
        for error in _audit(fixture)
    )


def test_absolute_source_config_target_fails(tmp_path: Path) -> None:
    fixture = _make_fixture(tmp_path)
    _change_after(
        fixture,
        path="provenance.source_config",
        value=r"E:\data\config.yaml",
        declared=True,
    )
    assert any(
        "after value is not a safe repository path" in error
        for error in _audit(fixture)
    )


def test_preview_does_not_pollute_actual_change_counts(tmp_path: Path) -> None:
    fixture = _make_fixture(tmp_path)
    old_a_text = _yaml_bytes(
        fixture.before_entries[0]["parsed_yaml"]  # type: ignore[arg-type]
    ).decode("utf-8")
    old_b_text = _yaml_bytes(
        fixture.before_entries[1]["parsed_yaml"]  # type: ignore[arg-type]
    ).decode("utf-8")
    current = {OLD_A: old_a_text, OLD_B: old_b_text}

    metrics, errors = preview_batch_migration(
        fixture.root,
        1,
        fixture.plan_path,
        tracked_yaml_override={OLD_A, OLD_B, LATER_OLD, MANUAL_OLD},
        working_text_override=current,
        index_text_override=current,
        expected_path_changes=1,
    )

    assert errors == []
    assert metrics == {
        "PREVIEW_PATH_CHANGES_REQUIRED": 1,
        "ACTUAL_FILE_CHANGES": 0,
        "APPROVED_ACTUAL_CHANGES": 0,
        "UNAPPROVED_ACTUAL_CHANGES": 0,
        "ACTUAL_YAML_CONTENT_MODIFICATIONS": 0,
        "ACTUAL_UNAPPROVED_SEMANTIC_CHANGES": 0,
    }


def test_preview_approved_source_config_is_not_unapproved_actual(
    tmp_path: Path,
) -> None:
    fixture = _make_fixture(tmp_path)
    index_a = _yaml_bytes(
        fixture.before_entries[0]["parsed_yaml"]  # type: ignore[arg-type]
    ).decode("utf-8")
    working_payload = copy.deepcopy(fixture.before_entries[0]["parsed_yaml"])
    working_payload["provenance"]["source_config"] = NEW_B  # type: ignore[index]
    working_a = _yaml_bytes(working_payload).decode("utf-8")
    old_b = _yaml_bytes(
        fixture.before_entries[1]["parsed_yaml"]  # type: ignore[arg-type]
    ).decode("utf-8")

    metrics, errors = preview_batch_migration(
        fixture.root,
        1,
        fixture.plan_path,
        tracked_yaml_override={OLD_A, OLD_B, LATER_OLD, MANUAL_OLD},
        working_text_override={OLD_A: working_a, OLD_B: old_b},
        index_text_override={OLD_A: index_a, OLD_B: old_b},
    )

    assert errors == []
    assert metrics["ACTUAL_FILE_CHANGES"] == 1
    assert metrics["APPROVED_ACTUAL_CHANGES"] == 1
    assert metrics["UNAPPROVED_ACTUAL_CHANGES"] == 0


def test_actual_approved_source_config_change_is_declared(tmp_path: Path) -> None:
    fixture = _make_fixture(tmp_path)
    _change_after(
        fixture,
        path="provenance.source_config",
        value=NEW_B,
        declared=True,
    )
    assert fixture.move_rows[0]["yaml_content_changed"] == "YES"
    assert fixture.move_rows[0]["approved_changed_keys"] == "source_config"
    assert fixture.semantic_rows[0]["changed_keys"] == "source_config"
    assert _audit(fixture) == []


def test_source_config_plus_non_approved_key_fails(tmp_path: Path) -> None:
    fixture = _make_fixture(tmp_path)
    _change_after_values(
        fixture,
        values={
            "provenance.source_config": NEW_B,
            "seed": 99,
        },
        approved_keys={"source_config", "seed"},
    )
    assert any(
        "non-approved semantic keys" in error for error in _audit(fixture)
    )


def test_unchanged_paper_aligned_configs_remain_identical(tmp_path: Path) -> None:
    fixture = _make_fixture(tmp_path)
    for plan_row, move_row in zip(fixture.plan_rows[:2], fixture.move_rows):
        plan_row["implementation"] = "paper_aligned"
        move_row["implementation"] = "paper_aligned"
    _persist(fixture)

    assert all(row["byte_identical"] == "YES" for row in fixture.semantic_rows)
    assert all(row["semantic_identical"] == "YES" for row in fixture.semantic_rows)
    assert _audit(fixture) == []


def test_later_documented_reference_update_preserves_batch_regression(
    tmp_path: Path,
) -> None:
    fixture = _make_fixture(tmp_path)
    payload = copy.deepcopy(fixture.after_entries[0]["parsed_yaml"])
    payload["provenance"]["source_config"] = NEW_B  # type: ignore[index]
    (fixture.root / NEW_A).write_bytes(_yaml_bytes(payload))
    later_reference_path = (
        fixture.root / "docs/refactors/CONFIG_BATCH3_REFERENCE_AUDIT.csv"
    )
    later_reference_path.parent.mkdir(parents=True, exist_ok=True)
    row = {column: "" for column in BATCH3_REFERENCE_COLUMNS}
    row.update(
        {
            "old_config_path": OLD_B,
            "new_config_path": NEW_B,
            "source_file": NEW_A,
            "source_line": "1",
            "reference_type": "yaml_reference",
            "historical_reference": "NO",
            "requires_update": "YES",
            "updated": "YES",
            "remaining_reference_allowed": "NO",
            "notes": "later batch active reference rewrite",
        }
    )
    _write_csv(later_reference_path, BATCH3_REFERENCE_COLUMNS, [row])

    assert _audit(fixture) == []


def test_later_reference_record_does_not_allow_unrelated_semantic_drift(
    tmp_path: Path,
) -> None:
    fixture = _make_fixture(tmp_path)
    payload = copy.deepcopy(fixture.after_entries[0]["parsed_yaml"])
    payload["seed"] = 99  # type: ignore[index]
    (fixture.root / NEW_A).write_bytes(_yaml_bytes(payload))
    later_reference_path = (
        fixture.root / "docs/refactors/CONFIG_BATCH3_REFERENCE_AUDIT.csv"
    )
    later_reference_path.parent.mkdir(parents=True, exist_ok=True)
    row = {column: "" for column in BATCH3_REFERENCE_COLUMNS}
    row.update(
        {
            "old_config_path": OLD_B,
            "new_config_path": NEW_B,
            "source_file": NEW_A,
            "source_line": "1",
            "reference_type": "yaml_reference",
            "historical_reference": "NO",
            "requires_update": "YES",
            "updated": "YES",
            "remaining_reference_allowed": "NO",
            "notes": "later batch active reference rewrite",
        }
    )
    _write_csv(later_reference_path, BATCH3_REFERENCE_COLUMNS, [row])

    errors = _audit(fixture)
    assert any("post-move SHA does not match disk/snapshot" in error for error in errors)
    assert any("after snapshot semantics do not match disk" in error for error in errors)


def test_batch2_git_head_baseline_supports_post_move_audit(tmp_path: Path) -> None:
    fixture = _make_batch2_fixture(tmp_path)

    assert all(
        not (fixture.root / row["old_path"]).exists()
        for row in fixture.plan_rows
        if row["migration_batch"] == "Batch 2"
    )
    assert _audit(fixture, expected_total=len(fixture.tracked)) == []


def test_batch2_four_source_config_changes_are_approved(tmp_path: Path) -> None:
    fixture = _make_batch2_fixture(tmp_path)

    assert sum(
        row["yaml_content_changed"] == "YES"
        and row["approved_changed_keys"] == "source_config"
        for row in fixture.move_rows
    ) == 4
    assert sum(row["status"] == "PASS" for row in fixture.semantic_rows) == 17
    assert _audit(fixture, expected_total=len(fixture.tracked)) == []


def test_batch2_before_snapshot_must_match_git_head(tmp_path: Path) -> None:
    fixture = _make_batch2_fixture(tmp_path)
    fixture.before_entries[0]["sha256"] = "0" * 64
    _persist(fixture)

    assert any(
        "before snapshot SHA does not match Git HEAD" in error
        for error in _audit(fixture, expected_total=len(fixture.tracked))
    )


def test_batch2_snapshot_identity_mismatch_fails(tmp_path: Path) -> None:
    fixture = _make_batch2_fixture(tmp_path)
    fixture.after_entries[0]["audit_fields"]["feature_set"] = "wrong"  # type: ignore[index]
    _persist(fixture)

    assert any(
        "snapshot identity mismatch on feature_set" in error
        for error in _audit(fixture, expected_total=len(fixture.tracked))
    )


def test_valid_batch3_multidag_fixture_passes(tmp_path: Path) -> None:
    fixture = _make_batch3_fixture(tmp_path)

    assert sum(
        row["implementation"] == "unified" for row in fixture.move_rows
    ) == 13
    assert sum(
        row["implementation"] == "paper_aligned"
        for row in fixture.move_rows
    ) == 4
    assert sum(
        row["yaml_content_changed"] == "YES"
        and row["approved_changed_keys"] == "source_config"
        for row in fixture.move_rows
    ) == 4
    assert _audit(fixture, expected_total=len(fixture.tracked)) == []


def test_batch3_non_multidag_config_fails(tmp_path: Path) -> None:
    fixture = _make_batch3_fixture(tmp_path)
    batch_row = next(
        row
        for row in fixture.plan_rows
        if row["migration_batch"] == "Batch 3"
    )
    batch_row["model"] = "dialoguegcn"
    fixture.move_rows[0]["model"] = "dialoguegcn"
    _persist(fixture)

    assert any(
        "Batch 3 must contain only MultiDAG" in error
        or "Batch 3 model must be multidag_cl" in error
        for error in _audit(fixture, expected_total=len(fixture.tracked))
    )


def test_batch3_author_official_fails(tmp_path: Path) -> None:
    fixture = _make_batch3_fixture(tmp_path)
    batch_row = next(
        row
        for row in fixture.plan_rows
        if row["migration_batch"] == "Batch 3"
    )
    batch_row["implementation"] = "author_official"
    fixture.move_rows[0]["implementation"] = "author_official"
    _persist(fixture)

    assert any(
        "author_official is forbidden" in error
        for error in _audit(fixture, expected_total=len(fixture.tracked))
    )


def test_batch3_paper_aligned_marked_official_fails(tmp_path: Path) -> None:
    fixture = _make_batch3_fixture(tmp_path)
    paper_index = next(
        index
        for index, row in enumerate(fixture.move_rows)
        if row["implementation"] == "paper_aligned"
    )
    fixture.move_rows[paper_index]["is_official"] = "YES"
    _persist(fixture)

    assert any(
        "must be non-official" in error
        for error in _audit(fixture, expected_total=len(fixture.tracked))
    )


def test_batch3_wrong_lineage_placement_fails(tmp_path: Path) -> None:
    fixture = _make_batch3_fixture(tmp_path)
    batch_row = next(
        row
        for row in fixture.plan_rows
        if row["migration_batch"] == "Batch 3"
    )
    batch_row["implementation"] = "paper_aligned"
    fixture.move_rows[0]["implementation"] = "paper_aligned"
    fixture.move_rows[0]["is_paper_aligned"] = "YES"
    _persist(fixture)

    assert any(
        "wrong canonical lineage" in error
        for error in _audit(fixture, expected_total=len(fixture.tracked))
    )


@pytest.mark.parametrize(
    ("path", "value"),
    [
        ("model.curriculum.warmup_epochs", 99),
        ("model.context.window_past", 99),
        ("model.context.causal", False),
        ("model.graph.layers", 9),
    ],
)
def test_batch3_training_semantics_change_fails(
    tmp_path: Path,
    path: str,
    value: object,
) -> None:
    fixture = _make_batch3_fixture(tmp_path)
    _change_after(
        fixture,
        path=path,
        value=value,
        declared=True,
        index=5,
    )

    assert any(
        "non-approved semantic keys" in error
        for error in _audit(fixture, expected_total=len(fixture.tracked))
    )


def test_batch3_fails_if_batch2_canonical_config_is_missing(
    tmp_path: Path,
) -> None:
    fixture = _make_batch3_fixture(tmp_path)
    batch2_row = next(
        row
        for row in fixture.plan_rows
        if row["migration_batch"] == "Batch 2"
    )
    candidate = batch2_row["candidate_new_path"]
    (fixture.root / candidate).unlink()
    fixture.tracked.remove(candidate)

    assert any(
        "earlier-batch canonical YAML is missing" in error
        for error in _audit(fixture, expected_total=len(fixture.tracked))
    )


def test_batch3_fails_if_batch4_yaml_is_moved(tmp_path: Path) -> None:
    fixture = _make_batch3_fixture(tmp_path)
    batch4_row = next(
        row
        for row in fixture.plan_rows
        if row["migration_batch"] == "Batch 4"
    )
    old_path = batch4_row["old_path"]
    new_path = batch4_row["candidate_new_path"]
    (fixture.root / old_path).unlink()
    target = fixture.root / new_path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(_yaml_bytes({"seed": "moved"}))
    fixture.tracked.remove(old_path)
    fixture.tracked.add(new_path)

    assert any(
        "non-Batch 3 YAML was moved or is missing" in error
        for error in _audit(fixture, expected_total=len(fixture.tracked))
    )


def test_valid_batch4_dialoguegcn_fixture_passes(tmp_path: Path) -> None:
    fixture = _make_batch4_fixture(tmp_path)

    assert sum(
        row["implementation"] == "unified" for row in fixture.move_rows
    ) == 6
    assert sum(
        row["implementation"] == "paper_aligned"
        for row in fixture.move_rows
    ) == 4
    assert all(
        row["yaml_content_changed"] == "NO"
        for row in fixture.move_rows
    )
    assert _audit(fixture, expected_total=len(fixture.tracked)) == []


def test_batch4_non_dialoguegcn_config_fails(tmp_path: Path) -> None:
    fixture = _make_batch4_fixture(tmp_path)
    batch_row = next(
        row
        for row in fixture.plan_rows
        if row["migration_batch"] == "Batch 4"
    )
    batch_row["model"] = "mmgcn"
    fixture.move_rows[0]["model"] = "mmgcn"
    _persist(fixture)

    assert any(
        "Batch 4 must contain only DialogueGCN" in error
        or "Batch 4 model must be dialoguegcn" in error
        for error in _audit(fixture, expected_total=len(fixture.tracked))
    )


def test_batch4_author_official_fails(tmp_path: Path) -> None:
    fixture = _make_batch4_fixture(tmp_path)
    batch_row = next(
        row
        for row in fixture.plan_rows
        if row["migration_batch"] == "Batch 4"
    )
    batch_row["implementation"] = "author_official"
    fixture.move_rows[0]["implementation"] = "author_official"
    _persist(fixture)

    assert any(
        "author_official is forbidden" in error
        for error in _audit(fixture, expected_total=len(fixture.tracked))
    )


def test_batch4_paper_aligned_marked_official_fails(
    tmp_path: Path,
) -> None:
    fixture = _make_batch4_fixture(tmp_path)
    paper_index = next(
        index
        for index, row in enumerate(fixture.move_rows)
        if row["implementation"] == "paper_aligned"
    )
    fixture.move_rows[paper_index]["is_official"] = "YES"
    _persist(fixture)

    assert any(
        "must be non-official" in error
        for error in _audit(fixture, expected_total=len(fixture.tracked))
    )


def test_batch4_wrong_lineage_placement_fails(tmp_path: Path) -> None:
    fixture = _make_batch4_fixture(tmp_path)
    batch_row = next(
        row
        for row in fixture.plan_rows
        if row["migration_batch"] == "Batch 4"
    )
    batch_row["implementation"] = "paper_aligned"
    fixture.move_rows[0]["implementation"] = "paper_aligned"
    fixture.move_rows[0]["is_paper_aligned"] = "YES"
    _persist(fixture)

    assert any(
        "wrong canonical lineage" in error
        for error in _audit(fixture, expected_total=len(fixture.tracked))
    )


@pytest.mark.parametrize(
    ("path", "value"),
    [
        ("graph.window_past", 99),
        ("graph.relation_count", 99),
        ("graph.speaker_count", 99),
        ("causal", False),
        ("graph.context_mode", "full_context"),
    ],
)
def test_batch4_graph_relation_speaker_context_change_fails(
    tmp_path: Path,
    path: str,
    value: object,
) -> None:
    fixture = _make_batch4_fixture(tmp_path)
    _change_after(
        fixture,
        path=path,
        value=value,
        declared=True,
    )

    assert any(
        "non-approved semantic keys" in error
        for error in _audit(fixture, expected_total=len(fixture.tracked))
    )


def test_batch4_fails_if_batch3_canonical_config_is_missing(
    tmp_path: Path,
) -> None:
    fixture = _make_batch4_fixture(tmp_path)
    batch3_row = next(
        row
        for row in fixture.plan_rows
        if row["migration_batch"] == "Batch 3"
    )
    candidate = batch3_row["candidate_new_path"]
    (fixture.root / candidate).unlink()
    fixture.tracked.remove(candidate)

    assert any(
        "earlier-batch canonical YAML is missing" in error
        for error in _audit(fixture, expected_total=len(fixture.tracked))
    )


def test_batch4_fails_if_batch5_yaml_is_moved(tmp_path: Path) -> None:
    fixture = _make_batch4_fixture(tmp_path)
    batch5_row = next(
        row
        for row in fixture.plan_rows
        if row["migration_batch"] == "Batch 5"
    )
    old_path = batch5_row["old_path"]
    new_path = batch5_row["candidate_new_path"]
    (fixture.root / old_path).unlink()
    target = fixture.root / new_path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(_yaml_bytes({"seed": "moved"}))
    fixture.tracked.remove(old_path)
    fixture.tracked.add(new_path)

    assert any(
        "non-Batch 4 YAML was moved or is missing" in error
        for error in _audit(fixture, expected_total=len(fixture.tracked))
    )


def test_valid_batch5_gsmcc_project_variant_fixture_passes(
    tmp_path: Path,
) -> None:
    fixture = _make_batch5_fixture(tmp_path)

    assert len(fixture.move_rows) == 13
    assert sum(
        row["context_mode"] == "causal_context"
        for row in fixture.move_rows
    ) == 7
    assert sum(
        row["context_mode"] == "full_context"
        for row in fixture.move_rows
    ) == 6
    assert all(
        row["implementation"] == "project_variant"
        and row["is_project_variant"] == "YES"
        and row["is_official"] == "NO"
        for row in fixture.move_rows
    )
    assert _audit(fixture, expected_total=len(fixture.tracked)) == []


def test_batch5_non_gsmcc_config_fails(tmp_path: Path) -> None:
    fixture = _make_batch5_fixture(tmp_path)
    batch_row = next(
        row
        for row in fixture.plan_rows
        if row["migration_batch"] == "Batch 5"
    )
    batch_row["model"] = "dialoguegcn"
    fixture.move_rows[0]["model"] = "dialoguegcn"
    _persist(fixture)

    assert any(
        "Batch 5 must contain only GS-MCC" in error
        or "Batch 5 model must be gsmcc" in error
        for error in _audit(fixture, expected_total=len(fixture.tracked))
    )


@pytest.mark.parametrize(
    "implementation",
    ["unified", "paper_aligned", "author_official"],
)
def test_batch5_non_project_variant_implementation_fails(
    tmp_path: Path,
    implementation: str,
) -> None:
    fixture = _make_batch5_fixture(tmp_path)
    batch_row = next(
        row
        for row in fixture.plan_rows
        if row["migration_batch"] == "Batch 5"
    )
    batch_row["implementation"] = implementation
    fixture.move_rows[0]["implementation"] = implementation
    _persist(fixture)

    assert any(
        "Batch 5 must contain only project_variant" in error
        or "Batch 5 implementation must be project_variant" in error
        or "author_official is forbidden" in error
        for error in _audit(fixture, expected_total=len(fixture.tracked))
    )


def test_batch5_official_flag_fails(tmp_path: Path) -> None:
    fixture = _make_batch5_fixture(tmp_path)
    fixture.move_rows[0]["is_official"] = "YES"
    _persist(fixture)

    assert any(
        "must be non-official" in error
        for error in _audit(fixture, expected_total=len(fixture.tracked))
    )


def test_batch5_missing_project_variant_marker_fails(
    tmp_path: Path,
) -> None:
    fixture = _make_batch5_fixture(tmp_path)
    fixture.move_rows[0]["is_project_variant"] = "NO"
    _persist(fixture)

    assert any(
        "project_variant marker must be YES" in error
        for error in _audit(fixture, expected_total=len(fixture.tracked))
    )


@pytest.mark.parametrize(
    ("index", "wrong_context"),
    [(0, "full_context"), (7, "causal_context")],
)
def test_batch5_wrong_context_placement_fails(
    tmp_path: Path,
    index: int,
    wrong_context: str,
) -> None:
    fixture = _make_batch5_fixture(tmp_path)
    plan_row = next(
        row
        for row in fixture.plan_rows
        if row["old_path"] == fixture.move_rows[index]["old_path"]
    )
    old_new_path = plan_row["candidate_new_path"]
    wrong_new_path = old_new_path.replace(
        plan_row["context_mode"],
        wrong_context,
    )
    source = fixture.root / old_new_path
    target = fixture.root / wrong_new_path
    target.parent.mkdir(parents=True, exist_ok=True)
    source.replace(target)
    fixture.tracked.remove(old_new_path)
    fixture.tracked.add(wrong_new_path)
    plan_row["candidate_new_path"] = wrong_new_path
    fixture.move_rows[index]["new_path"] = wrong_new_path
    fixture.before_entries[index]["new_path"] = wrong_new_path
    fixture.after_entries[index]["new_path"] = wrong_new_path
    fixture.semantic_rows[index]["new_path"] = wrong_new_path
    _persist(fixture)

    assert any(
        "wrong context directory" in error
        for error in _audit(fixture, expected_total=len(fixture.tracked))
    )


@pytest.mark.parametrize(
    ("path", "value"),
    [
        ("graph.window_past", 99),
        ("model.fusion_type", "sum"),
        ("model.active_modalities", ["text"]),
        ("missing_modalities.enabled", True),
        ("causal", False),
    ],
)
def test_batch5_graph_fusion_modality_causal_drift_fails(
    tmp_path: Path,
    path: str,
    value: object,
) -> None:
    fixture = _make_batch5_fixture(tmp_path)
    _change_after(
        fixture,
        path=path,
        value=value,
        declared=True,
    )

    assert any(
        "non-approved semantic keys" in error
        for error in _audit(fixture, expected_total=len(fixture.tracked))
    )


def test_batch5_fails_if_batch4_canonical_config_is_missing(
    tmp_path: Path,
) -> None:
    fixture = _make_batch5_fixture(tmp_path)
    batch4_row = next(
        row
        for row in fixture.plan_rows
        if row["migration_batch"] == "Batch 4"
    )
    candidate = batch4_row["candidate_new_path"]
    (fixture.root / candidate).unlink()
    fixture.tracked.remove(candidate)

    assert any(
        "earlier-batch canonical YAML is missing" in error
        for error in _audit(fixture, expected_total=len(fixture.tracked))
    )


def test_batch5_fails_if_batch6_yaml_is_moved(tmp_path: Path) -> None:
    fixture = _make_batch5_fixture(tmp_path)
    batch6_row = next(
        row
        for row in fixture.plan_rows
        if row["migration_batch"] == "Batch 6"
    )
    old_path = batch6_row["old_path"]
    new_path = batch6_row["candidate_new_path"]
    (fixture.root / old_path).unlink()
    target = fixture.root / new_path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(_yaml_bytes({"seed": "moved"}))
    fixture.tracked.remove(old_path)
    fixture.tracked.add(new_path)

    assert any(
        "non-Batch 5 YAML was moved or is missing" in error
        for error in _audit(fixture, expected_total=len(fixture.tracked))
    )


def test_batch5_active_old_reference_fails(tmp_path: Path) -> None:
    fixture = _make_batch5_fixture(tmp_path)
    old_path = fixture.move_rows[0]["old_path"]
    errors = _audit(
        fixture,
        tracked_text={"scripts/live.py": f'CONFIG = "{old_path}"\n'},
        expected_total=len(fixture.tracked),
    )
    assert any(
        "production code retains old config path" in error
        for error in errors
    )


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


def test_earlier_batch_audit_accepts_fully_recorded_later_batch(
    tmp_path: Path,
) -> None:
    fixture = _make_fixture(tmp_path)
    (fixture.root / LATER_OLD).unlink()
    later_target = fixture.root / LATER_NEW
    later_target.parent.mkdir(parents=True, exist_ok=True)
    later_target.write_bytes(_yaml_bytes({"seed": 3}))
    fixture.tracked.remove(LATER_OLD)
    fixture.tracked.add(LATER_NEW)
    _write_csv(
        fixture.root / "CONFIG_BATCH2_MOVES.csv",
        ("old_path", "new_path"),
        [{"old_path": LATER_OLD, "new_path": LATER_NEW}],
    )

    assert _audit(fixture) == []


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


def test_move_reference_counts_must_match_reference_audit(
    tmp_path: Path,
) -> None:
    fixture = _make_batch2_fixture(tmp_path)
    fixture.move_rows[0]["active_reference_count"] = "1"
    _persist(fixture)

    assert any(
        "active_reference_count mismatch" in error
        for error in _audit(fixture, expected_total=len(fixture.tracked))
    )


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


def test_arbitrary_yaml_cannot_be_whitelisted_as_historical(
    tmp_path: Path,
) -> None:
    fixture = _make_fixture(tmp_path)
    row = {column: "" for column in REFERENCE_COLUMNS}
    row.update(
        {
            "old_config_path": OLD_A,
            "source_file": "configs/live.yaml",
            "source_line": "3",
            "reference_type": "yaml_reference",
            "historical_reference": "YES",
            "updated": "NO",
            "new_config_path": NEW_A,
            "remaining_reference_allowed": "YES",
            "notes": "not present in the Phase 4A reference graph",
        }
    )
    fixture.reference_rows.append(row)
    _persist(fixture)

    assert any(
        "historical reference is outside the approved categories" in error
        for error in _audit(
            fixture,
            tracked_text={"configs/live.yaml": f"config: {OLD_A}\n"},
        )
    )


def test_manual_review_yaml_moved_fails(tmp_path: Path) -> None:
    fixture = _make_fixture(tmp_path)
    (fixture.root / MANUAL_OLD).unlink()
    assert any("manual-review YAML was moved" in error for error in _audit(fixture))

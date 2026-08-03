from __future__ import annotations

import shutil
import uuid
from collections import Counter
from pathlib import Path

import pytest
import yaml

from scripts.experiments.prepare_formal_full16_repaired import (
    EXPECTED_IDENTITIES,
    REPAIR_SIGNATURES,
    prepare_formal_full16_repaired,
)


ROOT = Path(__file__).resolve().parents[2]
MATRIX_PATH = Path(
    "configs/benchmarks/formal_rerun/iemocap_clean/"
    "full16_repaired_seed42.yaml"
)
LONG32_PATHS = (
    ROOT / "configs/benchmarks/long_training/iemocap_clean/primary_seed42.yaml",
    ROOT
    / "configs/benchmarks/long_training/iemocap_clean/base/"
    "mmgcn_full_context.yaml",
    ROOT
    / "configs/benchmarks/long_training/iemocap_clean/base/"
    "multidag_cl_full_context.yaml",
    ROOT
    / "configs/benchmarks/long_training/iemocap_clean/base/"
    "dialoguegcn_full_context.yaml",
    ROOT
    / "configs/benchmarks/long_training/iemocap_clean/base/"
    "gsmcc_project_variant_full_context.yaml",
)
REPAIR_SOURCE_PATHS = (
    ROOT
    / "configs/benchmarks/repairs/dialoguegcn_gsmcc_full/repair_matrix.yaml",
    ROOT / "scripts/workflows/benchmarks/prepare_full_repair.py",
)
REPAIR_DIAGNOSTIC_RELATIVE_PATHS = (
    Path(
        "outputs/20260727/dialoguegcn_gsmcc_full_repair/manifests/batches/"
        "formal_repair_diagnostic_20260727_120555/resolved_configs/"
        "dialoguegcn_ses03_delayed_early_stop.yaml"
    ),
    Path(
        "outputs/20260727/dialoguegcn_gsmcc_full_repair/manifests/batches/"
        "formal_repair_diagnostic_20260727_120555/resolved_configs/"
        "dialoguegcn_ses04_delayed_early_stop.yaml"
    ),
    Path(
        "outputs/20260727/dialoguegcn_gsmcc_full_repair/manifests/batches/"
        "formal_repair_diagnostic_20260727_120555/resolved_configs/"
        "gsmcc_ses03_best_candidate_control.yaml"
    ),
    Path(
        "outputs/20260727/dialoguegcn_gsmcc_full_repair/manifests/batches/"
        "formal_repair_diagnostic_20260727_120555/resolved_configs/"
        "gsmcc_ses04_best_candidate_control.yaml"
    ),
)


@pytest.fixture
def workspace_output_base() -> Path:
    parent = ROOT / "tests" / "_ff16"
    parent.mkdir(parents=True, exist_ok=True)
    case_root = parent / uuid.uuid4().hex[:8]
    case_root.mkdir()
    relative_output = case_root.relative_to(ROOT) / "outputs"
    try:
        yield relative_output
    finally:
        shutil.rmtree(case_root)
        try:
            parent.rmdir()
        except OSError:
            pass


def _load_yaml(path: Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as file:
        value = yaml.safe_load(file)
    assert isinstance(value, dict)
    return value


def _snapshot(paths: tuple[Path, ...]) -> dict[Path, bytes]:
    return {path: path.read_bytes() for path in paths}


def _optional_snapshot(paths: tuple[Path, ...]) -> dict[Path, bytes | None]:
    return {path: path.read_bytes() if path.is_file() else None for path in paths}


def test_check_expands_exact_full16_without_writing(
    workspace_output_base: Path,
) -> None:
    absolute_output = ROOT / workspace_output_base
    assert not absolute_output.exists()
    result = prepare_formal_full16_repaired(
        MATRIX_PATH,
        "check",
        "20260803",
        root=ROOT,
        batch_id="pytest_full16_check",
        output_base_override=workspace_output_base,
    )

    assert not absolute_output.exists()
    assert result["expanded_run_count"] == 16
    assert result["context_counts"] == Counter({"full_context": 16})
    assert result["context_counts"]["causal_context"] == 0
    assert result["model_counts"] == Counter(
        {
            "mmgcn": 4,
            "multidag_cl": 4,
            "dialoguegcn": 4,
            "gsmcc_project_variant": 4,
        }
    )
    assert result["validation_counts"] == Counter(
        {"Ses01": 4, "Ses02": 4, "Ses03": 4, "Ses04": 4}
    )
    assert result["test_session"] == "Ses05"
    assert result["seed_values"] == [42]
    assert result["test_selection_leakage_found"] == 0
    assert result["output_collision_count"] == 0
    assert result["duplicate_run_id_count"] == 0
    assert result["duplicate_output_root_count"] == 0
    assert result["runtime_validation_count"] == 16
    assert result["training_started"] == 0
    assert result["source_files_unchanged"] is True
    assert len(result["commands"]) == 16
    assert len(set(result["commands"])) == 16

    run_ids = [record["run_id"] for record in result["records"]]
    output_roots = [record["output_root"] for record in result["records"]]
    assert len(set(run_ids)) == 16
    assert len(set(output_roots)) == 16
    assert "ff16r_mmgcn_full_context_val_ses01_s42" in run_ids
    assert "ff16r_multidag_cl_full_context_val_ses01_s42" in run_ids
    assert "ff16r_dialoguegcn_full_context_val_ses01_s42" in run_ids
    assert (
        "ff16r_gsmcc_project_variant_full_context_val_ses01_s42" in run_ids
    )

    feature_keys = set()
    feature_hashes = set()
    for record in result["records"]:
        config = record["config"]
        metadata = config["formal_full16"]
        assert record["context_mode"] == "full_context"
        assert record["validation_session"] in {
            "Ses01",
            "Ses02",
            "Ses03",
            "Ses04",
        }
        assert record["test_session"] == "Ses05"
        assert record["seed"] == 42
        assert config["dataset"]["outer_test_session"] == "Ses05"
        assert config["training"]["select_best_by"] == "val_weighted_f1"
        assert config["protocol"]["checkpoint_selection_split"] == "validation"
        assert config["protocol"]["test_split_used_for_selection"] is False
        assert metadata["test_selection_leakage"] is False
        assert metadata["author_official_reproduction"] is False
        assert metadata["implementation_identity"] == EXPECTED_IDENTITIES[
            record["model_family"]
        ]
        feature_keys.add(config["provenance"]["feature_registry_key"])
        feature_hashes.add(config["dataset"]["feature_sha256"])

    assert feature_keys == {"clean_roberta_v1"}
    assert feature_hashes == {
        "c604c557bc3fbb129ca02b2acd57166b669a89ef76ff0cea1e14f9cb206324bf"
    }


def test_repair_signatures_are_uniform_and_unchanged_models_reuse_old_bases(
    workspace_output_base: Path,
) -> None:
    result = prepare_formal_full16_repaired(
        MATRIX_PATH,
        "check",
        "20260803",
        root=ROOT,
        batch_id="pytest_full16_signatures",
        output_base_override=workspace_output_base,
    )
    by_model: dict[str, list[dict]] = {}
    for record in result["records"]:
        by_model.setdefault(record["model_family"], []).append(record["config"])

    for model_family in ("dialoguegcn", "gsmcc_project_variant"):
        signatures = {
            tuple(config["formal_full16"]["training_signature"].items())
            for config in by_model[model_family]
        }
        assert signatures == {tuple(REPAIR_SIGNATURES[model_family].items())}

    source_paths = {
        "mmgcn": LONG32_PATHS[1],
        "multidag_cl": LONG32_PATHS[2],
    }
    for model_family, source_path in source_paths.items():
        source = _load_yaml(source_path)
        for config in by_model[model_family]:
            assert config["model"] == source["model"]
            assert config["training"] == source["training"]
            assert config["optimizer"] == source["optimizer"]
            assert config["scheduler"] == source["scheduler"]


def test_prepare_writes_only_resolved_configs_and_commands_and_refuses_reuse(
    workspace_output_base: Path,
) -> None:
    long32_before = _snapshot(LONG32_PATHS)
    repair_sources_before = _snapshot(REPAIR_SOURCE_PATHS)
    diagnostic_paths = tuple(ROOT / path for path in REPAIR_DIAGNOSTIC_RELATIVE_PATHS)
    diagnostic_before = _optional_snapshot(diagnostic_paths)

    result = prepare_formal_full16_repaired(
        MATRIX_PATH,
        "prepare",
        "20260803",
        root=ROOT,
        batch_id="pytest_ff16",
        output_base_override=workspace_output_base,
    )

    experiment_root = ROOT / result["experiment_root"]
    for name in ("runs", "logs", "manifests", "reports", "analysis"):
        assert (experiment_root / name).is_dir()
    assert (experiment_root / "review").is_dir()

    manifest_root = ROOT / result["manifest_root"]
    assert {path.name for path in manifest_root.iterdir()} == {
        "resolved_configs",
        "commands.txt",
    }
    resolved_files = sorted((manifest_root / "resolved_configs").glob("*.yaml"))
    assert len(resolved_files) == 16
    commands = (manifest_root / "commands.txt").read_text(
        encoding="utf-8"
    ).splitlines()
    assert len(commands) == 16
    assert len(set(commands)) == 16
    assert all("scripts/workflows/paper_aligned/train.py" in line for line in commands)

    with pytest.raises(FileExistsError, match="target batch already exists"):
        prepare_formal_full16_repaired(
            MATRIX_PATH,
            "prepare",
            "20260803",
            root=ROOT,
            batch_id="pytest_ff16",
            output_base_override=workspace_output_base,
        )

    assert _snapshot(LONG32_PATHS) == long32_before
    assert _snapshot(REPAIR_SOURCE_PATHS) == repair_sources_before
    assert _optional_snapshot(diagnostic_paths) == diagnostic_before


def test_prepare_refuses_existing_run_output(
    workspace_output_base: Path,
) -> None:
    collision = (
        ROOT
        / workspace_output_base
        / "20260803"
        / "formal_full16_repaired_seed42"
        / "runs"
        / "ff16r_mmgcn_full_context_val_ses01_s42"
    )
    collision.mkdir(parents=True)
    with pytest.raises(ValueError, match="output collisions found: 1"):
        prepare_formal_full16_repaired(
            MATRIX_PATH,
            "prepare",
            "20260803",
            root=ROOT,
            batch_id="formal_full16_repaired_collision",
            output_base_override=workspace_output_base,
        )

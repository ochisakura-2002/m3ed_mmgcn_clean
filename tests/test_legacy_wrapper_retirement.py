from __future__ import annotations

import ast
import csv
import importlib
import importlib.util
import os
from pathlib import Path
import subprocess
import sys

import pytest

from models.dialoguegcn.paper_aligned import OriginalReproDialogueGCN
from models.dialoguegcn.unified import CausalDialogueGCNBaseline
from models.gsmcc.project_variant.causal import CausalGSMCCInspiredBaseline
from models.gsmcc.project_variant.full_context import ProjectPaperOrientedGSMCC
from models.mmgcn.paper_aligned import OriginalReproMMGCN
from models.mmgcn.unified.mm_gcn import M3EDMMGCN
from models.multidag_cl.paper_aligned import OriginalReproMultiDAGCL
from models.multidag_cl.unified import MultiDAGCLBaseline
from models.registry.causal import build_new_causal_baseline
from models.registry.paper_aligned import build_original_repro_model


ROOT = Path(__file__).resolve().parents[1]
INVENTORY = (
    ROOT / "docs" / "refactors" / "LEGACY_WRAPPER_RETIREMENT_INVENTORY.csv"
)

CANONICAL_MODULES = (
    "models.common.causal_graph",
    "models.registry.causal",
    "models.registry.paper_aligned",
    "models.mmgcn.unified",
    "models.mmgcn.paper_aligned",
    "models.multidag_cl.unified",
    "models.multidag_cl.paper_aligned",
    "models.dialoguegcn.unified",
    "models.dialoguegcn.paper_aligned",
    "models.gsmcc.project_variant.causal",
    "models.gsmcc.project_variant.full_context",
    "models.simple_mlp.model",
    "models.experimental.sdt",
    "scripts.models.mmgcn.unified.train",
    "scripts.models.multidag_cl.unified.train",
    "scripts.evaluation.unified_checkpoint",
    "scripts.runtime.causal_graph",
    "scripts.runtime.paper_aligned",
    "scripts.workflows.run_pipeline",
    "scripts.workflows.causal_graph.train",
    "scripts.workflows.paper_aligned.train",
)

CANONICAL_CLI_PATHS = (
    "scripts/models/mmgcn/unified/train.py",
    "scripts/models/multidag_cl/unified/train.py",
    "scripts/models/simple_mlp/train.py",
    "scripts/evaluation/unified_checkpoint.py",
    "scripts/workflows/run_pipeline.py",
    "scripts/workflows/causal_graph/train.py",
    "scripts/workflows/paper_aligned/train.py",
)


def _inventory_rows() -> list[dict[str, str]]:
    with INVENTORY.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _tracked_python_paths() -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files", "models/*.py", "models/**/*.py", "scripts/*.py", "scripts/**/*.py"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return [ROOT / line for line in result.stdout.splitlines() if line]


def test_all_confirmed_legacy_wrapper_files_are_absent() -> None:
    rows = _inventory_rows()
    assert len(rows) == 98
    assert sum(row["old_path"].startswith("models/") for row in rows) == 34
    assert sum(row["old_path"].startswith("scripts/") for row in rows) == 64
    assert all(row["safe_to_delete"] == "YES" for row in rows)
    for row in rows:
        assert not (ROOT / row["old_path"]).exists(), row["old_path"]


@pytest.mark.parametrize(
    "legacy_module",
    (
        "models.baselines",
        "models.baselines.mmgcn.mm_gcn",
        "scripts.train_mmgcn",
        "scripts.baselines.train_multidag_cl",
        "scripts.analysis.analyze_original_merc_results",
    ),
)
def test_legacy_module_paths_are_not_importable(legacy_module: str) -> None:
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(legacy_module)
    try:
        spec = importlib.util.find_spec(legacy_module)
    except ModuleNotFoundError:
        spec = None
    assert spec is None


@pytest.mark.parametrize("module_name", CANONICAL_MODULES)
def test_canonical_modules_import(module_name: str) -> None:
    assert importlib.import_module(module_name) is not None


def test_canonical_registry_symbols_resolve() -> None:
    assert callable(build_new_causal_baseline)
    assert callable(build_original_repro_model)
    for symbol in (
        OriginalReproDialogueGCN,
        CausalDialogueGCNBaseline,
        CausalGSMCCInspiredBaseline,
        ProjectPaperOrientedGSMCC,
        OriginalReproMMGCN,
        OriginalReproMultiDAGCL,
        MultiDAGCLBaseline,
    ):
        assert callable(symbol)


def test_mmgcn_strict_state_dict_load_protocol_is_unchanged() -> None:
    constructor_args = {
        "text_dim": 5,
        "audio_dim": 4,
        "visual_dim": 3,
        "hidden_dim": 6,
        "num_classes": 2,
        "num_layers": 1,
        "dropout": 0.0,
    }
    source = M3EDMMGCN(**constructor_args)
    target = M3EDMMGCN(**constructor_args)
    state_dict = source.state_dict()

    assert tuple(state_dict) == tuple(target.state_dict())
    target.load_state_dict(state_dict, strict=True)


@pytest.mark.parametrize(
    ("case_name", "config_path"),
    (
        (
            "MMGCN_UNIFIED_FULL_CONTEXT",
            "configs/mmgcn/unified/iemocap/full_context/"
            "legacy_mmgcn_features/val_official_prefix.yaml",
        ),
        (
            "MMGCN_UNIFIED_CAUSAL_CONTEXT",
            "configs/mmgcn/unified/iemocap/causal_context/"
            "legacy_mmgcn_features/val_official_prefix.yaml",
        ),
        (
            "MULTIDAG_CL_UNIFIED_CAUSAL_CONTEXT",
            "configs/multidag_cl/unified/synthetic/causal_context/"
            "synthetic/smoke.yaml",
        ),
        (
            "DIALOGUEGCN_UNIFIED_CAUSAL_CONTEXT",
            "configs/dialoguegcn/unified/iemocap/causal_context/"
            "legacy_mmgcn_features/val_official_prefix.yaml",
        ),
        (
            "GSMCC_PROJECT_VARIANT_CAUSAL_CONTEXT",
            "configs/gsmcc/project_variant/iemocap/causal_context/"
            "legacy_mmgcn_features/val_official_prefix.yaml",
        ),
        (
            "GSMCC_PROJECT_VARIANT_FULL_CONTEXT",
            "configs/gsmcc/project_variant/iemocap/full_context/"
            "legacy_mmgcn_features/screening.yaml",
        ),
    ),
)
def test_safe_config_model_dry_run_without_epochs_or_outputs(
    case_name: str,
    config_path: str,
) -> None:
    from models.registry.causal import build_new_causal_baseline
    from models.registry.paper_aligned import build_original_repro_model
    from scripts.models.mmgcn.unified.train import (
        build_model as build_mmgcn,
        load_yaml_config as load_mmgcn,
    )
    from scripts.models.multidag_cl.unified.smoke import (
        _build_model as build_multidag_smoke,
        _validate_config as validate_multidag_smoke,
    )
    from scripts.runtime.causal_graph import (
        load_yaml_config as load_causal,
        normalized_training_config as normalize_causal,
        validate_runtime_config as validate_causal,
    )
    from scripts.runtime.paper_aligned import (
        load_yaml_config as load_paper,
        normalized_training_config as normalize_paper,
        validate_runtime_config as validate_paper,
    )
    from utils.io import load_yaml

    path = ROOT / config_path
    assert path.is_file()
    if case_name.startswith("MMGCN_"):
        model = build_mmgcn(load_mmgcn(path))
    elif case_name.startswith("MULTIDAG_"):
        config = load_yaml(str(path))
        validate_multidag_smoke(config)
        model = build_multidag_smoke(config)
    elif case_name.endswith("FULL_CONTEXT"):
        config = normalize_paper(load_paper(path))
        validate_paper(config)
        model = build_original_repro_model(config)
    else:
        config = normalize_causal(load_causal(path))
        validate_causal(config)
        model = build_new_causal_baseline(config)

    assert model is not None
    print(
        f"DRY_RUN_CASE={case_name} CONFIG={config_path} "
        f"MODEL={type(model).__name__} EPOCHS_STARTED=0 OUTPUTS_CREATED=0"
    )


@pytest.mark.parametrize("relative_path", CANONICAL_CLI_PATHS)
def test_canonical_cli_help_starts_without_training(relative_path: str) -> None:
    environment = dict(os.environ)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    result = subprocess.run(
        [sys.executable, relative_path, "--help"],
        cwd=ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout.lower()


def test_training_and_evaluation_command_targets_remain_canonical() -> None:
    from scripts.workflows import run_pipeline
    from scripts.workflows.ablations import evaluate_missing_modalities
    from scripts.workflows.benchmarks import run_causal8_original8

    assert set(run_pipeline.TRAIN_SCRIPT_REGISTRY.values()) - {
        "scripts/train_gsmcc.py"
    } <= {
        "scripts/models/mmgcn/unified/train.py",
        "scripts/models/simple_mlp/train.py",
        "scripts/models/multidag_cl/unified/train.py",
        "scripts/workflows/causal_graph/train.py",
    }
    assert set(run_pipeline.EVALUATE_SCRIPT_REGISTRY.values()) == {
        "scripts/evaluation/unified_checkpoint.py",
        "scripts/models/multidag_cl/unified/evaluate.py",
        "scripts/workflows/causal_graph/evaluate.py",
    }
    assert set(evaluate_missing_modalities.EVALUATE_SCRIPT_BY_MODEL.values()) == {
        "scripts/evaluation/unified_checkpoint.py",
        "scripts/models/multidag_cl/unified/evaluate.py",
    }
    assert {job.entrypoint.as_posix() for job in run_causal8_original8.build_run_plan()} == {
        "scripts/workflows/run_pipeline.py",
        "scripts/workflows/paper_aligned/train.py",
    }


def test_configs_do_not_reference_retired_script_paths() -> None:
    retired_script_paths = {
        row["old_path"]
        for row in _inventory_rows()
        if row["old_path"].startswith("scripts/")
    }
    for path in (ROOT / "configs").rglob("*.yaml"):
        text = path.read_text(encoding="utf-8")
        assert not retired_script_paths.intersection(text.split()), path
        for retired_path in retired_script_paths:
            assert retired_path not in text, (path, retired_path)


def test_tracked_models_and_scripts_contain_no_compatibility_loader_or_thin_main_forwarder() -> None:
    for path in _tracked_python_paths():
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
        assert "_load_compat_module" not in source, path
        if path.name == "__init__.py":
            continue
        top_level_definitions = [
            node
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        ]
        imports_main = any(
            isinstance(node, ast.ImportFrom)
            and any(alias.name == "main" for alias in node.names)
            for node in tree.body
        )
        calls_main = any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "main"
            for node in ast.walk(tree)
        )
        assert top_level_definitions or not (imports_main and calls_main), path

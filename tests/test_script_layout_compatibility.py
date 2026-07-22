from __future__ import annotations

import ast
import importlib
import os
from pathlib import Path
import runpy
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]

ENTRYPOINTS = {
    "scripts.train_mmgcn": "scripts.models.mmgcn.unified.train",
    "scripts.train_simple_mlp": "scripts.models.simple_mlp.train",
    "scripts.evaluate_checkpoint": "scripts.evaluation.unified_checkpoint",
    "scripts.train": "scripts.workflows.train_from_config",
    "scripts.run_experiment_pipeline": "scripts.workflows.run_pipeline",
    "scripts.run_original_merc_reproduction_pipeline": (
        "scripts.workflows.paper_aligned.run_reproduction"
    ),
    "scripts.baselines.train_multidag_cl": (
        "scripts.models.multidag_cl.unified.train"
    ),
    "scripts.baselines.evaluate_multidag_cl_checkpoint": (
        "scripts.models.multidag_cl.unified.evaluate"
    ),
    "scripts.baselines.train_multidag_cl_smoke": (
        "scripts.models.multidag_cl.unified.smoke"
    ),
    "scripts.baselines.new_causal_graph_runtime": "scripts.runtime.causal_graph",
    "scripts.baselines.train_new_causal_graph_baseline": (
        "scripts.workflows.causal_graph.train"
    ),
    "scripts.baselines.evaluate_new_causal_graph_checkpoint": (
        "scripts.workflows.causal_graph.evaluate"
    ),
    "scripts.baselines.original_merc_runtime": "scripts.runtime.paper_aligned",
    "scripts.baselines.train_original_merc_baseline": (
        "scripts.workflows.paper_aligned.train"
    ),
    "scripts.baselines.evaluate_original_merc_checkpoint": (
        "scripts.workflows.paper_aligned.evaluate"
    ),
    "scripts.baselines.run_original_merc_pipeline": (
        "scripts.workflows.paper_aligned.run_pipeline"
    ),
    "scripts.experiments.run_causal8_original8_formal": (
        "scripts.workflows.benchmarks.run_causal8_original8"
    ),
    "scripts.experiments.evaluate_missing_modalities": (
        "scripts.workflows.ablations.evaluate_missing_modalities"
    ),
    "scripts.experiments.generate_modality_ablation_configs": (
        "scripts.workflows.ablations.generate_modality_configs"
    ),
}

CLI_CASES = [
    (
        "scripts/train_mmgcn.py",
        "scripts/models/mmgcn/unified/train.py",
        ("--config", "--experiment-date"),
    ),
    (
        "scripts/baselines/train_multidag_cl.py",
        "scripts/models/multidag_cl/unified/train.py",
        ("--config", "--experiment-date"),
    ),
    (
        "scripts/baselines/train_new_causal_graph_baseline.py",
        "scripts/workflows/causal_graph/train.py",
        ("--config", "--output-root", "--experiment-date"),
    ),
    (
        "scripts/baselines/train_original_merc_baseline.py",
        "scripts/workflows/paper_aligned/train.py",
        ("--config", "--resume", "--dry-run"),
    ),
    (
        "scripts/evaluate_checkpoint.py",
        "scripts/evaluation/unified_checkpoint.py",
        ("--checkpoint", "--split", "--active-modalities"),
    ),
    (
        "scripts/experiments/run_causal8_original8_formal.py",
        "scripts/workflows/benchmarks/run_causal8_original8.py",
        ("--start-index", "--end-index", "--continue-on-error"),
    ),
]


def _module_path(module_name: str) -> Path:
    return ROOT / (module_name.replace(".", "/") + ".py")


def _import_targets(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    targets: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            targets.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            targets.append(node.module)
    return targets


@pytest.mark.parametrize(("legacy_name", "canonical_name"), ENTRYPOINTS.items())
def test_legacy_import_is_canonical_module(
    legacy_name: str,
    canonical_name: str,
) -> None:
    legacy = importlib.import_module(legacy_name)
    canonical = importlib.import_module(canonical_name)
    assert legacy is canonical


@pytest.mark.parametrize(("legacy_name", "canonical_name"), ENTRYPOINTS.items())
def test_wrapper_is_thin_and_does_not_import_models(
    legacy_name: str,
    canonical_name: str,
) -> None:
    path = _module_path(legacy_name)
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    assert "argparse" not in _import_targets(path)
    assert not any(target == "models" or target.startswith("models.") for target in _import_targets(path))
    assert canonical_name in source
    assert not any(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) for node in ast.walk(tree)
    )


def test_canonical_execution_scripts_have_no_legacy_model_imports() -> None:
    for canonical_name in ENTRYPOINTS.values():
        path = _module_path(canonical_name)
        targets = _import_targets(path)
        assert not any(
            target == "models.baselines" or target.startswith("models.baselines.")
            for target in targets
        ), path


@pytest.mark.parametrize(("legacy_path", "canonical_path", "options"), CLI_CASES)
def test_old_and_new_help_are_safe_and_keep_key_options(
    legacy_path: str,
    canonical_path: str,
    options: tuple[str, ...],
) -> None:
    environment = dict(os.environ)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    for script_path in (legacy_path, canonical_path):
        result = subprocess.run(
            [sys.executable, script_path, "--help"],
            cwd=ROOT,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
        for option in options:
            assert option in result.stdout


@pytest.mark.parametrize(
    ("legacy_path", "canonical_name"),
    [
        ("scripts/train_mmgcn.py", "scripts.models.mmgcn.unified.train"),
        (
            "scripts/baselines/train_multidag_cl.py",
            "scripts.models.multidag_cl.unified.train",
        ),
        (
            "scripts/baselines/train_new_causal_graph_baseline.py",
            "scripts.workflows.causal_graph.train",
        ),
        (
            "scripts/baselines/train_original_merc_baseline.py",
            "scripts.workflows.paper_aligned.train",
        ),
        (
            "scripts/evaluate_checkpoint.py",
            "scripts.evaluation.unified_checkpoint",
        ),
        (
            "scripts/experiments/run_causal8_original8_formal.py",
            "scripts.workflows.benchmarks.run_causal8_original8",
        ),
    ],
)
def test_legacy_cli_delegates_to_canonical_main_without_training(
    monkeypatch: pytest.MonkeyPatch,
    legacy_path: str,
    canonical_name: str,
) -> None:
    canonical = importlib.import_module(canonical_name)
    monkeypatch.setattr(canonical, "main", lambda: 17)
    with pytest.raises(SystemExit) as exit_info:
        runpy.run_path(str(ROOT / legacy_path), run_name="__main__")
    assert exit_info.value.code == 17


def test_canonical_paths_and_subprocess_targets_resolve_from_repo_root() -> None:
    from scripts.workflows import run_pipeline
    from scripts.workflows.ablations import evaluate_missing_modalities
    from scripts.workflows.benchmarks import run_causal8_original8

    assert run_pipeline.PROJECT_ROOT == ROOT
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
    jobs = run_causal8_original8.build_run_plan()
    assert len(jobs) == 16
    assert {job.entrypoint.as_posix() for job in jobs} == {
        "scripts/workflows/run_pipeline.py",
        "scripts/workflows/paper_aligned/train.py",
    }
    for relative_path in (
        *run_pipeline.TRAIN_SCRIPT_REGISTRY.values(),
        *run_pipeline.EVALUATE_SCRIPT_REGISTRY.values(),
        *evaluate_missing_modalities.EVALUATE_SCRIPT_BY_MODEL.values(),
    ):
        if relative_path == "scripts/train_gsmcc.py":
            continue
        assert (ROOT / relative_path).is_file()

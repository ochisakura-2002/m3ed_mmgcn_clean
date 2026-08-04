from __future__ import annotations

import ast
import re
from pathlib import Path

import models.multidag_cl.paper_reimplementation as public_package
import models.multidag_cl.paper_reimplementation.curriculum as curriculum_package
from models.multidag_cl.paper_reimplementation.model import (
    MultiDAGCLPaperReimplementation,
)


SOURCE_ROOT = Path("models/multidag_cl/paper_reimplementation")
EXPECTED_SOURCE_FILES = {
    "__init__.py",
    "attention.py",
    "config.py",
    "contracts.py",
    "curriculum/__init__.py",
    "curriculum/buckets.py",
    "curriculum/difficulty.py",
    "curriculum/schedule.py",
    "dag_layer.py",
    "encoders.py",
    "graph.py",
    "model.py",
}


def _source_files() -> list[Path]:
    return sorted(SOURCE_ROOT.rglob("*.py"))


def test_lineage_contains_only_the_expected_independent_core_files() -> None:
    relative = {path.relative_to(SOURCE_ROOT).as_posix() for path in _source_files()}
    assert relative == EXPECTED_SOURCE_FILES


def test_static_imports_exclude_official_clone_old_cores_and_runtime_dependencies() -> None:
    forbidden_prefixes = (
        "models.multidag_cl.unified",
        "models.multidag_cl.paper_aligned",
        "scripts.models.multidag_cl",
        "scripts.runtime",
        "scripts.workflows",
        "datasets",
        "yaml",
        "argparse",
        "pandas",
        "sklearn",
        "importlib",
    )
    for path in _source_files():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                imported = [node.module or ""] if node.level == 0 else []
            else:
                imported = []
            for module in imported:
                assert not module.startswith(forbidden_prefixes), (path, module)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                assert node.func.id != "__import__", path


def test_source_has_no_absolute_paths_or_path_injection() -> None:
    windows_absolute = re.compile(r"[A-Za-z]:[\\/]")
    for path in _source_files():
        text = path.read_text(encoding="utf-8")
        assert windows_absolute.search(text) is None, path
        assert "sys.path" not in text, path
        assert ".cuda(" not in text, path


def test_public_api_exports_only_the_frozen_symbols() -> None:
    assert public_package.__all__ == [
        "MultiDAGCLConfig",
        "MultiDAGCLPaperReimplementation",
    ]
    assert curriculum_package.__all__ == [
        "CurriculumBucketPartitioner",
        "CurriculumSchedule",
        "DialogueDifficultyScorer",
    ]


def test_identity_is_reimplementation_and_never_an_official_implementation_claim() -> None:
    assert MultiDAGCLPaperReimplementation.model_key == "multidag_cl_paper_reimplementation"
    assert MultiDAGCLPaperReimplementation.implementation_identity == "paper_reimplementation"
    forbidden_assignment = re.compile(
        r"implementation_identity\s*=\s*['\"]author_official['\"]"
    )
    for path in _source_files():
        assert forbidden_assignment.search(path.read_text(encoding="utf-8")) is None, path

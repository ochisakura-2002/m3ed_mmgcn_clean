from __future__ import annotations

import ast
import csv
import importlib
import os
from pathlib import Path
import subprocess
import sys

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]


MIGRATED_ANALYSIS = (
    (
        "scripts.analyze.analyze_original_merc_results",
        "scripts.analysis.paper_aligned.analyze_original_merc_results",
        "scripts/analyze/analyze_original_merc_results.py",
        "scripts/analysis/paper_aligned/analyze_original_merc_results.py",
        ("--runs-root", "--results-dir", "--experiment-date", "--paper-targets"),
    ),
    (
        "scripts.analyze.audit_model_causality",
        "scripts.analysis.causal.audit_model_causality",
        "scripts/analyze/audit_model_causality.py",
        "scripts/analysis/causal/audit_model_causality.py",
        ("--config", "--output-dir", "--mode", "--target-policy"),
    ),
    (
        "scripts.analyze.build_analysis_tables",
        "scripts.analysis.common.build_analysis_tables",
        "scripts/analyze/build_analysis_tables.py",
        "scripts/analysis/common/build_analysis_tables.py",
        ("--runs-dir", "--output-dir", "--experiment-date"),
    ),
    (
        "scripts.analyze.export_paper_artifacts",
        "scripts.analysis.common.export_paper_artifacts",
        "scripts/analyze/export_paper_artifacts.py",
        "scripts/analysis/common/export_paper_artifacts.py",
        ("--run-dir", "--split", "--also-val", "--output-subdir"),
    ),
    (
        "scripts.analyze.export_paper_multi_run_tables",
        "scripts.analysis.common.export_paper_multi_run_tables",
        "scripts/analyze/export_paper_multi_run_tables.py",
        "scripts/analysis/common/export_paper_multi_run_tables.py",
        ("--run-ids", "--config", "--runs-dir", "--output-dir"),
    ),
    (
        "scripts.analyze.plot_missing_modality_summary",
        "scripts.analysis.common.plot_missing_modality_summary",
        "scripts/analyze/plot_missing_modality_summary.py",
        "scripts/analysis/common/plot_missing_modality_summary.py",
        ("--summary", "--output-dir"),
    ),
    (
        "scripts.analyze.plot_multi_run_final_analysis",
        "scripts.analysis.common.plot_multi_run_final_analysis",
        "scripts/analyze/plot_multi_run_final_analysis.py",
        "scripts/analysis/common/plot_multi_run_final_analysis.py",
        ("--config", "--output-dir", "--evaluation-master"),
    ),
    (
        "scripts.analyze.plot_multi_run_training_curves",
        "scripts.analysis.common.plot_multi_run_training_curves",
        "scripts/analyze/plot_multi_run_training_curves.py",
        "scripts/analysis/common/plot_multi_run_training_curves.py",
        ("--config", "--output-dir", "--epoch-master"),
    ),
    (
        "scripts.analyze.plot_multidag_cl_stabilization_compare",
        "scripts.analysis.models.multidag_cl.plot_stabilization_compare",
        "scripts/analyze/plot_multidag_cl_stabilization_compare.py",
        "scripts/analysis/models/multidag_cl/plot_stabilization_compare.py",
        ("--config", "--output-dir", "--experiment-date"),
    ),
    (
        "scripts.analyze.plot_single_run_final_analysis",
        "scripts.analysis.common.plot_single_run_final_analysis",
        "scripts/analyze/plot_single_run_final_analysis.py",
        "scripts/analysis/common/plot_single_run_final_analysis.py",
        ("--run-id", "--eval-name"),
    ),
    (
        "scripts.analyze.plot_single_run_training_curves",
        "scripts.analysis.common.plot_single_run_training_curves",
        "scripts/analyze/plot_single_run_training_curves.py",
        "scripts/analysis/common/plot_single_run_training_curves.py",
        ("--run-id",),
    ),
    (
        "scripts.analyze.run_four_model_causal_audit",
        "scripts.analysis.causal.run_four_model_causal_audit",
        "scripts/analyze/run_four_model_causal_audit.py",
        "scripts/analysis/causal/run_four_model_causal_audit.py",
        ("--config", "--output-dir", "--experiment-date", "--strict"),
    ),
    (
        "scripts.analyze.summarize_causal_benchmark_runs",
        "scripts.analysis.causal.summarize_causal_benchmark_runs",
        "scripts/analyze/summarize_causal_benchmark_runs.py",
        "scripts/analysis/causal/summarize_causal_benchmark_runs.py",
        ("--config", "--runs-root", "--output-dir", "--report-path", "--strict"),
    ),
)


ADDITIONAL_COMPATIBILITY_ENTRIES = (
    (
        "scripts.analysis.analyze_original_merc_results",
        "scripts.analysis.paper_aligned.analyze_original_merc_results",
        "scripts/analysis/analyze_original_merc_results.py",
    ),
    (
        "scripts.analyze.export_original_merc_reproduction_report",
        "scripts.analysis.paper_aligned.analyze_original_merc_results",
        "scripts/analyze/export_original_merc_reproduction_report.py",
    ),
)


def _help(script: str, work_dir: Path) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["MPLCONFIGDIR"] = str(work_dir / "mplconfig")
    return subprocess.run(
        [sys.executable, str(PROJECT_ROOT / script), "--help"],
        cwd=work_dir,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )


@pytest.mark.parametrize(
    ("legacy_module", "canonical_module", "legacy_path", "canonical_path", "_flags"),
    MIGRATED_ANALYSIS,
)
def test_migrated_analysis_imports_alias_the_canonical_module(
    legacy_module: str,
    canonical_module: str,
    legacy_path: str,
    canonical_path: str,
    _flags: tuple[str, ...],
) -> None:
    canonical = importlib.import_module(canonical_module)
    legacy = importlib.import_module(legacy_module)

    assert legacy is canonical
    assert legacy.main is canonical.main
    assert (PROJECT_ROOT / legacy_path).is_file()
    assert (PROJECT_ROOT / canonical_path).is_file()


@pytest.mark.parametrize(
    ("legacy_module", "canonical_module", "legacy_path"),
    ADDITIONAL_COMPATIBILITY_ENTRIES,
)
def test_existing_analysis_aliases_delegate_directly_to_canonical(
    legacy_module: str,
    canonical_module: str,
    legacy_path: str,
) -> None:
    canonical = importlib.import_module(canonical_module)
    legacy = importlib.import_module(legacy_module)

    assert legacy is canonical
    assert legacy.main is canonical.main
    wrapper_source = (PROJECT_ROOT / legacy_path).read_text(encoding="utf-8")
    assert canonical_module in wrapper_source
    assert "scripts.analyze.analyze_original_merc_results" not in wrapper_source


@pytest.mark.parametrize(
    ("_legacy_module", "_canonical_module", "legacy_path", "canonical_path", "flags"),
    MIGRATED_ANALYSIS,
)
def test_legacy_and_canonical_cli_help_keep_key_options(
    _legacy_module: str,
    _canonical_module: str,
    legacy_path: str,
    canonical_path: str,
    flags: tuple[str, ...],
    tmp_path: Path,
) -> None:
    legacy = _help(legacy_path, tmp_path)
    canonical = _help(canonical_path, tmp_path)

    assert legacy.returncode == canonical.returncode == 0
    legacy_help = legacy.stdout + legacy.stderr
    canonical_help = canonical.stdout + canonical.stderr
    for flag in flags:
        assert flag in legacy_help
        assert flag in canonical_help


def test_canonical_analysis_has_no_reverse_legacy_imports() -> None:
    forbidden_prefixes = (
        "models.baselines",
        "scripts.analyze",
        "scripts.baselines",
        "scripts.train_mmgcn",
    )
    canonical_paths = [PROJECT_ROOT / item[3] for item in MIGRATED_ANALYSIS]

    for path in canonical_paths:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        imported_modules: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_modules.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported_modules.append(node.module)
        assert not any(
            module == prefix or module.startswith(prefix + ".")
            for module in imported_modules
            for prefix in forbidden_prefixes
        ), path


def test_canonical_analysis_repo_roots_resolve_from_new_layout() -> None:
    modules_with_project_root = [
        item[1]
        for item in MIGRATED_ANALYSIS
        if item[1]
        not in {
            "scripts.analysis.common.plot_missing_modality_summary",
        }
    ]
    for module_name in modules_with_project_root:
        module = importlib.import_module(module_name)
        root = getattr(module, "PROJECT_ROOT", getattr(module, "ROOT", None))
        assert root is not None, module_name
        assert Path(root).resolve() == PROJECT_ROOT.resolve()


def _run_missing_modality_plot(
    script: str,
    summary_path: Path,
    output_dir: Path,
    tmp_path: Path,
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["MPLCONFIGDIR"] = str(tmp_path / "mplconfig")
    return subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / script),
            "--summary",
            str(summary_path),
            "--output-dir",
            str(output_dir),
        ],
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )


def test_missing_modality_synthetic_outputs_match_across_entrypoints(
    tmp_path: Path,
) -> None:
    summary_path = tmp_path / "summary.csv"
    columns = ["setting", "loss", "acc", "uar", "macro_f1", "weighted_f1"]
    rows = [
        ["T", 1.2, 0.55, 0.50, 0.48, 0.54],
        ["TAV", 0.8, 0.70, 0.66, 0.64, 0.69],
    ]
    with summary_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(columns)
        writer.writerows(rows)

    legacy_dir = tmp_path / "legacy"
    canonical_dir = tmp_path / "canonical"
    legacy = _run_missing_modality_plot(
        "scripts/analyze/plot_missing_modality_summary.py",
        summary_path,
        legacy_dir,
        tmp_path,
    )
    canonical = _run_missing_modality_plot(
        "scripts/analysis/common/plot_missing_modality_summary.py",
        summary_path,
        canonical_dir,
        tmp_path,
    )

    assert legacy.returncode == canonical.returncode == 0, (
        legacy.stdout,
        legacy.stderr,
        canonical.stdout,
        canonical.stderr,
    )
    legacy_files = sorted(path.name for path in legacy_dir.iterdir())
    canonical_files = sorted(path.name for path in canonical_dir.iterdir())
    assert legacy_files == canonical_files
    assert (legacy_dir / "summary_sorted.csv").read_text(
        encoding="utf-8-sig"
    ) == (canonical_dir / "summary_sorted.csv").read_text(encoding="utf-8-sig")

    with (legacy_dir / "summary_sorted.csv").open(
        encoding="utf-8-sig", newline=""
    ) as file:
        reader = csv.DictReader(file)
        assert reader.fieldnames == columns
        legacy_rows = list(reader)
    with (canonical_dir / "summary_sorted.csv").open(
        encoding="utf-8-sig", newline=""
    ) as file:
        canonical_rows = list(csv.DictReader(file))
    assert legacy_rows == canonical_rows

    for filename in legacy_files:
        legacy_path = legacy_dir / filename
        canonical_path = canonical_dir / filename
        assert legacy_path.stat().st_size > 0
        assert canonical_path.stat().st_size > 0

    image_module = pytest.importorskip("PIL.Image")
    for filename in (name for name in legacy_files if name.endswith(".png")):
        with image_module.open(legacy_dir / filename) as legacy_image:
            with image_module.open(canonical_dir / filename) as canonical_image:
                assert legacy_image.size == canonical_image.size
                assert legacy_image.info.get("dpi") == canonical_image.info.get("dpi")

"""Plan or execute the versioned original-MERC experiment manifest."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = next(
    candidate
    for candidate in Path(__file__).resolve().parents
    if (candidate / "AGENTS.md").is_file() and (candidate / "scripts").is_dir()
)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.runtime.paper_aligned import (  # noqa: E402
    NUMERIC_STATUS_FINITE,
    NumericValidationError,
    load_yaml_config,
    resolve_path,
)
from scripts.workflows.paper_aligned.train import run_training  # noqa: E402
from utils.output_paths import (  # noqa: E402
    configured_output_root,
    create_unique_category_dir,
    discover_analysis_directories,
    resolve_experiment_date,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        default="configs/benchmarks/original_merc/pipeline_manifest.yaml",
    )
    parser.add_argument(
        "--stage",
        choices=(
            "smoke",
            "legacy_paper_adjacent_screening",
            "clean_screening",
            "legacy_folds",
            "clean_folds",
            "top2",
        ),
        default="smoke",
    )
    parser.add_argument("--execute", action="store_true", help="Actually run jobs; default is plan only")
    parser.add_argument("--device", default=None)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--experiment-date", default=None)
    return parser.parse_args()


def load_manifest(path: Path) -> dict[str, Any]:
    with resolve_path(str(path)).open("r", encoding="utf-8") as file:
        manifest = yaml.safe_load(file)
    if not isinstance(manifest, dict) or "stages" not in manifest:
        raise TypeError("pipeline manifest must contain a stages mapping")
    manifest["_manifest_path"] = str(resolve_path(str(path)))
    return manifest


def expand_jobs(manifest: dict[str, Any], stage: str) -> list[dict[str, Any]]:
    jobs = manifest["stages"].get(stage)
    if not isinstance(jobs, list):
        raise TypeError(f"manifest stage {stage!r} must be a list")
    if stage == "top2" and not jobs:
        selection_path = manifest.get("top2_selection_file")
        candidates: list[Path] = []
        if selection_path:
            candidates.append(resolve_path(str(selection_path)))
        candidates.extend(
            directory / "top2_selection.yaml"
            for directory in discover_analysis_directories(PROJECT_ROOT / "outputs")
        )
        resolved_selection = next(
            (candidate for candidate in candidates if candidate.is_file()),
            None,
        )
        if resolved_selection is None:
            raise FileNotFoundError(
                "top2 selection is unavailable; run analyze_original_merc_results.py "
                "after clean screening. Searched an explicit manifest path (when "
                "configured), then dated and legacy analysis directories."
            )
        with resolved_selection.open("r", encoding="utf-8") as file:
            selection = yaml.safe_load(file)
        if selection.get("status") != "ready" or len(selection.get("jobs", [])) != 2:
            raise RuntimeError(
                "top2 selection is pending; complete all four clean screening validation runs"
            )
        if selection.get("test_split_used_for_selection") is not False:
            raise ValueError("top2 selection file does not prove validation-only ranking")
        if selection.get("selection_source") != "clean_single_fold_screening_validation_only":
            raise ValueError("top2 selection did not originate from clean screening validation")
        if selection.get("ordered_criteria", [None])[0] != (
            "clean_screening_validation_weighted_f1"
        ):
            raise ValueError("top2 selection criteria do not prioritize clean validation")
        jobs = selection["jobs"]
    expanded: list[dict[str, Any]] = []
    for job in jobs:
        seeds = job.get("seeds", [None])
        folds = job.get("outer_test_sessions", [None])
        for seed in seeds:
            for fold in folds:
                expanded.append({"config": job["config"], "seed": seed, "outer_test_session": fold})
    return expanded


def materialize_overrides(job: dict[str, Any]) -> Path:
    config_path = resolve_path(str(job["config"]))
    if job["seed"] is None and job["outer_test_session"] is None:
        return config_path
    config = copy.deepcopy(load_yaml_config(config_path))
    suffixes = []
    if job["seed"] is not None:
        seed = int(job["seed"])
        config.setdefault("system", {})["seed"] = seed
        config.setdefault("training", {})["seed"] = seed
        suffixes.append(f"seed{seed}")
    if job["outer_test_session"] is not None:
        fold = str(job["outer_test_session"])
        config.setdefault("dataset", {})["outer_test_session"] = fold
        suffixes.append(fold.lower())
    config["run_name"] = f"{config.get('run_name', config['model']['name'])}_{'_'.join(suffixes)}"
    generated = PROJECT_ROOT / "tmp" / "original_merc_pipeline_configs"
    generated.mkdir(parents=True, exist_ok=True)
    destination = generated / f"{config_path.stem}_{'_'.join(suffixes)}.yaml"
    with destination.open("w", encoding="utf-8") as file:
        yaml.safe_dump(config, file, sort_keys=False, allow_unicode=True)
    return destination


def run_pipeline(
    manifest_path: Path,
    stage: str,
    execute: bool,
    device: str | None,
    output_root: Path | None,
    experiment_date: str | None = None,
) -> list[dict[str, Any]]:
    manifest = load_manifest(manifest_path)
    frozen_date = resolve_experiment_date(
        cli_date=experiment_date,
        config=manifest,
    )
    frozen_output_root = configured_output_root(manifest, override=output_root)
    frozen_output_root = resolve_path(str(frozen_output_root))
    jobs = expand_jobs(manifest, stage)
    plan = {
        "stage": stage,
        "execute": execute,
        "jobs": jobs,
        "experiment_date": frozen_date,
        "day_output_root": str(frozen_output_root / frozen_date),
    }
    print(json.dumps(plan, ensure_ascii=False, indent=2))
    if not execute:
        return jobs
    manifest_dir = create_unique_category_dir(
        "manifests",
        f"original_merc_{stage}",
        frozen_date,
        frozen_output_root,
    )
    (manifest_dir / "pipeline_plan.json").write_text(
        json.dumps(plan, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    results = []
    for job in jobs:
        config_path = materialize_overrides(job)
        try:
            result = run_training(
                config_path,
                output_root_override=frozen_output_root,
                device_override=device,
                experiment_date=frozen_date,
            )
        except FloatingPointError as error:
            failure = (
                error.as_dict()
                if isinstance(error, NumericValidationError)
                else {
                    "numeric_status": "NONFINITE_FORWARD",
                    "first_nonfinite_stage": "unclassified_floating_point_error",
                    "error": str(error),
                }
            )
            (manifest_dir / "run_status.json").write_text(
                json.dumps(
                    {
                        **plan,
                        "manifest_dir": str(manifest_dir),
                        "run_status": "NUMERICALLY_INVALID",
                        "exit_code": 1,
                        **failure,
                        "checkpoint_parameters_finite": False,
                        "final_metrics_finite": False,
                        "prediction_count_correct": False,
                        "completed_runs": results,
                    },
                    ensure_ascii=False,
                    indent=2,
                    default=str,
                )
                + "\n",
                encoding="utf-8",
            )
            raise
        results.append(result)
        passed = (
            result.get("run_status") == "PASS"
            and result.get("numeric_status") == NUMERIC_STATUS_FINITE
            and result.get("checkpoint_parameters_finite") is True
            and result.get("final_metrics_finite") is True
            and result.get("prediction_count_correct") is True
        )
        (manifest_dir / "run_status.json").write_text(
            json.dumps(
                {
                    **plan,
                    "manifest_dir": str(manifest_dir),
                    "run_status": "PASS" if passed else "NUMERICALLY_INVALID",
                    "exit_code": 0,
                    "numeric_status": result.get("numeric_status"),
                    "checkpoint_parameters_finite": result.get(
                        "checkpoint_parameters_finite"
                    ),
                    "final_metrics_finite": result.get("final_metrics_finite"),
                    "prediction_count_correct": result.get("prediction_count_correct"),
                    "completed_runs": results,
                },
                ensure_ascii=False,
                indent=2,
                default=str,
            )
            + "\n",
            encoding="utf-8",
        )
    return results


def main() -> None:
    args = parse_args()
    run_pipeline(
        Path(args.manifest),
        args.stage,
        args.execute,
        args.device,
        None if args.output_root is None else Path(args.output_root),
        args.experiment_date,
    )


if __name__ == "__main__":
    main()

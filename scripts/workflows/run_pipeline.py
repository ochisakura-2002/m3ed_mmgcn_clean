"""
Run a full experiment pipeline from one YAML config.

This script is only a scheduler. It does not implement training, evaluation,
model building, dataset loading, or plotting by itself.

Pipeline stages:
    1. Train model
    2. Evaluate checkpoint
    3. Build analysis master tables
    4. Single-run training curve analysis
    5. Single-run final analysis
    6. Single-run paper artifact export
    7. Multi-run training curve analysis
    8. Multi-run final analysis

Typical usage:
    python scripts/workflows/run_pipeline.py \
      --config configs/benchmarks/ablations/missing_modality/pipelines/mmgcn/m3ed_features/mmgcn_pipeline_m3ed.yaml

Design rule:
    Do not modify train_mmgcn.py just because the pipeline has an import problem.
    This pipeline runs child scripts through a bootstrap process that inserts
    PROJECT_ROOT into sys.path before running the target script.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time

import yaml


PROJECT_ROOT = next(
    candidate
    for candidate in Path(__file__).resolve().parents
    if (candidate / "AGENTS.md").is_file() and (candidate / "scripts").is_dir()
)
RUN_INFO_PREFIX = "CODEX_RUN_INFO_JSON="
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


TRAIN_SCRIPT_REGISTRY = {
    "MMGCN": "scripts/models/mmgcn/unified/train.py",
    "SimpleMLP": "scripts/models/simple_mlp/train.py",
    "GS_MCC": "scripts/train_gsmcc.py",
    "MultiDAGCL": "scripts/models/multidag_cl/unified/train.py",
    "causal_gsmcc_inspired": "scripts/workflows/causal_graph/train.py",
    "causal_dialoguegcn": "scripts/workflows/causal_graph/train.py",
}


EVALUATE_SCRIPT = "scripts/evaluation/unified_checkpoint.py"
EVALUATE_SCRIPT_REGISTRY = {
    "MMGCN": EVALUATE_SCRIPT,
    "SimpleMLP": EVALUATE_SCRIPT,
    "MultiDAGCL": "scripts/models/multidag_cl/unified/evaluate.py",
    "causal_gsmcc_inspired": "scripts/workflows/causal_graph/evaluate.py",
    "causal_dialoguegcn": "scripts/workflows/causal_graph/evaluate.py",
}

BUILD_ANALYSIS_TABLES_SCRIPT = "scripts/analysis/common/build_analysis_tables.py"
PLOT_SINGLE_RUN_TRAINING_CURVES_SCRIPT = (
    "scripts/analysis/common/plot_single_run_training_curves.py"
)
PLOT_SINGLE_RUN_FINAL_ANALYSIS_SCRIPT = (
    "scripts/analysis/common/plot_single_run_final_analysis.py"
)
PLOT_MULTI_RUN_TRAINING_CURVES_SCRIPT = (
    "scripts/analysis/common/plot_multi_run_training_curves.py"
)
PLOT_MULTI_RUN_FINAL_ANALYSIS_SCRIPT = (
    "scripts/analysis/common/plot_multi_run_final_analysis.py"
)
EVALUATE_MISSING_MODALITIES_SCRIPT = "scripts/workflows/ablations/evaluate_missing_modalities.py"
PLOT_MISSING_MODALITY_SUMMARY_SCRIPT = (
    "scripts/analysis/common/plot_missing_modality_summary.py"
)
EXPORT_PAPER_ARTIFACTS_SCRIPT = "scripts/analysis/common/export_paper_artifacts.py"

from utils.output_paths import (  # noqa: E402
    EXPERIMENT_DATE_ENV,
    configured_output_root,
    create_unique_category_dir,
    find_run_directory,
    resolve_experiment_date,
    resolve_output_category,
    sanitize_run_name,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a complete experiment pipeline from YAML."
    )

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to pipeline YAML config.",
    )
    parser.add_argument(
        "--force-unlock",
        action="store_true",
        help="Explicitly remove this pipeline config's existing lock before starting.",
    )
    parser.add_argument(
        "--experiment-date",
        default=None,
        help="Frozen pipeline launch date in YYYYMMDD format.",
    )

    return parser.parse_args()


def pipeline_lock_path(config_path: Path) -> Path:
    resolved = str(Path(config_path).resolve())
    digest = hashlib.sha256(resolved.encode("utf-8")).hexdigest()[:16]
    stem = "".join(
        character if character.isalnum() or character in "-_" else "_"
        for character in Path(config_path).stem
    )
    return PROJECT_ROOT / "tmp" / "pipeline_locks" / f"{stem}-{digest}.lock"


class PipelineLock:
    """Atomic per-pipeline lock with explicit force removal."""

    def __init__(self, path: Path, config_path: Path) -> None:
        self.path = Path(path)
        self.config_path = Path(config_path)
        self.acquired = False

    def acquire(self, *, force_unlock: bool = False) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if force_unlock and self.path.exists():
            self.path.unlink()
        payload = json.dumps(
            {
                "pid": os.getpid(),
                "created_at_unix": time.time(),
                "config_path": str(self.config_path),
            },
            ensure_ascii=False,
        ).encode("utf-8")
        try:
            descriptor = os.open(
                str(self.path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600
            )
        except FileExistsError as error:
            try:
                owner = self.path.read_text(encoding="utf-8").strip()
            except OSError:
                owner = "unreadable"
            raise RuntimeError(
                f"Pipeline is already locked: {self.path}. Owner: {owner}. "
                "Use --force-unlock only after confirming no matching batch is running."
            ) from error
        try:
            os.write(descriptor, payload)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        self.acquired = True

    def release(self) -> None:
        if not self.acquired:
            return
        try:
            self.path.unlink()
        except FileNotFoundError:
            pass
        self.acquired = False


def resolve_path(path_text: str) -> Path:
    path = Path(path_text)

    if path.is_absolute():
        return path

    return PROJECT_ROOT / path


def path_to_command_arg(path: Path) -> str:
    """
    Prefer project-relative paths for readable logs and compatibility with
    existing scripts.
    """
    resolved = path.resolve()

    try:
        return str(resolved.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(resolved)


def load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"YAML config not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if config is None:
        raise RuntimeError(f"Empty YAML config: {path}")

    if not isinstance(config, dict):
        raise TypeError(f"YAML config must be a dict: {path}")

    return config


def safe_get(config: Dict[str, Any], keys: List[str], default: Any = None) -> Any:
    current: Any = config

    for key in keys:
        if not isinstance(current, dict):
            return default

        if key not in current:
            return default

        current = current[key]

    if current is None:
        return default

    return current


def get_bool(config: Dict[str, Any], keys: List[str], default: bool = False) -> bool:
    value = safe_get(config, keys, default)
    return bool(value)


def ensure_project_script_exists(script_relative_path: str) -> None:
    script_path = PROJECT_ROOT / script_relative_path

    if not script_path.exists():
        raise FileNotFoundError(
            f"Project script not found: {script_path}"
        )


def run_project_script(
    script_relative_path: str,
    script_args: List[str],
    dry_run: bool = False,
) -> Dict[str, str]:
    """
    Run a project script without modifying the script itself.

    This function starts a new Python process, inserts PROJECT_ROOT into sys.path,
    changes cwd to PROJECT_ROOT, then runs the target script via runpy.run_path.

    This avoids editing train_mmgcn.py / evaluate_checkpoint.py just to fix imports.
    """
    ensure_project_script_exists(script_relative_path)

    display_command = [
        sys.executable,
        script_relative_path,
    ] + script_args

    print("\n" + "-" * 100)
    print("[Run]", " ".join(display_command))
    print("[Mode] bootstrap with PROJECT_ROOT in sys.path")
    print("-" * 100)

    if dry_run:
        print("[Dry run] Command not executed.")
        return {}

    project_root_text = str(PROJECT_ROOT.resolve())

    bootstrap_code = (
        "import os, sys, runpy; "
        f"project_root = {project_root_text!r}; "
        "os.chdir(project_root); "
        "sys.path.insert(0, project_root); "
        f"sys.argv = {[script_relative_path] + script_args!r}; "
        "runpy.run_path(sys.argv[0], run_name='__main__')"
    )

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUNBUFFERED"] = "1"
    old_pythonpath = env.get("PYTHONPATH", "")

    if old_pythonpath:
        env["PYTHONPATH"] = project_root_text + os.pathsep + old_pythonpath
    else:
        env["PYTHONPATH"] = project_root_text

    command = [sys.executable, "-c", bootstrap_code]
    process = subprocess.Popen(
        command,
        cwd=PROJECT_ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    run_info: Dict[str, str] = {}
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="", flush=True)
        stripped = line.strip()
        if stripped.startswith(RUN_INFO_PREFIX):
            payload = json.loads(stripped[len(RUN_INFO_PREFIX) :])
            if not isinstance(payload, dict):
                raise TypeError("Training run-info marker must contain a JSON object.")
            run_info = {str(key): str(value) for key, value in payload.items()}
    return_code = process.wait()
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, command)
    return run_info


def choose_train_script(model_name: str) -> str:
    if model_name not in TRAIN_SCRIPT_REGISTRY:
        supported = ", ".join(sorted(TRAIN_SCRIPT_REGISTRY.keys()))
        raise ValueError(
            f"Unsupported model name: {model_name}. "
            f"Supported models: {supported}"
        )

    script_relative_path = TRAIN_SCRIPT_REGISTRY[model_name]
    ensure_project_script_exists(script_relative_path)

    return script_relative_path


def choose_evaluate_script(model_name: str) -> str:
    if model_name not in EVALUATE_SCRIPT_REGISTRY:
        supported = ", ".join(sorted(EVALUATE_SCRIPT_REGISTRY.keys()))
        raise ValueError(
            f"Unsupported model name for evaluation: {model_name}. "
            f"Supported models: {supported}"
        )

    script_relative_path = EVALUATE_SCRIPT_REGISTRY[model_name]
    ensure_project_script_exists(script_relative_path)

    return script_relative_path


def validate_train_config(
    pipeline_config: Dict[str, Any],
    train_config_path: Path,
) -> None:
    train_config = load_yaml(train_config_path)

    pipeline_dataset = safe_get(pipeline_config, ["dataset", "name"], "")
    pipeline_model = safe_get(pipeline_config, ["model", "name"], "")

    train_dataset = safe_get(train_config, ["dataset", "name"], "")
    train_model = safe_get(train_config, ["model", "name"], "")

    if train_dataset and pipeline_dataset and train_dataset != pipeline_dataset:
        raise ValueError(
            "Dataset mismatch between pipeline YAML and train YAML:\n"
            f"  pipeline dataset: {pipeline_dataset}\n"
            f"  train config dataset: {train_dataset}"
        )

    if train_model and pipeline_model and train_model != pipeline_model:
        raise ValueError(
            "Model mismatch between pipeline YAML and train YAML:\n"
            f"  pipeline model: {pipeline_model}\n"
            f"  train config model: {train_model}"
        )


def get_existing_run_info(
    run_id: str,
    output_root: Path = PROJECT_ROOT / "outputs",
) -> Dict[str, str]:
    run_dir = find_run_directory(run_id, output_root)

    return {
        "run_id": run_id,
        "run_dir": str(run_dir.resolve()),
    }


def pipeline_needs_current_run(config: Dict[str, Any]) -> bool:
    evaluation_enabled = get_bool(config, ["evaluation", "enabled"], False)
    single_enabled = get_bool(config, ["single_run_analysis", "enabled"], False)
    paper_enabled = get_bool(config, ["paper_artifacts", "enabled"], False)
    missing_enabled = get_bool(config, ["missing_modalities", "enabled"], False)

    return bool(evaluation_enabled or single_enabled or paper_enabled or missing_enabled)


def resolve_current_run_info(
    config: Dict[str, Any],
    train_was_executed: bool,
    dry_run: bool,
    training_run_info: Optional[Dict[str, str]] = None,
) -> Optional[Dict[str, str]]:
    """
    Decide which run_id is the target of evaluation and single-run analysis.

    Cases:
    1. train.enabled=true:
       use the machine-readable run info emitted by this exact subprocess.

    2. train.enabled=false and run_control.skip_train_use_run_id is set:
       use that existing run.

    3. train.enabled=false and no target run is needed:
       return None.

    4. train.enabled=false but evaluation/single-run analysis enabled:
       error, because there is no target run.
    """
    if dry_run and train_was_executed:
        print(
            "\n[Dry run] Training is enabled, so the new run_id is unknown. "
            "Stop before evaluation and single-run analysis."
        )
        return None

    if train_was_executed:
        if training_run_info is None:
            raise RuntimeError(
                "Training completed without a machine-readable run-info marker; "
                "refusing to consult shared latest_run.txt."
            )
        run_id = str(training_run_info.get("run_id", "")).strip()
        run_dir_text = str(training_run_info.get("run_dir", "")).strip()
        if not run_id or not run_dir_text:
            raise ValueError("Training run-info marker must include run_id and run_dir.")
        run_dir = Path(run_dir_text)
        if not run_dir.is_absolute():
            run_dir = PROJECT_ROOT / run_dir
        if not run_dir.is_dir():
            raise FileNotFoundError(f"Training-reported run_dir does not exist: {run_dir}")
        return {"run_id": run_id, "run_dir": str(run_dir.resolve())}

    specified_run_id = str(
        safe_get(config, ["run_control", "skip_train_use_run_id"], "") or ""
    ).strip()

    if specified_run_id:
        output_root = configured_output_root(config)
        if not output_root.is_absolute():
            output_root = PROJECT_ROOT / output_root
        return get_existing_run_info(specified_run_id, output_root)

    if pipeline_needs_current_run(config):
        raise ValueError(
            "This pipeline needs a target run for evaluation or single-run analysis, "
            "but train.enabled=false and run_control.skip_train_use_run_id is empty."
        )

    return None


def get_checkpoint_path(
    run_info: Dict[str, str],
    checkpoint_name: str,
) -> Path:
    run_dir = Path(run_info["run_dir"])
    checkpoint_path = run_dir / "checkpoints" / checkpoint_name

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    return checkpoint_path


def print_pipeline_header(
    config_path: Path,
    config: Dict[str, Any],
) -> None:
    print("=" * 100)
    print("Run experiment pipeline")
    print("=" * 100)
    print("Project root:", PROJECT_ROOT)
    print("Pipeline config:", config_path)
    print("Pipeline name:", safe_get(config, ["project", "pipeline_name"], ""))
    print("Dataset:", safe_get(config, ["dataset", "name"], ""))
    print("Model:", safe_get(config, ["model", "name"], ""))
    print("Train enabled:", get_bool(config, ["train", "enabled"], False))
    print("Evaluation enabled:", get_bool(config, ["evaluation", "enabled"], False))
    print("Analysis tables enabled:", get_bool(config, ["analysis_tables", "enabled"], True))
    print("Single-run analysis enabled:", get_bool(config, ["single_run_analysis", "enabled"], False))
    print("Paper artifacts enabled:", get_bool(config, ["paper_artifacts", "enabled"], False))
    print("Missing modalities enabled:", get_bool(config, ["missing_modalities", "enabled"], False))
    print("Multi-run training curves enabled:", get_bool(config, ["multi_run_training_curves", "enabled"], False))
    print("Multi-run final analysis enabled:", get_bool(config, ["multi_run_final_analysis", "enabled"], False))
    print("Dry run:", get_bool(config, ["execution", "dry_run"], False))

    train_config_path = safe_get(config, ["train", "train_config_path"], "")
    if train_config_path:
        print("Train config path:", train_config_path)

    skip_run_id = safe_get(config, ["run_control", "skip_train_use_run_id"], "")
    if skip_run_id:
        print("Skip-train target run_id:", skip_run_id)

    print("=" * 100)


def print_pipeline_summary(
    run_info: Optional[Dict[str, str]],
    config: Dict[str, Any],
) -> None:
    print("\n" + "=" * 100)
    print("Pipeline finished")
    print("=" * 100)

    if run_info is not None:
        run_id = run_info["run_id"]
        run_dir = Path(run_info["run_dir"])

        print("Target run_id:", run_id)
        print("Target run_dir:", run_dir)

        print("\nSingle-run outputs:")
        print("  Training curves:", run_dir / "figures" / "training_curves")
        print("  Final analysis:", run_dir / "figures" / "final_analysis")

        if get_bool(config, ["paper_artifacts", "enabled"], False):
            output_subdir = str(
                safe_get(config, ["paper_artifacts", "output_subdir"], "paper_artifacts")
            )
            print("  Paper artifacts:", run_dir / output_subdir)

        if get_bool(config, ["missing_modalities", "enabled"], False):
            output_subdir = str(
                safe_get(config, ["missing_modalities", "output_subdir"], "missing_modalities")
            )
            print("  Missing modalities:", run_dir / "logs" / output_subdir)
            print("  Missing modality figures:", run_dir / "figures" / "missing_modalities")
    else:
        print("Target run_id: NONE")

    print("\nMaster tables:")
    analysis_tables_dir = Path(
        str(safe_get(config, ["output", "analysis_dir"], "outputs"))
    ) / "analysis_tables"
    for filename in (
        "run_file_status.csv",
        "run_summary_master.csv",
        "epoch_metrics_master.csv",
        "evaluation_master.csv",
        "per_class_master.csv",
    ):
        print(" ", analysis_tables_dir / filename)

    multi_train_enabled = get_bool(config, ["multi_run_training_curves", "enabled"], False)
    multi_final_enabled = get_bool(config, ["multi_run_final_analysis", "enabled"], False)

    if multi_train_enabled:
        print("\nMulti-run training analysis config:")
        print(" ", safe_get(config, ["multi_run_training_curves", "config_path"], ""))

    if multi_final_enabled:
        print("\nMulti-run final analysis config:")
        print(" ", safe_get(config, ["multi_run_final_analysis", "config_path"], ""))

    print("=" * 100)


def run_training_stage(
    config: Dict[str, Any],
    dry_run: bool,
    experiment_date: str,
) -> Tuple[bool, Optional[Dict[str, str]]]:
    train_enabled = get_bool(config, ["train", "enabled"], False)
    if not train_enabled:
        return False, None

    model_name = str(safe_get(config, ["model", "name"], ""))
    train_config_text = safe_get(config, ["train", "train_config_path"], "")
    if not train_config_text:
        raise ValueError("train.enabled=true but train.train_config_path is empty.")
    train_config_path = resolve_path(str(train_config_text))
    validate_train_config(
        pipeline_config=config,
        train_config_path=train_config_path,
    )
    train_script = choose_train_script(model_name)
    run_info = run_project_script(
        script_relative_path=train_script,
        script_args=[
            "--config",
            path_to_command_arg(train_config_path),
            "--experiment-date",
            experiment_date,
        ],
        dry_run=dry_run,
    )
    if dry_run:
        return True, None
    if "run_id" not in run_info or "run_dir" not in run_info:
        raise RuntimeError(
            f"Training script {train_script} did not emit {RUN_INFO_PREFIX}<json>."
        )
    return True, run_info


def run_evaluation_stage(
    config: Dict[str, Any],
    run_info: Optional[Dict[str, str]],
    dry_run: bool,
) -> None:
    evaluation_enabled = get_bool(config, ["evaluation", "enabled"], False)

    if not evaluation_enabled:
        return

    if run_info is None:
        raise RuntimeError("Evaluation enabled but no target run is available.")

    checkpoint_name = str(
        safe_get(config, ["evaluation", "checkpoint_name"], "best_model.pt")
    )

    splits = safe_get(config, ["evaluation", "splits"], ["val", "test"])

    if not isinstance(splits, list) or len(splits) == 0:
        raise ValueError("evaluation.splits must be a non-empty list.")

    checkpoint_path = get_checkpoint_path(
        run_info=run_info,
        checkpoint_name=checkpoint_name,
    )

    evaluate_script = choose_evaluate_script(
        str(safe_get(config, ["model", "name"], ""))
    )
    max_batches = safe_get(config, ["evaluation", "max_batches"], None)

    for split in splits:
        script_args = [
            "--checkpoint",
            path_to_command_arg(checkpoint_path),
            "--split",
            str(split),
        ]

        if max_batches is not None:
            if evaluate_script != EVALUATE_SCRIPT_REGISTRY["MultiDAGCL"]:
                raise ValueError(
                    "evaluation.max_batches is currently supported only for "
                    "MultiDAGCL evaluation."
                )
            script_args.extend(["--max-batches", str(max_batches)])

        run_project_script(
            script_relative_path=evaluate_script,
            script_args=script_args,
            dry_run=dry_run,
        )


def run_analysis_tables_stage(
    config: Dict[str, Any],
    dry_run: bool,
) -> None:
    analysis_tables_enabled = get_bool(config, ["analysis_tables", "enabled"], True)

    if not analysis_tables_enabled:
        return

    run_project_script(
        script_relative_path=BUILD_ANALYSIS_TABLES_SCRIPT,
        script_args=[],
        dry_run=dry_run,
    )


def run_single_run_analysis_stage(
    config: Dict[str, Any],
    run_info: Optional[Dict[str, str]],
    dry_run: bool,
) -> None:
    single_enabled = get_bool(config, ["single_run_analysis", "enabled"], False)

    if not single_enabled:
        return

    if run_info is None:
        raise RuntimeError("Single-run analysis enabled but no target run is available.")

    run_id = run_info["run_id"]

    training_curves_enabled = get_bool(
        config,
        ["single_run_analysis", "training_curves"],
        True,
    )

    final_analysis_enabled = get_bool(
        config,
        ["single_run_analysis", "final_analysis"],
        True,
    )

    if training_curves_enabled:
        run_project_script(
            script_relative_path=PLOT_SINGLE_RUN_TRAINING_CURVES_SCRIPT,
            script_args=[
                "--run-id",
                run_id,
            ],
            dry_run=dry_run,
        )

    if final_analysis_enabled:
        run_project_script(
            script_relative_path=PLOT_SINGLE_RUN_FINAL_ANALYSIS_SCRIPT,
            script_args=[
                "--run-id",
                run_id,
            ],
            dry_run=dry_run,
        )


def run_paper_artifacts_stage(
    config: Dict[str, Any],
    run_info: Optional[Dict[str, str]],
    dry_run: bool,
) -> None:
    paper_enabled = get_bool(config, ["paper_artifacts", "enabled"], False)

    if not paper_enabled:
        return

    if run_info is None:
        raise RuntimeError("paper_artifacts.enabled=true but no target run is available.")

    run_dir = Path(run_info["run_dir"])
    split = str(safe_get(config, ["paper_artifacts", "split"], "test"))
    output_subdir = str(
        safe_get(config, ["paper_artifacts", "output_subdir"], "paper_artifacts")
    )
    also_val = get_bool(config, ["paper_artifacts", "also_val"], False)

    script_args = [
        "--run-dir",
        path_to_command_arg(run_dir),
        "--split",
        split,
        "--output-subdir",
        output_subdir,
    ]

    if also_val:
        script_args.append("--also-val")

    run_project_script(
        script_relative_path=EXPORT_PAPER_ARTIFACTS_SCRIPT,
        script_args=script_args,
        dry_run=dry_run,
    )



def normalize_checkpoint_name(checkpoint_name: str) -> str:
    checkpoint_name = str(checkpoint_name).strip()

    if not checkpoint_name:
        checkpoint_name = "best_model.pt"

    if not checkpoint_name.endswith(".pt"):
        checkpoint_name = checkpoint_name + ".pt"

    return checkpoint_name


def run_missing_modalities_stage(
    config: Dict[str, Any],
    run_info: Optional[Dict[str, str]],
    dry_run: bool,
) -> None:
    missing_enabled = get_bool(config, ["missing_modalities", "enabled"], False)

    if not missing_enabled:
        return

    if run_info is None:
        raise RuntimeError("missing_modalities.enabled=true but no target run is available.")

    checkpoint_name_text = safe_get(
        config,
        ["missing_modalities", "checkpoint_name"],
        None,
    )

    if checkpoint_name_text is None:
        checkpoint_name_text = safe_get(
            config,
            ["missing_modalities", "checkpoint"],
            safe_get(config, ["evaluation", "checkpoint_name"], "best_model.pt"),
        )

    checkpoint_name = normalize_checkpoint_name(str(checkpoint_name_text))

    split = str(
        safe_get(config, ["missing_modalities", "split"], "test")
    )

    settings = safe_get(
        config,
        ["missing_modalities", "settings"],
        ["TAV", "TA", "TV", "AV", "T", "A", "V"],
    )

    if not isinstance(settings, list) or len(settings) == 0:
        raise ValueError("missing_modalities.settings must be a non-empty list.")

    output_subdir = str(
        safe_get(config, ["missing_modalities", "output_subdir"], "missing_modalities")
    )

    make_figures = get_bool(
        config,
        ["missing_modalities", "make_figures"],
        True,
    )

    skip_if_not_full = get_bool(
        config,
        ["missing_modalities", "skip_if_not_full_train_modalities"],
        True,
    )

    checkpoint_path = get_checkpoint_path(
        run_info=run_info,
        checkpoint_name=checkpoint_name,
    )

    script_args = [
        "--checkpoint",
        path_to_command_arg(checkpoint_path),
        "--split",
        split,
        "--output-subdir",
        output_subdir,
        "--settings",
        *[str(setting) for setting in settings],
    ]

    if skip_if_not_full:
        script_args.append("--skip-if-not-full-train-modalities")

    run_project_script(
        script_relative_path=EVALUATE_MISSING_MODALITIES_SCRIPT,
        script_args=script_args,
        dry_run=dry_run,
    )

    if not make_figures:
        return

    run_dir = Path(run_info["run_dir"])
    stage_name = f"{split}_{Path(checkpoint_name).stem}"
    summary_path = run_dir / "logs" / output_subdir / stage_name / "summary.csv"

    if not dry_run and not summary_path.exists():
        print(
            "[Skip] Missing-modality summary not found. "
            f"Maybe evaluation was skipped: {summary_path}"
        )
        return

    run_project_script(
        script_relative_path=PLOT_MISSING_MODALITY_SUMMARY_SCRIPT,
        script_args=[
            "--summary",
            path_to_command_arg(summary_path),
        ],
        dry_run=dry_run,
    )


def run_multi_run_training_curves_stage(
    config: Dict[str, Any],
    dry_run: bool,
) -> None:
    multi_training_enabled = get_bool(
        config,
        ["multi_run_training_curves", "enabled"],
        False,
    )

    if not multi_training_enabled:
        return

    multi_training_config_text = safe_get(
        config,
        ["multi_run_training_curves", "config_path"],
        "",
    )

    if not multi_training_config_text:
        raise ValueError(
            "multi_run_training_curves.enabled=true but config_path is empty."
        )

    multi_training_config_path = resolve_path(str(multi_training_config_text))

    run_project_script(
        script_relative_path=PLOT_MULTI_RUN_TRAINING_CURVES_SCRIPT,
        script_args=[
            "--config",
            path_to_command_arg(multi_training_config_path),
        ],
        dry_run=dry_run,
    )


def run_multi_run_final_analysis_stage(
    config: Dict[str, Any],
    dry_run: bool,
) -> None:
    multi_final_enabled = get_bool(
        config,
        ["multi_run_final_analysis", "enabled"],
        False,
    )

    if not multi_final_enabled:
        return

    multi_final_config_text = safe_get(
        config,
        ["multi_run_final_analysis", "config_path"],
        "",
    )

    if not multi_final_config_text:
        raise ValueError(
            "multi_run_final_analysis.enabled=true but config_path is empty."
        )

    multi_final_config_path = resolve_path(str(multi_final_config_text))

    run_project_script(
        script_relative_path=PLOT_MULTI_RUN_FINAL_ANALYSIS_SCRIPT,
        script_args=[
            "--config",
            path_to_command_arg(multi_final_config_path),
        ],
        dry_run=dry_run,
    )


def main() -> Optional[str]:
    args = parse_args()

    config_path = resolve_path(args.config)
    config = load_yaml(config_path)
    frozen_date = resolve_experiment_date(
        cli_date=args.experiment_date,
        config=config,
    )
    output_root = configured_output_root(config)
    if not output_root.is_absolute():
        output_root = PROJECT_ROOT / output_root
    pipeline_name = sanitize_run_name(
        str(safe_get(config, ["project", "experiment_name"], config_path.stem))
    )
    config.setdefault("output", {})
    config["output"].update(
        {
            "root": str(output_root),
            "experiment_date": frozen_date,
            "output_root": str(output_root),
            "day_output_root": str(output_root / frozen_date),
            "log_dir": str(
                resolve_output_category("logs", frozen_date, output_root)
                / pipeline_name
            ),
            "analysis_dir": str(
                resolve_output_category("analysis", frozen_date, output_root)
            ),
        }
    )
    lock = PipelineLock(pipeline_lock_path(config_path), config_path)
    lock.acquire(force_unlock=bool(args.force_unlock))
    old_experiment_date = os.environ.get(EXPERIMENT_DATE_ENV)
    os.environ[EXPERIMENT_DATE_ENV] = frozen_date
    try:
        print_pipeline_header(config_path, config)
        dry_run = get_bool(config, ["execution", "dry_run"], False)
        manifest_dir = None
        if not dry_run:
            manifest_dir = create_unique_category_dir(
                "manifests",
                pipeline_name,
                frozen_date,
                output_root,
            )
            config["output"]["manifest_dir"] = str(manifest_dir)
            with (manifest_dir / "pipeline_config_resolved.yaml").open(
                "w", encoding="utf-8"
            ) as file:
                yaml.safe_dump(config, file, sort_keys=False, allow_unicode=True)
        train_was_executed, training_run_info = run_training_stage(
            config=config,
            dry_run=dry_run,
            experiment_date=frozen_date,
        )
        run_info = resolve_current_run_info(
            config=config,
            train_was_executed=train_was_executed,
            dry_run=dry_run,
            training_run_info=training_run_info,
        )
        if dry_run and train_was_executed:
            print_pipeline_summary(run_info=None, config=config)
            return None
        run_evaluation_stage(config=config, run_info=run_info, dry_run=dry_run)
        run_analysis_tables_stage(config=config, dry_run=dry_run)
        run_single_run_analysis_stage(
            config=config, run_info=run_info, dry_run=dry_run
        )
        run_paper_artifacts_stage(config=config, run_info=run_info, dry_run=dry_run)
        run_missing_modalities_stage(config=config, run_info=run_info, dry_run=dry_run)
        run_multi_run_training_curves_stage(config=config, dry_run=dry_run)
        run_multi_run_final_analysis_stage(config=config, dry_run=dry_run)
        if manifest_dir is not None:
            (manifest_dir / "run_status.json").write_text(
                json.dumps(
                    {
                        "status": "completed",
                        "experiment_date": frozen_date,
                        "day_output_root": str(output_root / frozen_date),
                        "run_info": run_info,
                    },
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
        print_pipeline_summary(run_info=run_info, config=config)
        return None if run_info is None else str(run_info["run_id"])
    finally:
        if old_experiment_date is None:
            os.environ.pop(EXPERIMENT_DATE_ENV, None)
        else:
            os.environ[EXPERIMENT_DATE_ENV] = old_experiment_date
        lock.release()


if __name__ == "__main__":
    main()

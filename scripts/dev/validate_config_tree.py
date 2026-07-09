"""Validate canonical MultiDAG+CL config wiring without running experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[2]

FORMAL_PIPELINE_DIR = ROOT / "configs" / "pipeline" / "multidag_cl" / "iemocap" / "formal"
STABILIZE_PIPELINE_DIR = ROOT / "configs" / "pipeline" / "multidag_cl" / "iemocap" / "stabilize"
FORMAL_TRAIN_DIR = ROOT / "configs" / "baselines" / "multidag_cl" / "iemocap" / "formal"
STABILIZE_TRAIN_DIR = ROOT / "configs" / "baselines" / "multidag_cl" / "iemocap" / "stabilize"

STABLE_CONTEXTS = {
    "context_w0_tav_stable_candidate.yaml": ("causal", 0),
    "context_w3_tav_stable_candidate.yaml": ("causal", 3),
    "context_w5_tav_stable_candidate.yaml": ("causal", 5),
    "context_past_all_causal_tav_stable_candidate.yaml": ("past_all_causal", None),
}


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)
    if not isinstance(data, dict):
        raise TypeError(f"{path} must contain a YAML mapping")
    return data


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def require(condition: bool, message: str, errors: list[str]) -> None:
    if not condition:
        errors.append(message)


def bool_at(config: dict[str, Any], *keys: str) -> bool:
    current: Any = config
    for key in keys:
        if not isinstance(current, dict):
            return False
        current = current.get(key)
    return bool(current)


def validate_pipeline(path: Path, errors: list[str]) -> None:
    config = load_yaml(path)
    train_path_text = str(config.get("train", {}).get("train_config_path", ""))
    train_path = ROOT / train_path_text

    require(train_path_text != "", f"{rel(path)} missing train.train_config_path", errors)
    require(train_path.exists(), f"{rel(path)} points to missing {train_path_text}", errors)
    require(bool_at(config, "evaluation", "enabled"), f"{rel(path)} must enable evaluation", errors)
    require(bool_at(config, "analysis_tables", "enabled"), f"{rel(path)} must enable analysis_tables", errors)
    require(bool_at(config, "single_run_analysis", "enabled"), f"{rel(path)} must enable single_run_analysis", errors)
    require(
        bool_at(config, "single_run_analysis", "training_curves"),
        f"{rel(path)} must enable training_curves",
        errors,
    )
    require(
        bool_at(config, "single_run_analysis", "final_analysis"),
        f"{rel(path)} must enable final_analysis",
        errors,
    )


def validate_training(path: Path, errors: list[str]) -> None:
    config = load_yaml(path)
    training = config.get("training", {})
    graph = config.get("graph", {})

    require(training.get("max_train_batches") is None, f"{rel(path)} caps max_train_batches", errors)
    require(training.get("max_val_batches") is None, f"{rel(path)} caps max_val_batches", errors)

    if path.parent == FORMAL_TRAIN_DIR:
        require("full" not in path.name.lower(), f"{rel(path)} uses legacy full naming", errors)
        require(graph.get("context_mode") != "full", f"{rel(path)} uses graph.context_mode: full", errors)

    if path.name in STABLE_CONTEXTS:
        expected_mode, expected_window = STABLE_CONTEXTS[path.name]
        model = config.get("model", {})
        require(model.get("modality_encoder_type") == "causal_gru", f"{rel(path)} stable encoder mismatch", errors)
        require(model.get("num_graph_layers") == 1, f"{rel(path)} stable graph layer mismatch", errors)
        require(model.get("dropout") == 0.2, f"{rel(path)} stable dropout mismatch", errors)
        require(training.get("lr") == 0.0005, f"{rel(path)} stable lr mismatch", errors)
        require(training.get("weight_decay") == 0.0001, f"{rel(path)} stable weight_decay mismatch", errors)
        require(training.get("grad_clip") == 1.0, f"{rel(path)} stable grad_clip mismatch", errors)
        require(training.get("epochs") == 30, f"{rel(path)} stable epochs mismatch", errors)
        require(training.get("batch_size") == 8, f"{rel(path)} stable batch_size mismatch", errors)
        require(training.get("select_best_by") == "val_weighted_f1", f"{rel(path)} stable selector mismatch", errors)
        require(graph.get("context_mode") == expected_mode, f"{rel(path)} stable context_mode mismatch", errors)
        require(graph.get("window_past") == expected_window, f"{rel(path)} stable window_past mismatch", errors)


def main() -> None:
    errors: list[str] = []

    for path in sorted(FORMAL_PIPELINE_DIR.glob("*.yaml")):
        validate_pipeline(path, errors)
    for path in sorted(STABILIZE_PIPELINE_DIR.glob("*.yaml")):
        validate_pipeline(path, errors)

    for path in sorted(FORMAL_TRAIN_DIR.glob("*.yaml")):
        validate_training(path, errors)
    for path in sorted(STABILIZE_TRAIN_DIR.glob("*.yaml")):
        validate_training(path, errors)

    for name in STABLE_CONTEXTS:
        require((STABILIZE_TRAIN_DIR / name).exists(), f"missing stable training YAML {name}", errors)
        require((STABILIZE_PIPELINE_DIR / name).exists(), f"missing stable pipeline YAML {name}", errors)

    legacy_patterns = [
        ROOT / "configs" / "baselines" / "multidag_cl",
        ROOT / "configs" / "pipeline",
        ROOT / "configs" / "analysis",
    ]
    legacy_names = []
    for directory in legacy_patterns:
        legacy_names.extend(path for path in directory.glob("multidag_cl_iemocap_*.yaml"))
        legacy_names.extend(path for path in directory.glob("iemocap_multidag_cl_*.yaml"))
    require(not legacy_names, "legacy root-level MultiDAG+CL YAMLs remain", errors)

    if errors:
        print("Config validation failed:")
        for error in errors:
            print(f"- {error}")
        raise SystemExit(1)

    print("Config validation passed.")


if __name__ == "__main__":
    main()

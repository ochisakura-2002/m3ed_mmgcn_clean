"""Validate canonical MultiDAG+CL config wiring without running experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[2]

FORMAL_PIPELINE_DIR = ROOT / "configs" / "pipeline" / "multidag_cl" / "iemocap" / "formal"
STABILIZE_PIPELINE_DIR = ROOT / "configs" / "pipeline" / "multidag_cl" / "iemocap" / "stabilize"
LOSS_STABILITY_PIPELINE_DIR = ROOT / "configs" / "pipeline" / "multidag_cl" / "iemocap" / "loss_stability"
FORMAL_TRAIN_DIR = ROOT / "configs" / "baselines" / "multidag_cl" / "iemocap" / "formal"
STABILIZE_TRAIN_DIR = ROOT / "configs" / "baselines" / "multidag_cl" / "iemocap" / "stabilize"
LOSS_STABILITY_TRAIN_DIR = ROOT / "configs" / "baselines" / "multidag_cl" / "iemocap" / "loss_stability"

STABLE_CONTEXTS = {
    "context_w0_tav_stable_candidate.yaml": ("causal", 0),
    "context_w3_tav_stable_candidate.yaml": ("causal", 3),
    "context_w5_tav_stable_candidate.yaml": ("causal", 5),
    "context_past_all_causal_tav_stable_candidate.yaml": ("past_all_causal", None),
}

LOSS_STABILITY_CONFIGS = {
    "context_w5_tav_stable_lr3e4_plateau_valloss.yaml": {
        "lr": 0.0003,
        "scheduler_type": "reduce_on_plateau",
        "label_smoothing": 0.0,
        "early_stopping": False,
    },
    "context_w5_tav_stable_lr5e4_plateau_valloss.yaml": {
        "lr": 0.0005,
        "scheduler_type": "reduce_on_plateau",
        "label_smoothing": 0.0,
        "early_stopping": False,
    },
    "context_w5_tav_stable_lr3e4_cosine.yaml": {
        "lr": 0.0003,
        "scheduler_type": "cosine",
        "label_smoothing": 0.0,
        "early_stopping": False,
    },
    "context_w5_tav_stable_lr3e4_label_smoothing005.yaml": {
        "lr": 0.0003,
        "scheduler_type": "none",
        "label_smoothing": 0.05,
        "early_stopping": False,
    },
    "context_w5_tav_stable_lr3e4_earlystop_valloss.yaml": {
        "lr": 0.0003,
        "scheduler_type": "none",
        "label_smoothing": 0.0,
        "early_stopping": True,
    },
}

MONITOR_MODES = {
    "val_loss": "min",
    "val_acc": "max",
    "val_uar": "max",
    "val_macro_f1": "max",
    "val_weighted_f1": "max",
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


def nested(config: dict[str, Any], *keys: str, default: Any = None) -> Any:
    current: Any = config
    for key in keys:
        if not isinstance(current, dict):
            return default
        if key not in current:
            return default
        current = current[key]
    return current


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


def validate_training_control_fields(path: Path, training: dict[str, Any], errors: list[str]) -> None:
    select_best_by = str(training.get("select_best_by", "val_weighted_f1"))
    require(select_best_by in MONITOR_MODES, f"{rel(path)} invalid select_best_by", errors)

    loss = training.get("loss", {})
    require(isinstance(loss, dict), f"{rel(path)} training.loss must be a mapping", errors)
    if isinstance(loss, dict):
        label_smoothing = float(loss.get("label_smoothing", 0.0))
        class_weights = str(loss.get("class_weights", "none"))
        require(
            0.0 <= label_smoothing < 1.0,
            f"{rel(path)} invalid label_smoothing",
            errors,
        )
        require(
            class_weights in {"none", "balanced"},
            f"{rel(path)} invalid class_weights",
            errors,
        )

    optimizer = training.get("optimizer", {})
    require(isinstance(optimizer, dict), f"{rel(path)} training.optimizer must be a mapping", errors)
    if isinstance(optimizer, dict) and optimizer:
        require(
            str(optimizer.get("type", "adamw")) == "adamw",
            f"{rel(path)} invalid optimizer.type",
            errors,
        )
        betas = optimizer.get("betas", [0.9, 0.999])
        require(
            isinstance(betas, list) and len(betas) == 2,
            f"{rel(path)} optimizer.betas must have two values",
            errors,
        )
        if isinstance(betas, list) and len(betas) == 2:
            require(
                all(0.0 <= float(beta) < 1.0 for beta in betas),
                f"{rel(path)} optimizer.betas out of range",
                errors,
            )
        require(float(optimizer.get("eps", 1.0e-8)) > 0.0, f"{rel(path)} optimizer.eps invalid", errors)

    scheduler = training.get("scheduler", {"type": "none"})
    require(isinstance(scheduler, dict), f"{rel(path)} training.scheduler must be a mapping", errors)
    if isinstance(scheduler, dict):
        scheduler_type = str(scheduler.get("type", "none"))
        require(
            scheduler_type in {"none", "reduce_on_plateau", "cosine", "step"},
            f"{rel(path)} invalid scheduler.type",
            errors,
        )
        if scheduler_type == "reduce_on_plateau":
            monitor = str(scheduler.get("monitor", "val_loss"))
            mode = str(scheduler.get("mode", MONITOR_MODES.get(monitor, "")))
            require(
                monitor in {"val_loss", "val_weighted_f1"},
                f"{rel(path)} invalid reduce_on_plateau monitor",
                errors,
            )
            require(
                monitor in MONITOR_MODES and mode == MONITOR_MODES[monitor],
                f"{rel(path)} reduce_on_plateau mode mismatch",
                errors,
            )
            require(
                0.0 < float(scheduler.get("factor", 0.5)) < 1.0,
                f"{rel(path)} invalid scheduler.factor",
                errors,
            )
            require(int(scheduler.get("patience", 3)) >= 0, f"{rel(path)} invalid scheduler.patience", errors)
            require(float(scheduler.get("min_lr", 0.0)) >= 0.0, f"{rel(path)} invalid scheduler.min_lr", errors)
        elif scheduler_type == "cosine":
            require(int(scheduler.get("t_max", 0)) > 0, f"{rel(path)} invalid cosine t_max", errors)
            require(float(scheduler.get("eta_min", 0.0)) >= 0.0, f"{rel(path)} invalid cosine eta_min", errors)
        elif scheduler_type == "step":
            require(int(scheduler.get("step_size", 0)) > 0, f"{rel(path)} invalid step_size", errors)
            require(float(scheduler.get("gamma", 0.5)) > 0.0, f"{rel(path)} invalid scheduler.gamma", errors)

    early = training.get("early_stopping", {"enabled": False})
    require(isinstance(early, dict), f"{rel(path)} training.early_stopping must be a mapping", errors)
    if isinstance(early, dict) and bool(early.get("enabled", False)):
        monitor = str(early.get("monitor", "val_loss"))
        mode = str(early.get("mode", MONITOR_MODES.get(monitor, "")))
        require(monitor in MONITOR_MODES, f"{rel(path)} invalid early_stopping monitor", errors)
        require(
            monitor in MONITOR_MODES and mode == MONITOR_MODES[monitor],
            f"{rel(path)} early_stopping mode mismatch",
            errors,
        )
        require(int(early.get("patience", 8)) > 0, f"{rel(path)} invalid early_stopping patience", errors)
        require(float(early.get("min_delta", 0.0)) >= 0.0, f"{rel(path)} invalid early_stopping min_delta", errors)


def validate_loss_stability_training(path: Path, config: dict[str, Any], errors: list[str]) -> None:
    expected = LOSS_STABILITY_CONFIGS.get(path.name)
    if expected is None:
        errors.append(f"unexpected loss-stability YAML {rel(path)}")
        return

    training = config.get("training", {})
    model = config.get("model", {})
    graph = config.get("graph", {})
    loss = training.get("loss", {})
    scheduler = training.get("scheduler", {"type": "none"})
    early = training.get("early_stopping", {"enabled": False})

    require(model.get("modality_encoder_type") == "causal_gru", f"{rel(path)} loss-stability encoder mismatch", errors)
    require(model.get("num_graph_layers") == 1, f"{rel(path)} loss-stability graph layer mismatch", errors)
    require(model.get("dropout") == 0.2, f"{rel(path)} loss-stability dropout mismatch", errors)
    require(model.get("active_modalities") == ["text", "audio", "visual"], f"{rel(path)} loss-stability modalities mismatch", errors)
    require(graph.get("context_mode") == "causal", f"{rel(path)} loss-stability context_mode mismatch", errors)
    require(graph.get("window_past") == 5, f"{rel(path)} loss-stability window_past mismatch", errors)
    require(training.get("epochs") == 30, f"{rel(path)} loss-stability epochs mismatch", errors)
    require(training.get("batch_size") == 8, f"{rel(path)} loss-stability batch_size mismatch", errors)
    require(training.get("lr") == expected["lr"], f"{rel(path)} loss-stability lr mismatch", errors)
    require(training.get("weight_decay") == 0.0001, f"{rel(path)} loss-stability weight_decay mismatch", errors)
    require(training.get("grad_clip") == 1.0, f"{rel(path)} loss-stability grad_clip mismatch", errors)
    require(training.get("select_best_by") == "val_weighted_f1", f"{rel(path)} loss-stability selector mismatch", errors)
    require(loss.get("label_smoothing", 0.0) == expected["label_smoothing"], f"{rel(path)} label_smoothing mismatch", errors)
    require(str(loss.get("class_weights", "none")) == "none", f"{rel(path)} class_weights mismatch", errors)
    require(str(scheduler.get("type", "none")) == expected["scheduler_type"], f"{rel(path)} scheduler type mismatch", errors)
    require(bool(early.get("enabled", False)) == expected["early_stopping"], f"{rel(path)} early_stopping mismatch", errors)

    if expected["scheduler_type"] == "reduce_on_plateau":
        require(scheduler.get("monitor") == "val_loss", f"{rel(path)} plateau monitor mismatch", errors)
        require(scheduler.get("mode") == "min", f"{rel(path)} plateau mode mismatch", errors)
    if expected["scheduler_type"] == "cosine":
        require(scheduler.get("t_max") == 30, f"{rel(path)} cosine t_max mismatch", errors)
    if expected["early_stopping"]:
        require(early.get("monitor") == "val_loss", f"{rel(path)} early monitor mismatch", errors)
        require(early.get("patience") == 8, f"{rel(path)} early patience mismatch", errors)


def validate_training(path: Path, errors: list[str]) -> None:
    config = load_yaml(path)
    training = config.get("training", {})
    graph = config.get("graph", {})

    validate_training_control_fields(path, training, errors)

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

    if path.parent == LOSS_STABILITY_TRAIN_DIR:
        validate_loss_stability_training(path, config, errors)


def main() -> None:
    errors: list[str] = []

    for path in sorted(FORMAL_PIPELINE_DIR.glob("*.yaml")):
        validate_pipeline(path, errors)
    for path in sorted(STABILIZE_PIPELINE_DIR.glob("*.yaml")):
        validate_pipeline(path, errors)
    for path in sorted(LOSS_STABILITY_PIPELINE_DIR.glob("*.yaml")):
        validate_pipeline(path, errors)

    for path in sorted(FORMAL_TRAIN_DIR.glob("*.yaml")):
        validate_training(path, errors)
    for path in sorted(STABILIZE_TRAIN_DIR.glob("*.yaml")):
        validate_training(path, errors)
    for path in sorted(LOSS_STABILITY_TRAIN_DIR.glob("*.yaml")):
        validate_training(path, errors)

    for name in STABLE_CONTEXTS:
        require((STABILIZE_TRAIN_DIR / name).exists(), f"missing stable training YAML {name}", errors)
        require((STABILIZE_PIPELINE_DIR / name).exists(), f"missing stable pipeline YAML {name}", errors)
    for name in LOSS_STABILITY_CONFIGS:
        require((LOSS_STABILITY_TRAIN_DIR / name).exists(), f"missing loss-stability training YAML {name}", errors)
        require((LOSS_STABILITY_PIPELINE_DIR / name).exists(), f"missing loss-stability pipeline YAML {name}", errors)

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

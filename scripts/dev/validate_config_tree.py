"""Validate canonical baseline config wiring without running experiments."""

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

CAUSAL_BENCHMARK_TRAIN_DIRS = {
    "mmgcn": ROOT / "configs" / "baselines" / "mmgcn" / "iemocap" / "causal_benchmark",
    "multidag_cl": ROOT / "configs" / "baselines" / "multidag_cl" / "iemocap" / "causal_benchmark",
}
CAUSAL_BENCHMARK_PIPELINE_DIRS = {
    "mmgcn": ROOT / "configs" / "pipeline" / "mmgcn" / "iemocap" / "causal_benchmark",
    "multidag_cl": ROOT / "configs" / "pipeline" / "multidag_cl" / "iemocap" / "causal_benchmark",
}

CAUSAL_CONTRACT_VERSION = "1.0"
IEMOCAP_FEATURE_PKL = "third_party/MMGCN_official/IEMOCAP_features/IEMOCAP_features.pkl"
IEMOCAP_FEATURE_SHA256 = "ceb5cc9a45d1792998438e27f86a12642320aa6315546e4425316140ea503fb3"
IEMOCAP_LABELS = ["Happy", "Sad", "Neutral", "Angry", "Excited", "Frustrated"]
CAUSAL_BENCHMARK_VARIANTS = {
    "official_prefix.yaml": ("official_prefix", None),
    "val_ses01.yaml": ("session_holdout", "Ses01"),
    "val_ses02.yaml": ("session_holdout", "Ses02"),
    "val_ses03.yaml": ("session_holdout", "Ses03"),
    "val_ses04.yaml": ("session_holdout", "Ses04"),
}

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


def values_for_key(value: Any, wanted_key: str) -> list[Any]:
    matches: list[Any] = []
    if isinstance(value, dict):
        for key, child in value.items():
            if str(key) == wanted_key:
                matches.append(child)
            matches.extend(values_for_key(child, wanted_key))
    elif isinstance(value, list):
        for child in value:
            matches.extend(values_for_key(child, wanted_key))
    return matches


def test_monitor_paths(value: Any, prefix: str = "") -> list[str]:
    paths: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            key_text = str(key)
            child_path = f"{prefix}.{key_text}" if prefix else key_text
            is_selection_field = (
                "monitor" in key_text.lower()
                or key_text in {"select_best_by", "checkpoint_selection_metric"}
            )
            if is_selection_field and "test" in str(child).lower():
                paths.append(child_path)
            paths.extend(test_monitor_paths(child, child_path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            paths.extend(test_monitor_paths(child, f"{prefix}[{index}]"))
    return paths


def validate_causal_benchmark_common(
    path: Path,
    config: dict[str, Any],
    errors: list[str],
) -> None:
    dataset = config.get("dataset", {})
    require(config.get("causal") is True, f"{rel(path)} must set top-level causal: true", errors)
    require(
        str(config.get("causal_contract_version", "")) == CAUSAL_CONTRACT_VERSION,
        f"{rel(path)} causal_contract_version mismatch",
        errors,
    )
    require(dataset.get("name") == "IEMOCAP", f"{rel(path)} must use IEMOCAP", errors)
    require(
        dataset.get("feature_pkl_path") == IEMOCAP_FEATURE_PKL,
        f"{rel(path)} feature_pkl_path mismatch",
        errors,
    )
    require(
        dataset.get("feature_sha256") == IEMOCAP_FEATURE_SHA256,
        f"{rel(path)} feature_sha256 mismatch",
        errors,
    )
    require(
        not any(bool(value) for value in values_for_key(config, "bidirectional")),
        f"{rel(path)} enables a bidirectional encoder",
        errors,
    )
    monitor_paths = test_monitor_paths(config)
    require(
        not monitor_paths,
        f"{rel(path)} uses test for monitoring/selection at {monitor_paths}",
        errors,
    )


def validate_causal_benchmark_training(
    path: Path,
    family: str,
    errors: list[str],
) -> None:
    config = load_yaml(path)
    validate_causal_benchmark_common(path, config, errors)

    expected_split = CAUSAL_BENCHMARK_VARIANTS.get(path.name)
    if expected_split is None:
        errors.append(f"unexpected causal benchmark training YAML {rel(path)}")
        return

    expected_strategy, expected_session = expected_split
    dataset = config.get("dataset", {})
    model = config.get("model", {})
    graph = config.get("graph", {})

    require(dataset.get("label_list") == IEMOCAP_LABELS, f"{rel(path)} label mapping mismatch", errors)
    require(dataset.get("num_classes") == 6, f"{rel(path)} num_classes mismatch", errors)
    require(
        dataset.get("val_split_strategy") == expected_strategy,
        f"{rel(path)} val_split_strategy does not match filename",
        errors,
    )
    require(
        dataset.get("val_session_id") == expected_session,
        f"{rel(path)} val_session_id does not match filename",
        errors,
    )

    if family == "mmgcn":
        train = config.get("train", {})
        logging = config.get("logging", {})
        require(model.get("name") == "MMGCN", f"{rel(path)} model.name mismatch", errors)
        require(model.get("hidden_dim") == 256, f"{rel(path)} MMGCN hidden_dim mismatch", errors)
        require(model.get("dropout") == 0.5, f"{rel(path)} MMGCN dropout mismatch", errors)
        require(graph.get("context_mode") == "causal", f"{rel(path)} MMGCN graph is not causal", errors)
        require(graph.get("window_future") == 0, f"{rel(path)} MMGCN window_future must be 0", errors)
        require(graph.get("num_layers") == 4, f"{rel(path)} MMGCN num_layers mismatch", errors)
        require(train.get("max_epochs") == 30, f"{rel(path)} MMGCN epochs mismatch", errors)
        require(train.get("batch_size") == 8, f"{rel(path)} MMGCN batch_size mismatch", errors)
        require(train.get("learning_rate") == 0.0001, f"{rel(path)} MMGCN learning_rate mismatch", errors)
        require(train.get("weight_decay") == 0.0001, f"{rel(path)} MMGCN weight_decay mismatch", errors)
        require(train.get("max_train_batches") is None, f"{rel(path)} caps max_train_batches", errors)
        require(train.get("max_val_batches") is None, f"{rel(path)} caps max_val_batches", errors)
        require(
            logging.get("monitor_metric") == "val_weighted_f1",
            f"{rel(path)} MMGCN must select by val_weighted_f1",
            errors,
        )
        return

    if family == "multidag_cl":
        training = config.get("training", {})
        validate_training_control_fields(path, training, errors)
        require(model.get("name") == "MultiDAGCL", f"{rel(path)} model.name mismatch", errors)
        require(model.get("hidden_dim") == 128, f"{rel(path)} MultiDAG hidden_dim mismatch", errors)
        require(model.get("dropout") == 0.2, f"{rel(path)} MultiDAG dropout mismatch", errors)
        require(model.get("num_graph_layers") == 1, f"{rel(path)} MultiDAG graph layer mismatch", errors)
        require(
            model.get("active_modalities") == ["text", "audio", "visual"],
            f"{rel(path)} MultiDAG modalities mismatch",
            errors,
        )
        require(
            model.get("modality_encoder_type") == "causal_gru",
            f"{rel(path)} MultiDAG encoder must be causal_gru",
            errors,
        )
        require(graph.get("context_mode") == "causal", f"{rel(path)} MultiDAG graph is not causal", errors)
        require(graph.get("window_past") == 5, f"{rel(path)} MultiDAG window_past mismatch", errors)
        require(training.get("epochs") == 30, f"{rel(path)} MultiDAG epochs mismatch", errors)
        require(training.get("batch_size") == 8, f"{rel(path)} MultiDAG batch_size mismatch", errors)
        require(training.get("lr") == 0.0005, f"{rel(path)} MultiDAG lr mismatch", errors)
        require(training.get("weight_decay") == 0.0001, f"{rel(path)} MultiDAG weight_decay mismatch", errors)
        require(training.get("grad_clip") == 1.0, f"{rel(path)} MultiDAG grad_clip mismatch", errors)
        require(
            training.get("select_best_by") == "val_weighted_f1",
            f"{rel(path)} MultiDAG must select by val_weighted_f1",
            errors,
        )
        require(training.get("max_train_batches") is None, f"{rel(path)} caps max_train_batches", errors)
        require(training.get("max_val_batches") is None, f"{rel(path)} caps max_val_batches", errors)
        return

    errors.append(f"unsupported causal benchmark family {family}")


def validate_causal_benchmark_pipeline(
    path: Path,
    family: str,
    errors: list[str],
) -> None:
    config = load_yaml(path)
    validate_causal_benchmark_common(path, config, errors)

    expected_train_path = CAUSAL_BENCHMARK_TRAIN_DIRS[family] / path.name
    train_path_text = str(config.get("train", {}).get("train_config_path", ""))
    train_path = ROOT / train_path_text
    expected_model_name = "MMGCN" if family == "mmgcn" else "MultiDAGCL"
    evaluation = config.get("evaluation", {})

    require(path.name in CAUSAL_BENCHMARK_VARIANTS, f"unexpected causal benchmark pipeline YAML {rel(path)}", errors)
    require(config.get("model", {}).get("name") == expected_model_name, f"{rel(path)} model.name mismatch", errors)
    require(bool_at(config, "train", "enabled"), f"{rel(path)} must enable training", errors)
    require(train_path.exists(), f"{rel(path)} points to missing {train_path_text}", errors)
    require(
        train_path.resolve() == expected_train_path.resolve(),
        f"{rel(path)} must point to matching causal benchmark training YAML",
        errors,
    )
    require(bool_at(config, "evaluation", "enabled"), f"{rel(path)} must enable evaluation", errors)
    require(evaluation.get("checkpoint_name") == "best_model.pt", f"{rel(path)} checkpoint mismatch", errors)
    require(evaluation.get("splits") == ["val", "test"], f"{rel(path)} must evaluate full val/test", errors)
    require(evaluation.get("max_batches") is None, f"{rel(path)} caps evaluation batches", errors)
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


def validate_causal_benchmark_tree(errors: list[str]) -> None:
    expected_names = set(CAUSAL_BENCHMARK_VARIANTS)
    for family, train_dir in CAUSAL_BENCHMARK_TRAIN_DIRS.items():
        pipeline_dir = CAUSAL_BENCHMARK_PIPELINE_DIRS[family]
        train_names = {path.name for path in train_dir.glob("*.yaml")}
        pipeline_names = {path.name for path in pipeline_dir.glob("*.yaml")}
        require(
            train_names == expected_names,
            f"{rel(train_dir)} must contain exactly {sorted(expected_names)}; got {sorted(train_names)}",
            errors,
        )
        require(
            pipeline_names == expected_names,
            f"{rel(pipeline_dir)} must contain exactly {sorted(expected_names)}; got {sorted(pipeline_names)}",
            errors,
        )
        for name in sorted(expected_names):
            train_path = train_dir / name
            pipeline_path = pipeline_dir / name
            if train_path.exists():
                validate_causal_benchmark_training(train_path, family, errors)
            if pipeline_path.exists():
                validate_causal_benchmark_pipeline(pipeline_path, family, errors)


def main() -> None:
    errors: list[str] = []

    validate_causal_benchmark_tree(errors)

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

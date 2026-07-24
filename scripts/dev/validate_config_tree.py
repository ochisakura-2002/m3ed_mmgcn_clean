"""Validate canonical baseline config wiring without running experiments."""

from __future__ import annotations

import copy
from collections import Counter
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
    "mmgcn": (
        ROOT
        / "configs"
        / "mmgcn"
        / "unified"
        / "iemocap"
        / "causal_context"
        / "legacy_mmgcn_features"
    ),
    "multidag_cl": (
        ROOT
        / "configs"
        / "multidag_cl"
        / "unified"
        / "iemocap"
        / "causal_context"
        / "legacy_mmgcn_features"
    ),
    "gsmcc": ROOT / "configs" / "baselines" / "gsmcc" / "iemocap" / "causal_benchmark",
    "dialoguegcn": (
        ROOT
        / "configs"
        / "dialoguegcn"
        / "unified"
        / "iemocap"
        / "causal_context"
        / "legacy_mmgcn_features"
    ),
}
CAUSAL_BENCHMARK_PIPELINE_DIRS = {
    "mmgcn": ROOT / "configs" / "pipeline" / "mmgcn" / "iemocap" / "causal_benchmark",
    "multidag_cl": ROOT / "configs" / "pipeline" / "multidag_cl" / "iemocap" / "causal_benchmark",
    "gsmcc": ROOT / "configs" / "pipeline" / "gsmcc" / "iemocap" / "causal_benchmark",
    "dialoguegcn": ROOT / "configs" / "pipeline" / "dialoguegcn" / "iemocap" / "causal_benchmark",
}
CLEAN_ROBERTA_V1_TRAIN_DIRS = {
    "mmgcn": (
        ROOT
        / "configs"
        / "mmgcn"
        / "unified"
        / "iemocap"
        / "causal_context"
        / "clean_roberta_features"
    ),
    "multidag_cl": (
        ROOT
        / "configs"
        / "multidag_cl"
        / "unified"
        / "iemocap"
        / "causal_context"
        / "clean_roberta_features"
    ),
    "dialoguegcn": (
        ROOT
        / "configs"
        / "dialoguegcn"
        / "unified"
        / "iemocap"
        / "causal_context"
        / "clean_roberta_features"
    ),
    "gsmcc": (
        ROOT
        / "configs"
        / "baselines"
        / "gsmcc"
        / "iemocap"
        / "clean_roberta_v1"
    ),
}
CLEAN_ROBERTA_V1_FORMAL_FAMILIES = ("mmgcn", "multidag_cl")
CLEAN_ROBERTA_V1_PIPELINE_DIRS = {
    family: ROOT / "configs" / "pipeline" / family / "iemocap" / "clean_roberta_v1"
    for family in CLEAN_ROBERTA_V1_FORMAL_FAMILIES
}
CLEAN_ROBERTA_V1_MANIFEST = (
    ROOT / "configs" / "experiments" / "iemocap_clean_roberta_v1_8run.yaml"
)

CAUSAL_CONTRACT_VERSION = "1.0"
IEMOCAP_FEATURE_PKL = "third_party/MMGCN_official/IEMOCAP_features/IEMOCAP_features.pkl"
IEMOCAP_FEATURE_SHA256 = "ceb5cc9a45d1792998438e27f86a12642320aa6315546e4425316140ea503fb3"
IEMOCAP_LABELS = ["Happy", "Sad", "Neutral", "Angry", "Excited", "Frustrated"]
CLEAN_ROBERTA_V1_PATH = (
    "data/processed/iemocap/"
    "IEMOCAP_features_clean_roberta_base_c8b8a37_utterance_mean_v1.pkl"
)
CLEAN_ROBERTA_V1_NAME = "iemocap_clean_roberta_base_utterance_mean_v1"
CLEAN_ROBERTA_V1_SHA256 = "c604c557bc3fbb129ca02b2acd57166b669a89ef76ff0cea1e14f9cb206324bf"
IEMOCAP_FEATURE_REGISTRY = (
    ROOT / "configs" / "_shared" / "data" / "iemocap" / "feature_sets.yaml"
)
ORIGINAL_REPRO_LEGACY_SMOKE_DIR = ROOT / "configs" / "smoke" / "original_repro"
ORIGINAL_MERC_EXPERIMENT_DIR = ROOT / "configs" / "experiments" / "original_merc"
ORIGINAL_MERC_CANONICAL_FORMAL_CONFIGS = {
    "mmgcn": {
        "screening": (
            ROOT
            / "configs/mmgcn/paper_aligned/iemocap/full_context/"
            "legacy_mmgcn_features/screening.yaml"
        ),
        "clean_screening": (
            ROOT
            / "configs/mmgcn/paper_aligned/iemocap/full_context/"
            "clean_roberta_features/mmgcn_clean.yaml"
        ),
        "legacy_fold_bases": (
            ROOT
            / "configs/mmgcn/paper_aligned/iemocap/full_context/"
            "legacy_mmgcn_features/fivefold_base.yaml"
        ),
        "clean_fold_bases": (
            ROOT
            / "configs/mmgcn/paper_aligned/iemocap/full_context/"
            "clean_roberta_features/fivefold_base.yaml"
        ),
    },
    "multidag_cl": {
        "screening": (
            ROOT
            / "configs/multidag_cl/paper_aligned/iemocap/full_context/"
            "legacy_mmgcn_features/screening.yaml"
        ),
        "clean_screening": (
            ROOT
            / "configs/multidag_cl/paper_aligned/iemocap/full_context/"
            "clean_roberta_features/multidag_cl_clean.yaml"
        ),
        "legacy_fold_bases": (
            ROOT
            / "configs/multidag_cl/paper_aligned/iemocap/full_context/"
            "legacy_mmgcn_features/fivefold_base.yaml"
        ),
        "clean_fold_bases": (
            ROOT
            / "configs/multidag_cl/paper_aligned/iemocap/full_context/"
            "clean_roberta_features/fivefold_base.yaml"
        ),
    },
    "dialoguegcn": {
        "screening": (
            ROOT
            / "configs/dialoguegcn/paper_aligned/iemocap/full_context/"
            "legacy_mmgcn_features/screening.yaml"
        ),
        "clean_screening": (
            ROOT
            / "configs/dialoguegcn/paper_aligned/iemocap/full_context/"
            "clean_roberta_features/dialoguegcn_clean.yaml"
        ),
        "legacy_fold_bases": (
            ROOT
            / "configs/dialoguegcn/paper_aligned/iemocap/full_context/"
            "legacy_mmgcn_features/fivefold_base.yaml"
        ),
        "clean_fold_bases": (
            ROOT
            / "configs/dialoguegcn/paper_aligned/iemocap/full_context/"
            "clean_roberta_features/fivefold_base.yaml"
        ),
    },
}
ORIGINAL_REPRO_MODELS = {
    "mmgcn": "original_repro_mmgcn",
    "multidag_cl": "original_repro_multidag_cl",
    "gsmcc": "project_paper_oriented_gsmcc",
    "dialoguegcn": "original_repro_dialoguegcn",
}
ORIGINAL_REPRO_SMOKE_CONFIGS = {
    ("mmgcn", "legacy"): (
        ROOT
        / "configs"
        / "mmgcn"
        / "paper_aligned"
        / "iemocap"
        / "full_context"
        / "legacy_mmgcn_features"
        / "smoke.yaml"
    ),
    ("mmgcn", "clean"): (
        ROOT
        / "configs"
        / "mmgcn"
        / "paper_aligned"
        / "iemocap"
        / "full_context"
        / "clean_roberta_features"
        / "smoke.yaml"
    ),
    ("multidag_cl", "legacy"): (
        ROOT
        / "configs"
        / "multidag_cl"
        / "paper_aligned"
        / "iemocap"
        / "full_context"
        / "legacy_mmgcn_features"
        / "smoke.yaml"
    ),
    ("multidag_cl", "clean"): (
        ROOT
        / "configs"
        / "multidag_cl"
        / "paper_aligned"
        / "iemocap"
        / "full_context"
        / "clean_roberta_features"
        / "smoke.yaml"
    ),
    ("dialoguegcn", "legacy"): (
        ROOT
        / "configs"
        / "dialoguegcn"
        / "paper_aligned"
        / "iemocap"
        / "full_context"
        / "legacy_mmgcn_features"
        / "smoke.yaml"
    ),
    ("dialoguegcn", "clean"): (
        ROOT
        / "configs"
        / "dialoguegcn"
        / "paper_aligned"
        / "iemocap"
        / "full_context"
        / "clean_roberta_features"
        / "smoke.yaml"
    ),
    ("gsmcc", "legacy"): ORIGINAL_REPRO_LEGACY_SMOKE_DIR / "gsmcc_legacy.yaml",
    ("gsmcc", "clean"): ORIGINAL_REPRO_LEGACY_SMOKE_DIR / "gsmcc_clean.yaml",
}
ORIGINAL_MERC_TRACKS = {
    "legacy_official_split_safe_selection": (
        "official_train_stratified",
        "paper_adjacent_not_exact",
    ),
    "legacy_fivefold_fair_comparison": (
        "outer_session_stratified",
        "fair_comparison_not_paper_reproduction",
    ),
    "clean_roberta_fivefold_fair_comparison": (
        "outer_session_stratified",
        "fair_comparison_not_paper_reproduction",
    ),
}
CAUSAL_BENCHMARK_VARIANTS = {
    "official_prefix.yaml": ("official_prefix", None),
    "val_ses01.yaml": ("session_holdout", "Ses01"),
    "val_ses02.yaml": ("session_holdout", "Ses02"),
    "val_ses03.yaml": ("session_holdout", "Ses03"),
    "val_ses04.yaml": ("session_holdout", "Ses04"),
}
CAUSAL_BENCHMARK_ALLOWED_TRAIN_EXTRAS = {
    "dialoguegcn": {"smoke_real_2epoch.yaml"},
    "multidag_cl": {
        "context_past_all_causal_tav_smoke.yaml",
        "context_w5_tav_quick.yaml",
        "context_w5_tav_smoke.yaml",
    },
}


def causal_training_name(family: str, pipeline_name: str) -> str:
    if (
        family in {"mmgcn", "multidag_cl", "dialoguegcn"}
        and pipeline_name == "official_prefix.yaml"
    ):
        return "val_official_prefix.yaml"
    return pipeline_name


def causal_pipeline_name(family: str, training_name: str) -> str:
    if (
        family in {"mmgcn", "multidag_cl", "dialoguegcn"}
        and training_name == "val_official_prefix.yaml"
    ):
        return "official_prefix.yaml"
    return training_name


CLEAN_ROBERTA_V1_FORMAL_VARIANTS = {
    name: split
    for name, split in CAUSAL_BENCHMARK_VARIANTS.items()
    if split[0] == "session_holdout"
}
CLEAN_ROBERTA_V1_MODEL_NAMES = {
    "mmgcn": "MMGCN",
    "multidag_cl": "MultiDAGCL",
}
CLEAN_ROBERTA_V1_FORBIDDEN_CAPS = (
    "train_batch_cap",
    "val_batch_cap",
    "test_batch_cap",
    "max_train_batches",
    "max_val_batches",
    "max_test_batches",
)

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


def string_values(value: Any) -> list[str]:
    matches: list[str] = []
    if isinstance(value, dict):
        for child in value.values():
            matches.extend(string_values(child))
    elif isinstance(value, list):
        for child in value:
            matches.extend(string_values(child))
    elif isinstance(value, str):
        matches.append(value)
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

    expected_split = CAUSAL_BENCHMARK_VARIANTS.get(
        causal_pipeline_name(family, path.name)
    )
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

    if family in {"gsmcc", "dialoguegcn"}:
        training = config.get("training", {})
        optimizer = config.get("optimizer", {})
        scheduler = config.get("scheduler", {})
        expected_name = (
            "causal_gsmcc_inspired" if family == "gsmcc" else "causal_dialoguegcn"
        )
        require(model.get("name") == expected_name, f"{rel(path)} model.name mismatch", errors)
        require(model.get("text_dim") == 100, f"{rel(path)} text_dim mismatch", errors)
        require(model.get("audio_dim") == 1582, f"{rel(path)} audio_dim mismatch", errors)
        require(model.get("visual_dim") == 342, f"{rel(path)} visual_dim mismatch", errors)
        require(model.get("hidden_dim") == 128, f"{rel(path)} hidden_dim mismatch", errors)
        require(model.get("num_classes") == 6, f"{rel(path)} model num_classes mismatch", errors)
        require(model.get("dropout") == 0.2, f"{rel(path)} dropout mismatch", errors)
        require(model.get("num_graph_layers") == 1, f"{rel(path)} graph layer mismatch", errors)
        require(graph.get("context_mode") == "causal", f"{rel(path)} graph is not causal", errors)
        require(graph.get("window_past") == 5, f"{rel(path)} window_past mismatch", errors)
        require(graph.get("window_future") == 0, f"{rel(path)} window_future must be 0", errors)
        require(training.get("epochs") == 30, f"{rel(path)} epochs mismatch", errors)
        require(training.get("batch_size") == 8, f"{rel(path)} batch_size mismatch", errors)
        require(training.get("seed") == 42, f"{rel(path)} training seed mismatch", errors)
        require(training.get("grad_clip") == 1.0, f"{rel(path)} grad_clip mismatch", errors)
        require(
            training.get("select_best_by") == "val_weighted_f1",
            f"{rel(path)} must select by val_weighted_f1",
            errors,
        )
        for cap_name in (
            "train_batch_cap",
            "val_batch_cap",
            "test_batch_cap",
            "max_train_batches",
            "max_val_batches",
            "max_test_batches",
        ):
            require(cap_name not in training, f"{rel(path)} contains forbidden {cap_name}", errors)
        require(str(optimizer.get("name", "")).lower() == "adamw", f"{rel(path)} optimizer mismatch", errors)
        require(optimizer.get("learning_rate") == 0.0005, f"{rel(path)} learning_rate mismatch", errors)
        require(optimizer.get("weight_decay") == 0.0001, f"{rel(path)} weight_decay mismatch", errors)
        require(str(scheduler.get("name", "")).lower() == "none", f"{rel(path)} scheduler mismatch", errors)
        require(dataset.get("text_feature_dim") == 100, f"{rel(path)} dataset text dim mismatch", errors)
        require(dataset.get("audio_feature_dim") == 1582, f"{rel(path)} dataset audio dim mismatch", errors)
        require(dataset.get("visual_feature_dim") == 342, f"{rel(path)} dataset visual dim mismatch", errors)
        if family == "gsmcc":
            loss = config.get("loss", {})
            require(model.get("modality_encoder_type") == "linear", f"{rel(path)} GS-MCC encoder mismatch", errors)
            require(model.get("fusion_type") == "concat", f"{rel(path)} GS-MCC fusion mismatch", errors)
            require(model.get("num_filter_steps") == 2, f"{rel(path)} filter step mismatch", errors)
            require(loss.get("classification_weight") == 1.0, f"{rel(path)} classification weight mismatch", errors)
            require(loss.get("consistency_weight") == 0.0, f"{rel(path)} consistency loss must be disabled", errors)
            require(loss.get("complementarity_weight") == 0.0, f"{rel(path)} complementarity loss must be disabled", errors)
            require(
                config.get("official_fgo_reproduced") is not True,
                f"{rel(path)} incorrectly claims official FGO reproduction",
                errors,
            )
        else:
            require(model.get("context_hidden_dim") == 128, f"{rel(path)} context hidden mismatch", errors)
            require(model.get("graph_hidden_dim") == 128, f"{rel(path)} graph hidden mismatch", errors)
            require(model.get("context_encoder_type") == "causal_gru", f"{rel(path)} context encoder mismatch", errors)
            require(model.get("num_speakers") == 2, f"{rel(path)} num_speakers mismatch", errors)
            require(model.get("nodal_attention") == "none", f"{rel(path)} full nodal attention is forbidden", errors)
            require(config.get("loss", {}).get("class_weight_mode") == "none", f"{rel(path)} class weights must be disabled", errors)
        return

    errors.append(f"unsupported causal benchmark family {family}")


def validate_causal_benchmark_pipeline(
    path: Path,
    family: str,
    errors: list[str],
) -> None:
    config = load_yaml(path)
    validate_causal_benchmark_common(path, config, errors)

    expected_train_path = (
        CAUSAL_BENCHMARK_TRAIN_DIRS[family]
        / causal_training_name(family, path.name)
    )
    train_path_text = str(config.get("train", {}).get("train_config_path", ""))
    train_path = ROOT / train_path_text
    expected_model_names = {
        "mmgcn": "MMGCN",
        "multidag_cl": "MultiDAGCL",
        "gsmcc": "causal_gsmcc_inspired",
        "dialoguegcn": "causal_dialoguegcn",
    }
    expected_model_name = expected_model_names[family]
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
    pipeline_names_expected = set(CAUSAL_BENCHMARK_VARIANTS)
    for family, train_dir in CAUSAL_BENCHMARK_TRAIN_DIRS.items():
        pipeline_dir = CAUSAL_BENCHMARK_PIPELINE_DIRS[family]
        train_names_expected = {
            causal_training_name(family, name)
            for name in pipeline_names_expected
        }
        train_names_expected.update(
            CAUSAL_BENCHMARK_ALLOWED_TRAIN_EXTRAS.get(family, set())
        )
        train_names = {path.name for path in train_dir.glob("*.yaml")}
        pipeline_names = {path.name for path in pipeline_dir.glob("*.yaml")}
        require(
            train_names == train_names_expected,
            f"{rel(train_dir)} must contain exactly "
            f"{sorted(train_names_expected)}; got {sorted(train_names)}",
            errors,
        )
        require(
            pipeline_names == pipeline_names_expected,
            f"{rel(pipeline_dir)} must contain exactly "
            f"{sorted(pipeline_names_expected)}; got {sorted(pipeline_names)}",
            errors,
        )
        for pipeline_name in sorted(pipeline_names_expected):
            train_path = train_dir / causal_training_name(
                family, pipeline_name
            )
            pipeline_path = pipeline_dir / pipeline_name
            if train_path.exists():
                validate_causal_benchmark_training(train_path, family, errors)
            if pipeline_path.exists():
                validate_causal_benchmark_pipeline(pipeline_path, family, errors)

        if family in {"gsmcc", "dialoguegcn"}:
            normalized_folds = []
            for name in ("val_ses01.yaml", "val_ses02.yaml", "val_ses03.yaml", "val_ses04.yaml"):
                fold = copy.deepcopy(load_yaml(train_dir / name))
                fold.pop("run_name", None)
                fold.get("dataset", {}).pop("val_session_id", None)
                normalized_folds.append(fold)
            require(
                all(fold == normalized_folds[0] for fold in normalized_folds[1:]),
                f"{rel(train_dir)} session-holdout folds differ beyond run_name/val_session_id",
                errors,
            )


def validate_clean_roberta_v1_registry(
    registry: dict[str, Any], errors: list[str]
) -> None:
    """Validate frozen Clean v1 fields while preserving the legacy contract."""

    legacy = registry.get("legacy_mmgcn_textcnn", {})
    clean = registry.get("clean_roberta_v1", {})
    require(
        legacy.get("path") == IEMOCAP_FEATURE_PKL,
        "IEMOCAP legacy registry path mismatch",
        errors,
    )
    require(
        legacy.get("sha256") == IEMOCAP_FEATURE_SHA256,
        "IEMOCAP legacy registry SHA256 mismatch",
        errors,
    )
    require(legacy.get("text_dim") == 100, "IEMOCAP legacy registry text_dim mismatch", errors)
    require(clean.get("path") == CLEAN_ROBERTA_V1_PATH, "clean v1 registry path mismatch", errors)
    require(
        clean.get("feature_set_name") == CLEAN_ROBERTA_V1_NAME,
        "clean v1 registry feature_set_name mismatch",
        errors,
    )
    require(clean.get("sha256") == CLEAN_ROBERTA_V1_SHA256, "clean v1 registry SHA mismatch", errors)
    require(clean.get("text_dim") == 768, "clean v1 registry text_dim mismatch", errors)
    require(
        clean.get("status") == "frozen_for_main_experiments",
        "clean v1 registry status mismatch",
        errors,
    )


def validate_clean_roberta_v1_smoke_config(
    path: Path,
    family: str,
    config: dict[str, Any],
    errors: list[str],
) -> None:
    """Validate one smoke config against the immutable Clean v1 contract."""

    dataset = config.get("dataset", {})
    model = config.get("model", {})
    require(dataset.get("name") == "IEMOCAP", f"{rel(path)} must use IEMOCAP", errors)
    require(
        dataset.get("feature_pkl_path") == CLEAN_ROBERTA_V1_PATH,
        f"{rel(path)} clean feature path mismatch",
        errors,
    )
    require(
        dataset.get("feature_set_name") == CLEAN_ROBERTA_V1_NAME,
        f"{rel(path)} feature_set_name mismatch",
        errors,
    )
    require(
        dataset.get("feature_sha256") == CLEAN_ROBERTA_V1_SHA256,
        f"{rel(path)} clean feature SHA mismatch",
        errors,
    )
    require(
        dataset.get("allow_unpinned_feature_for_smoke") is not True,
        f"{rel(path)} must not enable unpinned clean features",
        errors,
    )
    if family in {"mmgcn", "multidag_cl"}:
        require(
            model.get("text_feature_dim") == 768,
            f"{rel(path)} text_feature_dim mismatch",
            errors,
        )
    else:
        require(model.get("text_dim") == 768, f"{rel(path)} text_dim mismatch", errors)
        require(
            dataset.get("text_feature_dim") == 768,
            f"{rel(path)} dataset text_feature_dim mismatch",
            errors,
        )
    epochs = (
        config.get("train", {}).get("max_epochs")
        if family == "mmgcn"
        else config.get("training", {}).get("epochs")
    )
    require(epochs == 2, f"{rel(path)} must run exactly two smoke epochs", errors)


def clean_roberta_v1_training_expected(path: Path, family: str) -> dict[str, Any]:
    """Derive the only permitted Clean v1 formal config from its legacy template."""

    source_path = CAUSAL_BENCHMARK_TRAIN_DIRS[family] / path.name
    expected = copy.deepcopy(load_yaml(source_path))
    session = CLEAN_ROBERTA_V1_FORMAL_VARIANTS[path.name][1]
    experiment_name = f"{family}_iemocap_clean_roberta_v1_val_{session.lower()}"

    expected["project"]["experiment_name"] = experiment_name
    expected["dataset"]["feature_pkl_path"] = CLEAN_ROBERTA_V1_PATH
    expected["dataset"]["feature_set_name"] = CLEAN_ROBERTA_V1_NAME
    expected["dataset"]["feature_sha256"] = CLEAN_ROBERTA_V1_SHA256
    expected["dataset"]["test_session_id"] = "Ses05"
    expected["model"]["text_feature_dim"] = 768
    if family == "multidag_cl":
        expected["graph"]["window_future"] = 0
        expected["training"].pop("max_train_batches", None)
        expected["training"].pop("max_val_batches", None)
        expected["output"]["experiment_name"] = experiment_name
    expected["protocol"] = {
        "run_type": "formal",
        "checkpoint_selection_metric": "val_weighted_f1",
        "test_split_used_for_selection": False,
        "future_context_allowed": False,
    }
    expected["provenance"] = {
        "source_config": rel(source_path),
        "feature_registry_key": "clean_roberta_v1",
        "feature_set_name": CLEAN_ROBERTA_V1_NAME,
        "feature_sha256": CLEAN_ROBERTA_V1_SHA256,
    }
    return expected


def clean_roberta_v1_pipeline_expected(path: Path, family: str) -> dict[str, Any]:
    """Derive the only permitted Clean v1 pipeline config from its legacy template."""

    source_path = CAUSAL_BENCHMARK_PIPELINE_DIRS[family] / path.name
    expected = copy.deepcopy(load_yaml(source_path))
    session = CLEAN_ROBERTA_V1_FORMAL_VARIANTS[path.name][1]
    pipeline_name = f"{family}_iemocap_clean_roberta_v1_val_{session.lower()}"
    train_path = CLEAN_ROBERTA_V1_TRAIN_DIRS[family] / path.name

    expected["project"]["pipeline_name"] = pipeline_name
    expected["dataset"]["feature_pkl_path"] = CLEAN_ROBERTA_V1_PATH
    expected["dataset"]["feature_set_name"] = CLEAN_ROBERTA_V1_NAME
    expected["dataset"]["feature_sha256"] = CLEAN_ROBERTA_V1_SHA256
    expected["dataset"]["val_split_strategy"] = "session_holdout"
    expected["dataset"]["val_session_id"] = session
    expected["dataset"]["test_session_id"] = "Ses05"
    expected["model"]["text_feature_dim"] = 768
    expected["model"]["audio_feature_dim"] = 1582
    expected["model"]["visual_feature_dim"] = 342
    expected["protocol"] = {
        "run_type": "formal",
        "seed": 42,
        "epochs": 30,
        "checkpoint_selection_metric": "val_weighted_f1",
        "test_split_used_for_selection": False,
        "future_context_allowed": False,
    }
    expected["train"]["train_config_path"] = rel(train_path)
    expected["evaluation"]["checkpoint_selected_by"] = "val_weighted_f1"
    expected["evaluation"]["test_session_id"] = "Ses05"
    expected["evaluation"]["test_split_used_for_selection"] = False
    expected["provenance"] = {
        "source_config": rel(source_path),
        "feature_registry_key": "clean_roberta_v1",
    }
    return expected


def validate_clean_roberta_v1_formal_common(
    path: Path,
    config: dict[str, Any],
    expected_session: str,
    errors: list[str],
) -> None:
    dataset = config.get("dataset", {})
    model = config.get("model", {})
    protocol = config.get("protocol", {})

    require(config.get("causal") is True, f"{rel(path)} must set causal: true", errors)
    require(
        config.get("causal_contract_version") == CAUSAL_CONTRACT_VERSION,
        f"{rel(path)} causal contract mismatch",
        errors,
    )
    require(dataset.get("name") == "IEMOCAP", f"{rel(path)} must use IEMOCAP", errors)
    require(
        dataset.get("feature_pkl_path") == CLEAN_ROBERTA_V1_PATH,
        f"{rel(path)} clean feature path mismatch",
        errors,
    )
    require(
        dataset.get("feature_set_name") == CLEAN_ROBERTA_V1_NAME,
        f"{rel(path)} clean feature set mismatch",
        errors,
    )
    require(
        dataset.get("feature_sha256") == CLEAN_ROBERTA_V1_SHA256,
        f"{rel(path)} clean feature SHA mismatch",
        errors,
    )
    require(
        "allow_unpinned_feature_for_smoke" not in dataset,
        f"{rel(path)} must not allow an unpinned feature",
        errors,
    )
    require(
        dataset.get("val_split_strategy") == "session_holdout",
        f"{rel(path)} must use session_holdout validation",
        errors,
    )
    require(
        dataset.get("val_session_id") == expected_session,
        f"{rel(path)} validation session does not match filename",
        errors,
    )
    require(dataset.get("test_session_id") == "Ses05", f"{rel(path)} test must be Ses05", errors)
    require(model.get("text_feature_dim") == 768, f"{rel(path)} text dim mismatch", errors)
    require(model.get("audio_feature_dim") == 1582, f"{rel(path)} audio dim mismatch", errors)
    require(model.get("visual_feature_dim") == 342, f"{rel(path)} visual dim mismatch", errors)
    require(
        protocol.get("checkpoint_selection_metric") == "val_weighted_f1",
        f"{rel(path)} must select checkpoints by val_weighted_f1",
        errors,
    )
    test_selection_flags = values_for_key(config, "test_split_used_for_selection")
    require(
        bool(test_selection_flags) and all(value is False for value in test_selection_flags),
        f"{rel(path)} must explicitly exclude test from selection",
        errors,
    )
    require(
        protocol.get("future_context_allowed") is False,
        f"{rel(path)} must forbid future context",
        errors,
    )
    require(
        not any(bool(value) for value in values_for_key(config, "bidirectional")),
        f"{rel(path)} enables a bidirectional encoder",
        errors,
    )
    require(
        not test_monitor_paths(config),
        f"{rel(path)} uses test for checkpoint or epoch selection",
        errors,
    )
    for cap_name in CLEAN_ROBERTA_V1_FORBIDDEN_CAPS:
        require(
            not values_for_key(config, cap_name),
            f"{rel(path)} contains forbidden {cap_name}",
            errors,
        )
    lowered_values = [value.lower() for value in string_values(config)]
    forbidden_markers = ("placeholder", "to_be_filled", "smoke", "quick", "debug")
    found_markers = sorted(
        marker
        for marker in forbidden_markers
        if any(marker in value for value in lowered_values)
    )
    require(
        not found_markers,
        f"{rel(path)} contains forbidden formal-run markers {found_markers}",
        errors,
    )


def validate_clean_roberta_v1_formal_training(
    path: Path,
    family: str,
    config: dict[str, Any],
    errors: list[str],
) -> None:
    expected_variant = CLEAN_ROBERTA_V1_FORMAL_VARIANTS.get(path.name)
    if expected_variant is None:
        errors.append(f"unexpected Clean RoBERTa v1 training YAML {rel(path)}")
        return
    expected_session = expected_variant[1]
    assert expected_session is not None
    validate_clean_roberta_v1_formal_common(path, config, expected_session, errors)

    graph = config.get("graph", {})
    require(config.get("system", {}).get("seed") == 42, f"{rel(path)} seed mismatch", errors)
    require(graph.get("context_mode") == "causal", f"{rel(path)} graph is not causal", errors)
    require(graph.get("window_future") == 0, f"{rel(path)} window_future must be 0", errors)
    if family == "mmgcn":
        require(config.get("train", {}).get("max_epochs") == 30, f"{rel(path)} epochs mismatch", errors)
        require(
            config.get("logging", {}).get("monitor_metric") == "val_weighted_f1",
            f"{rel(path)} MMGCN selection metric mismatch",
            errors,
        )
    else:
        require(config.get("training", {}).get("epochs") == 30, f"{rel(path)} epochs mismatch", errors)
        require(
            config.get("training", {}).get("select_best_by") == "val_weighted_f1",
            f"{rel(path)} MultiDAG selection metric mismatch",
            errors,
        )
        require(graph.get("window_past") == 5, f"{rel(path)} MultiDAG window_past mismatch", errors)

    expected = clean_roberta_v1_training_expected(path, family)
    require(
        config == expected,
        f"{rel(path)} differs from its legacy formal template beyond allowed fields",
        errors,
    )


def validate_clean_roberta_v1_formal_pipeline(
    path: Path,
    family: str,
    config: dict[str, Any],
    errors: list[str],
) -> None:
    expected_variant = CLEAN_ROBERTA_V1_FORMAL_VARIANTS.get(path.name)
    if expected_variant is None:
        errors.append(f"unexpected Clean RoBERTa v1 pipeline YAML {rel(path)}")
        return
    expected_session = expected_variant[1]
    assert expected_session is not None
    validate_clean_roberta_v1_formal_common(path, config, expected_session, errors)

    protocol = config.get("protocol", {})
    evaluation = config.get("evaluation", {})
    train_path_text = str(config.get("train", {}).get("train_config_path", ""))
    train_path = ROOT / train_path_text
    expected_train_path = CLEAN_ROBERTA_V1_TRAIN_DIRS[family] / path.name
    require(config.get("model", {}).get("name") == CLEAN_ROBERTA_V1_MODEL_NAMES[family], f"{rel(path)} model mismatch", errors)
    require(protocol.get("seed") == 42, f"{rel(path)} seed mismatch", errors)
    require(protocol.get("epochs") == 30, f"{rel(path)} epochs mismatch", errors)
    require(bool_at(config, "train", "enabled"), f"{rel(path)} must enable training", errors)
    require(train_path.exists(), f"{rel(path)} points to missing {train_path_text}", errors)
    require(
        train_path.resolve() == expected_train_path.resolve(),
        f"{rel(path)} must point to its matching Clean v1 training YAML",
        errors,
    )
    require(bool_at(config, "evaluation", "enabled"), f"{rel(path)} must enable evaluation", errors)
    require(evaluation.get("checkpoint_name") == "best_model.pt", f"{rel(path)} checkpoint mismatch", errors)
    require(
        evaluation.get("checkpoint_selected_by") == "val_weighted_f1",
        f"{rel(path)} evaluated checkpoint was not validation-selected",
        errors,
    )
    require(evaluation.get("splits") == ["val", "test"], f"{rel(path)} must evaluate val/test", errors)
    require(evaluation.get("test_session_id") == "Ses05", f"{rel(path)} test must be Ses05", errors)
    require(evaluation.get("max_batches") is None, f"{rel(path)} caps evaluation batches", errors)

    expected = clean_roberta_v1_pipeline_expected(path, family)
    require(
        config == expected,
        f"{rel(path)} differs from its legacy pipeline template beyond allowed fields",
        errors,
    )


def validate_clean_roberta_v1_manifest(errors: list[str]) -> None:
    path = CLEAN_ROBERTA_V1_MANIFEST
    if not path.exists():
        errors.append(f"missing {rel(path)}")
        return
    manifest = load_yaml(path)
    feature = manifest.get("feature", {})
    protocol = manifest.get("protocol", {})
    execution = manifest.get("execution", {})
    runs = manifest.get("runs", [])

    require(manifest.get("expected_run_count") == 8, f"{rel(path)} expected run count mismatch", errors)
    require(execution.get("mode") == "sequential", f"{rel(path)} must execute sequentially", errors)
    require(execution.get("max_concurrent_runs") == 1, f"{rel(path)} concurrency must be one", errors)
    require(feature.get("feature_set_name") == CLEAN_ROBERTA_V1_NAME, f"{rel(path)} feature set mismatch", errors)
    require(feature.get("path") == CLEAN_ROBERTA_V1_PATH, f"{rel(path)} feature path mismatch", errors)
    require(feature.get("sha256") == CLEAN_ROBERTA_V1_SHA256, f"{rel(path)} feature SHA mismatch", errors)
    require(feature.get("text_dim") == 768, f"{rel(path)} text dim mismatch", errors)
    require(feature.get("audio_dim") == 1582, f"{rel(path)} audio dim mismatch", errors)
    require(feature.get("visual_dim") == 342, f"{rel(path)} visual dim mismatch", errors)
    require(feature.get("pinned") is True, f"{rel(path)} feature must be pinned", errors)
    require(protocol.get("seed") == 42, f"{rel(path)} seed mismatch", errors)
    require(protocol.get("epochs") == 30, f"{rel(path)} epochs mismatch", errors)
    require(
        protocol.get("validation_split_strategy") == "session_holdout",
        f"{rel(path)} validation strategy mismatch",
        errors,
    )
    require(
        protocol.get("validation_sessions") == ["Ses01", "Ses02", "Ses03", "Ses04"],
        f"{rel(path)} validation sessions mismatch",
        errors,
    )
    require(
        protocol.get("checkpoint_selection_metric") == "val_weighted_f1",
        f"{rel(path)} checkpoint metric mismatch",
        errors,
    )
    require(
        protocol.get("checkpoint_selection_split") == "validation",
        f"{rel(path)} checkpoint selection must be validation-only",
        errors,
    )
    require(protocol.get("checkpoint_name_for_test") == "best_model.pt", f"{rel(path)} test checkpoint mismatch", errors)
    require(protocol.get("test_session") == "Ses05", f"{rel(path)} test session mismatch", errors)
    require(protocol.get("test_split_used_for_selection") is False, f"{rel(path)} test participates in selection", errors)
    require(protocol.get("future_context_allowed") is False, f"{rel(path)} allows future context", errors)

    require(isinstance(runs, list), f"{rel(path)} runs must be a list", errors)
    if not isinstance(runs, list):
        return
    require(len(runs) == 8, f"{rel(path)} must list eight runs", errors)
    require([run.get("order") for run in runs] == list(range(1, 9)), f"{rel(path)} run order mismatch", errors)
    require(
        len({run.get("config_path") for run in runs}) == 8,
        f"{rel(path)} config paths must be unique",
        errors,
    )
    require(
        len({run.get("train_config_path") for run in runs}) == 8,
        f"{rel(path)} training config paths must be unique",
        errors,
    )
    model_counts = Counter(run.get("model") for run in runs)
    require(model_counts == Counter({"MMGCN": 4, "MultiDAGCL": 4}), f"{rel(path)} model counts mismatch", errors)
    expected_pairs = Counter(
        (model_name, session)
        for model_name in ("MMGCN", "MultiDAGCL")
        for session in ("Ses01", "Ses02", "Ses03", "Ses04")
    )
    actual_pairs = Counter((run.get("model"), run.get("validation_session")) for run in runs)
    require(actual_pairs == expected_pairs, f"{rel(path)} model/session coverage mismatch", errors)

    family_for_model = {"MMGCN": "mmgcn", "MultiDAGCL": "multidag_cl"}
    for run in runs:
        model_name = run.get("model")
        family = family_for_model.get(model_name)
        session = run.get("validation_session")
        if family is None or session not in {"Ses01", "Ses02", "Ses03", "Ses04"}:
            continue
        file_name = f"val_{session.lower()}.yaml"
        expected_pipeline_path = CLEAN_ROBERTA_V1_PIPELINE_DIRS[family] / file_name
        expected_train_path = CLEAN_ROBERTA_V1_TRAIN_DIRS[family] / file_name
        require(run.get("config_path") == rel(expected_pipeline_path), f"{rel(path)} run config path mismatch", errors)
        require(run.get("train_config_path") == rel(expected_train_path), f"{rel(path)} run training path mismatch", errors)
        require(expected_pipeline_path.exists(), f"{rel(path)} references missing {rel(expected_pipeline_path)}", errors)
        require(expected_train_path.exists(), f"{rel(path)} references missing {rel(expected_train_path)}", errors)
        require(run.get("seed") == 42, f"{rel(path)} run seed mismatch", errors)
        require(run.get("epochs") == 30, f"{rel(path)} run epochs mismatch", errors)
        require(run.get("feature_path") == CLEAN_ROBERTA_V1_PATH, f"{rel(path)} run feature path mismatch", errors)
        require(run.get("feature_sha256") == CLEAN_ROBERTA_V1_SHA256, f"{rel(path)} run feature SHA mismatch", errors)
        require(run.get("checkpoint_selection_metric") == "val_weighted_f1", f"{rel(path)} run selection metric mismatch", errors)
        require(run.get("test_session") == "Ses05", f"{rel(path)} run test session mismatch", errors)
        require(run.get("test_split_used_for_selection") is False, f"{rel(path)} run uses test for selection", errors)

    for cap_name in CLEAN_ROBERTA_V1_FORBIDDEN_CAPS:
        require(not values_for_key(manifest, cap_name), f"{rel(path)} contains forbidden {cap_name}", errors)
    lowered_values = [value.lower() for value in string_values(manifest)]
    require(
        not any("placeholder" in value or "to_be_filled" in value for value in lowered_values),
        f"{rel(path)} contains a placeholder",
        errors,
    )


def validate_clean_roberta_v1_formal_tree(errors: list[str]) -> None:
    expected_names = set(CLEAN_ROBERTA_V1_FORMAL_VARIANTS)
    for family in CLEAN_ROBERTA_V1_FORMAL_FAMILIES:
        train_dir = CLEAN_ROBERTA_V1_TRAIN_DIRS[family]
        pipeline_dir = CLEAN_ROBERTA_V1_PIPELINE_DIRS[family]
        pipeline_names = {path.name for path in pipeline_dir.glob("*.yaml")}
        require(
            pipeline_names == expected_names,
            f"{rel(pipeline_dir)} must contain exactly {sorted(expected_names)}; got {sorted(pipeline_names)}",
            errors,
        )
        for name in sorted(expected_names):
            train_path = train_dir / name
            pipeline_path = pipeline_dir / name
            require(train_path.exists(), f"missing {rel(train_path)}", errors)
            require(pipeline_path.exists(), f"missing {rel(pipeline_path)}", errors)
            if train_path.exists():
                validate_clean_roberta_v1_formal_training(
                    train_path, family, load_yaml(train_path), errors
                )
            if pipeline_path.exists():
                validate_clean_roberta_v1_formal_pipeline(
                    pipeline_path, family, load_yaml(pipeline_path), errors
                )
    validate_clean_roberta_v1_manifest(errors)


def validate_clean_roberta_v1_tree(errors: list[str]) -> None:
    if not IEMOCAP_FEATURE_REGISTRY.exists():
        errors.append(f"missing {rel(IEMOCAP_FEATURE_REGISTRY)}")
    else:
        validate_clean_roberta_v1_registry(load_yaml(IEMOCAP_FEATURE_REGISTRY), errors)

    for family, directory in CLEAN_ROBERTA_V1_TRAIN_DIRS.items():
        expected_names = {"smoke_real_2epoch.yaml"}
        if family in CLEAN_ROBERTA_V1_FORMAL_FAMILIES:
            expected_names.update(CLEAN_ROBERTA_V1_FORMAL_VARIANTS)
        names = {path.name for path in directory.glob("*.yaml")}
        require(
            names == expected_names,
            f"{rel(directory)} must contain exactly {sorted(expected_names)}; got {sorted(names)}",
            errors,
        )
        path = directory / "smoke_real_2epoch.yaml"
        if not path.exists():
            continue
        validate_clean_roberta_v1_smoke_config(path, family, load_yaml(path), errors)

    validate_clean_roberta_v1_formal_tree(errors)


def validate_original_merc_config(
    path: Path,
    family: str,
    feature_track: str,
    experiment_track: str,
    formal: bool,
    errors: list[str],
) -> None:
    config = load_yaml(path)
    model = config.get("model", {})
    dataset = config.get("dataset", {})
    training = config.get("training", {})
    require(model.get("name") == ORIGINAL_REPRO_MODELS[family], f"{rel(path)} model mismatch", errors)
    require(
        model.get("causal_grade") == "noncausal_offline_full_context",
        f"{rel(path)} causal grade mismatch",
        errors,
    )
    expected_strategy, expected_comparability = ORIGINAL_MERC_TRACKS[experiment_track]
    require(dataset.get("experiment_track") == experiment_track, f"{rel(path)} experiment track mismatch", errors)
    require(dataset.get("protocol_comparability") == expected_comparability, f"{rel(path)} comparability mismatch", errors)
    require(dataset.get("val_split_strategy") == expected_strategy, f"{rel(path)} split strategy mismatch", errors)
    require(dataset.get("outer_test_session") in {"Ses01", "Ses02", "Ses03", "Ses04", "Ses05"}, f"{rel(path)} outer fold missing", errors)
    if experiment_track == "legacy_official_split_safe_selection":
        require(dataset.get("outer_test_session") == "Ses05", f"{rel(path)} official-safe test must be Ses05", errors)
    require(0 < float(dataset.get("inner_val_ratio", 0)) < 1, f"{rel(path)} inner ratio invalid", errors)
    require(training.get("select_best_by") == "val_weighted_f1", f"{rel(path)} selects by test or wrong metric", errors)
    if family == "mmgcn":
        require(model.get("use_residual") is True, f"{rel(path)} disables MMGCN residual", errors)
    if family == "gsmcc":
        require(model.get("fidelity_status") == "PROJECT_VARIANT_NOT_PAPER_REPRODUCTION", f"{rel(path)} GS-MCC status mismatch", errors)
    if family == "dialoguegcn" and formal:
        require(model.get("base_model") == "LSTM", f"{rel(path)} DialogueGCN base model mismatch", errors)
        require(model.get("dropout") == 0.4, f"{rel(path)} DialogueGCN dropout mismatch", errors)
        require(training.get("batch_size") == 32, f"{rel(path)} DialogueGCN batch size mismatch", errors)
        require(config.get("optimizer", {}).get("learning_rate") == 0.0003, f"{rel(path)} DialogueGCN learning rate mismatch", errors)
        require(config.get("optimizer", {}).get("weight_decay") == 0.0, f"{rel(path)} DialogueGCN L2 mismatch", errors)
        require(model.get("use_class_weight") is True, f"{rel(path)} DialogueGCN class weight disabled", errors)
        require(model.get("use_nodal_attention") is True, f"{rel(path)} DialogueGCN nodal attention disabled", errors)
    if feature_track == "legacy":
        require(dataset.get("feature_set_name") == "legacy_mmgcn_textcnn", f"{rel(path)} legacy feature name mismatch", errors)
        require(dataset.get("feature_pkl_path") == IEMOCAP_FEATURE_PKL, f"{rel(path)} legacy feature path mismatch", errors)
        require(dataset.get("feature_sha256") == IEMOCAP_FEATURE_SHA256, f"{rel(path)} legacy SHA mismatch", errors)
        require(model.get("text_feature_dim") == 100, f"{rel(path)} legacy text dimension mismatch", errors)
        require(dataset.get("feature_protocol") == "legacy_mmgcn_features_v1", f"{rel(path)} legacy feature protocol mismatch", errors)
        require(dataset.get("feature_cleanliness") == "legacy_emotion_supervised_text_features", f"{rel(path)} legacy cleanliness mismatch", errors)
        expected_usage = (
            "reproduction_diagnostic"
            if experiment_track == "legacy_official_split_safe_selection"
            else "fair_legacy_feature_comparison"
        )
        require(dataset.get("usage") == expected_usage, f"{rel(path)} legacy usage mismatch", errors)
    else:
        require(dataset.get("feature_set_name") == CLEAN_ROBERTA_V1_NAME, f"{rel(path)} clean feature name mismatch", errors)
        require(dataset.get("feature_pkl_path") == CLEAN_ROBERTA_V1_PATH, f"{rel(path)} clean feature path mismatch", errors)
        require(dataset.get("feature_sha256") == CLEAN_ROBERTA_V1_SHA256, f"{rel(path)} clean SHA mismatch", errors)
        require(model.get("text_feature_dim") == 768, f"{rel(path)} clean text dimension mismatch", errors)
        require(dataset.get("feature_protocol") == "clean_roberta_v1", f"{rel(path)} clean feature protocol mismatch", errors)
        require(dataset.get("feature_cleanliness") == "frozen_self_supervised_utterance_only", f"{rel(path)} clean cleanliness mismatch", errors)
        require(dataset.get("usage") == "fair_main_experiment", f"{rel(path)} clean usage mismatch", errors)
    caps = [training.get("max_train_batches"), training.get("max_eval_batches")]
    if formal:
        require(all(value is None for value in caps), f"{rel(path)} formal config contains batch caps", errors)
    else:
        require(training.get("epochs") == 2, f"{rel(path)} smoke must use two epochs", errors)
        require(all(value == 1 for value in caps), f"{rel(path)} smoke caps must equal one", errors)


def validate_original_merc_tree(errors: list[str]) -> None:
    missing_smoke = [
        rel(path)
        for path in ORIGINAL_REPRO_SMOKE_CONFIGS.values()
        if not path.exists()
    ]
    require(
        not missing_smoke,
        f"original reproduction smoke config matrix missing {missing_smoke}",
        errors,
    )
    expected_legacy_smoke = {"gsmcc_legacy.yaml", "gsmcc_clean.yaml"}
    actual_legacy_smoke = {
        path.name for path in ORIGINAL_REPRO_LEGACY_SMOKE_DIR.glob("*.yaml")
    }
    require(
        actual_legacy_smoke == expected_legacy_smoke,
        "legacy original reproduction smoke directory must contain only "
        "unmigrated GS-MCC configs",
        errors,
    )
    for family in ORIGINAL_REPRO_MODELS:
        for track in ("legacy", "clean"):
            path = ORIGINAL_REPRO_SMOKE_CONFIGS[(family, track)]
            if path.exists():
                experiment_track = (
                    "legacy_official_split_safe_selection"
                    if track == "legacy"
                    else "clean_roberta_fivefold_fair_comparison"
                )
                validate_original_merc_config(
                    path, family, track, experiment_track, False, errors
                )

    screening_dir = ORIGINAL_MERC_EXPERIMENT_DIR / "screening"
    clean_screening_dir = ORIGINAL_MERC_EXPERIMENT_DIR / "clean_screening"
    legacy_fold_dir = ORIGINAL_MERC_EXPERIMENT_DIR / "legacy_fold_bases"
    clean_dir = ORIGINAL_MERC_EXPERIMENT_DIR / "clean_fold_bases"
    canonical_families = set(ORIGINAL_MERC_CANONICAL_FORMAL_CONFIGS)
    legacy_families = set(ORIGINAL_REPRO_MODELS) - canonical_families
    expected_screening = {
        f"{family}_legacy.yaml" for family in legacy_families
    }
    expected_clean = {f"{family}_clean.yaml" for family in legacy_families}
    require({path.name for path in screening_dir.glob("*.yaml")} == expected_screening, "original screening matrix mismatch", errors)
    require({path.name for path in clean_screening_dir.glob("*.yaml")} == expected_clean, "original clean screening matrix mismatch", errors)
    require({path.name for path in legacy_fold_dir.glob("*.yaml")} == expected_screening, "original legacy fold-base matrix mismatch", errors)
    require({path.name for path in clean_dir.glob("*.yaml")} == expected_clean, "original clean fold-base matrix mismatch", errors)
    for family in ORIGINAL_REPRO_MODELS:
        if family in ORIGINAL_MERC_CANONICAL_FORMAL_CONFIGS:
            canonical = ORIGINAL_MERC_CANONICAL_FORMAL_CONFIGS[family]
            screening = canonical["screening"]
            clean_screening = canonical[
                "clean_screening"
            ]
            legacy_fold = canonical[
                "legacy_fold_bases"
            ]
            clean = canonical["clean_fold_bases"]
        else:
            screening = screening_dir / f"{family}_legacy.yaml"
            clean_screening = clean_screening_dir / f"{family}_clean.yaml"
            legacy_fold = legacy_fold_dir / f"{family}_legacy.yaml"
            clean = clean_dir / f"{family}_clean.yaml"
        if screening.exists():
            validate_original_merc_config(screening, family, "legacy", "legacy_official_split_safe_selection", True, errors)
        if clean_screening.exists():
            validate_original_merc_config(clean_screening, family, "clean", "clean_roberta_fivefold_fair_comparison", True, errors)
        if legacy_fold.exists():
            validate_original_merc_config(legacy_fold, family, "legacy", "legacy_fivefold_fair_comparison", True, errors)
        if clean.exists():
            validate_original_merc_config(clean, family, "clean", "clean_roberta_fivefold_fair_comparison", True, errors)

    manifest_path = ORIGINAL_MERC_EXPERIMENT_DIR / "pipeline_manifest.yaml"
    require(manifest_path.exists(), "missing original MERC pipeline manifest", errors)
    if manifest_path.exists():
        manifest = load_yaml(manifest_path)
        stages = manifest.get("stages", {})
        require(len(stages.get("smoke", [])) == 8, "original manifest must list eight smoke runs", errors)
        require(len(stages.get("legacy_paper_adjacent_screening", [])) == 4, "original manifest must list four legacy paper-adjacent screening runs", errors)
        require(len(stages.get("clean_screening", [])) == 4, "original manifest must list four clean screening runs", errors)
        require(len(stages.get("legacy_folds", [])) == 4, "original manifest must list four legacy fold bases", errors)
        require(len(stages.get("clean_folds", [])) == 4, "original manifest must list four clean fold bases", errors)
        require(manifest.get("protocol", {}).get("test_split_used_for_selection") is False, "original manifest permits test selection", errors)
        require(manifest.get("top2_policy", {}).get("seeds") == [13, 42, 77], "original top2 seed template mismatch", errors)
        require(manifest.get("top2_policy", {}).get("selection_source") == "clean_single_fold_screening_validation_only", "original top2 source mismatch", errors)
        require(manifest.get("top2_policy", {}).get("test_metrics_allowed") is False, "original top2 permits test metrics", errors)


def main() -> None:
    errors: list[str] = []

    validate_causal_benchmark_tree(errors)
    validate_clean_roberta_v1_tree(errors)
    validate_original_merc_tree(errors)

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

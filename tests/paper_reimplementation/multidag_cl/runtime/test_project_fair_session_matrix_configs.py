from __future__ import annotations

from copy import deepcopy
import hashlib
from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[4]
CONFIG_ROOT = (
    ROOT
    / "configs/multidag_cl/paper_reimplementation/iemocap/full_context/clean_roberta_features"
)
BASE_CONFIG_SHA256 = "55075c9d7a8e09c04d7c4ab3502598c5859a71d58116b50c620bb6e0e291b7f7"
FEATURE_SHA256 = "c604c557bc3fbb129ca02b2acd57166b669a89ef76ff0cea1e14f9cb206324bf"
SESSION_CONFIGS = {
    "Ses01": CONFIG_ROOT / "project_fair.yaml",
    "Ses02": CONFIG_ROOT / "project_fair_val_ses02_seed100.yaml",
    "Ses03": CONFIG_ROOT / "project_fair_val_ses03_seed100.yaml",
    "Ses04": CONFIG_ROOT / "project_fair_val_ses04_seed100.yaml",
}


def _load(path: Path) -> dict:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _without_allowed_differences(config: dict) -> dict:
    normalized = deepcopy(config)
    del normalized["run_name"]
    del normalized["dataset"]["validation_session"]
    return normalized


@pytest.mark.parametrize("session", tuple(SESSION_CONFIGS))
def test_project_fair_session_config_exists_and_parses(session: str) -> None:
    path = SESSION_CONFIGS[session]
    assert path.is_file()
    config = _load(path)
    assert config["dataset"]["validation_session"] == session


def test_ses01_base_config_bytes_remain_frozen() -> None:
    digest = hashlib.sha256(SESSION_CONFIGS["Ses01"].read_bytes()).hexdigest()
    assert digest == BASE_CONFIG_SHA256


def test_project_fair_matrix_has_one_config_per_validation_session() -> None:
    configs = [_load(SESSION_CONFIGS[session]) for session in SESSION_CONFIGS]
    sessions = [config["dataset"]["validation_session"] for config in configs]
    assert sessions == ["Ses01", "Ses02", "Ses03", "Ses04"]
    assert len(sessions) == len(set(sessions)) == 4


@pytest.mark.parametrize("session", tuple(SESSION_CONFIGS))
def test_project_fair_matrix_fixed_identity_and_protocol(session: str) -> None:
    config = _load(SESSION_CONFIGS[session])
    assert config["registry_key"] == "multidag_cl_paper_reimplementation"
    assert config["dataset"]["test_session"] == "Ses05"
    assert config["dataset"]["feature_registry"] == "clean_roberta_v1"
    assert config["dataset"]["feature_sha256"] == FEATURE_SHA256
    assert (
        config["dataset"]["split_protocol"]
        == "clean_roberta_session_holdout_fair_comparison"
    )
    assert config["dataset"]["protocol_comparability"] == (
        "fair_comparison_not_paper_reproduction"
    )
    assert config["model_core"]["identity"]["conformance_profile"] == (
        "paper_formula_behavior"
    )
    assert config["model_core"]["training"]["seed"] == 100
    assert config["runtime"]["seed"] == 100
    assert config["model_core"]["training"]["epochs"] == 30
    assert config["runtime"]["epochs"] == 30
    assert config["runtime"]["mode"] == "train"
    assert config["output"]["experiment_group"] == (
        "multidag_cl_paper_reimplementation_project_fair"
    )
    assert config["model_core"]["training"]["gradient_clip_norm"] == 5.0
    assert config["runtime"]["gradient_clipping"] == {
        "mode": "global_norm",
        "max_norm": 5.0,
        "norm_type": 2.0,
        "error_if_nonfinite": True,
    }


def test_project_fair_matrix_only_changes_run_name_and_validation_session() -> None:
    normalized = {
        session: _without_allowed_differences(_load(path))
        for session, path in SESSION_CONFIGS.items()
    }
    expected = normalized["Ses01"]
    assert all(config == expected for config in normalized.values())


@pytest.mark.parametrize("session", tuple(SESSION_CONFIGS))
def test_project_fair_matrix_run_names_match_session_and_seed(session: str) -> None:
    config = _load(SESSION_CONFIGS[session])
    expected = (
        "multidag_cl_paper_reimplementation_project_fair_"
        f"val_{session.lower()}_seed100"
    )
    assert config["run_name"] == expected
    assert config["dataset"]["validation_session"] == session


@pytest.mark.parametrize("session", tuple(SESSION_CONFIGS))
def test_project_fair_matrix_keeps_test_out_of_selection(session: str) -> None:
    config = _load(SESSION_CONFIGS[session])
    assert config["model_core"]["checkpoint"]["test_split_used_for_selection"] is False
    assert config["checkpoint"]["test_split_used_for_selection"] is False
    assert config["checkpoint"]["test_evaluation_count"] == 1

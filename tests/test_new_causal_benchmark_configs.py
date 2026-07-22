from __future__ import annotations

import copy
from pathlib import Path

import yaml

from scripts.run_experiment_pipeline import choose_evaluate_script, choose_train_script


ROOT = Path(__file__).resolve().parents[1]
VARIANTS = {
    "official_prefix.yaml",
    "val_ses01.yaml",
    "val_ses02.yaml",
    "val_ses03.yaml",
    "val_ses04.yaml",
}


def _load(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def test_new_benchmark_and_pipeline_trees_are_complete_and_consistent() -> None:
    for family, model_name in (
        ("gsmcc", "causal_gsmcc_inspired"),
        ("dialoguegcn", "causal_dialoguegcn"),
    ):
        train_dir = ROOT / "configs" / "baselines" / family / "iemocap" / "causal_benchmark"
        pipeline_dir = ROOT / "configs" / "pipeline" / family / "iemocap" / "causal_benchmark"
        assert {path.name for path in train_dir.glob("*.yaml")} == VARIANTS
        assert {path.name for path in pipeline_dir.glob("*.yaml")} == VARIANTS
        folds = []
        for session in range(1, 5):
            config = _load(train_dir / f"val_ses0{session}.yaml")
            assert config["dataset"]["val_session_id"] == f"Ses0{session}"
            assert config["training"]["select_best_by"] == "val_weighted_f1"
            assert config["graph"]["window_future"] == 0
            assert not any(
                key in config["training"]
                for key in (
                    "train_batch_cap",
                    "val_batch_cap",
                    "test_batch_cap",
                    "max_train_batches",
                    "max_val_batches",
                    "max_test_batches",
                )
            )
            normalized = copy.deepcopy(config)
            normalized.pop("run_name")
            normalized["dataset"].pop("val_session_id")
            folds.append(normalized)
        assert all(fold == folds[0] for fold in folds[1:])
        for name in VARIANTS:
            pipeline = _load(pipeline_dir / name)
            assert pipeline["model"]["name"] == model_name
            target = ROOT / pipeline["train"]["train_config_path"]
            assert target == train_dir / name
            assert target.is_file()


def test_pipeline_uses_exact_new_model_routes() -> None:
    for name in ("causal_gsmcc_inspired", "causal_dialoguegcn"):
        assert choose_train_script(name) == "scripts/workflows/causal_graph/train.py"
        assert (
            choose_evaluate_script(name)
            == "scripts/workflows/causal_graph/evaluate.py"
        )

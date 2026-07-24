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
    for family, model_name, train_dir, train_name_by_pipeline_name in (
        (
            "gsmcc",
            "causal_gsmcc_inspired",
            ROOT
            / "configs"
            / "baselines"
            / "gsmcc"
            / "iemocap"
            / "causal_benchmark",
            {name: name for name in VARIANTS},
        ),
        (
            "dialoguegcn",
            "causal_dialoguegcn",
            ROOT
            / "configs"
            / "dialoguegcn"
            / "unified"
            / "iemocap"
            / "causal_context"
            / "legacy_mmgcn_features",
            {
                name: (
                    "val_official_prefix.yaml"
                    if name == "official_prefix.yaml"
                    else name
                )
                for name in VARIANTS
            },
        ),
    ):
        pipeline_dir = ROOT / "configs" / "pipeline" / family / "iemocap" / "causal_benchmark"
        expected_train_names = set(train_name_by_pipeline_name.values())
        if family == "dialoguegcn":
            expected_train_names.add("smoke_real_2epoch.yaml")
        assert {
            path.name for path in train_dir.glob("*.yaml")
        } == expected_train_names
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
            assert target == train_dir / train_name_by_pipeline_name[name]
            assert target.is_file()


def test_pipeline_uses_exact_new_model_routes() -> None:
    for name in ("causal_gsmcc_inspired", "causal_dialoguegcn"):
        assert choose_train_script(name) == "scripts/workflows/causal_graph/train.py"
        assert (
            choose_evaluate_script(name)
            == "scripts/workflows/causal_graph/evaluate.py"
        )

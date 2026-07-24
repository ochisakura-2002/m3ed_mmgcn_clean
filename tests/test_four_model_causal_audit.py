from __future__ import annotations

import copy
from pathlib import Path

import yaml

from scripts.analysis.causal.run_four_model_causal_audit import run_four_model_audit


ROOT = Path(__file__).resolve().parents[1]


def _load(relative: str) -> dict:
    with (ROOT / relative).open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def _write(path: Path, value: dict) -> None:
    with path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(value, file, sort_keys=False)


def test_four_model_synthetic_audit_passes_and_preserves_legacy_models(
    tmp_path: Path,
) -> None:
    configs = {
        "mmgcn": _load("configs/mmgcn/unified/iemocap/causal_context/legacy_mmgcn_features/val_official_prefix.yaml"),
        "multidag": _load("configs/multidag_cl/unified/iemocap/causal_context/legacy_mmgcn_features/val_official_prefix.yaml"),
        "gsmcc": _load("configs/gsmcc/project_variant/iemocap/causal_context/legacy_mmgcn_features/val_official_prefix.yaml"),
        "dialoguegcn": _load("configs/dialoguegcn/unified/iemocap/causal_context/legacy_mmgcn_features/val_official_prefix.yaml"),
    }
    configs["mmgcn"]["model"].update(
        {"text_feature_dim": 4, "audio_feature_dim": 3, "visual_feature_dim": 2, "hidden_dim": 8}
    )
    configs["mmgcn"]["graph"]["num_layers"] = 1
    configs["multidag"]["model"].update(
        {"text_feature_dim": 4, "audio_feature_dim": 3, "visual_feature_dim": 2, "hidden_dim": 8}
    )
    for key in ("gsmcc", "dialoguegcn"):
        configs[key]["dataset"].update(
            {"text_feature_dim": 4, "audio_feature_dim": 3, "visual_feature_dim": 2}
        )
        configs[key]["model"].update(
            {"text_dim": 4, "audio_dim": 3, "visual_dim": 2, "hidden_dim": 8}
        )
    configs["dialoguegcn"]["model"].update(
        {"context_hidden_dim": 8, "graph_hidden_dim": 8}
    )
    model_entries = {}
    for key, config in configs.items():
        path = tmp_path / f"{key}.yaml"
        _write(path, config)
        model_entries[key] = {"config": str(path)}
    runner = {
        "models": model_entries,
        "audit": {
            "mode": "synthetic",
            "device": "cpu",
            "batch_size": 2,
            "sequence_length": 5,
            "target_policy": "auto_multiple",
            "gradient_target": "history_squared_logit_sum",
            "perturbation_seeds": [42],
        },
        "output_dir": str(tmp_path / "audit"),
    }
    runner_path = tmp_path / "runner.yaml"
    _write(runner_path, runner)
    rows, passed = run_four_model_audit(runner_path)
    assert passed
    assert {row["model"] for row in rows} == {
        "mmgcn",
        "multidag",
        "gsmcc",
        "dialoguegcn",
    }
    assert all(row["strict_pass_1e6"] is True for row in rows)
    assert (tmp_path / "audit" / "four_model_causal_audit_summary.csv").is_file()

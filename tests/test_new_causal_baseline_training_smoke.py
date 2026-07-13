from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from scripts.baselines.train_new_causal_graph_baseline import (
    run_training,
    strictly_better_val_weighted_f1,
)


@pytest.mark.parametrize(
    "config_path",
    [
        Path("configs/smoke/train_causal_gsmcc_end_to_end.yaml"),
        Path("configs/smoke/train_causal_dialoguegcn_end_to_end.yaml"),
    ],
)
def test_synthetic_training_writes_complete_reloadable_run(
    config_path: Path, tmp_path: Path
) -> None:
    result = run_training(config_path, output_root_override=tmp_path / "runs")
    run_dir = Path(result["run_dir"])
    assert (run_dir / "checkpoints" / "best_model.pt").is_file()
    assert (run_dir / "checkpoints" / "last_model.pt").is_file()
    assert (run_dir / "logs" / "epoch_metrics.csv").is_file()
    assert (run_dir / "run_metadata.json").is_file()
    for split in ("val", "test"):
        output = run_dir / "logs" / "evaluations" / f"{split}_best_model"
        assert {path.name for path in output.iterdir()} == {
            "metrics.csv",
            "predictions.csv",
            "confusion_matrix.csv",
            "per_class_recall.csv",
        }
    assert result["best_epoch"] in {1, 2}
    if "gsmcc" in config_path.name:
        epoch_metrics = pd.read_csv(run_dir / "logs" / "epoch_metrics.csv")
        assert (epoch_metrics["train_consistency_loss"] == 0.0).all()
        assert (epoch_metrics["train_complementarity_loss"] == 0.0).all()


def test_best_checkpoint_comparison_is_strict() -> None:
    assert strictly_better_val_weighted_f1(0.6, 0.5)
    assert not strictly_better_val_weighted_f1(0.5, 0.5)

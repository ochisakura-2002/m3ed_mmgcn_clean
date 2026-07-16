from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pandas as pd
import yaml

from scripts.analyze.analyze_original_merc_results import (
    CLEAN_FIVEFOLD_TRACK,
    LEGACY_FIVEFOLD_TRACK,
    LEGACY_OFFICIAL_TRACK,
    aggregate,
    write_top2_selection,
)


MODELS = [
    "original_repro_mmgcn",
    "original_repro_multidag_cl",
    "project_paper_oriented_gsmcc",
    "original_repro_dialoguegcn",
]


def _validation_rows(tmp_path: Path) -> pd.DataFrame:
    rows = []
    for index, model in enumerate(MODELS):
        run_dir = tmp_path / model
        (run_dir / "logs").mkdir(parents=True)
        pd.DataFrame(
            {
                "epoch": [1, 2, 3],
                "train_loss": [1.0, 0.8, 0.7],
                "val_loss": [1.1, 0.9, 0.85],
                "val_weighted_f1": [0.3, 0.4, 0.45],
            }
        ).to_csv(run_dir / "logs" / "epoch_metrics.csv", index=False)
        rows.append(
            {
                "profile": "clean_screening",
                "experiment_track": CLEAN_FIVEFOLD_TRACK,
                "model_name": model,
                "weighted_f1": 0.80 - index * 0.05,
                "run_dir": str(run_dir),
                # A deliberately reversed Test-like column must be ignored.
                "test_weighted_f1": index,
            }
        )
    return pd.DataFrame(rows)


def test_top2_requires_all_clean_candidates_and_ignores_test_metrics() -> None:
    test_root = Path("tmp") / f"pytest_original_analysis_{uuid4().hex}"
    output_dir = test_root / "results"
    output_dir.mkdir(parents=True)
    rows = _validation_rows(test_root)

    write_top2_selection(output_dir, rows.iloc[:3])
    pending = yaml.safe_load((output_dir / "top2_selection.yaml").read_text(encoding="utf-8"))
    assert pending["status"] == "pending_all_four_clean_screening_results"
    assert pending["selected_models"] == []

    ranking = write_top2_selection(output_dir, rows)
    ready = yaml.safe_load((output_dir / "top2_selection.yaml").read_text(encoding="utf-8"))
    assert ready["status"] == "ready"
    assert ready["test_split_used_for_selection"] is False
    assert ready["selected_models"] == MODELS[:2]
    assert ranking.iloc[0]["model_name"] == MODELS[0]
    assert "test_weighted_f1" not in ranking.columns


def test_aggregate_keeps_protocol_tracks_separate() -> None:
    rows = []
    for track, score in (
        (LEGACY_OFFICIAL_TRACK, 0.61),
        (LEGACY_FIVEFOLD_TRACK, 0.62),
        (CLEAN_FIVEFOLD_TRACK, 0.70),
    ):
        rows.append(
            {
                "model_name": "original_repro_mmgcn",
                "experiment_track": track,
                "weighted_f1": score,
                "macro_f1": score,
                "uar": score,
                "accuracy": score,
                "paper_weighted_f1_points": 66.22,
                "paper_gap_weighted_f1_points": 0.0 if track == LEGACY_OFFICIAL_TRACK else float("nan"),
            }
        )
    summary = aggregate(pd.DataFrame(rows))
    assert set(summary["experiment_track"]) == {
        LEGACY_OFFICIAL_TRACK,
        LEGACY_FIVEFOLD_TRACK,
        CLEAN_FIVEFOLD_TRACK,
    }
    assert len(summary) == 3

#!/usr/bin/env python
"""
Batch evaluate one checkpoint under multiple test-time active modality settings.

Output layout:
    outputs/<YYYYMMDD>/runs/<run_id>/
      logs/
        missing_modalities/
          test_best_model/
            raw/
              TAV/
              TA/
              TV/
              AV/
              T/
              A/
              V/
            summary.csv
            metadata.json
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch


FULL_MODALITIES = ("text", "audio", "visual")
EVALUATE_SCRIPT_BY_MODEL = {
    "MMGCN": "scripts/evaluate_checkpoint.py",
    "MultiDAGCL": "scripts/baselines/evaluate_multidag_cl_checkpoint.py",
}

SETTINGS: List[Tuple[str, List[str]]] = [
    ("TAV", ["text", "audio", "visual"]),
    ("TA", ["text", "audio"]),
    ("TV", ["text", "visual"]),
    ("AV", ["audio", "visual"]),
    ("T", ["text"]),
    ("A", ["audio"]),
    ("V", ["visual"]),
    ("missing_text", ["audio", "visual"]),
    ("missing_audio", ["text", "visual"]),
    ("missing_visual", ["text", "audio"]),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate one checkpoint under multiple test-time active modality settings."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to a dated or legacy run checkpoint.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Dataset split to evaluate.",
    )
    parser.add_argument(
        "--settings",
        nargs="+",
        default=[name for name, _ in SETTINGS],
        choices=[name for name, _ in SETTINGS],
        help="Subset of modality settings to evaluate.",
    )
    parser.add_argument(
        "--output-subdir",
        type=str,
        default="missing_modalities",
        help="Subdirectory under logs/ to save missing-modality results.",
    )
    parser.add_argument(
        "--skip-if-not-full-train-modalities",
        action="store_true",
        help=(
            "Skip evaluation if checkpoint config was not trained with full "
            "text+audio+visual modalities."
        ),
    )
    return parser.parse_args()


def normalize_modalities(modalities: Optional[Sequence[str]]) -> List[str]:
    if modalities is None:
        return list(FULL_MODALITIES)

    valid = set(FULL_MODALITIES)
    normalized: List[str] = []

    for name in modalities:
        value = str(name).strip().lower()

        if value not in valid:
            raise ValueError(f"Unknown modality: {name}")

        if value not in normalized:
            normalized.append(value)

    return [name for name in FULL_MODALITIES if name in normalized]


def read_checkpoint_payload(checkpoint: Path) -> Dict[str, Any]:
    try:
        payload = torch.load(
            checkpoint,
            map_location="cpu",
            weights_only=False,
        )
    except TypeError:
        payload = torch.load(
            checkpoint,
            map_location="cpu",
        )
    if not isinstance(payload, dict):
        return {}
    return payload


def get_checkpoint_config(payload: Dict[str, Any]) -> Dict[str, Any]:
    config = payload.get("config", {})

    if not isinstance(config, dict):
        return {}

    return config


def get_checkpoint_model_name(payload: Dict[str, Any], config: Dict[str, Any]) -> str:
    model_name = payload.get("model_name", "")
    if model_name:
        return str(model_name)
    return str(config.get("model", {}).get("name", ""))


def get_trained_active_modalities(config: Dict[str, Any]) -> List[str]:
    model_config = config.get("model", {})

    if isinstance(model_config, dict) and "active_modalities" in model_config:
        return normalize_modalities(
            model_config.get("active_modalities", None)
        )

    modality_config = config.get("modality", {})

    if not isinstance(modality_config, dict):
        return list(FULL_MODALITIES)

    return normalize_modalities(
        modality_config.get("active_modalities", None)
    )


def read_metrics(metrics_path: Path) -> Dict[str, str]:
    with metrics_path.open("r", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    if len(rows) != 1:
        raise RuntimeError(
            f"Expected exactly one metric row in {metrics_path}, got {len(rows)}."
        )

    return rows[0]


def write_metadata(
    path: Path,
    checkpoint: Path,
    model_name: str,
    split: str,
    settings: List[str],
    trained_active_modalities: List[str],
    skipped: bool,
    reason: str = "",
) -> None:
    metadata = {
        "checkpoint": str(checkpoint),
        "model_name": model_name,
        "split": split,
        "settings": settings,
        "trained_active_modalities": trained_active_modalities,
        "skipped": skipped,
        "reason": reason,
    }

    with path.open("w", encoding="utf-8") as f:
        json.dump(
            metadata,
            f,
            ensure_ascii=False,
            indent=2,
        )


def main() -> None:
    args = parse_args()

    project_root = Path(__file__).resolve().parents[2]

    checkpoint = Path(args.checkpoint).resolve()

    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    if checkpoint.parent.name != "checkpoints":
        raise RuntimeError(
            "Expected checkpoint path like <run_dir>/checkpoints/best_model.pt. "
            f"Got: {checkpoint}"
        )

    run_dir = checkpoint.parent.parent
    checkpoint_stem = checkpoint.stem

    source_eval_dir = (
        run_dir
        / "logs"
        / "evaluations"
        / f"{args.split}_{checkpoint_stem}"
    )

    stage_dir = (
        run_dir
        / "logs"
        / str(args.output_subdir)
        / f"{args.split}_{checkpoint_stem}"
    )
    raw_root = stage_dir / "raw"

    stage_dir.mkdir(parents=True, exist_ok=True)
    raw_root.mkdir(parents=True, exist_ok=True)

    summary_path = stage_dir / "summary.csv"
    metadata_path = stage_dir / "metadata.json"

    checkpoint_payload = read_checkpoint_payload(checkpoint)
    checkpoint_config = get_checkpoint_config(checkpoint_payload)
    model_name = get_checkpoint_model_name(checkpoint_payload, checkpoint_config)
    evaluate_script_relative = EVALUATE_SCRIPT_BY_MODEL.get(model_name)
    if evaluate_script_relative is None:
        supported = ", ".join(sorted(EVALUATE_SCRIPT_BY_MODEL))
        raise ValueError(
            f"Unsupported checkpoint model for missing-modality evaluation: "
            f"{model_name!r}. Supported models: {supported}"
        )

    evaluate_script = project_root / evaluate_script_relative
    if not evaluate_script.exists():
        raise FileNotFoundError(f"Evaluation script not found: {evaluate_script}")

    trained_active_modalities = get_trained_active_modalities(checkpoint_config)

    trained_is_full = trained_active_modalities == list(FULL_MODALITIES)

    if args.skip_if_not_full_train_modalities and not trained_is_full:
        reason = (
            "Checkpoint was not trained with full text+audio+visual modalities. "
            f"trained_active_modalities={trained_active_modalities}"
        )

        print("[Skip]", reason)

        write_metadata(
            path=metadata_path,
            checkpoint=checkpoint,
            model_name=model_name,
            split=args.split,
            settings=list(args.settings),
            trained_active_modalities=trained_active_modalities,
            skipped=True,
            reason=reason,
        )

        return

    setting_map = dict(SETTINGS)
    summary_rows: List[Dict[str, str]] = []

    for setting_name in args.settings:
        modalities = setting_map[setting_name]

        print("=" * 100)
        print(f"Evaluating setting={setting_name}, active_modalities={modalities}")
        print("=" * 100)

        cmd = [
            sys.executable,
            str(evaluate_script),
            "--checkpoint",
            str(checkpoint),
            "--split",
            args.split,
            "--active-modalities",
            *modalities,
        ]

        subprocess.run(
            cmd,
            cwd=str(project_root),
            check=True,
        )

        if not source_eval_dir.exists():
            raise RuntimeError(f"Evaluation output dir not found: {source_eval_dir}")

        dest_dir = raw_root / setting_name

        if dest_dir.exists():
            shutil.rmtree(dest_dir)

        shutil.copytree(
            source_eval_dir,
            dest_dir,
        )

        metrics = read_metrics(dest_dir / "metrics.csv")
        metrics = {
            **metrics,
            "setting": setting_name,
            "active_modalities": "+".join(modalities),
            "trained_active_modalities": "+".join(trained_active_modalities),
            "output_dir": str(dest_dir),
        }
        summary_rows.append(metrics)

    fieldnames = [
        "setting",
        "active_modalities",
        "trained_active_modalities",
        "split",
        "checkpoint",
        "loss",
        "accuracy",
        "acc",
        "uar",
        "macro_f1",
        "weighted_f1",
        "output_dir",
    ]

    with summary_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    write_metadata(
        path=metadata_path,
        checkpoint=checkpoint,
        model_name=model_name,
        split=args.split,
        settings=list(args.settings),
        trained_active_modalities=trained_active_modalities,
        skipped=False,
    )

    print("=" * 100)
    print(f"Saved raw outputs to: {raw_root}")
    print(f"Saved summary to: {summary_path}")
    print(f"Saved metadata to: {metadata_path}")
    print("=" * 100)

    with summary_path.open("r", encoding="utf-8") as f:
        print(f.read())


if __name__ == "__main__":
    main()

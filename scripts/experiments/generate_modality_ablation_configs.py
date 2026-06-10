#!/usr/bin/env python
"""
Generate training-time modality ablation YAML configs.

This script copies one base training config and creates seven configs:
TAV, TA, TV, AV, T, A, V.
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml


SETTINGS: List[Tuple[str, List[str]]] = [
    ("TAV", ["text", "audio", "visual"]),
    ("TA", ["text", "audio"]),
    ("TV", ["text", "visual"]),
    ("AV", ["audio", "visual"]),
    ("T", ["text"]),
    ("A", ["audio"]),
    ("V", ["visual"]),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate modality-ablation training YAML configs."
    )
    parser.add_argument("--base-config", required=True, type=str)
    parser.add_argument("--output-dir", required=True, type=str)
    parser.add_argument("--dataset-tag", required=True, type=str)
    parser.add_argument("--model-tag", default="mmgcn", type=str)
    parser.add_argument("--experiment-prefix", required=True, type=str)
    return parser.parse_args()


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if not isinstance(config, dict):
        raise RuntimeError(f"Expected YAML dict: {path}")

    return config


def main() -> None:
    args = parse_args()

    base_path = Path(args.base_config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    base_config = load_yaml(base_path)
    generated_paths: List[Path] = []

    for setting_name, modalities in SETTINGS:
        config = copy.deepcopy(base_config)

        config.setdefault("project", {})
        config["project"]["experiment_name"] = (
            f"{args.experiment_prefix}_{setting_name}"
        )

        config["modality"] = {
            "active_modalities": modalities,
        }

        config.setdefault("notes", {})
        config["notes"]["modality_ablation"] = {
            "setting": setting_name,
            "active_modalities": modalities,
            "base_config": str(base_path),
            "description": "Training-time modality ablation config.",
        }

        output_path = (
            output_dir
            / f"train_{args.model_tag}_{args.dataset_tag}_{setting_name}.yaml"
        )

        with output_path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(
                config,
                f,
                allow_unicode=True,
                sort_keys=False,
            )

        generated_paths.append(output_path)

    print("=" * 100)
    print("Generated modality-ablation configs")
    print("=" * 100)
    for path in generated_paths:
        print(path)


if __name__ == "__main__":
    main()

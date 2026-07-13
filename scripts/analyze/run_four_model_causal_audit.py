"""Run the unified causal audit for MMGCN, MultiDAG, GS-MCC, and DialogueGCN."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple

import yaml


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze.audit_model_causality import run_audit  # noqa: E402


SUMMARY_FIELDS = [
    "model",
    "mode",
    "cutoff_count",
    "max_future_zero_diff",
    "max_future_noise_diff",
    "max_future_shuffle_diff",
    "max_joint_future_diff",
    "max_prefix_full_diff",
    "max_future_text_grad",
    "max_future_audio_grad",
    "max_future_visual_grad",
    "future_edge_violations",
    "padding_edge_violations",
    "cross_dialogue_edge_violations",
    "strict_pass_1e6",
    "strict_pass_1e5",
    "feature_causality_status",
    "error",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


def resolve_path(value: str) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else ROOT / path


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        value = yaml.safe_load(file)
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain a YAML mapping.")
    return value


def summary_row(model_key: str, summary: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "model": model_key,
        "mode": summary["mode"],
        "cutoff_count": summary["cutoff_count"],
        "max_future_zero_diff": summary["future_zero_max_abs_diff"],
        "max_future_noise_diff": summary["future_random_noise_max_abs_diff"],
        "max_future_shuffle_diff": summary["future_cross_sample_shuffle_max_abs_diff"],
        "max_joint_future_diff": summary["joint_future_max_abs_diff"],
        "max_prefix_full_diff": summary["prefix_full_max_abs_diff"],
        "max_future_text_grad": summary["max_future_text_grad"],
        "max_future_audio_grad": summary["max_future_audio_grad"],
        "max_future_visual_grad": summary["max_future_visual_grad"],
        "future_edge_violations": summary["future_adjacency_violations"],
        "padding_edge_violations": summary["padding_edge_violations"],
        "cross_dialogue_edge_violations": summary["cross_dialogue_edge_violations"],
        "strict_pass_1e6": summary["strict_pass_at_1e6"],
        "strict_pass_1e5": summary["strict_pass_at_1e5"],
        "feature_causality_status": summary["feature_causality_status"],
        "error": "",
    }


def write_outputs(
    output_dir: Path,
    rows: List[Dict[str, Any]],
    sources: List[str],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "four_model_causal_audit_summary.csv").open(
        "w", encoding="utf-8", newline=""
    ) as file:
        writer = csv.DictWriter(file, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    with (output_dir / "source_configs.txt").open("w", encoding="utf-8") as file:
        file.write("\n".join(sources) + "\n")
    lines = [
        "# Four-model causal audit",
        "",
        "| Model | Mode | Cutoffs | Strict 1e-6 | Strict 1e-5 | Error |",
        "|---|---|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['model']} | {row['mode']} | {row['cutoff_count']} | "
            f"{row['strict_pass_1e6']} | {row['strict_pass_1e5']} | {row['error']} |"
        )
    lines.extend(
        [
            "",
            "Upstream feature extraction causality remains unverified; this is a model-level audit.",
            "",
        ]
    )
    with (output_dir / "four_model_causal_audit_report.md").open(
        "w", encoding="utf-8"
    ) as file:
        file.write("\n".join(lines))


def run_four_model_audit(config_path: Path) -> Tuple[List[Dict[str, Any]], bool]:
    config = load_yaml(resolve_path(str(config_path)))
    models = config.get("models", {})
    if not isinstance(models, dict) or not models:
        raise ValueError("models must be a non-empty mapping.")
    audit = config.get("audit", {})
    output_dir = resolve_path(
        str(config.get("output_dir", "outputs/analysis/four_model_causal_audit"))
    )
    rows: List[Dict[str, Any]] = []
    sources: List[str] = []
    for model_key, entry in models.items():
        model_output = output_dir / str(model_key)
        config_text = str(entry.get("config", "")) if isinstance(entry, dict) else ""
        source = resolve_path(config_text)
        sources.append(f"{model_key}={source.relative_to(ROOT).as_posix() if source.is_relative_to(ROOT) else source}")
        try:
            summary = run_audit(
                config_path=source,
                output_dir=model_output,
                mode=str(audit.get("mode", "real_batch")),
                device_name=str(audit.get("device", "cpu")),
                seed=int(audit.get("seed", 2026)),
                batch_size=int(audit.get("batch_size", 2)),
                sequence_length=int(audit.get("sequence_length", 5)),
                target_index=int(audit.get("target_index", 2)),
                class_id=int(audit.get("class_id", 0)),
                tolerance=float(audit.get("tolerance", 1.0e-6)),
                real_split=str(audit.get("real_split", "val")),
                target_policy=str(audit.get("target_policy", "auto_multiple")),
                target_indices=audit.get("target_indices"),
                gradient_target=str(
                    audit.get("gradient_target", "history_squared_logit_sum")
                ),
                perturbation_seeds=audit.get("perturbation_seeds", [42, 43, 44]),
            )
            rows.append(summary_row(str(model_key), summary))
        except Exception as error:  # Continue so one failure does not erase other audits.
            rows.append(
                {
                    "model": str(model_key),
                    "mode": str(audit.get("mode", "real_batch")),
                    "cutoff_count": 0,
                    "max_future_zero_diff": "",
                    "max_future_noise_diff": "",
                    "max_future_shuffle_diff": "",
                    "max_joint_future_diff": "",
                    "max_prefix_full_diff": "",
                    "max_future_text_grad": "",
                    "max_future_audio_grad": "",
                    "max_future_visual_grad": "",
                    "future_edge_violations": "",
                    "padding_edge_violations": "",
                    "cross_dialogue_edge_violations": "",
                    "strict_pass_1e6": False,
                    "strict_pass_1e5": False,
                    "feature_causality_status": "UNCONFIRMED",
                    "error": f"{type(error).__name__}: {error}",
                }
            )
    write_outputs(output_dir, rows, sources)
    passed = len(rows) == 4 and all(row["strict_pass_1e6"] is True for row in rows)
    return rows, passed


def main() -> None:
    args = parse_args()
    rows, passed = run_four_model_audit(Path(args.config))
    for row in rows:
        print(
            f"model={row['model']} strict_pass_1e6={row['strict_pass_1e6']} "
            f"error={row['error']}"
        )
    if args.strict and not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

"""
Diagnose IEMOCAP train/val/test split size and label distribution.

This script is read-only with respect to the dataset. It uses the existing
IEMOCAP official feature adapter to report split statistics and figures that
help interpret validation-test gaps.

Usage:
    python scripts/analyze/diagnose_iemocap_splits.py \
      --config configs/baselines/multidag_cl/iemocap/formal/context_w5_tav.yaml \
      --output-dir outputs/paper_artifacts/split_diagnostics/iemocap
"""

from __future__ import annotations

import argparse
from collections import Counter
from math import ceil
from pathlib import Path
import sys
from typing import Any, Dict, Iterable, List

import pandas as pd
import yaml

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SPLITS = ["train", "val", "test"]
DEFAULT_LABEL_LIST = [
    "Happy",
    "Sad",
    "Neutral",
    "Angry",
    "Excited",
    "Frustrated",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export IEMOCAP split diagnostics tables and figures."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="MultiDAG+CL IEMOCAP training YAML.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/paper_artifacts/split_diagnostics/iemocap",
        help="Directory for diagnostics tables, figures, and notes.",
    )
    return parser.parse_args()


def resolve_path(path_text: str) -> Path:
    path = Path(str(path_text))
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)
    if data is None:
        raise RuntimeError(f"Empty config: {path}")
    if not isinstance(data, dict):
        raise TypeError(f"Config must be a dict: {path}")
    return data


def safe_get(config: Dict[str, Any], keys: Iterable[str], default: Any = "") -> Any:
    current: Any = config
    for key in keys:
        if not isinstance(current, dict):
            return default
        if key not in current:
            return default
        current = current[key]
    if current is None:
        return default
    return current


def get_training_config(config: Dict[str, Any]) -> Dict[str, Any]:
    training = dict(config.get("training", {}))
    legacy_train = dict(config.get("train", {}))
    if "batch_size" not in training and "batch_size" in legacy_train:
        training["batch_size"] = legacy_train["batch_size"]
    training.setdefault("batch_size", 1)
    return training


def get_label_list(config: Dict[str, Any]) -> List[str]:
    num_classes = int(safe_get(config, ["dataset", "num_classes"], 0))
    labels = safe_get(config, ["dataset", "label_list"], None)
    if labels is None:
        labels = DEFAULT_LABEL_LIST
    labels = [str(label) for label in labels]
    if num_classes and len(labels) != num_classes:
        raise ValueError(
            f"label_list length {len(labels)} != dataset.num_classes {num_classes}"
        )
    return labels


def build_dataset(config: Dict[str, Any], split: str) -> IEMOCAPOfficialFeatureDataset:
    from datasets.iemocap.official_feature_dataset import (  # noqa: WPS433
        IEMOCAPOfficialFeatureDataset,
    )

    dataset_config = config["dataset"]
    return IEMOCAPOfficialFeatureDataset(
        feature_pkl_path=resolve_path(str(dataset_config["feature_pkl_path"])),
        split=split,
        valid_ratio=float(dataset_config.get("valid_ratio", 0.1)),
        val_split_strategy=str(dataset_config.get("val_split_strategy", "official_prefix")),
        seed=int(safe_get(config, ["system", "seed"], 42)),
    )


def collect_split_info(
    config: Dict[str, Any],
    split: str,
    batch_size: int,
    label_list: List[str],
) -> tuple[Dict[str, Any], List[Dict[str, Any]], List[int]]:
    dataset = build_dataset(config, split)

    lengths = [int(len(dataset.videoLabels[dialogue_id])) for dialogue_id in dataset.keys]
    labels: List[int] = []
    for dialogue_id in dataset.keys:
        labels.extend(int(label) for label in dataset.videoLabels[dialogue_id])

    num_dialogues = len(lengths)
    num_utterances = int(sum(lengths))
    num_classes = len(label_list)

    stats_row = {
        "split": split,
        "num_dialogues": int(num_dialogues),
        "num_utterances": int(num_utterances),
        "avg_utterances_per_dialogue": (
            float(num_utterances) / float(num_dialogues) if num_dialogues else 0.0
        ),
        "min_utterances_per_dialogue": int(min(lengths)) if lengths else 0,
        "max_utterances_per_dialogue": int(max(lengths)) if lengths else 0,
        "batch_size": int(batch_size),
        "estimated_num_batches": int(ceil(num_dialogues / float(batch_size))) if batch_size else 0,
        "num_classes": int(num_classes),
    }

    counts = Counter(labels)
    label_rows: List[Dict[str, Any]] = []
    for class_id, class_name in enumerate(label_list):
        count = int(counts.get(class_id, 0))
        percentage = count / float(num_utterances) if num_utterances else 0.0
        label_rows.append(
            {
                "split": split,
                "class_id": int(class_id),
                "class_name": class_name,
                "count": count,
                "percentage": float(percentage),
            }
        )

    return stats_row, label_rows, lengths


def format_cell(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    columns = [str(column) for column in df.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(format_cell(row[column]) for column in df.columns) + " |")
    return "\n".join(lines) + "\n"


def save_table_bundle(df: pd.DataFrame, output_dir: Path, stem: str) -> None:
    csv_path = output_dir / f"{stem}.csv"
    md_path = output_dir / f"{stem}.md"
    tex_path = output_dir / f"{stem}.tex"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    md_path.write_text(dataframe_to_markdown(df), encoding="utf-8")
    tex_path.write_text(df.to_latex(index=False), encoding="utf-8")
    print(f"  [Table] {stem}: {csv_path}")


def save_label_distribution_plot(label_df: pd.DataFrame, output_dir: Path) -> None:
    pivot_df = label_df.pivot(
        index="class_name",
        columns="split",
        values="percentage",
    )
    pivot_df = pivot_df.reindex(columns=[split for split in SPLITS if split in pivot_df.columns])

    ax = pivot_df.plot(kind="bar", figsize=(10, 5))
    ax.set_ylabel("Percentage of Utterances")
    ax.set_xlabel("Emotion Class")
    ax.set_title("IEMOCAP Label Distribution by Split")
    ax.set_ylim(0, max(1.0, float(pivot_df.max().max()) * 1.15 if len(pivot_df) else 1.0))
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()

    png_path = output_dir / "label_distribution_by_split.png"
    pdf_path = output_dir / "label_distribution_by_split.pdf"
    plt.savefig(png_path, dpi=300)
    plt.savefig(pdf_path)
    plt.close()
    print(f"  [Figure] label distribution: {png_path}")


def save_utterance_count_plot(stats_df: pd.DataFrame, output_dir: Path) -> None:
    plt.figure(figsize=(7, 5))
    plt.bar(stats_df["split"], stats_df["num_utterances"])
    plt.xlabel("Split")
    plt.ylabel("Number of Utterances")
    plt.title("IEMOCAP Utterance Count by Split")
    plt.tight_layout()

    png_path = output_dir / "utterance_count_by_split.png"
    pdf_path = output_dir / "utterance_count_by_split.pdf"
    plt.savefig(png_path, dpi=300)
    plt.savefig(pdf_path)
    plt.close()
    print(f"  [Figure] utterance counts: {png_path}")


def save_dialogue_length_distribution_plot(
    lengths_by_split: Dict[str, List[int]],
    output_dir: Path,
) -> None:
    labels = [split for split in SPLITS if split in lengths_by_split]
    values = [lengths_by_split[split] for split in labels]

    plt.figure(figsize=(8, 5))
    plt.boxplot(values, labels=labels, showmeans=True)
    plt.xlabel("Split")
    plt.ylabel("Utterances per Dialogue")
    plt.title("IEMOCAP Dialogue Length Distribution by Split")
    plt.tight_layout()

    png_path = output_dir / "dialogue_length_distribution.png"
    pdf_path = output_dir / "dialogue_length_distribution.pdf"
    plt.savefig(png_path, dpi=300)
    plt.savefig(pdf_path)
    plt.close()
    print(f"  [Figure] dialogue length distribution: {png_path}")


def row_value(stats_df: pd.DataFrame, split: str, column: str) -> Any:
    row_df = stats_df[stats_df["split"] == split]
    if len(row_df) == 0:
        return ""
    return row_df.iloc[0][column]


def save_notes(stats_df: pd.DataFrame, output_dir: Path) -> None:
    train_dialogues = row_value(stats_df, "train", "num_dialogues")
    val_dialogues = row_value(stats_df, "val", "num_dialogues")
    test_dialogues = row_value(stats_df, "test", "num_dialogues")
    train_batches = row_value(stats_df, "train", "estimated_num_batches")
    test_batches = row_value(stats_df, "test", "estimated_num_batches")

    notes = f"""# IEMOCAP Split Diagnostics Notes

## Summary

- Train/validation/test dialogue counts: train={train_dialogues}, val={val_dialogues}, test={test_dialogues}.
- Estimated batch counts with the configured batch size: train={train_batches}, test={test_batches}.
- It is normal for train to have many more batches than test because batching is done at dialogue level and the test split has fewer dialogues.

## Split Semantics

- The validation split is derived from the official training dialogues using the configured validation strategy.
- The test split comes from the official test dialogues.
- Validation curves are process metrics used for checkpoint selection.
- Final test metrics are computed once for the validation-best checkpoint.

## Interpreting A High Validation / Moderate Test Gap

- A high validation score does not guarantee a high test score, especially when validation has few dialogues.
- The gap may come from a small validation split, different label distributions, dialogue length differences, or a harder official test split.
- Do not use test metrics to select the best epoch or checkpoint.
- Recommended checks: multiple seeds, label distribution by split, per-class recall, and confusion matrices.
"""

    notes_path = output_dir / "split_diagnostics_notes.md"
    notes_path.write_text(notes, encoding="utf-8")
    print(f"  [Notes] {notes_path}")


def main() -> None:
    args = parse_args()
    config_path = resolve_path(args.config)
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = load_yaml(config_path)
    training = get_training_config(config)
    batch_size = int(training["batch_size"])
    label_list = get_label_list(config)

    print("=" * 100)
    print("Diagnose IEMOCAP splits")
    print("=" * 100)
    print("Project root:", PROJECT_ROOT)
    print("Config:", config_path)
    print("Output dir:", output_dir)
    print("Batch size:", batch_size)
    print("=" * 100)

    stats_rows: List[Dict[str, Any]] = []
    label_rows: List[Dict[str, Any]] = []
    lengths_by_split: Dict[str, List[int]] = {}

    for split in SPLITS:
        stats_row, split_label_rows, lengths = collect_split_info(
            config=config,
            split=split,
            batch_size=batch_size,
            label_list=label_list,
        )
        stats_rows.append(stats_row)
        label_rows.extend(split_label_rows)
        lengths_by_split[split] = lengths
        print(
            f"[OK] {split}: dialogues={stats_row['num_dialogues']} "
            f"utterances={stats_row['num_utterances']} "
            f"estimated_batches={stats_row['estimated_num_batches']}"
        )

    stats_df = pd.DataFrame(stats_rows)
    label_df = pd.DataFrame(label_rows)

    save_table_bundle(stats_df, output_dir, "split_statistics")
    save_table_bundle(label_df, output_dir, "label_distribution_by_split")
    save_label_distribution_plot(label_df, output_dir)
    save_utterance_count_plot(stats_df, output_dir)
    save_dialogue_length_distribution_plot(lengths_by_split, output_dir)
    save_notes(stats_df, output_dir)

    print("=" * 100)
    print("Finished split diagnostics.")
    print("=" * 100)


if __name__ == "__main__":
    main()

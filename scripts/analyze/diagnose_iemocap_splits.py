"""
Diagnose IEMOCAP train/val/test split size and label distribution.

This script is read-only with respect to the dataset. It uses the existing
IEMOCAP official feature adapter to report split statistics and figures that
help interpret validation-test gaps.

Usage:
    python scripts/analyze/diagnose_iemocap_splits.py \
      --config configs/baselines/multidag_cl/iemocap/formal/context_w5_tav.yaml \
      --experiment-date 20260716
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from math import ceil, sqrt
from pathlib import Path
import re
import sys
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, TYPE_CHECKING

import pandas as pd
import yaml

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

if TYPE_CHECKING:
    from datasets.iemocap.official_feature_dataset import IEMOCAPOfficialFeatureDataset


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from utils.output_paths import (  # noqa: E402
    configured_output_root,
    resolve_experiment_date,
    resolve_output_category,
)

from datasets.iemocap import parse_iemocap_session_id  # noqa: E402

SPLITS = ["train", "val", "test"]
SPLIT_PAIRS = [("train", "val"), ("train", "test"), ("val", "test")]
DEFAULT_LABEL_LIST = [
    "Happy",
    "Sad",
    "Neutral",
    "Angry",
    "Excited",
    "Frustrated",
]
UTTERANCE_SPEAKER_PATTERN = re.compile(r"_([FM])\d{3}$", re.IGNORECASE)
DIALOGUE_MANIFEST_COLUMNS = [
    "split",
    "dialogue_id",
    "session_id",
    "speaker_ids",
    "num_speakers",
    "num_utterances",
    "majority_class_id",
    "majority_class_name",
    "majority_class_ratio",
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
        default=None,
        help="Directory for diagnostics tables, figures, and notes.",
    )
    parser.add_argument("--experiment-date", default=None)
    return parser.parse_args()


def resolve_path(path_text: str) -> Path:
    path = Path(str(path_text))
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def project_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path)


def decode_text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("latin1")
    return str(value)


def parse_session_id(dialogue_id: Any) -> Optional[str]:
    """Use the shared strict parser while retaining diagnostic issue reporting."""
    try:
        return parse_iemocap_session_id(decode_text(dialogue_id))
    except ValueError:
        return None


def parse_speaker_ids(
    dialogue_id: Any,
    session_id: Optional[str],
    raw_speakers: Sequence[Any],
    utterance_ids: Sequence[Any],
) -> Tuple[List[str], Counter[str], str]:
    """Return session-qualified speakers after cross-checking utterance IDs."""
    if session_id is None:
        return [], Counter(), "session_id_unparsed"
    if len(raw_speakers) != len(utterance_ids):
        return [], Counter(), "speaker_and_utterance_length_mismatch"
    if not raw_speakers:
        return [], Counter(), "no_speaker_observations"

    dialogue_text = decode_text(dialogue_id)
    speaker_ids: List[str] = []
    for raw_speaker, utterance_id in zip(raw_speakers, utterance_ids):
        speaker_code = decode_text(raw_speaker).strip().upper()
        utterance_text = decode_text(utterance_id)
        utterance_match = UTTERANCE_SPEAKER_PATTERN.search(utterance_text)
        if speaker_code not in {"F", "M"}:
            return [], Counter(), "unsupported_video_speaker_value"
        if (
            utterance_match is None
            or utterance_text[: utterance_match.start()] != dialogue_text
        ):
            return [], Counter(), "utterance_id_speaker_unparsed"
        if utterance_match.group(1).upper() != speaker_code:
            return [], Counter(), "video_speaker_utterance_id_mismatch"
        speaker_ids.append(f"{session_id}{speaker_code}")

    counts = Counter(speaker_ids)
    return sorted(counts), counts, ""


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
    dataset_config = config["dataset"]
    feature_pkl_path = resolve_path(str(dataset_config["feature_pkl_path"]))
    if not feature_pkl_path.exists():
        raise FileNotFoundError(f"IEMOCAP feature pkl not found: {feature_pkl_path}")

    from datasets.iemocap.official_feature_dataset import (  # noqa: WPS433
        IEMOCAPOfficialFeatureDataset,
    )

    return IEMOCAPOfficialFeatureDataset(
        feature_pkl_path=feature_pkl_path,
        split=split,
        valid_ratio=float(dataset_config.get("valid_ratio", 0.1)),
        val_split_strategy=str(
            dataset_config.get("val_split_strategy", "official_prefix")
        ),
        val_session_id=dataset_config.get("val_session_id"),
        seed=int(safe_get(config, ["system", "seed"], 42)),
    )


def collect_split_info(
    config: Dict[str, Any],
    split: str,
    batch_size: int,
    label_list: List[str],
) -> tuple[
    Dict[str, Any],
    List[Dict[str, Any]],
    List[int],
    List[Dict[str, Any]],
    Dict[str, Any],
]:
    dataset = build_dataset(config, split)

    lengths: List[int] = []
    labels: List[int] = []
    dialogue_rows: List[Dict[str, Any]] = []
    session_issue_counts: Counter[str] = Counter()
    speaker_issue_counts: Counter[str] = Counter()

    for dialogue_id in dataset.keys:
        dialogue_labels = [int(label) for label in dataset.videoLabels[dialogue_id]]
        raw_speakers = list(dataset.videoSpeakers[dialogue_id])
        utterance_ids = list(dataset.videoIDs[dialogue_id])
        num_utterances = len(dialogue_labels)
        lengths.append(int(num_utterances))
        labels.extend(dialogue_labels)

        session_id = parse_session_id(dialogue_id)
        if session_id is None:
            session_issue_counts["dialogue_id_session_unparsed"] += 1

        if len(raw_speakers) != num_utterances or len(utterance_ids) != num_utterances:
            speaker_ids = []
            speaker_utterance_counts: Counter[str] = Counter()
            speaker_issue = "label_speaker_utterance_length_mismatch"
        else:
            speaker_ids, speaker_utterance_counts, speaker_issue = parse_speaker_ids(
                dialogue_id=dialogue_id,
                session_id=session_id,
                raw_speakers=raw_speakers,
                utterance_ids=utterance_ids,
            )
        if speaker_issue:
            speaker_issue_counts[speaker_issue] += 1

        dialogue_label_counts = Counter(dialogue_labels)
        if dialogue_label_counts:
            majority_count = max(dialogue_label_counts.values())
            majority_class_id: Optional[int] = min(
                class_id
                for class_id, count in dialogue_label_counts.items()
                if count == majority_count
            )
            majority_class_name = (
                label_list[majority_class_id]
                if 0 <= majority_class_id < len(label_list)
                else ""
            )
            majority_class_ratio: Optional[float] = float(majority_count) / float(
                num_utterances
            )
        else:
            majority_class_id = None
            majority_class_name = ""
            majority_class_ratio = None

        dialogue_rows.append(
            {
                "split": split,
                "dialogue_id": decode_text(dialogue_id),
                "session_id": session_id,
                "speaker_ids": speaker_ids,
                "num_speakers": len(speaker_ids) if not speaker_issue else None,
                "num_utterances": int(num_utterances),
                "majority_class_id": majority_class_id,
                "majority_class_name": majority_class_name,
                "majority_class_ratio": majority_class_ratio,
                "_speaker_utterance_counts": dict(speaker_utterance_counts),
            }
        )

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
        "estimated_num_batches": (
            int(ceil(num_dialogues / float(batch_size))) if batch_size else 0
        ),
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

    parse_summary = {
        "num_dialogues": int(num_dialogues),
        "session_ids_parsed": int(num_dialogues - sum(session_issue_counts.values())),
        "speaker_ids_parsed": int(num_dialogues - sum(speaker_issue_counts.values())),
        "session_issue_counts": dict(session_issue_counts),
        "speaker_issue_counts": dict(speaker_issue_counts),
        "official_train_dialogues": int(len(dataset.trainVid)),
        "official_test_dialogues": int(len(dataset.testVid)),
    }

    return stats_row, label_rows, lengths, dialogue_rows, parse_summary


def json_list(values: Iterable[Any]) -> str:
    return json.dumps(sorted(str(value) for value in values), ensure_ascii=False)


def build_dialogue_manifest(dialogue_rows: List[Dict[str, Any]]) -> pd.DataFrame:
    output_rows: List[Dict[str, Any]] = []
    for row in dialogue_rows:
        output_row = {column: row.get(column) for column in DIALOGUE_MANIFEST_COLUMNS}
        output_row["speaker_ids"] = (
            json_list(row.get("speaker_ids", []))
            if row.get("num_speakers") is not None
            else None
        )
        output_rows.append(output_row)
    return pd.DataFrame(output_rows, columns=DIALOGUE_MANIFEST_COLUMNS)


def build_session_distribution(dialogue_rows: List[Dict[str, Any]]) -> pd.DataFrame:
    columns = [
        "split",
        "session_id",
        "num_dialogues",
        "num_utterances",
        "percentage_dialogues",
        "percentage_utterances",
    ]
    totals = {
        split: {
            "dialogues": sum(1 for row in dialogue_rows if row["split"] == split),
            "utterances": sum(
                int(row["num_utterances"])
                for row in dialogue_rows
                if row["split"] == split
            ),
        }
        for split in SPLITS
    }
    grouped: Dict[Tuple[str, str], Dict[str, int]] = defaultdict(
        lambda: {"dialogues": 0, "utterances": 0}
    )
    for row in dialogue_rows:
        session_id = row.get("session_id")
        if not session_id:
            continue
        key = (str(row["split"]), str(session_id))
        grouped[key]["dialogues"] += 1
        grouped[key]["utterances"] += int(row["num_utterances"])

    rows: List[Dict[str, Any]] = []
    for split in SPLITS:
        for _, session_id in sorted(key for key in grouped if key[0] == split):
            values = grouped[(split, session_id)]
            split_totals = totals[split]
            rows.append(
                {
                    "split": split,
                    "session_id": session_id,
                    "num_dialogues": int(values["dialogues"]),
                    "num_utterances": int(values["utterances"]),
                    "percentage_dialogues": (
                        float(values["dialogues"]) / float(split_totals["dialogues"])
                        if split_totals["dialogues"]
                        else 0.0
                    ),
                    "percentage_utterances": (
                        float(values["utterances"]) / float(split_totals["utterances"])
                        if split_totals["utterances"]
                        else 0.0
                    ),
                }
            )
    return pd.DataFrame(rows, columns=columns)


def build_speaker_distribution(dialogue_rows: List[Dict[str, Any]]) -> pd.DataFrame:
    columns = [
        "split",
        "speaker_id",
        "num_dialogues",
        "num_utterances",
        "percentage_dialogues",
        "percentage_utterances",
    ]
    totals = {
        split: {
            "dialogues": sum(1 for row in dialogue_rows if row["split"] == split),
            "utterances": sum(
                int(row["num_utterances"])
                for row in dialogue_rows
                if row["split"] == split
            ),
        }
        for split in SPLITS
    }
    grouped: Dict[Tuple[str, str], Dict[str, int]] = defaultdict(
        lambda: {"dialogues": 0, "utterances": 0}
    )
    for row in dialogue_rows:
        split = str(row["split"])
        speaker_counts = row.get("_speaker_utterance_counts", {})
        for speaker_id, utterance_count in speaker_counts.items():
            key = (split, str(speaker_id))
            grouped[key]["dialogues"] += 1
            grouped[key]["utterances"] += int(utterance_count)

    rows: List[Dict[str, Any]] = []
    for split in SPLITS:
        for _, speaker_id in sorted(key for key in grouped if key[0] == split):
            values = grouped[(split, speaker_id)]
            split_totals = totals[split]
            rows.append(
                {
                    "split": split,
                    "speaker_id": speaker_id,
                    "num_dialogues": int(values["dialogues"]),
                    "num_utterances": int(values["utterances"]),
                    "percentage_dialogues": (
                        float(values["dialogues"]) / float(split_totals["dialogues"])
                        if split_totals["dialogues"]
                        else 0.0
                    ),
                    "percentage_utterances": (
                        float(values["utterances"]) / float(split_totals["utterances"])
                        if split_totals["utterances"]
                        else 0.0
                    ),
                }
            )
    return pd.DataFrame(rows, columns=columns)


def build_split_overlap_summary(dialogue_rows: List[Dict[str, Any]]) -> pd.DataFrame:
    columns = [
        "split_a",
        "split_b",
        "dialogue_id_overlap_count",
        "session_id_overlap_count",
        "speaker_id_overlap_count",
        "dialogue_ids_overlap",
        "session_ids_overlap",
        "speaker_ids_overlap",
    ]
    dialogue_ids = {
        split: {
            str(row["dialogue_id"]) for row in dialogue_rows if row["split"] == split
        }
        for split in SPLITS
    }
    session_ids = {
        split: {
            str(row["session_id"])
            for row in dialogue_rows
            if row["split"] == split and row.get("session_id")
        }
        for split in SPLITS
    }
    speaker_ids = {
        split: {
            str(speaker_id)
            for row in dialogue_rows
            if row["split"] == split
            for speaker_id in row.get("speaker_ids", [])
        }
        for split in SPLITS
    }

    rows: List[Dict[str, Any]] = []
    for split_a, split_b in SPLIT_PAIRS:
        dialogue_overlap = dialogue_ids[split_a] & dialogue_ids[split_b]
        session_overlap = session_ids[split_a] & session_ids[split_b]
        speaker_overlap = speaker_ids[split_a] & speaker_ids[split_b]
        rows.append(
            {
                "split_a": split_a,
                "split_b": split_b,
                "dialogue_id_overlap_count": len(dialogue_overlap),
                "session_id_overlap_count": len(session_overlap),
                "speaker_id_overlap_count": len(speaker_overlap),
                "dialogue_ids_overlap": json_list(dialogue_overlap),
                "session_ids_overlap": json_list(session_overlap),
                "speaker_ids_overlap": json_list(speaker_overlap),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def build_dialogue_level_statistics(
    dialogue_rows: List[Dict[str, Any]],
) -> pd.DataFrame:
    columns = [
        "split",
        "num_dialogues",
        "mean_num_utterances",
        "std_num_utterances",
        "median_num_utterances",
        "num_dialogues_with_majority_class_ratio",
        "percentage_dialogues_with_majority_class_ratio",
        "mean_majority_class_ratio",
        "std_majority_class_ratio",
        "median_majority_class_ratio",
    ]
    rows: List[Dict[str, Any]] = []
    for split in SPLITS:
        split_rows = [row for row in dialogue_rows if row["split"] == split]
        lengths = pd.Series(
            [float(row["num_utterances"]) for row in split_rows],
            dtype="float64",
        )
        majority_ratios = pd.Series(
            [
                float(row["majority_class_ratio"])
                for row in split_rows
                if row.get("majority_class_ratio") is not None
            ],
            dtype="float64",
        )
        rows.append(
            {
                "split": split,
                "num_dialogues": len(split_rows),
                "mean_num_utterances": lengths.mean() if not lengths.empty else None,
                "std_num_utterances": (
                    lengths.std(ddof=0) if not lengths.empty else None
                ),
                "median_num_utterances": (
                    lengths.median() if not lengths.empty else None
                ),
                "num_dialogues_with_majority_class_ratio": int(len(majority_ratios)),
                "percentage_dialogues_with_majority_class_ratio": (
                    float(len(majority_ratios)) / float(len(split_rows))
                    if split_rows
                    else None
                ),
                "mean_majority_class_ratio": (
                    majority_ratios.mean() if not majority_ratios.empty else None
                ),
                "std_majority_class_ratio": (
                    majority_ratios.std(ddof=0) if not majority_ratios.empty else None
                ),
                "median_majority_class_ratio": (
                    majority_ratios.median() if not majority_ratios.empty else None
                ),
            }
        )
    return pd.DataFrame(rows, columns=columns)


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
        lines.append(
            "| " + " | ".join(format_cell(row[column]) for column in df.columns) + " |"
        )
    return "\n".join(lines) + "\n"


def save_table_bundle(df: pd.DataFrame, output_dir: Path, stem: str) -> None:
    csv_path = output_dir / f"{stem}.csv"
    md_path = output_dir / f"{stem}.md"
    tex_path = output_dir / f"{stem}.tex"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    md_path.write_text(dataframe_to_markdown(df), encoding="utf-8")
    tex_path.write_text(df.to_latex(index=False), encoding="utf-8")
    print(f"  [Table] {stem}: {csv_path}")


def save_csv(df: pd.DataFrame, output_dir: Path, stem: str) -> None:
    csv_path = output_dir / f"{stem}.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig", na_rep="")
    print(f"  [CSV] {stem}: {csv_path}")


def save_label_distribution_plot(label_df: pd.DataFrame, output_dir: Path) -> None:
    pivot_df = label_df.pivot(
        index="class_name",
        columns="split",
        values="percentage",
    )
    pivot_df = pivot_df.reindex(
        columns=[split for split in SPLITS if split in pivot_df.columns]
    )

    ax = pivot_df.plot(kind="bar", figsize=(10, 5))
    ax.set_ylabel("Percentage of Utterances")
    ax.set_xlabel("Emotion Class")
    ax.set_title("IEMOCAP Label Distribution by Split")
    ax.set_ylim(
        0, max(1.0, float(pivot_df.max().max()) * 1.15 if len(pivot_df) else 1.0)
    )
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


def overlap_count(
    overlap_df: pd.DataFrame,
    split_a: str,
    split_b: str,
    column: str,
) -> int:
    rows = overlap_df[
        (overlap_df["split_a"] == split_a) & (overlap_df["split_b"] == split_b)
    ]
    if rows.empty:
        return 0
    return int(rows.iloc[0][column])


def build_label_comparison(label_df: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "class_id",
        "class_name",
        "val_percentage",
        "test_percentage",
        "val_minus_test_percentage_points",
    ]
    pivot = label_df.pivot_table(
        index=["class_id", "class_name"],
        columns="split",
        values="percentage",
        aggfunc="first",
    ).reset_index()
    if "val" not in pivot:
        pivot["val"] = 0.0
    if "test" not in pivot:
        pivot["test"] = 0.0
    comparison = pd.DataFrame(
        {
            "class_id": pivot["class_id"],
            "class_name": pivot["class_name"],
            "val_percentage": pivot["val"] * 100.0,
            "test_percentage": pivot["test"] * 100.0,
            "val_minus_test_percentage_points": (pivot["val"] - pivot["test"]) * 100.0,
        }
    )
    return comparison[columns]


def optional_float(value: Any) -> Optional[float]:
    if value is None or (isinstance(value, str) and value == ""):
        return None
    if bool(pd.isna(value)):
        return None
    return float(value)


def format_percentage_points(value: Optional[float]) -> str:
    if value is None or pd.isna(value):
        return "unable_to_determine"
    return f"{float(value):.3f} percentage points"


def dialogue_ease_assessment(dialogue_stats_df: pd.DataFrame) -> Dict[str, Any]:
    val_mean = optional_float(
        row_value(dialogue_stats_df, "val", "mean_majority_class_ratio")
    )
    test_mean = optional_float(
        row_value(dialogue_stats_df, "test", "mean_majority_class_ratio")
    )
    val_median = optional_float(
        row_value(dialogue_stats_df, "val", "median_majority_class_ratio")
    )
    test_median = optional_float(
        row_value(dialogue_stats_df, "test", "median_majority_class_ratio")
    )
    val_std = optional_float(
        row_value(dialogue_stats_df, "val", "std_majority_class_ratio")
    )
    test_std = optional_float(
        row_value(dialogue_stats_df, "test", "std_majority_class_ratio")
    )

    values = [val_mean, test_mean, val_median, test_median, val_std, test_std]
    if any(value is None for value in values):
        return {
            "direction": "unable_to_determine",
            "mean_difference_percentage_points": None,
            "median_difference_percentage_points": None,
            "standardized_mean_difference": None,
            "text": (
                "At least one split has no usable majority-class ratio, so dialogue-level "
                "ease cannot be assessed from this proxy. Empty-label dialogues are excluded "
                "from the ratio summaries and their coverage is reported in the table."
            ),
        }

    assert val_mean is not None
    assert test_mean is not None
    assert val_median is not None
    assert test_median is not None
    assert val_std is not None
    assert test_std is not None

    mean_difference = val_mean - test_mean
    median_difference = val_median - test_median
    pooled_scale = sqrt((val_std**2 + test_std**2) / 2.0)
    standardized_difference = (
        mean_difference / pooled_scale if pooled_scale > 0.0 else None
    )

    if mean_difference > 0.0 and median_difference > 0.0:
        direction = "directionally_consistent"
        conclusion = (
            "Both the mean and median majority-class ratios are higher on validation than "
            "on test. This is directionally consistent with validation containing more "
            "single-emotion-dominated dialogues. The proxy alone cannot establish that "
            "validation is noticeably or intrinsically easier, especially when validation "
            "contains few dialogue units."
        )
    elif mean_difference <= 0.0 and median_difference <= 0.0:
        direction = "does_not_support"
        conclusion = (
            "Neither the mean nor the median majority-class ratio is higher on validation "
            "than on test. This proxy does not support the claim that validation is easier."
        )
    else:
        direction = "mixed"
        conclusion = (
            "The mean and median majority-class-ratio comparisons point in different "
            "directions, so this proxy gives no clear dialogue-level ease signal."
        )

    effect_text = (
        f" The standardized mean difference is {standardized_difference:.3f}."
        if standardized_difference is not None
        else " The standardized mean difference is undefined because the pooled dispersion is zero."
    )
    return {
        "direction": direction,
        "mean_difference_percentage_points": mean_difference * 100.0,
        "median_difference_percentage_points": median_difference * 100.0,
        "standardized_mean_difference": standardized_difference,
        "text": conclusion + effect_text,
    }


def assess_split_risk(
    stats_df: pd.DataFrame,
    overlap_df: pd.DataFrame,
    parse_summaries: Dict[str, Dict[str, Any]],
    ease_assessment: Dict[str, Any],
) -> Tuple[str, List[str]]:
    dialogue_overlap_total = sum(
        int(value) for value in overlap_df["dialogue_id_overlap_count"].tolist()
    )
    parse_complete = all(
        int(summary["session_ids_parsed"]) == int(summary["num_dialogues"])
        and int(summary["speaker_ids_parsed"]) == int(summary["num_dialogues"])
        for summary in parse_summaries.values()
    )
    train_val_session = overlap_count(
        overlap_df, "train", "val", "session_id_overlap_count"
    )
    train_val_speaker = overlap_count(
        overlap_df, "train", "val", "speaker_id_overlap_count"
    )
    train_test_session = overlap_count(
        overlap_df, "train", "test", "session_id_overlap_count"
    )
    val_test_session = overlap_count(
        overlap_df, "val", "test", "session_id_overlap_count"
    )
    train_test_speaker = overlap_count(
        overlap_df, "train", "test", "speaker_id_overlap_count"
    )
    val_test_speaker = overlap_count(
        overlap_df, "val", "test", "speaker_id_overlap_count"
    )
    val_dialogues = int(row_value(stats_df, "val", "num_dialogues"))
    test_dialogues = int(row_value(stats_df, "test", "num_dialogues"))
    val_batches = int(row_value(stats_df, "val", "estimated_num_batches"))

    reasons: List[str] = []
    if dialogue_overlap_total > 0:
        reasons.append(
            f"The summed pairwise dialogue-ID overlap count is {dialogue_overlap_total}; "
            "dialogue membership is therefore not disjoint in the realized data."
        )
        return "high", reasons

    reasons.append("No cross-split dialogue-ID overlap was observed.")
    if not parse_complete:
        reasons.append(
            "Session or speaker parsing was incomplete, so person/session overlap cannot be "
            "fully determined without inventing identities."
        )
        return "unable_to_determine", reasons

    train_val_overlap = train_val_session > 0 or train_val_speaker > 0
    held_out_test_regime = (
        train_test_session == 0
        and val_test_session == 0
        and train_test_speaker == 0
        and val_test_speaker == 0
    )
    if train_val_overlap:
        reasons.append(
            "Validation shares session and/or session-qualified speaker identities with "
            "training. This is not automatically data leakage, but it can make validation "
            "more familiar than a held-out-session test split."
        )
    if held_out_test_regime and train_val_overlap:
        reasons.append(
            "Test is disjoint from both train and validation at the parsed session and "
            "speaker levels, while train and validation are not; the validation and test "
            "domain relationships are therefore structurally different."
        )
    if val_dialogues < test_dialogues:
        reasons.append(
            f"Validation contains {val_dialogues} dialogues ({val_batches} estimated batches) "
            f"versus {test_dialogues} test dialogues, so its metric is based on fewer "
            "dialogue units."
        )
    else:
        reasons.append(
            f"Validation contains {val_dialogues} dialogues ({val_batches} estimated batches) "
            f"versus {test_dialogues} test dialogues."
        )
    if ease_assessment["direction"] == "directionally_consistent":
        reasons.append(
            "Both mean and median majority-class ratios are higher on validation than test, "
            "which is a directional dialogue-composition signal but not proof of easier data."
        )

    if train_val_overlap or val_dialogues < test_dialogues:
        return "moderate", reasons
    return "low", reasons


def save_protocol_report(
    config_path: Path,
    config: Dict[str, Any],
    stats_df: pd.DataFrame,
    label_df: pd.DataFrame,
    session_df: pd.DataFrame,
    speaker_df: pd.DataFrame,
    overlap_df: pd.DataFrame,
    dialogue_stats_df: pd.DataFrame,
    parse_summaries: Dict[str, Dict[str, Any]],
    output_dir: Path,
) -> None:
    dataset_config = config["dataset"]
    valid_ratio = float(dataset_config.get("valid_ratio", 0.1))
    strategy = str(dataset_config.get("val_split_strategy", "official_prefix"))
    normalized_strategy = strategy.lower()
    val_session_id = dataset_config.get("val_session_id")
    configured_seed = int(safe_get(config, ["system", "seed"], 42))
    official_train_count = int(parse_summaries["train"]["official_train_dialogues"])
    official_test_count = int(parse_summaries["train"]["official_test_dialogues"])
    n_val = int(valid_ratio * official_train_count)

    if normalized_strategy == "official_prefix":
        construction = (
            f"`n_val = int({valid_ratio} * {official_train_count}) = {n_val}`; "
            "`val_ids = trainVid[:n_val]`; `train_ids = trainVid[n_val:]`."
        )
        seed_effect = (
            f"The configured seed is `{configured_seed}`, but `official_prefix` performs no "
            "shuffle and does not read the seed; validation membership therefore has no "
            "effective random seed."
        )
        strategy_description = (
            "ordered dialogue-prefix split in the pickle's `trainVid` order; not random and "
            "not session-aware"
        )
    elif normalized_strategy == "random":
        construction = (
            f"`n_val = int({valid_ratio} * {official_train_count}) = {n_val}`; copy "
            "`trainVid`, shuffle it with `random.Random(seed)`, use the first `n_val` IDs "
            "for validation, and use the remainder for training."
        )
        seed_effect = (
            f"The validation shuffle uses configured seed `{configured_seed}`."
        )
        strategy_description = "seeded random dialogue split; not session-aware"
    elif normalized_strategy == "session_holdout":
        construction = (
            f"Iterate through the official `trainVid` in its stored order; assign every "
            f"dialogue whose strict session ID is `{val_session_id}` to validation and "
            "assign the remaining Ses01-Ses04 dialogues to training. Keep the official "
            "`testVid` unchanged as the Ses05 test split."
        )
        seed_effect = (
            f"The configured seed is `{configured_seed}`, but `session_holdout` performs "
            "no shuffle and does not use the seed for split membership."
        )
        strategy_description = (
            f"whole-session validation holdout `{val_session_id}` from the official "
            "training pool; order-preserving and session-aware"
        )
    else:
        raise ValueError(f"Unsupported val_split_strategy: {strategy}")

    val_session_text = (
        f", `val_session_id={val_session_id}`"
        if normalized_strategy == "session_holdout"
        else ""
    )
    if normalized_strategy == "session_holdout":
        protocol_recommendations = f"""1. Keep this declared `{val_session_id}` holdout and its results as a fully documented baseline; do not choose a different validation session after seeing test performance.
2. Keep the official Ses05 `testVid` untouched and use validation metrics only for checkpoint selection.
3. For broader coverage, predeclare rotations across Ses01-Ses04 and aggregate checkpoint-selected results across folds or seeds.
4. Report Weighted-F1, Macro-F1, UAR, and Accuracy together with session/speaker manifests, per-class metrics, and the number of dialogue-level batches.
5. Never use test metrics to select epochs, checkpoints, validation sessions, or seeds."""
    else:
        protocol_recommendations = """1. Keep the current split and its results as a fully documented baseline; do not reinterpret or silently replace it.
2. For the next protocol, keep the official test session untouched and construct validation by whole session (or whole speaker pair) from the official training pool, so checkpoint selection and final testing both measure transfer to unseen identities.
3. Rotate the validation session across the available training sessions and aggregate checkpoint-selected test results across the predeclared folds or seeds. Do not choose the validation fold after seeing test performance.
4. Report Weighted-F1, Macro-F1, UAR, and Accuracy together with session/speaker manifests, per-class metrics, and the number of dialogue-level batches.
5. Never use test metrics to select epochs, checkpoints, validation sessions, or seeds."""

    coverage_rows: List[Dict[str, Any]] = []
    for split in SPLITS:
        summary = parse_summaries[split]
        coverage_rows.append(
            {
                "split": split,
                "dialogues": summary["num_dialogues"],
                "session_ids_parsed": summary["session_ids_parsed"],
                "speaker_ids_parsed": summary["speaker_ids_parsed"],
                "session_parse_issues": json.dumps(
                    summary["session_issue_counts"], sort_keys=True
                ),
                "speaker_parse_issues": json.dumps(
                    summary["speaker_issue_counts"], sort_keys=True
                ),
            }
        )
    coverage_df = pd.DataFrame(coverage_rows)
    label_comparison_df = build_label_comparison(label_df)
    ease = dialogue_ease_assessment(dialogue_stats_df)
    risk_level, risk_reasons = assess_split_risk(
        stats_df=stats_df,
        overlap_df=overlap_df,
        parse_summaries=parse_summaries,
        ease_assessment=ease,
    )

    dialogue_overlap_df = overlap_df[
        ["split_a", "split_b", "dialogue_id_overlap_count", "dialogue_ids_overlap"]
    ]
    session_overlap_df = overlap_df[
        ["split_a", "split_b", "session_id_overlap_count", "session_ids_overlap"]
    ]
    speaker_overlap_df = overlap_df[
        ["split_a", "split_b", "speaker_id_overlap_count", "speaker_ids_overlap"]
    ]
    risk_bullets = "\n".join(f"- {reason}" for reason in risk_reasons)

    report = f"""# IEMOCAP split protocol report

## 1. Active split configuration

- Config: `{project_relative(config_path)}`
- Feature pickle: `{dataset_config['feature_pkl_path']}`
- Official ID source: the `trainVid` and `testVid` objects stored as the last two entries of the 9-item feature pickle; IDs are not inferred from filenames.
- Official dialogue counts read from the pickle: train pool={official_train_count}, test={official_test_count}.
- `valid_ratio={valid_ratio}`, `val_split_strategy={strategy}`{val_session_text}, configured `system.seed={configured_seed}`.
- Active strategy classification: {strategy_description}.

## 2. Exact validation construction

{construction}

{seed_effect}

The training entry point and this diagnostic both pass the same feature path, validation ratio, strategy, validation session ID, and system seed to `IEMOCAPOfficialFeatureDataset`. This report inspects the resulting IDs and does not change them.

## 3. Dialogue overlap

Dialogue IDs should be disjoint in a valid realized split, but the adapter does not assert uniqueness inside `trainVid` or disjointness between `trainVid` and `testVid`; the table below reports the actual set intersections.

{dataframe_to_markdown(dialogue_overlap_df)}

## 4. Session overlap

`session_id` is accepted only when a dialogue ID matches the anchored IEMOCAP form `^(Ses0[1-5])[FM]_`. Unparseable values are left empty rather than guessed.

{dataframe_to_markdown(session_overlap_df)}

Parsing coverage:

{dataframe_to_markdown(coverage_df)}

Session distribution by split:

{dataframe_to_markdown(session_df)}

The distribution CSV stores percentage columns as fractions in `[0, 1]`.

Session overlap is not automatically labelled data leakage. With disjoint dialogue IDs, it means the model is validated on conversations involving a session already represented in training, which can make validation more familiar than a held-out-session test set.

## 5. Speaker overlap

Speaker IDs are session-qualified (`SesNNF` / `SesNNM`), not raw `F/M` role labels. A speaker field is populated only when every utterance's raw `videoSpeakers` value agrees with the speaker suffix in its `videoIDs` utterance ID and the utterance ID belongs to the dialogue. No real name or cross-session identity is inferred.

{dataframe_to_markdown(speaker_overlap_df)}

Speaker distribution by split:

{dataframe_to_markdown(speaker_df)}

In `speaker_distribution_by_split.csv`, percentage columns are stored as fractions in `[0, 1]`, and `num_utterances` counts utterances actually spoken by that speaker. `percentage_dialogues` can sum above 100% because a dialogue can contain more than one speaker.

As with session overlap, speaker overlap alone is not automatically data leakage. It is a diagnostic for whether validation contains speakers already represented during training while test does not.

## 6. Label-distribution observations

Percentages below are utterance-level; the final column is validation minus test in percentage points.

{dataframe_to_markdown(label_comparison_df)}

Label-composition differences can contribute to a validation-test gap, but this table alone cannot establish causality.

## 7. Dialogue-level observations

The standard deviations below use `ddof=0` because each row describes the complete realized split rather than estimates from a sampled subset.

{dataframe_to_markdown(dialogue_stats_df)}

- Validation minus test mean majority-class ratio: {format_percentage_points(ease['mean_difference_percentage_points'])}.
- Validation minus test median majority-class ratio: {format_percentage_points(ease['median_difference_percentage_points'])}.
- {ease['text']}

`majority_class_ratio` is only a composition proxy: higher values may make a dialogue easier for a model that benefits from emotion persistence, but difficulty also depends on context, modality quality, speaker turns, class identity, and annotation ambiguity.

## 8. Risk assessment

Risk level: **{risk_level}**

{risk_bullets}

The level assesses the risk that the current validation score is an optimistic or unstable proxy for held-out test performance. It is not a declaration of data leakage based solely on shared sessions or speakers.

## 9. Recommended next experiment protocol

{protocol_recommendations}
"""

    report_path = output_dir / "split_protocol_report.md"
    report_path.write_text(report, encoding="utf-8")
    print(f"  [Report] {report_path}")


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

    config = load_yaml(config_path)
    frozen_date = resolve_experiment_date(
        cli_date=args.experiment_date,
        config=config,
    )
    output_root = resolve_path(str(configured_output_root(config)))
    output_dir = (
        resolve_path(args.output_dir)
        if args.output_dir is not None
        else resolve_output_category("audits", frozen_date, output_root)
        / "iemocap_split_diagnostics"
    )
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
    dialogue_rows: List[Dict[str, Any]] = []
    parse_summaries: Dict[str, Dict[str, Any]] = {}

    for split in SPLITS:
        (
            stats_row,
            split_label_rows,
            lengths,
            split_dialogue_rows,
            parse_summary,
        ) = collect_split_info(
            config=config,
            split=split,
            batch_size=batch_size,
            label_list=label_list,
        )
        stats_rows.append(stats_row)
        label_rows.extend(split_label_rows)
        lengths_by_split[split] = lengths
        dialogue_rows.extend(split_dialogue_rows)
        parse_summaries[split] = parse_summary
        print(
            f"[OK] {split}: dialogues={stats_row['num_dialogues']} "
            f"utterances={stats_row['num_utterances']} "
            f"estimated_batches={stats_row['estimated_num_batches']}"
        )

    stats_df = pd.DataFrame(stats_rows)
    label_df = pd.DataFrame(label_rows)
    dialogue_manifest_df = build_dialogue_manifest(dialogue_rows)
    session_distribution_df = build_session_distribution(dialogue_rows)
    speaker_distribution_df = build_speaker_distribution(dialogue_rows)
    overlap_df = build_split_overlap_summary(dialogue_rows)
    dialogue_stats_df = build_dialogue_level_statistics(dialogue_rows)

    output_dir.mkdir(parents=True, exist_ok=True)
    save_table_bundle(stats_df, output_dir, "split_statistics")
    save_table_bundle(label_df, output_dir, "label_distribution_by_split")
    save_csv(dialogue_manifest_df, output_dir, "dialogue_split_manifest")
    save_csv(session_distribution_df, output_dir, "session_distribution_by_split")
    save_csv(speaker_distribution_df, output_dir, "speaker_distribution_by_split")
    save_csv(overlap_df, output_dir, "split_overlap_summary")
    save_csv(dialogue_stats_df, output_dir, "dialogue_level_statistics")
    save_label_distribution_plot(label_df, output_dir)
    save_utterance_count_plot(stats_df, output_dir)
    save_dialogue_length_distribution_plot(lengths_by_split, output_dir)
    save_notes(stats_df, output_dir)
    save_protocol_report(
        config_path=config_path,
        config=config,
        stats_df=stats_df,
        label_df=label_df,
        session_df=session_distribution_df,
        speaker_df=speaker_distribution_df,
        overlap_df=overlap_df,
        dialogue_stats_df=dialogue_stats_df,
        parse_summaries=parse_summaries,
        output_dir=output_dir,
    )

    print("=" * 100)
    print("Finished split diagnostics.")
    print("=" * 100)


if __name__ == "__main__":
    main()

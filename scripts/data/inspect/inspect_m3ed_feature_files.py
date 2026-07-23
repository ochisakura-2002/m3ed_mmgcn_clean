"""
检查 M3ED 官方 feature pkl 文件。

这个脚本会读取 data/processed/M3ED/features/ 下的所有 pkl 文件，
检查它们是否符合 DialogueRNN-style pkl 格式，并和 m3ed_metadata.csv 对齐。

它不训练模型，不依赖 torch，也不会生成新文件。

运行方式：
    python scripts/data/inspect/inspect_m3ed_feature_files.py
"""

from pathlib import Path
import sys
from typing import Any, Dict, List, Set

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT))

from datasets.common.dialogue_feature_dataset import DialogueFeatureDataset  # noqa: E402


FEATURE_DIR = PROJECT_ROOT / "data" / "processed" / "M3ED" / "features"
METADATA_PATH = PROJECT_ROOT / "data" / "metadata" / "m3ed_metadata.csv"


def print_section(title: str) -> None:
    """打印分隔标题。"""
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)


def describe_value(value: Any) -> str:
    """
    简要描述一个字段的类型和 shape。
    """
    if value is None:
        return "None"

    array = np.asarray(value)

    if array.dtype == object:
        return f"type={type(value).__name__}, len={len(value)}"

    return f"shape={array.shape}, dtype={array.dtype}"


def load_metadata_dialogue_ids() -> Dict[str, Set[str]]:
    """
    从 m3ed_metadata.csv 中读取每个 split 的 dialogue_id 集合。
    """
    if not METADATA_PATH.exists():
        raise FileNotFoundError(f"Metadata file not found: {METADATA_PATH}")

    df = pd.read_csv(METADATA_PATH)

    split_to_dialogue_ids = {}

    for split in ["train", "val", "test"]:
        split_df = df[df["split"] == split]
        dialogue_ids = set(split_df["dialogue_id"].astype(str).unique())
        split_to_dialogue_ids[split] = dialogue_ids

    split_to_dialogue_ids["all"] = set(df["dialogue_id"].astype(str).unique())

    return split_to_dialogue_ids


def collect_feature_dialogue_ids(
    pkl_path: Path,
) -> Dict[str, Set[str]]:
    """
    读取 feature pkl 中 train / val / test 的 dialogue_id 集合。

    如果某个 split 不存在，则返回空集合。
    """
    split_to_ids = {}

    for split in ["train", "val", "test"]:
        try:
            dataset = DialogueFeatureDataset(
                feature_pkl_path=str(pkl_path),
                split=split,
            )
            split_to_ids[split] = set(str(x) for x in dataset.dialogue_ids)
        except Exception:
            split_to_ids[split] = set()

    all_ids = set()
    for ids in split_to_ids.values():
        all_ids.update(ids)

    split_to_ids["all"] = all_ids

    return split_to_ids


def print_alignment_report(
    metadata_ids: Dict[str, Set[str]],
    feature_ids: Dict[str, Set[str]],
) -> None:
    """
    打印 metadata dialogue_id 和 feature dialogue_id 的对齐情况。
    """
    print("\nDialogue ID alignment with metadata:")

    for split in ["train", "val", "test", "all"]:
        meta_set = metadata_ids[split]
        feat_set = feature_ids[split]

        matched = meta_set & feat_set
        missing_in_feature = meta_set - feat_set
        extra_in_feature = feat_set - meta_set

        print(f"\n  Split: {split}")
        print(f"    metadata dialogues: {len(meta_set)}")
        print(f"    feature dialogues: {len(feat_set)}")
        print(f"    matched dialogues: {len(matched)}")
        print(f"    missing in feature: {len(missing_in_feature)}")
        print(f"    extra in feature: {len(extra_in_feature)}")

        if len(missing_in_feature) > 0:
            print(f"    first missing: {sorted(list(missing_in_feature))[:5]}")

        if len(extra_in_feature) > 0:
            print(f"    first extra: {sorted(list(extra_in_feature))[:5]}")


def inspect_sample(dataset: DialogueFeatureDataset) -> None:
    """
    打印第一个 dialogue 样本。
    """
    sample = dataset[0]

    print("\nFirst sample preview:")
    print("  dialogue_id:", sample["dialogue_id"])
    print("  num_utterances:", sample["num_utterances"])
    print("  speakers:", describe_value(sample["speakers"]))
    print("  labels:", describe_value(sample["labels"]))
    print("  text_features:", describe_value(sample["text_features"]))
    print("  audio_features:", describe_value(sample["audio_features"]))
    print("  visual_features:", describe_value(sample["visual_features"]))
    print("  sentences:", describe_value(sample["sentences"]))

    labels = sample["labels"]
    sentences = sample["sentences"]

    if labels is not None:
        print("  first 10 labels:", list(labels)[:10])

    if sentences is not None:
        print("  first 3 sentences:", list(sentences)[:3])


def inspect_split(pkl_path: Path, split: str) -> None:
    """
    检查一个 pkl 文件的一个 split。
    """
    print(f"\n--- split={split} ---")

    try:
        dataset = DialogueFeatureDataset(
            feature_pkl_path=str(pkl_path),
            split=split,
        )
    except Exception as error:
        print(f"[SKIP/ERROR] split={split}: {repr(error)}")
        return

    print("Number of dialogues:", len(dataset))

    print("Split summary:")
    for key, value in dataset.split_summary().items():
        print(f"  {key}: {value}")

    print("Feature dimensions:")
    for key, value in dataset.feature_dimensions().items():
        print(f"  {key}: {value}")

    inspect_sample(dataset)


def inspect_one_feature_file(
    pkl_path: Path,
    metadata_ids: Dict[str, Set[str]],
) -> None:
    """
    检查一个 M3ED feature pkl 文件。
    """
    print_section(f"Inspect feature file: {pkl_path.name}")

    print("Path:", pkl_path)
    print("File size:", f"{pkl_path.stat().st_size / 1024 / 1024:.2f} MB")

    feature_ids = collect_feature_dialogue_ids(pkl_path)
    print_alignment_report(metadata_ids, feature_ids)

    for split in ["train", "val", "test"]:
        inspect_split(pkl_path, split)


def main() -> None:
    """主函数。"""
    print("=" * 100)
    print("Inspect M3ED feature pkl files")
    print("=" * 100)

    print("Project root:", PROJECT_ROOT)
    print("Feature dir:", FEATURE_DIR)
    print("Metadata path:", METADATA_PATH)

    if not FEATURE_DIR.exists():
        raise FileNotFoundError(f"Feature directory not found: {FEATURE_DIR}")

    pkl_files = sorted(FEATURE_DIR.glob("*.pkl"))

    if len(pkl_files) == 0:
        raise FileNotFoundError(f"No pkl files found in: {FEATURE_DIR}")

    print("\nFound pkl files:")
    for path in pkl_files:
        print(" ", path.name)

    metadata_ids = load_metadata_dialogue_ids()

    print("\nMetadata dialogue counts:")
    for split in ["train", "val", "test", "all"]:
        print(f"  {split}: {len(metadata_ids[split])}")

    for pkl_path in pkl_files:
        inspect_one_feature_file(pkl_path, metadata_ids)

    print("\nInspection finished.")
    print("=" * 100)


if __name__ == "__main__":
    main()

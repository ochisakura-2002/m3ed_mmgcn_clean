"""
测试 M3EDFeatureDataset。

这个脚本用于确认：
1. M3ED feature pkl 能否读取
2. metadata 和 feature dialogue_id 是否对齐
3. metadata label 和 feature label 是否一致
4. val/test split 互换问题是否已经修正
5. 三模态特征维度是否正确

运行方式：
    python scripts/debug_m3ed_feature_dataset.py
"""

from pathlib import Path
import sys
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from utils.io import load_yaml  # noqa: E402
from datasets.m3ed.feature_dataset import M3EDFeatureDataset  # noqa: E402


def print_section(title: str) -> None:
    """打印分隔标题。"""
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def describe_array(value: Any) -> str:
    """
    简要描述数组或列表。
    """
    if value is None:
        return "None"

    array = np.asarray(value)

    return f"shape={array.shape}, dtype={array.dtype}"


def inspect_split(feature_pkl_path: str, split: str) -> M3EDFeatureDataset:
    """
    检查一个 split。
    """
    print_section(f"M3EDFeatureDataset | split={split}")

    dataset = M3EDFeatureDataset(
        feature_pkl_path=feature_pkl_path,
        split=split,
    )

    print("Number of dialogues:", len(dataset))

    print("\nSplit summary:")
    for key, value in dataset.split_summary().items():
        print(f"  {key}: {value}")

    print("\nFeature dimensions:")
    for key, value in dataset.feature_dimensions().items():
        print(f"  {key}: {value}")

    sample = dataset[0]

    print("\nFirst sample:")
    print("  movie_id:", sample["movie_id"])
    print("  dialogue_id:", sample["dialogue_id"])
    print("  split:", sample["split"])
    print("  num_utterances:", sample["num_utterances"])
    print("  text_features:", describe_array(sample["text_features"]))
    print("  audio_features:", describe_array(sample["audio_features"]))
    print("  visual_features:", describe_array(sample["visual_features"]))
    print("  first 10 labels:", sample["labels"][:10])
    print("  first 10 label_names:", sample["label_names"][:10])
    print("  first 3 texts:", sample["texts"][:3])

    return dataset


def main() -> None:
    """主函数。"""
    print("=" * 80)
    print("Debug M3EDFeatureDataset")
    print("=" * 80)

    print("Project root:", PROJECT_ROOT)

    config = load_yaml("configs/train_mmgcn_m3ed.yaml")
    feature_pkl_path = config["dataset"]["feature_pkl_path"]

    print("Feature pkl path:", feature_pkl_path)

    train_dataset = inspect_split(feature_pkl_path, "train")
    val_dataset = inspect_split(feature_pkl_path, "val")
    test_dataset = inspect_split(feature_pkl_path, "test")

    total_dialogues = len(train_dataset) + len(val_dataset) + len(test_dataset)
    total_utterances = (
        train_dataset.num_utterances()
        + val_dataset.num_utterances()
        + test_dataset.num_utterances()
    )

    print_section("Total check")
    print("Total dialogues:", total_dialogues)
    print("Total utterances:", total_utterances)

    print("\nDebug M3EDFeatureDataset finished.")
    print("=" * 80)


if __name__ == "__main__":
    main()
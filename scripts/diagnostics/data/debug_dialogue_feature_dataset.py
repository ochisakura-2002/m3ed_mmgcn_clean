"""
测试 DialogueFeatureDataset。

这个脚本读取 MMGCN 官方自带的 IEMOCAP / MELD pkl 文件，
用于验证 DialogueRNN-style pkl adapter 是否能正常工作。

它不训练模型，不需要 torch，也不会生成新文件。

运行方式：
    python scripts/diagnostics/data/debug_dialogue_feature_dataset.py
"""

from pathlib import Path
import sys
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT))

from datasets.common.dialogue_feature_dataset import DialogueFeatureDataset  # noqa: E402


IEMOCAP_PKL = (
    "third_party/MMGCN_official/IEMOCAP_features/IEMOCAP_features.pkl"
)

MELD_PKL = (
    "third_party/MMGCN_official/MELD_features/MELD_features_raw1.pkl"
)


def print_section(title: str) -> None:
    """打印分隔标题。"""
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


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


def inspect_sample(dataset: DialogueFeatureDataset) -> None:
    """
    打印第一个 dialogue 样本。
    """
    sample = dataset[0]

    print("\nFirst sample:")
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


def inspect_dataset(name: str, pkl_path: str, split: str) -> None:
    """
    检查某个 pkl 的某个 split。
    """
    print_section(f"{name} | split={split}")

    try:
        dataset = DialogueFeatureDataset(
            feature_pkl_path=pkl_path,
            split=split,
        )
    except Exception as error:
        print(f"[SKIP/ERROR] {name} split={split}: {repr(error)}")
        return

    print("PKL path:", pkl_path)
    print("Number of dialogues:", len(dataset))

    print("\nSplit summary:")
    for key, value in dataset.split_summary().items():
        print(f"  {key}: {value}")

    print("\nFeature dimensions:")
    for key, value in dataset.feature_dimensions().items():
        print(f"  {key}: {value}")

    inspect_sample(dataset)


def main() -> None:
    """主函数。"""
    print("=" * 80)
    print("Debug DialogueFeatureDataset")
    print("=" * 80)

    print("Project root:", PROJECT_ROOT)

    inspect_dataset("IEMOCAP", IEMOCAP_PKL, "train")
    inspect_dataset("IEMOCAP", IEMOCAP_PKL, "test")

    inspect_dataset("MELD", MELD_PKL, "train")
    inspect_dataset("MELD", MELD_PKL, "val")
    inspect_dataset("MELD", MELD_PKL, "test")

    print("\nDebug DialogueFeatureDataset finished.")
    print("=" * 80)


if __name__ == "__main__":
    main()

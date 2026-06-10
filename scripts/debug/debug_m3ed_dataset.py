"""
测试 M3EDDialogueDataset。

这个脚本只检查 metadata dataset 是否能正常工作。
它不读取真实音频/视频/特征，也不需要 torch。

运行方式：
    python scripts/debug_m3ed_dataset.py
"""

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from datasets.m3ed.metadata_dataset import M3EDDialogueDataset  # noqa: E402


def print_section(title: str) -> None:
    """打印分隔标题。"""
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def inspect_split(split: str) -> M3EDDialogueDataset:
    """
    检查某个 split 的 dataset。
    """
    dataset = M3EDDialogueDataset(split=split)

    print_section(f"Split: {split}")

    print("Number of dialogues:", len(dataset))
    print("Number of utterances:", dataset.num_utterances())

    print("\nDialogue length stats:")
    for key, value in dataset.dialogue_length_stats().items():
        print(f"  {key}: {value}")

    print("\nLabel distribution:")
    print(dataset.label_distribution())

    return dataset


def inspect_first_sample(dataset: M3EDDialogueDataset) -> None:
    """
    打印第一个 dialogue 样本的摘要。
    """
    print_section("First train dialogue sample")

    sample = dataset[0]

    print("Sample keys:", list(sample.keys()))
    print("movie_id:", sample["movie_id"])
    print("dialogue_id:", sample["dialogue_id"])
    print("split:", sample["split"])
    print("num_utterances:", sample["num_utterances"])

    print("\nFirst 5 utterances:")
    max_show = min(5, sample["num_utterances"])

    for i in range(max_show):
        print("-" * 40)
        print("utterance_id:", sample["utterance_ids"][i])
        print("speaker:", sample["speaker_ids"][i])
        print("label:", sample["label_names"][i], sample["labels"][i])
        print("text:", sample["texts"][i])


def main() -> None:
    """主函数。"""
    print("=" * 60)
    print("Debug M3EDDialogueDataset")
    print("=" * 60)

    print("Project root:", PROJECT_ROOT)

    train_dataset = inspect_split("train")
    val_dataset = inspect_split("val")
    test_dataset = inspect_split("test")

    total_utterances = (
        train_dataset.num_utterances()
        + val_dataset.num_utterances()
        + test_dataset.num_utterances()
    )

    print_section("Total check")
    print("Total utterances across splits:", total_utterances)

    inspect_first_sample(train_dataset)

    print("\nDebug M3EDDialogueDataset finished.")
    print("=" * 60)


if __name__ == "__main__":
    main()
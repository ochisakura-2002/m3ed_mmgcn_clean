"""
测试 M3ED PyTorch DataLoader。

这个脚本用于确认：
1. M3EDTorchDataset 能否正常返回 torch Tensor
2. collate_fn 能否正确 padding 变长 dialogue
3. DataLoader 能否产出 batch
4. batch 的 shape 是否符合后续模型输入要求

运行方式：
    python scripts/diagnostics/data/debug_m3ed_dataloader.py
"""

from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from utils.io import load_yaml  # noqa: E402
from datasets.m3ed.torch_dataset import M3EDTorchDataset  # noqa: E402
from datasets.collators.m3ed_collate import m3ed_dialogue_collate_fn, IGNORE_INDEX  # noqa: E402


def print_section(title: str) -> None:
    """
    打印分隔标题。
    """
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def build_dataloader(
    feature_pkl_path: str,
    split: str,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    """
    构建 DataLoader。
    """
    dataset = M3EDTorchDataset(
        feature_pkl_path=feature_pkl_path,
        split=split,
        check_label_consistency=True,
        raise_on_label_mismatch=False,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        collate_fn=m3ed_dialogue_collate_fn,
    )

    return loader


def inspect_batch(batch: dict) -> None:
    """
    打印一个 batch 的结构。
    """
    print("\nTensor shapes:")
    print("  text_features:", batch["text_features"].shape)
    print("  audio_features:", batch["audio_features"].shape)
    print("  visual_features:", batch["visual_features"].shape)
    print("  labels:", batch["labels"].shape)
    print("  speaker_ids_int:", batch["speaker_ids_int"].shape)
    print("  attention_mask:", batch["attention_mask"].shape)
    print("  lengths:", batch["lengths"].shape)

    print("\nTensor dtypes:")
    print("  text_features:", batch["text_features"].dtype)
    print("  audio_features:", batch["audio_features"].dtype)
    print("  visual_features:", batch["visual_features"].dtype)
    print("  labels:", batch["labels"].dtype)
    print("  attention_mask:", batch["attention_mask"].dtype)

    valid_positions = batch["attention_mask"]
    valid_label_count = int((batch["labels"] != IGNORE_INDEX).sum().item())

    print("\nMask / label check:")
    print("  valid positions from attention_mask:", int(valid_positions.sum().item()))
    print("  valid labels:", valid_label_count)
    print("  padded labels:", int((batch["labels"] == IGNORE_INDEX).sum().item()))

    valid_labels = batch["labels"][batch["labels"] != IGNORE_INDEX]

    if valid_labels.numel() > 0:
        print("  valid label min:", int(valid_labels.min().item()))
        print("  valid label max:", int(valid_labels.max().item()))
        print("  valid label unique:", sorted(valid_labels.unique().tolist()))

    print("\nBatch metadata:")
    print("  dialogue_ids:", batch["dialogue_ids"])
    print("  lengths:", batch["lengths"].tolist())
    print("  has_label_mismatch:", batch["has_label_mismatch"])

    print("\nFirst dialogue preview:")
    print("  dialogue_id:", batch["dialogue_ids"][0])
    print("  num_utterances:", int(batch["lengths"][0].item()))
    print("  first 3 texts:", batch["texts"][0][:3])
    print("  first 10 label names:", batch["label_names"][0][:10])


def inspect_split(
    feature_pkl_path: str,
    split: str,
    batch_size: int,
) -> None:
    """
    检查一个 split 的 DataLoader。
    """
    print_section(f"DataLoader debug | split={split}")

    loader = build_dataloader(
        feature_pkl_path=feature_pkl_path,
        split=split,
        batch_size=batch_size,
        shuffle=False,
    )

    dataset = loader.dataset

    print("Number of dialogues:", len(dataset))

    print("\nDataset summary:")
    for key, value in dataset.split_summary().items():
        print(f"  {key}: {value}")

    print("\nFeature dimensions:")
    for key, value in dataset.feature_dimensions().items():
        print(f"  {key}: {value}")

    batch = next(iter(loader))

    inspect_batch(batch)


def main() -> None:
    """
    主函数。
    """
    print("=" * 80)
    print("Debug M3ED DataLoader")
    print("=" * 80)

    print("Project root:", PROJECT_ROOT)
    print("Torch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())

    config = load_yaml("configs/train_mmgcn_m3ed.yaml")

    feature_pkl_path = config["dataset"]["feature_pkl_path"]
    batch_size = int(config["train"]["batch_size"])

    print("Feature pkl path:", feature_pkl_path)
    print("Batch size:", batch_size)

    inspect_split(feature_pkl_path, "train", batch_size)
    inspect_split(feature_pkl_path, "val", batch_size)
    inspect_split(feature_pkl_path, "test", batch_size)

    print("\nDebug M3ED DataLoader finished.")
    print("=" * 80)


if __name__ == "__main__":
    main()

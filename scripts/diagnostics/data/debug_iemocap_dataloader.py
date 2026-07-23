"""
Debug IEMOCAP official feature dataloader.

Run:
    python scripts/diagnostics/data/debug_iemocap_dataloader.py

Optional:
    python scripts/diagnostics/data/debug_iemocap_dataloader.py \
      --feature-pkl-path third_party/MMGCN_official/IEMOCAP_features/IEMOCAP_features.pkl \
      --batch-size 8
"""

from __future__ import annotations

from pathlib import Path
import argparse
import sys
from collections import Counter
from typing import Optional

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[3]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from datasets.iemocap import (  # noqa: E402
    IEMOCAPOfficialFeatureDataset,
    build_iemocap_dataloader,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Debug IEMOCAP official feature dataloader."
    )

    parser.add_argument(
        "--feature-pkl-path",
        type=str,
        default="third_party/MMGCN_official/IEMOCAP_features/IEMOCAP_features.pkl",
        help="Path to official IEMOCAP_features.pkl.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for debug dataloader.",
    )
    parser.add_argument(
        "--valid-ratio",
        type=float,
        default=0.1,
        help="Validation ratio split from official trainVid.",
    )
    parser.add_argument(
        "--val-split-strategy",
        type=str,
        default="official_prefix",
        choices=["official_prefix", "random", "session_holdout"],
        help="How to split official trainVid into train/val.",
    )
    parser.add_argument(
        "--val-session-id",
        type=str,
        default=None,
        choices=["Ses01", "Ses02", "Ses03", "Ses04"],
        help="Whole validation session required by session_holdout.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed used only when val_split_strategy=random.",
    )

    return parser.parse_args()


def describe_dataset(
    feature_pkl_path: str,
    split: str,
    valid_ratio: float,
    val_split_strategy: str,
    seed: int,
    val_session_id: Optional[str] = None,
) -> IEMOCAPOfficialFeatureDataset:
    dataset = IEMOCAPOfficialFeatureDataset(
        feature_pkl_path=feature_pkl_path,
        split=split,
        valid_ratio=valid_ratio,
        val_split_strategy=val_split_strategy,
        val_session_id=val_session_id,
        seed=seed,
    )

    labels = dataset.return_labels()
    label_counter = Counter(labels)
    split_ids = dataset.get_split_ids()

    print("-" * 100)
    print(f"Dataset split: {split}")
    print("-" * 100)
    print("num dialogues:", len(dataset))
    print("feature dims:", dataset.get_feature_dims())
    print("label counts:", dict(sorted(label_counter.items())))
    print("first 5 dialogue ids:", dataset.keys[:5])
    print(
        "global split sizes:",
        {
            "train": len(split_ids["train"]),
            "val": len(split_ids["val"]),
            "test": len(split_ids["test"]),
        },
    )

    item = dataset[0]
    print("\nFirst item:")
    print("  dialogue_id:", item["dialogue_id"])
    print("  length:", item["length"])
    print("  text_features:", tuple(item["text_features"].shape), item["text_features"].dtype)
    print("  audio_features:", tuple(item["audio_features"].shape), item["audio_features"].dtype)
    print("  visual_features:", tuple(item["visual_features"].shape), item["visual_features"].dtype)
    print("  labels:", tuple(item["labels"].shape), item["labels"].dtype)
    print("  speaker_ids_int:", tuple(item["speaker_ids_int"].shape), item["speaker_ids_int"].dtype)
    print("  first 10 speakers:", item["speaker_ids_int"][:10].tolist())
    print("  first 10 labels:", item["labels"][:10].tolist())

    return dataset


def describe_batch(
    feature_pkl_path: str,
    split: str,
    batch_size: int,
    valid_ratio: float,
    val_split_strategy: str,
    seed: int,
    val_session_id: Optional[str] = None,
) -> None:
    loader = build_iemocap_dataloader(
        feature_pkl_path=feature_pkl_path,
        split=split,
        batch_size=batch_size,
        valid_ratio=valid_ratio,
        val_split_strategy=val_split_strategy,
        val_session_id=val_session_id,
        seed=seed,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    batch = next(iter(loader))

    print("\nBatch:")
    print("  dialogue_ids:", batch["dialogue_ids"][:3])
    print("  text_features:", tuple(batch["text_features"].shape), batch["text_features"].dtype)
    print("  audio_features:", tuple(batch["audio_features"].shape), batch["audio_features"].dtype)
    print("  visual_features:", tuple(batch["visual_features"].shape), batch["visual_features"].dtype)
    print("  labels:", tuple(batch["labels"].shape), batch["labels"].dtype)
    print("  attention_mask:", tuple(batch["attention_mask"].shape), batch["attention_mask"].dtype)
    print("  lengths:", batch["lengths"].tolist())
    print("  speaker_ids_int:", tuple(batch["speaker_ids_int"].shape), batch["speaker_ids_int"].dtype)

    valid_count = int((batch["labels"] != -100).sum().item())
    mask_count = int(batch["attention_mask"].sum().item())

    print("  valid label count:", valid_count)
    print("  attention mask count:", mask_count)

    if valid_count != mask_count:
        raise RuntimeError(
            f"valid label count != attention mask count: {valid_count} vs {mask_count}"
        )


def main() -> None:
    args = parse_args()

    print("=" * 100)
    print("Debug IEMOCAP official feature dataloader")
    print("=" * 100)
    print("Project root:", PROJECT_ROOT)
    print("Feature pkl:", args.feature_pkl_path)
    print("Batch size:", args.batch_size)
    print("Valid ratio:", args.valid_ratio)
    print("Val split strategy:", args.val_split_strategy)
    print("Val session ID:", args.val_session_id)
    print("Seed:", args.seed)

    for split in ["train", "val", "test"]:
        describe_dataset(
            feature_pkl_path=args.feature_pkl_path,
            split=split,
            valid_ratio=args.valid_ratio,
            val_split_strategy=args.val_split_strategy,
            val_session_id=args.val_session_id,
            seed=args.seed,
        )
        describe_batch(
            feature_pkl_path=args.feature_pkl_path,
            split=split,
            batch_size=args.batch_size,
            valid_ratio=args.valid_ratio,
            val_split_strategy=args.val_split_strategy,
            val_session_id=args.val_session_id,
            seed=args.seed,
        )

    print("=" * 100)
    print("IEMOCAP dataloader debug finished.")
    print("=" * 100)


if __name__ == "__main__":
    main()

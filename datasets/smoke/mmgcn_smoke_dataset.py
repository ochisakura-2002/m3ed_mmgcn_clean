"""
Small deterministic dialogue dataset for MMGCN smoke tests.

This dataset does not read files from data/. It only produces synthetic
dialogue-level batches with the same fields as the M3ED torch dataset, so the
training and checkpoint-evaluation loops can be tested without real features.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import torch
from torch.utils.data import DataLoader, Dataset

from datasets.collators.m3ed_collate import m3ed_dialogue_collate_fn


DEFAULT_SPLIT_SIZES = {
    "train": 4,
    "val": 2,
    "test": 2,
}


class MMGCNSmokeDialogueDataset(Dataset):
    """Deterministic fake dialogue dataset with the MMGCN batch interface."""

    def __init__(
        self,
        split: str,
        num_classes: int,
        label_list: Optional[Sequence[str]],
        text_dim: int,
        audio_dim: int,
        visual_dim: int,
        split_sizes: Optional[Dict[str, int]] = None,
        dialogue_lengths: Optional[Sequence[int]] = None,
        seed: int = 1234,
    ) -> None:
        if split not in {"train", "val", "test"}:
            raise ValueError(f"Unsupported smoke split: {split}")

        self.split = split
        self.num_classes = int(num_classes)
        self.label_list = (
            [str(x) for x in label_list]
            if label_list is not None
            else [str(i) for i in range(self.num_classes)]
        )
        self.text_dim = int(text_dim)
        self.audio_dim = int(audio_dim)
        self.visual_dim = int(visual_dim)
        self.seed = int(seed)

        if self.num_classes <= 0:
            raise ValueError("num_classes must be positive for smoke dataset.")

        if len(self.label_list) != self.num_classes:
            raise ValueError(
                f"label_list length {len(self.label_list)} != num_classes {self.num_classes}"
            )

        self.split_sizes = dict(DEFAULT_SPLIT_SIZES)
        if split_sizes is not None:
            self.split_sizes.update({str(k): int(v) for k, v in split_sizes.items()})

        self.dialogue_lengths = (
            [3, 4]
            if dialogue_lengths is None
            else [int(x) for x in dialogue_lengths]
        )

        if len(self.dialogue_lengths) == 0 or min(self.dialogue_lengths) <= 0:
            raise ValueError("dialogue_lengths must contain positive integers.")

    def __len__(self) -> int:
        return int(self.split_sizes.get(self.split, 0))

    def __getitem__(self, index: int) -> Dict[str, Any]:
        length = self.dialogue_lengths[index % len(self.dialogue_lengths)]
        labels = torch.as_tensor(
            [(index + time_index) % self.num_classes for time_index in range(length)],
            dtype=torch.long,
        )

        text_features = self._make_features(
            index=index,
            labels=labels,
            dim=self.text_dim,
            split_offset=0,
        )
        audio_features = self._make_features(
            index=index,
            labels=labels,
            dim=self.audio_dim,
            split_offset=100,
        )
        visual_features = self._make_features(
            index=index,
            labels=labels,
            dim=self.visual_dim,
            split_offset=200,
        )

        speaker_ids = ["A" if time_index % 2 == 0 else "B" for time_index in range(length)]
        speaker_ids_int = torch.as_tensor(
            [0 if speaker_id == "A" else 1 for speaker_id in speaker_ids],
            dtype=torch.long,
        )

        dialogue_id = f"smoke_{self.split}_{index:03d}"

        return {
            "movie_id": f"smoke_movie_{self.split}",
            "dialogue_id": dialogue_id,
            "split": self.split,
            "num_utterances": length,
            "utterance_ids": [
                f"{dialogue_id}_utt_{time_index:03d}"
                for time_index in range(length)
            ],
            "utterance_indices": list(range(length)),
            "texts": [
                f"smoke utterance {time_index}"
                for time_index in range(length)
            ],
            "speaker_ids": speaker_ids,
            "speaker_names": speaker_ids,
            "speaker_genders": ["" for _ in range(length)],
            "label_names": [
                self.label_list[int(label_id)]
                for label_id in labels.tolist()
            ],
            "text_features": text_features,
            "audio_features": audio_features,
            "visual_features": visual_features,
            "labels": labels,
            "speaker_ids_int": speaker_ids_int,
            "has_label_mismatch": False,
            "label_mismatch_positions": [],
        }

    def _make_features(
        self,
        index: int,
        labels: torch.Tensor,
        dim: int,
        split_offset: int,
    ) -> torch.Tensor:
        generator = torch.Generator()
        generator.manual_seed(
            self.seed
            + self._split_seed_offset()
            + split_offset
            + int(index)
        )

        features = torch.randn(
            labels.shape[0],
            int(dim),
            generator=generator,
            dtype=torch.float32,
        ) * 0.05

        for time_index, label_id in enumerate(labels.tolist()):
            features[time_index, int(label_id) % int(dim)] += 1.0

        return features

    def _split_seed_offset(self) -> int:
        return {
            "train": 0,
            "val": 1000,
            "test": 2000,
        }[self.split]


def build_mmgcn_smoke_dataloader(
    dataset_config: Dict[str, Any],
    model_config: Dict[str, Any],
    train_config: Dict[str, Any],
    split: str,
    shuffle: bool,
    batch_size: Optional[int] = None,
    pin_memory: bool = False,
) -> DataLoader:
    """Build a DataLoader for the deterministic MMGCN smoke dataset."""
    dataset = MMGCNSmokeDialogueDataset(
        split=split,
        num_classes=int(dataset_config["num_classes"]),
        label_list=dataset_config.get("label_list"),
        text_dim=int(model_config["text_feature_dim"]),
        audio_dim=int(model_config["audio_feature_dim"]),
        visual_dim=int(model_config["visual_feature_dim"]),
        split_sizes=dataset_config.get("split_sizes"),
        dialogue_lengths=dataset_config.get("dialogue_lengths"),
        seed=int(dataset_config.get("seed", 1234)),
    )

    return DataLoader(
        dataset,
        batch_size=int(batch_size or train_config["batch_size"]),
        shuffle=bool(shuffle),
        num_workers=int(train_config.get("num_workers", 0)),
        collate_fn=m3ed_dialogue_collate_fn,
        pin_memory=bool(pin_memory),
    )

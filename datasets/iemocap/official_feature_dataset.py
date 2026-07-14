"""
IEMOCAP official feature dataset adapter.

This adapter reads the official MMGCN IEMOCAP feature pkl and converts it
to the dialogue-batch interface used by this project.

Official pkl structure:
    [
        videoIDs,
        videoSpeakers,
        videoLabels,
        videoText,
        videoAudio,
        videoVisual,
        videoSentence,
        trainVid,
        testVid,
    ]

Official feature names:
    videoText   -> text_features
    videoAudio  -> audio_features
    videoVisual -> visual_features

Project batch format:
    text_features:    [B, T, D_text]  # legacy=100, clean RoBERTa v1=768
    audio_features:   [B, T, 1582]
    visual_features:  [B, T, 342]
    labels:           [B, T]
    attention_mask:   [B, T]
    lengths:          [B]
    speaker_ids_int:  [B, T]
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import pickle
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .split_utils import (
    IEMOCAP_TEST_SESSION_ID,
    IEMOCAP_TRAIN_SESSION_IDS,
    parse_iemocap_session_id,
)


IGNORE_INDEX = -100


class IEMOCAPOfficialFeatureDataset(Dataset):
    """
    Dialogue-level IEMOCAP dataset based on official MMGCN features.

    Splits:
        train:
            official trainVid after removing the validation prefix.

        val / valid:
            first valid_ratio portion of official trainVid.

        test:
            official testVid.

    Validation strategies:
        official_prefix:
            Preserve the official prefix sampler behavior.
        random:
            Seeded dialogue-level split of official trainVid.
        session_holdout:
            Hold out all trainVid dialogues from ``val_session_id`` and keep
            the other three training sessions for train. Official testVid is
            never changed.

    By default, the split follows the official loader's valid sampler logic:
        val   = trainVid[:int(valid_ratio * len(trainVid))]
        train = trainVid[int(valid_ratio * len(trainVid)):]
    """

    DEFAULT_LABEL_LIST = [
        "Happy",
        "Sad",
        "Neutral",
        "Angry",
        "Excited",
        "Frustrated",
    ]

    SPEAKER_TO_ID = {
        "M": 0,
        "F": 1,
    }

    def __init__(
        self,
        feature_pkl_path: Union[str, Path],
        split: str,
        valid_ratio: float = 0.1,
        val_split_strategy: str = "official_prefix",
        val_session_id: Optional[str] = None,
        seed: int = 42,
    ) -> None:
        super().__init__()

        self.feature_pkl_path = Path(feature_pkl_path)
        self.split = self._normalize_split(split)
        self.valid_ratio = float(valid_ratio)
        self.val_split_strategy = str(val_split_strategy)
        self.val_session_id = (
            None if val_session_id is None else str(val_session_id).strip()
        )
        self.seed = int(seed)

        if not self.feature_pkl_path.exists():
            raise FileNotFoundError(
                f"IEMOCAP feature pkl not found: {self.feature_pkl_path}"
            )

        with open(self.feature_pkl_path, "rb") as f:
            data = pickle.load(f, encoding="latin1")

        if not isinstance(data, (list, tuple)) or len(data) != 9:
            raise ValueError(
                "Expected IEMOCAP official feature pkl to be a list/tuple "
                f"of length 9, got type={type(data)}, len={len(data) if hasattr(data, '__len__') else 'NA'}"
            )

        (
            self.videoIDs,
            self.videoSpeakers,
            self.videoLabels,
            self.videoText,
            self.videoAudio,
            self.videoVisual,
            self.videoSentence,
            self.trainVid,
            self.testVid,
        ) = data

        self.train_ids, self.val_ids, self.test_ids = self._build_split_ids()

        if self.split == "train":
            self.keys = list(self.train_ids)
        elif self.split == "val":
            self.keys = list(self.val_ids)
        elif self.split == "test":
            self.keys = list(self.test_ids)
        else:
            raise ValueError(f"Unsupported split: {self.split}")

        if len(self.keys) == 0:
            raise ValueError(
                f"IEMOCAP split={self.split} has 0 dialogues. "
                f"valid_ratio={self.valid_ratio}, strategy={self.val_split_strategy}"
            )

        self.text_feature_dim, self.audio_feature_dim, self.visual_feature_dim = (
            self._infer_feature_dims()
        )

    @staticmethod
    def _normalize_split(split: str) -> str:
        split = str(split).lower()

        if split in {"valid", "validation", "dev"}:
            return "val"

        if split in {"train", "val", "test"}:
            return split

        raise ValueError(
            f"Unsupported split={split}. Use one of train / val / test."
        )

    def _build_split_ids(self) -> Tuple[List[str], List[str], List[str]]:
        train_vid = list(self.trainVid)
        test_vid = list(self.testVid)

        strategy = self.val_split_strategy.lower()

        if strategy == "session_holdout":
            if self.val_session_id not in IEMOCAP_TRAIN_SESSION_IDS:
                supported = ", ".join(IEMOCAP_TRAIN_SESSION_IDS)
                raise ValueError(
                    "val_session_id must be one of "
                    f"{supported} for session_holdout; got {self.val_session_id!r}."
                )

            train_ids: List[str] = []
            val_ids: List[str] = []
            for dialogue_id in train_vid:
                session_id = parse_iemocap_session_id(dialogue_id)
                if session_id not in IEMOCAP_TRAIN_SESSION_IDS:
                    raise ValueError(
                        "Official trainVid contains a dialogue outside Ses01-Ses04: "
                        f"{dialogue_id!r} ({session_id})."
                    )
                if session_id == self.val_session_id:
                    val_ids.append(dialogue_id)
                else:
                    train_ids.append(dialogue_id)

            for dialogue_id in test_vid:
                session_id = parse_iemocap_session_id(dialogue_id)
                if session_id != IEMOCAP_TEST_SESSION_ID:
                    raise ValueError(
                        "Official testVid contains a dialogue outside Ses05: "
                        f"{dialogue_id!r} ({session_id})."
                    )

            overlap = set(train_vid) & set(test_vid)
            if overlap:
                raise ValueError(
                    "Official trainVid/testVid dialogue IDs overlap: "
                    f"{sorted(overlap)}"
                )

            if not train_ids or not val_ids:
                raise ValueError(
                    "session_holdout produced an empty train or validation split: "
                    f"val_session_id={self.val_session_id!r}, "
                    f"train={len(train_ids)}, val={len(val_ids)}."
                )
            return train_ids, val_ids, test_vid

        if self.valid_ratio <= 0:
            return train_vid, [], test_vid

        if self.valid_ratio >= 1:
            raise ValueError("valid_ratio must be in [0, 1).")

        n_val = int(self.valid_ratio * len(train_vid))

        if n_val <= 0:
            return train_vid, [], test_vid

        if strategy == "official_prefix":
            # Matches official get_train_valid_sampler:
            # valid uses idx[:split], train uses idx[split:].
            val_ids = train_vid[:n_val]
            train_ids = train_vid[n_val:]

        elif strategy == "random":
            rng = random.Random(self.seed)
            shuffled = list(train_vid)
            rng.shuffle(shuffled)
            val_ids = shuffled[:n_val]
            train_ids = shuffled[n_val:]

        else:
            raise ValueError(
                "Unsupported val_split_strategy: "
                f"{self.val_split_strategy}. Use official_prefix, random, or "
                "session_holdout."
            )

        return train_ids, val_ids, test_vid

    def _infer_feature_dims(self) -> Tuple[int, int, int]:
        first_id = self.keys[0]

        text = np.asarray(self.videoText[first_id], dtype=np.float32)
        audio = np.asarray(self.videoAudio[first_id], dtype=np.float32)
        visual = np.asarray(self.videoVisual[first_id], dtype=np.float32)

        if text.ndim != 2 or audio.ndim != 2 or visual.ndim != 2:
            raise ValueError(
                "Expected text/audio/visual features to be 2D arrays "
                f"for dialogue {first_id}, got "
                f"text={text.shape}, audio={audio.shape}, visual={visual.shape}"
            )

        return int(text.shape[1]), int(audio.shape[1]), int(visual.shape[1])

    def __len__(self) -> int:
        return len(self.keys)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        dialogue_id = self.keys[index]

        utterance_ids = list(self.videoIDs[dialogue_id])
        speakers = list(self.videoSpeakers[dialogue_id])
        labels = list(self.videoLabels[dialogue_id])
        sentences = list(self.videoSentence[dialogue_id])

        text_features = torch.tensor(
            np.asarray(self.videoText[dialogue_id], dtype=np.float32),
            dtype=torch.float32,
        )
        audio_features = torch.tensor(
            np.asarray(self.videoAudio[dialogue_id], dtype=np.float32),
            dtype=torch.float32,
        )
        visual_features = torch.tensor(
            np.asarray(self.videoVisual[dialogue_id], dtype=np.float32),
            dtype=torch.float32,
        )

        labels_tensor = torch.tensor(labels, dtype=torch.long)

        speaker_ids = [
            self.SPEAKER_TO_ID.get(str(speaker), 0)
            for speaker in speakers
        ]
        speaker_ids_int = torch.tensor(speaker_ids, dtype=torch.long)

        length = int(labels_tensor.shape[0])

        self._validate_item_shapes(
            dialogue_id=dialogue_id,
            length=length,
            text_features=text_features,
            audio_features=audio_features,
            visual_features=visual_features,
            speaker_ids_int=speaker_ids_int,
            utterance_ids=utterance_ids,
            sentences=sentences,
        )

        return {
            "dialogue_id": dialogue_id,
            "utterance_ids": utterance_ids,
            "sentences": sentences,
            "text_features": text_features,
            "audio_features": audio_features,
            "visual_features": visual_features,
            "labels": labels_tensor,
            "speaker_ids_int": speaker_ids_int,
            "length": length,
        }

    @staticmethod
    def _validate_item_shapes(
        dialogue_id: str,
        length: int,
        text_features: torch.Tensor,
        audio_features: torch.Tensor,
        visual_features: torch.Tensor,
        speaker_ids_int: torch.Tensor,
        utterance_ids: Sequence[str],
        sentences: Sequence[str],
    ) -> None:
        values = {
            "text_features": text_features.shape[0],
            "audio_features": audio_features.shape[0],
            "visual_features": visual_features.shape[0],
            "speaker_ids_int": speaker_ids_int.shape[0],
            "utterance_ids": len(utterance_ids),
            "sentences": len(sentences),
        }

        bad = {
            name: value
            for name, value in values.items()
            if int(value) != int(length)
        }

        if bad:
            raise ValueError(
                f"Length mismatch in dialogue {dialogue_id}: "
                f"labels length={length}, others={bad}"
            )

    def get_feature_dims(self) -> Dict[str, int]:
        return {
            "text_feature_dim": self.text_feature_dim,
            "audio_feature_dim": self.audio_feature_dim,
            "visual_feature_dim": self.visual_feature_dim,
        }

    def return_labels(self) -> List[int]:
        labels: List[int] = []

        for dialogue_id in self.keys:
            labels.extend([int(x) for x in self.videoLabels[dialogue_id]])

        return labels

    def get_split_ids(self) -> Dict[str, List[str]]:
        return {
            "train": list(self.train_ids),
            "val": list(self.val_ids),
            "test": list(self.test_ids),
        }


def iemocap_dialogue_collate_fn(
    batch: List[Dict[str, Any]],
    ignore_index: int = IGNORE_INDEX,
) -> Dict[str, Any]:
    if len(batch) == 0:
        raise ValueError("Empty batch received by iemocap_dialogue_collate_fn.")

    batch_size = len(batch)
    lengths = torch.tensor(
        [int(item["length"]) for item in batch],
        dtype=torch.long,
    )
    max_len = int(lengths.max().item())

    text_dim = int(batch[0]["text_features"].shape[-1])
    audio_dim = int(batch[0]["audio_features"].shape[-1])
    visual_dim = int(batch[0]["visual_features"].shape[-1])

    text_features = torch.zeros(
        (batch_size, max_len, text_dim),
        dtype=torch.float32,
    )
    audio_features = torch.zeros(
        (batch_size, max_len, audio_dim),
        dtype=torch.float32,
    )
    visual_features = torch.zeros(
        (batch_size, max_len, visual_dim),
        dtype=torch.float32,
    )

    labels = torch.full(
        (batch_size, max_len),
        fill_value=int(ignore_index),
        dtype=torch.long,
    )
    attention_mask = torch.zeros(
        (batch_size, max_len),
        dtype=torch.long,
    )
    speaker_ids_int = torch.zeros(
        (batch_size, max_len),
        dtype=torch.long,
    )

    dialogue_ids: List[str] = []
    utterance_ids: List[List[str]] = []
    sentences: List[List[str]] = []

    for batch_index, item in enumerate(batch):
        length = int(item["length"])

        text_features[batch_index, :length] = item["text_features"]
        audio_features[batch_index, :length] = item["audio_features"]
        visual_features[batch_index, :length] = item["visual_features"]

        labels[batch_index, :length] = item["labels"]
        attention_mask[batch_index, :length] = 1
        speaker_ids_int[batch_index, :length] = item["speaker_ids_int"]

        dialogue_ids.append(str(item["dialogue_id"]))
        utterance_ids.append(list(item["utterance_ids"]))
        sentences.append(list(item["sentences"]))

    return {
        "dialogue_ids": dialogue_ids,
        "utterance_ids": utterance_ids,
        "sentences": sentences,
        "text_features": text_features,
        "audio_features": audio_features,
        "visual_features": visual_features,
        "labels": labels,
        "attention_mask": attention_mask,
        "lengths": lengths,
        "speaker_ids_int": speaker_ids_int,
    }


def build_iemocap_dataloader(
    feature_pkl_path: Union[str, Path],
    split: str,
    batch_size: int,
    valid_ratio: float = 0.1,
    val_split_strategy: str = "official_prefix",
    val_session_id: Optional[str] = None,
    seed: int = 42,
    shuffle: Optional[bool] = None,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> DataLoader:
    dataset = IEMOCAPOfficialFeatureDataset(
        feature_pkl_path=feature_pkl_path,
        split=split,
        valid_ratio=valid_ratio,
        val_split_strategy=val_split_strategy,
        val_session_id=val_session_id,
        seed=seed,
    )

    normalized_split = IEMOCAPOfficialFeatureDataset._normalize_split(split)

    if shuffle is None:
        shuffle = normalized_split == "train"

    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        num_workers=int(num_workers),
        collate_fn=iemocap_dialogue_collate_fn,
        pin_memory=bool(pin_memory),
    )

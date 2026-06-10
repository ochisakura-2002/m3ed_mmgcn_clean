"""
M3ED PyTorch dataset.

这个文件把 M3EDFeatureDataset 包装成 torch.utils.data.Dataset。

M3EDFeatureDataset 负责：
    metadata + feature pkl 对齐

M3EDTorchDataset 负责：
    把 numpy array / list 转成 torch Tensor，供 DataLoader 使用

当前每个 item 仍然是一个 dialogue。
不同 dialogue 的 utterance 数量不同，所以 batch 拼接需要 collate_fn 负责 padding。
"""

from typing import Any, Dict

import torch
from torch.utils.data import Dataset

from datasets.m3ed.feature_dataset import M3EDFeatureDataset


class M3EDTorchDataset(Dataset):
    """
    M3ED dialogue-level PyTorch Dataset.

    每个样本是一个 dialogue，包含变长 utterance 序列。
    """

    def __init__(
        self,
        feature_pkl_path: str,
        metadata_path: str = "data/metadata/m3ed_metadata.csv",
        label_mapping_path: str = "data/metadata/m3ed_label_mapping.csv",
        split: str = "train",
        check_label_consistency: bool = True,
        raise_on_label_mismatch: bool = False,
    ) -> None:
        """
        初始化 PyTorch dataset。

        参数：
            feature_pkl_path:
                M3ED feature pkl 路径。
            metadata_path:
                m3ed_metadata.csv 路径。
            label_mapping_path:
                m3ed_label_mapping.csv 路径。
            split:
                train / val / test。
            check_label_consistency:
                是否检查 metadata label 和 feature pkl label。
            raise_on_label_mismatch:
                是否在 label mismatch 时直接报错。
        """
        self.feature_dataset = M3EDFeatureDataset(
            feature_pkl_path=feature_pkl_path,
            metadata_path=metadata_path,
            label_mapping_path=label_mapping_path,
            split=split,
            check_label_consistency=check_label_consistency,
            raise_on_label_mismatch=raise_on_label_mismatch,
        )

    def __len__(self) -> int:
        """
        返回 dialogue 数量。
        """
        return len(self.feature_dataset)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        """
        返回一个 dialogue 样本。

        Tensor 字段：
            text_features   : [num_utterances, text_dim]
            audio_features  : [num_utterances, audio_dim]
            visual_features : [num_utterances, visual_dim]
            labels          : [num_utterances]
            speaker_ids_int : [num_utterances]

        非 Tensor 字段保留为 list / str，方便后续分析和保存预测结果。
        """
        sample = self.feature_dataset[index]

        speaker_ids_int = self._convert_speaker_ids_to_int(sample["speaker_ids"])

        torch_sample = {
            "movie_id": sample["movie_id"],
            "dialogue_id": sample["dialogue_id"],
            "split": sample["split"],
            "num_utterances": sample["num_utterances"],

            "utterance_ids": sample["utterance_ids"],
            "utterance_indices": sample["utterance_indices"],
            "texts": sample["texts"],
            "speaker_ids": sample["speaker_ids"],
            "speaker_names": sample["speaker_names"],
            "speaker_genders": sample["speaker_genders"],
            "label_names": sample["label_names"],

            "text_features": torch.as_tensor(
                sample["text_features"],
                dtype=torch.float32,
            ),
            "audio_features": torch.as_tensor(
                sample["audio_features"],
                dtype=torch.float32,
            ),
            "visual_features": torch.as_tensor(
                sample["visual_features"],
                dtype=torch.float32,
            ),
            "labels": torch.as_tensor(
                sample["labels"],
                dtype=torch.long,
            ),
            "speaker_ids_int": torch.as_tensor(
                speaker_ids_int,
                dtype=torch.long,
            ),

            "has_label_mismatch": sample["has_label_mismatch"],
            "label_mismatch_positions": sample["label_mismatch_positions"],
        }

        return torch_sample

    @staticmethod
    def _convert_speaker_ids_to_int(speaker_ids):
        """
        把 speaker_id 转成整数。

        M3ED 中通常是 A / B 两个说话人：
            A -> 0
            B -> 1
            其他或缺失 -> -1

        后续构建 speaker-aware graph 时会用到。
        """
        mapping = {
            "A": 0,
            "B": 1,
        }

        return [
            mapping.get(str(speaker_id), -1)
            for speaker_id in speaker_ids
        ]

    def feature_dimensions(self) -> Dict[str, int]:
        """
        返回三模态特征维度。
        """
        return self.feature_dataset.feature_dimensions()

    def split_summary(self) -> Dict[str, Any]:
        """
        返回当前 split 的统计信息。
        """
        return self.feature_dataset.split_summary()
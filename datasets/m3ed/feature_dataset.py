"""
M3ED feature dataset.

这个文件把 M3ED metadata 和官方 DialogueRNN-style feature pkl 对齐。

核心职责：
1. 读取 m3ed_metadata.csv
2. 读取 M3ED 官方 feature pkl
3. 修正官方 feature pkl 中 val/test 字段和 metadata split 不一致的问题
4. 使用 metadata 中的 label_id 作为训练标签
5. 将 text/audio/visual 三模态特征组织成 dialogue-level 样本

重要说明：
M3ED 官方 feature pkl 中自带 videoLabels。
但我们诊断发现，metadata labels 和 feature labels 只有极少数不一致：
    2 / 24449 utterances
因此训练时以 metadata labels 为准。
feature labels 只作为一致性检查参考，不作为最终训练标签。
"""

from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from datasets.m3ed.metadata_dataset import M3EDDialogueDataset
from datasets.common.dialogue_feature_dataset import DialogueFeatureDataset


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class M3EDFeatureDataset:
    """
    M3ED dialogue-level feature dataset.

    每个 item 对应一个 dialogue，包含：
    1. metadata 中的 movie_id、dialogue_id、utterance_id、text、speaker、label
    2. feature pkl 中的 text/audio/visual 三模态特征

    返回样本示例：
        {
            "movie_id": ...,
            "dialogue_id": ...,
            "split": ...,
            "num_utterances": ...,
            "utterance_ids": [...],
            "texts": [...],
            "speaker_ids": [...],
            "labels": [...],
            "label_names": [...],
            "text_features": np.ndarray,
            "audio_features": np.ndarray,
            "visual_features": np.ndarray,
            "has_label_mismatch": bool,
            "label_mismatch_positions": [...]
        }
    """

    # 根据 scripts/data/inspect/inspect_m3ed_feature_files.py 的输出确定：
    # metadata 的 val split 对应 feature pkl 里的 generic test split；
    # metadata 的 test split 对应 feature pkl 里的 generic val split。
    #
    # 这不是优雅设计，这是数据集现实。人类数据工程经常像盲盒。
    FEATURE_SPLIT_MAP = {
        "train": "train",
        "val": "test",
        "test": "val",
    }

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
        初始化 M3ED feature dataset。

        参数：
            feature_pkl_path:
                M3ED 官方 feature pkl 路径。
            metadata_path:
                m3ed_metadata.csv 路径。
            label_mapping_path:
                m3ed_label_mapping.csv 路径。
            split:
                train / val / test。
            check_label_consistency:
                是否检查 metadata labels 和 feature pkl labels 是否一致。
            raise_on_label_mismatch:
                如果为 True，发现 label mismatch 时直接报错。
                如果为 False，只打印 warning，并继续使用 metadata labels。

        当前推荐：
            check_label_consistency=True
            raise_on_label_mismatch=False

        原因：
            我们已经诊断过，只有 2 / 24449 个 utterances 的标签不一致。
            训练标签应该以 annotation.json 展平得到的 metadata 为准。
        """
        if split not in ["train", "val", "test"]:
            raise ValueError(f"Unsupported split: {split}")

        self.feature_pkl_path = feature_pkl_path
        self.metadata_path = metadata_path
        self.label_mapping_path = label_mapping_path
        self.split = split
        self.check_label_consistency = check_label_consistency
        self.raise_on_label_mismatch = raise_on_label_mismatch

        # 用于避免同一个 dialogue 的 label mismatch warning 被重复打印。
        self._warned_label_mismatch_dialogues = set()

        # metadata dataset 使用我们从 annotation.json 生成的 m3ed_metadata.csv。
        self.metadata_dataset = M3EDDialogueDataset(
            metadata_path=metadata_path,
            label_mapping_path=label_mapping_path,
            split=split,
        )

        # feature pkl 的 split 字段和 metadata split 不完全一致，
        # 所以这里用 FEATURE_SPLIT_MAP 做修正。
        feature_split = self.FEATURE_SPLIT_MAP[split]

        self.feature_dataset = DialogueFeatureDataset(
            feature_pkl_path=feature_pkl_path,
            split=feature_split,
        )

        self.metadata_dialogue_ids = [
            self.metadata_dataset[index]["dialogue_id"]
            for index in range(len(self.metadata_dataset))
        ]

        self.feature_index_map = self._build_feature_index_map()

        self._check_dialogue_alignment()

    def _build_feature_index_map(self) -> Dict[str, int]:
        """
        建立 dialogue_id -> feature_dataset index 的映射。

        这样后续可以根据 metadata 中的 dialogue_id 快速找到对应 feature。
        """
        feature_index_map = {}

        for index, dialogue_id in enumerate(self.feature_dataset.dialogue_ids):
            dialogue_id = str(dialogue_id)

            if dialogue_id in feature_index_map:
                raise ValueError(f"Duplicated dialogue_id in feature pkl: {dialogue_id}")

            feature_index_map[dialogue_id] = index

        return feature_index_map

    def _check_dialogue_alignment(self) -> None:
        """
        检查当前 split 下 metadata 和 feature pkl 的 dialogue_id 是否完全一致。

        如果这里失败，说明数据层还没对齐，不能继续训练。
        """
        metadata_ids = set(str(x) for x in self.metadata_dialogue_ids)
        feature_ids = set(self.feature_index_map.keys())

        missing_in_feature = metadata_ids - feature_ids
        extra_in_feature = feature_ids - metadata_ids

        if len(missing_in_feature) > 0 or len(extra_in_feature) > 0:
            raise ValueError(
                "Dialogue id mismatch between metadata and feature pkl.\n"
                f"split={self.split}\n"
                f"metadata dialogues={len(metadata_ids)}\n"
                f"feature dialogues={len(feature_ids)}\n"
                f"missing in feature={len(missing_in_feature)}\n"
                f"extra in feature={len(extra_in_feature)}\n"
                f"first missing={sorted(list(missing_in_feature))[:5]}\n"
                f"first extra={sorted(list(extra_in_feature))[:5]}"
            )

    def __len__(self) -> int:
        """
        返回当前 split 的 dialogue 数量。
        """
        return len(self.metadata_dataset)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        """
        返回一个 dialogue 样本。

        注意：
            最终训练标签使用 metadata_sample["labels"]。
            feature_sample["labels"] 只用于一致性检查。
        """
        metadata_sample = self.metadata_dataset[index]
        dialogue_id = metadata_sample["dialogue_id"]

        feature_index = self.feature_index_map[str(dialogue_id)]
        feature_sample = self.feature_dataset[feature_index]

        if metadata_sample["num_utterances"] != feature_sample["num_utterances"]:
            raise ValueError(
                f"Utterance number mismatch for dialogue {dialogue_id}: "
                f"metadata={metadata_sample['num_utterances']}, "
                f"feature={feature_sample['num_utterances']}"
            )

        metadata_labels = [int(x) for x in metadata_sample["labels"]]

        feature_labels = [
            int(x)
            for x in np.asarray(feature_sample["labels"]).tolist()
        ]

        label_mismatch_positions = []

        if self.check_label_consistency and metadata_labels != feature_labels:
            label_mismatch_positions = [
                position
                for position, (metadata_label, feature_label) in enumerate(
                    zip(metadata_labels, feature_labels)
                )
                if metadata_label != feature_label
            ]

            message = (
                f"Label mismatch for dialogue {dialogue_id}. "
                f"mismatch_positions={label_mismatch_positions[:10]}, "
                f"num_mismatches={len(label_mismatch_positions)}. "
                "Use metadata labels as training labels."
            )

            if self.raise_on_label_mismatch:
                raise ValueError(
                    message
                    + "\n"
                    + f"metadata labels first 20={metadata_labels[:20]}\n"
                    + f"feature labels first 20={feature_labels[:20]}"
                )

            if dialogue_id not in self._warned_label_mismatch_dialogues:
                print("[WARN]", message)
                self._warned_label_mismatch_dialogues.add(dialogue_id)

        sample = {
            "movie_id": metadata_sample["movie_id"],
            "dialogue_id": dialogue_id,
            "split": self.split,
            "num_utterances": metadata_sample["num_utterances"],

            # utterance-level metadata
            "utterance_ids": metadata_sample["utterance_ids"],
            "utterance_indices": metadata_sample["utterance_indices"],
            "texts": metadata_sample["texts"],
            "sentences": feature_sample["sentences"],

            # speaker information
            "speaker_ids": metadata_sample["speaker_ids"],
            "speaker_names": metadata_sample["speaker_names"],
            "speaker_genders": metadata_sample["speaker_genders"],
            "feature_speakers": feature_sample["speakers"],

            # labels
            # 训练时只使用 metadata labels。
            "labels": metadata_labels,
            "label_names": metadata_sample["label_names"],

            # multimodal features
            "text_features": np.asarray(
                feature_sample["text_features"],
                dtype=np.float32,
            ),
            "audio_features": np.asarray(
                feature_sample["audio_features"],
                dtype=np.float32,
            ),
            "visual_features": np.asarray(
                feature_sample["visual_features"],
                dtype=np.float32,
            ),

            # diagnostic information
            "has_label_mismatch": len(label_mismatch_positions) > 0,
            "label_mismatch_positions": label_mismatch_positions,
        }

        return sample

    def num_utterances(self) -> int:
        """
        返回当前 split 的 utterance 总数。
        """
        total = 0

        for index in range(len(self)):
            total += self[index]["num_utterances"]

        return int(total)

    def feature_dimensions(self) -> Dict[str, int]:
        """
        返回三模态特征维度。
        """
        sample = self[0]

        return {
            "text_dim": int(sample["text_features"].shape[-1]),
            "audio_dim": int(sample["audio_features"].shape[-1]),
            "visual_dim": int(sample["visual_features"].shape[-1]),
        }

    def split_summary(self) -> Dict[str, Any]:
        """
        返回当前 split 的基础统计。
        """
        lengths = []
        mismatch_dialogues = 0
        mismatch_utterances = 0

        for index in range(len(self)):
            sample = self[index]
            lengths.append(sample["num_utterances"])

            if sample["has_label_mismatch"]:
                mismatch_dialogues += 1
                mismatch_utterances += len(sample["label_mismatch_positions"])

        if len(lengths) == 0:
            return {
                "split": self.split,
                "num_dialogues": 0,
                "num_utterances": 0,
                "min_len": None,
                "max_len": None,
                "mean_len": None,
                "mismatch_dialogues": 0,
                "mismatch_utterances": 0,
            }

        return {
            "split": self.split,
            "num_dialogues": int(len(lengths)),
            "num_utterances": int(sum(lengths)),
            "min_len": int(min(lengths)),
            "max_len": int(max(lengths)),
            "mean_len": float(np.mean(lengths)),
            "mismatch_dialogues": int(mismatch_dialogues),
            "mismatch_utterances": int(mismatch_utterances),
        }

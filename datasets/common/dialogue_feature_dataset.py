"""
DialogueRNN-style feature dataset.

这个文件负责读取 DialogueRNN / MMGCN 常见的 pkl 特征格式。

常见 pkl 顶层结构是 list 或 tuple：

9 字段：
    videoIDs
    videoSpeakers
    videoLabels
    videoText
    videoAudio
    videoVisual
    videoSentence
    trainVid
    testVid

10 字段：
    videoIDs
    videoSpeakers
    videoLabels
    videoText
    videoAudio
    videoVisual
    videoSentence
    trainVid
    testVid
    valVid

当前文件不依赖 torch。
后续 M3ED feature pkl 下载后，如果格式一致，可以复用这个读取器。
"""

from pathlib import Path
from typing import Any, Dict, List, Optional
import pickle

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]


FIELD_NAMES_9 = [
    "videoIDs",
    "videoSpeakers",
    "videoLabels",
    "videoText",
    "videoAudio",
    "videoVisual",
    "videoSentence",
    "trainVid",
    "testVid",
]

FIELD_NAMES_10 = [
    "videoIDs",
    "videoSpeakers",
    "videoLabels",
    "videoText",
    "videoAudio",
    "videoVisual",
    "videoSentence",
    "trainVid",
    "testVid",
    "valVid",
]


class DialogueFeatureDataset:
    """
    Dialogue-level feature dataset.

    每个 item 是一个 dialogue。
    """

    def __init__(
        self,
        feature_pkl_path: str,
        split: str = "train",
    ) -> None:
        """
        初始化 feature dataset。

        参数：
            feature_pkl_path:
                pkl 文件路径，可以是项目根目录下的相对路径，也可以是绝对路径。
            split:
                train / val / test。
                如果 pkl 不提供 valVid，则 val 会报错。
        """
        self.feature_pkl_path = self._resolve_path(feature_pkl_path)
        self.split = split

        self.raw_data = self._load_pickle(self.feature_pkl_path)
        self.field_map = self._parse_feature_pkl(self.raw_data)
        self.dialogue_ids = self._select_dialogue_ids(split)

    @staticmethod
    def _resolve_path(path_str: str) -> Path:
        """把相对路径解析成项目绝对路径。"""
        path = Path(path_str)
        if path.is_absolute():
            return path
        return PROJECT_ROOT / path

    @staticmethod
    def _load_pickle(path: Path) -> Any:
        """
        读取 pkl 文件。

        某些旧论文代码保存的 pkl 可能需要 latin1 兼容。
        """
        if not path.exists():
            raise FileNotFoundError(f"Feature pkl not found: {path}")

        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except UnicodeDecodeError:
            with open(path, "rb") as f:
                return pickle.load(f, encoding="latin1")

    @staticmethod
    def _parse_feature_pkl(data: Any) -> Dict[str, Any]:
        """
        解析 DialogueRNN-style pkl 顶层结构。
        """
        if not isinstance(data, (list, tuple)):
            raise TypeError(
                f"Expected top-level list or tuple, but got {type(data).__name__}"
            )

        if len(data) == 9:
            field_names = FIELD_NAMES_9
        elif len(data) == 10:
            field_names = FIELD_NAMES_10
        else:
            raise ValueError(
                f"Unsupported pkl field number: {len(data)}. "
                "Expected 9 or 10 fields."
            )

        return {
            field_name: data[index]
            for index, field_name in enumerate(field_names)
        }

    def _select_dialogue_ids(self, split: str) -> List[Any]:
        """
        根据 split 选择 dialogue id。
        """
        if split == "train":
            split_ids = self.field_map.get("trainVid")
        elif split == "test":
            split_ids = self.field_map.get("testVid")
        elif split == "val":
            split_ids = self.field_map.get("valVid")
        else:
            raise ValueError(f"Unsupported split: {split}")

        if split_ids is None:
            raise ValueError(
                f"Split '{split}' is not available in this pkl file."
            )

        return list(split_ids)

    def __len__(self) -> int:
        """返回 dialogue 数量。"""
        return len(self.dialogue_ids)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        """
        返回一个 dialogue 样本。
        """
        dialogue_id = self.dialogue_ids[index]

        sample = {
            "dialogue_id": dialogue_id,
            "speakers": self._get_dialogue_field("videoSpeakers", dialogue_id),
            "labels": self._get_dialogue_field("videoLabels", dialogue_id),
            "text_features": self._get_dialogue_field("videoText", dialogue_id),
            "audio_features": self._get_dialogue_field("videoAudio", dialogue_id),
            "visual_features": self._get_dialogue_field("videoVisual", dialogue_id),
            "sentences": self._get_dialogue_field("videoSentence", dialogue_id),
        }

        sample["num_utterances"] = self._infer_num_utterances(sample)

        return sample

    def _get_dialogue_field(self, field_name: str, dialogue_id: Any) -> Any:
        """
        从某个字段里取指定 dialogue 的内容。
        """
        field_value = self.field_map.get(field_name)

        if field_value is None:
            return None

        if not isinstance(field_value, dict):
            raise TypeError(
                f"Field {field_name} is expected to be dict, "
                f"but got {type(field_value).__name__}"
            )

        if dialogue_id not in field_value:
            raise KeyError(
                f"Dialogue id {dialogue_id} not found in field {field_name}"
            )

        return field_value[dialogue_id]

    @staticmethod
    def _infer_num_utterances(sample: Dict[str, Any]) -> int:
        """
        推断当前 dialogue 的 utterance 数量。

        优先使用 labels 的长度。
        """
        labels = sample.get("labels")

        if labels is not None:
            return int(len(labels))

        for key in ["text_features", "audio_features", "visual_features", "sentences"]:
            value = sample.get(key)
            if value is not None:
                return int(len(value))

        return 0

    def feature_dimensions(self) -> Dict[str, Optional[int]]:
        """
        推断 text/audio/visual 的特征维度。

        返回：
            {
                "text_dim": ...,
                "audio_dim": ...,
                "visual_dim": ...
            }
        """
        if len(self) == 0:
            return {
                "text_dim": None,
                "audio_dim": None,
                "visual_dim": None,
            }

        sample = self[0]

        return {
            "text_dim": self._infer_feature_dim(sample["text_features"]),
            "audio_dim": self._infer_feature_dim(sample["audio_features"]),
            "visual_dim": self._infer_feature_dim(sample["visual_features"]),
        }

    @staticmethod
    def _infer_feature_dim(feature_value: Any) -> Optional[int]:
        """
        从一个 dialogue 的特征中推断单个 utterance 的特征维度。
        """
        if feature_value is None:
            return None

        array = np.asarray(feature_value)

        if array.ndim == 1:
            return int(array.shape[0])

        if array.ndim >= 2:
            return int(array.shape[-1])

        return None

    def split_summary(self) -> Dict[str, Any]:
        """
        返回当前 split 的基础摘要。
        """
        lengths = []

        for index in range(len(self)):
            sample = self[index]
            lengths.append(sample["num_utterances"])

        if len(lengths) == 0:
            return {
                "split": self.split,
                "num_dialogues": 0,
                "num_utterances": 0,
                "min_len": None,
                "max_len": None,
                "mean_len": None,
            }

        return {
            "split": self.split,
            "num_dialogues": int(len(lengths)),
            "num_utterances": int(sum(lengths)),
            "min_len": int(min(lengths)),
            "max_len": int(max(lengths)),
            "mean_len": float(np.mean(lengths)),
        }
"""
M3ED dialogue-level dataset.

当前版本只读取 metadata，不读取音频、视觉、文本特征。
它的目标是把 utterance-level metadata 组织成 dialogue-level 样本。

后续 MMGCN 需要构建对话图，所以这里每个样本对应一个 dialogue，
而不是单个 utterance。

当前不依赖 torch。
等后续安装 torch 后，可以再把它改造成 torch.utils.data.Dataset 的子类。
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class M3EDDialogueDataset:
    """
    M3ED dialogue-level dataset.

    每个 item 是一个 dialogue，包含多个 utterances。

    返回样例：
        {
            "movie_id": ...,
            "dialogue_id": ...,
            "split": ...,
            "num_utterances": ...,
            "utterance_ids": [...],
            "texts": [...],
            "speaker_ids": [...],
            "labels": [...],
            "label_names": [...]
        }
    """

    required_columns = [
        "movie_id",
        "dialogue_id",
        "utterance_id",
        "utterance_index",
        "split",
        "speaker_id",
        "text",
        "final_main_emo",
        "label_id",
    ]

    def __init__(
        self,
        metadata_path: str = "data/metadata/m3ed_metadata.csv",
        label_mapping_path: str = "data/metadata/m3ed_label_mapping.csv",
        split: Optional[str] = None,
    ) -> None:
        """
        初始化 dataset。

        参数：
            metadata_path:
                metadata csv 路径。可以是相对项目根目录的路径。
            label_mapping_path:
                label mapping csv 路径。
            split:
                指定 train / val / test。
                如果为 None，则读取全部 split。
        """
        self.metadata_path = self._resolve_path(metadata_path)
        self.label_mapping_path = self._resolve_path(label_mapping_path)
        self.split = split

        self.metadata = self._load_metadata()
        self.label_mapping = self._load_label_mapping()

        if self.split is not None:
            self.metadata = self.metadata[self.metadata["split"] == self.split].copy()

        self.metadata = self.metadata.sort_values(
            by=["movie_id", "dialogue_id", "utterance_index"]
        ).reset_index(drop=True)

        self.dialogue_keys = self._build_dialogue_keys()

    @staticmethod
    def _resolve_path(path_str: str) -> Path:
        """把相对路径解析为项目绝对路径。"""
        path = Path(path_str)
        if path.is_absolute():
            return path
        return PROJECT_ROOT / path

    def _load_metadata(self) -> pd.DataFrame:
        """读取 metadata csv，并检查必要字段。"""
        if not self.metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {self.metadata_path}")

        df = pd.read_csv(self.metadata_path)

        missing_columns = [
            column for column in self.required_columns if column not in df.columns
        ]

        if len(missing_columns) > 0:
            raise ValueError(f"Missing required columns: {missing_columns}")

        return df

    def _load_label_mapping(self) -> Dict[int, str]:
        """读取 label_id -> label_name 映射。"""
        if not self.label_mapping_path.exists():
            raise FileNotFoundError(
                f"Label mapping file not found: {self.label_mapping_path}"
            )

        df = pd.read_csv(self.label_mapping_path)

        mapping = {
            int(row["label_id"]): str(row["label_name"])
            for _, row in df.iterrows()
        }

        return mapping

    def _build_dialogue_keys(self) -> List[Tuple[str, str]]:
        """
        构建 dialogue 索引。

        一个 dialogue 由 movie_id + dialogue_id 唯一确定。
        """
        dialogue_df = (
            self.metadata[["movie_id", "dialogue_id"]]
            .drop_duplicates()
            .reset_index(drop=True)
        )

        dialogue_keys = [
            (row["movie_id"], row["dialogue_id"])
            for _, row in dialogue_df.iterrows()
        ]

        return dialogue_keys

    def __len__(self) -> int:
        """返回 dialogue 数量。"""
        return len(self.dialogue_keys)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        """
        返回一个 dialogue 样本。

        注意：
            labels 是该 dialogue 中每个 utterance 的标签序列，
            不是单个标签。
        """
        movie_id, dialogue_id = self.dialogue_keys[index]

        dialogue_df = self.metadata[
            (self.metadata["movie_id"] == movie_id)
            & (self.metadata["dialogue_id"] == dialogue_id)
        ].copy()

        dialogue_df = dialogue_df.sort_values("utterance_index")

        labels = dialogue_df["label_id"].astype(int).tolist()
        label_names = [
            self.label_mapping.get(label_id, "Unknown") for label_id in labels
        ]

        sample = {
            "movie_id": movie_id,
            "dialogue_id": dialogue_id,
            "split": str(dialogue_df["split"].iloc[0]),
            "num_utterances": int(len(dialogue_df)),
            "utterance_ids": dialogue_df["utterance_id"].astype(str).tolist(),
            "utterance_indices": dialogue_df["utterance_index"].astype(int).tolist(),
            "texts": dialogue_df["text"].fillna("").astype(str).tolist(),
            "speaker_ids": dialogue_df["speaker_id"].fillna("").astype(str).tolist(),
            "speaker_names": dialogue_df["speaker_name"].fillna("").astype(str).tolist(),
            "speaker_genders": dialogue_df["speaker_gender"].fillna("").astype(str).tolist(),
            "utterance_start_times": dialogue_df["utterance_start_time"]
            .fillna("")
            .astype(str)
            .tolist(),
            "utterance_end_times": dialogue_df["utterance_end_time"]
            .fillna("")
            .astype(str)
            .tolist(),
            "labels": labels,
            "label_names": label_names,
        }

        return sample

    def num_utterances(self) -> int:
        """返回当前 split 下 utterance 总数。"""
        return int(len(self.metadata))

    def label_distribution(self) -> pd.Series:
        """返回当前 split 下的标签分布。"""
        return self.metadata["final_main_emo"].value_counts()

    def dialogue_length_stats(self) -> Dict[str, float]:
        """
        统计 dialogue 长度。

        返回：
            dialogue 数量、最短长度、最长长度、平均长度、中位数长度。
        """
        lengths = (
            self.metadata.groupby(["movie_id", "dialogue_id"])
            .size()
            .astype(int)
        )

        return {
            "num_dialogues": float(len(lengths)),
            "min_len": float(lengths.min()),
            "max_len": float(lengths.max()),
            "mean_len": float(lengths.mean()),
            "median_len": float(lengths.median()),
        }
"""
准备 M3ED metadata。

这个脚本把 RUCM3ED 官方 annotation.json 展平成 utterance 级别的 CSV。

输入：
    third_party/RUCM3ED_official/annotation.json
    third_party/RUCM3ED_official/relation_annotation_release.json
    third_party/RUCM3ED_official/splitInfo/movie_list_train.txt
    third_party/RUCM3ED_official/splitInfo/movie_list_val.txt
    third_party/RUCM3ED_official/splitInfo/movie_list_test.txt

输出：
    data/metadata/m3ed_metadata.csv
    data/metadata/m3ed_label_mapping.csv

当前阶段只整理文本、说话人、时间戳、情绪标签和 split。
音频、视频、特征文件后续再接入。
"""

from pathlib import Path
import json
import sys
from typing import Dict, List, Any, Optional

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT))

from utils.io import ensure_dir  # noqa: E402


RUCM3ED_DIR = PROJECT_ROOT / "third_party" / "RUCM3ED_official"
ANNOTATION_PATH = RUCM3ED_DIR / "annotation.json"
RELATION_PATH = RUCM3ED_DIR / "relation_annotation_release.json"
SPLIT_DIR = RUCM3ED_DIR / "splitInfo"

METADATA_DIR = PROJECT_ROOT / "data" / "metadata"
METADATA_OUTPUT_PATH = METADATA_DIR / "m3ed_metadata.csv"
LABEL_MAPPING_OUTPUT_PATH = METADATA_DIR / "m3ed_label_mapping.csv"


# 这里先使用 RUCM3ED README 中的 7 类情绪。
# 注意大小写要和 annotation.json 中的 final_main_emo 保持一致。
LABEL_LIST = [
    "Happy",
    "Neutral",
    "Sad",
    "Disgust",
    "Anger",
    "Fear",
    "Surprise",
]

LABEL_TO_ID = {
    label_name: label_id
    for label_id, label_name in enumerate(LABEL_LIST)
}


def load_json(path: Path) -> Any:
    """
    读取 JSON 文件。
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_movie_list(path: Path) -> List[str]:
    """
    读取 splitInfo 中的 movie list。
    """
    with open(path, "r", encoding="utf-8") as f:
        movies = [line.strip() for line in f if line.strip()]
    return movies


def build_movie_split_map() -> Dict[str, str]:
    """
    建立 movie_id -> split 的映射。

    M3ED 官方 split 是 movie 级别：
        movie_list_train.txt
        movie_list_val.txt
        movie_list_test.txt
    """
    split_files = {
        "train": SPLIT_DIR / "movie_list_train.txt",
        "val": SPLIT_DIR / "movie_list_val.txt",
        "test": SPLIT_DIR / "movie_list_test.txt",
    }

    movie_to_split = {}

    for split_name, split_path in split_files.items():
        movies = read_movie_list(split_path)

        for movie_id in movies:
            if movie_id in movie_to_split:
                raise ValueError(
                    f"Movie {movie_id} appears in multiple splits: "
                    f"{movie_to_split[movie_id]} and {split_name}"
                )

            movie_to_split[movie_id] = split_name

    return movie_to_split


def get_relation_info(
    relation_annotation: Dict[str, Any],
    movie_id: str,
    dialogue_id: str,
) -> Dict[str, Optional[str]]:
    """
    从 relation_annotation_release.json 中读取 dialogue 级别信息。

    这个文件中包含 episode、dialogue start_time、dialogue end_time。
    如果找不到对应信息，就返回空值。
    """
    default_info = {
        "episode": None,
        "dialogue_start_time": None,
        "dialogue_end_time": None,
    }

    movie_relation = relation_annotation.get(movie_id)
    if movie_relation is None:
        return default_info

    dialogue_relation = movie_relation.get(dialogue_id)
    if dialogue_relation is None:
        return default_info

    return {
        "episode": dialogue_relation.get("episode"),
        "dialogue_start_time": dialogue_relation.get("start_time"),
        "dialogue_end_time": dialogue_relation.get("end_time"),
    }


def get_speaker_profile(
    speaker_info: Dict[str, Any],
    speaker_id: str,
) -> Dict[str, Optional[str]]:
    """
    根据说话人 ID 读取说话人信息。

    speaker_id 通常是 A 或 B。
    """
    profile = speaker_info.get(speaker_id, {})

    return {
        "speaker_name": profile.get("Name"),
        "speaker_age": profile.get("Age"),
        "speaker_gender": profile.get("Gender"),
    }


def parse_utterance_rows(
    annotation: Dict[str, Any],
    relation_annotation: Dict[str, Any],
    movie_to_split: Dict[str, str],
) -> List[Dict[str, Any]]:
    """
    把 annotation.json 展平成 utterance 级别的行。

    输出的每一行对应一个 utterance。
    """
    rows = []
    unknown_labels = set()
    movies_without_split = set()

    for movie_id, movie_dialogues in annotation.items():
        split = movie_to_split.get(movie_id)

        if split is None:
            movies_without_split.add(movie_id)
            split = "unknown"

        for dialogue_id, dialogue_data in movie_dialogues.items():
            speaker_info = dialogue_data.get("SpeakerInfo", {})
            dialog = dialogue_data.get("Dialog", {})

            relation_info = get_relation_info(
                relation_annotation=relation_annotation,
                movie_id=movie_id,
                dialogue_id=dialogue_id,
            )

            # dialogue 内 utterance 的顺序按 key 中最后的数字排序。
            # 例如 shaonianpai_1_2 的最后一段是 2。
            utterance_items = sorted(
                dialog.items(),
                key=lambda item: int(item[0].split("_")[-1]),
            )

            for utterance_index, (utterance_id, utterance_data) in enumerate(
                utterance_items
            ):
                emo_annotation = utterance_data.get("EmoAnnotation", {})

                final_main_emo = emo_annotation.get("final_main_emo")
                final_mul_emo = emo_annotation.get("final_mul_emo")

                if final_main_emo not in LABEL_TO_ID:
                    unknown_labels.add(str(final_main_emo))
                    label_id = -1
                else:
                    label_id = LABEL_TO_ID[final_main_emo]

                speaker_id = utterance_data.get("Speaker")
                speaker_profile = get_speaker_profile(
                    speaker_info=speaker_info,
                    speaker_id=speaker_id,
                )

                row = {
                    "movie_id": movie_id,
                    "dialogue_id": dialogue_id,
                    "utterance_id": utterance_id,
                    "utterance_index": utterance_index,
                    "split": split,
                    "episode": relation_info["episode"],
                    "dialogue_start_time": relation_info["dialogue_start_time"],
                    "dialogue_end_time": relation_info["dialogue_end_time"],
                    "utterance_start_time": utterance_data.get("StartTime"),
                    "utterance_end_time": utterance_data.get("EndTime"),
                    "speaker_id": speaker_id,
                    "speaker_name": speaker_profile["speaker_name"],
                    "speaker_age": speaker_profile["speaker_age"],
                    "speaker_gender": speaker_profile["speaker_gender"],
                    "text": utterance_data.get("Text"),
                    "emo_annotator1": emo_annotation.get("EmoAnnotator1"),
                    "emo_annotator2": emo_annotation.get("EmoAnnotator2"),
                    "emo_annotator3": emo_annotation.get("EmoAnnotator3"),
                    "final_mul_emo": final_mul_emo,
                    "final_main_emo": final_main_emo,
                    "label_id": label_id,
                }

                rows.append(row)

    if len(movies_without_split) > 0:
        print("[WARN] Movies without split:")
        for movie_id in sorted(movies_without_split):
            print("  ", movie_id)

    if len(unknown_labels) > 0:
        print("[WARN] Unknown labels found:")
        for label_name in sorted(unknown_labels):
            print("  ", label_name)

    return rows


def save_label_mapping() -> None:
    """
    保存 label 映射表。

    输出：
        data/metadata/m3ed_label_mapping.csv
    """
    label_rows = [
        {
            "label_id": label_id,
            "label_name": label_name,
        }
        for label_name, label_id in LABEL_TO_ID.items()
    ]

    df = pd.DataFrame(label_rows)
    df = df.sort_values("label_id")

    df.to_csv(
        LABEL_MAPPING_OUTPUT_PATH,
        index=False,
        encoding="utf-8-sig",
    )


def main() -> None:
    """
    主函数。
    """
    print("=" * 60)
    print("Prepare M3ED metadata")
    print("=" * 60)

    print("Project root:", PROJECT_ROOT)
    print("RUCM3ED official dir:", RUCM3ED_DIR)
    print("Annotation path:", ANNOTATION_PATH)
    print("Relation path:", RELATION_PATH)
    print("Split dir:", SPLIT_DIR)

    if not ANNOTATION_PATH.exists():
        raise FileNotFoundError(f"annotation.json not found: {ANNOTATION_PATH}")

    if not RELATION_PATH.exists():
        raise FileNotFoundError(
            f"relation_annotation_release.json not found: {RELATION_PATH}"
        )

    annotation = load_json(ANNOTATION_PATH)
    relation_annotation = load_json(RELATION_PATH)
    movie_to_split = build_movie_split_map()

    print("\nLoaded official files:")
    print("Number of movies in annotation:", len(annotation))
    print("Number of movies in split map:", len(movie_to_split))

    rows = parse_utterance_rows(
        annotation=annotation,
        relation_annotation=relation_annotation,
        movie_to_split=movie_to_split,
    )

    df = pd.DataFrame(rows)

    ensure_dir(METADATA_DIR)

    df.to_csv(
        METADATA_OUTPUT_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    save_label_mapping()

    print("\nMetadata saved successfully.")
    print("Metadata path:", METADATA_OUTPUT_PATH)
    print("Label mapping path:", LABEL_MAPPING_OUTPUT_PATH)

    print("\nTotal utterances:", len(df))

    print("\nSplit distribution:")
    print(df["split"].value_counts())

    print("\nLabel distribution:")
    print(df["final_main_emo"].value_counts())

    print("\nPreview:")
    print(df.head())

    print("\nPrepare M3ED metadata finished.")
    print("=" * 60)


if __name__ == "__main__":
    main()

"""
检查官方 MMGCN 特征 pkl 文件结构。

这个脚本只读取 third_party/MMGCN_official 中自带的 IEMOCAP / MELD 特征文件，
用于理解官方 MMGCN 期待的数据格式。

它不训练模型，不需要 torch，也不会生成新文件。

运行方式：
    python scripts/data/inspect/inspect_mmgcn_feature_pkl.py
"""

from pathlib import Path
import pickle
from typing import Any, Dict, Optional

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]

IEMOCAP_PKL = (
    PROJECT_ROOT
    / "third_party"
    / "MMGCN_official"
    / "IEMOCAP_features"
    / "IEMOCAP_features.pkl"
)

MELD_PKL = (
    PROJECT_ROOT
    / "third_party"
    / "MMGCN_official"
    / "MELD_features"
    / "MELD_features_raw1.pkl"
)


# DialogueRNN / MMGCN 常见 pkl 顺序。
# 这里不是强行假设，只是辅助打印时给字段起名字。
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


def print_section(title: str) -> None:
    """打印分隔标题。"""
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def load_pickle(path: Path) -> Any:
    """
    读取 pickle 文件。

    有些旧论文代码的 pkl 是 Python2 / 旧 Python 保存的，
    因此这里先正常读取，失败后再用 latin1 encoding 兼容。
    """
    if not path.exists():
        raise FileNotFoundError(f"Feature pkl not found: {path}")

    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except UnicodeDecodeError:
        with open(path, "rb") as f:
            return pickle.load(f, encoding="latin1")


def get_short_repr(obj: Any, max_len: int = 200) -> str:
    """返回较短的 repr，避免终端刷屏。"""
    text = repr(obj)
    if len(text) > max_len:
        text = text[:max_len] + " ... [truncated]"
    return text


def describe_basic_object(obj: Any) -> str:
    """
    简要描述一个对象的类型、长度或 shape。
    """
    if isinstance(obj, np.ndarray):
        return f"np.ndarray shape={obj.shape}, dtype={obj.dtype}"

    if isinstance(obj, dict):
        keys = list(obj.keys())
        preview_keys = keys[:5]
        return f"dict len={len(obj)}, first_keys={preview_keys}"

    if isinstance(obj, list):
        preview = obj[:3]
        return f"list len={len(obj)}, first_items={get_short_repr(preview)}"

    if isinstance(obj, tuple):
        return f"tuple len={len(obj)}"

    if isinstance(obj, set):
        preview = list(obj)[:5]
        return f"set len={len(obj)}, first_items={get_short_repr(preview)}"

    return f"{type(obj).__name__}: {get_short_repr(obj)}"


def get_field_names(data: Any) -> Optional[list]:
    """
    根据 tuple/list 长度猜测字段名。
    """
    if not isinstance(data, (tuple, list)):
        return None

    if len(data) == 9:
        return FIELD_NAMES_9

    if len(data) == 10:
        return FIELD_NAMES_10

    return None


def summarize_top_level(data: Any) -> Dict[str, Any]:
    """
    打印 pkl 顶层结构，并在可能时返回 field_map。
    """
    print("Top-level type:", type(data).__name__)

    field_map = {}

    if isinstance(data, (tuple, list)):
        print("Top-level length:", len(data))

        field_names = get_field_names(data)

        for index, item in enumerate(data):
            if field_names is not None:
                field_name = field_names[index]
            else:
                field_name = f"field_{index}"

            field_map[field_name] = item
            print(f"[{index}] {field_name}: {describe_basic_object(item)}")

    elif isinstance(data, dict):
        print("Top-level dict:", describe_basic_object(data))
        field_map = data

    else:
        print("Unsupported top-level structure:")
        print(describe_basic_object(data))

    return field_map


def get_first_dialogue_id(field_map: Dict[str, Any]) -> Optional[Any]:
    """
    尝试找到一个 dialogue id。

    优先从 trainVid 中取第一个；
    如果没有，再从 videoIDs 的 key 中取第一个。
    """
    train_vid = field_map.get("trainVid")
    video_ids = field_map.get("videoIDs")

    if isinstance(train_vid, (list, tuple, set)) and len(train_vid) > 0:
        return list(train_vid)[0]

    if isinstance(video_ids, dict) and len(video_ids) > 0:
        return list(video_ids.keys())[0]

    if isinstance(video_ids, (list, tuple)) and len(video_ids) > 0:
        return video_ids[0]

    return None


def summarize_value_for_dialogue(
    field_name: str,
    field_value: Any,
    dialogue_id: Any,
) -> None:
    """
    打印某个字段在指定 dialogue_id 下的内容摘要。
    """
    if isinstance(field_value, dict):
        if dialogue_id not in field_value:
            print(f"  {field_name}: dialogue_id not found")
            return

        value = field_value[dialogue_id]
        print(f"  {field_name}: {describe_basic_object(value)}")

        if isinstance(value, list) and len(value) > 0:
            first_item = value[0]
            print(f"    first item: {describe_basic_object(first_item)}")

        if isinstance(value, np.ndarray):
            if value.ndim > 0 and value.shape[0] > 0:
                print(f"    first row/item shape: {np.asarray(value[0]).shape}")

    else:
        print(f"  {field_name}: not a dict, skip dialogue-level preview")


def summarize_sample_dialogue(field_map: Dict[str, Any]) -> None:
    """
    打印一个样例 dialogue 的各字段信息。
    """
    dialogue_id = get_first_dialogue_id(field_map)

    if dialogue_id is None:
        print("\nNo dialogue id found for sample preview.")
        return

    print("\nSample dialogue preview")
    print("-" * 80)
    print("Sample dialogue id:", dialogue_id)

    candidate_fields = [
        "videoIDs",
        "videoSpeakers",
        "videoLabels",
        "videoText",
        "videoAudio",
        "videoVisual",
        "videoSentence",
    ]

    for field_name in candidate_fields:
        if field_name in field_map:
            summarize_value_for_dialogue(
                field_name=field_name,
                field_value=field_map[field_name],
                dialogue_id=dialogue_id,
            )


def inspect_one_pkl(name: str, path: Path) -> None:
    """
    检查一个 pkl 文件。
    """
    print_section(f"Inspect {name}")

    print("Path:", path)
    print("File size:", f"{path.stat().st_size / 1024 / 1024:.2f} MB")

    data = load_pickle(path)
    field_map = summarize_top_level(data)

    if len(field_map) > 0:
        summarize_sample_dialogue(field_map)


def main() -> None:
    """主函数。"""
    print("=" * 80)
    print("Inspect official MMGCN feature pkl files")
    print("=" * 80)
    print("Project root:", PROJECT_ROOT)

    inspect_one_pkl("IEMOCAP_features.pkl", IEMOCAP_PKL)
    inspect_one_pkl("MELD_features_raw1.pkl", MELD_PKL)

    print("\nInspection finished.")
    print("=" * 80)


if __name__ == "__main__":
    main()

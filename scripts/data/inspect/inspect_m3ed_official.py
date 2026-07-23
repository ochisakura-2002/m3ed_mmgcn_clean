"""
检查 RUCM3ED 官方仓库中的 M3ED 标注和 split 结构。

这个脚本只负责读取官方文件并打印结构信息。
它不训练模型，不读取特征，不依赖 torch。

检查对象：
1. third_party/RUCM3ED_official/annotation.json
2. third_party/RUCM3ED_official/relation_annotation_release.json
3. third_party/RUCM3ED_official/splitInfo/*.txt

运行方式：
    python scripts/data/inspect/inspect_m3ed_official.py
"""

from pathlib import Path
import json
from collections import Counter
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]
RUCM3ED_DIR = PROJECT_ROOT / "third_party" / "RUCM3ED_official"


def load_json(path: Path) -> Any:
    """读取 JSON 文件。"""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def print_section(title: str) -> None:
    """打印分隔标题。"""
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def short_repr(obj: Any, max_len: int = 1000) -> str:
    """
    把对象转成较短字符串，避免终端一次打印太多。
    """
    text = repr(obj)
    if len(text) > max_len:
        text = text[:max_len] + " ... [truncated]"
    return text


def inspect_json_structure(name: str, data: Any) -> None:
    """
    打印 JSON 的基础结构。

    支持 list / dict 两种常见情况。
    """
    print_section(f"Inspect {name}")

    print("Type:", type(data).__name__)

    if isinstance(data, dict):
        print("Number of top-level keys:", len(data))
        keys = list(data.keys())
        print("First 10 keys:", keys[:10])

        if len(keys) > 0:
            first_key = keys[0]
            print("\nFirst key:", first_key)
            print("First value type:", type(data[first_key]).__name__)
            print("First value preview:")
            print(short_repr(data[first_key]))

    elif isinstance(data, list):
        print("Number of items:", len(data))

        if len(data) > 0:
            print("First item type:", type(data[0]).__name__)
            print("First item preview:")
            print(short_repr(data[0]))

    else:
        print("Unsupported top-level JSON type.")
        print(short_repr(data))


def try_count_labels_from_annotation(annotation: Any) -> None:
    """
    尝试从 annotation 中统计标签分布。

    由于我们还不知道官方 annotation 的字段名，
    这里会尝试常见字段：
    label, emotion, emotion_label, emotion_id
    """
    print_section("Try counting labels from annotation.json")

    possible_label_keys = [
        "label",
        "emotion",
        "emotion_label",
        "emotion_id",
        "Emotion",
        "Label",
    ]

    counter = Counter()

    def visit(obj: Any) -> None:
        if isinstance(obj, dict):
            for key in possible_label_keys:
                if key in obj:
                    counter[str(obj[key])] += 1

            for value in obj.values():
                if isinstance(value, (dict, list)):
                    visit(value)

        elif isinstance(obj, list):
            for item in obj:
                visit(item)

    visit(annotation)

    if len(counter) == 0:
        print("No obvious label field found with keys:", possible_label_keys)
        print("This is not an error. We need to inspect the actual JSON schema.")
    else:
        print("Possible label distribution:")
        for label, count in counter.most_common():
            print(f"  {label}: {count}")


def inspect_split_files() -> None:
    """
    读取 splitInfo 下的 movie list 文件。
    """
    print_section("Inspect splitInfo")

    split_dir = RUCM3ED_DIR / "splitInfo"

    if not split_dir.exists():
        print("splitInfo directory not found:", split_dir)
        return

    split_files = [
        "movie_list_train.txt",
        "movie_list_val.txt",
        "movie_list_test.txt",
        "movie_list_total.txt",
    ]

    for file_name in split_files:
        path = split_dir / file_name

        if not path.exists():
            print(f"[MISS] {file_name}")
            continue

        with open(path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]

        print(f"\n[OK] {file_name}")
        print("  Number of lines:", len(lines))
        print("  First 10 lines:", lines[:10])


def main() -> None:
    """主函数。"""
    print("=" * 60)
    print("Inspect official RUCM3ED files")
    print("=" * 60)

    print("Project root:", PROJECT_ROOT)
    print("RUCM3ED official dir:", RUCM3ED_DIR)

    annotation_path = RUCM3ED_DIR / "annotation.json"
    relation_path = RUCM3ED_DIR / "relation_annotation_release.json"

    if not annotation_path.exists():
        raise FileNotFoundError(f"annotation.json not found: {annotation_path}")

    if not relation_path.exists():
        raise FileNotFoundError(
            f"relation_annotation_release.json not found: {relation_path}"
        )

    annotation = load_json(annotation_path)
    relation_annotation = load_json(relation_path)

    inspect_json_structure("annotation.json", annotation)
    try_count_labels_from_annotation(annotation)

    inspect_json_structure(
        "relation_annotation_release.json",
        relation_annotation,
    )

    inspect_split_files()

    print("\nInspection finished.")
    print("=" * 60)


if __name__ == "__main__":
    main()

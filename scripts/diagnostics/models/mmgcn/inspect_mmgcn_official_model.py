"""
Inspect official MMGCN model code.

这个脚本只读取 third_party/MMGCN_official 里的官方代码文本，
用于理解 MMGCN 官方实现的模型结构、图构建方式和依赖。

它不 import 官方代码，不训练模型，不安装依赖。
这样可以避免旧版 torch_geometric / torch 依赖污染当前环境。

运行方式：
    python scripts/diagnostics/models/mmgcn/inspect_mmgcn_official_model.py
"""

from pathlib import Path
import re
from typing import List, Dict


PROJECT_ROOT = Path(__file__).resolve().parents[4]
OFFICIAL_DIR = PROJECT_ROOT / "third_party" / "MMGCN_official"

TARGET_FILES = [
    "model_GCN.py",
    "model_mm.py",
    "model.py",
    "train.py",
    "dataloader.py",
    "run.sh",
]

KEYWORDS = [
    "torch_geometric",
    "GCNConv",
    "edge_index",
    "edge_type",
    "edge_type_mapping",
    "batch_graphify",
    "edge_perms",
    "windowp",
    "windowf",
    "qmask",
    "umask",
    "forward",
    "GraphNetwork",
    "MMGCN",
    "GCN",
]


def print_section(title: str) -> None:
    """打印分隔标题。"""
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)


def read_text(path: Path) -> str:
    """读取文本文件。"""
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def get_lines(text: str) -> List[str]:
    """按行切分文本。"""
    return text.splitlines()


def print_file_summary(file_name: str, text: str) -> None:
    """
    打印一个文件的基础统计信息。
    """
    lines = get_lines(text)

    print_section(f"File summary: {file_name}")
    print("Number of lines:", len(lines))

    import_lines = []
    class_lines = []
    def_lines = []

    for line_number, line in enumerate(lines, start=1):
        stripped = line.strip()

        if stripped.startswith("import ") or stripped.startswith("from "):
            import_lines.append((line_number, stripped))

        if re.match(r"^class\s+\w+", stripped):
            class_lines.append((line_number, stripped))

        if re.match(r"^def\s+\w+", stripped):
            def_lines.append((line_number, stripped))

    print("\nImports:")
    if len(import_lines) == 0:
        print("  No import lines found.")
    else:
        for line_number, line in import_lines[:30]:
            print(f"  L{line_number}: {line}")

    print("\nClasses:")
    if len(class_lines) == 0:
        print("  No class definitions found.")
    else:
        for line_number, line in class_lines:
            print(f"  L{line_number}: {line}")

    print("\nFunctions:")
    if len(def_lines) == 0:
        print("  No top-level function definitions found.")
    else:
        for line_number, line in def_lines[:50]:
            print(f"  L{line_number}: {line}")


def print_keyword_hits(file_name: str, text: str) -> None:
    """
    打印关键词命中位置。
    """
    lines = get_lines(text)

    print_section(f"Keyword hits: {file_name}")

    found_any = False

    for keyword in KEYWORDS:
        hits = []

        for line_number, line in enumerate(lines, start=1):
            if keyword in line:
                hits.append((line_number, line.strip()))

        if len(hits) == 0:
            continue

        found_any = True
        print(f"\nKeyword: {keyword}")
        for line_number, line in hits[:20]:
            print(f"  L{line_number}: {line}")

        if len(hits) > 20:
            print(f"  ... {len(hits) - 20} more hits")

    if not found_any:
        print("No keyword hits found.")


def print_forward_blocks(file_name: str, text: str) -> None:
    """
    打印 forward 函数附近的代码片段。

    这里只打印前若干行，用于看输入输出和关键张量。
    """
    lines = get_lines(text)

    print_section(f"Forward blocks: {file_name}")

    forward_line_numbers = []

    for line_number, line in enumerate(lines, start=1):
        if re.search(r"def\s+forward\s*\(", line):
            forward_line_numbers.append(line_number)

    if len(forward_line_numbers) == 0:
        print("No forward function found.")
        return

    for start_line in forward_line_numbers:
        print(f"\nForward block starting at L{start_line}:")
        print("-" * 100)

        start_index = max(start_line - 1, 0)
        end_index = min(start_index + 80, len(lines))

        for i in range(start_index, end_index):
            print(f"L{i + 1}: {lines[i]}")


def print_argparse_related(text: str) -> None:
    """
    打印 train.py 中和 argparse / window 相关的内容。
    """
    lines = get_lines(text)

    print_section("Argument parser related lines in train.py")

    patterns = [
        "add_argument",
        "windowp",
        "windowf",
        "graph-model",
        "nodal-attention",
        "Dataset",
        "base-model",
    ]

    for line_number, line in enumerate(lines, start=1):
        if any(pattern in line for pattern in patterns):
            print(f"L{line_number}: {line.strip()}")


def inspect_file(file_name: str) -> None:
    """
    检查一个官方文件。
    """
    path = OFFICIAL_DIR / file_name

    if not path.exists():
        print_section(f"Missing file: {file_name}")
        print("Path not found:", path)
        return

    text = read_text(path)

    print_file_summary(file_name, text)
    print_keyword_hits(file_name, text)

    if file_name.endswith(".py"):
        print_forward_blocks(file_name, text)

    if file_name == "train.py":
        print_argparse_related(text)


def main() -> None:
    """主函数。"""
    print("=" * 100)
    print("Inspect official MMGCN model implementation")
    print("=" * 100)

    print("Project root:", PROJECT_ROOT)
    print("Official MMGCN dir:", OFFICIAL_DIR)

    if not OFFICIAL_DIR.exists():
        raise FileNotFoundError(f"Official MMGCN dir not found: {OFFICIAL_DIR}")

    for file_name in TARGET_FILES:
        inspect_file(file_name)

    print("\nInspection finished.")
    print("=" * 100)


if __name__ == "__main__":
    main()

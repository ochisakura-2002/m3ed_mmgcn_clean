"""
Extract core code blocks from official MMGCN implementation.

这个脚本只读取官方代码文本，不 import 官方模块。
作用是把 MMGCN 的核心实现片段提取出来，方便阅读和过程记录。

提取内容：
1. simple_batch_graphify
2. batch_graphify
3. GraphNetwork
4. 官方 model.py 中包含 MMGCN 分支的 forward 片段
5. train.py 中 graph/window 参数

运行方式：
    python scripts/inspect/extract_mmgcn_core_blocks.py

输出：
    outputs/mmgcn_core_blocks.txt
"""

from pathlib import Path
from typing import List, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OFFICIAL_DIR = PROJECT_ROOT / "third_party" / "MMGCN_official"
OUTPUT_PATH = PROJECT_ROOT / "outputs" / "mmgcn_core_blocks.txt"


TARGET_MODEL_FILES = [
    OFFICIAL_DIR / "model.py",
    OFFICIAL_DIR / "model_GCN.py",
    OFFICIAL_DIR / "model_mm.py",
]


def read_lines(path: Path) -> List[str]:
    """读取文件行。"""
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.readlines()


def find_line_indices(lines: List[str], keyword: str) -> List[int]:
    """查找包含 keyword 的行号，返回 0-based index。"""
    return [
        index
        for index, line in enumerate(lines)
        if keyword in line
    ]


def extract_block(
    lines: List[str],
    start_index: int,
    before: int = 5,
    after: int = 120,
) -> Tuple[int, int, List[str]]:
    """
    从某个起始行附近提取代码块。
    """
    start = max(0, start_index - before)
    end = min(len(lines), start_index + after)

    return start, end, lines[start:end]


def format_block(
    file_path: Path,
    title: str,
    start: int,
    end: int,
    block_lines: List[str],
) -> str:
    """
    格式化代码块，带原始行号。
    """
    output = []

    output.append("=" * 100)
    output.append(f"{title}")
    output.append(f"File: {file_path}")
    output.append(f"Lines: {start + 1}-{end}")
    output.append("=" * 100)

    for offset, line in enumerate(block_lines, start=start + 1):
        output.append(f"L{offset}: {line.rstrip()}")

    output.append("")

    return "\n".join(output)


def extract_keyword_blocks(
    file_path: Path,
    keywords: List[str],
    before: int,
    after: int,
) -> List[str]:
    """
    从某个文件中按关键词提取代码块。
    """
    if not file_path.exists():
        return [f"[MISS] File not found: {file_path}\n"]

    lines = read_lines(file_path)
    outputs = []

    for keyword in keywords:
        indices = find_line_indices(lines, keyword)

        if len(indices) == 0:
            continue

        # 只取每个关键词第一次出现的位置，避免输出太长。
        index = indices[0]
        start, end, block_lines = extract_block(
            lines=lines,
            start_index=index,
            before=before,
            after=after,
        )

        outputs.append(
            format_block(
                file_path=file_path,
                title=f"Keyword block: {keyword}",
                start=start,
                end=end,
                block_lines=block_lines,
            )
        )

    return outputs


def extract_train_args() -> str:
    """
    提取 train.py 中与 graph/window 相关的参数行。
    """
    train_path = OFFICIAL_DIR / "train.py"

    if not train_path.exists():
        return f"[MISS] File not found: {train_path}\n"

    lines = read_lines(train_path)

    keywords = [
        "windowp",
        "windowf",
        "graph_type",
        "graph_construct",
        "multi_modal",
        "mm_fusion_mthd",
        "modals",
        "use_speaker",
        "nodal-attention",
    ]

    output = []
    output.append("=" * 100)
    output.append("Train.py graph-related arguments")
    output.append(f"File: {train_path}")
    output.append("=" * 100)

    for index, line in enumerate(lines, start=1):
        if any(keyword in line for keyword in keywords):
            output.append(f"L{index}: {line.rstrip()}")

    output.append("")

    return "\n".join(output)


def main() -> None:
    """主函数。"""
    print("=" * 100)
    print("Extract MMGCN official core blocks")
    print("=" * 100)

    print("Project root:", PROJECT_ROOT)
    print("Official dir:", OFFICIAL_DIR)
    print("Output path:", OUTPUT_PATH)

    all_outputs = []

    # 这几个关键词是当前已经确认的核心位置。
    keywords = [
        "def simple_batch_graphify",
        "def batch_graphify",
        "class GraphNetwork",
        "elif self.graph_type=='MMGCN'",
        'elif self.graph_type=="MMGCN"',
        "self.graph_net_a",
        "self.graph_net_v",
        "self.graph_net_l",
    ]

    for file_path in TARGET_MODEL_FILES:
        blocks = extract_keyword_blocks(
            file_path=file_path,
            keywords=keywords,
            before=8,
            after=150,
        )
        all_outputs.extend(blocks)

    all_outputs.append(extract_train_args())

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(all_outputs))

    print("\nExtraction finished.")
    print("Saved to:", OUTPUT_PATH)
    print("=" * 100)


if __name__ == "__main__":
    main()
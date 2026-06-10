"""
诊断 M3ED metadata label 和 feature pkl label 是否一致。

背景：
M3ED feature pkl 中自带 videoLabels。
我们自己从 annotation.json 生成了 m3ed_metadata.csv。
两者应该大体一致，但可能因为官方标注版本或处理流程不同，存在少量差异。

这个脚本只做诊断：
1. 不训练模型
2. 不修改文件
3. 不生成新文件
"""

from pathlib import Path
import sys
from collections import Counter
from typing import Dict, List, Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from utils.io import load_yaml  # noqa: E402
from datasets.m3ed.feature_dataset import M3EDFeatureDataset  # noqa: E402


ID_TO_LABEL = {
    0: "Happy",
    1: "Neutral",
    2: "Sad",
    3: "Disgust",
    4: "Anger",
    5: "Fear",
    6: "Surprise",
}


def print_section(title: str) -> None:
    """打印分隔标题。"""
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def compare_one_split(feature_pkl_path: str, split: str) -> Dict[str, Any]:
    """
    比较一个 split 下 metadata label 和 feature label 的一致性。
    """
    print_section(f"Diagnose split={split}")

    dataset = M3EDFeatureDataset(
        feature_pkl_path=feature_pkl_path,
        split=split,
        check_label_consistency=False,
    )

    mismatch_dialogues = 0
    mismatch_utterances = 0
    total_utterances = 0

    mismatch_examples: List[Dict[str, Any]] = []
    pair_counter = Counter()

    for index in range(len(dataset)):
        metadata_sample = dataset.metadata_dataset[index]
        dialogue_id = metadata_sample["dialogue_id"]

        feature_index = dataset.feature_index_map[str(dialogue_id)]
        feature_sample = dataset.feature_dataset[feature_index]

        metadata_labels = [int(x) for x in metadata_sample["labels"]]
        feature_labels = [
            int(x) for x in np.asarray(feature_sample["labels"]).tolist()
        ]

        if len(metadata_labels) != len(feature_labels):
            raise ValueError(
                f"Length mismatch for {dialogue_id}: "
                f"metadata={len(metadata_labels)}, feature={len(feature_labels)}"
            )

        total_utterances += len(metadata_labels)

        current_dialogue_has_mismatch = False

        for pos, (meta_label, feat_label) in enumerate(
            zip(metadata_labels, feature_labels)
        ):
            if meta_label != feat_label:
                mismatch_utterances += 1
                current_dialogue_has_mismatch = True

                pair_counter[(meta_label, feat_label)] += 1

                if len(mismatch_examples) < 20:
                    mismatch_examples.append(
                        {
                            "split": split,
                            "dialogue_id": dialogue_id,
                            "utterance_index": pos,
                            "utterance_id": metadata_sample["utterance_ids"][pos],
                            "text": metadata_sample["texts"][pos],
                            "metadata_label_id": meta_label,
                            "metadata_label_name": ID_TO_LABEL.get(meta_label, "Unknown"),
                            "feature_label_id": feat_label,
                            "feature_label_name": ID_TO_LABEL.get(feat_label, "Unknown"),
                        }
                    )

        if current_dialogue_has_mismatch:
            mismatch_dialogues += 1

    print("Total dialogues:", len(dataset))
    print("Total utterances:", total_utterances)
    print("Mismatch dialogues:", mismatch_dialogues)
    print("Mismatch utterances:", mismatch_utterances)

    if total_utterances > 0:
        print("Mismatch utterance ratio:", mismatch_utterances / total_utterances)

    print("\nFirst mismatch examples:")
    if len(mismatch_examples) == 0:
        print("  No mismatch found.")
    else:
        for item in mismatch_examples:
            print("-" * 80)
            print("split:", item["split"])
            print("dialogue_id:", item["dialogue_id"])
            print("utterance_index:", item["utterance_index"])
            print("utterance_id:", item["utterance_id"])
            print(
                "metadata label:",
                item["metadata_label_id"],
                item["metadata_label_name"],
            )
            print(
                "feature label:",
                item["feature_label_id"],
                item["feature_label_name"],
            )
            print("text:", item["text"])

    print("\nMismatch pair counts:")
    if len(pair_counter) == 0:
        print("  No mismatch pairs.")
    else:
        for (meta_label, feat_label), count in pair_counter.most_common():
            print(
                f"  metadata {meta_label} {ID_TO_LABEL.get(meta_label)} "
                f"-> feature {feat_label} {ID_TO_LABEL.get(feat_label)}: {count}"
            )

    return {
        "split": split,
        "num_dialogues": len(dataset),
        "total_utterances": total_utterances,
        "mismatch_dialogues": mismatch_dialogues,
        "mismatch_utterances": mismatch_utterances,
        "pair_counter": pair_counter,
    }


def main() -> None:
    """主函数。"""
    print("=" * 80)
    print("Diagnose M3ED label alignment")
    print("=" * 80)

    config = load_yaml("configs/train_mmgcn_m3ed.yaml")
    feature_pkl_path = config["dataset"]["feature_pkl_path"]

    print("Project root:", PROJECT_ROOT)
    print("Feature pkl path:", feature_pkl_path)

    summaries = []

    for split in ["train", "val", "test"]:
        summary = compare_one_split(feature_pkl_path, split)
        summaries.append(summary)

    print_section("Overall summary")

    total_dialogues = sum(item["num_dialogues"] for item in summaries)
    total_utterances = sum(item["total_utterances"] for item in summaries)
    total_mismatch_dialogues = sum(item["mismatch_dialogues"] for item in summaries)
    total_mismatch_utterances = sum(item["mismatch_utterances"] for item in summaries)

    print("Total dialogues:", total_dialogues)
    print("Total utterances:", total_utterances)
    print("Total mismatch dialogues:", total_mismatch_dialogues)
    print("Total mismatch utterances:", total_mismatch_utterances)

    if total_utterances > 0:
        print("Overall mismatch ratio:", total_mismatch_utterances / total_utterances)

    print("\nDiagnosis finished.")
    print("=" * 80)


if __name__ == "__main__":
    main()
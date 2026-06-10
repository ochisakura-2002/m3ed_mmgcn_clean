"""
分类指标工具。

这个文件只负责“基础指标怎么算”，不负责读文件、不负责画图、不负责分析某个具体实验。

后续调用关系大概是：

train.py / evaluate_best_model.py / analyze_run.py
    -> 调用 utils.metrics
    -> 得到 acc、UAR、macro F1、weighted F1 等指标

这样可以保证训练、评估、分析阶段使用同一套指标定义。
"""

from typing import Dict, List, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    recall_score,
)


def compute_classification_metrics(
    y_true,
    y_pred,
    labels: Optional[List[int]] = None,
) -> Dict[str, float]:
    """
    计算基础分类指标。

    参数：
        y_true:
            真实标签 ID 列表，例如 [0, 1, 2, 1]。
        y_pred:
            预测标签 ID 列表，例如 [0, 1, 1, 1]。
        labels:
            完整类别 ID 列表。
            如果有些类别在当前 batch 或验证集中没出现，传入 labels 可以保证指标维度一致。

    返回：
        一个字典：
        {
            "acc": 准确率,
            "uar": 非加权平均召回率，也就是 macro recall,
            "macro_f1": 每类 F1 的简单平均,
            "weighted_f1": 按样本数加权的 F1
        }
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    metrics = {
        "acc": float(accuracy_score(y_true, y_pred)),
        "uar": float(
            recall_score(
                y_true,
                y_pred,
                labels=labels,
                average="macro",
                zero_division=0,
            )
        ),
        "macro_f1": float(
            f1_score(
                y_true,
                y_pred,
                labels=labels,
                average="macro",
                zero_division=0,
            )
        ),
        "weighted_f1": float(
            f1_score(
                y_true,
                y_pred,
                labels=labels,
                average="weighted",
                zero_division=0,
            )
        ),
    }

    return metrics


def compute_per_class_recall(
    y_true,
    y_pred,
    labels: List[int],
) -> Dict[int, float]:
    """
    计算每个类别的 recall。

    参数：
        y_true:
            真实标签 ID。
        y_pred:
            预测标签 ID。
        labels:
            完整类别 ID 列表。

    返回：
        字典形式：
        {
            label_id: recall_value
        }
    """
    recalls = recall_score(
        y_true,
        y_pred,
        labels=labels,
        average=None,
        zero_division=0,
    )

    return {
        int(label_id): float(recall_value)
        for label_id, recall_value in zip(labels, recalls)
    }


def compute_confusion_matrix(
    y_true,
    y_pred,
    labels: Optional[List[int]] = None,
) -> np.ndarray:
    """
    计算混淆矩阵。

    参数：
        y_true:
            真实标签 ID。
        y_pred:
            预测标签 ID。
        labels:
            类别顺序。传入 labels 后，混淆矩阵的行列顺序会固定。

    返回：
        NumPy 数组形式的 confusion matrix。
    """
    return confusion_matrix(
        y_true,
        y_pred,
        labels=labels,
    )


def summarize_metrics(metrics: Dict[str, float]) -> str:
    """
    把指标字典整理成一行字符串，方便终端打印。

    示例输出：
        acc=0.7000 | uar=0.7148 | macro_f1=0.7012 | weighted_f1=0.7055
    """
    ordered_keys = ["acc", "uar", "macro_f1", "weighted_f1"]

    parts = []
    for key in ordered_keys:
        if key in metrics:
            parts.append(f"{key}={metrics[key]:.4f}")

    return " | ".join(parts)
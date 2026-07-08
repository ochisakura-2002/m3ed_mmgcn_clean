"""
Evaluate a trained checkpoint on M3ED or IEMOCAP.

这个脚本用于统一评估 checkpoint。

支持模型：
1. SimpleMLP
2. MMGCN

支持数据集：
1. M3ED
2. IEMOCAP

输入：
    --checkpoint outputs/runs/<run_id>/checkpoints/best_model.pt
    --split val 或 test

输出：
    outputs/runs/<run_id>/logs/evaluations/<split>_<checkpoint_name>/
        metrics.csv
        predictions.csv
        confusion_matrix.csv
        per_class_recall.csv
"""

from __future__ import annotations

from pathlib import Path
import argparse
import sys
from typing import Dict, List, Tuple, Any

import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, recall_score, confusion_matrix
from torch.utils.data import DataLoader
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from utils.seed import set_seed  # noqa: E402
from datasets.m3ed.torch_dataset import M3EDTorchDataset  # noqa: E402
from datasets.collators.m3ed_collate import m3ed_dialogue_collate_fn  # noqa: E402
from datasets.iemocap import build_iemocap_dataloader  # noqa: E402
from datasets.smoke import build_mmgcn_smoke_dataloader  # noqa: E402
from models.baselines.simple_mlp import M3EDConcatMLP  # noqa: E402
from models.baselines.mmgcn.mm_gcn import M3EDMMGCN  # noqa: E402


IGNORE_INDEX = -100


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="Evaluate a trained checkpoint."
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to checkpoint, e.g. outputs/runs/xxx/checkpoints/best_model.pt",
    )

    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Which split to evaluate.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override batch size. If not set, use config train.batch_size.",
    )

    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override device. If not set, use config system.device.",
    )

    parser.add_argument(

        "--active-modalities",

        nargs="+",

        choices=["text", "audio", "visual"],

        default=None,

        help="Override active modalities at evaluation time, e.g. --active-modalities text audio.",

    )


    return parser.parse_args()


def parse_bool(value: Any, default: bool = False) -> bool:
    """安全解析 bool，避免 bool('false') 这种人类灾难。"""
    if value is None:
        return bool(default)

    if isinstance(value, bool):
        return value

    if isinstance(value, (int, float)):
        return bool(value)

    if isinstance(value, str):
        value_lower = value.strip().lower()

        if value_lower in {"true", "1", "yes", "y"}:
            return True

        if value_lower in {"false", "0", "no", "n"}:
            return False

    raise ValueError(f"Cannot parse bool value: {value!r}")


def load_checkpoint(checkpoint_path: Path, device: torch.device) -> dict:
    """加载 checkpoint。"""
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    if "config" not in checkpoint:
        raise KeyError("Checkpoint does not contain config.")

    if "model_state_dict" not in checkpoint:
        raise KeyError("Checkpoint does not contain model_state_dict.")

    return checkpoint


def get_device(config: dict, override_device: str = None) -> torch.device:
    """选择 device。"""
    device_name = override_device or config["system"].get("device", "cuda")
    device_name = str(device_name)

    if device_name == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA requested but unavailable. Use CPU.")
        device_name = "cpu"

    return torch.device(device_name)


def get_label_list(config: dict) -> List[str]:
    """读取标签名。没有 label_list 时，用数字标签兜底。"""
    dataset_config = config["dataset"]
    num_classes = int(dataset_config["num_classes"])

    label_list = dataset_config.get("label_list", None)

    if label_list is None:
        return [str(i) for i in range(num_classes)]

    if len(label_list) != num_classes:
        raise ValueError(
            f"label_list length {len(label_list)} != num_classes {num_classes}"
        )

    return [str(x) for x in label_list]


def build_dataloader(
    config: dict,
    split: str,
    batch_size: int,
) -> DataLoader:
    """
    构建 DataLoader。

    当前支持：
        dataset.name = M3ED
        dataset.name = IEMOCAP
    """
    dataset_config = config["dataset"]
    train_config = config["train"]
    dataset_name = str(dataset_config["name"]).upper()

    if dataset_name == "M3ED":
        dataset = M3EDTorchDataset(
            feature_pkl_path=dataset_config["feature_pkl_path"],
            metadata_path=dataset_config["metadata_path"],
            label_mapping_path=dataset_config["label_mapping_path"],
            split=split,
            check_label_consistency=True,
            raise_on_label_mismatch=False,
        )

        loader = DataLoader(
            dataset,
            batch_size=int(batch_size),
            shuffle=False,
            num_workers=int(train_config.get("num_workers", 0)),
            collate_fn=m3ed_dialogue_collate_fn,
            pin_memory=torch.cuda.is_available(),
        )

        return loader

    if dataset_name == "IEMOCAP":
        loader = build_iemocap_dataloader(
            feature_pkl_path=dataset_config["feature_pkl_path"],
            split=split,
            batch_size=int(batch_size),
            valid_ratio=float(dataset_config.get("valid_ratio", 0.1)),
            val_split_strategy=str(
                dataset_config.get("val_split_strategy", "official_prefix")
            ),
            seed=int(config.get("system", {}).get("seed", 42)),
            shuffle=False,
            num_workers=int(train_config.get("num_workers", 0)),
            pin_memory=torch.cuda.is_available(),
        )

        return loader

    if dataset_name == "MMGCN_SMOKE":
        return build_mmgcn_smoke_dataloader(
            dataset_config=dataset_config,
            model_config=config["model"],
            train_config=train_config,
            split=split,
            shuffle=False,
            batch_size=int(batch_size),
            pin_memory=torch.cuda.is_available(),
        )

    raise ValueError(
        f"Unsupported dataset.name={dataset_config['name']}. "
        "Use M3ED, IEMOCAP, or MMGCN_SMOKE."
    )


def build_model(config: dict) -> nn.Module:
    """根据 checkpoint 内 config 构建模型。"""
    model_name = str(config["model"]["name"])

    if model_name == "SimpleMLP":
        model = M3EDConcatMLP(
            text_dim=int(config["model"]["text_feature_dim"]),
            audio_dim=int(config["model"]["audio_feature_dim"]),
            visual_dim=int(config["model"]["visual_feature_dim"]),
            hidden_dim=int(config["model"]["hidden_dim"]),
            num_classes=int(config["dataset"]["num_classes"]),
            dropout=float(config["model"].get("dropout", 0.3)),
        )
        return model

    if model_name == "MMGCN":
        graph_config = config.get("graph", {})
        modality_config = config.get("modality", {})
        active_modalities = modality_config.get(
            "active_modalities",
            ["text", "audio", "visual"],
        )

        model = M3EDMMGCN(
            text_dim=int(config["model"]["text_feature_dim"]),
            audio_dim=int(config["model"]["audio_feature_dim"]),
            visual_dim=int(config["model"]["visual_feature_dim"]),
            hidden_dim=int(config["model"]["hidden_dim"]),
            num_classes=int(config["dataset"]["num_classes"]),
            num_layers=int(graph_config.get("num_layers", 2)),
            dropout=float(config["model"].get("dropout", 0.3)),
            lamda=float(graph_config.get("lamda", 0.5)),
            alpha=float(graph_config.get("alpha", 0.1)),
            gamma=float(graph_config.get("gamma", 0.7)),
            use_speaker=parse_bool(graph_config.get("use_speaker", True), True),
            use_modal=parse_bool(graph_config.get("use_modal", True), True),
            use_residual=parse_bool(graph_config.get("use_residual", False), False),
            active_modalities=active_modalities,
            context_mode=str(graph_config.get("context_mode", "full")),
            window_past=graph_config.get("window_past", None),
            window_future=graph_config.get("window_future", None),
        )
        return model

    raise ValueError(f"Unsupported model name: {model_name}")


def move_tensor_batch_to_device(batch: dict, device: torch.device) -> dict:
    """把 batch 中 tensor 移动到 device。"""
    moved_batch = {}

    for key, value in batch.items():
        if torch.is_tensor(value):
            moved_batch[key] = value.to(device, non_blocking=True)
        else:
            moved_batch[key] = value

    return moved_batch


def forward_model(
    model: nn.Module,
    config: dict,
    batch: dict,
) -> torch.Tensor:
    """
    统一模型 forward。

    返回：
        logits: [B, T, num_classes]
    """
    model_name = str(config["model"]["name"])

    if model_name == "SimpleMLP":
        outputs = model(
            text_features=batch["text_features"],
            audio_features=batch["audio_features"],
            visual_features=batch["visual_features"],
        )
        return outputs["logits"]

    if model_name == "MMGCN":
        outputs = model(
            text_features=batch["text_features"],
            audio_features=batch["audio_features"],
            visual_features=batch["visual_features"],
            lengths=batch["lengths"],
            attention_mask=batch["attention_mask"],
            speaker_ids_int=batch["speaker_ids_int"],
            return_graph=False,
        )
        return outputs["logits"]

    raise ValueError(f"Unsupported model name: {model_name}")


def compute_metrics(
    y_true: List[int],
    y_pred: List[int],
    num_classes: int,
) -> Dict[str, float]:
    """计算分类指标。"""
    labels = list(range(num_classes))

    if len(y_true) == 0:
        return {
            "acc": 0.0,
            "uar": 0.0,
            "macro_f1": 0.0,
            "weighted_f1": 0.0,
        }

    return {
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


def compute_per_class_recall_rows(
    y_true: List[int],
    y_pred: List[int],
    label_list: List[str],
) -> List[dict]:
    """计算每一类 recall。"""
    num_classes = len(label_list)
    labels = list(range(num_classes))

    recalls = recall_score(
        y_true,
        y_pred,
        labels=labels,
        average=None,
        zero_division=0,
    )

    rows = []

    for label_id, recall_value in enumerate(recalls):
        rows.append(
            {
                "label_id": label_id,
                "label_name": label_list[label_id],
                "recall": float(recall_value),
            }
        )

    return rows


def build_confusion_matrix_df(
    y_true: List[int],
    y_pred: List[int],
    label_list: List[str],
) -> pd.DataFrame:
    """构造带标签名的混淆矩阵。"""
    num_classes = len(label_list)
    labels = list(range(num_classes))

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=labels,
    )

    row_names = [f"true_{name}" for name in label_list]
    col_names = [f"pred_{name}" for name in label_list]

    return pd.DataFrame(
        matrix,
        index=row_names,
        columns=col_names,
    )


@torch.no_grad()
def evaluate(
    model: nn.Module,
    config: dict,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    split: str,
) -> Tuple[float, Dict[str, float], List[dict], List[int], List[int]]:
    """评估一个 split。"""
    model.eval()

    label_list = get_label_list(config)
    num_classes = int(config["dataset"]["num_classes"])

    total_loss = 0.0
    total_valid_utterances = 0

    all_true: List[int] = []
    all_pred: List[int] = []
    prediction_rows: List[dict] = []

    for batch in tqdm(loader, desc=f"Evaluate {split}", ncols=100):
        batch = move_tensor_batch_to_device(batch, device)

        logits = forward_model(
            model=model,
            config=config,
            batch=batch,
        )

        loss = criterion(
            logits.reshape(-1, num_classes),
            batch["labels"].reshape(-1),
        )

        probabilities = torch.softmax(logits, dim=-1)
        predictions = torch.argmax(probabilities, dim=-1)
        confidences = torch.max(probabilities, dim=-1).values

        valid_mask = batch["labels"] != IGNORE_INDEX
        valid_count = int(valid_mask.sum().item())

        total_loss += float(loss.item()) * valid_count
        total_valid_utterances += valid_count

        y_true = batch["labels"][valid_mask].detach().cpu().numpy()
        y_pred = predictions[valid_mask].detach().cpu().numpy()

        all_true.extend(y_true.tolist())
        all_pred.extend(y_pred.tolist())

        batch_size = batch["labels"].shape[0]
        dialogue_ids = batch.get(
            "dialogue_ids",
            [f"dialogue_{i}" for i in range(batch_size)],
        )

        for batch_index in range(batch_size):
            length = int(batch["lengths"][batch_index].item())
            dialogue_id = str(dialogue_ids[batch_index])

            for time_index in range(length):
                true_label = int(batch["labels"][batch_index, time_index].item())

                if true_label == IGNORE_INDEX:
                    continue

                pred_label = int(predictions[batch_index, time_index].item())
                confidence = float(confidences[batch_index, time_index].item())

                prediction_rows.append(
                    {
                        "split": split,
                        "dialogue_id": dialogue_id,
                        "utterance_index": time_index,
                        "true_label_id": true_label,
                        "pred_label_id": pred_label,
                        "true_label_name": label_list[true_label],
                        "pred_label_name": label_list[pred_label],
                        "confidence": confidence,
                    }
                )

    metrics = compute_metrics(
        y_true=all_true,
        y_pred=all_pred,
        num_classes=num_classes,
    )

    avg_loss = total_loss / max(total_valid_utterances, 1)

    return avg_loss, metrics, prediction_rows, all_true, all_pred


def get_run_dir_from_checkpoint(checkpoint_path: Path) -> Path:
    """
    从 checkpoint 路径推断 run_dir。

    期望：
        outputs/runs/<run_id>/checkpoints/best_model.pt
    """
    if checkpoint_path.parent.name == "checkpoints":
        return checkpoint_path.parent.parent

    return checkpoint_path.parent


def save_evaluation_outputs(
    eval_dir: Path,
    split: str,
    checkpoint_path: Path,
    loss: float,
    metrics: Dict[str, float],
    prediction_rows: List[dict],
    y_true: List[int],
    y_pred: List[int],
    label_list: List[str],
) -> None:
    """保存评估结果。"""
    eval_dir.mkdir(parents=True, exist_ok=True)

    metrics_row = {
        "split": split,
        "checkpoint": str(checkpoint_path.resolve()),
        "loss": float(loss),
        "acc": metrics["acc"],
        "uar": metrics["uar"],
        "macro_f1": metrics["macro_f1"],
        "weighted_f1": metrics["weighted_f1"],
    }

    pd.DataFrame([metrics_row]).to_csv(
        eval_dir / "metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    pd.DataFrame(prediction_rows).to_csv(
        eval_dir / "predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )

    confusion_df = build_confusion_matrix_df(
        y_true=y_true,
        y_pred=y_pred,
        label_list=label_list,
    )
    confusion_df.to_csv(
        eval_dir / "confusion_matrix.csv",
        encoding="utf-8-sig",
    )

    per_class_rows = compute_per_class_recall_rows(
        y_true=y_true,
        y_pred=y_pred,
        label_list=label_list,
    )
    pd.DataFrame(per_class_rows).to_csv(
        eval_dir / "per_class_recall.csv",
        index=False,
        encoding="utf-8-sig",
    )


def main() -> None:
    args = parse_args()

    checkpoint_path = Path(args.checkpoint)

    if not checkpoint_path.is_absolute():
        checkpoint_path = PROJECT_ROOT / checkpoint_path

    # 先用 CPU 加载 config，再决定 device。
    temp_checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    if "config" not in temp_checkpoint:
        raise KeyError("Checkpoint does not contain config.")

    config = temp_checkpoint["config"]

    seed = int(config.get("system", {}).get("seed", 42))
    set_seed(seed)

    device = get_device(
        config=config,
        override_device=args.device,
    )

    checkpoint = load_checkpoint(
        checkpoint_path=checkpoint_path,
        device=device,
    )
    config = checkpoint["config"]

    batch_size = (
        int(args.batch_size)
        if args.batch_size is not None
        else int(config["train"]["batch_size"])
    )

    print("=" * 100)
    print("Evaluate checkpoint")
    print("=" * 100)
    print("Project root:", PROJECT_ROOT)
    print("Checkpoint:", checkpoint_path)
    print("Dataset:", config["dataset"]["name"])
    print("Model:", config["model"]["name"])
    print("Split:", args.split)
    print("Batch size:", batch_size)
    print("Device:", device)
    print("=" * 100)

    loader = build_dataloader(
        config=config,
        split=args.split,
        batch_size=batch_size,
    )

    # Evaluate-time active_modalities override.

    # This mutates only the in-memory config loaded from checkpoint.

    # It does not modify the checkpoint file.

    if getattr(args, "active_modalities", None) is not None:

        config.setdefault("modality", {})["active_modalities"] = list(args.active_modalities)

        print("[Override] active_modalities:", config["modality"]["active_modalities"])

    

    model = build_model(config).to(device)

    model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True,
    )

    criterion = nn.CrossEntropyLoss(
        ignore_index=IGNORE_INDEX,
    )

    loss, metrics, prediction_rows, y_true, y_pred = evaluate(
        model=model,
        config=config,
        loader=loader,
        criterion=criterion,
        device=device,
        split=args.split,
    )

    label_list = get_label_list(config)

    run_dir = get_run_dir_from_checkpoint(checkpoint_path)
    eval_dir = (
        run_dir
        / "logs"
        / "evaluations"
        / f"{args.split}_{checkpoint_path.stem}"
    )

    save_evaluation_outputs(
        eval_dir=eval_dir,
        split=args.split,
        checkpoint_path=checkpoint_path,
        loss=loss,
        metrics=metrics,
        prediction_rows=prediction_rows,
        y_true=y_true,
        y_pred=y_pred,
        label_list=label_list,
    )

    print("\n" + "=" * 100)
    print("Evaluation finished.")
    print("=" * 100)
    print("loss:", loss)
    print("acc:", metrics["acc"])
    print("uar:", metrics["uar"])
    print("macro_f1:", metrics["macro_f1"])
    print("weighted_f1:", metrics["weighted_f1"])
    print("Output dir:", eval_dir)
    print("=" * 100)


if __name__ == "__main__":
    main()

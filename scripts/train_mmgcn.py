"""
Train MMGCN on M3ED or IEMOCAP.

这个脚本用于训练 MMGCN baseline。

当前支持：
1. M3ED
2. IEMOCAP official MMGCN feature pkl

统一 batch 格式：
    text_features:   [B, T, D_text]
    audio_features:  [B, T, D_audio]
    visual_features: [B, T, D_visual]
    labels:          [B, T]
    attention_mask:  [B, T]
    lengths:         [B]
    speaker_ids_int: [B, T]

输出内容：
1. logs/epoch_metrics.csv
2. checkpoints/best_model.pt
3. checkpoints/last_model.pt
4. logs/val_predictions_best.csv
5. logs/confusion_matrix_best.csv
6. logs/per_class_recall_best.csv
7. logs/experiment_config.yaml
8. outputs/latest_run.txt
"""

from __future__ import annotations

from pathlib import Path
import argparse
import csv
import sys
from datetime import datetime
from typing import Dict, List, Tuple, Any

import pandas as pd
import torch
import torch.nn as nn
import yaml
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
from models.baselines.mmgcn.mm_gcn import M3EDMMGCN  # noqa: E402


IGNORE_INDEX = -100


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="Train MMGCN on M3ED or IEMOCAP."
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/train_mmgcn_m3ed.yaml",
        help="Path to YAML config file.",
    )

    return parser.parse_args()


def load_yaml_config(config_path: Path) -> dict:
    """读取 YAML 配置文件。"""
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if config is None:
        raise RuntimeError(f"Empty config file: {config_path}")

    if not isinstance(config, dict):
        raise TypeError(f"Config must be a dict: {config_path}")

    return config


def save_yaml_config(config: dict, output_path: Path) -> None:
    """保存本次实验实际使用的配置。"""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(
            config,
            f,
            allow_unicode=True,
            sort_keys=False,
        )


def parse_bool(value: Any, default: bool = False) -> bool:
    """
    安全解析 bool。

    注意：
        bool("false") 在 Python 里会得到 True。
        这种设计非常反人类，所以这里单独处理。
    """
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


def sanitize_name(name: str) -> str:
    """把实验名变成适合文件夹的名字。"""
    text = str(name).strip()

    if not text:
        return "unnamed_experiment"

    bad_chars = [" ", "/", "\\", ":", "*", "?", "\"", "<", ">", "|"]

    for ch in bad_chars:
        text = text.replace(ch, "_")

    return text


def prepare_run_environment(config: dict, config_path: Path) -> Dict[str, Path]:
    """
    创建本次实验目录。

    目录结构：
        outputs/runs/<timestamp>_<experiment_name>/
            checkpoints/
            logs/
            figures/
    """
    output_dir = PROJECT_ROOT / str(config["system"].get("output_dir", "outputs"))
    runs_dir = output_dir / "runs"

    experiment_name = sanitize_name(
        config.get("project", {}).get("experiment_name", "mmgcn_experiment")
    )
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"{timestamp}_{experiment_name}"

    run_dir = runs_dir / run_id
    logs_dir = run_dir / "logs"
    checkpoints_dir = run_dir / "checkpoints"
    figures_dir = run_dir / "figures"

    for path in [run_dir, logs_dir, checkpoints_dir, figures_dir]:
        path.mkdir(parents=True, exist_ok=True)

    save_yaml_config(
        config=config,
        output_path=logs_dir / "experiment_config.yaml",
    )

    latest_run_path = output_dir / "latest_run.txt"
    latest_run_path.parent.mkdir(parents=True, exist_ok=True)

    with open(latest_run_path, "w", encoding="utf-8") as f:
        f.write(f"run_id={run_id}\n")
        f.write(f"run_dir={run_dir.resolve()}\n")
        f.write(f"experiment_name={experiment_name}\n")
        f.write(f"config_path={config_path.resolve()}\n")

    return {
        "output_dir": output_dir,
        "run_dir": run_dir,
        "logs_dir": logs_dir,
        "checkpoints_dir": checkpoints_dir,
        "figures_dir": figures_dir,
        "latest_run_path": latest_run_path,
    }


def get_device(config: dict) -> torch.device:
    """根据配置选择 device。"""
    device_name = str(config["system"].get("device", "cuda"))

    if device_name == "cuda" and not torch.cuda.is_available():
        print("[WARN] Config requests cuda, but CUDA is not available. Use CPU.")
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
    shuffle: bool,
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
            batch_size=int(train_config["batch_size"]),
            shuffle=bool(shuffle),
            num_workers=int(train_config.get("num_workers", 0)),
            collate_fn=m3ed_dialogue_collate_fn,
            pin_memory=torch.cuda.is_available(),
        )

        return loader

    if dataset_name == "IEMOCAP":
        loader = build_iemocap_dataloader(
            feature_pkl_path=dataset_config["feature_pkl_path"],
            split=split,
            batch_size=int(train_config["batch_size"]),
            valid_ratio=float(dataset_config.get("valid_ratio", 0.1)),
            val_split_strategy=str(
                dataset_config.get("val_split_strategy", "official_prefix")
            ),
            seed=int(config.get("system", {}).get("seed", 42)),
            shuffle=bool(shuffle),
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
            shuffle=shuffle,
            pin_memory=torch.cuda.is_available(),
        )

    raise ValueError(
        f"Unsupported dataset.name={dataset_config['name']}. "
        "Use M3ED, IEMOCAP, or MMGCN_SMOKE."
    )


def build_model(config: dict) -> M3EDMMGCN:
    """根据 YAML 构建 MMGCN 模型。"""
    graph_config = config.get("graph", {})
    model_config = config["model"]
    dataset_config = config["dataset"]
    modality_config = config.get("modality", {})
    active_modalities = modality_config.get(
        "active_modalities",
        ["text", "audio", "visual"],
    )

    model = M3EDMMGCN(
        text_dim=int(model_config["text_feature_dim"]),
        audio_dim=int(model_config["audio_feature_dim"]),
        visual_dim=int(model_config["visual_feature_dim"]),
        hidden_dim=int(model_config["hidden_dim"]),
        num_classes=int(dataset_config["num_classes"]),
        num_layers=int(graph_config.get("num_layers", 2)),
        dropout=float(model_config.get("dropout", 0.3)),
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


def move_tensor_batch_to_device(batch: dict, device: torch.device) -> dict:
    """把 batch 中的 tensor 移到 device。"""
    moved_batch = {}

    for key, value in batch.items():
        if torch.is_tensor(value):
            moved_batch[key] = value.to(device, non_blocking=True)
        else:
            moved_batch[key] = value

    return moved_batch


def forward_loss(
    model: nn.Module,
    batch: dict,
    criterion: nn.Module,
    num_classes: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    前向传播并计算 loss。

    返回：
        loss
        logits
    """
    outputs = model(
        text_features=batch["text_features"],
        audio_features=batch["audio_features"],
        visual_features=batch["visual_features"],
        lengths=batch["lengths"],
        attention_mask=batch["attention_mask"],
        speaker_ids_int=batch["speaker_ids_int"],
        return_graph=False,
    )

    logits = outputs["logits"]

    loss = criterion(
        logits.reshape(-1, num_classes),
        batch["labels"].reshape(-1),
    )

    return loss, logits


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
    """计算每一类 recall，并整理成 CSV 行。"""
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
    """构造带标签名的混淆矩阵 DataFrame。"""
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


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    num_classes: int,
    epoch: int,
) -> float:
    """训练一个 epoch。"""
    model.train()

    total_loss = 0.0
    total_valid_utterances = 0

    progress = tqdm(
        loader,
        desc=f"Train epoch {epoch}",
        ncols=100,
    )

    for batch in progress:
        batch = move_tensor_batch_to_device(batch, device)

        optimizer.zero_grad()

        loss, _ = forward_loss(
            model=model,
            batch=batch,
            criterion=criterion,
            num_classes=num_classes,
        )

        loss.backward()
        optimizer.step()

        valid_count = int((batch["labels"] != IGNORE_INDEX).sum().item())
        total_loss += float(loss.item()) * valid_count
        total_valid_utterances += valid_count

        progress.set_postfix(
            loss=f"{float(loss.item()):.4f}",
        )

    return total_loss / max(total_valid_utterances, 1)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int,
    label_list: List[str],
    split: str,
    collect_predictions: bool = False,
) -> Tuple[float, Dict[str, float], List[dict], List[int], List[int]]:
    """
    评估一个 split。

    collect_predictions=True 时，返回 utterance 级预测明细。
    """
    model.eval()

    total_loss = 0.0
    total_valid_utterances = 0

    all_true: List[int] = []
    all_pred: List[int] = []
    prediction_rows: List[dict] = []

    for batch in tqdm(loader, desc=f"Evaluate {split}", ncols=100):
        batch = move_tensor_batch_to_device(batch, device)

        loss, logits = forward_loss(
            model=model,
            batch=batch,
            criterion=criterion,
            num_classes=num_classes,
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

        if collect_predictions:
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

                    row = {
                        "split": split,
                        "dialogue_id": dialogue_id,
                        "utterance_index": time_index,
                        "true_label_id": true_label,
                        "pred_label_id": pred_label,
                        "true_label_name": label_list[true_label],
                        "pred_label_name": label_list[pred_label],
                        "confidence": confidence,
                    }

                    prediction_rows.append(row)

    metrics = compute_metrics(
        y_true=all_true,
        y_pred=all_pred,
        num_classes=num_classes,
    )

    avg_loss = total_loss / max(total_valid_utterances, 1)

    return avg_loss, metrics, prediction_rows, all_true, all_pred


def initialize_epoch_metrics_csv(path: Path) -> None:
    """初始化 epoch 指标 CSV。"""
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "epoch",
                "train_loss",
                "val_loss",
                "val_acc",
                "val_uar",
                "val_macro_f1",
                "val_weighted_f1",
            ]
        )


def append_epoch_metrics_csv(
    path: Path,
    epoch: int,
    train_loss: float,
    val_loss: float,
    val_metrics: Dict[str, float],
) -> None:
    """追加一行 epoch 指标。"""
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                epoch,
                train_loss,
                val_loss,
                val_metrics["acc"],
                val_metrics["uar"],
                val_metrics["macro_f1"],
                val_metrics["weighted_f1"],
            ]
        )


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    config: dict,
    epoch: int,
    train_loss: float,
    val_loss: float,
    val_metrics: Dict[str, float],
    monitor_metric: str,
    monitor_value: float,
) -> None:
    """保存 checkpoint。"""
    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "epoch": int(epoch),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": config,
        "train_loss": float(train_loss),
        "val_loss": float(val_loss),
        "val_metrics": val_metrics,
        "monitor_metric": monitor_metric,
        "monitor_value": float(monitor_value),
    }

    torch.save(checkpoint, path)


def get_monitor_value(
    monitor_metric: str,
    val_loss: float,
    val_metrics: Dict[str, float],
) -> float:
    """根据 monitor_metric 取当前监控值。"""
    metric = str(monitor_metric)

    if metric == "val_loss":
        return float(val_loss)

    if metric.startswith("val_"):
        metric = metric[len("val_") :]

    if metric not in val_metrics:
        raise KeyError(
            f"monitor_metric={monitor_metric} is not available. "
            f"Available: val_loss, val_acc, val_uar, "
            f"val_macro_f1, val_weighted_f1"
        )

    return float(val_metrics[metric])


def is_better_monitor_value(
    monitor_metric: str,
    current_value: float,
    best_value: float,
) -> bool:
    """判断当前监控值是否优于历史最好值。"""
    if str(monitor_metric) == "val_loss":
        return current_value < best_value

    return current_value > best_value


def initial_best_value(monitor_metric: str) -> float:
    """初始化 best 值。"""
    if str(monitor_metric) == "val_loss":
        return float("inf")

    return -float("inf")


def save_best_validation_outputs(
    logs_dir: Path,
    prediction_rows: List[dict],
    y_true: List[int],
    y_pred: List[int],
    label_list: List[str],
) -> None:
    """保存 best epoch 对应的验证集预测与类别指标。"""
    pd.DataFrame(prediction_rows).to_csv(
        logs_dir / "val_predictions_best.csv",
        index=False,
        encoding="utf-8-sig",
    )

    confusion_df = build_confusion_matrix_df(
        y_true=y_true,
        y_pred=y_pred,
        label_list=label_list,
    )
    confusion_df.to_csv(
        logs_dir / "confusion_matrix_best.csv",
        encoding="utf-8-sig",
    )

    per_class_rows = compute_per_class_recall_rows(
        y_true=y_true,
        y_pred=y_pred,
        label_list=label_list,
    )
    pd.DataFrame(per_class_rows).to_csv(
        logs_dir / "per_class_recall_best.csv",
        index=False,
        encoding="utf-8-sig",
    )


def print_config_summary(config: dict, run_paths: Dict[str, Path]) -> None:
    """打印实验配置摘要。"""
    dataset_name = config["dataset"]["name"]
    model_name = config["model"]["name"]
    graph_config = config.get("graph", {})

    print("=" * 100)
    print("Train MMGCN")
    print("=" * 100)
    print("Project root:", PROJECT_ROOT)
    print("Run dir:", run_paths["run_dir"])
    print("Dataset:", dataset_name)
    print("Model:", model_name)
    print("Feature path:", config["dataset"].get("feature_pkl_path", ""))
    print("Batch size:", config["train"]["batch_size"])
    print("Learning rate:", config["train"]["learning_rate"])
    print("Weight decay:", config["train"].get("weight_decay", 0.0))
    print("Max epochs:", config["train"]["max_epochs"])
    print("Monitor metric:", config.get("logging", {}).get("monitor_metric", "val_uar"))

    print("\nGraph config:")
    print("  context_mode:", graph_config.get("context_mode", "full"))
    print("  window_past:", graph_config.get("window_past", None))
    print("  window_future:", graph_config.get("window_future", None))
    print("  num_layers:", graph_config.get("num_layers", 2))
    print("  lamda:", graph_config.get("lamda", 0.5))
    print("  alpha:", graph_config.get("alpha", 0.1))
    print("  gamma:", graph_config.get("gamma", 0.7))
    print("  use_speaker:", graph_config.get("use_speaker", True))
    print("  use_modal:", graph_config.get("use_modal", True))
    print("  use_residual:", graph_config.get("use_residual", False))
    print("  active_modalities:", config.get("modality", {}).get("active_modalities", ["text", "audio", "visual"]))
    print("=" * 100)


def main() -> None:
    args = parse_args()

    config_path = Path(args.config)

    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path

    config = load_yaml_config(config_path)

    seed = int(config.get("system", {}).get("seed", 42))
    set_seed(seed)

    run_paths = prepare_run_environment(
        config=config,
        config_path=config_path,
    )

    print_config_summary(config, run_paths)

    device = get_device(config)

    num_classes = int(config["dataset"]["num_classes"])
    label_list = get_label_list(config)

    train_loader = build_dataloader(
        config=config,
        split="train",
        shuffle=True,
    )

    val_loader = build_dataloader(
        config=config,
        split="val",
        shuffle=False,
    )

    model = build_model(config).to(device)

    print(
        "[MMGCN model config]",
        "use_speaker=", getattr(model, "use_speaker", None),
        "use_modal=", getattr(model, "use_modal", None),
        "use_residual=", getattr(model, "use_residual", None),
        "gamma=", getattr(model, "gamma", None),
    )

    criterion = nn.CrossEntropyLoss(
        ignore_index=IGNORE_INDEX,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["train"]["learning_rate"]),
        weight_decay=float(config["train"].get("weight_decay", 0.0)),
    )

    max_epochs = int(config["train"]["max_epochs"])
    logging_config = config.get("logging", {})
    save_best = parse_bool(logging_config.get("save_best", True), True)
    save_last = parse_bool(logging_config.get("save_last", True), True)
    monitor_metric = str(logging_config.get("monitor_metric", "val_uar"))

    epoch_metrics_path = run_paths["logs_dir"] / "epoch_metrics.csv"
    initialize_epoch_metrics_csv(epoch_metrics_path)

    best_value = initial_best_value(monitor_metric)
    best_epoch = -1

    for epoch in range(1, max_epochs + 1):
        print("\n" + "=" * 100)
        print(f"Epoch {epoch}/{max_epochs}")
        print("=" * 100)

        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            num_classes=num_classes,
            epoch=epoch,
        )

        val_loss, val_metrics, val_predictions, y_true, y_pred = evaluate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            num_classes=num_classes,
            label_list=label_list,
            split="val",
            collect_predictions=True,
        )

        append_epoch_metrics_csv(
            path=epoch_metrics_path,
            epoch=epoch,
            train_loss=train_loss,
            val_loss=val_loss,
            val_metrics=val_metrics,
        )

        monitor_value = get_monitor_value(
            monitor_metric=monitor_metric,
            val_loss=val_loss,
            val_metrics=val_metrics,
        )

        print(
            f"[Epoch {epoch}] "
            f"train_loss={train_loss:.6f} "
            f"val_loss={val_loss:.6f} "
            f"val_acc={val_metrics['acc']:.6f} "
            f"val_uar={val_metrics['uar']:.6f} "
            f"val_macro_f1={val_metrics['macro_f1']:.6f} "
            f"val_weighted_f1={val_metrics['weighted_f1']:.6f}"
        )

        if save_last:
            save_checkpoint(
                path=run_paths["checkpoints_dir"] / "last_model.pt",
                model=model,
                optimizer=optimizer,
                config=config,
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                val_metrics=val_metrics,
                monitor_metric=monitor_metric,
                monitor_value=monitor_value,
            )

        if is_better_monitor_value(
            monitor_metric=monitor_metric,
            current_value=monitor_value,
            best_value=best_value,
        ):
            best_value = monitor_value
            best_epoch = epoch

            print(
                f"[BEST] epoch={best_epoch}, "
                f"{monitor_metric}={best_value:.6f}"
            )

            if save_best:
                save_checkpoint(
                    path=run_paths["checkpoints_dir"] / "best_model.pt",
                    model=model,
                    optimizer=optimizer,
                    config=config,
                    epoch=epoch,
                    train_loss=train_loss,
                    val_loss=val_loss,
                    val_metrics=val_metrics,
                    monitor_metric=monitor_metric,
                    monitor_value=monitor_value,
                )

                save_best_validation_outputs(
                    logs_dir=run_paths["logs_dir"],
                    prediction_rows=val_predictions,
                    y_true=y_true,
                    y_pred=y_pred,
                    label_list=label_list,
                )

    print("\n" + "=" * 100)
    print("Training finished.")
    print("=" * 100)
    print("Run dir:", run_paths["run_dir"])
    print("Best epoch:", best_epoch)
    print("Best monitor value:", best_value)
    print("Epoch metrics:", epoch_metrics_path)
    print("Best checkpoint:", run_paths["checkpoints_dir"] / "best_model.pt")
    print("Last checkpoint:", run_paths["checkpoints_dir"] / "last_model.pt")
    print("=" * 100)


if __name__ == "__main__":
    main()

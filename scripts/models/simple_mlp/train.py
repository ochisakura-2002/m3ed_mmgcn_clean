"""
Train SimpleMLP on M3ED.

这个脚本用于跑通完整训练闭环：

1. train / val DataLoader
2. SimpleMLP forward
3. loss / backward / optimizer
4. validation metrics
5. checkpoint 保存
6. prediction 明细保存
7. experiment_summary.csv 记录

注意：
SimpleMLP 不是最终 MMGCN baseline。
它只是用于验证训练管线是否稳定。
"""

from pathlib import Path
import argparse
import csv
import json
import sys
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


PROJECT_ROOT = next(
    candidate
    for candidate in Path(__file__).resolve().parents
    if (candidate / "AGENTS.md").is_file() and (candidate / "scripts").is_dir()
)
sys.path.insert(0, str(PROJECT_ROOT))

from utils.io import load_yaml, prepare_run_environment  # noqa: E402
from utils.output_paths import resolve_output_category  # noqa: E402
from utils.seed import set_seed  # noqa: E402
from utils.metrics import (  # noqa: E402
    compute_classification_metrics,
    compute_confusion_matrix,
    compute_per_class_recall,
    summarize_metrics,
)
from datasets.m3ed.torch_dataset import M3EDTorchDataset  # noqa: E402
from datasets.collators.m3ed_collate import m3ed_dialogue_collate_fn, IGNORE_INDEX  # noqa: E402
from models.simple_mlp.model import M3EDConcatMLP  # noqa: E402


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="Train SimpleMLP on M3ED."
    )

    parser.add_argument(
        "--config",
        type=str,
        default=(
            "configs/simple_mlp/unified/m3ed/full_context/"
            "m3ed_features/development.yaml"
        ),
        help="Path to YAML config file.",
    )
    parser.add_argument(
        "--experiment-date",
        default=None,
        help="Frozen experiment launch date in YYYYMMDD format.",
    )
    parser.add_argument("--experiment-group", default=None)

    return parser.parse_args()


def get_device(config: dict) -> torch.device:
    """根据配置选择设备。"""
    device_name = config["system"]["device"]

    if device_name == "cuda" and not torch.cuda.is_available():
        print("[WARN] Config requests cuda, but CUDA is not available. Use CPU.")
        device_name = "cpu"

    return torch.device(device_name)


def build_dataloader(
    config: dict,
    split: str,
    shuffle: bool,
) -> DataLoader:
    """构建 M3ED DataLoader。"""
    dataset = M3EDTorchDataset(
        feature_pkl_path=config["dataset"]["feature_pkl_path"],
        metadata_path=config["dataset"]["metadata_path"],
        label_mapping_path=config["dataset"]["label_mapping_path"],
        split=split,
        check_label_consistency=True,
        raise_on_label_mismatch=False,
    )

    loader = DataLoader(
        dataset,
        batch_size=int(config["train"]["batch_size"]),
        shuffle=shuffle,
        num_workers=int(config["train"]["num_workers"]),
        collate_fn=m3ed_dialogue_collate_fn,
        pin_memory=torch.cuda.is_available(),
    )

    return loader


def build_model(config: dict) -> M3EDConcatMLP:
    """构建 SimpleMLP 模型。"""
    model = M3EDConcatMLP(
        text_dim=int(config["model"]["text_feature_dim"]),
        audio_dim=int(config["model"]["audio_feature_dim"]),
        visual_dim=int(config["model"]["visual_feature_dim"]),
        hidden_dim=int(config["model"]["hidden_dim"]),
        num_classes=int(config["dataset"]["num_classes"]),
        dropout=float(config["model"]["dropout"]),
    )

    return model


def move_tensor_batch_to_device(batch: dict, device: torch.device) -> dict:
    """把 batch 中的 tensor 移动到 device。"""
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
    )

    logits = outputs["logits"]

    loss = criterion(
        logits.reshape(-1, num_classes),
        batch["labels"].reshape(-1),
    )

    return loss, logits


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
    collect_predictions: bool = False,
) -> Tuple[float, Dict[str, float], List[dict]]:
    """
    在验证集上评估。

    如果 collect_predictions=True，会返回 utterance 级预测明细。
    """
    model.eval()

    total_loss = 0.0
    total_valid_utterances = 0

    all_true = []
    all_pred = []
    prediction_rows = []

    for batch in tqdm(loader, desc="Evaluate", ncols=100):
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

            for batch_index in range(batch_size):
                seq_len = int(batch["lengths"][batch_index].item())

                for pos in range(seq_len):
                    true_label_id = int(batch["labels"][batch_index, pos].item())
                    pred_label_id = int(predictions[batch_index, pos].item())
                    confidence = float(confidences[batch_index, pos].item())

                    prediction_rows.append(
                        {
                            "movie_id": batch["movie_ids"][batch_index],
                            "dialogue_id": batch["dialogue_ids"][batch_index],
                            "utterance_id": batch["utterance_ids"][batch_index][pos],
                            "utterance_index": int(
                                batch["utterance_indices"][batch_index][pos]
                            ),
                            "text": batch["texts"][batch_index][pos],
                            "true_label_id": true_label_id,
                            "true_label_name": label_list[true_label_id],
                            "pred_label_id": pred_label_id,
                            "pred_label_name": label_list[pred_label_id],
                            "confidence": confidence,
                        }
                    )

    avg_loss = total_loss / max(total_valid_utterances, 1)

    metrics = compute_classification_metrics(
        y_true=all_true,
        y_pred=all_pred,
        labels=list(range(num_classes)),
    )

    return avg_loss, metrics, prediction_rows


def save_epoch_metrics(
    rows: List[dict],
    save_path: Path,
) -> None:
    """保存 epoch_metrics.csv。"""
    df = pd.DataFrame(rows)
    df.to_csv(save_path, index=False, encoding="utf-8-sig")


def save_prediction_outputs(
    prediction_rows: List[dict],
    label_list: List[str],
    logs_dir: Path,
) -> None:
    """保存 best validation predictions 和分析文件。"""
    predictions_path = logs_dir / "val_predictions_best.csv"
    df = pd.DataFrame(prediction_rows)
    df.to_csv(predictions_path, index=False, encoding="utf-8-sig")

    y_true = df["true_label_id"].astype(int).tolist()
    y_pred = df["pred_label_id"].astype(int).tolist()
    labels = list(range(len(label_list)))

    confusion = compute_confusion_matrix(
        y_true=y_true,
        y_pred=y_pred,
        labels=labels,
    )

    confusion_df = pd.DataFrame(
        confusion,
        index=[f"true_{name}" for name in label_list],
        columns=[f"pred_{name}" for name in label_list],
    )
    confusion_df.to_csv(
        logs_dir / "confusion_matrix_best.csv",
        encoding="utf-8-sig",
    )

    per_class_recall = compute_per_class_recall(
        y_true=y_true,
        y_pred=y_pred,
        labels=labels,
    )

    per_class_rows = [
        {
            "label_id": label_id,
            "label_name": label_list[label_id],
            "recall": recall_value,
        }
        for label_id, recall_value in per_class_recall.items()
    ]

    pd.DataFrame(per_class_rows).to_csv(
        logs_dir / "per_class_recall_best.csv",
        index=False,
        encoding="utf-8-sig",
    )


def save_checkpoint(
    save_path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    config: dict,
    metrics: Dict[str, float],
) -> None:
    """保存 checkpoint。"""
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": config,
        "metrics": metrics,
        "test_split_used_for_selection": False,
    }

    torch.save(checkpoint, save_path)


def append_experiment_summary(
    config: dict,
    run_info: dict,
    best_epoch: int,
    best_metrics: Dict[str, float],
) -> None:
    """Append to this experiment date's analysis summary."""
    summary_path = (
        resolve_output_category(
            "analysis",
            str(run_info["experiment_date"]),
            Path(run_info["output_root"]),
            experiment_group=str(run_info["experiment_group"]),
        )
        / "experiment_summary"
        / "experiment_summary.csv"
    )

    row = {
        "run_id": run_info["run_id"],
        "experiment_name": config["project"]["experiment_name"],
        "dataset": config["dataset"]["name"],
        "model": config["model"]["name"],
        "feature_pkl_path": config["dataset"]["feature_pkl_path"],
        "batch_size": config["train"]["batch_size"],
        "learning_rate": config["train"]["learning_rate"],
        "weight_decay": config["train"]["weight_decay"],
        "max_epochs": config["train"]["max_epochs"],
        "best_epoch": best_epoch,
        "best_val_loss": best_metrics["val_loss"],
        "best_val_acc": best_metrics["acc"],
        "best_val_uar": best_metrics["uar"],
        "best_val_macro_f1": best_metrics["macro_f1"],
        "best_val_weighted_f1": best_metrics["weighted_f1"],
        "run_dir": str(run_info["run_dir"]),
    }

    file_exists = summary_path.exists()

    with open(summary_path, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))

        if not file_exists:
            writer.writeheader()

        writer.writerow(row)


def print_start_summary(
    config: dict,
    run_info: dict,
    device: torch.device,
) -> None:
    """打印训练启动摘要。"""
    print("=" * 80)
    print("Train SimpleMLP on M3ED")
    print("=" * 80)

    print("Project root:", PROJECT_ROOT)
    print("Experiment:", config["project"]["experiment_name"])
    print("Dataset:", config["dataset"]["name"])
    print("Model:", config["model"]["name"])
    print("Device:", device)

    print("\nFeature:")
    print("  feature_pkl_path:", config["dataset"]["feature_pkl_path"])
    print("  text_dim:", config["model"]["text_feature_dim"])
    print("  audio_dim:", config["model"]["audio_feature_dim"])
    print("  visual_dim:", config["model"]["visual_feature_dim"])

    print("\nTrain:")
    print("  batch_size:", config["train"]["batch_size"])
    print("  learning_rate:", config["train"]["learning_rate"])
    print("  weight_decay:", config["train"]["weight_decay"])
    print("  max_epochs:", config["train"]["max_epochs"])

    print("\nRun:")
    print("  run_id:", run_info["run_id"])
    print("  run_dir:", run_info["run_dir"])
    print("  logs_dir:", run_info["logs_dir"])
    print("  checkpoints_dir:", run_info["checkpoints_dir"])
    print("=" * 80)


def main() -> None:
    """主函数。"""
    args = parse_args()

    config = load_yaml(args.config)

    set_seed(int(config["system"]["seed"]))

    device = get_device(config)

    run_info = prepare_run_environment(
        config,
        experiment_date=args.experiment_date,
        experiment_group=args.experiment_group,
    )
    print(
        "CODEX_RUN_INFO_JSON="
        + json.dumps(
            {
                "run_id": str(run_info["run_id"]),
                "run_dir": str(Path(run_info["run_dir"]).resolve()),
            }
        ),
        flush=True,
    )

    print_start_summary(
        config=config,
        run_info=run_info,
        device=device,
    )

    label_list = config["dataset"]["label_list"]
    num_classes = int(config["dataset"]["num_classes"])

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

    criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["train"]["learning_rate"]),
        weight_decay=float(config["train"]["weight_decay"]),
    )

    max_epochs = int(config["train"]["max_epochs"])
    monitor_metric = config["logging"]["monitor_metric"]

    best_epoch = -1
    best_monitor_value = -float("inf")
    best_metrics = None

    epoch_rows = []

    for epoch in range(1, max_epochs + 1):
        print(f"\nEpoch {epoch}/{max_epochs}")

        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            num_classes=num_classes,
            epoch=epoch,
        )

        val_loss, val_metrics, _ = evaluate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            num_classes=num_classes,
            label_list=label_list,
            collect_predictions=False,
        )

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_acc": val_metrics["acc"],
            "val_uar": val_metrics["uar"],
            "val_macro_f1": val_metrics["macro_f1"],
            "val_weighted_f1": val_metrics["weighted_f1"],
        }

        epoch_rows.append(row)

        save_epoch_metrics(
            rows=epoch_rows,
            save_path=run_info["logs_dir"] / "epoch_metrics.csv",
        )

        print(
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"{summarize_metrics(val_metrics)}"
        )

        monitor_value = row[monitor_metric]

        if monitor_value > best_monitor_value:
            best_monitor_value = monitor_value
            best_epoch = epoch
            best_metrics = {
                "val_loss": val_loss,
                **val_metrics,
            }

            save_checkpoint(
                save_path=run_info["checkpoints_dir"] / "best_model.pt",
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                config=config,
                metrics=best_metrics,
            )

            print(
                f"[Best] epoch={best_epoch}, "
                f"{monitor_metric}={best_monitor_value:.4f}"
            )

    last_metrics = {
        "val_loss": val_loss,
        **val_metrics,
    }

    save_checkpoint(
        save_path=run_info["checkpoints_dir"] / "last_model.pt",
        model=model,
        optimizer=optimizer,
        epoch=max_epochs,
        config=config,
        metrics=last_metrics,
    )

    print("\nLoad best checkpoint for prediction export.")

    best_checkpoint = torch.load(
        run_info["checkpoints_dir"] / "best_model.pt",
        map_location=device,
        weights_only=False,
    )
    model.load_state_dict(best_checkpoint["model_state_dict"])

    best_val_loss, best_val_metrics, prediction_rows = evaluate(
        model=model,
        loader=val_loader,
        criterion=criterion,
        device=device,
        num_classes=num_classes,
        label_list=label_list,
        collect_predictions=True,
    )

    save_prediction_outputs(
        prediction_rows=prediction_rows,
        label_list=label_list,
        logs_dir=run_info["logs_dir"],
    )
    print("[Info] Global experiment_summary.csv is managed by "
          "scripts/maintenance/rebuild_experiment_summary.py")

    print("\nTraining finished.")
    print("Best epoch      :", best_epoch)
    print("Best val_loss   :", best_metrics["val_loss"])
    print("Best val_acc    :", best_metrics["acc"])
    print("Best val_uar    :", best_metrics["uar"])
    print("Best val_macro_f1:", best_metrics["macro_f1"])
    print("Metrics csv     :", run_info["logs_dir"] / "epoch_metrics.csv")
    print("Best checkpoint :", run_info["checkpoints_dir"] / "best_model.pt")
    print("Last checkpoint :", run_info["checkpoints_dir"] / "last_model.pt")
    print("Val predictions :", run_info["logs_dir"] / "val_predictions_best.csv")
    print("Experiment summary: managed by dated analysis outputs")
    print("=" * 80)


if __name__ == "__main__":
    main()

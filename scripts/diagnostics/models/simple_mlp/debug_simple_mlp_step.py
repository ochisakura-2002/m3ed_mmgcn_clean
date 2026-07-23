"""
测试最小三模态 MLP 的一个训练步。

这个脚本用于确认：
1. DataLoader 能提供 batch
2. 模型 forward 能跑通
3. CrossEntropyLoss 能正确忽略 padding label
4. backward 能跑通
5. optimizer.step 能跑通
6. metrics 能在有效 utterance 上计算

它不是正式训练脚本。
它只跑一个 batch。
"""

from pathlib import Path
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from utils.io import load_yaml  # noqa: E402
from utils.seed import set_seed  # noqa: E402
from utils.metrics import compute_classification_metrics, summarize_metrics  # noqa: E402
from datasets.m3ed.torch_dataset import M3EDTorchDataset  # noqa: E402
from datasets.collators.m3ed_collate import m3ed_dialogue_collate_fn, IGNORE_INDEX  # noqa: E402
from models.simple_mlp.model import M3EDConcatMLP  # noqa: E402


def print_section(title: str) -> None:
    """
    打印分隔标题。
    """
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def move_tensor_batch_to_device(batch: dict, device: torch.device) -> dict:
    """
    把 batch 中的 tensor 移到指定 device。

    非 tensor 字段，比如 dialogue_ids / texts，不移动。
    """
    moved_batch = {}

    for key, value in batch.items():
        if torch.is_tensor(value):
            moved_batch[key] = value.to(device)
        else:
            moved_batch[key] = value

    return moved_batch


def compute_grad_norm(model: torch.nn.Module) -> float:
    """
    计算模型参数梯度范数。

    这个值只用于 debug：
        如果是 0 或 nan，说明 backward 可能有问题。
    """
    total_norm_square = 0.0

    for parameter in model.parameters():
        if parameter.grad is None:
            continue

        grad_norm = parameter.grad.detach().data.norm(2).item()
        total_norm_square += grad_norm ** 2

    return total_norm_square ** 0.5


def main() -> None:
    """
    主函数。
    """
    print("=" * 80)
    print("Debug simple multimodal MLP one training step")
    print("=" * 80)

    config = load_yaml("configs/train_mmgcn_m3ed.yaml")

    seed = int(config["system"]["seed"])
    set_seed(seed)

    device_name = config["system"]["device"]
    if device_name == "cuda" and not torch.cuda.is_available():
        print("[WARN] Config requests cuda, but CUDA is not available. Use CPU.")
        device_name = "cpu"

    device = torch.device(device_name)

    feature_pkl_path = config["dataset"]["feature_pkl_path"]
    batch_size = int(config["train"]["batch_size"])

    text_dim = int(config["model"]["text_feature_dim"])
    audio_dim = int(config["model"]["audio_feature_dim"])
    visual_dim = int(config["model"]["visual_feature_dim"])
    hidden_dim = int(config["model"]["hidden_dim"])
    num_classes = int(config["dataset"]["num_classes"])
    dropout = float(config["model"]["dropout"])

    learning_rate = float(config["train"]["learning_rate"])
    weight_decay = float(config["train"]["weight_decay"])

    print("Project root:", PROJECT_ROOT)
    print("Device:", device)
    print("Feature pkl path:", feature_pkl_path)
    print("Batch size:", batch_size)

    print_section("Build DataLoader")

    dataset = M3EDTorchDataset(
        feature_pkl_path=feature_pkl_path,
        split="train",
        check_label_consistency=True,
        raise_on_label_mismatch=False,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=m3ed_dialogue_collate_fn,
    )

    batch = next(iter(loader))
    batch = move_tensor_batch_to_device(batch, device)

    print("Batch loaded.")
    print("  text_features:", batch["text_features"].shape)
    print("  audio_features:", batch["audio_features"].shape)
    print("  visual_features:", batch["visual_features"].shape)
    print("  labels:", batch["labels"].shape)
    print("  attention_mask:", batch["attention_mask"].shape)
    print("  lengths:", batch["lengths"].tolist())

    print_section("Build model")

    model = M3EDConcatMLP(
        text_dim=text_dim,
        audio_dim=audio_dim,
        visual_dim=visual_dim,
        hidden_dim=hidden_dim,
        num_classes=num_classes,
        dropout=dropout,
    ).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    num_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print("Model:", model.__class__.__name__)
    print("Trainable parameters:", num_parameters)

    print_section("Forward / loss / backward")

    model.train()
    optimizer.zero_grad()

    outputs = model(
        text_features=batch["text_features"],
        audio_features=batch["audio_features"],
        visual_features=batch["visual_features"],
    )

    logits = outputs["logits"]

    print("Logits shape:", logits.shape)

    logits_flat = logits.reshape(-1, num_classes)
    labels_flat = batch["labels"].reshape(-1)

    loss = criterion(logits_flat, labels_flat)

    print("Loss:", float(loss.item()))

    loss.backward()

    grad_norm = compute_grad_norm(model)
    print("Gradient norm:", grad_norm)

    optimizer.step()

    print("Optimizer step completed.")

    print_section("One-batch metrics")

    with torch.no_grad():
        preds_flat = torch.argmax(logits_flat, dim=-1)
        valid_mask = labels_flat != IGNORE_INDEX

        y_true = labels_flat[valid_mask].detach().cpu().numpy()
        y_pred = preds_flat[valid_mask].detach().cpu().numpy()

    metrics = compute_classification_metrics(
        y_true=y_true,
        y_pred=y_pred,
        labels=list(range(num_classes)),
    )

    print("Valid utterances:", int(valid_mask.sum().item()))
    print("Metrics:", metrics)
    print("Summary:", summarize_metrics(metrics))

    print("\nDebug simple MLP step finished.")
    print("=" * 80)


if __name__ == "__main__":
    main()

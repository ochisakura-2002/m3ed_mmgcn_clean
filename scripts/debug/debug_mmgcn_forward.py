"""
Debug MMGCN forward on one M3ED batch.

这个脚本用于验证：
1. M3ED DataLoader 能否接入 MMGCN
2. MMGCN forward 是否输出 [B, T, num_classes]
3. loss 是否能计算
4. backward 是否能跑通
5. dense adjacency 是否构造正常

运行方式：
    python scripts/debug/debug_mmgcn_forward.py
"""

from pathlib import Path
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from utils.io import load_yaml  # noqa: E402
from utils.seed import set_seed  # noqa: E402
from utils.metrics import compute_classification_metrics, summarize_metrics  # noqa: E402
from datasets.m3ed.torch_dataset import M3EDTorchDataset  # noqa: E402
from datasets.collators.m3ed_collate import (  # noqa: E402
    m3ed_dialogue_collate_fn,
    IGNORE_INDEX,
)
from models.baselines.mmgcn.mm_gcn import M3EDMMGCN  # noqa: E402


def print_section(title: str) -> None:
    """打印分隔标题。"""
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def move_tensor_batch_to_device(batch: dict, device: torch.device) -> dict:
    """
    把 batch 中的 tensor 移到 device。
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
    计算梯度范数。
    """
    total_norm_square = 0.0

    for parameter in model.parameters():
        if parameter.grad is None:
            continue

        grad_norm = parameter.grad.detach().data.norm(2).item()
        total_norm_square += grad_norm ** 2

    return total_norm_square ** 0.5


def build_model(config: dict) -> M3EDMMGCN:
    """
    根据 YAML 构建 MMGCN。
    """
    graph_config = config.get("graph", {})

    model = M3EDMMGCN(
        text_dim=int(config["model"]["text_feature_dim"]),
        audio_dim=int(config["model"]["audio_feature_dim"]),
        visual_dim=int(config["model"]["visual_feature_dim"]),
        hidden_dim=int(config["model"]["hidden_dim"]),
        num_classes=int(config["dataset"]["num_classes"]),
        num_layers=int(graph_config.get("num_layers", 2)),
        dropout=float(config["model"]["dropout"]),
        lamda=float(graph_config.get("lamda", 0.5)),
        alpha=float(graph_config.get("alpha", 0.1)),
        use_speaker=bool(graph_config.get("use_speaker", True)),
        use_modal=bool(graph_config.get("use_modal", True)),
        use_residual=bool(graph_config.get("use_residual", False)),
        context_mode=str(graph_config.get("context_mode", "full")),
        window_past=graph_config.get("window_past", None),
        window_future=graph_config.get("window_future", None),
    )

    return model


def main() -> None:
    """主函数。"""
    print("=" * 80)
    print("Debug MMGCN forward")
    print("=" * 80)

    config = load_yaml("configs/train_mmgcn_m3ed.yaml")

    set_seed(int(config["system"]["seed"]))

    device_name = config["system"]["device"]

    if device_name == "cuda" and not torch.cuda.is_available():
        print("[WARN] Config requests cuda, but CUDA is not available. Use CPU.")
        device_name = "cpu"

    device = torch.device(device_name)

    feature_pkl_path = config["dataset"]["feature_pkl_path"]
    batch_size = int(config["train"]["batch_size"])
    num_classes = int(config["dataset"]["num_classes"])

    print("Project root:", PROJECT_ROOT)
    print("Device:", device)
    print("Feature pkl path:", feature_pkl_path)
    print("Batch size:", batch_size)

    print_section("Build DataLoader")

    dataset = M3EDTorchDataset(
        feature_pkl_path=feature_pkl_path,
        metadata_path=config["dataset"]["metadata_path"],
        label_mapping_path=config["dataset"]["label_mapping_path"],
        split="train",
        check_label_consistency=True,
        raise_on_label_mismatch=False,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=int(config["train"]["num_workers"]),
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

    print_section("Build MMGCN model")

    model = build_model(config).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["train"]["learning_rate"]),
        weight_decay=float(config["train"]["weight_decay"]),
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
        lengths=batch["lengths"],
        attention_mask=batch["attention_mask"],
        speaker_ids_int=batch["speaker_ids_int"],
        return_graph=True,
    )

    logits = outputs["logits"]

    print("Logits shape:", logits.shape)
    print("Logits flat shape:", outputs["logits_flat"].shape)
    print("Utterance nodes:", int(outputs["num_utterance_nodes"].item()))
    print("Graph nodes:", int(outputs["num_graph_nodes"].item()))
    print("Adjacency shape:", outputs["adjacency"].shape)
    print("Adjacency density:", float(outputs["adjacency_density"].item()))

    loss = criterion(
        logits.reshape(-1, num_classes),
        batch["labels"].reshape(-1),
    )

    print("Loss:", float(loss.item()))

    loss.backward()

    grad_norm = compute_grad_norm(model)

    print("Gradient norm:", grad_norm)

    optimizer.step()

    print("Optimizer step completed.")

    print_section("One-batch metrics")

    with torch.no_grad():
        logits_flat = logits.reshape(-1, num_classes)
        labels_flat = batch["labels"].reshape(-1)

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

    print("\nDebug MMGCN forward finished.")
    print("=" * 80)


if __name__ == "__main__":
    main()
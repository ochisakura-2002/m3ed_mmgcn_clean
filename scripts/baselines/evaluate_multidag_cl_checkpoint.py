"""Evaluate a MultiDAG+CL checkpoint on IEMOCAP official features."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import torch
from sklearn.metrics import confusion_matrix, recall_score
from torch.utils.data import DataLoader
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.iemocap import build_iemocap_dataloader  # noqa: E402
from models.baselines.multidag_cl import MultiDAGCLBaseline  # noqa: E402
from utils.io import ensure_dir  # noqa: E402
from utils.metrics import compute_classification_metrics  # noqa: E402
from utils.seed import set_seed  # noqa: E402


IGNORE_INDEX = -100
DEFAULT_PAST_ALL_WINDOW = 1_000_000
METRIC_KEYS = ("acc", "uar", "macro_f1", "weighted_f1")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained MultiDAG+CL checkpoint."
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to checkpoint, e.g. outputs/runs/<run_id>/checkpoints/best_model.pt.",
    )
    parser.add_argument(
        "--split",
        default="val",
        choices=("train", "val", "test"),
        help="Split to evaluate. Test is for final reporting, not checkpoint selection.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override evaluation batch size.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Override device. Defaults to config system.device.",
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="Optional tiny evaluation cap for smoke checks.",
    )
    return parser.parse_args()


def project_relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT.resolve()))
    except ValueError:
        return str(path)


def resolve_path(path_text: str) -> Path:
    path = Path(str(path_text))
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def get_training_config(config: Dict[str, Any]) -> Dict[str, Any]:
    training = dict(config.get("training", {}))
    legacy_train = dict(config.get("train", {}))
    if "batch_size" not in training and "batch_size" in legacy_train:
        training["batch_size"] = legacy_train["batch_size"]
    if "num_workers" not in training and "num_workers" in legacy_train:
        training["num_workers"] = legacy_train["num_workers"]
    training.setdefault("batch_size", 2)
    training.setdefault("num_workers", 0)
    return training


def load_checkpoint(checkpoint_path: Path, device: torch.device) -> Dict[str, Any]:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)

    if "config" not in checkpoint:
        raise KeyError("Checkpoint does not contain config.")
    if "model_state_dict" not in checkpoint:
        raise KeyError("Checkpoint does not contain model_state_dict.")
    return checkpoint


def resolve_graph_window_past(config: Dict[str, Any]) -> int:
    model_config = config.get("model", {})
    graph_config = config.get("graph", {})
    context_mode = str(graph_config.get("context_mode", "causal")).lower()
    graph_window = graph_config.get("window_past", model_config.get("window_past", 5))

    if context_mode == "causal":
        if graph_window is None:
            graph_window = model_config.get("window_past", 5)
        return int(graph_window)

    if context_mode in {"full", "past_all_causal"}:
        if graph_window is None:
            graph_window = graph_config.get("effective_window_past", DEFAULT_PAST_ALL_WINDOW)
        return int(graph_window)

    raise ValueError(
        "graph.context_mode must be one of causal, full, or past_all_causal; "
        f"got {context_mode!r}."
    )


def get_device(config: Dict[str, Any], override_device: Optional[str]) -> torch.device:
    device_name = str(override_device or config.get("system", {}).get("device", "cuda"))
    if device_name == "cuda" and not torch.cuda.is_available():
        print("[WARN] Config requests cuda, but CUDA is not available. Use CPU.")
        device_name = "cpu"
    return torch.device(device_name)


def get_label_list(config: Dict[str, Any]) -> List[str]:
    dataset_config = config["dataset"]
    num_classes = int(dataset_config["num_classes"])
    labels = dataset_config.get("label_list")
    if labels is None:
        return [str(index) for index in range(num_classes)]
    if len(labels) != num_classes:
        raise ValueError(
            f"label_list length {len(labels)} != dataset.num_classes {num_classes}"
        )
    return [str(label) for label in labels]


def validate_config(config: Dict[str, Any]) -> None:
    dataset_name = str(config.get("dataset", {}).get("name", "")).upper()
    model_name = str(config.get("model", {}).get("name", ""))
    if dataset_name != "IEMOCAP":
        raise ValueError(
            "evaluate_multidag_cl_checkpoint.py currently supports "
            f"dataset.name=IEMOCAP, got {dataset_name!r}."
        )
    if model_name != "MultiDAGCL":
        raise ValueError(f"model.name must be MultiDAGCL, got {model_name!r}.")


def build_dataloader(
    config: Dict[str, Any],
    split: str,
    batch_size: int,
) -> DataLoader:
    dataset_config = config["dataset"]
    training = get_training_config(config)
    device_name = str(config.get("system", {}).get("device", "cuda"))
    return build_iemocap_dataloader(
        feature_pkl_path=resolve_path(str(dataset_config["feature_pkl_path"])),
        split=split,
        batch_size=int(batch_size),
        valid_ratio=float(dataset_config.get("valid_ratio", 0.1)),
        val_split_strategy=str(dataset_config.get("val_split_strategy", "official_prefix")),
        seed=int(config.get("system", {}).get("seed", 42)),
        shuffle=False,
        num_workers=int(training.get("num_workers", 0)),
        pin_memory=device_name == "cuda" and torch.cuda.is_available(),
    )


def build_model(config: Dict[str, Any]) -> MultiDAGCLBaseline:
    model_config = config["model"]
    dataset_config = config["dataset"]
    active_modalities = model_config.get(
        "active_modalities",
        config.get("modality", {}).get("active_modalities", ["text", "audio", "visual"]),
    )
    return MultiDAGCLBaseline(
        text_dim=int(model_config["text_feature_dim"]),
        audio_dim=int(model_config["audio_feature_dim"]),
        visual_dim=int(model_config["visual_feature_dim"]),
        hidden_dim=int(model_config["hidden_dim"]),
        num_classes=int(model_config.get("num_classes", dataset_config["num_classes"])),
        num_speakers=int(model_config.get("num_speakers", 2)),
        window_past=resolve_graph_window_past(config),
        dropout=float(model_config.get("dropout", 0.1)),
        active_modalities=tuple(str(name) for name in active_modalities),
        num_graph_layers=int(model_config.get("num_graph_layers", config.get("graph", {}).get("num_layers", 2))),
        modality_encoder_type=str(model_config.get("modality_encoder_type", "causal_gru")),
        modality_encoder_layers=int(model_config.get("modality_encoder_layers", 1)),
    )


def move_tensor_batch(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def forward_batch(model: MultiDAGCLBaseline, batch: Dict[str, Any]) -> Dict[str, Any]:
    return model(
        text_features=batch["text_features"],
        audio_features=batch["audio_features"],
        visual_features=batch["visual_features"],
        attention_mask=batch["attention_mask"],
        speaker_ids_int=batch["speaker_ids_int"],
        labels=batch["labels"],
    )


def valid_mask(batch: Dict[str, Any]) -> torch.Tensor:
    return batch["attention_mask"].to(dtype=torch.bool) & (batch["labels"] != IGNORE_INDEX)


def collect_predictions(
    batch: Dict[str, Any],
    logits: torch.Tensor,
    split: str,
    label_list: List[str],
) -> Tuple[List[dict], List[int], List[int]]:
    probabilities = torch.softmax(logits, dim=-1)
    predictions = torch.argmax(probabilities, dim=-1)
    confidences = torch.max(probabilities, dim=-1).values
    mask = valid_mask(batch)

    y_true = [int(x) for x in batch["labels"][mask].detach().cpu().tolist()]
    y_pred = [int(x) for x in predictions[mask].detach().cpu().tolist()]

    rows: List[dict] = []
    batch_size = int(batch["labels"].shape[0])
    dialogue_ids = batch.get("dialogue_ids", [f"dialogue_{i}" for i in range(batch_size)])
    lengths = batch.get("lengths", mask.to(dtype=torch.long).sum(dim=1))

    for batch_index in range(batch_size):
        dialogue_id = str(dialogue_ids[batch_index])
        length = int(lengths[batch_index].item())
        for time_index in range(length):
            true_label = int(batch["labels"][batch_index, time_index].item())
            if true_label == IGNORE_INDEX:
                continue
            pred_label = int(predictions[batch_index, time_index].item())
            rows.append(
                {
                    "split": split,
                    "dialogue_id": dialogue_id,
                    "utterance_index": int(time_index),
                    "true_label_id": true_label,
                    "pred_label_id": pred_label,
                    "true_label_name": label_list[true_label],
                    "pred_label_name": label_list[pred_label],
                    "confidence": float(confidences[batch_index, time_index].item()),
                }
            )

    return rows, y_true, y_pred


def compute_metrics(y_true: List[int], y_pred: List[int], num_classes: int) -> Dict[str, float]:
    if not y_true:
        return {key: 0.0 for key in METRIC_KEYS}
    return compute_classification_metrics(
        y_true=y_true,
        y_pred=y_pred,
        labels=list(range(int(num_classes))),
    )


def build_confusion_matrix_df(
    y_true: List[int],
    y_pred: List[int],
    label_list: List[str],
) -> pd.DataFrame:
    labels = list(range(len(label_list)))
    if not y_true:
        matrix = [[0 for _ in labels] for _ in labels]
    else:
        matrix = confusion_matrix(y_true, y_pred, labels=labels)
    return pd.DataFrame(
        matrix,
        index=[f"true_{label}" for label in label_list],
        columns=[f"pred_{label}" for label in label_list],
    )


def compute_per_class_recall_rows(
    y_true: List[int],
    y_pred: List[int],
    label_list: List[str],
) -> List[dict]:
    labels = list(range(len(label_list)))
    if not y_true:
        recalls = [0.0 for _ in labels]
    else:
        recalls = recall_score(
            y_true,
            y_pred,
            labels=labels,
            average=None,
            zero_division=0,
        )
    return [
        {
            "label_id": int(label_id),
            "label_name": label_list[label_id],
            "recall": float(recall_value),
        }
        for label_id, recall_value in zip(labels, recalls)
    ]


@torch.no_grad()
def evaluate(
    model: MultiDAGCLBaseline,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
    label_list: List[str],
    split: str,
    max_batches: Optional[int],
) -> Tuple[float, Dict[str, float], List[dict], List[int], List[int]]:
    model.eval()
    total_loss = 0.0
    total_items = 0
    prediction_rows: List[dict] = []
    all_true: List[int] = []
    all_pred: List[int] = []

    for batch_index, batch in enumerate(tqdm(loader, desc=f"Evaluate {split}", ncols=100), start=1):
        if max_batches is not None and batch_index > int(max_batches):
            break

        batch = move_tensor_batch(batch, device)
        output = forward_batch(model, batch)
        loss = output["loss"]
        if loss is None:
            raise RuntimeError("MultiDAGCLBaseline returned no loss for a labeled batch.")

        item_count = max(int(valid_mask(batch).sum().item()), 1)
        total_loss += float(loss.item()) * float(item_count)
        total_items += item_count

        rows, y_true, y_pred = collect_predictions(
            batch=batch,
            logits=output["logits"],
            split=split,
            label_list=label_list,
        )
        prediction_rows.extend(rows)
        all_true.extend(y_true)
        all_pred.extend(y_pred)

    metrics = compute_metrics(all_true, all_pred, int(num_classes))
    avg_loss = total_loss / float(max(total_items, 1))
    return avg_loss, metrics, prediction_rows, all_true, all_pred


def get_run_dir_from_checkpoint(checkpoint_path: Path) -> Path:
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
    ensure_dir(eval_dir)
    metrics_row = {
        "split": split,
        "checkpoint": project_relative(checkpoint_path),
        "loss": float(loss),
        "accuracy": float(metrics["acc"]),
        "acc": float(metrics["acc"]),
        "uar": float(metrics["uar"]),
        "macro_f1": float(metrics["macro_f1"]),
        "weighted_f1": float(metrics["weighted_f1"]),
    }
    pd.DataFrame([metrics_row]).to_csv(eval_dir / "metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(prediction_rows).to_csv(eval_dir / "predictions.csv", index=False, encoding="utf-8-sig")
    build_confusion_matrix_df(y_true, y_pred, label_list).to_csv(
        eval_dir / "confusion_matrix.csv",
        encoding="utf-8-sig",
    )
    pd.DataFrame(compute_per_class_recall_rows(y_true, y_pred, label_list)).to_csv(
        eval_dir / "per_class_recall.csv",
        index=False,
        encoding="utf-8-sig",
    )


def metrics_summary(loss: float, metrics: Dict[str, float]) -> str:
    return (
        f"loss={loss:.6f} "
        f"acc={metrics['acc']:.6f} "
        f"uar={metrics['uar']:.6f} "
        f"macro_f1={metrics['macro_f1']:.6f} "
        f"weighted_f1={metrics['weighted_f1']:.6f}"
    )


def main() -> None:
    args = parse_args()
    checkpoint_path = resolve_path(args.checkpoint)

    temp_checkpoint = load_checkpoint(checkpoint_path, torch.device("cpu"))
    config = temp_checkpoint["config"]
    validate_config(config)

    seed = int(config.get("system", {}).get("seed", 42))
    set_seed(seed)
    device = get_device(config, args.device)
    checkpoint = load_checkpoint(checkpoint_path, device)
    config = checkpoint["config"]

    batch_size = int(args.batch_size or get_training_config(config)["batch_size"])
    label_list = get_label_list(config)
    num_classes = int(config["dataset"]["num_classes"])

    print("=" * 100)
    print("Evaluate MultiDAG+CL checkpoint")
    print("=" * 100)
    print("Project root:", PROJECT_ROOT)
    print("Checkpoint:", checkpoint_path)
    print("Dataset:", config["dataset"]["name"])
    print("Model:", config["model"]["name"])
    print("Split:", args.split)
    print("Batch size:", batch_size)
    print("Device:", device)
    print("=" * 100)

    loader = build_dataloader(config=config, split=args.split, batch_size=batch_size)
    model = build_model(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)

    loss, metrics, prediction_rows, y_true, y_pred = evaluate(
        model=model,
        loader=loader,
        device=device,
        num_classes=num_classes,
        label_list=label_list,
        split=args.split,
        max_batches=args.max_batches,
    )

    run_dir = get_run_dir_from_checkpoint(checkpoint_path)
    eval_dir = run_dir / "logs" / "evaluations" / f"{args.split}_{checkpoint_path.stem}"
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
    print(metrics_summary(loss, metrics))
    print("Output dir:", eval_dir)
    print("=" * 100)


if __name__ == "__main__":
    main()

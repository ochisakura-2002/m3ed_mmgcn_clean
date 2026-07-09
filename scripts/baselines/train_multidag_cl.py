"""Train the project MultiDAG+CL baseline on IEMOCAP official features.

This entry stays separate from the MMGCN training script while writing the
same run-directory schema expected by the existing analysis scripts.
"""

from __future__ import annotations

import argparse
import copy
from datetime import datetime
from itertools import islice
from pathlib import Path
import sys
from typing import Any, Dict, Iterable, List, Optional, Tuple

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
from utils.io import ensure_dir, load_yaml, sanitize_name, save_yaml  # noqa: E402
from utils.metrics import compute_classification_metrics  # noqa: E402
from utils.seed import set_seed  # noqa: E402


IGNORE_INDEX = -100
DEFAULT_PAST_ALL_WINDOW = 1_000_000
METRIC_KEYS = ("acc", "uar", "macro_f1", "weighted_f1")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train MultiDAG+CL on IEMOCAP official MMGCN features."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to a MultiDAG+CL YAML training config.",
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


def optional_int(value: Any) -> Optional[int]:
    if value is None or value == "":
        return None
    return int(value)


def get_training_config(config: Dict[str, Any]) -> Dict[str, Any]:
    training = dict(config.get("training", {}))
    legacy_train = dict(config.get("train", {}))

    if "batch_size" not in training and "batch_size" in legacy_train:
        training["batch_size"] = legacy_train["batch_size"]
    if "lr" not in training and "learning_rate" in legacy_train:
        training["lr"] = legacy_train["learning_rate"]
    if "epochs" not in training and "max_epochs" in legacy_train:
        training["epochs"] = legacy_train["max_epochs"]
    if "weight_decay" not in training and "weight_decay" in legacy_train:
        training["weight_decay"] = legacy_train["weight_decay"]
    if "num_workers" not in training and "num_workers" in legacy_train:
        training["num_workers"] = legacy_train["num_workers"]

    training.setdefault("epochs", 1)
    training.setdefault("batch_size", 2)
    training.setdefault("lr", 1e-3)
    training.setdefault("weight_decay", 0.0)
    training.setdefault("grad_clip", 0.0)
    training.setdefault("select_best_by", "val_weighted_f1")
    training.setdefault("num_workers", 0)
    return training


def validate_config(config: Dict[str, Any]) -> None:
    dataset_name = str(config.get("dataset", {}).get("name", "")).upper()
    if dataset_name != "IEMOCAP":
        raise ValueError(
            "scripts/baselines/train_multidag_cl.py currently supports "
            f"dataset.name=IEMOCAP, got {dataset_name!r}."
        )
    feature_path = resolve_path(str(config["dataset"].get("feature_pkl_path", "")))
    if not feature_path.exists():
        raise FileNotFoundError(
            "IEMOCAP smoke training was not run because feature file is missing: "
            f"{project_relative(feature_path)}"
        )

    model_name = str(config.get("model", {}).get("name", ""))
    if model_name != "MultiDAGCL":
        raise ValueError(f"model.name must be MultiDAGCL, got {model_name!r}.")

    training = get_training_config(config)
    if str(training.get("select_best_by")) != "val_weighted_f1":
        raise ValueError(
            "MultiDAG+CL checkpoint selection is restricted to "
            "training.select_best_by=val_weighted_f1 for this integration."
        )

    dataset_classes = int(config["dataset"]["num_classes"])
    model_classes = int(config["model"].get("num_classes", dataset_classes))
    if dataset_classes != model_classes:
        raise ValueError(
            "dataset.num_classes and model.num_classes must match: "
            f"{dataset_classes} != {model_classes}"
        )


def normalized_config(config: Dict[str, Any]) -> Dict[str, Any]:
    run_config = copy.deepcopy(config)
    training = get_training_config(run_config)

    run_config["training"] = training
    run_config.setdefault("project", {})
    run_config["project"].setdefault("name", "m3ed_mmgcn_clean")
    run_config["project"].setdefault(
        "experiment_name",
        run_config.get("output", {}).get(
            "experiment_name",
            "iemocap_multidag_cl_experiment",
        ),
    )
    run_config.setdefault("system", {})
    run_config["system"].setdefault("seed", 42)
    run_config["system"].setdefault("device", "cuda")
    run_config["system"].setdefault("output_dir", "outputs")

    run_config["train"] = {
        "batch_size": int(training["batch_size"]),
        "learning_rate": float(training["lr"]),
        "weight_decay": float(training.get("weight_decay", 0.0)),
        "max_epochs": int(training["epochs"]),
        "num_workers": int(training.get("num_workers", 0)),
    }

    graph = run_config.setdefault("graph", {})
    effective_window, semantics = resolve_graph_window_past(run_config)
    graph["effective_window_past"] = int(effective_window)
    graph["effective_context_semantics"] = semantics

    runtime = run_config.setdefault("runtime", {})
    runtime["train_script"] = "scripts/baselines/train_multidag_cl.py"
    runtime["checkpoint_selection"] = "validation_weighted_f1"
    runtime["test_split_used_for_selection"] = False
    return run_config


def resolve_graph_window_past(config: Dict[str, Any]) -> Tuple[int, str]:
    model_config = config.get("model", {})
    graph_config = config.get("graph", {})
    context_mode = str(graph_config.get("context_mode", "causal")).lower()
    graph_window = graph_config.get("window_past", model_config.get("window_past", 5))

    if context_mode == "causal":
        if graph_window is None:
            graph_window = model_config.get("window_past", 5)
        return int(graph_window), "causal"

    if context_mode in {"full", "past_all_causal"}:
        if graph_window is None:
            graph_window = DEFAULT_PAST_ALL_WINDOW
        return int(graph_window), "past_all_causal"

    raise ValueError(
        "graph.context_mode must be one of causal, full, or past_all_causal; "
        f"got {context_mode!r}."
    )


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


def get_device(config: Dict[str, Any]) -> torch.device:
    device_name = str(config.get("system", {}).get("device", "cuda"))
    if device_name == "cuda" and not torch.cuda.is_available():
        print("[WARN] Config requests cuda, but CUDA is not available. Use CPU.")
        device_name = "cpu"
    return torch.device(device_name)


def prepare_run_environment(config: Dict[str, Any], config_path: Path) -> Dict[str, Path]:
    output_config = config.get("output", {})
    run_root_text = output_config.get("run_root")
    if not run_root_text:
        run_root_text = str(Path(str(config["system"].get("output_dir", "outputs"))) / "runs")

    run_root = resolve_path(str(run_root_text))
    experiment_name = sanitize_name(
        str(output_config.get("experiment_name", config["project"]["experiment_name"]))
    )
    run_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{experiment_name}"
    run_dir = run_root / run_id

    logs_dir = run_dir / "logs"
    checkpoints_dir = run_dir / "checkpoints"
    figures_dir = run_dir / "figures"
    for path in (logs_dir, checkpoints_dir, figures_dir):
        ensure_dir(path)

    save_yaml(config, logs_dir / "experiment_config.yaml")

    latest_path = run_root.parent / "latest_run.txt"
    ensure_dir(latest_path.parent)
    with latest_path.open("w", encoding="utf-8") as file:
        file.write(f"run_id={run_id}\n")
        file.write(f"run_dir={project_relative(run_dir)}\n")
        file.write(f"experiment_name={experiment_name}\n")
        file.write(f"config_path={project_relative(config_path)}\n")

    return {
        "run_id": run_id,
        "run_dir": run_dir,
        "logs_dir": logs_dir,
        "checkpoints_dir": checkpoints_dir,
        "figures_dir": figures_dir,
        "latest_run_path": latest_path,
    }


def build_dataloader(
    config: Dict[str, Any],
    split: str,
    shuffle: bool,
    batch_size: Optional[int] = None,
) -> DataLoader:
    dataset_config = config["dataset"]
    training = get_training_config(config)
    effective_batch_size = int(batch_size or training["batch_size"])
    device_name = str(config.get("system", {}).get("device", "cuda"))

    return build_iemocap_dataloader(
        feature_pkl_path=resolve_path(str(dataset_config["feature_pkl_path"])),
        split=split,
        batch_size=effective_batch_size,
        valid_ratio=float(dataset_config.get("valid_ratio", 0.1)),
        val_split_strategy=str(dataset_config.get("val_split_strategy", "official_prefix")),
        seed=int(config.get("system", {}).get("seed", 42)),
        shuffle=bool(shuffle),
        num_workers=int(training.get("num_workers", 0)),
        pin_memory=device_name == "cuda" and torch.cuda.is_available(),
    )


def build_model(config: Dict[str, Any]) -> MultiDAGCLBaseline:
    model_config = config["model"]
    dataset_config = config["dataset"]
    effective_window_past, _ = resolve_graph_window_past(config)
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
        window_past=int(effective_window_past),
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


def forward_batch(
    model: MultiDAGCLBaseline,
    batch: Dict[str, Any],
) -> Dict[str, Any]:
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


def valid_count(batch: Dict[str, Any]) -> int:
    return int(valid_mask(batch).sum().item())


def limited_batches(
    loader: DataLoader,
    max_batches: Optional[int],
) -> Iterable[Dict[str, Any]]:
    if max_batches is None:
        return loader
    return islice(loader, max(0, int(max_batches)))


def effective_num_batches(
    loader: DataLoader,
    max_batches: Optional[int],
) -> Optional[int]:
    try:
        loader_length = len(loader)
    except TypeError:
        loader_length = None

    if max_batches is None:
        return loader_length

    cap = max(0, int(max_batches))
    if loader_length is None:
        return cap
    return min(loader_length, cap)


def current_learning_rate(optimizer: torch.optim.Optimizer) -> float:
    if not optimizer.param_groups:
        return 0.0
    return float(optimizer.param_groups[0].get("lr", 0.0))


def compute_gradient_norm(parameters: Iterable[torch.nn.Parameter]) -> float:
    total_norm_sq = 0.0
    for parameter in parameters:
        if parameter.grad is None:
            continue
        parameter_norm = parameter.grad.detach().data.norm(2)
        total_norm_sq += float(parameter_norm.item()) ** 2
    return total_norm_sq ** 0.5


def collect_metric_labels(
    batch: Dict[str, Any],
    logits: torch.Tensor,
) -> Tuple[List[int], List[int]]:
    predictions = torch.argmax(logits, dim=-1)
    mask = valid_mask(batch)
    y_true = [int(x) for x in batch["labels"][mask].detach().cpu().tolist()]
    y_pred = [int(x) for x in predictions[mask].detach().cpu().tolist()]
    return y_true, y_pred


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
    labels = list(range(int(num_classes)))
    return compute_classification_metrics(y_true=y_true, y_pred=y_pred, labels=labels)


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


def train_one_epoch(
    model: MultiDAGCLBaseline,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip: float,
    max_train_batches: Optional[int],
    num_classes: int,
    epoch: int,
    total_epochs: int,
) -> Tuple[float, Dict[str, float], float]:
    model.train()
    total_loss = 0.0
    total_items = 0
    total_grad_norm = 0.0
    grad_steps = 0
    all_true: List[int] = []
    all_pred: List[int] = []

    progress = tqdm(
        limited_batches(loader, max_train_batches),
        total=effective_num_batches(loader, max_train_batches),
        desc=f"Train epoch {epoch}/{total_epochs}",
        ncols=100,
    )
    for batch_index, batch in enumerate(progress, start=1):
        batch = move_tensor_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        output = forward_batch(model, batch)
        loss = output["loss"]
        if loss is None:
            raise RuntimeError("MultiDAGCLBaseline returned no loss for a labeled batch.")
        loss.backward()

        if float(grad_clip) > 0.0:
            grad_norm = float(
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip)).item()
            )
        else:
            grad_norm = compute_gradient_norm(model.parameters())

        optimizer.step()

        valid_items = valid_count(batch)
        item_count = max(valid_items, 1)
        batch_loss = float(loss.item())
        total_loss += batch_loss * float(item_count)
        total_items += item_count
        total_grad_norm += float(grad_norm)
        grad_steps += 1
        y_true, y_pred = collect_metric_labels(batch=batch, logits=output["logits"])
        all_true.extend(y_true)
        all_pred.extend(y_pred)
        running_avg = total_loss / float(max(total_items, 1))
        progress.set_postfix(
            loss=f"{batch_loss:.4f}",
            avg=f"{running_avg:.4f}",
            lr=f"{current_learning_rate(optimizer):.3g}",
            grad=f"{grad_norm:.3g}",
            valid=valid_items,
        )

    train_metrics = compute_metrics(all_true, all_pred, int(num_classes))
    avg_grad_norm = total_grad_norm / float(max(grad_steps, 1))
    return total_loss / float(max(total_items, 1)), train_metrics, avg_grad_norm


@torch.no_grad()
def evaluate(
    model: MultiDAGCLBaseline,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
    label_list: List[str],
    split: str,
    max_eval_batches: Optional[int],
) -> Tuple[float, Dict[str, float], List[dict], List[int], List[int]]:
    model.eval()
    total_loss = 0.0
    total_items = 0
    prediction_rows: List[dict] = []
    all_true: List[int] = []
    all_pred: List[int] = []

    progress = tqdm(
        limited_batches(loader, max_eval_batches),
        total=effective_num_batches(loader, max_eval_batches),
        desc=f"Eval {split}",
        ncols=100,
    )
    for batch_index, batch in enumerate(progress, start=1):
        batch = move_tensor_batch(batch, device)
        output = forward_batch(model, batch)
        loss = output["loss"]
        if loss is None:
            raise RuntimeError("MultiDAGCLBaseline returned no loss for a labeled batch.")

        valid_items = valid_count(batch)
        item_count = max(valid_items, 1)
        batch_loss = float(loss.item())
        total_loss += batch_loss * float(item_count)
        total_items += item_count
        running_avg = total_loss / float(max(total_items, 1))
        progress.set_postfix(
            loss=f"{batch_loss:.4f}",
            avg=f"{running_avg:.4f}",
            valid=valid_items,
        )

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
    loss = total_loss / float(max(total_items, 1))
    return loss, metrics, prediction_rows, all_true, all_pred


def save_checkpoint(
    path: Path,
    model: MultiDAGCLBaseline,
    optimizer: torch.optim.Optimizer,
    config: Dict[str, Any],
    epoch: int,
    train_loss: float,
    val_loss: float,
    val_metrics: Dict[str, float],
    monitor_value: float,
) -> None:
    ensure_dir(path.parent)
    torch.save(
        {
            "epoch": int(epoch),
            "model_name": "MultiDAGCL",
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": config,
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "val_metrics": val_metrics,
            "monitor_metric": "val_weighted_f1",
            "monitor_value": float(monitor_value),
            "checkpoint_selection": "validation_weighted_f1",
        },
        path,
    )


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


def save_best_validation_outputs(
    logs_dir: Path,
    checkpoint_path: Path,
    val_loss: float,
    val_metrics: Dict[str, float],
    prediction_rows: List[dict],
    y_true: List[int],
    y_pred: List[int],
    label_list: List[str],
) -> None:
    pd.DataFrame(prediction_rows).to_csv(
        logs_dir / "val_predictions_best.csv",
        index=False,
        encoding="utf-8-sig",
    )
    build_confusion_matrix_df(y_true, y_pred, label_list).to_csv(
        logs_dir / "confusion_matrix_best.csv",
        encoding="utf-8-sig",
    )
    pd.DataFrame(compute_per_class_recall_rows(y_true, y_pred, label_list)).to_csv(
        logs_dir / "per_class_recall_best.csv",
        index=False,
        encoding="utf-8-sig",
    )
    save_evaluation_outputs(
        eval_dir=logs_dir / "evaluations" / "val_best_model",
        split="val",
        checkpoint_path=checkpoint_path,
        loss=val_loss,
        metrics=val_metrics,
        prediction_rows=prediction_rows,
        y_true=y_true,
        y_pred=y_pred,
        label_list=label_list,
    )


def save_epoch_metrics(logs_dir: Path, epoch_rows: List[Dict[str, Any]]) -> None:
    pd.DataFrame(epoch_rows).to_csv(
        logs_dir / "epoch_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )


def print_config_summary(config: Dict[str, Any], run_paths: Dict[str, Path]) -> None:
    graph = config.get("graph", {})
    training = get_training_config(config)
    print("=" * 100)
    print("Train MultiDAG+CL")
    print("=" * 100)
    print("Project root:", PROJECT_ROOT)
    print("Run dir:", run_paths["run_dir"])
    print("Dataset:", config["dataset"]["name"])
    print("Model:", config["model"]["name"])
    print("Feature path:", config["dataset"].get("feature_pkl_path", ""))
    print("Batch size:", training["batch_size"])
    print("Learning rate:", training["lr"])
    print("Epochs:", training["epochs"])
    print("Max train batches:", training.get("max_train_batches"))
    print("Max val batches:", training.get("max_val_batches"))
    print("Monitor metric: val_weighted_f1")
    print("Graph context_mode:", graph.get("context_mode", "causal"))
    print("Graph effective_context_semantics:", graph.get("effective_context_semantics"))
    print("Graph effective_window_past:", graph.get("effective_window_past"))
    print("Active modalities:", config["model"].get("active_modalities", ["text", "audio", "visual"]))
    print("Checkpoint selection: validation Weighted-F1 only; test split is not used.")
    print("=" * 100)


def main() -> None:
    args = parse_args()
    config_path = resolve_path(args.config)
    raw_config = load_yaml(str(config_path))
    validate_config(raw_config)
    config = normalized_config(raw_config)

    seed = int(config.get("system", {}).get("seed", 42))
    set_seed(seed)
    device = get_device(config)

    run_paths = prepare_run_environment(config, config_path)
    print_config_summary(config, run_paths)

    training = get_training_config(config)
    train_loader = build_dataloader(config, split="train", shuffle=True)
    val_loader = build_dataloader(config, split="val", shuffle=False)
    model = build_model(config).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["lr"]),
        weight_decay=float(training.get("weight_decay", 0.0)),
    )

    label_list = get_label_list(config)
    num_classes = int(config["dataset"]["num_classes"])
    max_train_batches = optional_int(training.get("max_train_batches"))
    max_val_batches = optional_int(training.get("max_val_batches"))
    grad_clip = float(training.get("grad_clip", 0.0))
    total_epochs = int(training["epochs"])

    best_value = -float("inf")
    best_epoch = -1
    epoch_rows: List[Dict[str, Any]] = []

    for epoch in range(1, total_epochs + 1):
        print("\n" + "=" * 100)
        print(f"Epoch {epoch}/{total_epochs}")
        print("=" * 100)

        train_loss, train_metrics, grad_norm = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            grad_clip=grad_clip,
            max_train_batches=max_train_batches,
            num_classes=num_classes,
            epoch=epoch,
            total_epochs=total_epochs,
        )
        val_loss, val_metrics, val_predictions, y_true, y_pred = evaluate(
            model=model,
            loader=val_loader,
            device=device,
            num_classes=num_classes,
            label_list=label_list,
            split="val",
            max_eval_batches=max_val_batches,
        )

        row = {
            "epoch": int(epoch),
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "val_acc": float(val_metrics["acc"]),
            "val_weighted_f1": float(val_metrics["weighted_f1"]),
            "val_macro_f1": float(val_metrics["macro_f1"]),
            "val_uar": float(val_metrics["uar"]),
            "train_acc": float(train_metrics["acc"]),
            "train_weighted_f1": float(train_metrics["weighted_f1"]),
            "train_macro_f1": float(train_metrics["macro_f1"]),
            "train_uar": float(train_metrics["uar"]),
            "lr": float(current_learning_rate(optimizer)),
            "grad_norm": float(grad_norm),
        }
        epoch_rows.append(row)
        save_epoch_metrics(run_paths["logs_dir"], epoch_rows)

        monitor_value = float(row["val_weighted_f1"])
        last_checkpoint = run_paths["checkpoints_dir"] / "last_model.pt"
        save_checkpoint(
            path=last_checkpoint,
            model=model,
            optimizer=optimizer,
            config=config,
            epoch=epoch,
            train_loss=train_loss,
            val_loss=val_loss,
            val_metrics=val_metrics,
            monitor_value=monitor_value,
        )

        print(
            f"Epoch {epoch:02d}/{total_epochs:02d} | "
            f"train_loss={train_loss:.6f} | "
            f"train_acc={train_metrics['acc']:.6f} | "
            f"train_weighted_f1={train_metrics['weighted_f1']:.6f} | "
            f"val_loss={val_loss:.6f} | "
            f"val_acc={val_metrics['acc']:.6f} | "
            f"val_weighted_f1={val_metrics['weighted_f1']:.6f} | "
            f"val_macro_f1={val_metrics['macro_f1']:.6f} | "
            f"val_uar={val_metrics['uar']:.6f} | "
            f"lr={row['lr']:.6g} | "
            f"grad_norm={grad_norm:.6f}",
            flush=True,
        )

        if monitor_value > best_value:
            best_value = monitor_value
            best_epoch = epoch
            best_checkpoint = run_paths["checkpoints_dir"] / "best_model.pt"
            save_checkpoint(
                path=best_checkpoint,
                model=model,
                optimizer=optimizer,
                config=config,
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                val_metrics=val_metrics,
                monitor_value=monitor_value,
            )
            save_best_validation_outputs(
                logs_dir=run_paths["logs_dir"],
                checkpoint_path=best_checkpoint,
                val_loss=val_loss,
                val_metrics=val_metrics,
                prediction_rows=val_predictions,
                y_true=y_true,
                y_pred=y_pred,
                label_list=label_list,
            )
            print(
                f"[BEST] epoch={best_epoch:02d} val_weighted_f1={best_value:.6f}",
                flush=True,
            )

    save_epoch_metrics(run_paths["logs_dir"], epoch_rows)

    if best_epoch < 0:
        raise RuntimeError("No best checkpoint was selected.")

    print("\n" + "=" * 100)
    print("Training finished.")
    print("=" * 100)
    print("Run dir:", run_paths["run_dir"])
    print("Best epoch:", best_epoch)
    print("Best val_weighted_f1:", best_value)
    print("Epoch metrics:", run_paths["logs_dir"] / "epoch_metrics.csv")
    print("Best checkpoint:", run_paths["checkpoints_dir"] / "best_model.pt")
    print("Last checkpoint:", run_paths["checkpoints_dir"] / "last_model.pt")
    print("=" * 100)


if __name__ == "__main__":
    main()

"""Shared runtime utilities for the two Task 2 causal graph baselines."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Dataset


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.iemocap.official_feature_dataset import (  # noqa: E402
    build_iemocap_dataloader,
    iemocap_dialogue_collate_fn,
)
from models.baselines.causal_baseline_registry import (  # noqa: E402
    CAUSAL_GSMCC_NAME,
    build_new_causal_baseline,
    get_new_causal_model_family,
    normalize_new_causal_model_name,
    validate_new_causal_model_config,
)
from models.baselines.gsmcc import compute_causal_gsmcc_loss  # noqa: E402
from utils.metrics import (  # noqa: E402
    compute_classification_metrics,
    compute_confusion_matrix,
    compute_per_class_recall,
)
from utils.iemocap_features import validate_iemocap_feature_config  # noqa: E402
from utils.evaluation import build_prediction_row, compute_calibration_metrics  # noqa: E402


IGNORE_INDEX = -100


def resolve_path(path_text: str) -> Path:
    path = Path(str(path_text)).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def project_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def load_yaml_config(path: Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise TypeError(f"{path} must contain a YAML mapping.")
    return config


def save_yaml_config(config: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(dict(config), file, sort_keys=False, allow_unicode=True)


def normalized_training_config(config: Mapping[str, Any]) -> Dict[str, Any]:
    result = copy.deepcopy(dict(config))
    training = dict(result.get("training", {}))
    optimizer = dict(result.get("optimizer", {}))
    scheduler = dict(result.get("scheduler", {}))
    training.setdefault("epochs", 30)
    training.setdefault("batch_size", 8)
    training.setdefault("seed", result.get("system", {}).get("seed", 42))
    training.setdefault("select_best_by", "val_weighted_f1")
    training.setdefault("grad_clip", 1.0)
    training.setdefault("num_workers", 0)
    optimizer.setdefault("name", "AdamW")
    optimizer.setdefault("learning_rate", 5.0e-4)
    optimizer.setdefault("weight_decay", 1.0e-4)
    scheduler.setdefault("name", "none")
    result["training"] = training
    result["optimizer"] = optimizer
    result["scheduler"] = scheduler
    result.setdefault("system", {})
    result["system"].setdefault("seed", int(training["seed"]))
    result["system"].setdefault("device", "cuda")
    result.setdefault("output", {})
    result["output"].setdefault("run_root", "outputs/runs")
    result.setdefault("run_name", normalize_new_causal_model_name(result["model"]["name"]))
    result.setdefault("causal_contract_version", "1.0")
    return result


def validate_runtime_config(config: Mapping[str, Any]) -> None:
    validate_new_causal_model_config(config)
    training = config.get("training", {})
    optimizer = config.get("optimizer", {})
    scheduler = config.get("scheduler", {})
    if int(training.get("epochs", 0)) <= 0:
        raise ValueError("training.epochs must be positive.")
    if int(training.get("batch_size", 0)) <= 0:
        raise ValueError("training.batch_size must be positive.")
    if str(training.get("select_best_by", "")) != "val_weighted_f1":
        raise ValueError("training.select_best_by must be val_weighted_f1.")
    if float(training.get("grad_clip", 0.0)) < 0:
        raise ValueError("training.grad_clip must be non-negative.")
    if str(optimizer.get("name", "")).lower() not in {"adam", "adamw"}:
        raise ValueError("optimizer.name must be Adam or AdamW.")
    if float(optimizer.get("learning_rate", 0.0)) <= 0:
        raise ValueError("optimizer.learning_rate must be positive.")
    if str(scheduler.get("name", "none")).lower() != "none":
        raise ValueError("Only scheduler.name=none is supported for the initial benchmark.")
    dataset_name = str(config.get("dataset", {}).get("name", "")).upper()
    if dataset_name not in {"IEMOCAP", "SYNTHETIC"}:
        raise ValueError("dataset.name must be IEMOCAP or SYNTHETIC.")


def get_device(config: Mapping[str, Any], override: Optional[str] = None) -> torch.device:
    requested = str(override or config.get("system", {}).get("device", "cuda"))
    if requested.startswith("cuda") and not torch.cuda.is_available():
        print("[WARN] CUDA requested but unavailable; using CPU.")
        requested = "cpu"
    return torch.device(requested)


def get_label_list(config: Mapping[str, Any]) -> List[str]:
    dataset = config["dataset"]
    num_classes = int(dataset["num_classes"])
    labels = dataset.get("label_list")
    if labels is None:
        return [str(index) for index in range(num_classes)]
    if len(labels) != num_classes:
        raise ValueError("dataset.label_list length must equal dataset.num_classes.")
    return [str(value) for value in labels]


def verify_feature_sha256(config: Mapping[str, Any]) -> Optional[str]:
    return validate_iemocap_feature_config(config, PROJECT_ROOT)


class SyntheticDialogueDataset(Dataset):
    """Small deterministic dialogue dataset used only by local smoke tests."""

    _SPLIT_OFFSET = {"train": 0, "val": 10_000, "test": 20_000}

    def __init__(self, config: Mapping[str, Any], split: str) -> None:
        self.config = config
        self.split = str(split)
        if self.split not in self._SPLIT_OFFSET:
            raise ValueError(f"Unsupported synthetic split: {split}")
        synthetic = config.get("synthetic", {})
        self.count = int(synthetic.get("split_sizes", {}).get(self.split, 4))
        self.seq_len = int(synthetic.get("sequence_length", 5))
        self.seed = int(config.get("system", {}).get("seed", 42))
        if self.count <= 0 or self.seq_len < 2:
            raise ValueError("Synthetic split size must be positive and sequence_length >= 2.")

    def __len__(self) -> int:
        return self.count

    def __getitem__(self, index: int) -> Dict[str, Any]:
        model = self.config["model"]
        generator = torch.Generator(device="cpu").manual_seed(
            self.seed + self._SPLIT_OFFSET[self.split] + int(index)
        )
        length = self.seq_len - (int(index) % 2)
        num_classes = int(model["num_classes"])
        dialogue_id = f"synthetic_{self.split}_{index:03d}"
        return {
            "dialogue_id": dialogue_id,
            "utterance_ids": [f"{dialogue_id}_utt_{time:03d}" for time in range(length)],
            "sentences": ["synthetic" for _ in range(length)],
            "text_features": torch.randn(length, int(model["text_dim"]), generator=generator),
            "audio_features": torch.randn(length, int(model["audio_dim"]), generator=generator),
            "visual_features": torch.randn(length, int(model["visual_dim"]), generator=generator),
            "labels": torch.randint(0, num_classes, (length,), generator=generator),
            "speaker_ids_int": torch.arange(length, dtype=torch.long) % int(model.get("num_speakers", 2)),
            "length": length,
        }


def build_dataloader(
    config: Mapping[str, Any],
    split: str,
    shuffle: bool,
    batch_size: Optional[int] = None,
) -> DataLoader:
    dataset = config["dataset"]
    training = config["training"]
    size = int(batch_size or training["batch_size"])
    workers = int(training.get("num_workers", 0))
    if str(dataset.get("name", "")).upper() == "IEMOCAP":
        return build_iemocap_dataloader(
            feature_pkl_path=resolve_path(str(dataset["feature_pkl_path"])),
            split=split,
            batch_size=size,
            valid_ratio=float(dataset.get("valid_ratio", 0.1)),
            val_split_strategy=str(dataset.get("val_split_strategy", "official_prefix")),
            val_session_id=dataset.get("val_session_id"),
            seed=int(config.get("system", {}).get("seed", 42)),
            shuffle=bool(shuffle),
            num_workers=workers,
            pin_memory=str(config.get("system", {}).get("device", "")).startswith("cuda"),
        )
    generator = torch.Generator(device="cpu").manual_seed(
        int(config.get("system", {}).get("seed", 42))
    )
    return DataLoader(
        SyntheticDialogueDataset(config, split),
        batch_size=size,
        shuffle=bool(shuffle),
        num_workers=workers,
        collate_fn=iemocap_dialogue_collate_fn,
        generator=generator,
    )


def move_batch(batch: Mapping[str, Any], device: torch.device) -> Dict[str, Any]:
    moved = {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }
    assert_batch_lengths(moved)
    return moved


def assert_batch_lengths(batch: Mapping[str, Any]) -> None:
    mask_lengths = batch["attention_mask"].long().sum(dim=1)
    if not torch.equal(mask_lengths, batch["lengths"].long()):
        raise ValueError(
            "attention_mask.long().sum(dim=1) must exactly equal lengths; "
            f"mask={mask_lengths.tolist()}, lengths={batch['lengths'].tolist()}."
        )


def forward_batch(
    model: torch.nn.Module,
    config: Mapping[str, Any],
    batch: Mapping[str, Any],
    return_aux: bool = True,
) -> Dict[str, Any]:
    output = model(
        text_features=batch["text_features"],
        audio_features=batch["audio_features"],
        visual_features=batch["visual_features"],
        attention_mask=batch["attention_mask"],
        lengths=batch["lengths"],
        speaker_ids_int=batch["speaker_ids_int"],
        return_aux=bool(return_aux),
    )
    if isinstance(output, dict):
        return output
    return {"logits": output}


def compute_batch_loss(
    config: Mapping[str, Any],
    output: Mapping[str, Any],
    batch: Mapping[str, Any],
) -> Dict[str, torch.Tensor]:
    family = get_new_causal_model_family(config)
    if family == "gsmcc":
        loss = config.get("loss", {})
        result = compute_causal_gsmcc_loss(
            output["logits"],
            batch["labels"],
            batch["attention_mask"],
            output["low_frequency_modal_repr"],
            output["high_frequency_modal_repr"],
            classification_weight=float(loss.get("classification_weight", 1.0)),
            consistency_weight=float(loss.get("consistency_weight", 0.0)),
            complementarity_weight=float(loss.get("complementarity_weight", 0.0)),
        )
        if float(loss.get("consistency_weight", 0.0)) == 0.0:
            result["consistency_loss"] = result["consistency_loss"] * 0.0
        if float(loss.get("complementarity_weight", 0.0)) == 0.0:
            result["complementarity_loss"] = result["complementarity_loss"] * 0.0
        return result
    mode = str(config.get("loss", {}).get("class_weight_mode", "none")).lower()
    if mode != "none":
        raise ValueError("Only loss.class_weight_mode=none is supported in Task 2.")
    valid = batch["attention_mask"].bool() & (batch["labels"] >= 0)
    classification = F.cross_entropy(output["logits"][valid], batch["labels"][valid].long())
    zero = classification.detach() * 0.0
    return {
        "classification_loss": classification,
        "consistency_loss": zero,
        "complementarity_loss": zero,
        "total_loss": classification,
    }


def evaluate_model(
    model: torch.nn.Module,
    config: Mapping[str, Any],
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, Any]:
    model.eval()
    total_loss = 0.0
    total_count = 0
    y_true: List[int] = []
    y_pred: List[int] = []
    all_probabilities: List[List[float]] = []
    prediction_rows: List[Dict[str, Any]] = []
    label_list = get_label_list(config)
    with torch.no_grad():
        for raw_batch in loader:
            batch = move_batch(raw_batch, device)
            output = forward_batch(model, config, batch, return_aux=True)
            losses = compute_batch_loss(config, output, batch)
            valid = batch["attention_mask"].bool() & (batch["labels"] >= 0)
            valid_count = int(valid.sum().item())
            total_loss += float(losses["total_loss"].item()) * valid_count
            total_count += valid_count
            probabilities = torch.softmax(output["logits"], dim=-1)
            predicted = probabilities.argmax(dim=-1)
            y_true.extend(int(value) for value in batch["labels"][valid].cpu().tolist())
            y_pred.extend(int(value) for value in predicted[valid].cpu().tolist())
            all_probabilities.extend(
                probabilities[valid].detach().cpu().to(torch.float64).tolist()
            )
            for batch_index, time_index in valid.nonzero(as_tuple=False).cpu().tolist():
                utterance_ids = raw_batch.get("utterance_ids", [])
                utterance_id = (
                    utterance_ids[batch_index][time_index]
                    if batch_index < len(utterance_ids) and time_index < len(utterance_ids[batch_index])
                    else f"utt_{time_index}"
                )
                true_label_id = int(batch["labels"][batch_index, time_index].item())
                predicted_label_id = int(predicted[batch_index, time_index].item())
                prediction_rows.append(
                    build_prediction_row(
                        split=str(getattr(loader.dataset, "split", "unknown")),
                        dialogue_id=raw_batch["dialogue_ids"][batch_index],
                        utterance_id=utterance_id,
                        utterance_index=time_index,
                        true_label_id=true_label_id,
                        predicted_label_id=predicted_label_id,
                        probabilities=probabilities[batch_index, time_index]
                        .detach()
                        .cpu()
                        .tolist(),
                        label_list=label_list,
                        extra={
                            "time_index": int(time_index),
                            "label_id": true_label_id,
                            "predicted_id": predicted_label_id,
                        },
                    )
                )
    if total_count == 0:
        raise RuntimeError("Evaluation produced no valid utterances.")
    labels = list(range(int(config["dataset"]["num_classes"])))
    metrics = compute_classification_metrics(y_true, y_pred, labels=labels)
    metrics.update(compute_calibration_metrics(y_true, all_probabilities))
    return {
        "loss": total_loss / total_count,
        "metrics": metrics,
        "prediction_rows": prediction_rows,
        "y_true": y_true,
        "y_pred": y_pred,
    }


def save_evaluation_outputs(
    output_dir: Path,
    config: Mapping[str, Any],
    split: str,
    result: Mapping[str, Any],
    checkpoint_path: Path,
    checkpoint_epoch: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = result["metrics"]
    model_name = normalize_new_causal_model_name(config["model"]["name"])
    metrics_row = {
        "split": str(split),
        "model_name": model_name,
        "checkpoint": project_relative(checkpoint_path),
        "checkpoint_epoch": int(checkpoint_epoch),
        "loss": float(result["loss"]),
        "accuracy": float(metrics["acc"]),
        "acc": float(metrics["acc"]),
        "weighted_f1": float(metrics["weighted_f1"]),
        "macro_f1": float(metrics["macro_f1"]),
        "uar": float(metrics["uar"]),
        "ece_10": float(metrics["ece_10"]),
        "nll": float(metrics["nll"]),
        "brier_score": float(metrics["brier_score"]),
        "mean_confidence": float(metrics["mean_confidence"]),
        "mean_confidence_correct": float(metrics["mean_confidence_correct"]),
        "mean_confidence_incorrect": float(metrics["mean_confidence_incorrect"]),
    }
    pd.DataFrame([metrics_row]).to_csv(output_dir / "metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(result["prediction_rows"]).to_csv(
        output_dir / "predictions.csv", index=False, encoding="utf-8-sig"
    )
    label_list = get_label_list(config)
    label_ids = list(range(len(label_list)))
    matrix = compute_confusion_matrix(result["y_true"], result["y_pred"], labels=label_ids)
    pd.DataFrame(matrix, index=label_list, columns=label_list).to_csv(
        output_dir / "confusion_matrix.csv", encoding="utf-8-sig"
    )
    recalls = compute_per_class_recall(result["y_true"], result["y_pred"], label_ids)
    support = {label: result["y_true"].count(label) for label in label_ids}
    pd.DataFrame(
        [
            {
                "label_id": label,
                "label": label_list[label],
                "support": support[label],
                "recall": recalls[label],
            }
            for label in label_ids
        ]
    ).to_csv(output_dir / "per_class_recall.csv", index=False, encoding="utf-8-sig")


def build_optimizer(model: torch.nn.Module, config: Mapping[str, Any]) -> torch.optim.Optimizer:
    optimizer = config["optimizer"]
    kwargs = {
        "lr": float(optimizer["learning_rate"]),
        "weight_decay": float(optimizer.get("weight_decay", 0.0)),
    }
    if str(optimizer["name"]).lower() == "adam":
        return torch.optim.Adam(model.parameters(), **kwargs)
    return torch.optim.AdamW(model.parameters(), **kwargs)


def load_checkpoint(path: Path, device: torch.device) -> Dict[str, Any]:
    checkpoint = torch.load(Path(path), map_location=device)
    if not isinstance(checkpoint, dict):
        raise TypeError("Checkpoint must contain a mapping.")
    for key in ("model_state_dict", "config", "epoch", "best_val_weighted_f1"):
        if key not in checkpoint:
            raise KeyError(f"Checkpoint is missing required key: {key}")
    return checkpoint


def rebuild_model_from_checkpoint(
    checkpoint: Mapping[str, Any], device: torch.device
) -> torch.nn.Module:
    config = checkpoint["config"]
    validate_runtime_config(config)
    model = build_new_causal_baseline(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    return model


def dump_json(value: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(dict(value), file, ensure_ascii=False, indent=2)
        file.write("\n")


__all__ = [
    "PROJECT_ROOT",
    "assert_batch_lengths",
    "build_dataloader",
    "build_new_causal_baseline",
    "build_optimizer",
    "compute_batch_loss",
    "evaluate_model",
    "forward_batch",
    "get_device",
    "get_label_list",
    "load_checkpoint",
    "load_yaml_config",
    "move_batch",
    "normalized_training_config",
    "project_relative",
    "rebuild_model_from_checkpoint",
    "resolve_path",
    "save_evaluation_outputs",
    "save_yaml_config",
    "validate_runtime_config",
    "verify_feature_sha256",
]

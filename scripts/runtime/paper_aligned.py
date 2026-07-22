"""Shared runtime for the four isolated original MERC reproductions."""

from __future__ import annotations

import copy
import json
import math
import random
import sys
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import precision_recall_fscore_support
from torch.utils.data import DataLoader, Dataset, Subset


PROJECT_ROOT = next(
    candidate
    for candidate in Path(__file__).resolve().parents
    if (candidate / "AGENTS.md").is_file() and (candidate / "scripts").is_dir()
)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.iemocap.official_feature_dataset import (  # noqa: E402
    IEMOCAPOfficialFeatureDataset,
    build_iemocap_dataloader,
    iemocap_dialogue_collate_fn,
)
from models.registry.paper_aligned import (  # noqa: E402
    build_original_repro_model,
    get_model_constructor_args,
)
from models.multidag_cl.paper_aligned import (  # noqa: E402
    curriculum_baby_step_indices,
    dialogue_difficulty_from_sequences,
)
from utils.evaluation import build_prediction_row  # noqa: E402
from utils.iemocap_features import validate_iemocap_feature_config  # noqa: E402
from utils.metrics import (  # noqa: E402
    compute_classification_metrics,
    compute_confusion_matrix,
)


IGNORE_INDEX = -100
LEGACY_OFFICIAL_TRACK = "legacy_official_split_safe_selection"
LEGACY_FIVEFOLD_TRACK = "legacy_fivefold_fair_comparison"
CLEAN_FIVEFOLD_TRACK = "clean_roberta_fivefold_fair_comparison"
EXPERIMENT_TRACKS = {
    LEGACY_OFFICIAL_TRACK: {
        "split_strategy": "official_train_stratified",
        "protocol_comparability": "paper_adjacent_not_exact",
    },
    LEGACY_FIVEFOLD_TRACK: {
        "split_strategy": "outer_session_stratified",
        "protocol_comparability": "fair_comparison_not_paper_reproduction",
    },
    CLEAN_FIVEFOLD_TRACK: {
        "split_strategy": "outer_session_stratified",
        "protocol_comparability": "fair_comparison_not_paper_reproduction",
    },
}
NUMERIC_STATUS_FINITE = "FINITE"
NUMERIC_STATUS_FORWARD = "NONFINITE_FORWARD"
NUMERIC_STATUS_LOSS = "NONFINITE_LOSS"
NUMERIC_STATUS_GRADIENT = "NONFINITE_GRADIENT"
NUMERIC_STATUS_PARAMETER = "NONFINITE_PARAMETER"
NUMERIC_STATUS_CHECKPOINT = "NONFINITE_CHECKPOINT"


def _loss_value(value: Any) -> Any:
    if torch.is_tensor(value):
        if value.numel() == 1:
            return float(value.detach().cpu().item())
        return f"tensor(shape={list(value.shape)})"
    if isinstance(value, Mapping):
        return {str(key): _loss_value(item) for key, item in value.items()}
    return value


class NumericValidationError(FloatingPointError):
    """A fail-fast error carrying machine-readable numerical failure context."""

    def __init__(
        self,
        *,
        numeric_status: str,
        model_name: str,
        epoch: Optional[int],
        batch_index: Optional[int],
        stage: str,
        tensor_or_parameter: str,
        classification_loss: Any = None,
        auxiliary_losses: Any = None,
        total_loss: Any = None,
        learning_rate: Any = None,
        amp_enabled: bool = False,
    ) -> None:
        self.numeric_status = numeric_status
        self.model_name = str(model_name)
        self.epoch = epoch
        self.batch_index = batch_index
        self.stage = str(stage)
        self.tensor_or_parameter = str(tensor_or_parameter)
        self.classification_loss = _loss_value(classification_loss)
        self.auxiliary_losses = _loss_value(auxiliary_losses or {})
        self.total_loss = _loss_value(total_loss)
        self.learning_rate = _loss_value(learning_rate)
        self.amp_enabled = bool(amp_enabled)
        fields = {
            "model_name": self.model_name,
            "epoch": self.epoch,
            "batch_index": self.batch_index,
            "stage": self.stage,
            "tensor_or_parameter": self.tensor_or_parameter,
            "classification_loss": self.classification_loss,
            "auxiliary_losses": self.auxiliary_losses,
            "total_loss": self.total_loss,
            "learning_rate": self.learning_rate,
            "amp_enabled": self.amp_enabled,
        }
        super().__init__(
            f"{numeric_status}: "
            + ", ".join(f"{key}={value!r}" for key, value in fields.items())
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "numeric_status": self.numeric_status,
            "first_nonfinite_stage": self.stage,
            "nonfinite_epoch": self.epoch,
            "nonfinite_batch": self.batch_index,
            "tensor_or_parameter": self.tensor_or_parameter,
            "error": str(self),
        }


def iter_floating_tensors(value: Any, prefix: str = ""):
    """Yield all floating/complex tensors from nested checkpoint-like values."""

    if torch.is_tensor(value):
        if value.is_floating_point() or value.is_complex():
            yield prefix or "tensor", value
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            child = f"{prefix}.{key}" if prefix else str(key)
            yield from iter_floating_tensors(item, child)
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            child = f"{prefix}[{index}]" if prefix else f"[{index}]"
            yield from iter_floating_tensors(item, child)


def tensor_collection_numeric_summary(value: Any) -> dict[str, Any]:
    tensor_count = 0
    element_count = 0
    nonfinite_tensor_count = 0
    nonfinite_element_count = 0
    first_nonfinite_tensor = None
    for name, tensor in iter_floating_tensors(value):
        tensor_count += 1
        element_count += tensor.numel()
        finite = torch.isfinite(tensor)
        invalid = int((~finite).sum().item())
        if invalid:
            nonfinite_tensor_count += 1
            nonfinite_element_count += invalid
            if first_nonfinite_tensor is None:
                first_nonfinite_tensor = name
    return {
        "tensor_count": tensor_count,
        "element_count": element_count,
        "nonfinite_tensor_count": nonfinite_tensor_count,
        "nonfinite_element_count": nonfinite_element_count,
        "first_nonfinite_tensor": first_nonfinite_tensor,
        "is_finite": nonfinite_tensor_count == 0,
    }


def checkpoint_numeric_summary(checkpoint: Mapping[str, Any]) -> dict[str, Any]:
    overall = tensor_collection_numeric_summary(checkpoint)
    parameters = tensor_collection_numeric_summary(checkpoint.get("model_state_dict", {}))
    return {
        "checkpoint_numeric_validation": "passed" if overall["is_finite"] else "failed",
        "checkpoint_nonfinite_tensor_count": overall["nonfinite_tensor_count"],
        "checkpoint_nonfinite_element_count": overall["nonfinite_element_count"],
        "checkpoint_first_nonfinite_tensor": overall["first_nonfinite_tensor"],
        "checkpoint_parameters_finite": bool(parameters["is_finite"]),
        "checkpoint_parameter_nonfinite_tensor_count": parameters[
            "nonfinite_tensor_count"
        ],
        "checkpoint_parameter_nonfinite_element_count": parameters[
            "nonfinite_element_count"
        ],
    }


def _first_nonfinite_tensor(named_tensors) -> Optional[str]:
    for name, tensor in named_tensors:
        if tensor is not None and (tensor.is_floating_point() or tensor.is_complex()):
            if not bool(torch.isfinite(tensor).all()):
                return str(name)
    return None


def validate_model_output_finite(
    output: Mapping[str, Any],
    *,
    model_name: str,
    epoch: Optional[int],
    batch_index: Optional[int],
    learning_rate: Any,
    amp_enabled: bool,
) -> None:
    context = {
        "model_name": model_name,
        "epoch": epoch,
        "batch_index": batch_index,
        "classification_loss": output.get("classification_loss"),
        "auxiliary_losses": output.get("aux_losses", {}),
        "total_loss": output.get("loss"),
        "learning_rate": learning_rate,
        "amp_enabled": amp_enabled,
    }
    if not bool(torch.isfinite(output["logits"]).all()):
        raise NumericValidationError(
            numeric_status=NUMERIC_STATUS_FORWARD,
            stage="logits",
            tensor_or_parameter="logits",
            **context,
        )
    losses = [("classification_loss", output["classification_loss"])]
    losses.extend((f"auxiliary_losses.{name}", value) for name, value in output["aux_losses"].items())
    losses.append(("total_loss", output["loss"]))
    invalid = _first_nonfinite_tensor(losses)
    if invalid is not None:
        raise NumericValidationError(
            numeric_status=NUMERIC_STATUS_LOSS,
            stage=invalid,
            tensor_or_parameter=invalid,
            **context,
        )


def validate_named_tensors_finite(
    named_tensors,
    *,
    numeric_status: str,
    stage: str,
    model_name: str,
    epoch: Optional[int],
    batch_index: Optional[int],
    output: Mapping[str, Any],
    learning_rate: Any,
    amp_enabled: bool,
) -> None:
    invalid = _first_nonfinite_tensor(named_tensors)
    if invalid is None:
        return
    raise NumericValidationError(
        numeric_status=numeric_status,
        model_name=model_name,
        epoch=epoch,
        batch_index=batch_index,
        stage=stage,
        tensor_or_parameter=invalid,
        classification_loss=output.get("classification_loss"),
        auxiliary_losses=output.get("aux_losses", {}),
        total_loss=output.get("loss"),
        learning_rate=learning_rate,
        amp_enabled=amp_enabled,
    )


def all_finite_numbers(value: Any) -> bool:
    if torch.is_tensor(value):
        return bool(torch.isfinite(value).all()) if (value.is_floating_point() or value.is_complex()) else True
    if isinstance(value, Mapping):
        return all(all_finite_numbers(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(all_finite_numbers(item) for item in value)
    if isinstance(value, (float, np.floating)):
        return math.isfinite(float(value))
    return True


def resolve_path(path_text: str) -> Path:
    path = Path(str(path_text)).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def project_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def load_yaml_config(path: Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise TypeError(f"{path} must contain a YAML mapping")
    return config


def save_yaml_config(config: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(dict(config), file, sort_keys=False, allow_unicode=True)


def dump_json(value: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(dict(value), file, ensure_ascii=False, indent=2)
        file.write("\n")


def normalized_training_config(config: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(dict(config))
    training = dict(result.get("training", {}))
    optimizer = dict(result.get("optimizer", {}))
    scheduler = dict(result.get("scheduler", {}))
    system = dict(result.get("system", {}))
    output = dict(result.get("output", {}))
    training.setdefault("epochs", 30)
    training.setdefault("batch_size", 16)
    training.setdefault("seed", system.get("seed", 42))
    training.setdefault("select_best_by", "val_weighted_f1")
    training.setdefault("grad_clip", 1.0)
    training.setdefault("num_workers", 0)
    training.setdefault("early_stopping_patience", 0)
    training.setdefault("amp", False)
    optimizer.setdefault("name", "Adam")
    optimizer.setdefault("learning_rate", 5e-4)
    optimizer.setdefault("weight_decay", 0.0)
    scheduler.setdefault("name", "none")
    system.setdefault("seed", int(training["seed"]))
    system.setdefault("device", "cuda")
    if "root" not in output and "run_root" not in output:
        output["root"] = "outputs"
    output.setdefault("experiment_date", None)
    result["training"] = training
    result["optimizer"] = optimizer
    result["scheduler"] = scheduler
    result["system"] = system
    result["output"] = output
    result.setdefault("run_name", str(result.get("model", {}).get("name", "original_merc")))
    result.setdefault("protocol_version", "original_merc_three_track_v2")
    return result


def validate_runtime_config(config: Mapping[str, Any]) -> None:
    dataset = config.get("dataset", {})
    training = config.get("training", {})
    optimizer = config.get("optimizer", {})
    scheduler = config.get("scheduler", {})
    if str(dataset.get("name", "")).upper() not in {"IEMOCAP", "SYNTHETIC"}:
        raise ValueError("dataset.name must be IEMOCAP or SYNTHETIC")
    if str(dataset.get("name", "")).upper() == "IEMOCAP":
        experiment_track = str(dataset.get("experiment_track", ""))
        if experiment_track not in EXPERIMENT_TRACKS:
            raise ValueError(f"unknown dataset.experiment_track: {experiment_track!r}")
        track_policy = EXPERIMENT_TRACKS[experiment_track]
        if str(dataset.get("val_split_strategy")) != track_policy["split_strategy"]:
            raise ValueError("experiment track and split strategy do not match")
        if str(dataset.get("protocol_comparability")) != track_policy["protocol_comparability"]:
            raise ValueError("experiment track and protocol comparability do not match")
        if str(dataset.get("outer_test_session", "")) not in {
            "Ses01", "Ses02", "Ses03", "Ses04", "Ses05"
        }:
            raise ValueError("dataset.outer_test_session must be Ses01-Ses05")
        if (
            experiment_track == LEGACY_OFFICIAL_TRACK
            and str(dataset.get("outer_test_session")) != "Ses05"
        ):
            raise ValueError("legacy official-safe selection must keep Ses05 as test")
        required_feature_fields = {
            "feature_protocol",
            "feature_cleanliness",
            "usage",
            "feature_sha256",
        }
        missing_feature_fields = required_feature_fields - set(dataset)
        if missing_feature_fields:
            raise ValueError(
                f"IEMOCAP config missing feature audit fields: {sorted(missing_feature_fields)}"
            )
    if int(training.get("epochs", 0)) <= 0 or int(training.get("batch_size", 0)) <= 0:
        raise ValueError("training.epochs and training.batch_size must be positive")
    if str(training.get("select_best_by")) != "val_weighted_f1":
        raise ValueError("checkpoint selection must use val_weighted_f1")
    if int(training.get("early_stopping_patience", 0)) < 0:
        raise ValueError("early_stopping_patience must be non-negative")
    if str(optimizer.get("name", "")).lower() not in {"adam", "adamw"}:
        raise ValueError("optimizer.name must be Adam or AdamW")
    if float(optimizer.get("learning_rate", 0)) <= 0:
        raise ValueError("optimizer.learning_rate must be positive")
    if str(scheduler.get("name", "none")).lower() not in {"none", "plateau", "cosine"}:
        raise ValueError("scheduler.name must be none, plateau, or cosine")
    get_model_constructor_args({"model": config["model"]})


def verify_feature_sha256(config: Mapping[str, Any]) -> Optional[str]:
    return validate_iemocap_feature_config(config, PROJECT_ROOT)


def get_device(config: Mapping[str, Any], override: Optional[str] = None) -> torch.device:
    requested = str(override or config.get("system", {}).get("device", "cuda"))
    if requested.startswith("cuda") and not torch.cuda.is_available():
        print("[WARN] CUDA requested but unavailable; using CPU")
        requested = "cpu"
    return torch.device(requested)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_label_list(config: Mapping[str, Any]) -> list[str]:
    dataset = config["dataset"]
    count = int(dataset.get("num_classes", config["model"].get("num_classes", 6)))
    labels = dataset.get("label_list")
    if labels is None:
        return [str(index) for index in range(count)]
    if len(labels) != count:
        raise ValueError("dataset.label_list length must equal num_classes")
    return [str(value) for value in labels]


class SyntheticDialogueDataset(Dataset):
    """Deterministic dataset reserved for unit and CPU pipeline checks."""

    _OFFSET = {"train": 0, "val": 10_000, "test": 20_000}

    def __init__(self, config: Mapping[str, Any], split: str) -> None:
        self.config = config
        self.split = split
        synthetic = config.get("synthetic", {})
        self.count = int(synthetic.get("split_sizes", {}).get(split, 4))
        self.sequence_length = int(synthetic.get("sequence_length", 5))

    def __len__(self) -> int:
        return self.count

    def __getitem__(self, index: int) -> dict[str, Any]:
        model = self.config["model"]
        generator = torch.Generator().manual_seed(
            int(self.config["system"]["seed"]) + self._OFFSET[self.split] + index
        )
        length = self.sequence_length - index % 2
        dialogue_id = f"synthetic_{self.split}_{index:03d}"
        return {
            "dialogue_id": dialogue_id,
            "utterance_ids": [f"{dialogue_id}_{time:03d}" for time in range(length)],
            "sentences": ["synthetic"] * length,
            "text_features": torch.randn(length, int(model["text_feature_dim"]), generator=generator),
            "audio_features": torch.randn(length, int(model["audio_feature_dim"]), generator=generator),
            "visual_features": torch.randn(length, int(model["visual_feature_dim"]), generator=generator),
            "labels": torch.randint(0, int(model["num_classes"]), (length,), generator=generator),
            "speaker_ids_int": torch.arange(length) % int(model.get("num_speakers", 2)),
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
            valid_ratio=float(dataset.get("inner_val_ratio", 0.1)),
            val_split_strategy=str(dataset["val_split_strategy"]),
            outer_test_session=str(dataset["outer_test_session"]),
            inner_val_ratio=float(dataset.get("inner_val_ratio", 0.1)),
            seed=int(dataset.get("split_seed", config["system"]["seed"])),
            shuffle=shuffle,
            num_workers=workers,
            pin_memory=str(config["system"].get("device", "")).startswith("cuda"),
        )
    generator = torch.Generator().manual_seed(int(config["system"]["seed"]))
    return DataLoader(
        SyntheticDialogueDataset(config, split),
        batch_size=size,
        shuffle=shuffle,
        num_workers=workers,
        collate_fn=iemocap_dialogue_collate_fn,
        generator=generator,
    )


def curriculum_train_loader(
    loader: DataLoader,
    model: torch.nn.Module,
    epoch: int,
    seed: int,
) -> DataLoader:
    if not bool(getattr(model, "use_curriculum_learning", False)):
        return loader
    dataset_split = str(getattr(loader.dataset, "split", "train")).lower()
    if dataset_split != "train":
        raise ValueError("curriculum difficulty may only be computed from the train split")
    difficulties: list[float] = []
    for index in range(len(loader.dataset)):
        item = loader.dataset[index]
        difficulties.append(
            dialogue_difficulty_from_sequences(
                item["labels"].tolist(), item["speaker_ids_int"].tolist()
            )
        )
    visible = curriculum_baby_step_indices(
        difficulties,
        epoch,
        int(getattr(model, "curriculum_bucket_count")),
    )
    generator = torch.Generator().manual_seed(seed + epoch)
    return DataLoader(
        Subset(loader.dataset, visible),
        batch_size=loader.batch_size,
        shuffle=True,
        num_workers=loader.num_workers,
        collate_fn=iemocap_dialogue_collate_fn,
        generator=generator,
    )


def move_batch(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    moved = {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}
    if not torch.equal(
        moved["attention_mask"].long().sum(dim=1), moved["lengths"].long()
    ):
        raise ValueError("attention_mask sums must exactly match lengths")
    return moved


def forward_batch(model: torch.nn.Module, batch: Mapping[str, Any]) -> dict[str, Any]:
    output = model(
        text_features=batch["text_features"],
        audio_features=batch["audio_features"],
        visual_features=batch["visual_features"],
        attention_mask=batch["attention_mask"],
        lengths=batch["lengths"],
        speaker_ids_int=batch["speaker_ids_int"],
        labels=batch.get("labels"),
    )
    required = {"logits", "loss", "classification_loss", "aux_losses", "features", "diagnostics"}
    missing = required - set(output)
    if missing:
        raise KeyError(f"model output missing keys: {sorted(missing)}")
    return output


def evaluate_model(
    model: torch.nn.Module,
    config: Mapping[str, Any],
    loader: DataLoader,
    device: torch.device,
    max_batches: Optional[int] = None,
) -> dict[str, Any]:
    model.eval()
    total_loss = 0.0
    total_count = 0
    y_true: list[int] = []
    y_pred: list[int] = []
    prediction_rows: list[dict[str, Any]] = []
    label_list = get_label_list(config)
    with torch.no_grad():
        for batch_number, raw_batch in enumerate(loader, start=1):
            if max_batches is not None and batch_number > max_batches:
                break
            batch = move_batch(raw_batch, device)
            output = forward_batch(model, batch)
            validate_model_output_finite(
                output,
                model_name=str(config["model"]["name"]),
                epoch=None,
                batch_index=batch_number,
                learning_rate=None,
                amp_enabled=False,
            )
            valid = batch["attention_mask"].bool() & (batch["labels"] >= 0)
            count = int(valid.sum().item())
            total_loss += float(output["loss"].item()) * count
            total_count += count
            probabilities = torch.softmax(output["logits"], dim=-1)
            if not bool(torch.isfinite(probabilities).all()):
                raise NumericValidationError(
                    numeric_status=NUMERIC_STATUS_FORWARD,
                    model_name=str(config["model"]["name"]),
                    epoch=None,
                    batch_index=batch_number,
                    stage="probabilities",
                    tensor_or_parameter="probabilities",
                    classification_loss=output["classification_loss"],
                    auxiliary_losses=output["aux_losses"],
                    total_loss=output["loss"],
                    learning_rate=None,
                    amp_enabled=False,
                )
            predicted = probabilities.argmax(dim=-1)
            y_true.extend(int(value) for value in batch["labels"][valid].cpu().tolist())
            y_pred.extend(int(value) for value in predicted[valid].cpu().tolist())
            for batch_index, time_index in valid.nonzero(as_tuple=False).cpu().tolist():
                true_id = int(batch["labels"][batch_index, time_index].item())
                predicted_id = int(predicted[batch_index, time_index].item())
                prediction_rows.append(
                    build_prediction_row(
                        split=str(getattr(loader.dataset, "split", "unknown")),
                        dialogue_id=raw_batch["dialogue_ids"][batch_index],
                        utterance_id=raw_batch["utterance_ids"][batch_index][time_index],
                        utterance_index=time_index,
                        true_label_id=true_id,
                        predicted_label_id=predicted_id,
                        probabilities=probabilities[batch_index, time_index].cpu().tolist(),
                        label_list=label_list,
                    )
                )
    if total_count == 0:
        raise RuntimeError("evaluation produced no valid utterances")
    label_ids = list(range(len(label_list)))
    result = {
        "loss": total_loss / total_count,
        "metrics": compute_classification_metrics(y_true, y_pred, labels=label_ids),
        "prediction_rows": prediction_rows,
        "y_true": y_true,
        "y_pred": y_pred,
    }
    if not all_finite_numbers({"loss": result["loss"], "metrics": result["metrics"]}):
        raise NumericValidationError(
            numeric_status=NUMERIC_STATUS_LOSS,
            model_name=str(config["model"]["name"]),
            epoch=None,
            batch_index=None,
            stage="evaluation_metrics",
            tensor_or_parameter="evaluation_metrics",
            classification_loss=None,
            auxiliary_losses={},
            total_loss=result["loss"],
            learning_rate=None,
            amp_enabled=False,
        )
    return result


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
    pd.DataFrame(
        [{
            "split": split,
            "model_name": config["model"]["name"],
            "feature_set_name": config["dataset"].get("feature_set_name"),
            "feature_protocol": config["dataset"].get("feature_protocol"),
            "feature_cleanliness": config["dataset"].get("feature_cleanliness"),
            "usage": config["dataset"].get("usage"),
            "outer_test_session": config["dataset"].get("outer_test_session"),
            "checkpoint": project_relative(checkpoint_path),
            "checkpoint_epoch": checkpoint_epoch,
            "loss": float(result["loss"]),
            "accuracy": float(metrics["acc"]),
            "weighted_f1": float(metrics["weighted_f1"]),
            "macro_f1": float(metrics["macro_f1"]),
            "uar": float(metrics["uar"]),
        }]
    ).to_csv(output_dir / "metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(result["prediction_rows"]).to_csv(
        output_dir / "predictions.csv", index=False, encoding="utf-8-sig"
    )
    label_list = get_label_list(config)
    label_ids = list(range(len(label_list)))
    matrix = compute_confusion_matrix(result["y_true"], result["y_pred"], labels=label_ids)
    pd.DataFrame(matrix, index=label_list, columns=label_list).to_csv(
        output_dir / "confusion_matrix.csv", encoding="utf-8-sig"
    )
    precision, recall, f1, support = precision_recall_fscore_support(
        result["y_true"], result["y_pred"], labels=label_ids, zero_division=0
    )
    pd.DataFrame(
        [{
            "label_id": label_id,
            "label": label_list[label_id],
            "precision": float(precision[label_id]),
            "recall": float(recall[label_id]),
            "f1": float(f1[label_id]),
            "support": int(support[label_id]),
        } for label_id in label_ids]
    ).to_csv(output_dir / "per_class_metrics.csv", index=False, encoding="utf-8-sig")


def build_optimizer(model: torch.nn.Module, config: Mapping[str, Any]) -> torch.optim.Optimizer:
    options = config["optimizer"]
    kwargs = {
        "lr": float(options["learning_rate"]),
        "weight_decay": float(options.get("weight_decay", 0.0)),
    }
    if str(options["name"]).lower() == "adamw":
        return torch.optim.AdamW(model.parameters(), **kwargs)
    return torch.optim.Adam(model.parameters(), **kwargs)


def build_scheduler(optimizer: torch.optim.Optimizer, config: Mapping[str, Any]):
    scheduler = config["scheduler"]
    name = str(scheduler.get("name", "none")).lower()
    if name == "none":
        return None
    if name == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=float(scheduler.get("factor", 0.5)),
            patience=int(scheduler.get("patience", 3)),
        )
    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=int(config["training"]["epochs"])
    )


def load_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    checkpoint = torch.load(Path(path), map_location=device)
    if not isinstance(checkpoint, dict):
        raise TypeError("checkpoint must contain a mapping")
    required = {
        "model_state_dict", "optimizer_state_dict", "config", "epoch",
        "best_val_weighted_f1", "feature_sha256", "test_split_used_for_selection",
    }
    missing = required - set(checkpoint)
    if missing:
        raise KeyError(f"checkpoint missing keys: {sorted(missing)}")
    numeric = checkpoint_numeric_summary(checkpoint)
    if numeric["checkpoint_numeric_validation"] != "passed":
        raise NumericValidationError(
            numeric_status=NUMERIC_STATUS_CHECKPOINT,
            model_name=str(checkpoint.get("model_key", "unknown")),
            epoch=checkpoint.get("epoch"),
            batch_index=None,
            stage="checkpoint_reload",
            tensor_or_parameter=str(numeric["checkpoint_first_nonfinite_tensor"]),
            classification_loss=None,
            auxiliary_losses={},
            total_loss=None,
            learning_rate=None,
            amp_enabled=False,
        )
    checkpoint["checkpoint_numeric_validation"] = numeric
    return checkpoint


def rebuild_model_from_checkpoint(
    checkpoint: Mapping[str, Any], device: torch.device
) -> torch.nn.Module:
    config = checkpoint["config"]
    validate_runtime_config(config)
    model = build_original_repro_model(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    return model


__all__ = [
    "CLEAN_FIVEFOLD_TRACK",
    "EXPERIMENT_TRACKS",
    "LEGACY_FIVEFOLD_TRACK",
    "LEGACY_OFFICIAL_TRACK",
    "NUMERIC_STATUS_CHECKPOINT",
    "NUMERIC_STATUS_FINITE",
    "NUMERIC_STATUS_FORWARD",
    "NUMERIC_STATUS_GRADIENT",
    "NUMERIC_STATUS_LOSS",
    "NUMERIC_STATUS_PARAMETER",
    "NumericValidationError",
    "PROJECT_ROOT",
    "all_finite_numbers",
    "build_dataloader",
    "build_optimizer",
    "build_scheduler",
    "checkpoint_numeric_summary",
    "curriculum_train_loader",
    "dump_json",
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
    "seed_everything",
    "tensor_collection_numeric_summary",
    "validate_model_output_finite",
    "validate_named_tensors_finite",
    "validate_runtime_config",
    "verify_feature_sha256",
]

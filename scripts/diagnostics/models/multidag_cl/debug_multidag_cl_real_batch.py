"""One-batch real-data dry run for the project MultiDAG+CL baseline.

This script intentionally stays separate from the MMGCN train/evaluate path.
It builds one existing real-data dataloader, consumes the first batch only,
runs MultiDAG+CL forward/loss/backward, checks tensor contracts, and exits
without saving checkpoints or formal experiment outputs.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any, Dict, Iterable, Optional

import torch
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.collators.m3ed_collate import m3ed_dialogue_collate_fn  # noqa: E402
from datasets.iemocap import build_iemocap_dataloader  # noqa: E402
from datasets.m3ed.torch_dataset import M3EDTorchDataset  # noqa: E402
from models.multidag_cl.unified import MultiDAGCLBaseline  # noqa: E402
from utils.io import load_yaml, resolve_project_path  # noqa: E402
from utils.seed import set_seed  # noqa: E402


CORE_FIELDS = (
    "text_features",
    "audio_features",
    "visual_features",
    "labels",
    "attention_mask",
    "speaker_ids_int",
    "lengths",
)


class DryRunUnavailable(RuntimeError):
    """Raised when the requested real-data batch cannot be built locally."""


class DryRunCheckError(RuntimeError):
    """Raised when the first real-data batch violates the model contract."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a one-batch MultiDAG+CL dry run on an existing real dataset."
    )
    parser.add_argument(
        "--config",
        default="configs/mmgcn/unified/m3ed/full_context/m3ed_features/skeleton.yaml",
        help="Path to an existing M3ED or IEMOCAP YAML config.",
    )
    parser.add_argument(
        "--dataset",
        default="M3ED",
        choices=("M3ED", "IEMOCAP"),
        help="Dataset adapter to use. Must match dataset.name in the config.",
    )
    parser.add_argument(
        "--split",
        default="train",
        choices=("train", "val"),
        help="Real-data split to dry-run. Test split is intentionally disabled.",
    )
    parser.add_argument(
        "--batch-size-override",
        type=int,
        default=2,
        help="Batch size for this one-batch dry run.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device for the dry run. Use 'config' to honor system.device.",
    )
    return parser.parse_args()


def _resolve_required_data_path(
    dataset_name: str,
    key: str,
    raw_path: Optional[str],
) -> Path:
    if not raw_path:
        raise DryRunUnavailable(
            f"{dataset_name} real-data dry run was not completed because "
            f"config key is missing: dataset.{key}"
        )

    resolved = resolve_project_path(str(raw_path))
    if resolved is None or not resolved.exists():
        raise DryRunUnavailable(
            f"{dataset_name} real-data dry run was not completed because "
            f"data file is missing: {raw_path}"
        )

    return resolved


def _validate_dataset_choice(config: Dict[str, Any], dataset_name: str) -> None:
    config_dataset = str(config.get("dataset", {}).get("name", "")).upper()
    if config_dataset != dataset_name:
        raise DryRunCheckError(
            f"Config dataset.name={config_dataset!r} does not match "
            f"--dataset {dataset_name!r}."
        )


def _select_device(config: Dict[str, Any], requested: str) -> torch.device:
    if requested == "config":
        requested = str(config.get("system", {}).get("device", "cpu"))

    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA requested but unavailable; using CPU for this dry run.")
        device = torch.device("cpu")

    return device


def _batch_size(args: argparse.Namespace) -> int:
    if int(args.batch_size_override) <= 0:
        raise DryRunCheckError("--batch-size-override must be positive.")
    return int(args.batch_size_override)


def _num_workers(config: Dict[str, Any]) -> int:
    return int(config.get("train", {}).get("num_workers", 0))


def _build_m3ed_loader(
    config: Dict[str, Any],
    split: str,
    batch_size: int,
    device: torch.device,
) -> DataLoader:
    dataset_config = config["dataset"]
    feature_pkl_path = _resolve_required_data_path(
        "M3ED",
        "feature_pkl_path",
        dataset_config.get("feature_pkl_path"),
    )
    metadata_path = _resolve_required_data_path(
        "M3ED",
        "metadata_path",
        dataset_config.get("metadata_path"),
    )
    label_mapping_path = _resolve_required_data_path(
        "M3ED",
        "label_mapping_path",
        dataset_config.get("label_mapping_path"),
    )

    dataset = M3EDTorchDataset(
        feature_pkl_path=str(feature_pkl_path),
        metadata_path=str(metadata_path),
        label_mapping_path=str(label_mapping_path),
        split=split,
        check_label_consistency=True,
        raise_on_label_mismatch=False,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=_num_workers(config),
        collate_fn=m3ed_dialogue_collate_fn,
        pin_memory=device.type == "cuda",
    )


def _build_iemocap_loader(
    config: Dict[str, Any],
    split: str,
    batch_size: int,
    device: torch.device,
) -> DataLoader:
    dataset_config = config["dataset"]
    feature_pkl_path = _resolve_required_data_path(
        "IEMOCAP",
        "feature_pkl_path",
        dataset_config.get("feature_pkl_path"),
    )

    return build_iemocap_dataloader(
        feature_pkl_path=feature_pkl_path,
        split=split,
        batch_size=batch_size,
        valid_ratio=float(dataset_config.get("valid_ratio", 0.1)),
        val_split_strategy=str(
            dataset_config.get("val_split_strategy", "official_prefix")
        ),
        val_session_id=dataset_config.get("val_session_id"),
        seed=int(config.get("system", {}).get("seed", 42)),
        shuffle=False,
        num_workers=_num_workers(config),
        pin_memory=device.type == "cuda",
    )


def build_dataloader(
    config: Dict[str, Any],
    dataset_name: str,
    split: str,
    batch_size: int,
    device: torch.device,
) -> DataLoader:
    if dataset_name == "M3ED":
        return _build_m3ed_loader(config, split, batch_size, device)
    if dataset_name == "IEMOCAP":
        return _build_iemocap_loader(config, split, batch_size, device)
    raise DryRunCheckError(f"Unsupported dataset for real dry run: {dataset_name}")


def first_batch(loader: DataLoader, dataset_name: str, split: str) -> Dict[str, Any]:
    try:
        return next(iter(loader))
    except StopIteration as exc:
        raise DryRunUnavailable(
            f"{dataset_name} real-data dry run was not completed because "
            f"split={split} produced no batches."
        ) from exc


def move_tensor_batch(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def _shape(tensor: torch.Tensor) -> str:
    return "[" + ", ".join(str(dim) for dim in tensor.shape) + "]"


def print_batch_summary(batch: Dict[str, Any]) -> None:
    print("Batch keys:", ", ".join(sorted(batch.keys())))
    print("Core tensor shapes:")
    for field in CORE_FIELDS:
        if field not in batch:
            print(f"  {field}: MISSING")
            continue
        value = batch[field]
        if torch.is_tensor(value):
            print(
                f"  {field}: shape={_shape(value)} "
                f"dtype={value.dtype} device={value.device}"
            )
        else:
            print(f"  {field}: type={type(value).__name__}")


def _active_modalities(config: Dict[str, Any]) -> tuple[str, ...]:
    model_config = config.get("model", {})
    modality_config = config.get("modality", {})
    active = model_config.get(
        "active_modalities",
        modality_config.get("active_modalities", ["text", "audio", "visual"]),
    )
    return tuple(str(name) for name in active)


def _window_past(config: Dict[str, Any], seq_len: int) -> int:
    model_config = config.get("model", {})
    graph_config = config.get("graph", {})
    value = model_config.get("window_past", graph_config.get("window_past", None))
    if value is None:
        return max(int(seq_len) - 1, 0)
    return int(value)


def _num_graph_layers(config: Dict[str, Any]) -> int:
    model_config = config.get("model", {})
    graph_config = config.get("graph", {})
    return int(model_config.get("num_graph_layers", graph_config.get("num_layers", 2)))


def _num_speakers(config: Dict[str, Any], batch: Dict[str, Any]) -> int:
    configured = config.get("model", {}).get("num_speakers")
    if configured is not None:
        return int(configured)

    speaker_ids = batch["speaker_ids_int"]
    valid = batch["attention_mask"].to(dtype=torch.bool)
    if not valid.any():
        return 2
    max_id = int(speaker_ids[valid].clamp_min(0).max().item())
    return max(max_id + 1, 2)


def build_model(config: Dict[str, Any], batch: Dict[str, Any]) -> MultiDAGCLBaseline:
    text_features = batch["text_features"]
    audio_features = batch["audio_features"]
    visual_features = batch["visual_features"]
    _, seq_len, text_dim = text_features.shape

    model_config = config["model"]
    dataset_config = config["dataset"]
    return MultiDAGCLBaseline(
        text_dim=int(text_dim),
        audio_dim=int(audio_features.shape[-1]),
        visual_dim=int(visual_features.shape[-1]),
        hidden_dim=int(model_config["hidden_dim"]),
        num_classes=int(dataset_config["num_classes"]),
        num_speakers=_num_speakers(config, batch),
        window_past=_window_past(config, int(seq_len)),
        dropout=float(model_config.get("dropout", 0.1)),
        active_modalities=_active_modalities(config),
        num_graph_layers=_num_graph_layers(config),
        modality_encoder_type=str(model_config.get("modality_encoder_type", "causal_gru")),
        modality_encoder_layers=int(model_config.get("modality_encoder_layers", 1)),
    )


def _required_tensors(batch: Dict[str, Any], fields: Iterable[str]) -> None:
    for field in fields:
        if field not in batch:
            raise DryRunCheckError(f"Batch is missing required field: {field}")
        if not torch.is_tensor(batch[field]):
            raise DryRunCheckError(f"Batch field is not a tensor: {field}")


def valid_count(batch: Dict[str, Any]) -> int:
    valid = batch["attention_mask"].to(dtype=torch.bool) & (batch["labels"] >= 0)
    return int(valid.sum().item())


def has_future_leakage(adjacency: torch.Tensor) -> bool:
    if adjacency.dim() != 3:
        raise DryRunCheckError(
            f"adjacency must be [B, T, T], got {_shape(adjacency)}"
        )
    seq_len = int(adjacency.shape[1])
    future_mask = torch.triu(
        torch.ones(seq_len, seq_len, dtype=torch.bool, device=adjacency.device),
        diagonal=1,
    )
    leakage = adjacency.to(dtype=torch.bool) & future_mask.unsqueeze(0)
    return bool(leakage.any().item())


def backward_ok(model: MultiDAGCLBaseline) -> bool:
    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    return bool(gradients) and all(
        bool(torch.isfinite(gradient).all().item()) for gradient in gradients
    )


def run_forward_checks(
    model: MultiDAGCLBaseline,
    batch: Dict[str, Any],
) -> Dict[str, Any]:
    _required_tensors(batch, CORE_FIELDS)

    batch_size, seq_len, _ = batch["text_features"].shape
    num_classes = int(model.num_classes)
    expected_logits_shape = (int(batch_size), int(seq_len), num_classes)
    expected_adjacency_shape = (int(batch_size), int(seq_len), int(seq_len))

    model.train()
    output = model(
        text_features=batch["text_features"],
        audio_features=batch["audio_features"],
        visual_features=batch["visual_features"],
        attention_mask=batch["attention_mask"],
        speaker_ids_int=batch["speaker_ids_int"],
        labels=batch["labels"],
    )

    logits = output["logits"]
    if tuple(logits.shape) != expected_logits_shape:
        raise DryRunCheckError(
            "logits shape mismatch: "
            f"expected {expected_logits_shape}, got {tuple(logits.shape)}"
        )

    loss = output["loss"]
    if loss is None:
        raise DryRunCheckError("loss should not be None when labels are provided.")
    loss_finite = bool(torch.isfinite(loss).item())
    if not loss_finite:
        raise DryRunCheckError(f"loss is not finite: {float(loss.item())}")

    adjacency = output["adjacency"]
    if tuple(adjacency.shape) != expected_adjacency_shape:
        raise DryRunCheckError(
            "adjacency shape mismatch: "
            f"expected {expected_adjacency_shape}, got {tuple(adjacency.shape)}"
        )

    count = valid_count(batch)
    if count <= 0:
        raise DryRunCheckError("valid_count must be > 0.")

    future_leakage = has_future_leakage(adjacency)
    if future_leakage:
        raise DryRunCheckError("adjacency contains future leakage.")

    model.zero_grad(set_to_none=True)
    loss.backward()
    gradients_finite = backward_ok(model)
    if not gradients_finite:
        raise DryRunCheckError("backward produced missing or non-finite gradients.")

    return {
        "logits_shape": tuple(logits.shape),
        "loss_value": float(loss.item()),
        "loss_finite": loss_finite,
        "backward_ok": gradients_finite,
        "adjacency_shape": tuple(adjacency.shape),
        "future_leakage": future_leakage,
        "valid_count": count,
    }


def _print_result(result: Dict[str, Any]) -> None:
    print("Dry-run checks:")
    print(f"  logits shape: {result['logits_shape']}")
    print(f"  loss: {result['loss_value']:.6f}")
    print(f"  loss finite: {result['loss_finite']}")
    print(f"  backward ok: {result['backward_ok']}")
    print(f"  adjacency shape: {result['adjacency_shape']}")
    print(f"  future leakage: {result['future_leakage']}")
    print(f"  valid_count: {result['valid_count']}")


def main() -> int:
    args = parse_args()
    dataset_name = str(args.dataset).upper()

    try:
        config = load_yaml(args.config)
        _validate_dataset_choice(config, dataset_name)

        seed = int(config.get("system", {}).get("seed", config.get("seed", 42)))
        set_seed(seed)
        device = _select_device(config, str(args.device))
        batch_size = _batch_size(args)

        loader = build_dataloader(
            config=config,
            dataset_name=dataset_name,
            split=str(args.split),
            batch_size=batch_size,
            device=device,
        )
        batch = move_tensor_batch(first_batch(loader, dataset_name, args.split), device)

        print("MultiDAG+CL real-data one-batch dry run")
        print(f"Config: {args.config}")
        print(f"Dataset: {dataset_name}")
        print(f"Split: {args.split}")
        print(f"Batch size override: {batch_size}")
        print(f"Device: {device}")
        print("This is a one-batch dry run only. It is not a formal training experiment.")
        print_batch_summary(batch)

        model = build_model(config, batch).to(device)
        print("Model config:")
        print(f"  hidden_dim: {model.hidden_dim}")
        print(f"  num_classes: {model.num_classes}")
        print(f"  num_speakers: {model.num_speakers}")
        print(f"  window_past: {model.window_past}")
        print(f"  active_modalities: {model.active_modalities}")
        print(f"  num_graph_layers: {model.num_graph_layers}")
        print(f"  modality_encoder_type: {model.modality_encoder_type}")
        print(f"  modality_encoder_layers: {model.modality_encoder_layers}")

        result = run_forward_checks(model, batch)
        _print_result(result)
        print(f"{dataset_name} real-data one-batch dry run passed.")
        return 0

    except DryRunUnavailable as exc:
        print(str(exc))
        return 2
    except DryRunCheckError as exc:
        print(f"{dataset_name} real-data dry run failed contract check: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

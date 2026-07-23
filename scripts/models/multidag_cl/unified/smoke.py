"""Local smoke training entry for the project MultiDAG+CL baseline.

This script intentionally stays separate from the MMGCN train/evaluate path.
It uses the deterministic fake dialogue smoke dataset, selects the best
checkpoint only on validation Weighted-F1, and reloads the saved checkpoint for
one validation evaluation pass.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys
from typing import Any, Dict, Iterable, List, Tuple

import torch


PROJECT_ROOT = next(
    candidate
    for candidate in Path(__file__).resolve().parents
    if (candidate / "AGENTS.md").is_file() and (candidate / "scripts").is_dir()
)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.smoke.mmgcn_smoke_dataset import build_mmgcn_smoke_dataloader  # noqa: E402
from models.multidag_cl.unified import MultiDAGCLBaseline  # noqa: E402
from utils.io import ensure_dir, load_yaml, resolve_project_path, sanitize_name, save_yaml  # noqa: E402
from utils.seed import set_seed  # noqa: E402
from utils.output_paths import (  # noqa: E402
    configured_output_root,
    create_unique_run_dir,
    resolve_experiment_date,
    resolve_output_category,
)


METRIC_KEYS = ("acc", "weighted_f1", "macro_f1", "uar")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a tiny MultiDAG+CL local smoke training loop."
    )
    parser.add_argument(
        "--config",
        default=(
            "configs/multidag_cl/unified/synthetic/causal_context/"
            "synthetic/smoke.yaml"
        ),
        help="Path to the smoke YAML config.",
    )
    parser.add_argument("--experiment-date", default=None)
    return parser.parse_args()


def _project_relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT.resolve()))
    except ValueError:
        return str(path)


def _make_run_dir(config: Dict[str, Any], experiment_date: str | None) -> Path:
    output_config = config.setdefault("output", {})
    output_root = resolve_project_path(str(configured_output_root(config)))
    if output_root is None:
        raise ValueError("output root must be set")
    frozen_date = resolve_experiment_date(
        cli_date=experiment_date,
        config=config,
    )
    run_name = output_config.get("run_name", "multidag_cl_smoke")
    run_dir = create_unique_run_dir(
        sanitize_name(str(run_name)), frozen_date, output_root
    )
    manifest_dir = resolve_output_category(
        "manifests", frozen_date, output_root
    ) / run_dir.name
    ensure_dir(manifest_dir)
    output_config.pop("run_root", None)
    output_config.update(
        {
            "root": str(output_root),
            "experiment_date": frozen_date,
            "output_root": str(output_root),
            "day_output_root": str(output_root / frozen_date),
            "run_dir": str(run_dir),
            "log_dir": str(run_dir),
            "analysis_dir": str(
                resolve_output_category("analysis", frozen_date, output_root)
            ),
            "manifest_dir": str(manifest_dir),
        }
    )

    latest_path = manifest_dir / "latest_run.txt"
    with latest_path.open("w", encoding="utf-8") as file:
        file.write(f"run_dir={_project_relative(run_dir)}\n")

    return run_dir


def _validate_config(config: Dict[str, Any]) -> None:
    dataset_name = config.get("dataset", {}).get("name")
    if dataset_name != "MMGCN_SMOKE":
        raise ValueError(f"Only MMGCN_SMOKE is supported, got {dataset_name!r}")

    model_name = config.get("model", {}).get("name")
    if model_name != "MultiDAGCL":
        raise ValueError(f"Only MultiDAGCL is supported, got {model_name!r}")

    select_best_by = config.get("training", {}).get("select_best_by")
    if select_best_by != "val_weighted_f1":
        raise ValueError(
            "This smoke entry only supports validation-best selection by "
            "training.select_best_by=val_weighted_f1"
        )


def _build_dataloaders(config: Dict[str, Any]) -> Tuple[Any, Any]:
    dataset_config = dict(config["dataset"])
    model_config = dict(config["model"])
    training_config = {
        "batch_size": int(dataset_config.get("batch_size", 2)),
        "num_workers": int(dataset_config.get("num_workers", 0)),
    }
    dataset_config.setdefault("num_classes", int(model_config["num_classes"]))

    train_loader = build_mmgcn_smoke_dataloader(
        dataset_config=dataset_config,
        model_config=model_config,
        train_config=training_config,
        split="train",
        shuffle=True,
    )
    val_loader = build_mmgcn_smoke_dataloader(
        dataset_config=dataset_config,
        model_config=model_config,
        train_config=training_config,
        split="val",
        shuffle=False,
    )
    return train_loader, val_loader


def _build_model(config: Dict[str, Any]) -> MultiDAGCLBaseline:
    model_config = config["model"]
    return MultiDAGCLBaseline(
        text_dim=int(model_config["text_feature_dim"]),
        audio_dim=int(model_config["audio_feature_dim"]),
        visual_dim=int(model_config["visual_feature_dim"]),
        hidden_dim=int(model_config["hidden_dim"]),
        num_classes=int(model_config["num_classes"]),
        num_speakers=int(model_config.get("num_speakers", 2)),
        window_past=int(model_config.get("window_past", 2)),
        dropout=float(model_config.get("dropout", 0.1)),
        active_modalities=tuple(model_config.get("active_modalities", ["text", "audio", "visual"])),
        num_graph_layers=int(model_config.get("num_graph_layers", 1)),
        modality_encoder_type=str(model_config.get("modality_encoder_type", "causal_gru")),
        modality_encoder_layers=int(model_config.get("modality_encoder_layers", 1)),
    )


def _move_tensor_batch(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def _forward_batch(
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


def _valid_count(batch: Dict[str, Any]) -> int:
    valid = batch["attention_mask"].to(dtype=torch.bool) & (batch["labels"] >= 0)
    return int(valid.sum().item())


def _collect_predictions(
    logits: torch.Tensor,
    labels: torch.Tensor,
    attention_mask: torch.Tensor,
) -> Tuple[List[int], List[int]]:
    valid = attention_mask.to(dtype=torch.bool) & (labels >= 0)
    if not valid.any():
        return [], []
    predictions = logits.argmax(dim=-1)
    y_true = labels[valid].detach().cpu().tolist()
    y_pred = predictions[valid].detach().cpu().tolist()
    return [int(x) for x in y_true], [int(x) for x in y_pred]


def _compute_metrics(
    y_true: Iterable[int],
    y_pred: Iterable[int],
    num_classes: int,
) -> Dict[str, float]:
    true_values = [int(x) for x in y_true]
    pred_values = [int(x) for x in y_pred]
    total = len(true_values)
    if total == 0:
        return {key: 0.0 for key in METRIC_KEYS}

    correct = sum(1 for target, pred in zip(true_values, pred_values) if target == pred)
    f1_values = []
    recall_values = []
    weighted_f1 = 0.0

    for label_id in range(int(num_classes)):
        tp = sum(
            1
            for target, pred in zip(true_values, pred_values)
            if target == label_id and pred == label_id
        )
        fp = sum(
            1
            for target, pred in zip(true_values, pred_values)
            if target != label_id and pred == label_id
        )
        fn = sum(
            1
            for target, pred in zip(true_values, pred_values)
            if target == label_id and pred != label_id
        )
        support = tp + fn
        precision = float(tp) / float(tp + fp) if (tp + fp) > 0 else 0.0
        recall = float(tp) / float(support) if support > 0 else 0.0
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if (precision + recall) > 0.0
            else 0.0
        )
        f1_values.append(f1)
        recall_values.append(recall)
        weighted_f1 += f1 * float(support)

    return {
        "acc": float(correct) / float(total),
        "weighted_f1": weighted_f1 / float(total),
        "macro_f1": sum(f1_values) / float(num_classes),
        "uar": sum(recall_values) / float(num_classes),
    }


def train_one_epoch(
    model: MultiDAGCLBaseline,
    loader: Any,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip: float,
) -> float:
    model.train()
    total_loss = 0.0
    total_items = 0

    for batch in loader:
        batch = _move_tensor_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        output = _forward_batch(model, batch)
        loss = output["loss"]
        if loss is None:
            raise RuntimeError("Model returned no loss for a labeled smoke batch")
        loss.backward()
        if grad_clip > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        valid_items = max(_valid_count(batch), 1)
        total_loss += float(loss.item()) * float(valid_items)
        total_items += valid_items

    return total_loss / float(max(total_items, 1))


@torch.no_grad()
def evaluate(
    model: MultiDAGCLBaseline,
    loader: Any,
    device: torch.device,
    num_classes: int,
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_items = 0
    all_true: List[int] = []
    all_pred: List[int] = []

    for batch in loader:
        batch = _move_tensor_batch(batch, device)
        output = _forward_batch(model, batch)
        loss = output["loss"]
        if loss is None:
            raise RuntimeError("Model returned no loss for a labeled smoke batch")

        valid_items = max(_valid_count(batch), 1)
        total_loss += float(loss.item()) * float(valid_items)
        total_items += valid_items

        y_true, y_pred = _collect_predictions(
            logits=output["logits"],
            labels=batch["labels"],
            attention_mask=batch["attention_mask"],
        )
        all_true.extend(y_true)
        all_pred.extend(y_pred)

    metrics = _compute_metrics(all_true, all_pred, int(num_classes))
    metrics["loss"] = total_loss / float(max(total_items, 1))
    return metrics


def _write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _save_checkpoint(
    path: Path,
    model: MultiDAGCLBaseline,
    optimizer: torch.optim.Optimizer,
    config: Dict[str, Any],
    epoch: int,
    metrics: Dict[str, float],
) -> None:
    torch.save(
        {
            "model_name": "MultiDAGCL",
            "epoch": int(epoch),
            "config": config,
            "metrics": metrics,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "checkpoint_selection": "validation_weighted_f1",
        },
        path,
    )


def _load_checkpoint(path: Path, device: torch.device) -> Dict[str, Any]:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def _metric_summary(metrics: Dict[str, float], prefix: str = "") -> str:
    parts = []
    for key in METRIC_KEYS:
        metric_key = f"{prefix}{key}" if prefix else key
        if metric_key in metrics:
            parts.append(f"{metric_key}={metrics[metric_key]:.4f}")
    return " | ".join(parts)


def main() -> None:
    args = _parse_args()
    config = load_yaml(args.config)
    _validate_config(config)

    seed = int(config.get("seed", 42))
    set_seed(seed)

    device_name = config.get("system", {}).get("device", "cpu")
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Config requested CUDA, but CUDA is not available")

    run_dir = _make_run_dir(config, args.experiment_date)
    save_yaml(config, run_dir / "config.yaml")

    train_loader, val_loader = _build_dataloaders(config)
    model = _build_model(config).to(device)

    training_config = config["training"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_config.get("lr", 1e-3)),
        weight_decay=float(training_config.get("weight_decay", 0.0)),
    )

    epochs = int(training_config.get("epochs", 2))
    grad_clip = float(training_config.get("grad_clip", 0.0))
    num_classes = int(config["model"]["num_classes"])
    best_metric_name = "val_weighted_f1"
    best_metric = float("-inf")
    best_epoch = -1
    epoch_rows: List[Dict[str, Any]] = []

    print("MultiDAG+CL smoke training")
    print(f"Config: {args.config}")
    print(f"Output: {_project_relative(run_dir)}")
    print("Checkpoint selection: validation Weighted-F1 only; test split is not used.")

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            grad_clip=grad_clip,
        )
        val_metrics = evaluate(
            model=model,
            loader=val_loader,
            device=device,
            num_classes=num_classes,
        )
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "val_acc": val_metrics["acc"],
            "val_weighted_f1": val_metrics["weighted_f1"],
            "val_macro_f1": val_metrics["macro_f1"],
            "val_uar": val_metrics["uar"],
        }
        epoch_rows.append(row)

        if row[best_metric_name] > best_metric:
            best_metric = float(row[best_metric_name])
            best_epoch = epoch
            _save_checkpoint(
                path=run_dir / "best_model.pt",
                model=model,
                optimizer=optimizer,
                config=config,
                epoch=epoch,
                metrics=row,
            )

        _save_checkpoint(
            path=run_dir / "last_model.pt",
            model=model,
            optimizer=optimizer,
            config=config,
            epoch=epoch,
            metrics=row,
        )

        print(
            f"Epoch {epoch:02d}: train_loss={train_loss:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} | "
            f"{_metric_summary(row, prefix='val_')}"
        )

    epoch_metrics_path = run_dir / "epoch_metrics.csv"
    _write_csv(
        path=epoch_metrics_path,
        rows=epoch_rows,
        fieldnames=[
            "epoch",
            "train_loss",
            "val_loss",
            "val_acc",
            "val_weighted_f1",
            "val_macro_f1",
            "val_uar",
        ],
    )

    best_path = run_dir / "best_model.pt"
    if not best_path.exists():
        raise RuntimeError("Best checkpoint was not saved")

    checkpoint = _load_checkpoint(best_path, device)
    reloaded_model = _build_model(config).to(device)
    reloaded_model.load_state_dict(checkpoint["model_state_dict"])
    reload_metrics = evaluate(
        model=reloaded_model,
        loader=val_loader,
        device=device,
        num_classes=num_classes,
    )
    reload_row = {
        "split": "val",
        "checkpoint": "best_model.pt",
        "epoch": int(checkpoint["epoch"]),
        "loss": reload_metrics["loss"],
        "acc": reload_metrics["acc"],
        "weighted_f1": reload_metrics["weighted_f1"],
        "macro_f1": reload_metrics["macro_f1"],
        "uar": reload_metrics["uar"],
    }
    _write_csv(
        path=run_dir / "smoke_eval_reload_metrics.csv",
        rows=[reload_row],
        fieldnames=[
            "split",
            "checkpoint",
            "epoch",
            "loss",
            "acc",
            "weighted_f1",
            "macro_f1",
            "uar",
        ],
    )

    print(f"Best epoch: {best_epoch} ({best_metric_name}={best_metric:.4f})")
    print(f"Reload val: loss={reload_metrics['loss']:.4f} | {_metric_summary(reload_metrics)}")
    print(f"Saved best checkpoint: {_project_relative(best_path)}")
    print(f"Saved reload metrics: {_project_relative(run_dir / 'smoke_eval_reload_metrics.csv')}")


if __name__ == "__main__":
    main()

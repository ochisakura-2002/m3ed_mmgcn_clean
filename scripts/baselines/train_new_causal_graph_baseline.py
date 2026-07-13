"""Train Project causal GS-MCC-inspired or Project causal DialogueGCN."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import pandas as pd
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.baselines.causal_baseline_registry import (  # noqa: E402
    build_new_causal_baseline,
    get_new_causal_model_family,
    normalize_new_causal_model_name,
)
from scripts.baselines.new_causal_graph_runtime import (  # noqa: E402
    build_dataloader,
    build_optimizer,
    compute_batch_loss,
    evaluate_model,
    forward_batch,
    get_device,
    load_checkpoint,
    load_yaml_config,
    move_batch,
    normalized_training_config,
    project_relative,
    rebuild_model_from_checkpoint,
    resolve_path,
    save_evaluation_outputs,
    save_yaml_config,
    validate_runtime_config,
    verify_feature_sha256,
)
from utils.run_metadata import build_run_metadata, write_run_metadata  # noqa: E402
from utils.seed import set_seed  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Training YAML path.")
    parser.add_argument(
        "--output-root",
        default=None,
        help="Optional run-root override, primarily for isolated tests.",
    )
    return parser.parse_args()


def sanitize_name(value: str) -> str:
    cleaned = "".join(character if character.isalnum() or character in "-_" else "_" for character in str(value))
    return cleaned.strip("_") or "new_causal_graph_baseline"


def prepare_run_environment(
    config: Dict[str, Any], config_path: Path
) -> Dict[str, Any]:
    run_root = resolve_path(str(config["output"]["run_root"]))
    run_name = sanitize_name(str(config.get("run_name", config["model"]["name"])))
    run_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{run_name}"
    run_dir = run_root / run_id
    checkpoints_dir = run_dir / "checkpoints"
    logs_dir = run_dir / "logs"
    checkpoints_dir.mkdir(parents=True, exist_ok=False)
    logs_dir.mkdir(parents=True, exist_ok=True)
    save_yaml_config(config, logs_dir / "experiment_config.yaml")
    write_run_metadata(config, run_dir / "run_metadata.json", PROJECT_ROOT)

    latest_dir = run_root.parent if run_root.name == "runs" else run_root
    latest_path = latest_dir / "latest_run.txt"
    latest_path.parent.mkdir(parents=True, exist_ok=True)
    with latest_path.open("w", encoding="utf-8") as file:
        file.write(f"run_id={run_id}\n")
        file.write(f"run_dir={project_relative(run_dir)}\n")
        file.write(f"model_name={normalize_new_causal_model_name(config['model']['name'])}\n")
        file.write(f"config_path={project_relative(config_path)}\n")
    return {
        "run_id": run_id,
        "run_dir": run_dir,
        "checkpoints_dir": checkpoints_dir,
        "logs_dir": logs_dir,
        "latest_path": latest_path,
    }


def strictly_better_val_weighted_f1(current: float, best: float) -> bool:
    """Use strict comparison so exact ties retain the earlier epoch."""

    return float(current) > float(best)


def train_one_epoch(
    model: torch.nn.Module,
    config: Mapping[str, Any],
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Dict[str, float]:
    model.train()
    totals = {
        "total_loss": 0.0,
        "classification_loss": 0.0,
        "consistency_loss": 0.0,
        "complementarity_loss": 0.0,
    }
    total_count = 0
    grad_clip = float(config["training"].get("grad_clip", 0.0))
    for raw_batch in loader:
        batch = move_batch(raw_batch, device)
        optimizer.zero_grad(set_to_none=True)
        output = forward_batch(model, config, batch, return_aux=True)
        losses = compute_batch_loss(config, output, batch)
        losses["total_loss"].backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        valid_count = int((batch["attention_mask"].bool() & (batch["labels"] >= 0)).sum().item())
        total_count += valid_count
        for key in totals:
            totals[key] += float(losses[key].detach().item()) * valid_count
    if total_count == 0:
        raise RuntimeError("Training loader produced no valid utterances.")
    return {key: value / total_count for key, value in totals.items()}


def checkpoint_payload(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    config: Mapping[str, Any],
    epoch: int,
    best_val_weighted_f1: float,
) -> Dict[str, Any]:
    metadata = build_run_metadata(config, PROJECT_ROOT)
    dataset = config.get("dataset", {})
    name = normalize_new_causal_model_name(config["model"]["name"])
    return {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": int(epoch),
        "best_val_weighted_f1": float(best_val_weighted_f1),
        "config": dict(config),
        "model_family": metadata["model_family"],
        "model_variant": metadata["model_variant"],
        "model_name": name,
        "feature_sha256": dataset.get("feature_sha256"),
        "val_split_strategy": dataset.get("val_split_strategy"),
        "val_session_id": dataset.get("val_session_id"),
        "seed": int(config.get("system", {}).get("seed", 42)),
        "checkpoint_selection": "validation_val_weighted_f1",
        "test_split_used_for_selection": False,
    }


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    config: Mapping[str, Any],
    epoch: int,
    best_val_weighted_f1: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        checkpoint_payload(model, optimizer, config, epoch, best_val_weighted_f1),
        path,
    )


def run_training(
    config_path: Path, output_root_override: Optional[Path] = None
) -> Dict[str, Any]:
    resolved_config_path = resolve_path(str(config_path))
    config = normalized_training_config(load_yaml_config(resolved_config_path))
    if output_root_override is not None:
        config["output"]["run_root"] = str(Path(output_root_override))
    validate_runtime_config(config)
    verified_sha256 = verify_feature_sha256(config)

    seed = int(config.get("system", {}).get("seed", config["training"]["seed"]))
    set_seed(seed)
    device = get_device(config)
    paths = prepare_run_environment(config, resolved_config_path)
    train_loader = build_dataloader(config, "train", shuffle=True)
    val_loader = build_dataloader(config, "val", shuffle=False)
    model = build_new_causal_baseline(config).to(device)
    optimizer = build_optimizer(model, config)

    best_val_weighted_f1 = float("-inf")
    best_epoch = -1
    last_epoch = 0
    epoch_rows = []
    for epoch in range(1, int(config["training"]["epochs"]) + 1):
        train_losses = train_one_epoch(model, config, train_loader, optimizer, device)
        val_result = evaluate_model(model, config, val_loader, device)
        val_metrics = val_result["metrics"]
        current = float(val_metrics["weighted_f1"])
        row = {
            "epoch": int(epoch),
            "train_loss": float(train_losses["total_loss"]),
            "val_loss": float(val_result["loss"]),
            "val_accuracy": float(val_metrics["acc"]),
            "val_weighted_f1": current,
            "val_macro_f1": float(val_metrics["macro_f1"]),
            "val_uar": float(val_metrics["uar"]),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
        }
        if get_new_causal_model_family(config) == "gsmcc":
            row.update(
                {
                    "train_classification_loss": float(train_losses["classification_loss"]),
                    "train_consistency_loss": float(train_losses["consistency_loss"]),
                    "train_complementarity_loss": float(train_losses["complementarity_loss"]),
                }
            )
        epoch_rows.append(row)
        pd.DataFrame(epoch_rows).to_csv(
            paths["logs_dir"] / "epoch_metrics.csv", index=False, encoding="utf-8-sig"
        )
        if strictly_better_val_weighted_f1(current, best_val_weighted_f1):
            best_val_weighted_f1 = current
            best_epoch = epoch
            save_checkpoint(
                paths["checkpoints_dir"] / "best_model.pt",
                model,
                optimizer,
                config,
                epoch,
                best_val_weighted_f1,
            )
        last_epoch = epoch
        print(
            f"epoch={epoch} train_loss={train_losses['total_loss']:.6f} "
            f"val_loss={val_result['loss']:.6f} val_weighted_f1={current:.6f}"
        )

    if best_epoch < 0:
        raise RuntimeError("No best checkpoint was selected.")
    save_checkpoint(
        paths["checkpoints_dir"] / "last_model.pt",
        model,
        optimizer,
        config,
        last_epoch,
        best_val_weighted_f1,
    )

    best_path = paths["checkpoints_dir"] / "best_model.pt"
    best_checkpoint = load_checkpoint(best_path, device)
    reloaded_model = rebuild_model_from_checkpoint(best_checkpoint, device)
    reloaded_val = evaluate_model(reloaded_model, config, val_loader, device)
    test_loader = build_dataloader(config, "test", shuffle=False)
    reloaded_test = evaluate_model(reloaded_model, config, test_loader, device)
    evaluations = paths["logs_dir"] / "evaluations"
    save_evaluation_outputs(
        evaluations / "val_best_model",
        config,
        "val",
        reloaded_val,
        best_path,
        int(best_checkpoint["epoch"]),
    )
    save_evaluation_outputs(
        evaluations / "test_best_model",
        config,
        "test",
        reloaded_test,
        best_path,
        int(best_checkpoint["epoch"]),
    )
    print(f"run_id={paths['run_id']}")
    print(f"run_dir={paths['run_dir']}")
    print(f"best_epoch={best_epoch}")
    print("checkpoint_reload=passed")
    return {
        "run_id": str(paths["run_id"]),
        "run_dir": paths["run_dir"],
        "best_epoch": int(best_epoch),
        "best_val_weighted_f1": float(best_val_weighted_f1),
        "verified_feature_sha256": verified_sha256,
        "val_result": reloaded_val,
        "test_result": reloaded_test,
    }


def main() -> None:
    args = parse_args()
    run_training(
        Path(args.config),
        None if args.output_root is None else Path(args.output_root),
    )


if __name__ == "__main__":
    main()

"""Train one original MERC baseline with validation-only checkpoint selection."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Optional

import pandas as pd
import torch


PROJECT_ROOT = next(
    candidate
    for candidate in Path(__file__).resolve().parents
    if (candidate / "AGENTS.md").is_file() and (candidate / "scripts").is_dir()
)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.registry.paper_aligned import (  # noqa: E402
    build_original_repro_model,
    get_source_metadata,
)
from scripts.runtime.paper_aligned import (  # noqa: E402
    NUMERIC_STATUS_FINITE,
    NUMERIC_STATUS_GRADIENT,
    NUMERIC_STATUS_PARAMETER,
    NumericValidationError,
    all_finite_numbers,
    build_dataloader,
    build_optimizer,
    build_scheduler,
    checkpoint_numeric_summary,
    curriculum_train_loader,
    dump_json,
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
    seed_everything,
    validate_runtime_config,
    validate_model_output_finite,
    validate_named_tensors_finite,
    verify_feature_sha256,
)
from utils.output_paths import (  # noqa: E402
    allocate_configured_run,
    configured_output_root,
    configured_run_id,
    infer_experiment_date_from_run,
    infer_experiment_group_from_run,
    resolve_experiment_date,
    resolve_experiment_group,
    resolve_output_paths,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--experiment-date", default=None)
    parser.add_argument("--experiment-group", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--resume", default=None, help="Resume from a last_model.pt checkpoint")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def sanitize_name(value: str) -> str:
    clean = "".join(char if char.isalnum() or char in "-_" else "_" for char in str(value))
    return clean.strip("_") or "original_merc"


def current_git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unavailable"


def prepare_run_environment(
    config: dict[str, Any],
    config_path: Path,
    experiment_date: str,
    output_root: Path,
    experiment_group: Optional[str] = None,
    resume_run_dir: Optional[Path] = None,
) -> dict[str, Any]:
    run_name = sanitize_name(str(config.get("run_name", config["model"]["name"])))
    layout = allocate_configured_run(
        config=config,
        config_path=config_path,
        experiment_name=run_name,
        experiment_date=experiment_date,
        output_base=output_root,
        experiment_group=experiment_group,
        resume_run_dir=resume_run_dir,
    )
    assert layout.run_root is not None
    run_dir = layout.run_root
    run_id = run_dir.name
    checkpoints = run_dir / "checkpoints"
    logs = run_dir / "logs"
    metrics = run_dir / "metrics"
    artifacts = run_dir / "artifacts"
    manifest_dir = layout.manifest_root / "runs" / run_id
    for path in (checkpoints, logs, metrics, artifacts, manifest_dir):
        path.mkdir(parents=True, exist_ok=True)
    config.setdefault("output", {})
    config["output"].update(
        {
            "root": str(run_dir),
            "output_base": str(output_root),
            "experiment_date": experiment_date,
            "experiment_group": layout.experiment_group,
            "output_root": str(output_root),
            "day_output_root": str(output_root / experiment_date),
            "experiment_root": str(layout.experiment_root),
            "run_id": run_id,
            "run_root": str(run_dir),
            "run_dir": str(run_dir),
            "log_dir": str(logs),
            "metrics_dir": str(metrics),
            "artifacts_dir": str(artifacts),
            "analysis_dir": str(layout.analysis_root),
            "manifest_dir": str(manifest_dir),
            "review_dir": str(layout.review_root),
            "report_dir": str(layout.report_root),
        }
    )
    save_yaml_config(config, logs / "experiment_config.yaml")
    save_yaml_config(config, run_dir / "resolved_config.yaml")
    dump_json(
        {
            "run_id": run_id,
            "run_dir": str(run_dir),
            "experiment_date": experiment_date,
            "experiment_group": layout.experiment_group,
            "experiment_root": str(layout.experiment_root),
            "output_root": str(output_root),
            "day_output_root": str(output_root / experiment_date),
            "log_dir": str(logs),
            "metrics_dir": str(metrics),
            "artifacts_dir": str(artifacts),
            "analysis_dir": str(layout.analysis_root),
            "manifest_dir": str(manifest_dir),
            "review_dir": str(layout.review_root),
            "report_dir": str(layout.report_root),
            "config_path": project_relative(config_path),
            "model_source": get_source_metadata(config["model"]["name"]),
            "causal_grade": "noncausal_offline_full_context",
            "protocol_version": config["protocol_version"],
            "feature_path": config["dataset"].get("feature_pkl_path"),
            "feature_sha256": config["dataset"].get("feature_sha256"),
            "feature_protocol": config["dataset"].get("feature_protocol"),
            "experiment_track": config["dataset"].get("experiment_track"),
            "protocol_comparability": config["dataset"].get("protocol_comparability"),
            "outer_test_session": config["dataset"].get("outer_test_session"),
            "split_seed": config["dataset"].get("split_seed"),
            "seed": config["system"].get("seed"),
            "git_commit": current_git_commit(),
            "test_split_used_for_selection": False,
        },
        run_dir / "run_metadata.json",
    )
    latest_run = manifest_dir / "latest_run.txt"
    latest_run.write_text(
        f"run_id={run_id}\nrun_dir={run_dir.resolve()}\n",
        encoding="utf-8",
    )
    return {
        "run_id": run_id,
        "run_dir": run_dir,
        "checkpoints": checkpoints,
        "logs": logs,
        "manifest_dir": manifest_dir,
        "experiment_date": experiment_date,
        "experiment_group": layout.experiment_group,
        "experiment_root": layout.experiment_root,
        "day_output_root": output_root / experiment_date,
        "metrics": metrics,
        "artifacts": artifacts,
    }


def train_one_epoch(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    device: torch.device,
    grad_clip: float,
    amp_enabled: bool,
    max_batches: Optional[int],
    model_name: str = "unknown",
    epoch: int = 0,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    total_classification = 0.0
    total_count = 0
    for batch_number, raw_batch in enumerate(loader, start=1):
        if max_batches is not None and batch_number > max_batches:
            break
        batch = move_batch(raw_batch, device)
        optimizer.zero_grad(set_to_none=True)
        learning_rate = optimizer.param_groups[0]["lr"]
        try:
            with torch.autocast(device_type=device.type, enabled=amp_enabled):
                output = forward_batch(model, batch)
                loss = output["loss"]
        except NumericValidationError:
            raise
        except FloatingPointError as error:
            raise NumericValidationError(
                numeric_status="NONFINITE_FORWARD",
                model_name=model_name,
                epoch=epoch,
                batch_index=batch_number,
                stage="model_forward",
                tensor_or_parameter="model_internal",
                classification_loss=None,
                auxiliary_losses={},
                total_loss=None,
                learning_rate=learning_rate,
                amp_enabled=amp_enabled,
            ) from error
        validate_model_output_finite(
            output,
            model_name=model_name,
            epoch=epoch,
            batch_index=batch_number,
            learning_rate=learning_rate,
            amp_enabled=amp_enabled,
        )
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        validate_named_tensors_finite(
            ((name, parameter.grad) for name, parameter in model.named_parameters()),
            numeric_status=NUMERIC_STATUS_GRADIENT,
            stage="gradients_after_backward",
            model_name=model_name,
            epoch=epoch,
            batch_index=batch_number,
            output=output,
            learning_rate=learning_rate,
            amp_enabled=amp_enabled,
        )
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        validate_named_tensors_finite(
            ((name, parameter.grad) for name, parameter in model.named_parameters()),
            numeric_status=NUMERIC_STATUS_GRADIENT,
            stage="gradients_after_clipping",
            model_name=model_name,
            epoch=epoch,
            batch_index=batch_number,
            output=output,
            learning_rate=learning_rate,
            amp_enabled=amp_enabled,
        )
        scaler.step(optimizer)
        scaler.update()
        validate_named_tensors_finite(
            model.named_parameters(),
            numeric_status=NUMERIC_STATUS_PARAMETER,
            stage="parameters_after_optimizer_step",
            model_name=model_name,
            epoch=epoch,
            batch_index=batch_number,
            output=output,
            learning_rate=learning_rate,
            amp_enabled=amp_enabled,
        )
        count = int((batch["attention_mask"].bool() & (batch["labels"] >= 0)).sum().item())
        total_count += count
        total_loss += float(loss.detach().item()) * count
        total_classification += float(output["classification_loss"].detach().item()) * count
    if total_count == 0:
        raise RuntimeError("training produced no valid utterances")
    return {
        "loss": total_loss / total_count,
        "classification_loss": total_classification / total_count,
    }


def checkpoint_payload(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: Any,
    config: Mapping[str, Any],
    epoch: int,
    best_epoch: int,
    best_val_weighted_f1: float,
    split_ids: Mapping[str, list[str]],
) -> dict[str, Any]:
    dataset = config["dataset"]
    source = get_source_metadata(config["model"]["name"])
    constructor_arguments = {
        key: value
        for key, value in config["model"].items()
        if key not in {"name", "causal_grade", "fidelity_status"}
    }
    return {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": None if scheduler is None else scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "config": dict(config),
        "model_key": config["model"]["name"],
        "model_constructor_arguments": constructor_arguments,
        "epoch": int(epoch),
        "best_epoch": int(best_epoch),
        "best_val_weighted_f1": float(best_val_weighted_f1),
        "best_validation_weighted_f1": float(best_val_weighted_f1),
        "feature_sha256": dataset.get("feature_sha256"),
        "feature_path": dataset.get("feature_pkl_path"),
        "feature_set_name": dataset.get("feature_set_name"),
        "feature_protocol": dataset.get("feature_protocol"),
        "feature_cleanliness": dataset.get("feature_cleanliness"),
        "feature_usage": dataset.get("usage"),
        "experiment_track": dataset.get("experiment_track"),
        "protocol_comparability": dataset.get("protocol_comparability"),
        "outer_test_session": dataset.get("outer_test_session"),
        "inner_val_ratio": dataset.get("inner_val_ratio"),
        "split_seed": dataset.get("split_seed"),
        "split_ids": {key: list(value) for key, value in split_ids.items()},
        "split_protocol": {
            "name": dataset.get("experiment_track"),
            "protocol_comparability": dataset.get("protocol_comparability"),
            "validation_source": "train_pool_dialogues_only",
            "outer_test_session": dataset.get("outer_test_session"),
            "inner_val_ratio": dataset.get("inner_val_ratio"),
            "split_seed": dataset.get("split_seed"),
        },
        "seed": int(config["system"]["seed"]),
        "git_commit": current_git_commit(),
        "checkpoint_selection": "validation_val_weighted_f1",
        "test_split_used_for_selection": False,
        "causal_grade": "noncausal_offline_full_context",
        "source_paper": source["paper"],
        "source_repository": source["repository"],
        "source_commit": source["commit"],
        "model_source": source,
        "paper_reproduction_eligible": source.get("paper_reproduction_eligible", "true")
        != "false",
    }


def save_checkpoint(path: Path, **payload_args: Any) -> dict[str, Any]:
    payload = checkpoint_payload(**payload_args)
    numeric = checkpoint_numeric_summary(payload)
    if numeric["checkpoint_numeric_validation"] != "passed":
        config = payload_args["config"]
        optimizer = payload_args["optimizer"]
        error = NumericValidationError(
            numeric_status="NONFINITE_CHECKPOINT",
            model_name=str(config["model"]["name"]),
            epoch=int(payload_args["epoch"]),
            batch_index=None,
            stage="checkpoint_save",
            tensor_or_parameter=str(numeric["checkpoint_first_nonfinite_tensor"]),
            classification_loss=None,
            auxiliary_losses={},
            total_loss=None,
            learning_rate=optimizer.param_groups[0]["lr"],
            amp_enabled=bool(payload_args["scaler"].is_enabled()),
        )
        error.checkpoint_numeric = numeric
        raise error
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    return numeric


def _record_numeric_failure(
    paths: Mapping[str, Any],
    error: NumericValidationError,
    run_start: float,
) -> None:
    checkpoint_numeric = getattr(error, "checkpoint_numeric", {})
    summary = {
        "run_id": paths["run_id"],
        "run_dir": str(paths["run_dir"]),
        "run_status": "NUMERICALLY_INVALID",
        **error.as_dict(),
        "checkpoint_reload": "failed",
        "checkpoint_numeric_validation": checkpoint_numeric.get(
            "checkpoint_numeric_validation", "not_passed"
        ),
        "checkpoint_nonfinite_tensor_count": checkpoint_numeric.get(
            "checkpoint_nonfinite_tensor_count"
        ),
        "checkpoint_nonfinite_element_count": checkpoint_numeric.get(
            "checkpoint_nonfinite_element_count"
        ),
        "checkpoint_parameters_finite": bool(
            checkpoint_numeric.get("checkpoint_parameters_finite", False)
        ),
        "final_metrics_finite": False,
        "prediction_count_correct": False,
        "training_time_seconds": time.perf_counter() - run_start,
    }
    dump_json(summary, paths["logs"] / "run_summary.json")


def _split_ids(loader: torch.utils.data.DataLoader) -> dict[str, list[str]]:
    dataset = loader.dataset
    if hasattr(dataset, "get_split_ids"):
        return dataset.get_split_ids()
    return {"train": [], "val": [], "test": []}


def dry_run(config: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    loaders = {
        split: build_dataloader(config, split, shuffle=False, batch_size=2)
        for split in ("train", "val", "test")
    }
    model = build_original_repro_model(config).to(device)
    raw_batch = next(iter(loaders["train"]))
    output = forward_batch(model, move_batch(raw_batch, device))
    output["loss"].backward()
    result = {
        "model": config["model"]["name"],
        "device": str(device),
        "logits_shape": list(output["logits"].shape),
        "loss": float(output["loss"].detach().item()),
        "split_sizes": {split: len(loader.dataset) for split, loader in loaders.items()},
        "feature_validated_before_model_init": True,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def _config_without_runtime_output(config: Mapping[str, Any]) -> dict[str, Any]:
    """Exclude path/date-only fields from resume compatibility checks."""

    import copy

    comparable = copy.deepcopy(dict(config))
    comparable.pop("output", None)
    return comparable


def run_training(
    config_path: Path,
    output_root_override: Optional[Path] = None,
    device_override: Optional[str] = None,
    resume_path: Optional[Path] = None,
    dry_run_only: bool = False,
    experiment_date: Optional[str] = None,
    experiment_group: Optional[str] = None,
) -> dict[str, Any]:
    resolved_config = resolve_path(str(config_path))
    config = normalized_training_config(load_yaml_config(resolved_config))
    frozen_date = resolve_experiment_date(
        cli_date=experiment_date,
        config=config,
    )
    output_root = configured_output_root(config, override=output_root_override)
    output_root = resolve_path(str(output_root))
    frozen_group = resolve_experiment_group(
        cli_group=experiment_group,
        config=config,
        config_path=resolved_config,
    )
    config["output"]["output_base"] = str(output_root)
    config["output"]["experiment_date"] = frozen_date
    config["output"]["experiment_group"] = frozen_group
    fixed_run_id = configured_run_id(config)
    if fixed_run_id is None:
        config["output"].pop("run_root", None)
        config["output"]["root"] = str(output_root)
    else:
        fixed_layout = resolve_output_paths(
            output_base=output_root,
            experiment_date=frozen_date,
            experiment_group=frozen_group,
            run_id=fixed_run_id,
        )
        assert fixed_layout.run_root is not None
        config["output"]["root"] = str(fixed_layout.run_root)
    validate_runtime_config(config)
    verified_sha = verify_feature_sha256(config)  # Must precede model construction.
    seed = int(config["system"]["seed"])
    seed_everything(seed)
    device = get_device(config, device_override)
    if dry_run_only:
        result = dry_run(config, device)
        result["verified_feature_sha256"] = verified_sha
        result["experiment_date"] = frozen_date
        result["experiment_group"] = frozen_group
        return result

    resume_checkpoint = None
    resume_run_dir = None
    if resume_path is not None:
        resolved_resume = resolve_path(str(resume_path))
        resume_checkpoint = load_checkpoint(resolved_resume, device)
        if _config_without_runtime_output(resume_checkpoint["config"]) != (
            _config_without_runtime_output(config)
        ):
            raise ValueError("resume checkpoint config does not exactly match requested config")
        resume_run_dir = resolved_resume.parent.parent
        inferred_date = infer_experiment_date_from_run(resume_run_dir)
        inferred_group = infer_experiment_group_from_run(resume_run_dir)
        checkpoint_date = resume_checkpoint.get("config", {}).get("output", {}).get(
            "experiment_date"
        )
        if inferred_date is not None:
            frozen_date = inferred_date
        elif checkpoint_date is not None:
            frozen_date = resolve_experiment_date(cli_date=str(checkpoint_date))
        if inferred_group is not None:
            frozen_group = inferred_group
        if resume_run_dir.parent.name == "runs":
            runs_parent = resume_run_dir.parent.parent
            if (
                runs_parent.name == frozen_group
                and runs_parent.parent.name == frozen_date
            ):
                output_root = runs_parent.parent.parent
            elif runs_parent.name == frozen_date:
                # Read-only/resume compatibility for outputs/<date>/runs/<id>.
                output_root = runs_parent.parent

    paths = prepare_run_environment(
        config,
        resolved_config,
        frozen_date,
        output_root,
        experiment_group=frozen_group,
        resume_run_dir=resume_run_dir,
    )
    run_start = time.perf_counter()

    def guarded_numeric(operation):
        try:
            return operation()
        except NumericValidationError as error:
            _record_numeric_failure(paths, error, run_start)
            raise

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    print(
        "CODEX_RUN_INFO_JSON="
        + json.dumps({"run_id": paths["run_id"], "run_dir": str(paths["run_dir"].resolve())}),
        flush=True,
    )
    train_loader = build_dataloader(config, "train", shuffle=True)
    val_loader = build_dataloader(config, "val", shuffle=False)
    split_ids = _split_ids(train_loader)
    dump_json(
        {
            "outer_test_session": config["dataset"].get("outer_test_session"),
            "inner_val_ratio": config["dataset"].get("inner_val_ratio"),
            "split_seed": config["dataset"].get("split_seed"),
            "experiment_track": config["dataset"].get("experiment_track"),
            "protocol_comparability": config["dataset"].get("protocol_comparability"),
            "test_split_used_for_selection": False,
            "dialogue_ids": split_ids,
        },
        paths["logs"] / "split_manifest.json",
    )
    model = build_original_repro_model(config).to(device)
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config)
    amp_enabled = bool(config["training"].get("amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    start_epoch = 1
    best_score = float("-inf")
    best_epoch = -1
    if resume_checkpoint is not None:
        checkpoint = resume_checkpoint
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        scaler.load_state_dict(checkpoint.get("scaler_state_dict", {}))
        start_epoch = int(checkpoint["epoch"]) + 1
        best_score = float(checkpoint["best_val_weighted_f1"])
        best_epoch = int(checkpoint.get("best_epoch", checkpoint["epoch"]))

    history_path = paths["logs"] / "epoch_metrics.csv"
    if resume_checkpoint is not None and history_path.is_file():
        history = pd.read_csv(history_path).to_dict(orient="records")
    else:
        history = []
    patience = int(config["training"].get("early_stopping_patience", 0))
    epochs_without_improvement = 0
    last_epoch = start_epoch - 1
    max_train_batches = config["training"].get("max_train_batches")
    max_eval_batches = config["training"].get("max_eval_batches")
    for epoch in range(start_epoch, int(config["training"]["epochs"]) + 1):
        epoch_loader = curriculum_train_loader(train_loader, model, epoch, seed)
        train_result = guarded_numeric(
            lambda: train_one_epoch(
                model,
                epoch_loader,
                optimizer,
                scaler,
                device,
                float(config["training"].get("grad_clip", 0.0)),
                amp_enabled,
                None if max_train_batches is None else int(max_train_batches),
                model_name=str(config["model"]["name"]),
                epoch=epoch,
            )
        )
        val_result = guarded_numeric(
            lambda: evaluate_model(
                model,
                config,
                val_loader,
                device,
                None if max_eval_batches is None else int(max_eval_batches),
            )
        )
        score = float(val_result["metrics"]["weighted_f1"])
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(score)
            else:
                scheduler.step()
        row = {
            "epoch": epoch,
            "train_loss": train_result["loss"],
            "train_classification_loss": train_result["classification_loss"],
            "val_loss": val_result["loss"],
            "val_accuracy": val_result["metrics"]["acc"],
            "val_weighted_f1": score,
            "val_macro_f1": val_result["metrics"]["macro_f1"],
            "val_uar": val_result["metrics"]["uar"],
            "learning_rate": optimizer.param_groups[0]["lr"],
            "curriculum_dialogues": len(epoch_loader.dataset),
        }
        history.append(row)
        pd.DataFrame(history).to_csv(history_path, index=False)
        improved = score > best_score
        if improved:
            best_score, best_epoch = score, epoch
            epochs_without_improvement = 0
            guarded_numeric(
                lambda: save_checkpoint(
                    paths["checkpoints"] / "best_model.pt",
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    config=config,
                    epoch=epoch,
                    best_epoch=best_epoch,
                    best_val_weighted_f1=best_score,
                    split_ids=split_ids,
                )
            )
        else:
            epochs_without_improvement += 1
        last_epoch = epoch
        print(
            f"epoch={epoch} train_loss={train_result['loss']:.6f} "
            f"val_weighted_f1={score:.6f} best_epoch={best_epoch}"
        )
        if patience > 0 and epochs_without_improvement >= patience:
            break

    if best_epoch < 0:
        raise RuntimeError("no validation-selected checkpoint was produced")
    guarded_numeric(
        lambda: save_checkpoint(
            paths["checkpoints"] / "last_model.pt",
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            config=config,
            epoch=last_epoch,
            best_epoch=best_epoch,
            best_val_weighted_f1=best_score,
            split_ids=split_ids,
        )
    )

    best_path = paths["checkpoints"] / "best_model.pt"
    checkpoint = guarded_numeric(lambda: load_checkpoint(best_path, device))
    reloaded = rebuild_model_from_checkpoint(checkpoint, device)
    val_best = guarded_numeric(lambda: evaluate_model(reloaded, config, val_loader, device, None))
    test_loader = build_dataloader(config, "test", shuffle=False)
    test_best = guarded_numeric(
        lambda: evaluate_model(reloaded, config, test_loader, device, None)
    )
    evaluation_root = paths["logs"] / "evaluations"
    save_evaluation_outputs(
        evaluation_root / "val_best_model", config, "val", val_best, best_path, best_epoch
    )
    save_evaluation_outputs(
        evaluation_root / "test_best_model", config, "test", test_best, best_path, best_epoch
    )
    checkpoint_numeric = checkpoint["checkpoint_numeric_validation"]
    final_metrics_finite = all_finite_numbers(
        {
            "validation_loss": val_best["loss"],
            "validation_metrics": val_best["metrics"],
            "test_loss": test_best["loss"],
            "test_metrics": test_best["metrics"],
        }
    )
    prediction_count_correct = (
        len(val_best["prediction_rows"]) == len(val_best["y_true"])
        and len(test_best["prediction_rows"]) == len(test_best["y_true"])
    )
    run_pass = (
        checkpoint_numeric["checkpoint_numeric_validation"] == "passed"
        and bool(checkpoint_numeric["checkpoint_parameters_finite"])
        and final_metrics_finite
        and prediction_count_correct
    )
    summary = {
        "run_id": paths["run_id"],
        "run_dir": str(paths["run_dir"]),
        "run_status": "PASS" if run_pass else "NUMERICALLY_INVALID",
        "numeric_status": NUMERIC_STATUS_FINITE,
        "first_nonfinite_stage": None,
        "nonfinite_epoch": None,
        "nonfinite_batch": None,
        "best_epoch": best_epoch,
        "best_val_weighted_f1": best_score,
        "test_weighted_f1": test_best["metrics"]["weighted_f1"],
        "verified_feature_sha256": verified_sha,
        "checkpoint_reload": "passed",
        "checkpoint_numeric_validation": checkpoint_numeric[
            "checkpoint_numeric_validation"
        ],
        "checkpoint_nonfinite_tensor_count": checkpoint_numeric[
            "checkpoint_nonfinite_tensor_count"
        ],
        "checkpoint_nonfinite_element_count": checkpoint_numeric[
            "checkpoint_nonfinite_element_count"
        ],
        "checkpoint_parameters_finite": checkpoint_numeric[
            "checkpoint_parameters_finite"
        ],
        "final_metrics_finite": final_metrics_finite,
        "prediction_count_correct": prediction_count_correct,
        "test_split_used_for_selection": False,
        "final_validation_prediction_count": len(val_best["prediction_rows"]),
        "final_test_prediction_count": len(test_best["prediction_rows"]),
        "outer_test_valid_utterance_count": len(test_best["y_true"]),
        "training_time_seconds": time.perf_counter() - run_start,
        "peak_gpu_memory_mb": (
            torch.cuda.max_memory_allocated(device) / (1024**2)
            if device.type == "cuda"
            else 0.0
        ),
    }
    dump_json(summary, paths["logs"] / "run_summary.json")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def main() -> None:
    args = parse_args()
    run_training(
        Path(args.config),
        None if args.output_root is None else Path(args.output_root),
        args.device,
        None if args.resume is None else Path(args.resume),
        args.dry_run,
        args.experiment_date,
        args.experiment_group,
    )


if __name__ == "__main__":
    main()

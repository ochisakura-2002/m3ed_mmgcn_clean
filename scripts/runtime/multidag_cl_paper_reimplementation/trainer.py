"""Validation-clean Stage-B3 training, smoke, and locked evaluation runtime."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import random
import time
import tracemalloc
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from datasets.iemocap.official_feature_dataset import (
    IEMOCAPOfficialFeatureDataset,
    iemocap_dialogue_collate_fn,
)
from models.registry.paper_reimplementation import (
    REGISTRY_KEY,
    build_paper_reimplementation_model,
)

from models.multidag_cl.paper_reimplementation.config import MultiDAGCLConfig
from .adapter import FeatureRegistryMetadata, ProjectBatchAdapter
from .checkpoint import (
    TestEvaluationGate,
    ValidationCheckpointSelector,
    load_checkpoint,
    save_checkpoint_atomic,
    strict_reload_checkpoint,
)
from .curriculum import CurriculumRuntime
from .evaluation import (
    evaluate_model,
    export_evaluation,
    model_forward,
    move_batch,
)
from .manifest import (
    RunPaths,
    append_final_evaluation,
    append_resume_history,
    build_run_manifest,
    load_config,
    prepare_run_paths,
    write_manifest,
)
from .optimizer import audit_optimizer_parameters, build_optimizer
from .validation import LocalAssetUnavailable, validate_runtime_config


class SyntheticDialogueDataset(Dataset):
    """Deterministic canonical dialogue items for the controlled Stage-B3 smoke."""

    def __init__(self, config: Mapping[str, Any], split: str) -> None:
        synthetic = config["synthetic"]
        split_sizes = synthetic["split_sizes"]
        if split not in split_sizes:
            raise KeyError(f"unknown synthetic split: {split}")
        self.split = split
        self.size = int(split_sizes[split])
        self.sequence_length = int(synthetic["sequence_length"])
        self.seed = int(config["runtime"]["seed"])
        self.num_classes = int(config["model_core"]["data"]["num_classes"])
        dims = config["dataset"]["feature_dimensions"]
        self.dims = (int(dims["text"]), int(dims["audio"]), int(dims["visual"]))

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> dict[str, Any]:
        generator = torch.Generator().manual_seed(
            self.seed + {"train": 0, "val": 1000, "test": 2000}[self.split] + index
        )
        length = self.sequence_length - (index % 2)
        dialogue_id = f"synthetic_{self.split}_{index:03d}"
        text_dim, audio_dim, visual_dim = self.dims
        labels = torch.tensor(
            [(index + position) % self.num_classes for position in range(length)],
            dtype=torch.long,
        )
        speakers = torch.tensor([position % 2 for position in range(length)], dtype=torch.long)
        return {
            "dialogue_id": dialogue_id,
            "utterance_ids": [f"{dialogue_id}_utt_{position:03d}" for position in range(length)],
            "sentences": [f"synthetic utterance {position}" for position in range(length)],
            "text_features": torch.randn(length, text_dim, generator=generator),
            "audio_features": torch.randn(length, audio_dim, generator=generator),
            "visual_features": torch.randn(length, visual_dim, generator=generator),
            "labels": labels,
            "speaker_ids_int": speakers,
            "length": length,
        }


class ValidationCleanCoordinator:
    """Pure control-plane object used by both the real loop and leakage tests."""

    def __init__(self, *, test_evaluation_count: int) -> None:
        self.selector = ValidationCheckpointSelector()
        self.gate = TestEvaluationGate(maximum_evaluations=test_evaluation_count)

    def complete_validation(
        self,
        *,
        epoch: int,
        val_weighted_f1: float,
        val_loss: float,
        global_step: int = 0,
    ) -> bool:
        return self.selector.update(
            epoch=epoch,
            val_weighted_f1=val_weighted_f1,
            val_loss=val_loss,
            global_step=global_step,
        )

    def finish_and_lock(self):
        self.gate.mark_training_finished()
        best = self.selector.lock()
        self.gate.mark_checkpoint_locked()
        return best

    def mark_strict_reload(self) -> None:
        self.gate.mark_checkpoint_reloaded()

    def before_test(self) -> None:
        self.gate.before_test()


def seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))


def _project_root() -> Path:
    return next(
        parent
        for parent in Path(__file__).resolve().parents
        if (parent / "AGENTS.md").is_file() and (parent / "models").is_dir()
    )


def _device(config: Mapping[str, Any], override: Optional[str]) -> torch.device:
    requested = str(override or config["runtime"]["device"])
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def _make_iemocap_dataset(
    config: Mapping[str, Any],
    *,
    split: str,
    project_root: Path,
) -> IEMOCAPOfficialFeatureDataset:
    dataset_config = config["dataset"]
    feature_path = Path(dataset_config["feature_path"])
    if not feature_path.is_absolute():
        feature_path = project_root / feature_path
    return IEMOCAPOfficialFeatureDataset(
        feature_pkl_path=feature_path,
        split=split,
        val_split_strategy=dataset_config["val_split_strategy"],
        val_session_id=dataset_config["validation_session"],
        outer_test_session=dataset_config["test_session"],
        seed=int(config["runtime"]["seed"]),
    )


def _make_dataset(
    config: Mapping[str, Any],
    *,
    split: str,
    project_root: Path,
) -> Dataset:
    if str(config["dataset"]["name"]).upper() == "SYNTHETIC":
        return SyntheticDialogueDataset(config, split)
    return _make_iemocap_dataset(config, split=split, project_root=project_root)


def _loader(
    dataset: Dataset,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
    num_workers: int,
) -> DataLoader:
    generator = torch.Generator().manual_seed(int(seed))
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        generator=generator,
        num_workers=int(num_workers),
        collate_fn=iemocap_dialogue_collate_fn,
    )


def _dialogue_ids(dataset: Dataset) -> list[str]:
    if hasattr(dataset, "keys"):
        return [str(value) for value in getattr(dataset, "keys")]
    return [str(dataset[index]["dialogue_id"]) for index in range(len(dataset))]


def _split_membership_sha256(train_dataset: Dataset, val_dataset: Dataset) -> str:
    payload = json.dumps(
        {
            "train": _dialogue_ids(train_dataset),
            "validation": _dialogue_ids(val_dataset),
        },
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _expected_checkpoint_identity(
    config: Mapping[str, Any],
    core: MultiDAGCLConfig,
    feature: FeatureRegistryMetadata,
) -> dict[str, Any]:
    return {
        "registry_key": REGISTRY_KEY,
        "implementation_identity": core.implementation_identity,
        "conformance_profile": core.conformance_profile.value,
        "data_track": core.data_track.value,
        "model_config": core.to_mapping(),
        "num_classes": core.num_classes,
        "feature_dimensions": feature.feature_dimensions,
        "feature_registry": feature.registry_key,
        "feature_sha256": feature.feature_sha256,
        "split_protocol": config["dataset"]["split_protocol"],
    }


def _checkpoint_payload(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    config: Mapping[str, Any],
    core: MultiDAGCLConfig,
    feature: FeatureRegistryMetadata,
    curriculum: CurriculumRuntime,
    coordinator: ValidationCleanCoordinator,
    epoch: int,
    global_step: int,
    resolved_config_sha256: str,
    checkpoint_locked: bool,
) -> dict[str, Any]:
    best = coordinator.selector.best
    if best is None:
        raise RuntimeError("checkpoint payload requires a validation candidate")
    return {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": int(epoch),
        "global_step": int(global_step),
        "resolved_config_sha256": str(resolved_config_sha256),
        "split_membership_sha256": config["dataset"]["split_membership_sha256"],
        "best_validation_weighted_f1": best.val_weighted_f1,
        "best_validation_loss": best.val_loss,
        "best_epoch": best.epoch,
        "resolved_config": deepcopy(dict(config)),
        **_expected_checkpoint_identity(config, core, feature),
        "curriculum_membership_sha256": curriculum.manifest.membership_sha256,
        "configured_bucket_count": curriculum.manifest.configured_bucket_count,
        "actual_bucket_count": curriculum.manifest.actual_bucket_count,
        "selector_state": coordinator.selector.state_dict(),
        "test_gate_state": coordinator.gate.state_dict(),
        "test_split_used_for_selection": False,
        "checkpoint_locked": bool(checkpoint_locked),
        "training_finished": coordinator.gate.training_finished,
    }


def _write_epoch_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with Path(path).open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0]),
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def _total_gradient_norm(
    parameters: list[torch.nn.Parameter], norm_type: float
) -> float:
    gradients = [parameter.grad.detach() for parameter in parameters if parameter.grad is not None]
    if not gradients:
        return 0.0
    if math.isinf(norm_type):
        return max(float(gradient.abs().max().item()) for gradient in gradients)
    per_parameter = torch.stack(
        [
            torch.linalg.vector_norm(gradient, ord=norm_type).to(gradients[0].device)
            for gradient in gradients
        ]
    )
    return float(torch.linalg.vector_norm(per_parameter, ord=norm_type).item())


def _clip_gradients(
    model: torch.nn.Module, gradient_clipping: Mapping[str, Any]
) -> tuple[float, float]:
    parameters = [parameter for parameter in model.parameters() if parameter.grad is not None]
    if not parameters:
        raise RuntimeError("optimizer step has no gradients to clip")
    max_norm = float(gradient_clipping["max_norm"])
    norm_type = float(gradient_clipping["norm_type"])
    try:
        pre_clip = torch.nn.utils.clip_grad_norm_(
            parameters,
            max_norm=max_norm,
            norm_type=norm_type,
            error_if_nonfinite=bool(gradient_clipping["error_if_nonfinite"]),
        )
    except RuntimeError as error:
        if "non-finite" in str(error).lower() or "nonfinite" in str(error).lower():
            raise FloatingPointError(
                "non-finite gradient norm detected before optimizer.step"
            ) from error
        raise
    pre_clip_value = float(pre_clip.detach().cpu().item())
    if not math.isfinite(pre_clip_value):
        raise FloatingPointError("non-finite gradient norm detected before optimizer.step")
    post_clip_value = _total_gradient_norm(parameters, norm_type)
    if not math.isfinite(post_clip_value):
        raise FloatingPointError("non-finite gradient norm detected after clipping")
    return pre_clip_value, post_clip_value


def _training_summary_counts(
    *, epoch_rows: list[dict[str, Any]], total_train_batches: int, total_optimizer_steps: int
) -> dict[str, int]:
    return {
        "epoch_count": len(epoch_rows),
        "train_batch_count": int(total_train_batches),
        "optimizer_step_count": int(total_optimizer_steps),
    }


def _train_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    *,
    adapter: ProjectBatchAdapter,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    max_batches: Optional[int],
    maximum_optimizer_steps: Optional[int],
    global_step: int,
    gradient_clipping: Mapping[str, Any],
) -> tuple[dict[str, Any], int]:
    model.train()
    loss_numerator = 0.0
    utterance_count = 0
    batch_count = 0
    optimizer_steps = 0
    forward_seconds = 0.0
    backward_seconds = 0.0
    optimizer_seconds = 0.0
    gradient_clip_count = 0
    maximum_pre_clip_grad_norm = 0.0
    maximum_post_clip_grad_norm = 0.0
    nonfinite_gradient_count = 0
    predecessor_counts: list[int] = []
    for batch_index, raw_batch in enumerate(loader):
        if max_batches is not None and batch_index >= int(max_batches):
            break
        batch = move_batch(raw_batch, device)
        adapter.adapt(batch, split="train", require_labels=True)
        optimizer.zero_grad(set_to_none=True)
        started = time.perf_counter()
        output = model_forward(model, batch)
        forward_seconds += time.perf_counter() - started
        if output.loss is None:
            raise RuntimeError("training forward did not return loss")
        started = time.perf_counter()
        output.loss.backward()
        backward_seconds += time.perf_counter() - started
        if maximum_optimizer_steps is None or optimizer_steps < maximum_optimizer_steps:
            started = time.perf_counter()
            try:
                pre_clip_norm, post_clip_norm = _clip_gradients(
                    model, gradient_clipping
                )
            except FloatingPointError:
                nonfinite_gradient_count += 1
                raise
            gradient_clip_count += 1
            maximum_pre_clip_grad_norm = max(
                maximum_pre_clip_grad_norm, pre_clip_norm
            )
            maximum_post_clip_grad_norm = max(
                maximum_post_clip_grad_norm, post_clip_norm
            )
            optimizer.step()
            optimizer_seconds += time.perf_counter() - started
            optimizer_steps += 1
            global_step += 1
        count = int((batch["attention_mask"].bool() & (batch["labels"] != -100)).sum().item())
        loss_numerator += float(output.loss.detach().cpu().item()) * count
        utterance_count += count
        batch_count += 1
        adjacency = model.predecessor_builder.build(
            batch["speaker_ids_int"], batch["attention_mask"]
        )
        valid_counts = adjacency.sum(dim=-1)[batch["attention_mask"].bool()]
        predecessor_counts.extend(int(value) for value in valid_counts.detach().cpu().tolist())
    if batch_count == 0:
        raise RuntimeError("training loader produced no controlled batch")
    return (
        {
            "loss": loss_numerator / utterance_count,
            "batch_count": batch_count,
            "utterance_count": utterance_count,
            "optimizer_steps": optimizer_steps,
            "forward_seconds": forward_seconds,
            "backward_seconds": backward_seconds,
            "optimizer_step_seconds": optimizer_seconds,
            "gradient_clip_count": gradient_clip_count,
            "maximum_pre_clip_grad_norm": maximum_pre_clip_grad_norm,
            "maximum_post_clip_grad_norm": maximum_post_clip_grad_norm,
            "nonfinite_gradient_count": nonfinite_gradient_count,
            "mean_predecessor_count": (
                sum(predecessor_counts) / len(predecessor_counts)
                if predecessor_counts
                else 0.0
            ),
            "maximum_predecessor_count": max(predecessor_counts, default=0),
        },
        global_step,
    )


def _check_mode(
    config: Mapping[str, Any],
    *,
    core: MultiDAGCLConfig,
    feature: FeatureRegistryMetadata,
    device: torch.device,
) -> dict[str, Any]:
    model = build_paper_reimplementation_model(config).to(device)
    optimizer = build_optimizer(model, config["runtime"]["optimizer"])
    return {
        "status": "PASS",
        "mode": "check",
        "registry_key": REGISTRY_KEY,
        "model_parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "optimizer": audit_optimizer_parameters(model, optimizer),
        "optimizer_steps": 0,
        "gradient_clip_count": 0,
        "gradient_clipping": dict(config["runtime"]["gradient_clipping"]),
        "feature_registry": feature.registry_key,
        "feature_sha256": feature.feature_sha256,
        "feature_dimensions": feature.feature_dimensions,
        "split_protocol": config["dataset"]["split_protocol"],
        "conformance_profile": core.conformance_profile.value,
    }


def _lock_best_checkpoint(
    *,
    best_path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    config: Mapping[str, Any],
    core: MultiDAGCLConfig,
    feature: FeatureRegistryMetadata,
    curriculum: CurriculumRuntime,
    coordinator: ValidationCleanCoordinator,
    resolved_config_sha256: str,
) -> None:
    selected = load_checkpoint(best_path, torch.device("cpu"))
    model.load_state_dict(selected["model_state_dict"], strict=True)
    optimizer.load_state_dict(selected["optimizer_state_dict"])
    best = coordinator.selector.best
    if best is None:
        raise RuntimeError("best checkpoint identity is missing")
    selected_best = selected.get("selector_state", {}).get("best")
    expected_best = {
        "epoch": best.epoch,
        "val_weighted_f1": best.val_weighted_f1,
        "val_loss": best.val_loss,
        "global_step": best.global_step,
    }
    if selected_best != expected_best:
        raise RuntimeError("selected checkpoint selector.best is inconsistent")
    if int(selected.get("epoch", -1)) != best.epoch:
        raise RuntimeError("selected checkpoint epoch is inconsistent with best_epoch")
    if int(selected.get("global_step", -1)) != best.global_step:
        raise RuntimeError("selected checkpoint global_step is inconsistent with best epoch")
    if float(selected.get("best_validation_weighted_f1")) != best.val_weighted_f1:
        raise RuntimeError("selected checkpoint best Validation Weighted-F1 is inconsistent")
    if float(selected.get("best_validation_loss")) != best.val_loss:
        raise RuntimeError("selected checkpoint best Validation loss is inconsistent")
    locked_payload = _checkpoint_payload(
        model=model,
        optimizer=optimizer,
        config=config,
        core=core,
        feature=feature,
        curriculum=curriculum,
        coordinator=coordinator,
        epoch=int(selected["epoch"]),
        global_step=int(selected["global_step"]),
        resolved_config_sha256=resolved_config_sha256,
        checkpoint_locked=True,
    )
    save_checkpoint_atomic(best_path, locked_payload)


def run_runtime(
    config_path: Path,
    *,
    mode: str,
    output_root_override: Optional[Path] = None,
    experiment_date: Optional[str] = None,
    experiment_group: Optional[str] = None,
    device_override: Optional[str] = None,
    resume_checkpoint: Optional[Path] = None,
) -> dict[str, Any]:
    project_root = _project_root()
    raw_config = load_config(config_path)
    config = deepcopy(raw_config)
    core, feature = validate_runtime_config(
        config,
        mode=mode,
        project_root=project_root,
    )
    effective_output_root_override = output_root_override
    if config["runtime"]["smoke_only"] and effective_output_root_override is None:
        effective_output_root_override = Path(config["runtime"]["smoke_output_root"])
    if effective_output_root_override is not None and config["runtime"]["smoke_only"]:
        candidate = Path(effective_output_root_override)
        if not candidate.is_absolute():
            candidate = project_root / candidate
        allowed = project_root / (
            "tmp/assistant_work/paper_reproduction_stage_b3_runtime_correction/"
            "smoke_outputs"
        )
        try:
            candidate.resolve().relative_to(allowed.resolve())
        except ValueError as error:
            raise ValueError("smoke output override must stay under the Stage-B3 tmp root") from error
    device = _device(config, device_override)
    seed_everything(int(config["runtime"]["seed"]))
    if mode == "check":
        return _check_mode(config, core=core, feature=feature, device=device)
    if mode == "evaluate":
        if resume_checkpoint is None:
            raise ValueError("evaluate mode requires a locked checkpoint path")
        return evaluate_locked_checkpoint(
            config,
            checkpoint_path=resume_checkpoint,
            core=core,
            feature=feature,
            project_root=project_root,
            device=device,
        )

    train_dataset = _make_dataset(config, split="train", project_root=project_root)
    val_dataset = _make_dataset(config, split="val", project_root=project_root)
    config["dataset"]["split_membership_sha256"] = _split_membership_sha256(
        train_dataset, val_dataset
    )
    curriculum = CurriculumRuntime.from_training_dataset(
        train_dataset,
        split="train",
        bucket_count=core.bucket_count,
        partition_profile=core.curriculum_partition,
        schedule_profile=core.curriculum_schedule,
        enabled=core.curriculum_enabled,
    )
    resolved_config, paths = prepare_run_paths(
        config,
        config_path=config_path,
        project_root=project_root,
        output_root_override=effective_output_root_override,
        experiment_date=experiment_date,
        experiment_group=experiment_group,
        resume_run_dir=(
            None if resume_checkpoint is None else Path(resume_checkpoint).parent.parent
        ),
    )
    curriculum.export_manifest(paths.manifests / "curriculum_bucket_manifest.tsv")
    model = build_paper_reimplementation_model(resolved_config).to(device)
    adapter = ProjectBatchAdapter(core, feature)
    optimizer = build_optimizer(model, resolved_config["runtime"]["optimizer"])
    optimizer_audit = audit_optimizer_parameters(model, optimizer)
    manifest = build_run_manifest(
        config=resolved_config,
        paths=paths,
        project_root=project_root,
        feature=feature,
        bucket_membership_sha256=curriculum.manifest.membership_sha256,
        configured_bucket_count=curriculum.manifest.configured_bucket_count,
        actual_bucket_count=curriculum.manifest.actual_bucket_count,
        optimizer_audit=optimizer_audit,
        command_entrypoint="scripts/models/multidag_cl/paper_reimplementation/train.py",
    )
    resolved_config_sha256 = str(manifest["resolved_config_sha256"])
    if resume_checkpoint is None:
        write_manifest(paths.run_manifest, manifest)

    runtime = resolved_config["runtime"]
    limits = runtime["limits"]
    label_names = list(resolved_config["dataset"]["label_names"])
    label_ids = list(range(core.num_classes))
    coordinator = ValidationCleanCoordinator(
        test_evaluation_count=int(resolved_config["checkpoint"]["test_evaluation_count"])
    )
    global_step = 0
    start_epoch = 1
    if resume_checkpoint is not None:
        checkpoint = strict_reload_checkpoint(
            resume_checkpoint,
            model=model,
            optimizer=optimizer,
            device=device,
            expected_identity=_expected_checkpoint_identity(resolved_config, core, feature),
        )
        if checkpoint.get("resolved_config_sha256") != resolved_config_sha256:
            raise ValueError("resume resolved config SHA256 mismatch")
        if checkpoint["curriculum_membership_sha256"] != curriculum.manifest.membership_sha256:
            raise ValueError("resume curriculum membership SHA256 mismatch")
        coordinator.selector.load_state_dict(checkpoint["selector_state"])
        if coordinator.selector.locked:
            raise RuntimeError("a locked completed checkpoint cannot resume training")
        start_epoch = curriculum.resume_global_epoch(
            int(checkpoint["epoch"]), checkpoint["curriculum_membership_sha256"]
        )
        global_step = int(checkpoint["global_step"])
        append_resume_history(
            paths.run_manifest,
            checkpoint_path=Path(resume_checkpoint),
            checkpoint=checkpoint,
            run_dir=paths.run_dir,
            project_root=project_root,
            expected_identity={
                "registry_key": REGISTRY_KEY,
                "implementation_identity": core.implementation_identity,
                "conformance_profile": core.conformance_profile.value,
                "data_track": core.data_track.value,
                "feature_sha256": feature.feature_sha256,
                "split_protocol": resolved_config["dataset"]["split_protocol"],
                "split_membership_sha256": resolved_config["dataset"][
                    "split_membership_sha256"
                ],
                "bucket_membership_sha256": curriculum.manifest.membership_sha256,
                "resolved_config_sha256": resolved_config_sha256,
            },
        )

    val_loader = _loader(
        val_dataset,
        batch_size=int(runtime["batch_size"]),
        shuffle=False,
        seed=int(runtime["seed"]),
        num_workers=int(runtime["num_workers"]),
    )
    epoch_rows: list[dict[str, Any]] = []
    total_optimizer_steps = 0
    total_train_batches = 0
    total_gradient_clip_count = 0
    maximum_pre_clip_grad_norm = 0.0
    maximum_post_clip_grad_norm = 0.0
    total_nonfinite_gradient_count = 0
    total_forward_seconds = 0.0
    total_backward_seconds = 0.0
    total_optimizer_seconds = 0.0
    last_train_result: dict[str, Any] = {}
    last_epoch = 0
    tracemalloc.start()
    run_started = time.perf_counter()

    epoch_count = int(runtime["epochs"])
    if mode == "real-batch-smoke":
        epoch_numbers = [1]
    else:
        epoch_numbers = list(range(start_epoch, epoch_count + 1))
    for epoch in epoch_numbers:
        train_loader = curriculum.build_loader(
            train_dataset,
            global_epoch=epoch,
            batch_size=int(runtime["batch_size"]),
            seed=int(runtime["seed"]),
            collate_fn=iemocap_dialogue_collate_fn,
            num_workers=int(runtime["num_workers"]),
        )
        maximum_steps = None
        max_train_batches = None
        max_val_batches = None
        if mode == "synthetic-smoke":
            maximum_steps = int(limits["optimizer_steps"])
            max_train_batches = int(limits["train_batches"])
            max_val_batches = int(limits["validation_batches"])
        elif mode == "real-batch-smoke":
            maximum_steps = 0
            max_train_batches = int(limits["train_batches"])
            max_val_batches = int(limits["validation_batches"])
        train_result, global_step = _train_epoch(
            model,
            train_loader,
            adapter=adapter,
            optimizer=optimizer,
            device=device,
            max_batches=max_train_batches,
            maximum_optimizer_steps=maximum_steps,
            global_step=global_step,
            gradient_clipping=runtime["gradient_clipping"],
        )
        total_optimizer_steps += int(train_result["optimizer_steps"])
        total_train_batches += int(train_result["batch_count"])
        total_gradient_clip_count += int(train_result["gradient_clip_count"])
        maximum_pre_clip_grad_norm = max(
            maximum_pre_clip_grad_norm,
            float(train_result["maximum_pre_clip_grad_norm"]),
        )
        maximum_post_clip_grad_norm = max(
            maximum_post_clip_grad_norm,
            float(train_result["maximum_post_clip_grad_norm"]),
        )
        total_nonfinite_gradient_count += int(train_result["nonfinite_gradient_count"])
        total_forward_seconds += float(train_result["forward_seconds"])
        total_backward_seconds += float(train_result["backward_seconds"])
        total_optimizer_seconds += float(train_result["optimizer_step_seconds"])
        val_result = evaluate_model(
            model,
            val_loader,
            adapter=adapter,
            device=device,
            split="validation",
            labels=label_ids,
            max_batches=max_val_batches,
        )
        improved = coordinator.complete_validation(
            epoch=epoch,
            val_weighted_f1=float(val_result["metrics"]["weighted_f1"]),
            val_loss=float(val_result["metrics"]["loss"]),
            global_step=global_step,
        )
        visible_indices = curriculum.visible_indices(epoch)
        visible_utterances = sum(
            int(train_dataset[index].get("length", train_dataset[index].get("num_utterances")))
            for index in visible_indices
        )
        epoch_rows.append(
            {
                "epoch": epoch,
                "visible_bucket_count": curriculum.visible_bucket_count(epoch),
                "visible_dialogue_count": len(visible_indices),
                "visible_utterance_count": visible_utterances,
                "train_loss": f"{train_result['loss']:.9f}",
                "train_batch_count": train_result["batch_count"],
                "optimizer_step_count": train_result["optimizer_steps"],
                "gradient_clip_count": train_result["gradient_clip_count"],
                "maximum_pre_clip_grad_norm": f"{train_result['maximum_pre_clip_grad_norm']:.9f}",
                "maximum_post_clip_grad_norm": f"{train_result['maximum_post_clip_grad_norm']:.9f}",
                "nonfinite_gradient_count": train_result["nonfinite_gradient_count"],
                "val_accuracy": f"{val_result['metrics']['accuracy']:.9f}",
                "val_weighted_f1": f"{val_result['metrics']['weighted_f1']:.9f}",
                "val_loss": f"{val_result['metrics']['loss']:.9f}",
            }
        )
        if improved:
            save_checkpoint_atomic(
                paths.checkpoints / "best_model.pt",
                _checkpoint_payload(
                    model=model,
                    optimizer=optimizer,
                    config=resolved_config,
                    core=core,
                    feature=feature,
                    curriculum=curriculum,
                    coordinator=coordinator,
                    epoch=epoch,
                    global_step=global_step,
                    resolved_config_sha256=resolved_config_sha256,
                    checkpoint_locked=False,
                ),
            )
        last_train_result = train_result
        last_epoch = epoch
    if not epoch_rows:
        raise RuntimeError("training protocol produced no epoch/controlled probe")
    _write_epoch_rows(paths.logs / "epoch_metrics.tsv", epoch_rows)
    save_checkpoint_atomic(
        paths.checkpoints / "last_model.pt",
        _checkpoint_payload(
            model=model,
            optimizer=optimizer,
            config=resolved_config,
            core=core,
            feature=feature,
            curriculum=curriculum,
            coordinator=coordinator,
            epoch=last_epoch,
            global_step=global_step,
            resolved_config_sha256=resolved_config_sha256,
            checkpoint_locked=False,
        ),
    )

    best = coordinator.finish_and_lock()
    best_path = paths.checkpoints / "best_model.pt"
    _lock_best_checkpoint(
        best_path=best_path,
        model=model,
        optimizer=optimizer,
        config=resolved_config,
        core=core,
        feature=feature,
        curriculum=curriculum,
        coordinator=coordinator,
        resolved_config_sha256=resolved_config_sha256,
    )
    strict_reload_checkpoint(
        best_path,
        model=model,
        device=device,
        expected_identity=_expected_checkpoint_identity(resolved_config, core, feature),
        require_locked=True,
    )
    coordinator.mark_strict_reload()
    val_best = evaluate_model(
        model,
        val_loader,
        adapter=adapter,
        device=device,
        split="validation_best",
        labels=label_ids,
        max_batches=(
            int(limits["validation_batches"])
            if mode in {"synthetic-smoke", "real-batch-smoke"}
            else None
        ),
    )
    export_evaluation(
        val_best,
        reports_dir=paths.reports,
        predictions_dir=paths.predictions,
        label_names=label_names,
    )

    final_test = None
    if int(resolved_config["checkpoint"]["test_evaluation_count"]) == 1:
        coordinator.before_test()
        test_dataset = _make_dataset(resolved_config, split="test", project_root=project_root)
        test_loader = _loader(
            test_dataset,
            batch_size=int(runtime["batch_size"]),
            shuffle=False,
            seed=int(runtime["seed"]),
            num_workers=int(runtime["num_workers"]),
        )
        final_test = evaluate_model(
            model,
            test_loader,
            adapter=adapter,
            device=device,
            split="test" if runtime["formal_experiment"] else "fake_test",
            labels=label_ids,
            max_batches=(
                int(limits["test_batches"]) if mode == "synthetic-smoke" else None
            ),
        )
        export_evaluation(
            final_test,
            reports_dir=paths.reports,
            predictions_dir=paths.predictions,
            label_names=label_names,
        )

    current_memory, peak_memory = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    counts = _training_summary_counts(
        epoch_rows=epoch_rows,
        total_train_batches=total_train_batches,
        total_optimizer_steps=total_optimizer_steps,
    )
    summary = {
        "status": "PASS",
        "mode": mode,
        "run_id": paths.run_id,
        "run_dir": paths.run_dir.as_posix(),
        "best_epoch": best.epoch,
        "best_validation_weighted_f1": best.val_weighted_f1,
        "best_validation_loss": best.val_loss,
        "test_split_used_for_selection": False,
        "test_evaluation_count": coordinator.gate.test_evaluation_count,
        **counts,
        "optimizer_steps": counts["optimizer_step_count"],
        "train_batches": counts["train_batch_count"],
        "deprecated_aliases": {
            "optimizer_steps": "optimizer_step_count",
            "train_batches": "train_batch_count",
        },
        "gradient_clip_count": total_gradient_clip_count,
        "maximum_pre_clip_grad_norm": maximum_pre_clip_grad_norm,
        "maximum_post_clip_grad_norm": maximum_post_clip_grad_norm,
        "nonfinite_gradient_count": total_nonfinite_gradient_count,
        "forward_time_seconds": total_forward_seconds,
        "backward_time_seconds": total_backward_seconds,
        "optimizer_step_time_seconds": total_optimizer_seconds,
        "runtime_seconds": time.perf_counter() - run_started,
        "python_tracemalloc_current_bytes": current_memory,
        "python_tracemalloc_peak_bytes": peak_memory,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "dialogue_count": len(train_dataset),
        "utterance_count": sum(
            int(train_dataset[index].get("length", train_dataset[index].get("num_utterances")))
            for index in range(len(train_dataset))
        ),
        "maximum_dialogue_length": max(
            int(train_dataset[index].get("length", train_dataset[index].get("num_utterances")))
            for index in range(len(train_dataset))
        ),
        "mean_predecessor_count": last_train_result["mean_predecessor_count"],
        "maximum_predecessor_count": last_train_result["maximum_predecessor_count"],
        "validation_metrics": val_best["metrics"],
        "test_metrics": None if final_test is None else final_test["metrics"],
        "checkpoint_locked": True,
        "checkpoint_reloaded": True,
        "formal_training_started": 1 if runtime["formal_experiment"] else 0,
    }
    (paths.logs / "run_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    append_final_evaluation(
        paths.run_manifest,
        {
            "best_epoch": best.epoch,
            "validation_metrics": val_best["metrics"],
            "test_metrics": None if final_test is None else final_test["metrics"],
            "test_evaluation_count_actual": coordinator.gate.test_evaluation_count,
            "test_split_used_for_selection": False,
            "best_checkpoint_locked": True,
            "best_checkpoint_reloaded": True,
        },
    )
    return summary


def evaluate_locked_checkpoint(
    config: Mapping[str, Any],
    *,
    checkpoint_path: Path,
    core: MultiDAGCLConfig,
    feature: FeatureRegistryMetadata,
    project_root: Path,
    device: torch.device,
) -> dict[str, Any]:
    model = build_paper_reimplementation_model(config).to(device)
    checkpoint = strict_reload_checkpoint(
        checkpoint_path,
        model=model,
        device=device,
        expected_identity=_expected_checkpoint_identity(config, core, feature),
        require_locked=True,
    )
    if checkpoint.get("training_finished") is not True:
        raise ValueError("evaluation requires training_finished=true")
    test_dataset = _make_dataset(config, split="test", project_root=project_root)
    test_loader = _loader(
        test_dataset,
        batch_size=int(config["runtime"]["batch_size"]),
        shuffle=False,
        seed=int(config["runtime"]["seed"]),
        num_workers=int(config["runtime"]["num_workers"]),
    )
    adapter = ProjectBatchAdapter(core, feature)
    result = evaluate_model(
        model,
        test_loader,
        adapter=adapter,
        device=device,
        split="fake_test" if config["runtime"]["smoke_only"] else "test",
        labels=list(range(core.num_classes)),
        max_batches=(
            int(config["runtime"]["limits"]["test_batches"])
            if config["runtime"]["smoke_only"]
            else None
        ),
    )
    return {
        "status": "PASS",
        "mode": "evaluate",
        "evaluation_only": True,
        "checkpoint_locked": True,
        "test_split_used_for_selection": False,
        "training_manifest_mutated": False,
        "metrics": result["metrics"],
    }


__all__ = [
    "LocalAssetUnavailable",
    "SyntheticDialogueDataset",
    "ValidationCleanCoordinator",
    "evaluate_locked_checkpoint",
    "run_runtime",
    "seed_everything",
]

"""Audit model-level dialogue causality on synthetic or real IEMOCAP batches."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch
import yaml


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


FEATURE_KEYS = ("text_features", "audio_features", "visual_features")
PERTURBATIONS = ("zero", "random_noise", "cross_sample_shuffle")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test a configured dialogue model for future-information leakage."
    )
    parser.add_argument("--config", required=True, help="Training YAML path.")
    parser.add_argument("--output-dir", required=True, help="Audit output directory.")
    parser.add_argument("--mode", choices=("synthetic", "real_batch"), default="synthetic")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--perturbation-seeds", default=None, help="Comma-separated seeds.")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--sequence-length", type=int, default=5)
    parser.add_argument("--target-index", type=int, default=2)
    parser.add_argument(
        "--target-policy", choices=("single", "auto_multiple"), default="single"
    )
    parser.add_argument("--target-indices", default=None, help="Comma-separated cutoffs.")
    parser.add_argument("--class-id", type=int, default=0)
    parser.add_argument(
        "--gradient-target",
        choices=(
            "fixed_class_logit",
            "current_predicted_logit",
            "history_squared_logit_sum",
        ),
        default="fixed_class_logit",
    )
    parser.add_argument("--tolerance", type=float, default=1.0e-6)
    parser.add_argument("--real-split", choices=("train", "val", "test"), default="val")
    return parser.parse_args()


def parse_int_list(value: Optional[str]) -> Optional[List[int]]:
    if value is None or str(value).strip() == "":
        return None
    return [int(part.strip()) for part in str(value).split(",") if part.strip()]


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise TypeError(f"{path} must contain a YAML mapping.")
    return config


def model_family(config: Mapping[str, Any]) -> str:
    name = str(config.get("model", {}).get("name", ""))
    if name == "MMGCN":
        return "MMGCN"
    if name == "MultiDAGCL":
        return "MultiDAGCL"
    from models.baselines.causal_baseline_registry import normalize_new_causal_model_name

    normalized = normalize_new_causal_model_name(name)
    return normalized


def build_model(config: Dict[str, Any], family: str) -> torch.nn.Module:
    if family == "MMGCN":
        from scripts.train_mmgcn import build_model as project_builder

        return project_builder(config)
    if family == "MultiDAGCL":
        from scripts.baselines.train_multidag_cl import build_model as project_builder

        return project_builder(config)
    from models.baselines.causal_baseline_registry import build_new_causal_baseline

    return build_new_causal_baseline(config)


def _feature_dim(model: Mapping[str, Any], modality: str) -> int:
    direct = f"{modality}_dim"
    legacy = f"{modality}_feature_dim"
    if direct in model:
        return int(model[direct])
    return int(model[legacy])


def build_synthetic_batch(
    config: Dict[str, Any], batch_size: int, seq_len: int, seed: int
) -> Dict[str, Any]:
    if batch_size < 2:
        raise ValueError("Synthetic cross-sample shuffle requires batch_size >= 2.")
    if seq_len < 3:
        raise ValueError("Synthetic audit requires sequence_length >= 3.")
    model = config["model"]
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    batch: Dict[str, Any] = {
        f"{modality}_features": torch.randn(
            batch_size,
            seq_len,
            _feature_dim(model, modality),
            generator=generator,
        )
        for modality in ("text", "audio", "visual")
    }
    lengths = torch.full((batch_size,), seq_len, dtype=torch.long)
    lengths[-1] = seq_len - 1
    positions = torch.arange(seq_len).unsqueeze(0)
    attention_mask = positions < lengths.unsqueeze(1)
    batch.update(
        {
            "lengths": lengths,
            "attention_mask": attention_mask,
            "speaker_ids_int": positions.expand(batch_size, -1) % int(model.get("num_speakers", 2)),
            "labels": torch.zeros(batch_size, seq_len, dtype=torch.long),
            "dialogue_ids": [f"synthetic_{index}" for index in range(batch_size)],
        }
    )
    return batch


def build_real_batch(
    config: Dict[str, Any], family: str, split: str, batch_size: int
) -> Dict[str, Any]:
    if str(config.get("dataset", {}).get("name", "")).upper() != "IEMOCAP":
        raise ValueError("real_batch mode currently supports IEMOCAP configs only.")
    if family == "MMGCN":
        from scripts.train_mmgcn import build_dataloader

        loader = build_dataloader(config, split=split, shuffle=False)
    elif family == "MultiDAGCL":
        from scripts.baselines.train_multidag_cl import build_dataloader

        loader = build_dataloader(config, split=split, shuffle=False, batch_size=int(batch_size))
    else:
        from scripts.baselines.new_causal_graph_runtime import (
            build_dataloader,
            normalized_training_config,
            verify_feature_sha256,
        )

        normalized = normalized_training_config(config)
        config.clear()
        config.update(normalized)
        verify_feature_sha256(config)
        loader = build_dataloader(config, split=split, shuffle=False, batch_size=int(batch_size))
    try:
        return next(iter(loader))
    except StopIteration as error:
        raise RuntimeError(f"IEMOCAP split={split!r} yielded no batches.") from error


def move_batch(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def forward_model(
    model: torch.nn.Module,
    family: str,
    batch: Dict[str, Any],
    return_graph: bool = False,
) -> Dict[str, Any]:
    common = {
        "text_features": batch["text_features"],
        "audio_features": batch["audio_features"],
        "visual_features": batch["visual_features"],
        "attention_mask": batch["attention_mask"],
        "speaker_ids_int": batch["speaker_ids_int"],
    }
    if family == "MMGCN":
        return model(**common, lengths=batch["lengths"], return_graph=bool(return_graph))
    if family == "MultiDAGCL":
        return model(**common, labels=None)
    output = model(
        **common,
        lengths=batch["lengths"],
        return_aux=bool(return_graph),
    )
    return output if isinstance(output, dict) else {"logits": output}


def valid_history_mask(batch: Dict[str, Any], target_index: int) -> torch.Tensor:
    positions = torch.arange(
        batch["attention_mask"].shape[1], device=batch["attention_mask"].device
    ).unsqueeze(0)
    return batch["attention_mask"].bool() & (positions <= int(target_index))


def future_mask(batch: Dict[str, Any], target_index: int) -> torch.Tensor:
    positions = torch.arange(
        batch["attention_mask"].shape[1], device=batch["attention_mask"].device
    ).unsqueeze(0)
    return batch["attention_mask"].bool() & (positions > int(target_index))


def perturb_future(
    batch: Dict[str, Any], method: str, target_index: int, seed: int
) -> Dict[str, Any]:
    perturbed = {
        key: value.clone() if torch.is_tensor(value) else value
        for key, value in batch.items()
    }
    mask = future_mask(batch, target_index)
    generator = torch.Generator(device=batch["text_features"].device).manual_seed(int(seed))
    for key in FEATURE_KEYS:
        source = batch[key]
        if method == "zero":
            replacement = torch.zeros_like(source)
        elif method == "random_noise":
            replacement = torch.randn(
                source.shape,
                dtype=source.dtype,
                device=source.device,
                generator=generator,
            )
        elif method == "cross_sample_shuffle":
            replacement = torch.roll(source, shifts=1, dims=0)
        else:
            raise ValueError(f"Unsupported perturbation method: {method}")
        perturbed[key] = torch.where(mask.unsqueeze(-1), replacement, source)
    return perturbed


def max_abs_diff(left: torch.Tensor, right: torch.Tensor) -> float:
    if left.numel() == 0:
        return 0.0
    return float((left - right).abs().max().item())


def future_perturbation_test(
    model: torch.nn.Module,
    family: str,
    batch: Dict[str, Any],
    target_index: int,
    seed: int,
) -> Tuple[Dict[str, float], List[Dict[str, Any]]]:
    history = valid_history_mask(batch, target_index)
    with torch.no_grad():
        original = forward_model(model, family, batch)["logits"]
    maxima: Dict[str, float] = {}
    details: List[Dict[str, Any]] = []
    for offset, method in enumerate(PERTURBATIONS):
        changed = perturb_future(batch, method, target_index, seed + offset + 1)
        with torch.no_grad():
            changed_logits = forward_model(model, family, changed)["logits"]
        difference = (original - changed_logits).abs().amax(dim=-1)
        maxima[method] = float(difference[history].max().item())
        for batch_index, time_index in history.nonzero(as_tuple=False).tolist():
            details.append(
                {
                    "seed": int(seed),
                    "target_index": int(target_index),
                    "perturbation": method,
                    "batch_index": int(batch_index),
                    "time_index": int(time_index),
                    "max_abs_logit_diff": float(difference[batch_index, time_index].item()),
                }
            )
    return maxima, details


def prefix_batch(batch: Mapping[str, Any], target_index: int) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    full_length = batch["attention_mask"].shape[1]
    prefix_length = int(target_index) + 1
    for key, value in batch.items():
        if torch.is_tensor(value) and value.dim() >= 2 and value.shape[1] == full_length:
            result[key] = value[:, :prefix_length].clone()
        elif torch.is_tensor(value):
            result[key] = value.clone()
        else:
            result[key] = value
    result["lengths"] = batch["lengths"].clamp_max(prefix_length)
    return result


def prefix_full_equivalence_test(
    model: torch.nn.Module,
    family: str,
    batch: Dict[str, Any],
    target_index: int,
) -> Tuple[float, Dict[str, float]]:
    prefix = prefix_batch(batch, target_index)
    with torch.no_grad():
        full = forward_model(model, family, batch, return_graph=True)
        short = forward_model(model, family, prefix, return_graph=True)
    valid = batch["attention_mask"][:, target_index].bool()
    logits_diff = max_abs_diff(
        full["logits"][valid, target_index], short["logits"][valid, -1]
    )
    extras: Dict[str, float] = {}
    if family == "causal_gsmcc_inspired":
        extras["low_frequency_prefix_full_diff"] = max_abs_diff(
            full["low_frequency_repr"][valid, target_index],
            short["low_frequency_repr"][valid, -1],
        )
        extras["high_frequency_prefix_full_diff"] = max_abs_diff(
            full["high_frequency_repr"][valid, target_index],
            short["high_frequency_repr"][valid, -1],
        )
    elif family == "causal_dialoguegcn":
        extras["context_prefix_full_diff"] = max_abs_diff(
            full["context_repr"][valid, target_index], short["context_repr"][valid, -1]
        )
        extras["graph_prefix_full_diff"] = max_abs_diff(
            full["graph_repr"][valid, target_index], short["graph_repr"][valid, -1]
        )
    return logits_diff, extras


def future_gradient_test(
    model: torch.nn.Module,
    family: str,
    batch: Dict[str, Any],
    target_index: int,
    class_id: int,
    gradient_target: str = "fixed_class_logit",
) -> Dict[str, float]:
    gradient_batch = {
        key: value.clone() if torch.is_tensor(value) else value
        for key, value in batch.items()
    }
    inputs = []
    for key in FEATURE_KEYS:
        gradient_batch[key] = gradient_batch[key].detach().requires_grad_(True)
        inputs.append(gradient_batch[key])
    logits = forward_model(model, family, gradient_batch)["logits"]
    valid_current = gradient_batch["attention_mask"][:, target_index].bool()
    if gradient_target == "fixed_class_logit":
        if class_id < 0 or class_id >= logits.shape[-1]:
            raise ValueError(f"class_id={class_id} is outside the model output range.")
        scalar = logits[valid_current, target_index, int(class_id)].sum()
    elif gradient_target == "current_predicted_logit":
        current = logits[valid_current, target_index]
        predicted = current.detach().argmax(dim=-1, keepdim=True)
        scalar = current.gather(1, predicted).sum()
    elif gradient_target == "history_squared_logit_sum":
        history = valid_history_mask(gradient_batch, target_index)
        scalar = logits[history].pow(2).sum()
    else:
        raise ValueError(f"Unsupported gradient target: {gradient_target}")
    gradients = torch.autograd.grad(scalar, inputs, allow_unused=True)
    mask = future_mask(gradient_batch, target_index).unsqueeze(-1)
    result: Dict[str, float] = {}
    for key, gradient in zip(FEATURE_KEYS, gradients):
        short_name = key.replace("_features", "")
        if gradient is None or not bool(mask.any()):
            result[short_name] = 0.0
        else:
            selected = gradient.masked_select(mask.expand_as(gradient))
            result[short_name] = 0.0 if selected.numel() == 0 else float(selected.abs().max().item())
    return result


def edge_violation_counts(
    output: Mapping[str, Any], family: str, batch: Mapping[str, Any]
) -> Dict[str, int]:
    adjacency = output["adjacency"].to(dtype=torch.bool)
    future = 0
    padding = 0
    cross_dialogue = 0
    if family == "MMGCN":
        positions: List[Tuple[int, int]] = []
        for batch_index, length in enumerate(batch["lengths"].detach().cpu().tolist()):
            positions.extend((batch_index, time_index) for time_index in range(int(length)))
        node_positions = positions * 3
        for target_node, source_node in adjacency.nonzero().detach().cpu().tolist():
            target_batch, target_time = node_positions[target_node]
            source_batch, source_time = node_positions[source_node]
            cross_dialogue += int(source_batch != target_batch)
            future += int(source_batch == target_batch and source_time > target_time)
        return {"future": future, "padding": padding, "cross_dialogue": cross_dialogue}

    batch_size, target_nodes, _ = adjacency.shape
    seq_len = batch["attention_mask"].shape[1]
    if family == "causal_gsmcc_inspired":
        num_modalities = target_nodes // seq_len
        times = torch.arange(seq_len, device=adjacency.device).repeat_interleave(num_modalities)
        valid_nodes = batch["attention_mask"].bool().unsqueeze(-1).expand(
            batch_size, seq_len, num_modalities
        ).reshape(batch_size, target_nodes)
    else:
        times = torch.arange(seq_len, device=adjacency.device)
        valid_nodes = batch["attention_mask"].bool()
    illegal_future = times.view(1, 1, -1) > times.view(1, -1, 1)
    future = int((adjacency & illegal_future).sum().item())
    legal_node_pair = valid_nodes.unsqueeze(2) & valid_nodes.unsqueeze(1)
    padding = int((adjacency & ~legal_node_pair).sum().item())
    return {"future": future, "padding": padding, "cross_dialogue": 0}


def gsmcc_future_path_violations(
    output: Mapping[str, Any], config: Mapping[str, Any]
) -> int:
    adjacency = output["adjacency"].to(dtype=torch.bool)
    node_time = output["causal_diagnostics"]["node_time"]
    steps = int(config["model"].get("num_filter_steps", 2)) * int(
        config["model"].get("num_graph_layers", 1)
    )
    reach = adjacency.clone()
    violations = 0
    illegal = node_time.view(1, 1, -1) > node_time.view(1, -1, 1)
    for _ in range(steps):
        violations += int((reach & illegal).sum().item())
        reach = torch.bmm(reach.float(), adjacency.float()) > 0
    return violations


def model_specific_diagnostics(
    model: torch.nn.Module,
    family: str,
    batch: Dict[str, Any],
    config: Mapping[str, Any],
) -> Dict[str, Any]:
    with torch.no_grad():
        output = forward_model(model, family, batch, return_graph=True)
    if family == "causal_gsmcc_inspired":
        return {
            "future_filter_path_violations": gsmcc_future_path_violations(output, config)
        }
    if family == "causal_dialoguegcn":
        adjacency = output["adjacency"].bool()
        invalid_attention = output["edge_attention"].masked_select(~adjacency)
        return {
            "relation_mapping": str(output["relation_mapping"]),
            "future_relation_count": edge_violation_counts(output, family, batch)["future"],
            "max_illegal_edge_attention": (
                0.0 if invalid_attention.numel() == 0 else float(invalid_attention.abs().max().item())
            ),
        }
    return {}


def select_target_indices(
    batch: Mapping[str, Any],
    target_policy: str,
    target_index: int,
    target_indices: Optional[Sequence[int]],
) -> List[int]:
    min_length = int(batch["lengths"].min().item())
    maximum = min_length - 2
    if maximum < 0:
        raise ValueError("Every audited sample must contain a valid future position.")
    if target_indices is not None:
        selected = sorted(set(int(value) for value in target_indices))
    elif target_policy == "auto_multiple":
        selected = sorted({0, maximum // 2, maximum})
    else:
        selected = [int(target_index)]
    invalid = [value for value in selected if value < 0 or value > maximum]
    if invalid:
        raise ValueError(
            "Every cutoff must leave a valid future utterance in every sample; "
            f"invalid={invalid}, allowed=0..{maximum}."
        )
    return selected


def write_csv(path: Path, rows: Iterable[Dict[str, Any]], fieldnames: List[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_report(
    path: Path,
    config_path: Path,
    family: str,
    mode: str,
    summary: Mapping[str, Any],
) -> None:
    lines = [
        "# Causal model audit report",
        "",
        f"- Config: `{config_path.as_posix()}`",
        f"- Model family: `{family}`",
        f"- Mode: `{mode}`",
        f"- Cutoffs: `{summary['target_indices']}`",
        f"- Gradient target: `{summary['gradient_target']}`",
        "- Scope: model-level causality only; upstream feature extraction remains unverified.",
        "",
        "## Global maxima",
        "",
        f"- Future zero diff: `{summary['future_zero_max_abs_diff']:.12g}`",
        f"- Future noise diff: `{summary['future_random_noise_max_abs_diff']:.12g}`",
        f"- Future shuffle diff: `{summary['future_cross_sample_shuffle_max_abs_diff']:.12g}`",
        f"- Joint future diff: `{summary['joint_future_max_abs_diff']:.12g}`",
        f"- Prefix/full diff: `{summary['prefix_full_max_abs_diff']:.12g}`",
        f"- Future edge violations: `{summary['future_adjacency_violations']}`",
        f"- Padding edge violations: `{summary['padding_edge_violations']}`",
        f"- Cross-dialogue edge violations: `{summary['cross_dialogue_edge_violations']}`",
        "",
        "## Verdict",
        "",
        f"- Strict pass at 1e-6: **{summary['strict_pass_at_1e6']}**",
        f"- Strict pass at 1e-5: **{summary['strict_pass_at_1e5']}**",
        "",
    ]
    with path.open("w", encoding="utf-8") as file:
        file.write("\n".join(lines))


def run_audit(
    config_path: Path,
    output_dir: Path,
    mode: str = "synthetic",
    device_name: str = "cpu",
    seed: int = 2026,
    batch_size: int = 2,
    sequence_length: int = 5,
    target_index: int = 2,
    class_id: int = 0,
    tolerance: float = 1.0e-6,
    real_split: str = "val",
    target_policy: str = "single",
    target_indices: Optional[Sequence[int]] = None,
    gradient_target: str = "fixed_class_logit",
    perturbation_seeds: Optional[Sequence[int]] = None,
) -> Dict[str, Any]:
    torch.manual_seed(int(seed))
    device = torch.device(device_name)
    config = load_config(config_path)
    family = model_family(config)
    model = build_model(config, family).to(device).eval()
    if mode == "synthetic":
        batch = build_synthetic_batch(config, batch_size, sequence_length, seed)
    elif mode == "real_batch":
        batch = build_real_batch(config, family, real_split, batch_size)
    else:
        raise ValueError(f"Unsupported audit mode: {mode}")
    batch = move_batch(batch, device)
    cutoffs = select_target_indices(batch, target_policy, target_index, target_indices)
    seeds = [int(value) for value in (perturbation_seeds or [seed])]
    if not seeds:
        raise ValueError("At least one perturbation seed is required.")

    with torch.no_grad():
        graph_output = forward_model(model, family, batch, return_graph=True)
    edge_counts = edge_violation_counts(graph_output, family, batch)
    specific = model_specific_diagnostics(model, family, batch, config)
    detail_rows: List[Dict[str, Any]] = []
    cutoff_rows: List[Dict[str, Any]] = []
    for cutoff in cutoffs:
        perturbation_maxima = {name: 0.0 for name in PERTURBATIONS}
        joint_maximum = 0.0
        for perturb_seed in seeds:
            maxima, details = future_perturbation_test(
                model, family, batch, cutoff, perturb_seed
            )
            for name, value in maxima.items():
                perturbation_maxima[name] = max(perturbation_maxima[name], value)
            joint_changed = perturb_future(
                batch, "random_noise", cutoff, perturb_seed + 100_000
            )
            history = valid_history_mask(batch, cutoff)
            with torch.no_grad():
                original_logits = forward_model(model, family, batch)["logits"]
                joint_logits = forward_model(model, family, joint_changed)["logits"]
            joint_maximum = max(
                joint_maximum,
                max_abs_diff(original_logits[history], joint_logits[history]),
            )
            detail_rows.extend(details)
        prefix_diff, prefix_extras = prefix_full_equivalence_test(
            model, family, batch, cutoff
        )
        gradients = future_gradient_test(
            model,
            family,
            batch,
            cutoff,
            class_id,
            gradient_target=gradient_target,
        )
        row: Dict[str, Any] = {
            "target_index": int(cutoff),
            "future_zero_max_abs_diff": perturbation_maxima["zero"],
            "future_random_noise_max_abs_diff": perturbation_maxima["random_noise"],
            "future_cross_sample_shuffle_max_abs_diff": perturbation_maxima["cross_sample_shuffle"],
            "joint_future_max_abs_diff": joint_maximum,
            "prefix_full_max_abs_diff": prefix_diff,
            "max_future_text_grad": gradients["text"],
            "max_future_audio_grad": gradients["audio"],
            "max_future_visual_grad": gradients["visual"],
        }
        row.update(prefix_extras)
        cutoff_rows.append(row)

    def maximum(key: str) -> float:
        return max(float(row.get(key, 0.0)) for row in cutoff_rows)

    numeric_keys = (
        "future_zero_max_abs_diff",
        "future_random_noise_max_abs_diff",
        "future_cross_sample_shuffle_max_abs_diff",
        "joint_future_max_abs_diff",
        "prefix_full_max_abs_diff",
        "max_future_text_grad",
        "max_future_audio_grad",
        "max_future_visual_grad",
        "low_frequency_prefix_full_diff",
        "high_frequency_prefix_full_diff",
        "context_prefix_full_diff",
        "graph_prefix_full_diff",
    )
    numeric_checks = [maximum(key) for key in numeric_keys]
    if "max_illegal_edge_attention" in specific:
        numeric_checks.append(float(specific["max_illegal_edge_attention"]))
    structural_violations = sum(edge_counts.values()) + int(
        specific.get("future_filter_path_violations", 0)
    ) + int(specific.get("future_relation_count", 0))
    strict_1e6 = all(value <= 1.0e-6 for value in numeric_checks) and structural_violations == 0
    strict_1e5 = all(value <= 1.0e-5 for value in numeric_checks) and structural_violations == 0
    strict_requested = all(value <= float(tolerance) for value in numeric_checks) and structural_violations == 0
    summary: Dict[str, Any] = {
        "model_family": family,
        "mode": mode,
        "target_index": int(target_index),
        "target_indices": ",".join(str(value) for value in cutoffs),
        "cutoff_count": len(cutoffs),
        "class_id": int(class_id),
        "gradient_target": gradient_target,
        "tolerance_requested": float(tolerance),
        "tolerance_1e6": 1.0e-6,
        "tolerance_1e5": 1.0e-5,
        "future_zero_max_abs_diff": maximum("future_zero_max_abs_diff"),
        "future_random_noise_max_abs_diff": maximum("future_random_noise_max_abs_diff"),
        "future_cross_sample_shuffle_max_abs_diff": maximum("future_cross_sample_shuffle_max_abs_diff"),
        "joint_future_max_abs_diff": maximum("joint_future_max_abs_diff"),
        "prefix_full_max_abs_diff": maximum("prefix_full_max_abs_diff"),
        "max_future_text_grad": maximum("max_future_text_grad"),
        "max_future_audio_grad": maximum("max_future_audio_grad"),
        "max_future_visual_grad": maximum("max_future_visual_grad"),
        "future_adjacency_violations": int(edge_counts["future"]),
        "padding_edge_violations": int(edge_counts["padding"]),
        "cross_dialogue_edge_violations": int(edge_counts["cross_dialogue"]),
        "strict_pass_at_requested_tolerance": bool(strict_requested),
        "strict_pass_at_1e6": bool(strict_1e6),
        "strict_pass_at_1e5": bool(strict_1e5),
        "feature_causality_status": "utterance_level_but_extractor_not_fully_verified",
    }
    for key in (
        "low_frequency_prefix_full_diff",
        "high_frequency_prefix_full_diff",
        "context_prefix_full_diff",
        "graph_prefix_full_diff",
    ):
        if any(key in row for row in cutoff_rows):
            summary[f"max_{key}"] = maximum(key)
    summary.update(specific)

    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "causal_audit_summary.csv", [summary], list(summary))
    write_csv(output_dir / "causal_audit_cutoffs.csv", cutoff_rows, sorted({key for row in cutoff_rows for key in row}))
    write_csv(
        output_dir / "future_perturbation_details.csv",
        detail_rows,
        ["seed", "target_index", "perturbation", "batch_index", "time_index", "max_abs_logit_diff"],
    )
    write_report(output_dir / "causal_audit_report.md", config_path, family, mode, summary)
    return summary


def main() -> None:
    args = parse_args()
    summary = run_audit(
        config_path=Path(args.config),
        output_dir=Path(args.output_dir),
        mode=args.mode,
        device_name=args.device,
        seed=args.seed,
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
        target_index=args.target_index,
        class_id=args.class_id,
        tolerance=args.tolerance,
        real_split=args.real_split,
        target_policy=args.target_policy,
        target_indices=parse_int_list(args.target_indices),
        gradient_target=args.gradient_target,
        perturbation_seeds=parse_int_list(args.perturbation_seeds),
    )
    print(
        "Causal audit complete:",
        f"model={summary['model_family']}",
        f"strict_pass_at_1e6={summary['strict_pass_at_1e6']}",
    )


if __name__ == "__main__":
    main()

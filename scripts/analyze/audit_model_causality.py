"""Audit whether future dialogue inputs can affect current model logits.

The audit exercises the real project model builders and performs three checks:
future perturbation invariance, prefix/full equivalence, and future gradients.
It supports small synthetic batches locally and real IEMOCAP batches where the
configured feature PKL is available.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

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
    parser.add_argument(
        "--mode", choices=("synthetic", "real_batch"), default="synthetic"
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--sequence-length", type=int, default=5)
    parser.add_argument("--target-index", type=int, default=2)
    parser.add_argument("--class-id", type=int, default=0)
    parser.add_argument("--tolerance", type=float, default=1.0e-6)
    parser.add_argument("--real-split", choices=("train", "val", "test"), default="val")
    return parser.parse_args()


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise TypeError(f"{path} must contain a YAML mapping.")
    return config


def model_family(config: Dict[str, Any]) -> str:
    name = str(config.get("model", {}).get("name", ""))
    if name == "MMGCN":
        return "MMGCN"
    if name == "MultiDAGCL":
        return "MultiDAGCL"
    raise ValueError(
        "Causality audit currently supports model.name=MMGCN or MultiDAGCL; "
        f"got {name!r}."
    )


def build_model(config: Dict[str, Any], family: str) -> torch.nn.Module:
    if family == "MMGCN":
        from scripts.train_mmgcn import build_model as project_builder
    else:
        from scripts.baselines.train_multidag_cl import build_model as project_builder
    return project_builder(config)


def build_synthetic_batch(
    config: Dict[str, Any], batch_size: int, seq_len: int, seed: int
) -> Dict[str, Any]:
    if batch_size < 2:
        raise ValueError("Synthetic cross-sample shuffle requires batch_size >= 2.")
    if seq_len < 3:
        raise ValueError("Synthetic audit requires sequence_length >= 3.")

    model_config = config["model"]
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    dims = {
        "text_features": int(model_config["text_feature_dim"]),
        "audio_features": int(model_config["audio_feature_dim"]),
        "visual_features": int(model_config["visual_feature_dim"]),
    }
    batch: Dict[str, Any] = {
        key: torch.randn(batch_size, seq_len, dim, generator=generator)
        for key, dim in dims.items()
    }
    lengths = torch.full((batch_size,), seq_len, dtype=torch.long)
    if batch_size > 1:
        lengths[-1] = seq_len - 1
    positions = torch.arange(seq_len).unsqueeze(0)
    attention_mask = positions < lengths.unsqueeze(1)
    batch["lengths"] = lengths
    batch["attention_mask"] = attention_mask
    batch["speaker_ids_int"] = (
        torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1) % 2
    )
    batch["labels"] = torch.zeros(batch_size, seq_len, dtype=torch.long)
    return batch


def build_real_batch(
    config: Dict[str, Any], family: str, split: str, batch_size: int
) -> Dict[str, Any]:
    if str(config.get("dataset", {}).get("name", "")).upper() != "IEMOCAP":
        raise ValueError("real_batch mode currently supports IEMOCAP configs only.")
    if family == "MMGCN":
        from scripts.train_mmgcn import build_dataloader

        loader = build_dataloader(config, split=split, shuffle=False)
    else:
        from scripts.baselines.train_multidag_cl import build_dataloader

        loader = build_dataloader(
            config, split=split, shuffle=False, batch_size=int(batch_size)
        )
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
        return model(
            **common,
            lengths=batch["lengths"],
            return_graph=bool(return_graph),
        )
    return model(**common, labels=None)


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
    generator = torch.Generator(device=batch["text_features"].device)
    generator.manual_seed(int(seed))
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
        expanded_mask = mask.unsqueeze(-1).expand_as(source)
        perturbed[key] = torch.where(expanded_mask, replacement, source)
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
                    "perturbation": method,
                    "batch_index": int(batch_index),
                    "time_index": int(time_index),
                    "max_abs_logit_diff": float(
                        difference[batch_index, time_index].item()
                    ),
                }
            )
    return maxima, details


def prefix_full_equivalence_test(
    model: torch.nn.Module,
    family: str,
    batch: Dict[str, Any],
    target_index: int,
) -> float:
    prefix: Dict[str, Any] = {}
    prefix_length = int(target_index) + 1
    for key, value in batch.items():
        if torch.is_tensor(value) and value.dim() >= 2 and value.shape[1] == batch["attention_mask"].shape[1]:
            prefix[key] = value[:, :prefix_length].clone()
        elif torch.is_tensor(value):
            prefix[key] = value.clone()
        else:
            prefix[key] = value
    prefix["lengths"] = batch["lengths"].clamp_max(prefix_length)

    with torch.no_grad():
        full_logits = forward_model(model, family, batch)["logits"][:, target_index]
        prefix_logits = forward_model(model, family, prefix)["logits"][:, -1]
    valid = batch["attention_mask"][:, target_index].bool()
    return max_abs_diff(full_logits[valid], prefix_logits[valid])


def future_gradient_test(
    model: torch.nn.Module,
    family: str,
    batch: Dict[str, Any],
    target_index: int,
    class_id: int,
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
    if class_id < 0 or class_id >= logits.shape[-1]:
        raise ValueError(
            f"class_id={class_id} is outside [0, {int(logits.shape[-1]) - 1}]."
        )
    valid = gradient_batch["attention_mask"][:, target_index].bool()
    scalar = logits[valid, target_index, int(class_id)].sum()
    gradients = torch.autograd.grad(scalar, inputs, allow_unused=True)
    mask = future_mask(gradient_batch, target_index).unsqueeze(-1)
    result: Dict[str, float] = {}
    for key, gradient in zip(FEATURE_KEYS, gradients):
        short_name = key.removesuffix("_features")
        if gradient is None or not bool(mask.any()):
            result[short_name] = 0.0
        else:
            selected = gradient.masked_select(mask.expand_as(gradient))
            result[short_name] = (
                0.0 if selected.numel() == 0 else float(selected.abs().max().item())
            )
    return result


def adjacency_violation_count(
    model: torch.nn.Module,
    family: str,
    batch: Dict[str, Any],
) -> int:
    with torch.no_grad():
        output = forward_model(model, family, batch, return_graph=True)
    adjacency = output["adjacency"]
    violations = 0
    if family == "MultiDAGCL":
        for batch_index, target_index, source_index in (adjacency != 0).nonzero().tolist():
            if source_index > target_index:
                violations += 1
        return violations

    positions: List[Tuple[int, int]] = []
    for batch_index, length in enumerate(batch["lengths"].detach().cpu().tolist()):
        positions.extend((batch_index, time_index) for time_index in range(int(length)))
    node_positions = positions * 3
    for target_node, source_node in (adjacency != 0).nonzero().detach().cpu().tolist():
        target_batch, target_time = node_positions[target_node]
        source_batch, source_time = node_positions[source_node]
        if source_batch != target_batch or source_time > target_time:
            violations += 1
    return violations


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
    summary: Dict[str, Any],
) -> None:
    lines = [
        "# Causal model audit report",
        "",
        f"- Config: `{config_path.as_posix()}`",
        f"- Model family: `{family}`",
        f"- Mode: `{mode}`",
        f"- Target index: `{summary['target_index']}`",
        f"- Tolerance: `{summary['tolerance_1e6']}` (also reported at `1e-5`)",
        "- Scope: model-level causality only; upstream feature extraction remains unverified.",
        "",
        "## Results",
        "",
        "| Check | Maximum / count | Pass at 1e-6 | Pass at 1e-5 |",
        "|---|---:|---:|---:|",
    ]
    checks = [
        ("future zero perturbation", summary["future_zero_max_abs_diff"]),
        ("future random-noise perturbation", summary["future_random_noise_max_abs_diff"]),
        ("future cross-sample shuffle", summary["future_cross_sample_shuffle_max_abs_diff"]),
        ("prefix/full equivalence", summary["prefix_full_max_abs_diff"]),
        ("future text gradient", summary["max_future_text_grad"]),
        ("future audio gradient", summary["max_future_audio_grad"]),
        ("future visual gradient", summary["max_future_visual_grad"]),
    ]
    for name, value in checks:
        lines.append(
            f"| {name} | {float(value):.12g} | "
            f"{float(value) <= 1e-6} | {float(value) <= 1e-5} |"
        )
    lines.append(
        f"| future adjacency violations | {summary['future_adjacency_violations']} | "
        f"{summary['future_adjacency_violations'] == 0} | "
        f"{summary['future_adjacency_violations'] == 0} |"
    )
    lines.extend(
        [
            "",
            "## Verdict",
            "",
            f"- Strict pass at 1e-6: **{summary['strict_pass_at_1e6']}**",
            f"- Strict pass at 1e-5: **{summary['strict_pass_at_1e5']}**",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


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
) -> Dict[str, Any]:
    torch.manual_seed(int(seed))
    device = torch.device(device_name)
    config = load_config(config_path)
    family = model_family(config)
    model = build_model(config, family).to(device)
    model.eval()

    if mode == "synthetic":
        batch = build_synthetic_batch(config, batch_size, sequence_length, seed)
    elif mode == "real_batch":
        batch = build_real_batch(config, family, real_split, batch_size)
    else:
        raise ValueError(f"Unsupported audit mode: {mode}")
    batch = move_batch(batch, device)

    min_length = int(batch["lengths"].min().item())
    if target_index < 0 or target_index >= min_length - 1:
        raise ValueError(
            "target_index must leave at least one valid future utterance in every "
            f"audited dialogue; got target_index={target_index}, min_length={min_length}."
        )

    perturbation_maxima, details = future_perturbation_test(
        model, family, batch, target_index, seed
    )
    prefix_diff = prefix_full_equivalence_test(model, family, batch, target_index)
    gradients = future_gradient_test(model, family, batch, target_index, class_id)
    adjacency_violations = adjacency_violation_count(model, family, batch)

    numeric_checks = [*perturbation_maxima.values(), prefix_diff, *gradients.values()]
    strict_1e6 = all(value <= 1.0e-6 for value in numeric_checks) and adjacency_violations == 0
    strict_1e5 = all(value <= 1.0e-5 for value in numeric_checks) and adjacency_violations == 0
    strict_requested = all(value <= float(tolerance) for value in numeric_checks) and adjacency_violations == 0

    summary: Dict[str, Any] = {
        "model_family": family,
        "mode": mode,
        "target_index": int(target_index),
        "class_id": int(class_id),
        "tolerance_requested": float(tolerance),
        "tolerance_1e6": 1.0e-6,
        "tolerance_1e5": 1.0e-5,
        "future_zero_max_abs_diff": perturbation_maxima["zero"],
        "future_random_noise_max_abs_diff": perturbation_maxima["random_noise"],
        "future_cross_sample_shuffle_max_abs_diff": perturbation_maxima["cross_sample_shuffle"],
        "prefix_full_max_abs_diff": prefix_diff,
        "max_future_text_grad": gradients["text"],
        "max_future_audio_grad": gradients["audio"],
        "max_future_visual_grad": gradients["visual"],
        "future_adjacency_violations": int(adjacency_violations),
        "strict_pass_at_requested_tolerance": bool(strict_requested),
        "strict_pass_at_1e6": bool(strict_1e6),
        "strict_pass_at_1e5": bool(strict_1e5),
        "feature_causality_status": "utterance_level_but_extractor_not_fully_verified",
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(
        output_dir / "causal_audit_summary.csv",
        [summary],
        list(summary),
    )
    write_csv(
        output_dir / "future_perturbation_details.csv",
        details,
        ["perturbation", "batch_index", "time_index", "max_abs_logit_diff"],
    )
    write_report(
        output_dir / "causal_audit_report.md",
        config_path,
        family,
        mode,
        summary,
    )
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
    )
    print(
        "Causal audit complete:",
        f"model={summary['model_family']}",
        f"strict_pass_at_1e6={summary['strict_pass_at_1e6']}",
    )


if __name__ == "__main__":
    main()

"""Trace GS-MCC numerical finiteness on two real training batches."""

from __future__ import annotations

import argparse
import copy
from datetime import datetime
import json
from pathlib import Path
import sys
from typing import Any
from uuid import uuid4

import pandas as pd
import torch
from torch.nn import functional as F
from torch.nn.utils.rnn import PackedSequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.baselines.original_repro import build_original_repro_model  # noqa: E402
from models.baselines.original_repro.gsmcc.model import (  # noqa: E402
    FourierGraphOperator,
    angular_similarity,
)
from scripts.baselines.original_merc_runtime import (  # noqa: E402
    build_dataloader,
    build_optimizer,
    forward_batch,
    get_device,
    load_yaml_config,
    move_batch,
    normalized_training_config,
    resolve_path,
    seed_everything,
    validate_runtime_config,
    verify_feature_sha256,
)
from utils.output_paths import configured_output_root, resolve_experiment_date  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--experiment-date", default=None)
    parser.add_argument("--diagnosis-id", default=None)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-batches", type=int, default=2)
    parser.add_argument(
        "--contrastive-mode",
        choices=("configured", "on", "off", "both"),
        default="both",
    )
    return parser.parse_args()


def _tensor_row(name: str, tensor: torch.Tensor, mode: str, batch_index: int) -> dict[str, Any]:
    value = tensor.detach()
    finite = torch.isfinite(value)
    if value.is_complex():
        statistics = value.abs().to(torch.float64)
        positive_inf = torch.isposinf(value.real) | torch.isposinf(value.imag)
        negative_inf = torch.isneginf(value.real) | torch.isneginf(value.imag)
    else:
        statistics = value.to(torch.float64)
        positive_inf = torch.isposinf(value)
        negative_inf = torch.isneginf(value)
    finite_statistics = statistics[torch.isfinite(statistics)]
    if finite_statistics.numel():
        minimum = float(finite_statistics.min().item())
        maximum = float(finite_statistics.max().item())
        mean = float(finite_statistics.mean().item())
        std = float(finite_statistics.std(unbiased=False).item())
    else:
        minimum = maximum = mean = std = float("nan")
    return {
        "mode": mode,
        "batch_index": batch_index,
        "name": name,
        "shape": json.dumps(list(value.shape)),
        "dtype": str(value.dtype),
        "min": minimum,
        "max": maximum,
        "mean": mean,
        "std": std,
        "nan_count": int(torch.isnan(value).sum().item()),
        "positive_inf_count": int(positive_inf.sum().item()),
        "negative_inf_count": int(negative_inf.sum().item()),
        "is_finite": bool(finite.all()),
    }


def _gradient_rows(
    model: torch.nn.Module,
    mode: str,
    batch_index: int,
    stage: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name, parameter in model.named_parameters():
        gradient = parameter.grad
        exists = gradient is not None
        rows.append(
            {
                "mode": mode,
                "batch_index": batch_index,
                "stage": stage,
                "parameter_name": name,
                "gradient_exists": exists,
                "gradient_norm": (
                    float(torch.linalg.vector_norm(gradient.detach()).item())
                    if exists
                    else None
                ),
                "nan_count": int(torch.isnan(gradient).sum().item()) if exists else 0,
                "inf_count": int(torch.isinf(gradient).sum().item()) if exists else 0,
                "is_finite": bool(torch.isfinite(gradient).all()) if exists else True,
            }
        )
    return rows


def _parameter_rows(
    model: torch.nn.Module,
    mode: str,
    batch_index: int,
    stage: str,
) -> list[dict[str, Any]]:
    rows = []
    for name, parameter in model.named_parameters():
        row = _tensor_row(name, parameter, mode, batch_index)
        row["stage"] = stage
        rows.append(row)
    return rows


def _record_contrastive(
    model: torch.nn.Module,
    output: dict[str, Any],
    record,
) -> dict[str, Any]:
    node_mask = output["diagnostics"]["node_mask"].bool()
    low_projected = model.low_projector(output["features"]["low_frequency"])
    high_projected = model.high_projector(output["features"]["high_frequency"])
    low = F.normalize(low_projected[node_mask], dim=-1)
    high = F.normalize(high_projected[node_mask], dim=-1)
    record("contrastive_normalized_low", low)
    record("contrastive_normalized_high", high)
    details: dict[str, Any] = {
        "enabled": bool(model.use_contrastive_loss),
        "valid_node_count": int(node_mask.sum().item()),
        "temperature": float(model.contrastive_temperature),
    }
    if not model.use_contrastive_loss or low.numel() == 0:
        return details
    similarity = torch.matmul(low, high.transpose(0, 1))
    positive_logits = torch.ones(low.shape[0], device=low.device, dtype=low.dtype)
    positive_logits = positive_logits / model.contrastive_temperature
    negative_logits = similarity / model.contrastive_temperature
    logsumexp = torch.logsumexp(
        torch.cat((positive_logits.unsqueeze(1), negative_logits), dim=1), dim=1
    )
    record("contrastive_similarity_matrix", similarity)
    record("contrastive_positive_logits", positive_logits)
    record("contrastive_negative_logits", negative_logits)
    record("contrastive_logsumexp", logsumexp)
    return details


def _install_hooks(model: torch.nn.Module, trace: dict[str, Any]):
    handles = []

    def simple_hook(name: str):
        def hook(_module, _inputs, output):
            value = output
            if isinstance(value, tuple):
                value = value[0]
            if isinstance(value, PackedSequence):
                value = value.data
            if torch.is_tensor(value):
                trace["record"](name, value)

        return hook

    for name in ("text_input", "audio_input", "visual_input", "speaker_embedding"):
        handles.append(getattr(model, name).register_forward_hook(simple_hook(name)))
    for name in ("text_encoder", "audio_encoder", "visual_encoder"):
        handles.append(getattr(model, name).register_forward_hook(simple_hook(name)))

    def fourier_hook(layer_name: str):
        def hook(module: FourierGraphOperator, inputs, output):
            features, adjacency, _node_mask = inputs
            propagated = torch.bmm(adjacency, features)
            filtered = features + propagated if module.frequency == "low" else features - propagated
            spectrum = torch.fft.rfft(filtered, dim=1, norm="ortho")
            gain = torch.complex(module.real_gain, module.imag_gain).view(1, 1, -1)
            multiplied = spectrum * gain
            inverse = torch.fft.irfft(
                multiplied, n=filtered.shape[1], dim=1, norm="ortho"
            )
            trace["record"](f"{layer_name}.frequency_input", filtered)
            trace["record"](
                f"{module.frequency}_frequency_input.{layer_name}", filtered
            )
            trace["record"](f"{layer_name}.fft_input", filtered)
            trace["record"](f"{layer_name}.fft_output_real", spectrum.real)
            trace["record"](f"{layer_name}.fft_output_imag", spectrum.imag)
            trace["record"](f"{layer_name}.fft_output_magnitude", spectrum.abs())
            trace["record"](f"{layer_name}.complex_gain_real", gain.real)
            trace["record"](f"{layer_name}.complex_gain_imag", gain.imag)
            trace["record"](f"{layer_name}.complex_gain", gain)
            trace["record"](f"{layer_name}.gained_spectrum_real", multiplied.real)
            trace["record"](f"{layer_name}.gained_spectrum_imag", multiplied.imag)
            trace["record"](f"{layer_name}.inverse_fft", inverse)
            trace["record"](f"{layer_name}.branch_output", output)

        return hook

    for index, layer in enumerate(model.low_layers):
        handles.append(layer.register_forward_hook(fourier_hook(f"low_layer_{index}")))
    for index, layer in enumerate(model.high_layers):
        handles.append(layer.register_forward_hook(fourier_hook(f"high_layer_{index}")))
    return handles


def _record_graph_similarities(model, encoded: dict[str, torch.Tensor], valid_mask, record) -> None:
    modalities = (encoded["audio"], encoded["visual"], encoded["text"])
    temporal_cosines = []
    temporal_angular = []
    cross_cosines = []
    cross_angular = []
    for batch_index in range(valid_mask.shape[0]):
        length = int(valid_mask[batch_index].sum().item())
        for features in modalities:
            features = features[batch_index]
            for target in range(length):
                start = max(0, target - model.window)
                stop = min(length, target + model.window + 1)
                indices = torch.arange(start, stop, device=features.device)
                non_self = indices != target
                if bool(non_self.any()):
                    left = features[target].expand(int(non_self.sum().item()), -1)
                    right = features[start:stop][non_self]
                    temporal_cosines.append(F.cosine_similarity(left, right, dim=-1))
                    temporal_angular.append(
                        angular_similarity(left, right, eps=model.angular_similarity_eps)
                    )
        for utterance in range(length):
            for left_index in range(3):
                for right_index in range(left_index + 1, 3):
                    left = modalities[left_index][batch_index, utterance].unsqueeze(0)
                    right = modalities[right_index][batch_index, utterance].unsqueeze(0)
                    cross_cosines.append(F.cosine_similarity(left, right, dim=-1))
                    cross_angular.append(
                        angular_similarity(left, right, eps=model.angular_similarity_eps)
                    )
    if temporal_cosines:
        record("temporal_cosine_similarity_nonself", torch.cat(temporal_cosines))
        record("temporal_angular_similarity_nonself", torch.cat(temporal_angular))
    if cross_cosines:
        record("cross_modal_cosine_similarity", torch.cat(cross_cosines))
        record("cross_modal_angular_similarity", torch.cat(cross_angular))


def diagnose_mode(
    config: dict[str, Any],
    loader,
    device: torch.device,
    mode: str,
    max_batches: int,
    tensor_rows: list[dict[str, Any]],
    gradient_rows: list[dict[str, Any]],
    parameter_before: list[dict[str, Any]],
    parameter_after: list[dict[str, Any]],
    loss_components: dict[str, Any],
    adjacency_statistics: dict[str, Any],
) -> int:
    mode_config = copy.deepcopy(config)
    if mode in {"on", "off"}:
        mode_config["model"]["use_contrastive_loss"] = mode == "on"
    seed_everything(int(mode_config["system"]["seed"]))
    model = build_original_repro_model(mode_config).to(device)
    optimizer = build_optimizer(model, mode_config)
    model.train()
    trace: dict[str, Any] = {}
    current_batch = 0

    def record(name: str, value: torch.Tensor) -> None:
        tensor_rows.append(_tensor_row(name, value, mode, current_batch))

    trace["record"] = record
    handles = _install_hooks(model, trace)
    completed = 0
    try:
        for batch_index, raw_batch in enumerate(loader, start=1):
            if batch_index > max_batches:
                break
            current_batch = batch_index
            completed += 1
            batch = move_batch(raw_batch, device)
            for name in ("text_features", "audio_features", "visual_features"):
                record(f"raw_{name}", batch[name])
            parameter_before.extend(
                _parameter_rows(model, mode, batch_index, "before_optimizer_step")
            )
            optimizer.zero_grad(set_to_none=True)
            output = forward_batch(model, batch)
            diagnostics = output["diagnostics"]
            features = output["features"]
            for name, value in features["modality_encoder_output"].items():
                record(f"context_encoder_output_{name}", value)
            _record_graph_similarities(
                model,
                features["modality_encoder_output"],
                batch["attention_mask"].bool(),
                record,
            )
            for name in (
                "raw_adjacency",
                "degree",
                "safe_degree",
                "inv_sqrt_degree",
                "adjacency",
            ):
                record(name, diagnostics[name])
            self_loop_weights = diagnostics["raw_adjacency"].diagonal(dim1=1, dim2=2)
            record("self_loop_weights", self_loop_weights[diagnostics["node_mask"].bool()])
            record("low_frequency_output", features["low_frequency"])
            record("high_frequency_output", features["high_frequency"])
            record("fused_representation", features["fusion_output"])
            record("classification_logits", output["logits"])
            record("classification_loss", output["classification_loss"])
            contrastive = _record_contrastive(model, output, record)
            for name, value in output["aux_losses"].items():
                record(name, value)
            record("total_loss", output["loss"])
            valid = diagnostics["node_mask"].bool()
            degrees = diagnostics["degree"][valid]
            adjacency_statistics[f"{mode}.batch_{batch_index}"] = {
                "adjacency_finite": bool(torch.isfinite(diagnostics["raw_adjacency"]).all()),
                "degree_finite": bool(torch.isfinite(degrees).all()),
                "degree_strictly_positive": bool((degrees > 0).all()),
                "degree_min": float(degrees.min().item()),
                "inverse_sqrt_degree_finite": bool(
                    torch.isfinite(diagnostics["inv_sqrt_degree"]).all()
                ),
                "normalized_adjacency_finite": bool(
                    torch.isfinite(diagnostics["adjacency"]).all()
                ),
                "valid_node_count": int(valid.sum().item()),
            }
            loss_components[f"{mode}.batch_{batch_index}"] = {
                "classification_loss": float(output["classification_loss"].detach().item()),
                "auxiliary_losses": {
                    name: float(value.detach().item())
                    for name, value in output["aux_losses"].items()
                },
                "raw_contrastive_loss": float(
                    diagnostics["raw_contrastive_loss"].detach().item()
                ),
                "total_loss": float(output["loss"].detach().item()),
                **contrastive,
            }
            output["loss"].backward()
            gradient_rows.extend(
                _gradient_rows(model, mode, batch_index, "after_backward")
            )
            grad_clip = float(mode_config["training"].get("grad_clip", 0.0))
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            gradient_rows.extend(
                _gradient_rows(model, mode, batch_index, "after_gradient_clipping")
            )
            optimizer.step()
            parameter_after.extend(
                _parameter_rows(model, mode, batch_index, "after_optimizer_step")
            )
    finally:
        for handle in handles:
            handle.remove()
    return completed


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.max_batches < 2:
        raise ValueError("GS-MCC diagnosis requires at least two train batches")
    config_path = resolve_path(args.config)
    config = normalized_training_config(load_yaml_config(config_path))
    validate_runtime_config(config)
    if config["model"]["name"] != "project_paper_oriented_gsmcc":
        raise ValueError("diagnostic only supports project_paper_oriented_gsmcc")
    if str(config["dataset"].get("name", "")).upper() != "IEMOCAP":
        raise ValueError("diagnostic requires a real IEMOCAP Legacy or Clean config")
    verify_feature_sha256(config)
    experiment_date = resolve_experiment_date(
        cli_date=args.experiment_date,
        config=config,
    )
    output_root = configured_output_root(
        config,
        override=None if args.output_root is None else Path(args.output_root),
    )
    output_root = resolve_path(str(output_root))
    diagnosis_id = args.diagnosis_id or (
        f"gsmcc_{Path(args.config).stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        f"_{uuid4().hex[:6]}"
    )
    output_dir = (
        output_root / experiment_date / "audits" / "gsmcc_numerics" / diagnosis_id
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    device = get_device(config, args.device)
    loader = build_dataloader(
        config,
        "train",
        shuffle=False,
        batch_size=args.batch_size,
    )
    modes = {
        "both": ["on", "off"],
        "on": ["on"],
        "off": ["off"],
        "configured": ["configured"],
    }[args.contrastive_mode]
    tensor_rows: list[dict[str, Any]] = []
    gradient_rows: list[dict[str, Any]] = []
    parameter_before: list[dict[str, Any]] = []
    parameter_after: list[dict[str, Any]] = []
    loss_components: dict[str, Any] = {}
    adjacency_statistics: dict[str, Any] = {}
    batches_by_mode = {}
    for mode in modes:
        batches_by_mode[mode] = diagnose_mode(
            config,
            loader,
            device,
            mode,
            args.max_batches,
            tensor_rows,
            gradient_rows,
            parameter_before,
            parameter_after,
            loss_components,
            adjacency_statistics,
        )
    pd.DataFrame(tensor_rows).to_csv(output_dir / "tensor_finiteness.csv", index=False)
    pd.DataFrame(gradient_rows).to_csv(output_dir / "gradient_finiteness.csv", index=False)
    pd.DataFrame(parameter_before).to_csv(
        output_dir / "parameter_finiteness_before.csv", index=False
    )
    pd.DataFrame(parameter_after).to_csv(
        output_dir / "parameter_finiteness_after.csv", index=False
    )
    write_json(output_dir / "loss_components.json", loss_components)
    write_json(output_dir / "adjacency_statistics.json", adjacency_statistics)
    first_nonfinite = next(
        (
            f"tensor:{row['mode']}.batch_{row['batch_index']}:{row['name']}"
            for row in tensor_rows
            if not row["is_finite"]
        ),
        None,
    )
    if first_nonfinite is None:
        first_nonfinite = next(
            (
                f"gradient:{row['mode']}.batch_{row['batch_index']}:{row['stage']}:{row['parameter_name']}"
                for row in gradient_rows
                if not row["is_finite"]
            ),
            None,
        )
    if first_nonfinite is None:
        first_nonfinite = next(
            (
                f"parameter:{row['mode']}.batch_{row['batch_index']}:{row['name']}"
                for row in parameter_after
                if not row["is_finite"]
            ),
            None,
        )
    summary = {
        "diagnosis_id": diagnosis_id,
        "output_dir": str(output_dir),
        "config": str(config_path),
        "dataset_split": "train",
        "test_split_used": False,
        "device": str(device),
        "modes": modes,
        "batches_by_mode": batches_by_mode,
        "numeric_status": "FINITE" if first_nonfinite is None else "NUMERICALLY_INVALID",
        "first_nonfinite_stage": first_nonfinite,
        "all_tensors_finite": all(row["is_finite"] for row in tensor_rows),
        "all_gradients_finite": all(row["is_finite"] for row in gradient_rows),
        "all_parameters_after_step_finite": all(
            row["is_finite"] for row in parameter_after
        ),
        "angular_similarity_eps": config["model"].get("angular_similarity_eps", 1e-7),
        "adjacency_detached": False,
    }
    write_json(output_dir / "diagnosis_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if first_nonfinite is not None:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

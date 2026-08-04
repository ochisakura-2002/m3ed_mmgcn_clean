"""Explicit native-AdamW construction and parameter inventory."""

from __future__ import annotations

import inspect
from typing import Any, Mapping

import torch


REQUIRED_ADAMW_VALUES = {
    "name": "AdamW",
    "learning_rate": 5.0e-4,
    "betas": [0.9, 0.999],
    "eps": 1.0e-6,
    "weight_decay": 0.0,
    "amsgrad": False,
    "maximize": False,
    "foreach": False,
    "capturable": False,
    "differentiable": False,
    "fused": None,
    "parameter_grouping": "all_parameters_single_group",
    "bias_correction": "native AdamW standard bias correction",
}


def validate_optimizer_config(config: Mapping[str, Any]) -> None:
    if not isinstance(config, Mapping):
        raise TypeError("runtime.optimizer must be a mapping")
    unknown = sorted(set(config) - set(REQUIRED_ADAMW_VALUES))
    if unknown:
        raise ValueError(f"unknown runtime.optimizer fields: {unknown}")
    missing = sorted(set(REQUIRED_ADAMW_VALUES) - set(config))
    if missing:
        raise ValueError(f"missing explicit runtime.optimizer fields: {missing}")
    for name, expected in REQUIRED_ADAMW_VALUES.items():
        configured = config[name]
        if name == "betas":
            configured = list(configured)
        if configured != expected:
            raise ValueError(
                f"runtime.optimizer.{name} must be {expected!r}, got {configured!r}"
            )


def build_optimizer(
    model: torch.nn.Module,
    config: Mapping[str, Any],
) -> torch.optim.AdamW:
    """Build one AdamW group containing every trainable parameter exactly once."""

    validate_optimizer_config(config)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise ValueError("model exposes no trainable parameters")
    if len({id(parameter) for parameter in parameters}) != len(parameters):
        raise ValueError("model trainable parameter iterator contains duplicates")

    signature = inspect.signature(torch.optim.AdamW)
    optional_values = {
        "amsgrad": config["amsgrad"],
        "maximize": config["maximize"],
        "foreach": config["foreach"],
        "capturable": config["capturable"],
        "differentiable": config["differentiable"],
        "fused": config["fused"],
    }
    kwargs = {
        "lr": float(config["learning_rate"]),
        "betas": tuple(float(value) for value in config["betas"]),
        "eps": float(config["eps"]),
        "weight_decay": float(config["weight_decay"]),
    }
    kwargs.update(
        {
            name: value
            for name, value in optional_values.items()
            if name in signature.parameters
        }
    )
    optimizer = torch.optim.AdamW(parameters, **kwargs)
    audit_optimizer_parameters(model, optimizer)
    return optimizer


def audit_optimizer_parameters(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> dict[str, Any]:
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimized = [
        parameter
        for group in optimizer.param_groups
        for parameter in group["params"]
    ]
    trainable_ids = [id(parameter) for parameter in trainable]
    optimized_ids = [id(parameter) for parameter in optimized]
    missing = sorted(set(trainable_ids) - set(optimized_ids))
    unexpected = sorted(set(optimized_ids) - set(trainable_ids))
    duplicate_count = len(optimized_ids) - len(set(optimized_ids))
    if missing or unexpected or duplicate_count:
        raise ValueError(
            "optimizer parameter coverage mismatch: "
            f"missing={len(missing)}, unexpected={len(unexpected)}, "
            f"duplicates={duplicate_count}"
        )
    return {
        "parameter_group_count": len(optimizer.param_groups),
        "trainable_parameter_tensor_count": len(trainable),
        "optimized_parameter_tensor_count": len(optimized),
        "trainable_parameter_count": sum(parameter.numel() for parameter in trainable),
        "missing_parameter_count": 0,
        "duplicate_parameter_count": 0,
    }


__all__ = [
    "REQUIRED_ADAMW_VALUES",
    "audit_optimizer_parameters",
    "build_optimizer",
    "validate_optimizer_config",
]

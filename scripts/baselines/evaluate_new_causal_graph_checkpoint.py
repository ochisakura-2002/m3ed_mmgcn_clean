"""Evaluate a saved Task 2 causal graph checkpoint on one explicit split."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.baselines.causal_baseline_registry import normalize_new_causal_model_name  # noqa: E402
from scripts.baselines.new_causal_graph_runtime import (  # noqa: E402
    build_dataloader,
    evaluate_model,
    get_device,
    load_checkpoint,
    rebuild_model_from_checkpoint,
    resolve_path,
    save_evaluation_outputs,
    validate_runtime_config,
    verify_feature_sha256,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Path to best_model.pt.")
    parser.add_argument("--split", required=True, choices=("val", "test"))
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def default_output_dir(checkpoint_path: Path, split: str) -> Path:
    run_dir = checkpoint_path.parent.parent
    return run_dir / "logs" / "evaluations" / f"{split}_best_model"


def evaluate_checkpoint(
    checkpoint_path: Path,
    split: str,
    output_dir: Optional[Path] = None,
    device_override: Optional[str] = None,
) -> Dict[str, Any]:
    resolved_checkpoint = resolve_path(str(checkpoint_path))
    if not resolved_checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {resolved_checkpoint}")
    bootstrap_device = get_device({"system": {"device": device_override or "cpu"}})
    checkpoint = load_checkpoint(resolved_checkpoint, bootstrap_device)
    config = checkpoint["config"]
    validate_runtime_config(config)
    device = get_device(config, device_override)
    if device != bootstrap_device:
        checkpoint = load_checkpoint(resolved_checkpoint, device)

    configured_sha = config.get("dataset", {}).get("feature_sha256")
    checkpoint_sha = checkpoint.get("feature_sha256")
    if checkpoint_sha != configured_sha:
        raise ValueError(
            "Checkpoint feature_sha256 does not match its saved config: "
            f"checkpoint={checkpoint_sha!r}, config={configured_sha!r}."
        )
    verify_feature_sha256(config)
    model = rebuild_model_from_checkpoint(checkpoint, device)
    loader = build_dataloader(config, split, shuffle=False)
    result = evaluate_model(model, config, loader, device)
    destination = (
        default_output_dir(resolved_checkpoint, split)
        if output_dir is None
        else resolve_path(str(output_dir))
    )
    save_evaluation_outputs(
        destination,
        config,
        split,
        result,
        resolved_checkpoint,
        int(checkpoint["epoch"]),
    )
    metrics = result["metrics"]
    print(f"model_name={normalize_new_causal_model_name(config['model']['name'])}")
    print(f"checkpoint_epoch={int(checkpoint['epoch'])}")
    print(f"split={split}")
    print(
        f"loss={result['loss']:.6f} accuracy={metrics['acc']:.6f} "
        f"weighted_f1={metrics['weighted_f1']:.6f} "
        f"macro_f1={metrics['macro_f1']:.6f} uar={metrics['uar']:.6f}"
    )
    print(f"output_dir={destination}")
    return {
        "config": config,
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "output_dir": destination,
        "result": result,
    }


def main() -> None:
    args = parse_args()
    evaluate_checkpoint(
        Path(args.checkpoint),
        args.split,
        None if args.output_dir is None else Path(args.output_dir),
        args.device,
    )


if __name__ == "__main__":
    main()

"""Evaluate a strict original-MERC checkpoint on one explicit split."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional

import torch


PROJECT_ROOT = next(
    candidate
    for candidate in Path(__file__).resolve().parents
    if (candidate / "AGENTS.md").is_file() and (candidate / "scripts").is_dir()
)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.runtime.paper_aligned import (  # noqa: E402
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
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def evaluate_checkpoint(
    checkpoint_path: Path,
    split: str = "test",
    output_dir: Optional[Path] = None,
    device_override: Optional[str] = None,
) -> dict[str, Any]:
    resolved = resolve_path(str(checkpoint_path))
    bootstrap = torch.device("cpu")
    checkpoint = load_checkpoint(resolved, bootstrap)
    config = checkpoint["config"]
    validate_runtime_config(config)
    verified_sha = verify_feature_sha256(config)  # Must precede model construction.
    if checkpoint.get("feature_sha256") != config["dataset"].get("feature_sha256"):
        raise ValueError("checkpoint feature SHA does not match its embedded config")
    if checkpoint.get("test_split_used_for_selection") is not False:
        raise ValueError("checkpoint does not prove validation-only selection")
    device = get_device(config, device_override)
    checkpoint = load_checkpoint(resolved, device)
    model = rebuild_model_from_checkpoint(checkpoint, device)
    loader = build_dataloader(config, split, shuffle=False)
    result = evaluate_model(model, config, loader, device)
    destination = (
        resolve_path(str(output_dir))
        if output_dir is not None
        else resolved.parent.parent / "logs" / "evaluations" / f"{split}_manual"
    )
    save_evaluation_outputs(
        destination,
        config,
        split,
        result,
        resolved,
        int(checkpoint["epoch"]),
    )
    summary = {
        "checkpoint": str(resolved),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "split": split,
        "metrics": result["metrics"],
        "verified_feature_sha256": verified_sha,
        "test_split_used_for_selection": False,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


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

"""Thin CLI for Stage-B3 check, smoke, training, and locked evaluation modes."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = next(
    parent
    for parent in Path(__file__).resolve().parents
    if (parent / "AGENTS.md").is_file() and (parent / "models").is_dir()
)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.runtime.multidag_cl_paper_reimplementation import (  # noqa: E402
    LocalAssetUnavailable,
    OfficialAssetsUnavailable,
    run_runtime,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        required=True,
        choices=["check", "synthetic-smoke", "real-batch-smoke", "train", "evaluate"],
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--experiment-date", default=None)
    parser.add_argument("--experiment-group", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--checkpoint", default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> dict:
    args = parse_args(argv)
    checkpoint = args.checkpoint if args.mode == "evaluate" else args.resume
    try:
        result = run_runtime(
            Path(args.config),
            mode=args.mode,
            output_root_override=(
                None if args.output_root is None else Path(args.output_root)
            ),
            experiment_date=args.experiment_date,
            experiment_group=args.experiment_group,
            device_override=args.device,
            resume_checkpoint=None if checkpoint is None else Path(checkpoint),
        )
    except OfficialAssetsUnavailable as error:
        if args.mode != "check":
            raise
        result = {
            "status": "BLOCKED_OFFICIAL_ASSETS",
            "mode": args.mode,
            "optimizer_steps": 0,
            "optimizer_step_count": 0,
            "gradient_clip_count": 0,
            "real_test_batches_accessed": 0,
            "error": str(error),
        }
    except LocalAssetUnavailable as error:
        if args.mode != "real-batch-smoke":
            raise
        result = {
            "status": "BLOCKED_LOCAL_ASSET",
            "mode": args.mode,
            "optimizer_steps": 0,
            "optimizer_step_count": 0,
            "gradient_clip_count": 0,
            "real_test_batches_accessed": 0,
            "error": str(error),
        }
    print("MULTIDAG_STAGE_B3_RESULT=" + json.dumps(result, ensure_ascii=False), flush=True)
    return result


if __name__ == "__main__":
    main()

"""One config-selected entry for three independent causal TA GraphCore models."""

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import yaml

from scripts.runtime.multidag_cl_graph_core_variants.formal import (
    run_formal, synthetic_dry_run, validate_formal_config,
)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--synthetic-dry-run", action="store_true")
    modes.add_argument("--formal-train-dev", action="store_true")
    parser.add_argument("--asset-path")
    parser.add_argument("--model-manifest-path")
    parser.add_argument("--provenance-manifest-path")
    parser.add_argument("--output-root", help="output base; canonical date/group/runs layout is retained")
    parser.add_argument("--experiment-date")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--smoke-one-step", action="store_true")
    args = parser.parse_args(argv)
    path_names = ("asset_path", "model_manifest_path", "provenance_manifest_path")
    if args.formal_train_dev:
        for name in path_names:
            if not getattr(args, name):
                parser.error("formal mode requires --" + name.replace("_", "-"))
        result = run_formal(
            formal_train_dev=True, config_path=args.config,
            **{name: getattr(args, name) for name in path_names},
            output_root=args.output_root, experiment_date=args.experiment_date,
            device=args.device, one_step_smoke=args.smoke_one_step)
    else:
        if (any(getattr(args, name) for name in path_names)
                or args.output_root or args.experiment_date or args.device != "cpu" or args.smoke_one_step):
            parser.error("real-data/output/device options require --formal-train-dev")
        graph = validate_formal_config(
            yaml.safe_load(Path(args.config).read_text(encoding="utf-8")), allow_disabled=True)
        torch.set_num_threads(1)
        result = synthetic_dry_run(graph)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

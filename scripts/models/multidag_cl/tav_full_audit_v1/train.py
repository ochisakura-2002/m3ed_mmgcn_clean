"""Remote formal TAV training or a small in-memory CPU smoke."""

import argparse
import json
from pathlib import Path
import sys
import uuid

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import yaml

from scripts.runtime.multidag_cl_tav_full_audit_v1.formal import execute, run_formal, validate_training_config
from scripts.runtime.multidag_cl_tav_full_audit_v1.runtime import synthetic_runtime


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--check-config", action="store_true")
    modes.add_argument("--synthetic-smoke", action="store_true")
    modes.add_argument("--formal-train-dev", action="store_true")
    for name in ("asset-path", "model-manifest-path", "provenance-manifest-path", "output-root", "experiment-date", "resume"):
        parser.add_argument("--" + name)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args(argv)
    raw = validate_training_config(yaml.safe_load(Path(args.config).read_text(encoding="utf-8")))
    paths = (args.asset_path, args.model_manifest_path, args.provenance_manifest_path)
    if args.formal_train_dev:
        if not all(paths):
            parser.error("formal mode requires all three explicit asset/manifest paths")
        result = run_formal(formal_train_dev=True, config_path=args.config, asset_path=args.asset_path,
                            model_manifest_path=args.model_manifest_path, provenance_manifest_path=args.provenance_manifest_path,
                            output_root=args.output_root, experiment_date=args.experiment_date, device=args.device, resume=args.resume)
    else:
        if any(paths) or args.output_root or args.experiment_date or args.resume:
            parser.error("data/output/resume options require formal mode")
        if args.check_config:
            result = {"status": "PASS", "experiment_id": raw["experiment_id"], "test_evaluation_count": 0}
        else:
            torch.set_num_threads(1)
            directory = ROOT / "tmp" / ("tav_smoke_" + uuid.uuid4().hex)
            runtime = synthetic_runtime(raw)
            result = execute(runtime, raw, directory, synthetic_epochs=2)
            result = {key: result[key] for key in ("status", "experiment_id", "parameter_count", "best_epoch",
                                                    "best_dev_metrics", "test_loader_built", "test_evaluation_count")}
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

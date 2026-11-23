"""Frozen experiment arms. The upstream F0 model config is used verbatim."""

from pathlib import Path

import yaml

from scripts.runtime.multidag_cl_fusion.spec import FINAL_PROTOCOL, ROOT, make_final_config
from scripts.runtime.multidag_cl_ta_v0.formal import json_identity
from scripts.runtime.multidag_cl_tav_full_audit_v1.confirm import frozen_model_source, source_unchanged

GROUP = "post_fusion_parallel_fastkill_001"
SEED = 100
CTRL = "CTRL"
B1 = "B1_CLASS_ANCHOR"
A1 = "A1_PREFIX_WES"
ARMS = (CTRL, B1, A1)
CONFIG_DIR = Path("configs/multidag_cl/post_fusion_parallel_fastkill_001")
F0_SOURCE = "85590fbf97d0c5aad0c5547142a98fc20e342dc4"
TAU = 0.1
ANCHOR_WEIGHT = 0.1
WES_K, WES_B = 1.0, 0.4
WEIGHT_STRENGTH = 0.10
UNIFORM_EPOCH = 8
HOLD = {
    "test_policy": "HOLD", "test_loader_built": 0, "test_labels_inspected": 0,
    "test_evaluation_count": 0, "test_checkpoint_selection_count": 0, "test_used_for_selection": False,
}


def baseline():
    return make_final_config("F0", SEED)


def require_frozen_baseline():
    if not frozen_model_source() or not source_unchanged(
            F0_SOURCE, "--", "models/multidag_cl/fusion", "scripts/runtime/multidag_cl_fusion"):
        raise ValueError("frozen CT-MultiDAG F0 source lineage cannot be verified")


def arm_config(arm):
    if arm not in ARMS:
        raise ValueError("unknown Fast-Kill arm")
    return {
        "protocol": GROUP.upper(), "arm": arm, "seed": SEED,
        "baseline": {"protocol": FINAL_PROTOCOL, "variant": "F0"},
        "class_anchor": {"enabled": arm == B1, "temperature": TAU, "lambda_anchor": ANCHOR_WEIGHT},
        "prefix_wes": {"enabled": arm == A1, "k": WES_K, "b": WES_B, "strength": WEIGHT_STRENGTH,
                       "uniform_from_epoch": UNIFORM_EPOCH},
        "test_policy": "HOLD",
    }


def validate_arm(raw):
    if not isinstance(raw, dict):
        raise ValueError("Fast-Kill config must be a mapping")
    if all(isinstance(raw.get(key), dict) and raw[key].get("enabled") is True
           for key in ("class_anchor", "prefix_wes")):
        raise ValueError("UNAUTHORIZED_COMBINATION: CLASS_ANCHOR + PREFIX_WES")
    expected = arm_config(raw.get("arm"))
    if json_identity(raw) != json_identity(expected):
        raise ValueError("Fast-Kill arm differs from its frozen seed100 configuration")
    return expected


def load_configs():
    configs = {}
    for arm in ARMS:
        path = CONFIG_DIR / f"{arm}_seed100.yaml"
        config = validate_arm(yaml.safe_load((ROOT / path).read_text(encoding="utf-8")))
        if config["arm"] != arm:
            raise ValueError(f"wrong arm in {path}")
        configs[arm] = config
    return configs

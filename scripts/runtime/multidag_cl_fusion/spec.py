"""Frozen seed-100 screening protocol derived from the Track TAV baseline."""

from copy import deepcopy
from pathlib import Path

import yaml

from datasets.iemocap.tav_full_audit_dataset import COUNTS
from models.multidag_cl.fusion.registry import DIRECT, FULL_RANK, LOW_RANK, ROUTER
from scripts.runtime.multidag_cl_ta_v0.formal import json_identity
from scripts.runtime.multidag_cl_tav_full_audit_v1.spec import ROOT, make_config


SCHEMA = "multidag_tav_fusion_screen_v1"
GROUP = "fusion_seed100_screen_20261121"
CONFIG_DIR = Path("configs/multidag_cl/fusion_candidates/seed100_screen")
ARMS = {
    "F0": {"type": DIRECT},
    "Hybrid": {"type": LOW_RANK, "rank": 16, "residual_scale": 0.1},
    "Fusion-A": {"type": FULL_RANK, "residual_scale": 0.1},
    "Fusion-C": {"type": ROUTER, "router_hidden_dim": 64, "expert_hidden_dim": 256, "residual_scale": 0.1},
}


def make_screen_config(arm):
    raw = make_config("track")
    raw.update(schema=SCHEMA, experiment_id=f"tav_track_{arm.lower().replace('-', '_')}_seed100",
               fusion=deepcopy(ARMS[arm]))
    raw["dataset"]["population"] = {split: deepcopy(COUNTS[split]) for split in ("train", "dev")}
    raw["output"]["experiment_group"] = GROUP
    return raw


def validate_screen_config(raw):
    if not isinstance(raw, dict):
        raise ValueError("CT-MultiDAG fusion screening config must be a mapping")
    for arm, fusion in ARMS.items():
        if raw.get("fusion") == fusion and json_identity(raw) == json_identity(make_screen_config(arm)):
            return deepcopy(raw)
    raise ValueError("frozen CT-MultiDAG fusion screening config mismatch")


def load_screen_configs():
    configs = {}
    for arm in ARMS:
        path = CONFIG_DIR / (make_screen_config(arm)["experiment_id"] + ".yaml")
        raw = validate_screen_config(yaml.safe_load((ROOT / path).read_text(encoding="utf-8")))
        if raw != make_screen_config(arm):
            raise ValueError(f"wrong fusion arm in {path}")
        configs[arm] = (path, raw)
    return configs

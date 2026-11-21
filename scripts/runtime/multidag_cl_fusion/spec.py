"""Frozen seed-100 screening protocol derived from the Track TAV baseline."""

from copy import deepcopy
from pathlib import Path

import yaml

from datasets.iemocap.tav_full_audit_dataset import COUNTS
from models.multidag_cl.fusion.matching import EDGE_LABELS
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
TOPOLOGY_PROTOCOL = "FUSION_TOPOLOGY_RETRAIN_SCREEN_V1"
TOPOLOGY_SCHEMA = "multidag_tav_fusion_topology_retrain_v1"
TOPOLOGY_GROUP = "fusion_topology_retrain_screen_v1"
TOPOLOGY_CONFIG_DIR = CONFIG_DIR.parent / "topology_retrain_screen_v1"
TOPOLOGY_EDGES = {
    "F0": (),
    "H4_FULL_HYBRID": EDGE_LABELS,
    "H3_ASYMMETRIC": ("T<-A", "A<-T", "V<-T"),
    "H2_AUDIO_TEXT": ("T<-A", "A<-T"),
}
EDGE_PARAMETER_COUNT = 13_200
F0_PARAMETER_COUNT = 1_638_306


def make_screen_config(arm):
    raw = make_config("track")
    raw.update(schema=SCHEMA, experiment_id=f"tav_track_{arm.lower().replace('-', '_')}_seed100",
               fusion=deepcopy(ARMS[arm]))
    raw["dataset"]["population"] = {split: deepcopy(COUNTS[split]) for split in ("train", "dev")}
    raw["output"]["experiment_group"] = GROUP
    return raw


def make_topology_config(arm):
    edges = TOPOLOGY_EDGES[arm]
    raw = make_screen_config("Hybrid" if edges else "F0")
    raw.update(schema=TOPOLOGY_SCHEMA, experiment_id=f"tav_track_topology_{arm.lower()}_seed100")
    raw["output"]["experiment_group"] = TOPOLOGY_GROUP
    if edges:
        raw["fusion"]["active_edges"] = list(edges)
    return raw


def is_topology(raw):
    return raw.get("schema") == TOPOLOGY_SCHEMA


def validate_screen_config(raw):
    if not isinstance(raw, dict):
        raise ValueError("CT-MultiDAG fusion screening config must be a mapping")
    arms, make = (TOPOLOGY_EDGES, make_topology_config) if is_topology(raw) else (ARMS, make_screen_config)
    identity = json_identity(raw)
    for arm in arms:
        expected = make(arm)
        if raw.get("fusion") == expected["fusion"] and identity == json_identity(expected):
            return deepcopy(raw)
    raise ValueError("frozen CT-MultiDAG fusion screening config mismatch")


def load_screen_configs(*, topology=False):
    arms, make, directory = ((TOPOLOGY_EDGES, make_topology_config, TOPOLOGY_CONFIG_DIR) if topology
                             else (ARMS, make_screen_config, CONFIG_DIR))
    configs = {}
    for arm in arms:
        expected = make(arm)
        path = directory / (expected["experiment_id"] + ".yaml")
        raw = validate_screen_config(yaml.safe_load((ROOT / path).read_text(encoding="utf-8")))
        if raw != expected:
            raise ValueError(f"wrong fusion arm in {path}")
        configs[arm] = (path, raw)
    return configs

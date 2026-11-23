"""Frozen seed-100 screening protocol derived from the Track TAV baseline."""

from copy import deepcopy
from pathlib import Path

import yaml

from datasets.iemocap.tav_full_audit_dataset import COUNTS
from models.multidag_cl.fusion.matching import EDGE_LABELS
from models.multidag_cl.fusion.registry import DIRECT, FULL_RANK, LOW_RANK, ROUTER
from scripts.runtime.multidag_cl_ta_v0.formal import json_identity
from scripts.runtime.multidag_cl_tav_full_audit_v1.spec import CONFIRM_SEEDS, ROOT, confirm_config, make_config


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
FINAL_PROTOCOL = "FUSION_FINAL_PAIRED_5SEED_CONFIRMATION"
FINAL_SCHEMA = "multidag_tav_fusion_final_paired_5seed_v1"
FINAL_GROUP = "fusion_final_paired_5seed_confirmation"
FINAL_CONFIG = CONFIG_DIR.parent / FINAL_GROUP / "confirmation.yaml"
FINAL_ARMS = ("F0", "H4_FULL_HYBRID", "H2_AUDIO_TEXT")
BASELINE_CONFIGURATION = "track_tav_cloff"


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
    return raw.get("schema") in (TOPOLOGY_SCHEMA, FINAL_SCHEMA)


def make_final_config(arm, seed):
    if arm not in FINAL_ARMS or type(seed) is not int or seed not in CONFIRM_SEEDS:
        raise ValueError("configuration is outside the frozen Fusion final confirmation")
    raw = make_topology_config(arm)
    raw.update(schema=FINAL_SCHEMA, experiment_id=f"tav_track_fusion_final_{arm.lower()}_s{seed}")
    raw["training"]["seed"] = seed
    raw["output"]["experiment_group"] = FINAL_GROUP
    return raw


def validate_screen_config(raw):
    if not isinstance(raw, dict):
        raise ValueError("CT-MultiDAG fusion screening config must be a mapping")
    if raw.get("schema") == FINAL_SCHEMA:
        training = raw.get("training")
        seed = training.get("seed") if isinstance(training, dict) else None
        configs = (make_final_config(arm, seed) for arm in FINAL_ARMS)
    else:
        arms, make = (TOPOLOGY_EDGES, make_topology_config) if is_topology(raw) else (ARMS, make_screen_config)
        configs = (make(arm) for arm in arms)
    identity = json_identity(raw)
    for expected in configs:
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


def final_seed_source():
    from scripts.runtime.multidag_cl_tav_full_audit_v1.confirm import MATRIX_PATH, load_design

    sources = (f"{MATRIX_PATH.as_posix()} [{BASELINE_CONFIGURATION}] and "
               "scripts/runtime/multidag_cl_tav_full_audit_v1/spec.py [CONFIRM_SEEDS]")
    try:
        rows, configs, _ = load_design()
        baseline = [row for row in rows if row["configuration"] == BASELINE_CONFIGURATION]
        seeds = tuple(int(row["seed"]) for row in baseline)
        if (len(set(seeds)) != 5 or seeds != CONFIRM_SEEDS
                or any(configs[row["experiment_id"]] != confirm_config(make_config("track"), int(row["seed"]))
                       for row in baseline)):
            raise ValueError("frozen Track/TAV/CL_OFF/all-past identity or five-seed definition disagrees")
    except (OSError, ValueError, KeyError, TypeError, yaml.YAMLError) as error:
        raise ValueError(f"FAIL-STOP: historical five-seed list cannot be verified. Sources: {sources}. {error}") from error
    return {"matrix": MATRIX_PATH.as_posix(), "configuration": BASELINE_CONFIGURATION,
            "matrix_rows_sha256": json_identity(baseline), "seeds": list(seeds)}


def load_final_configs():
    from scripts.runtime.multidag_cl_tav_full_audit_v1.confirm import MATRIX_PATH

    seed_source = final_seed_source()
    raw = yaml.safe_load((ROOT / FINAL_CONFIG).read_text(encoding="utf-8"))
    expected = {"protocol": FINAL_PROTOCOL, "experiment_group": FINAL_GROUP, "variants": list(FINAL_ARMS),
                "seed_source": {"matrix": MATRIX_PATH.as_posix(), "configuration": BASELINE_CONFIGURATION},
                "test_policy": "HOLD"}
    if raw != expected:
        raise ValueError("frozen final Fusion confirmation YAML mismatch")
    configs = {}
    for arm in FINAL_ARMS:
        topology = make_topology_config(arm)
        path = TOPOLOGY_CONFIG_DIR / f"{topology['experiment_id']}.yaml"
        if validate_screen_config(yaml.safe_load((ROOT / path).read_text(encoding="utf-8"))) != topology:
            raise ValueError(f"wrong source topology for final Fusion arm {arm}: {path}")
        for seed in seed_source["seeds"]:
            configs[(seed, arm)] = (path, make_final_config(arm, seed))
    return seed_source, configs

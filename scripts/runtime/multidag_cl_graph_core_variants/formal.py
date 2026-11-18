"""Independent variant identities around the existing frozen Train/Dev fit loop."""

from copy import deepcopy
from datetime import datetime
import json
from pathlib import Path

import torch
from torch.utils.data import Dataset
import yaml

from models.multidag_cl.graph_core_variants import GraphCoreConfig, GraphCoreTA, VARIANTS
from models.multidag_cl.graph_core_variants.config import PARENT_BASELINE, baseline_config
from scripts.models.multidag_cl.ta_v0.train import require_development_split
from scripts.runtime.multidag_cl_ta_v0.formal import (
    fit, git_identity, json_identity, load_train_dev, validate_formal_config as validate_parent,
    write_json_atomic,
)
from scripts.runtime.multidag_cl_ta_v0.training import PROTOCOL, TrainDevRuntime, seed_everything
from utils.output_paths import resolve_experiment_date, resolve_output_paths
from .training import GraphCoreRuntime

PARENT_CONFIG = ("configs/multidag_cl/ta_v0/iemocap/causal_context/"
                 "asymmetric_merc_v1/formal_train_dev_t1024_ab1.yaml")


def config_path(variant):
    if variant not in VARIANTS:
        raise ValueError("unknown independent variant")
    return (f"configs/multidag_cl/{variant}_t1024_v0/iemocap/causal_context/"
            "asymmetric_merc_v1/formal_train_dev.yaml")


def validate_formal_config(raw, *, allow_disabled=False):
    expected = {"model", "dataset", "runtime", "training", "checkpoint", "output",
                "identity", "graph_core"}
    if not isinstance(raw, dict) or set(raw) != expected:
        raise ValueError("unexpected/missing variant formal config sections")
    graph = GraphCoreConfig.from_mapping(raw["graph_core"])
    if not graph.enabled and not allow_disabled:
        raise ValueError("formal variants require enabled=true; OFF is a synthetic ablation")
    identity = dict(VARIANTS[graph.variant], parent_baseline=PARENT_BASELINE)
    if raw["identity"] != identity:
        raise ValueError("variant model identity mismatch")
    group = f"multidag_{graph.variant}_t1024_formal"
    if raw["output"] != {"root": "outputs", "experiment_date": None, "experiment_group": group}:
        raise ValueError("variant output configuration is frozen")
    parent = {key: deepcopy(raw[key]) for key in expected - {"identity", "graph_core"}}
    parent["output"]["experiment_group"] = "ta_v0_formal"
    if validate_parent(parent) != baseline_config():
        raise ValueError("variant requires the frozen T1024 baseline")
    return graph


class SyntheticDialogues(Dataset):
    """Tiny in-memory fixture with Train/Dev only; no Test universe is created."""

    def __init__(self, split):
        require_development_split(split)
        self.split = split
        generator = torch.Generator().manual_seed(100 if split == "train" else 101)
        self.items = []
        for i, length in enumerate((5, 2)):
            did = f"synthetic_{split}_{i}"
            self.items.append({
                "dialogue_id": did, "utterance_ids": tuple(f"{did}_u{j}" for j in range(length)),
                "text_features": torch.randn(length, 1024, generator=generator),
                "audio_features": torch.randn(length, 1582, generator=generator),
                "speaker_ids_int": torch.arange(length) % 2,
                "labels": torch.arange(length) % 6, "split": split,
            })

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


def synthetic_runtime(graph):
    seed_everything("cpu")
    datasets = {split: SyntheticDialogues(split) for split in ("train", "dev")}
    runtime_type = GraphCoreRuntime if graph.enabled else TrainDevRuntime
    return runtime_type(GraphCoreTA(graph), datasets.__getitem__, synthetic_only=True,
                        data_identity={"kind": "synthetic_train_dev_only_v1"})


def synthetic_dry_run(graph):
    runtime = synthetic_runtime(graph)
    batch = next(iter(runtime.loader("train")))
    output, loss = runtime.forward_batch(batch, split="train")
    loss.backward()
    if any(p.grad is None or not torch.isfinite(p.grad).all() for p in runtime.model.parameters()):
        raise ValueError("missing/nonfinite synthetic gradients")
    dev = runtime.evaluate()
    identity = VARIANTS[graph.variant] if graph.enabled else {
        "model_name": PARENT_BASELINE, "model_lineage": PARENT_BASELINE}
    return {"status": "PASS", **identity, "graph_core": graph.resolved(), "synthetic_only": True,
            "h_ta_shape": list(output.h_ta.shape), "train_loss": float(loss.detach()),
            "dev_weighted_f1": dev["metrics"]["weighted_f1"], "optimizer_steps": 0,
            "test_evaluation_count": 0, "test_loader_built": 0}


def build_manifest(runtime, *, git, run_id, run_dir, paths):
    graph = runtime.model.graph_config
    count = sum(p.numel() for p in runtime.model.parameters())
    return {
        **VARIANTS[graph.variant], **git, "parent_baseline": PARENT_BASELINE,
        "graph_core_variant": graph.variant, "graph_core": graph.resolved(),
        "text_dim": 1024, "audio_dim": 1582, "hidden_dim": 300, "graph_layers": 4,
        "parameter_count": count, "model_parameter_count": count,
        "graph_topology_changed": False, "future_edges_added": False,
        "model_config": runtime.model.config.checkpoint_config(),
        "model_architecture_identity": json_identity(runtime.model.get_extra_state()),
        "data_identity": deepcopy(runtime.data_identity), "data_paths": deepcopy(paths),
        "training_config": deepcopy(PROTOCOL), "seed": 100,
        "device": str(runtime.device), "synthetic_only": runtime.synthetic_only,
        "formal_train_dev": runtime.formal_train_dev,
        "test_evaluation_count": 0, "test_loader_built": 0,
        "checkpoint_metric": "dev_weighted_f1",
        "checkpoint_tiebreak_rules": ["weighted_f1_desc", "dev_loss_asc", "epoch_asc"],
        "run_id": run_id, "run_dir": Path(run_dir).as_posix(),
    }


def execute(runtime, raw, *, paths, git, output_root=None, experiment_date=None,
            synthetic_epochs=None, one_step_smoke=False):
    graph = validate_formal_config(raw)
    if graph != runtime.model.graph_config:
        raise ValueError("runtime/config GraphCore identity mismatch")
    resolved = deepcopy(raw)
    resolved["graph_core"] = graph.resolved()
    resolved["dataset"].update(asset_path=paths.get("asset"),
                               model_manifest_path=paths.get("model_manifest"),
                               provenance_manifest_path=paths.get("provenance_manifest"))
    resolved["runtime"].update(device=str(runtime.device), synthetic_only=runtime.synthetic_only,
                               formal_train_dev=runtime.formal_train_dev,
                               one_step_smoke=one_step_smoke)
    stamp = datetime.now().astimezone().strftime("%Y%m%dT%H%M%S%f%z")
    digest = json_identity({"config": resolved, "data": runtime.data_identity})[:12]
    run_id = f"multidag_{graph.variant}_t1024_seed100_{stamp}_{digest}"
    date = resolve_experiment_date(experiment_date, raw)
    # '+' in a timezone is not accepted by the project's identifier validator.
    run_id = run_id.replace("+", "p")
    output = resolve_output_paths(output_base=output_root or raw["output"]["root"],
                                  experiment_date=date,
                                  experiment_group=raw["output"]["experiment_group"], run_id=run_id)
    resolved["output"].update(root=output.output_base.as_posix(), experiment_date=date,
                              run_id=run_id, run_dir=output.run_root.as_posix())
    manifest = build_manifest(runtime, git=git, run_id=run_id, run_dir=output.run_root, paths=paths)
    # Fail collisions before loaders or any training step. fit's atomic mkdir is
    # still authoritative against races between this check and directory creation.
    if output.run_root.exists():
        raise FileExistsError(output.run_root)
    ids = [uid for batch in runtime.loader("dev") for dialogue in batch["utterance_ids"] for uid in dialogue]
    result = fit(runtime, output.run_root, resolved_config=resolved, manifest=manifest,
                 expected_dev_ids=ids, synthetic_epochs=synthetic_epochs, one_step_smoke=one_step_smoke)
    # Reuse the parent's ID-joined Dev export, then annotate the new representation
    # identity without changing inference, rows, metrics or artifact ordering.
    metadata_path = output.run_root / "representations" / "validation_best_h_ta.metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata.update(representation_name=f"{VARIANTS[graph.variant]['model_lineage']}_preclassifier",
                    **VARIANTS[graph.variant], graph_core=graph.resolved())
    write_json_atomic(metadata_path, metadata)
    result["dev_export"] = metadata
    write_json_atomic(output.run_root / "manifests" / "run_manifest.json", result)
    return result


def run_formal(*, formal_train_dev=False, config_path, asset_path, model_manifest_path,
               provenance_manifest_path, output_root=None, experiment_date=None,
               device="cpu", one_step_smoke=False):
    if formal_train_dev is not True:
        raise ValueError("--formal-train-dev is required before real data access")
    raw = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    graph = validate_formal_config(raw)
    device = torch.device(device)
    if device.type not in ("cpu", "cuda") or device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("explicit device is unavailable")
    datasets, data_identity, paths = load_train_dev(
        asset_path, model_manifest_path, provenance_manifest_path, config=baseline_config())
    seed_everything(device)
    git = git_identity()
    runtime = GraphCoreRuntime(GraphCoreTA(graph), datasets.__getitem__, device=device,
                               formal_train_dev=True, data_identity=data_identity, git_head=git["git_head"])
    return execute(runtime, raw, paths=paths, git=git, output_root=output_root,
                   experiment_date=experiment_date, one_step_smoke=one_step_smoke)

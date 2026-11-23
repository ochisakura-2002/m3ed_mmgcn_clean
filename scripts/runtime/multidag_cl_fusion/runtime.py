"""Pair shared initialization and observed batch order, and collect Dev diagnostics."""

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import random
from typing import Optional

import numpy as np
import torch

from models.multidag_cl.fusion.base import MODALITIES
from models.multidag_cl.fusion.matching import EDGES
from models.multidag_cl.fusion.model import FusionTAV
from models.multidag_cl.fusion.registry import DIRECT
from models.multidag_cl.ta_v0 import LABEL_VOCAB
from scripts.runtime.multidag_cl_ta_v0.formal import json_identity
from scripts.runtime.multidag_cl_ta_v0.training import cpu_snapshot, seed_everything
from scripts.runtime.multidag_cl_tav_full_audit_v1.runtime import AuditRuntime
from .spec import ARMS, FINAL_SCHEMA, is_topology, make_screen_config, make_topology_config, validate_screen_config


@dataclass(frozen=True)
class F0Initialization:
    seed: int
    shared: dict
    rng: torch.Tensor
    canonical_edges: Optional[dict] = None


def shared_state(model):
    return {name: value for name, value in model.state_dict().items()
            if isinstance(value, torch.Tensor) and not name.startswith("fusion.")}


def state_digest(state):
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        digest.update(json_identity([name, list(value.shape), str(value.dtype)]).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def training_rng_identity(device):
    name, words, position, has_gauss, cached = np.random.get_state()
    identity = {"python": json_identity(random.getstate()),
                "numpy": json_identity([name, words.tolist(), position, has_gauss, cached]),
                "torch_cpu": state_digest({"rng": torch.get_rng_state()})}
    if torch.device(device).type == "cuda":
        identity["torch_cuda"] = state_digest(dict(enumerate(torch.cuda.get_rng_state_all())))
    return identity


def seeded_candidate(raw, seed):
    # Construct CPU modules without advancing the F0 CPU or CUDA training RNG.
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        return FusionTAV(raw)


def edge_state(model):
    return {EDGES[name][2]: matcher.state_dict() for name, matcher in model.fusion.matchers.items()}


def paired_model(raw, *, device, reference=None):
    raw = validate_screen_config(raw)
    seed = raw["training"]["seed"]
    topology = is_topology(raw)
    direct = raw["fusion"]["type"] == DIRECT
    if reference is not None and (reference.seed != seed or topology != (reference.canonical_edges is not None)):
        raise ValueError("CT-MultiDAG initialization reference uses a different seed or protocol")
    seed_everything(device, seed)
    f0 = FusionTAV(raw) if direct else None
    if reference is None:
        if f0 is None:
            f0_config = make_topology_config("F0") if topology else make_screen_config("F0")
            f0 = FusionTAV(f0_config)
        shared, rng = cpu_snapshot(shared_state(f0)), torch.get_rng_state()
        canonical = edge_state(seeded_candidate(make_screen_config("Hybrid"), seed + 1)) if topology else None
        reference = F0Initialization(seed, shared, rng, canonical)
    else:
        torch.set_rng_state(reference.rng)

    if direct:
        model, fusion_seed = f0, None
    else:
        index = 1 if topology else next(i for i, options in enumerate(ARMS.values()) if options == raw["fusion"])
        fusion_seed = seed + index
        model = seeded_candidate(raw, fusion_seed)
        state = model.state_dict()
        if shared_state(model).keys() != reference.shared.keys():
            raise ValueError("CT-MultiDAG shared state identity mismatch")
        state.update(reference.shared)
        model.load_state_dict(state)

    if not torch.equal(torch.get_rng_state(), reference.rng):
        raise ValueError("CT-MultiDAG training RNG differs from F0")
    actual = shared_state(model)
    difference = max(float((value - actual[name]).abs().max()) for name, value in reference.shared.items())
    digest = state_digest(actual)
    if difference != 0 or digest != state_digest(reference.shared):
        raise ValueError("CT-MultiDAG shared initialization differs from F0")
    initialization = {"shared_parameter_max_abs_diff": difference, "shared_initial_state_sha256": digest,
                      "fusion_initialization_seed": fusion_seed}
    if topology:
        differences, edge_hashes = [], {}
        if not direct:
            for name, matcher in model.fusion.matchers.items():
                label = EDGES[name][2]
                expected = reference.canonical_edges[label]
                matcher.load_state_dict(expected, strict=True)
                actual_edge = matcher.state_dict()
                differences.extend(float((value - actual_edge[key]).abs().max()) for key, value in expected.items())
                edge_hashes[label] = state_digest(actual_edge)
                if edge_hashes[label] != state_digest(expected):
                    raise ValueError(f"CT-MultiDAG initial parameters differ for {label}")
        edge_difference = max(differences, default=0.0)
        if edge_difference != 0:
            raise ValueError("CT-MultiDAG common edge initialization differs from canonical H4")
        initialization.update(common_edge_init_max_abs_diff=edge_difference, edge_initial_state_sha256=edge_hashes,
                              training_cpu_rng_sha256=state_digest({"rng": reference.rng}))
    return model, reference, initialization


def aggregate_dev(traces, labels, *, by_class):
    result = {}
    for _, _, edge in EDGES.values():
        if edge not in traces:
            continue
        samples = torch.cat(traces[edge])
        row = {"active_count": len(samples)}
        if len(samples):
            gate, message, receiver, ratio = samples[:, :4].T
            quantiles = gate.quantile(torch.tensor([0.05, 0.25, 0.50, 0.75, 0.95])).tolist()
            row.update(gate_mean=float(gate.mean()), gate_std=float(gate.std(unbiased=False)),
                       message_l2_mean=float(message.mean()), receiver_l2_mean=float(receiver.mean()),
                       message_receiver_ratio_mean=float(ratio.mean()))
            row.update(zip(("gate_q05", "gate_q25", "gate_q50", "gate_q75", "gate_q95"), quantiles))
            if samples.shape[1] == 6:
                applied_edge, applied_receiver = samples[:, 4:].T
                row.update(applied_edge_residual_receiver_ratio_mean=float(applied_edge.mean()),
                           applied_receiver_residual_receiver_ratio_mean=float(applied_receiver.mean()))
        result[edge] = row
    if "activity" in traces:
        ratio = torch.cat(traces["activity"])
        median = float(ratio.quantile(0.5)) if len(ratio) else None
        result["activity"] = {"active_count": len(ratio), "residual_receiver_ratio_median": median,
                              "FUSION_ACTIVITY_WARNING": "NEAR_IDENTITY" if median is not None and median < 1e-3 else None}
    if "router" in traces:
        probabilities, residual = torch.cat(traces["router"]).split(len(MODALITIES), dim=-1)
        entropy = -(probabilities * probabilities.clamp_min(1e-8).log()).sum(-1)
        router = {"router_entropy_mean": float(entropy.mean())}
        for index, name in enumerate(MODALITIES):
            p = probabilities[:, index]
            router.update({f"p_{name}_mean": float(p.mean()), f"p_{name}_std": float(p.std(unbiased=False)),
                           f"{name}_near_zero_rate": float((p < 1e-3).float().mean()),
                           f"expert_{name}_residual_norm": float(residual[:, index].mean())})
        result["router"] = router
        if by_class:
            targets = torch.cat(labels)
            result["router_by_class"] = {}
            for index, label in enumerate(LABEL_VOCAB):
                rows = probabilities[targets == index]
                result["router_by_class"][label] = {
                    f"p_{name}": float(rows[:, i].mean()) if len(rows) else None
                    for i, name in enumerate(MODALITIES)
                }
    return result


class FusionAuditRuntime(AuditRuntime):
    def __init__(self, model, dataset_factory, *, initialization, f0_pairing=None, **options):
        super().__init__(model, dataset_factory, **options)
        self.initialization = initialization
        self.f0_pairing = f0_pairing
        self._populations = {}
        self._dev_trace = None
        populations = {}
        for split in ("train", "dev"):
            dataset = self.dataset(split)
            examples = [(item["dialogue_id"], tuple(item["utterance_ids"]))
                        for item in (dataset[index] for index in range(len(dataset)))]
            self._populations[split] = sorted(examples)
            counts = {"dialogues": len(examples), "utterances": sum(len(ids) for _, ids in examples)}
            if not self.synthetic_only and counts != model.audit_config["dataset"]["population"][split]:
                raise ValueError(f"canonical {split} population mismatch")
            populations[split] = {"sha256": json_identity(examples), **counts}
        if f0_pairing is not None and populations != f0_pairing["population"]:
            raise ValueError("CT-MultiDAG Train/Dev population differs from F0")
        self.pairing = {"population": populations, "train": {}, "dev": None}

    def loader(self, split, *, epoch=1):
        loader = super().loader(split, epoch=epoch)
        return self._audit_batches(loader, split, epoch)

    def _audit_batches(self, loader, split, epoch):
        epoch_key = str(epoch)
        previous = self.pairing["dev"] if split == "dev" else self.pairing["train"].get(epoch_key)
        if self.f0_pairing is not None:
            previous = self.f0_pairing["dev"] if split == "dev" else self.f0_pairing["train"][epoch_key]
        audit_rng = split == "train" and self.model.audit_config["schema"] == FINAL_SCHEMA
        if audit_rng:
            rng_start = json_identity(training_rng_identity(self.device))
            if previous is not None and rng_start != previous["training_rng_start_sha256"]:
                raise ValueError(f"Train epoch {epoch} starts with different training RNG from F0")
        batch_hashes, observed = [], []
        for index, batch in enumerate(loader):
            identities = list(zip(batch["dialogue_ids"], batch["utterance_ids"]))
            digest = json_identity(identities)
            if previous is not None:
                expected = previous["batch_sha256"]
                if index >= len(expected) or digest != expected[index]:
                    raise ValueError(f"{split} epoch {epoch} batch {index + 1} differs from paired order")
            batch_hashes.append(digest)
            observed.extend(identities)
            yield batch
        if sorted(observed) != self._populations[split]:
            raise ValueError(f"{split} batch population differs from the canonical view")
        actual = {"dialogue_order_sha256": json_identity(observed), "batch_sha256": batch_hashes,
                  "batch_sequence_sha256": json_identity(batch_hashes)}
        if audit_rng:
            actual.update(training_rng_start_sha256=rng_start,
                          training_rng_end_sha256=json_identity(training_rng_identity(self.device)))
        if previous is not None and actual != previous:
            raise ValueError(f"{split} epoch {epoch} batch digest mismatch")
        if split == "dev":
            self.pairing["dev"] = actual
        else:
            self.pairing["train"][epoch_key] = actual

    def restore_pairing(self, manifest):
        paired = manifest["batch_pairing"]
        if (manifest["shared_initialization"] != self.initialization
                or paired["population"] != self.pairing["population"]
                or set(paired["train"]) != {str(epoch) for epoch in range(1, self.epoch + 1)}
                or self.epoch and paired["dev"] is None):
            raise ValueError("resume fusion initialization or batch pairing identity mismatch")
        self.pairing = deepcopy(paired)

    def forward_batch(self, batch, *, split, active_modalities=None):
        output, loss = super().forward_batch(batch, split=split, active_modalities=active_modalities)
        if self._dev_trace is not None:
            traces, labels = self._dev_trace
            for key, value in output.fusion_diagnostics.items():
                traces.setdefault(key, []).append((value[output.mask] if key == "router" else value).detach().cpu())
            if "router" in output.fusion_diagnostics:
                labels.append(batch["labels"][output.mask.cpu()])
        return output, loss

    def evaluate(self, *args, include_predictions=False, **kwargs):
        self._dev_trace = ({}, [])
        try:
            if include_predictions and is_topology(self.model.audit_config):
                kwargs["prediction_details"] = True
            report = super().evaluate(*args, include_predictions=include_predictions, **kwargs)
            report["fusion_mechanism"] = aggregate_dev(*self._dev_trace, by_class=include_predictions)
            return report
        finally:
            self._dev_trace = None


def paired_runtime(raw, datasets, *, device, reference=None, f0_pairing=None, **options):
    model, reference, initialization = paired_model(raw, device=device, reference=reference)
    runtime = FusionAuditRuntime(model, datasets.__getitem__, device=device, initialization=initialization,
                                 f0_pairing=f0_pairing, **options)
    if raw["schema"] == FINAL_SCHEMA:
        initialization["training_rng_identity"] = training_rng_identity(device)
    return runtime, reference

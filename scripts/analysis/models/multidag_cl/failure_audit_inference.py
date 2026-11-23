"""Train/Dev-only frozen F0 inputs and counterfactual inference."""

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from datasets.iemocap.ta_v0_dataset import UID
from datasets.iemocap.tav_full_audit_dataset import COUNTS, DIMS, collate, require_split
from models.multidag_cl.fusion.base import DirectConcatFusion
from models.multidag_cl.fusion.model import FusionTAV
from models.multidag_cl.graph_core_variants.track import TrackCore
from scripts.runtime.multidag_cl_fusion.spec import F0_PARAMETER_COUNT, make_final_config
from scripts.runtime.multidag_cl_paper_reimplementation.curriculum import CurriculumRuntime
from models.multidag_cl.paper_reimplementation.config import CurriculumPartitionProfile, CurriculumScheduleProfile
from scripts.runtime.multidag_cl_ta_v0.formal import json_identity, sha256_file
from scripts.runtime.multidag_cl_ta_v0.training import reload_model_checkpoint
from scripts.runtime.multidag_cl_tav_full_audit_v1.confirm import checked_completion, checked_epochs, test_hold
from scripts.runtime.multidag_cl_tav_full_audit_v1.runtime import MODEL_INPUTS
from scripts.runtime.multidag_cl_tav_full_audit_v1.spec import CONFIRM_SEEDS, MODALITIES
from .export_fusion_topology_bundle import check_hold, prediction_rows, read_json


REPEATS = (1701, 1702, 1703)
RECENT = 4
HISTORY = ("FULL_HISTORY", "CURRENT_ONLY", "SAME_SPEAKER_ONLY", "OTHER_SPEAKER_ONLY",
           "RECENT_ONLY", "DISTANT_ONLY")
MASKS = {"FULL_TAV": "TAV", "T_ONLY": "T", "A_ONLY": "A", "V_ONLY": "V",
         "DROP_T": "AV", "DROP_A": "TV", "DROP_V": "TA"}


@dataclass(frozen=True)
class Source:
    config: dict
    manifest: dict
    logits: np.ndarray
    population: tuple
    checkpoint: Path

    @property
    def identity(self):
        return {"seed": self.manifest["seed"], "epoch": self.manifest["best_epoch"],
                "git_head": self.manifest["git_head"], "checkpoint_sha256": self.manifest["best_checkpoint_sha256"]}


def sources_from_runs(runs, data_identity=None):
    sources = {}
    for seed in CONFIRM_SEEDS:
        config = make_final_config("F0", seed)
        run = Path(runs) / config["experiment_id"]
        manifest = read_json(run / "manifests/run_manifest.json")
        evidence = read_json(run / "reports/validation_best_dev_evidence.json")
        resolved = yaml.safe_load((run / "resolved_config.yaml").read_text(encoding="utf-8"))
        checked_completion(run, manifest)
        check_hold(manifest)
        check_hold(evidence)
        selected = checked_epochs(run, manifest["best_epoch"])
        checkpoint = run / "checkpoints/best_model.pt"
        if data_identity is None:
            data_identity = manifest["data_identity"]
        if (manifest["seed"] != seed or manifest["config_sha256"] != json_identity(config)
                or manifest["parameter_count"] != F0_PARAMETER_COUNT or manifest["active_edges"] != []
                or manifest["run_id"] != config["experiment_id"] or manifest["data_identity"] != data_identity
                or resolved["audit_config"] != config or resolved["data_identity"] != data_identity
                or evidence["split"] != "dev" or evidence["export_mode"] != "final_selected_checkpoint_reload"
                or evidence["optimizer_steps_performed"] != 0 or evidence["metrics"] != manifest["best_dev_metrics"]
                or evidence["checkpoint_sha256"] != manifest["best_checkpoint_sha256"]
                or any(not np.isclose(float(selected[f"dev_{metric}"]), value, rtol=1e-6, atol=1e-8)
                       for metric, value in evidence["metrics"].items()
                       if metric in ("weighted_f1", "macro_f1", "uar", "accuracy", "loss"))):
            raise ValueError(f"F0 Dev evidence identity mismatch for seed {seed}")
        pairing = manifest["batch_pairing"]
        dev = pairing["population"]["dev"]
        if any(dev[key] != value for key, value in COUNTS["dev"].items()):
            raise ValueError("F0 Dev population counts changed")
        _, population = prediction_rows(evidence, {}, dev, pairing["dev"]["dialogue_order_sha256"])
        if sources and tuple(population) != next(iter(sources.values())).population:
            raise ValueError("F0 Dev populations differ across seeds")
        if any(UID.fullmatch(uid) is None or UID.fullmatch(uid)[1] != dialogue for uid, dialogue, _ in population):
            raise ValueError("F0 Dev canonical utterance IDs changed")
        sources[seed] = Source(config, manifest, np.asarray([row["logits"] for row in evidence["predictions"]]),
                               tuple(population), checkpoint)
    return sources


def frozen_model(source, membership_sha, device):
    expected = source.identity
    if sha256_file(source.checkpoint) != expected["checkpoint_sha256"]:
        raise ValueError("frozen best_model.pt SHA256 mismatch")
    model = FusionTAV(source.config)

    def validate(payload):
        test_hold(payload)
        identity = {"schema": "tav_full_audit_checkpoint_v1", "audit_config": source.config,
                    "data_identity": source.manifest["data_identity"], "training_config": source.config["training"],
                    "curriculum_membership_sha256": membership_sha, "seed": expected["seed"],
                    "epoch": expected["epoch"], "git_head": expected["git_head"], "epoch_complete": True}
        if any(payload.get(key) != value for key, value in identity.items()):
            raise ValueError("structured frozen checkpoint identity mismatch")

    reload_model_checkpoint(model, source.checkpoint, validate=validate)
    if sha256_file(source.checkpoint) != expected["checkpoint_sha256"]:
        raise ValueError("checkpoint changed while loading")
    if not isinstance(model.graph_core, TrackCore) or model.paper_encoder.text_encoder.bidirectional:
        raise ValueError("history audit requires causal text encoding and TrackCore")
    return model.to(device).requires_grad_(False).eval()


def membership(train):
    return CurriculumRuntime.from_training_dataset(
        train, split="train", bucket_count=7,
        partition_profile=CurriculumPartitionProfile.BALANCED_STABLE_CONTIGUOUS,
        schedule_profile=CurriculumScheduleProfile.OFFICIAL_ONE_BUCKET_PER_EPOCH, enabled=False,
    ).manifest.membership_sha256


def modality_available(model):
    return (type(model) is FusionTAV and isinstance(model.fusion, DirectConcatFusion)
            and set(MASKS.values()) == set(MODALITIES) and model.audit_config["active_modalities"] == "TAV")


def modality_logits(model, inputs, original, *, dev):
    # Frozen F0 production forward zeros the encoded blocks via active_modalities.
    return {"FULL_TAV": original, **{name: model(**inputs, active_modalities=mask).logits_ta
                                    for name, mask in list(MASKS.items())[1:] if dev or name.endswith("_ONLY")}}


def keep_history(speakers, mask, target, condition):
    indices = torch.arange(mask.shape[1], device=mask.device)
    before = mask & (indices < target[:, None])
    same = speakers == speakers.gather(1, target[:, None])
    recent = indices >= target[:, None] - RECENT
    selection = {"CURRENT_ONLY": torch.zeros_like(before), "SAME_SPEAKER_ONLY": same,
                 "OTHER_SPEAKER_ONLY": ~same, "RECENT_ONLY": recent, "DISTANT_ONLY": ~recent}
    return (before & selection[condition]) | (indices == target[:, None])


def counterfactual_history(model, inputs, original, *, chunk_size):
    h0 = original.layer_states[0]
    locations = original.mask.nonzero()
    results = defaultdict(list)
    for chunk in locations.split(chunk_size):
        b, t = chunk.unbind(1)
        states, adjacency, mask = h0[b], original.adjacency[b], original.mask[b]
        speakers = inputs["speaker_ids_int"][b]
        relations = model.relation_builder.build(speakers, mask)
        row = torch.arange(len(b), device=b.device)
        for condition in HISTORY[2:]:
            keep = keep_history(speakers, mask, t, condition)
            edges = adjacency & keep[:, :, None] & keep[:, None, :]
            layers = model.graph_core(states, edges, relations, mask)
            results[condition].append(model.classifier(torch.cat(
                [value[row, t] for value in (states, *layers)], dim=-1)).cpu().numpy())
    return {**isolated_history(model, inputs, original),
            **{name: np.concatenate(chunks) for name, chunks in results.items()}}


def isolated_history(model, inputs, original):
    mask = original.mask
    h0 = original.layer_states[0]
    relations = model.relation_builder.build(inputs["speaker_ids_int"], mask)
    isolated = model.graph_core(h0, torch.zeros_like(original.adjacency), relations, mask)
    return {"FULL_HISTORY": original.logits_ta[mask].cpu().numpy(),
            "CURRENT_ONLY": model.classifier(torch.cat((h0, *isolated), -1))[mask].cpu().numpy()}


def donor_plans(train, dev):
    """Choose joint T/A/V donors once, independently of model seed and outcomes."""
    banks = defaultdict(list)
    for sample in train:
        if sample["split"] != "train":
            raise ValueError("context donors must come from Train")
        speakers = sample["speaker_ids_int"].tolist()
        for j, (uid, label) in enumerate(zip(sample["utterance_ids"], sample["labels"].tolist())):
            banks[speakers[j] == speakers[0]].append((sample, j, uid, label))
    plans = {}
    for repeat in REPEATS:
        rng = np.random.default_rng(repeat)
        for sample in dev:
            if sample["split"] != "dev":
                raise ValueError("context targets must come from Dev")
            speakers = sample["speaker_ids_int"].tolist()
            for condition in ("ECCS", "EICS"):
                donors = []
                for j, uid in enumerate(sample["utterance_ids"][:-1]):
                    label = int(sample["labels"][j])
                    candidates = [(donor, k) for donor, k, donor_uid, donor_label in banks[speakers[j] == speakers[0]]
                                  if donor["dialogue_id"] != sample["dialogue_id"] and donor_uid != uid
                                  and (donor_label == label) == (condition == "ECCS")]
                    if not candidates:
                        raise ValueError(f"no speaker-role-matched {condition} donor for {uid}")
                    donors.append(candidates[int(rng.integers(len(candidates)))])
                plans[repeat, sample["dialogue_id"], condition] = donors
    return plans


def substituted_prefixes(sample, donors, targets):
    """Replace strictly past raw features while keeping recipient speakers and target bytes."""
    return [{**sample, "utterance_ids": sample["utterance_ids"][:t + 1],
             "speaker_ids_int": sample["speaker_ids_int"][:t + 1], "labels": sample["labels"][:t + 1],
             **{key: torch.stack([donor[key][index] for donor, index in donors[:t]] + [sample[key][t]])
                for key in DIMS}} for t in targets]


@torch.inference_mode()
def context_logits(model, dev, plans, *, device, chunk_size):
    results = defaultdict(list)
    for repeat in REPEATS:
        for condition in ("ECCS", "EICS"):
            for sample in dev:
                donors = plans[repeat, sample["dialogue_id"], condition]
                for start in range(0, len(sample["utterance_ids"]), chunk_size):
                    targets = range(start, min(start + chunk_size, len(sample["utterance_ids"])))
                    batch = collate(substituted_prefixes(sample, donors, targets))
                    output = model(**{key: batch[key].to(device) for key in MODEL_INPUTS})
                    results[condition, repeat].append(output.logits_ta[
                        torch.arange(len(targets), device=device), batch["lengths"].to(device) - 1].cpu().numpy())
    return {key: np.concatenate(chunks) for key, chunks in results.items()}


@torch.inference_mode()
def evaluate(model, samples, split, *, device, chunk_size):
    require_split(split)
    collected = defaultdict(list)
    for batch in DataLoader(samples, batch_size=16, shuffle=False, collate_fn=collate):
        if batch["split"] != split:
            raise ValueError("invalid frozen forward split")
        inputs = {name: batch[name].to(device) for name in MODEL_INPUTS}
        full = model(**inputs)
        history = (counterfactual_history(model, inputs, full, chunk_size=chunk_size) if split == "dev"
                   else isolated_history(model, inputs, full))
        for name, values in history.items():
            collected[name].append(values)
        if modality_available(model):
            for name, values in modality_logits(model, inputs, full.logits_ta, dev=split == "dev").items():
                collected[name].append(values[full.mask].cpu().numpy())
    return {name: np.concatenate(chunks) for name, chunks in collected.items()}

"""Authenticate F0 and collect only certified Train/Dev predictions."""

from dataclasses import dataclass
import pickle
import re

import numpy as np
import torch
from torch.utils.data import DataLoader

from datasets.iemocap.tav_full_audit_dataset import COUNTS, SCHEMA, TAVAuditDataset, collate, identity_sha, require_split
from models.multidag_cl.fusion.base import DirectConcatFusion
from models.multidag_cl.ta_v0 import LABEL_VOCAB
from scripts.runtime.multidag_cl_fusion.spec import make_final_config
from scripts.runtime.multidag_cl_ta_v0.formal import json_identity, sha256_file
from scripts.runtime.multidag_cl_ta_v0.training import seed_everything
from scripts.runtime.multidag_cl_tav_full_audit_v1.runtime import MODEL_INPUTS
from scripts.runtime.multidag_cl_tav_full_audit_v1.spec import read_csv

from .export_fusion_topology_bundle import read_json
from .failure_audit_inference import HISTORY, frozen_model, membership, sources_from_runs
from .failure_audit_metrics import GROUPS, metadata, quartiles, selection_groups
from .pairwise_preflight_metrics import scores


SEED = 100
F0_IDENTITY = {"seed": SEED, "epoch": 21, "git_head": "85590fbf97d0c5aad0c5547142a98fc20e342dc4",
               "checkpoint_sha256": "997e7db663e3a3c2b8cb9ea34957e8fdbda909593ae7ad8360bc14ef1e3022ac"}
F0_DEV_CORRECT = 693
F0_DEV_WF1_PERCENT = 69.9627
Q4_SUPPORT = 245
SPLITS = ("train", "dev")
TRAIN_DEV_COUNTS = {split: COUNTS[split] for split in SPLITS}
CLASS_ORDER = ("hap", "sad", "neu", "ang", "exc", "fru")
FEATURE_DIMENSIONS = {"text": 1024, "audio": 1582, "visual": 342}
PROJECTION_SCHEMA = "ct_multidag_train_dev_projection_v1"
PROJECTION_MANIFEST = "train_dev_provenance.json"
ASSET = "train_dev_tav.pkl"
MODEL_MANIFEST = "model_input_manifest.csv"
HOLD_COUNTERS = ("test_loader_built", "test_labels_inspected", "test_forward_count", "test_evaluation_count",
                 "test_checkpoint_selection_count")
HOLD = {"test_policy": "HOLD", **dict.fromkeys(HOLD_COUNTERS, 0)}
H_GROUPS = ("U_ONLY", "H_ONLY", "U_INTERSECT_H", "NEITHER")


class TestHoldViolation(ValueError):
    """An input reports activity on the held-out split."""

    def __init__(self, message, *, field=None, value=None):
        super().__init__(message)
        self.evidence = {} if field is None else {field: value if type(value) is int else None}


def validate_hold(record, *, legacy=False):
    for key in HOLD_COUNTERS:
        value = record.get(key, 0) if legacy else record.get(key)
        if type(value) is not int or value != 0:
            raise TestHoldViolation(f"invalid Test HOLD counter: {key}", field=key, value=value)
    if (record.get("test_policy", "HOLD" if legacy else None) != "HOLD"
            or record.get("test_used_for_selection", False) is not False
            or record.get("TEST_TOUCHED", "NO") != "NO"):
        raise TestHoldViolation("Test HOLD policy or selection violation")


def read_f0(frozen_runs):
    config = make_final_config("F0", SEED)
    run = frozen_runs / config["experiment_id"]
    validate_hold(read_json(run / "manifests/run_manifest.json"), legacy=True)
    validate_hold(read_json(run / "reports/validation_best_dev_evidence.json"), legacy=True)
    source = sources_from_runs(frozen_runs, seeds=(SEED,))[SEED]
    manifest = source.manifest
    architecture = {key: config[key] for key in ("architecture_lineage", "frontend_source", "graph_core", "fusion")}
    if (source.identity != F0_IDENTITY or source.config != config
            or manifest["architecture_identity"] != json_identity(architecture)
            or manifest["context"] != "past_all_causal" or manifest["graph_core"] != {"enabled": True, "variant": "track"}
            or manifest["fusion"] != config["fusion"] or manifest["active_modalities"] != "TAV"
            or manifest["feature_dimensions"] != FEATURE_DIMENSIONS or tuple(LABEL_VOCAB) != CLASS_ORDER):
        raise ValueError("historical seed100 F0 run/architecture identity mismatch")
    if sha256_file(source.checkpoint) != F0_IDENTITY["checkpoint_sha256"]:
        raise ValueError("checkpoint bytes do not match the historical seed100 F0 SHA-256")
    base = manifest["best_dev_metrics"]
    if (len(source.population) != TRAIN_DEV_COUNTS["dev"]["utterances"] or base["valid_target_count"] != len(source.population)
            or base["accuracy"] != F0_DEV_CORRECT / len(source.population)
            or round(100 * base["weighted_f1"], 4) != F0_DEV_WF1_PERCENT):
        raise ValueError("historical seed100 F0 Dev metrics differ from epoch-21 reference")
    return source


def load_projected_train_dev(directory, source):
    """Verify the Train/Dev-only package before invoking any dataset factory."""
    projection = read_json(directory / PROJECTION_MANIFEST)
    validate_hold(projection)
    data_identity = source.manifest["data_identity"]
    original = data_identity["sha256"]
    hashes = projection["sha256"]
    if (projection["schema"] != PROJECTION_SCHEMA or projection["splits"] != list(SPLITS)
            or projection["source_data_identity"] != data_identity or projection["split_counts"] != TRAIN_DEV_COUNTS
            or data_identity["schema"] != SCHEMA or data_identity["split_counts"] != COUNTS
            or data_identity["feature_dimensions"] != FEATURE_DIMENSIONS
            or set(hashes) != {ASSET, MODEL_MANIFEST}
            or any(not isinstance(digest, str) or not re.fullmatch(r"[a-f0-9]{64}", digest) for digest in hashes.values())
            or set(hashes.values()) & set(original.values())):
        raise ValueError("expected a certified, distinct Train/Dev-only projection of the historical F0 inputs")
    if any(sha256_file(directory / name) != digest for name, digest in hashes.items()):
        raise ValueError("Train/Dev-only input SHA-256 mismatch")
    rows = read_csv(directory / MODEL_MANIFEST)
    if (any(row["split"] not in SPLITS for row in rows)
            or identity_sha(rows) != projection["row_order_identity_sha256"]):
        raise ValueError("Train/Dev manifest scope or row-order identity mismatch")
    # Only the separately certified split-only pickle is deserialized. No path
    # is resolved from the original run's paths or from the projection metadata.
    with (directory / ASSET).open("rb") as stream:
        asset = pickle.load(stream)
    if set(asset["splits"]) != set(SPLITS):
        raise ValueError("Train/Dev projection must contain exactly two split keys")
    if set(asset["dialogues"]) != set(asset["splits"]["train"]) | set(asset["splits"]["dev"]):
        raise ValueError("Train/Dev projection contains unmapped dialogue objects")
    # The canonical validator expects three keys even for a split-only view.
    # Supply a structural empty slot in memory. No Test row or dataset exists.
    view = {**asset, "splits": {**asset["splits"], "test": []}}
    counts = {**TRAIN_DEV_COUNTS, "test": {"dialogues": 0, "utterances": 0}}
    datasets = {split: TAVAuditDataset(view, rows, split=split, expected_counts=counts) for split in SPLITS}
    if any(sha256_file(directory / name) != digest for name, digest in hashes.items()):
        raise ValueError("Train/Dev-only input changed during validation")
    fields = ("schema", "splits", "split_counts", "source_data_identity", "sha256", "row_order_identity_sha256", *HOLD)
    return datasets, {key: projection[key] for key in fields}


def validate_population(datasets, source):
    identities = {}
    for split in SPLITS:
        dataset = datasets[split]
        order = [dataset.ordered[did] for did in dataset.dialogue_ids]
        population = [(uid, dataset.rows[uid]["dialogue_id"], int(dataset.rows[uid]["benchmark_label_id"]))
                      for dialogue in order for _, _, uid in dialogue]
        examples = [(dataset.rows[dialogue[0][2]]["dialogue_id"], tuple(item[2] for item in dialogue))
                    for dialogue in order]
        identity = {"dialogues": len(examples), "utterances": len(population), "sha256": json_identity(examples)}
        if identity != source.manifest["batch_pairing"]["population"][split]:
            raise ValueError(f"{split} canonical population differs from the historical F0 tape")
        if split == "dev" and tuple(population) != source.population:
            raise ValueError("Dev projection order/IDs/gold differ from the archived F0 prediction evidence")
        identities[split] = identity
    return identities


@dataclass(frozen=True)
class FrozenSplit:
    logits: np.ndarray
    population: tuple

    @property
    def labels(self):
        return np.array([label for _, _, label in self.population], dtype=np.int64)


@torch.inference_mode()
def frozen_forward(model, dataset, split, *, device, batch_size, expected_batches=None):
    require_split(split)
    if dataset.split != split or model.training or any(p.requires_grad for p in model.parameters()):
        raise ValueError("frozen Train/Dev forward requires the exact split and eval-only weights")
    logits, population, batch_hashes = [], [], []
    for batch in DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate):
        if batch["split"] != split:
            raise ValueError("frozen forward received a different split")
        examples = list(zip(batch["dialogue_ids"], batch["utterance_ids"]))
        digest = json_identity(examples)
        if expected_batches is not None and (len(batch_hashes) >= len(expected_batches)
                                             or digest != expected_batches[len(batch_hashes)]):
            raise ValueError("Dev batch tape differs from the archived F0 evidence")
        batch_hashes.append(digest)
        output = model(**{key: batch[key].to(device) for key in MODEL_INPUTS})
        valid = batch["attention_mask"]
        if not torch.equal(output.mask.cpu(), valid) or torch.any(batch["labels"][~valid] != -100):
            raise ValueError("frozen forward changed the valid mask or padding contract")
        logits.append(output.logits_ta[output.mask].cpu().numpy())
        uids = [(uid, did) for did, utterances in examples for uid in utterances]
        targets = batch["labels"][valid].tolist()
        if len(uids) != len(targets):
            raise ValueError("utterance key and label counts differ")
        population.extend((uid, did, target) for (uid, did), target in zip(uids, targets))
    if expected_batches is not None and batch_hashes != expected_batches:
        raise ValueError("incomplete Dev batch tape")
    if not logits or len(population) != len({uid for uid, _, _ in population}):
        raise ValueError("empty or duplicated frozen population")
    return FrozenSplit(np.concatenate(logits), tuple(population))


def forward_f0(source, datasets, device):
    curriculum_sha = membership(datasets["train"])
    if curriculum_sha != source.manifest["curriculum_membership_sha256"]:
        raise ValueError("Train curriculum membership differs from the frozen checkpoint")
    seed_everything(device, SEED)
    model = frozen_model(source, curriculum_sha, device)
    if not isinstance(model.fusion, DirectConcatFusion):
        raise ValueError("frozen F0 requires DirectConcat")
    batch_size = source.config["training"]["batch_size"]
    dev = frozen_forward(model, datasets["dev"], "dev", device=device, batch_size=batch_size,
                         expected_batches=source.manifest["batch_pairing"]["dev"]["batch_sha256"])
    if (dev.population != source.population or not np.allclose(dev.logits, source.logits, rtol=1e-4, atol=5e-4)
            or not np.array_equal(dev.logits.argmax(axis=1), source.logits.argmax(axis=1))):
        raise ValueError("live seed100 F0 does not reproduce archived Dev identity/logits/predictions")
    train = frozen_forward(model, datasets["train"], "train", device=device, batch_size=batch_size)
    if sha256_file(source.checkpoint) != source.identity["checkpoint_sha256"]:
        raise ValueError("F0 checkpoint changed during forward")
    return train, dev


def checked_trace_rows(directory, manifest, name):
    digest = manifest["outputs_sha256"][name]
    if sha256_file(directory / name) != digest:
        raise ValueError(f"Round 1 SHA-256 mismatch: {name}")
    rows = read_csv(directory / name)
    if sha256_file(directory / name) != digest:
        raise ValueError(f"Round 1 file changed during read: {name}")
    return [row for row in rows if int(row["seed"]) == SEED]


def align_trace(rows, meta):
    key = lambda row: (row["dialogue_id"], row["utterance_id"])
    indexed = {key(row): row for row in rows}
    if len(indexed) != len(rows) or set(indexed) != {key(row) for row in meta}:
        raise ValueError("Round 1 trace IDs are incomplete or duplicated")
    aligned = [indexed[key(row)] for row in meta]
    if any(left[field] != str(right[field]) for left, right in zip(aligned, meta)
           for field in ("speaker", "true_label", "shift_group", "target_index")):
        raise ValueError("Round 1 trace metadata differs from frozen Dev")
    return aligned


def history_groups(directory, manifest, meta, archived, mad):
    rows = checked_trace_rows(directory, manifest, "HISTORY_UTILITY_AUDIT.csv")
    samples = [row for row in rows if row["row_type"] == "sample"]
    if len(samples) != len(meta) * len(HISTORY) or {row["condition"] for row in samples} != set(HISTORY):
        raise ValueError("Round 1 history variants are incomplete")
    predictions = np.array([[CLASS_ORDER.index(row["predicted_label"]) for row in
                             align_trace([row for row in samples if row["condition"] == condition], meta)]
                            for condition in HISTORY])
    if not np.array_equal(predictions[0], archived.top1):
        raise ValueError("Round 1 FULL_HISTORY differs from historical seed100 F0")
    h_instability = (predictions[1:] != predictions[0]).sum(axis=0)
    u, h = (np.isin(np.arange(len(meta)), np.argsort(-values, kind="stable")[:Q4_SUPPORT])
            for values in (mad, h_instability))
    # Round 2's published seed100 captures are 35.54% (102/287) and 36.24% (104/287).
    if (int((u & archived.wrong).sum()), int((h & archived.wrong).sum())) != (102, 104):
        raise ValueError("reconstructed risk lists do not reproduce the Round 2 seed100 capture counts")
    return dict(zip(H_GROUPS, (u & ~h, h & ~u, u & h, ~(u | h))))


def audit_subgroups(source, directory):
    archived = scores(source.logits, [label for _, _, label in source.population])
    q4 = quartiles(archived.mad) == 4
    groups = dict.fromkeys((*GROUPS, *H_GROUPS))
    status = {"Q4_SOURCE": "ARCHIVED_F0_DEV_STABLE_MAD", "SHIFT_SUBGROUP_STATUS": "NOT_AVAILABLE",
              "H_SUBGROUP_STATUS": "NOT_AVAILABLE"}
    if directory is None:
        return q4, groups, {**status, "subgroup_reason": "Round 1 trace was not supplied"}
    try:
        manifest = read_json(directory / "failure_audit_manifest.json")
        validate_hold(manifest, legacy=True)
        if (manifest["data_identity"] != source.manifest["data_identity"]
                or [item for item in manifest["sources"] if item["seed"] == SEED] != [source.identity]):
            raise ValueError("Round 1 source is not the historical seed100 F0")
        meta = metadata(source.population)
        rows = align_trace(checked_trace_rows(directory, manifest, "AMBIGUITY_AUDIT.csv"), meta)
        if [row["predicted_label"] for row in rows] != [CLASS_ORDER[index] for index in archived.top1]:
            raise ValueError("Round 1 predictions differ from the historical seed100 F0")
        mad = np.array([float(row["mad"]) for row in rows])
        quartile = quartiles(mad)
        if (not np.all((mad >= 0) & (mad <= 1)) or not np.array_equal(quartile, [int(row["mad_quartile"]) for row in rows])
                or not np.allclose(mad, archived.mad, rtol=0, atol=5e-4)):
            raise ValueError("Round 1 MAD or frozen quartile assignments do not reproduce")
        q4 = quartile == 4
        shifts = selection_groups(meta)
        groups.update({name: shifts[name] for name in GROUPS})
        status.update(Q4_SOURCE="VERIFIED_ROUND1_TRACE", SHIFT_SUBGROUP_STATUS="AVAILABLE",
                      round1_manifest_sha256=sha256_file(directory / "failure_audit_manifest.json"))
        groups.update(history_groups(directory, manifest, meta, archived, mad))
        status["H_SUBGROUP_STATUS"] = "AVAILABLE"
    except TestHoldViolation:
        raise
    except (OSError, ValueError, KeyError, TypeError) as error:
        status["subgroup_reason"] = str(error)
    return q4, groups, status

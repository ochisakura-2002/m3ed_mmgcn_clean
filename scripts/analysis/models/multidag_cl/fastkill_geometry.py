"""Frozen pre-dropout representation diagnostics and raw arrays."""

from collections import defaultdict
from itertools import combinations
from pathlib import Path
import statistics

import numpy as np
import torch
from torch.utils.data import DataLoader
import yaml

from datasets.iemocap.tav_full_audit_dataset import COUNTS, collate, require_split
from models.multidag_cl.fusion.model import FusionTAV
from models.multidag_cl.ta_v0 import LABEL_VOCAB
from scripts.runtime.multidag_cl_fastkill.prefix_diagnostic import correlation
from scripts.runtime.multidag_cl_fastkill.runtime import capture_pre_dropout
from scripts.runtime.multidag_cl_fastkill.spec import HOLD
from scripts.runtime.multidag_cl_fusion.spec import F0_PARAMETER_COUNT, final_seed_source, make_final_config
from scripts.runtime.multidag_cl_ta_v0.formal import json_identity, sha256_file, write_json_atomic
from scripts.runtime.multidag_cl_ta_v0.training import reload_model_checkpoint
from scripts.runtime.multidag_cl_tav_full_audit_v1.confirm import checked_epochs, test_hold
from scripts.runtime.multidag_cl_tav_full_audit_v1.runtime import AuditRuntime, MODEL_INPUTS
from scripts.runtime.multidag_cl_tav_full_audit_v1.spec import write_csv
from utils.metrics import compute_confusion_matrix
from .export_fusion_topology_bundle import check_hold, confusion_rows, read_json

FOCUS_PAIRS = (("hap", "exc"), ("neu", "fru"), ("ang", "fru"))
CLASS_IDS = tuple(range(len(LABEL_VOCAB)))


def represent(runtime, split):
    require_split(split)
    dataset = runtime.dataset(split)
    loader = DataLoader(dataset, batch_size=16, collate_fn=collate, shuffle=False,
                        generator=torch.Generator().manual_seed(0))
    model = runtime.model
    was_training = model.training
    vectors, logits, labels, uids, dids = [], [], [], [], []
    model.eval()
    try:
        with torch.no_grad():
            for batch in loader:
                with capture_pre_dropout(model) as activations:
                    output = model(**{key: batch[key].to(runtime.device) for key in MODEL_INPUTS})
                if len(activations) != 1 or activations[0].shape != (*output.mask.shape, 300):
                    raise ValueError("geometry classifier activation changed")
                vectors.append(activations[0][output.mask].cpu().numpy())
                logits.append(output.logits_ta[output.mask].cpu().numpy())
                labels.append(batch["labels"][output.mask.cpu()].numpy())
                for did, ids in zip(batch["dialogue_ids"], batch["utterance_ids"]):
                    uids.extend(ids)
                    dids.extend([did] * len(ids))
    finally:
        model.train(was_training)
    if (not uids or len(set(uids)) != len(uids)
            or not runtime.synthetic_only and len(uids) != COUNTS[split]["utterances"]):
        raise ValueError("geometry population mismatch")
    return {"representation": np.concatenate(vectors), "logits": np.concatenate(logits),
            "label_id": np.concatenate(labels), "utterance_id": np.asarray(uids), "dialogue_id": np.asarray(dids)}


def normalize(values):
    return values / np.linalg.norm(values, axis=-1, keepdims=True).clip(min=1e-12)


def centroid_fit(data):
    vectors, labels = data["representation"].astype(np.float64), data["label_id"]
    if not np.isfinite(vectors).all() or not np.isfinite(data["logits"]).all():
        raise ValueError("nonfinite geometry inputs")
    support = np.bincount(labels, minlength=len(CLASS_IDS))
    return np.stack([vectors[labels == i].mean(axis=0) if support[i] else np.full(300, np.nan)
                     for i in CLASS_IDS]), support


def mean(values):
    return float(np.mean(values)) if len(values) else None


def margins(scores, labels):
    row = np.arange(len(labels))
    true_scores = scores[row, labels]
    alternatives = scores.copy()
    alternatives[row, labels] = -np.inf
    return true_scores - alternatives.max(axis=1), true_scores[:, None] - scores


def measure(data, reference, split, *, seed):
    labels = data["label_id"]
    centroids, support = centroid_fit(data)
    identity = {"seed": seed, "split": split}
    own_similarity = np.clip(normalize(data["representation"].astype(np.float64)) @ normalize(centroids).T, -1, 1)
    dispersion = [mean(1 - own_similarity[labels == i, i]) for i in CLASS_IDS]
    centroid_cosine = np.clip(normalize(centroids) @ normalize(centroids).T, -1, 1)

    similarity = np.clip(normalize(data["representation"].astype(np.float64)) @ normalize(reference).T, -1, 1)
    similarity = np.nan_to_num(similarity, nan=-np.inf)
    nearest = similarity.argmax(axis=1)
    inter_margin, pair_inter_margin = margins(similarity, labels)
    logit_margin, pair_logit_margin = margins(data["logits"], labels)
    classifier_predictions = data["logits"].argmax(axis=1)
    nearest_confusion = compute_confusion_matrix(labels, nearest, labels=list(CLASS_IDS))
    classifier_confusion = compute_confusion_matrix(labels, classifier_predictions, labels=list(CLASS_IDS))
    class_rows = [{**identity, "label": label, "support": int(support[i]), "intra_cosine_dispersion": dispersion[i],
                   "mean_inter_class_cosine_margin": mean(inter_margin[labels == i]),
                   "mean_classifier_logit_margin": mean(logit_margin[labels == i]),
                   "nearest_centroid_recall": float(nearest_confusion[i, i] / support[i]) if support[i] else None,
                   "classifier_recall": float(classifier_confusion[i, i] / support[i]) if support[i] else None}
                  for i, label in enumerate(LABEL_VOCAB)]
    pair_rows = []
    for i, j in combinations(CLASS_IDS, 2):
        a, b = LABEL_VOCAB[i], LABEL_VOCAB[j]
        distance = float(1 - centroid_cosine[i, j]) if support[i] and support[j] else None
        denominator = (dispersion[i] + dispersion[j]) / 2 if distance is not None else None
        for left, right in ((i, j), (j, i)):
            selection = labels == left
            pair_rows.append({**identity, "true_label": LABEL_VOCAB[left], "other_label": LABEL_VOCAB[right],
                              "preregistered_pair": (a, b) in FOCUS_PAIRS or (b, a) in FOCUS_PAIRS,
                              "centroid_cosine_distance": distance,
                              "separation_ratio": distance / denominator if denominator else None,
                              "mean_pair_inter_class_cosine_margin": mean(pair_inter_margin[selection, right])
                              if np.isfinite(reference[right]).all() else None,
                              "mean_pair_classifier_logit_margin": mean(pair_logit_margin[selection, right]),
                              "classifier_error_rate": float(classifier_confusion[left, right] / support[left])
                              if support[left] else None,
                              "nearest_centroid_error_rate": float(nearest_confusion[left, right] / support[left])
                              if support[left] else None})
    available = sorted({row["separation_ratio"] for row in pair_rows if row["separation_ratio"] is not None})
    for row in pair_rows:
        ratio = row["separation_ratio"]
        row["separation_rank_low_to_high"] = available.index(ratio) + 1 if ratio is not None else None
    paired = [row for row in pair_rows if row["separation_ratio"] is not None]
    comparison = {**identity, "directed_pairs": len(paired),
                  "classifier_error_vs_separation_pearson": correlation(
                      [row["classifier_error_rate"] for row in paired], [row["separation_ratio"] for row in paired])}
    arrays = {**data, "nearest_centroid_pred_id": nearest, "classifier_pred_id": classifier_predictions,
              "centroid": centroids, "centroid_cosine_distance": 1 - centroid_cosine,
              "train_reference_centroid": reference, "nearest_centroid_confusion": nearest_confusion,
              "classifier_confusion": classifier_confusion, "intra_cosine_dispersion": np.asarray(dispersion, dtype=float),
              "inter_class_cosine_margin": inter_margin, "pair_inter_class_cosine_margin": pair_inter_margin,
              "classifier_logit_margin": logit_margin, "pair_classifier_logit_margin": pair_logit_margin}
    return arrays, class_rows, pair_rows, comparison


def export_geometry(runtime, directory, seed):
    train = represent(runtime, "train")
    dev = represent(runtime, "dev")
    reference, _ = centroid_fit(train)
    classes, pairs, comparisons = [], [], []
    directory.mkdir(parents=True, exist_ok=False)
    for split, values in (("train", train), ("dev", dev)):
        if not runtime.synthetic_only and set(values["label_id"]) != set(CLASS_IDS):
            raise ValueError(f"{split} geometry is missing a frozen emotion class")
        raw, class_rows, pair_rows, comparison = measure(values, reference, split, seed=seed)
        np.savez_compressed(directory / f"{split}_geometry.npz", labels=np.asarray(LABEL_VOCAB), **raw)
        for normalized in (False, True):
            write_csv(directory / f"{split}_nearest_centroid_{'normalized' if normalized else 'counts'}.csv",
                      confusion_rows(raw["nearest_centroid_confusion"], normalized=normalized))
        classes.extend(class_rows)
        pairs.extend(pair_rows)
        comparisons.append(comparison)
    write_csv(directory / "class_geometry.csv", classes)
    write_csv(directory / "pair_geometry.csv", pairs)
    write_csv(directory / "geometry_confusion_association.csv", comparisons)
    return classes, pairs, comparisons


def multiseed(rows, group_fields, *, exclude=()):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in group_fields)].append(row)
    for group, members in groups.items():
        for key in members[0]:
            if key in (*group_fields, *exclude, "seed"):
                continue
            values = [member[key] for member in members if member[key] is not None]
            yield {**dict(zip(group_fields, group)), "metric": key, "n_seeds": len(values),
                   "mean": mean(values), "sample_sd": statistics.stdev(values) if len(values) > 1 else None}


def export_b0(datasets, data_identity, frozen_runs, output, *, device):
    seeds = final_seed_source()["seeds"]
    frozen_runs, output = Path(frozen_runs), Path(output)
    sources = []
    for seed in seeds:
        raw = make_final_config("F0", seed)
        run = frozen_runs / raw["experiment_id"]
        manifest = read_json(run / "manifests/run_manifest.json")
        evidence = read_json(run / "reports/validation_best_dev_evidence.json")
        resolved = yaml.safe_load((run / "resolved_config.yaml").read_text(encoding="utf-8"))
        checkpoint = run / "checkpoints/best_model.pt"
        check_hold(manifest)
        check_hold(evidence)
        if (manifest["status"] != "completed" or not manifest["best_checkpoint_reloaded"]
                or manifest["seed"] != seed
                or manifest["parameter_count"] != F0_PARAMETER_COUNT or manifest["config_sha256"] != json_identity(raw)
                or manifest["data_identity"] != data_identity or resolved["audit_config"] != raw
                or evidence["split"] != "dev" or evidence["metrics"] != manifest["best_dev_metrics"]
                or evidence["checkpoint_sha256"] != manifest["best_checkpoint_sha256"]
                or sha256_file(checkpoint) != manifest["best_checkpoint_sha256"]):
            raise ValueError(f"frozen F0 seed {seed} checkpoint evidence mismatch")
        checked_epochs(run, manifest["best_epoch"])
        sources.append((seed, raw, checkpoint, manifest))

    output.mkdir(parents=True, exist_ok=False)
    classes, pairs, comparisons, checkpoints = [], [], [], []
    for seed, raw, checkpoint, manifest in sources:
        runtime = AuditRuntime(FusionTAV(raw), datasets.__getitem__, device=device, formal_train_dev=True,
                               data_identity=data_identity, git_head=manifest["git_head"])
        payload = reload_model_checkpoint(runtime.model, checkpoint, validate=runtime.validate_checkpoint)
        test_hold(payload)
        if (payload["epoch"] != manifest["best_epoch"] or payload["git_head"] != manifest["git_head"]
                or sha256_file(checkpoint) != manifest["best_checkpoint_sha256"]):
            raise ValueError(f"frozen F0 seed {seed} checkpoint identity changed")
        seed_classes, seed_pairs, seed_comparisons = export_geometry(runtime, output / f"seed{seed}", seed)
        if runtime.optimizer_steps:
            raise ValueError("B0 performed an optimizer update")
        classes.extend(seed_classes)
        pairs.extend(seed_pairs)
        comparisons.extend(seed_comparisons)
        checkpoints.append({"seed": seed, "epoch": payload["epoch"], "git_head": payload["git_head"],
                            "checkpoint_sha256": manifest["best_checkpoint_sha256"]})
    write_csv(output / "class_geometry_five_seed.csv", list(multiseed(classes, ("split", "label"))))
    write_csv(output / "pair_geometry_five_seed.csv", list(multiseed(
        pairs, ("split", "true_label", "other_label"), exclude=("preregistered_pair",))))
    write_csv(output / "geometry_confusion_association_five_seed.csv", list(multiseed(comparisons, ("split",))))
    write_json_atomic(output / "b0_manifest.json", {**HOLD, "B0_DIAGNOSTIC": "PASS",
                      "optimizer_steps_performed": 0, "centroid_reference_split": "train",
                      "data_identity": data_identity, "checkpoints": checkpoints, "focus_pairs": FOCUS_PAIRS})

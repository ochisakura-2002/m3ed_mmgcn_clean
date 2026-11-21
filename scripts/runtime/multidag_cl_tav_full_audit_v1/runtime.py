"""TAV adaptation of the existing Train/Dev checkpoint and optimizer mechanics."""

from copy import deepcopy
import json
from pathlib import Path
import pickle

import torch
from torch.utils.data import Dataset

from datasets.iemocap.tav_full_audit_dataset import (
    COUNTS, DIMS, SCHEMA, TAVAuditDataset, collate, identity_sha, read_manifest, require_split,
)
from models.multidag_cl.tav_full_audit_v1 import TAVFullAudit
from scripts.runtime.multidag_cl_paper_reimplementation.curriculum import CurriculumRuntime
from models.multidag_cl.paper_reimplementation.config import CurriculumPartitionProfile, CurriculumScheduleProfile
from scripts.runtime.multidag_cl_paper_reimplementation.checkpoint import ValidationCheckpointSelector
from scripts.runtime.multidag_cl_ta_v0.formal import json_identity, sha256_file
from scripts.runtime.multidag_cl_ta_v0.training import (
    PROTOCOL, TrainDevRuntime, classification_report, masked_cross_entropy, seed_everything,
)

MODEL_INPUTS = (*DIMS, "speaker_ids_int", "lengths", "attention_mask")
BATCH_FIELDS = {*MODEL_INPUTS, "labels", "utterance_ids", "dialogue_ids", "split"}


def load_train_dev(asset_path, model_manifest_path, provenance_manifest_path):
    paths = {"asset": Path(asset_path), "model_manifest": Path(model_manifest_path),
             "provenance_manifest": Path(provenance_manifest_path)}
    hashes = {key: sha256_file(path) for key, path in paths.items()}
    provenance = json.loads(paths["provenance_manifest"].read_text(encoding="utf-8"))
    if (provenance.get("schema") != SCHEMA or provenance.get("status") != "complete"
            or provenance.get("asset_sha256") != hashes["asset"]
            or provenance.get("model_manifest_sha256") != hashes["model_manifest"]
            or provenance.get("alignment") != {"matched": 7433, "unmatched": 0, "ambiguous": 0, "split_conflict": 0}
            or provenance.get("split_counts") != COUNTS
            or provenance.get("test_labels_inspected") != 0):
        raise ValueError("TAV asset provenance/alignment/SHA256 not certified")
    rows = read_manifest(paths["model_manifest"])
    if identity_sha(rows) != provenance["row_order_identity_sha256"]:
        raise ValueError("canonical row/order identity SHA mismatch")
    with paths["asset"].open("rb") as stream:
        asset = pickle.load(stream)
    datasets = {split: TAVAuditDataset(asset, rows, split=split) for split in ("train", "dev")}
    if hashes != {key: sha256_file(path) for key, path in paths.items()}:
        raise ValueError("TAV package changed while loading")
    identity = {"schema": SCHEMA, "sha256": hashes, "row_order_identity_sha256": identity_sha(rows),
                "split_counts": COUNTS, "feature_dimensions": {"text": 1024, "audio": 1582, "visual": 342}}
    return datasets, identity, {key: path.as_posix() for key, path in paths.items()}


class AuditRuntime(TrainDevRuntime):
    """Parent selector/save/resume are reused; TAV batches and CL visibility adapt."""

    def __init__(self, model, dataset_factory, *, device="cpu", formal_train_dev=False,
                 synthetic_only=False, data_identity=None, git_head=None):
        if not isinstance(model, TAVFullAudit) or not (formal_train_dev is True or synthetic_only is True):
            raise ValueError("explicit TAV Train/Dev or synthetic mode required")
        self.device = torch.device(device)
        if self.device.type not in ("cpu", "cuda") or synthetic_only and self.device.type != "cpu":
            raise ValueError("invalid runtime device")
        self.model, self.dataset_factory = model.to(self.device), dataset_factory
        self.formal_train_dev, self.synthetic_only = formal_train_dev is True, synthetic_only is True
        self.data_identity, self.git_head = deepcopy(data_identity or {}), git_head
        protocol = PROTOCOL["optimizer"]
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=protocol["learning_rate"],
                                          betas=tuple(protocol["betas"]), eps=protocol["eps"],
                                          weight_decay=protocol["weight_decay"], foreach=False)
        self.selector, self.optimizer_steps, self.epoch, self.best_checkpoint = ValidationCheckpointSelector(), 0, 0, None
        train = self.dataset("train")
        self.curriculum = CurriculumRuntime.from_training_dataset(
            train, split="train", bucket_count=7,
            partition_profile=CurriculumPartitionProfile.BALANCED_STABLE_CONTIGUOUS,
            schedule_profile=CurriculumScheduleProfile.OFFICIAL_ONE_BUCKET_PER_EPOCH,
            enabled=model.audit_config["training"]["curriculum"])

    def dataset(self, split):
        require_split(split)
        dataset = self.dataset_factory(split)
        if dataset.split != split:
            raise ValueError("dataset split identity mismatch")
        return dataset

    @property
    def identity(self):
        return {"schema": "tav_full_audit_checkpoint_v1", "audit_config": deepcopy(self.model.audit_config),
                "data_identity": deepcopy(self.data_identity),
                "curriculum_membership_sha256": self.curriculum.manifest.membership_sha256,
                "training_config": deepcopy(self.model.audit_config["training"]),
                "seed": self.model.audit_config["training"]["seed"]}

    def loader(self, split, *, epoch=1):
        require_split(split)
        dataset = self.dataset(split)
        seed = self.model.audit_config["training"]["seed"]
        if split == "train":
            return self.curriculum.build_loader(dataset, global_epoch=epoch, batch_size=16, seed=seed, collate_fn=collate)
        return torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0, collate_fn=collate,
                                           generator=torch.Generator().manual_seed(seed))

    def forward_batch(self, batch, *, split, active_modalities=None):
        require_split(split)
        if set(batch) != BATCH_FIELDS or batch["split"] != split:
            raise ValueError("invalid TAV batch fields/split")
        if split == "train" and active_modalities is not None:
            raise ValueError("test-time modality mask is Dev-only")
        tensors = {key: batch[key].to(self.device) for key in (*MODEL_INPUTS, "labels")}
        self.model.train(split == "train")
        output = self.model(**{key: tensors[key] for key in MODEL_INPUTS}, active_modalities=active_modalities)
        if not torch.all(tensors["labels"][~output.mask] == -100):
            raise ValueError("invalid padding targets")
        loss, _ = masked_cross_entropy(output.logits_ta, tensors["labels"], output.mask)
        if any(not torch.isfinite(value).all() for value in (output.h_ta, output.logits_ta, output.prob_ta)):
            raise ValueError("nonfinite TAV output")
        return output, loss

    def train_step(self, batch):
        self.optimizer.zero_grad(set_to_none=True)
        output, loss = self.forward_batch(batch, split="train")
        loss.backward()
        gradients = [p.grad for p in self.model.parameters() if p.grad is not None]
        # w0 has no predecessor messages: its relation/attention parameters are
        # intentionally unused. Inactive modality branches have exact zero grads.
        if not gradients or any(not torch.isfinite(grad).all() for grad in gradients):
            raise ValueError("nonfinite/missing training gradients")
        norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0, error_if_nonfinite=True)
        self.optimizer.step()
        self.optimizer_steps += 1
        return {"loss": float(loss.detach()), "valid_target_count": int(output.mask.sum()),
                "gradient_norm": float(norm), "optimizer_steps": self.optimizer_steps}

    def evaluate(self, batches=None, *, split="dev", active_modalities=None, include_predictions=False,
                 prediction_details=False):
        if split != "dev":
            raise ValueError("Dev evaluation only; Test forbidden")
        batches = self.loader("dev") if batches is None else batches
        labels, predictions, utterance_ids, numerator, count = [], [], [], 0.0, 0
        details = []
        was_training = self.model.training
        try:
            with torch.no_grad():
                for batch in batches:
                    output, loss = self.forward_batch(batch, split="dev", active_modalities=active_modalities)
                    targets = batch["labels"][output.mask.cpu()].tolist()
                    labels.extend(targets)
                    predictions.extend(output.logits_ta[output.mask].argmax(-1).cpu().tolist())
                    if include_predictions:
                        utterance_ids.extend(uid for dialogue in batch["utterance_ids"] for uid in dialogue)
                        if prediction_details:
                            dialogue_ids = [did for did, ids in zip(batch["dialogue_ids"], batch["utterance_ids"])
                                            for _ in ids]
                            logits = output.logits_ta[output.mask].cpu().tolist()
                            probabilities = output.prob_ta[output.mask].cpu().tolist()
                            if len(dialogue_ids) != len(targets):
                                raise ValueError("Dev prediction dialogue identity mismatch")
                            details.extend(dict(dialogue_id=did, logits=row, probabilities=prob)
                                           for did, row, prob in zip(dialogue_ids, logits, probabilities))
                    numerator += float(loss) * len(targets)
                    count += len(targets)
        finally:
            self.model.train(was_training)
        if not count:
            raise ValueError("empty Dev evaluation")
        report = classification_report(labels, predictions, loss=numerator / count)
        if include_predictions:
            if len(utterance_ids) != count or len(set(utterance_ids)) != count:
                raise ValueError("Dev prediction utterance identity mismatch")
            report["predictions"] = [dict(utterance_id=uid, true_label=label, predicted_label=prediction)
                                     for uid, label, prediction in zip(utterance_ids, labels, predictions)]
            if prediction_details:
                for row, extra in zip(report["predictions"], details):
                    row.update(extra)
        return report


class SyntheticDialogues(Dataset):
    def __init__(self, split):
        require_split(split)
        self.split, self.items = split, []
        generator = torch.Generator().manual_seed(100 if split == "train" else 101)
        for index in range(7 if split == "train" else 2):
            length = 2 + index % 3
            did = f"synthetic_{split}_{index}"
            self.items.append({"dialogue_id": did, "utterance_ids": tuple(f"{did}_{j}" for j in range(length)),
                               **{key: torch.randn(length, dim, generator=generator) for key, dim in DIMS.items()},
                               "speaker_ids_int": torch.arange(length) % 2,
                               "labels": (torch.arange(length) + index) % 6, "split": split})

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


def synthetic_runtime(raw):
    datasets = {split: SyntheticDialogues(split) for split in ("train", "dev")}
    if "fusion" in raw:
        from scripts.runtime.multidag_cl_fusion.runtime import paired_runtime

        runtime, _ = paired_runtime(raw, datasets, device="cpu", synthetic_only=True,
                                    data_identity={"kind": "synthetic_train_dev_only"}, git_head="synthetic")
        return runtime
    seed_everything("cpu", raw["training"]["seed"])
    return AuditRuntime(TAVFullAudit(raw), datasets.__getitem__, synthetic_only=True,
                        data_identity={"kind": "synthetic_train_dev_only"}, git_head="synthetic")

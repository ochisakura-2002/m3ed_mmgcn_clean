"""TA-v0 Train/Dev mechanics; no Test path or historical trainer dependency."""

from copy import deepcopy
import math
import random

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from datasets.iemocap.ta_v0_dataset import ta_v0_collate
from models.multidag_cl.ta_v0 import LABEL_VOCAB, TAV0, TAV0Config
from models.multidag_cl.ta_v0.causal_history_gate import collect_gate_diagnostics
from scripts.models.multidag_cl.ta_v0.train import (
    BATCH_FIELDS, MODEL_INPUTS, require_development_split,
)
from scripts.runtime.multidag_cl_paper_reimplementation.checkpoint import (
    ValidationCheckpointSelector, load_checkpoint, save_checkpoint_atomic,
)
from utils.metrics import compute_classification_metrics, compute_confusion_matrix

PROTOCOL = {
    "seed": 100, "max_epochs": 30, "batch_size": 16,
    "optimizer": {"name": "AdamW", "learning_rate": 0.0005,
                  "betas": [0.9, 0.999], "eps": 1e-6, "weight_decay": 0.0},
    "scheduler": None, "gradient_clip_norm": 5.0, "class_weight": None,
    "label_smoothing": 0.0, "curriculum": False, "early_stopping": False,
    "train_shuffle": True, "dev_shuffle": False,
}
CHECKPOINT_SCHEMA = "ta_v0_train_dev_checkpoint_v1"


def seed_everything(device):
    random.seed(100)
    np.random.seed(100)
    torch.manual_seed(100)
    if torch.device(device).type == "cuda":
        torch.cuda.manual_seed_all(100)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def masked_cross_entropy(logits, targets, mask):
    """Padding is never indexed into CE, even if it contains invalid targets."""
    if (mask.dtype != torch.bool or targets.dtype != torch.int64
            or targets.shape != mask.shape or logits.shape != (*mask.shape, 6)):
        raise ValueError("invalid masked CE shapes/types")
    count = int(mask.sum().item())
    if count == 0 or torch.any((targets[mask] < 0) | (targets[mask] >= 6)):
        raise ValueError("invalid valid target count/labels")
    loss = F.cross_entropy(logits[mask], targets[mask])
    if not torch.isfinite(loss):
        raise ValueError("nonfinite cross entropy")
    return loss, count


def classification_report(labels, predictions, *, loss):
    if not len(labels) or len(labels) != len(predictions) or not math.isfinite(loss):
        raise ValueError("invalid Dev metrics inputs")
    labels, predictions = np.asarray(labels), np.asarray(predictions)
    if np.any((labels < 0) | (labels >= 6) | (predictions < 0) | (predictions >= 6)):
        raise ValueError("invalid metric labels")
    metrics = compute_classification_metrics(labels, predictions, labels=list(range(6)))
    metrics["accuracy"] = metrics.pop("acc")
    metrics.update(loss=float(loss), valid_target_count=len(labels))
    confusion = compute_confusion_matrix(labels, predictions, labels=list(range(6)))
    support, predicted = confusion.sum(1), confusion.sum(0)
    true_positive = confusion.diagonal()
    precision = np.divide(true_positive, predicted, out=np.zeros(6), where=predicted > 0)
    recall = np.divide(true_positive, support, out=np.zeros(6), where=support > 0)
    f1 = np.divide(2 * precision * recall, precision + recall,
                   out=np.zeros(6), where=(precision + recall) > 0)
    per_class = [dict(label_id=i, label=LABEL_VOCAB[i], precision=float(precision[i]),
                      recall=float(recall[i]), f1=float(f1[i]), support=int(support[i]))
                 for i in range(6)]
    return {"metrics": metrics, "per_class": per_class, "confusion_matrix": confusion.tolist()}


def cpu_snapshot(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu_snapshot(v) for k, v in value.items()}
    if isinstance(value, list):
        return [cpu_snapshot(v) for v in value]
    if isinstance(value, tuple):
        return tuple(cpu_snapshot(v) for v in value)
    return deepcopy(value)


class TrainDevRuntime:
    """Explicitly authorized real training or explicit synthetic unit tests.

    Loaders are created lazily after split validation. Each epoch's shuffle has
    its own generator, so reloads and Dev inference cannot change Train order.
    """

    def __init__(self, model, dataset_factory, *, device="cpu", formal_train_dev=False,
                 synthetic_only=False, data_identity=None, git_head=None):
        if formal_train_dev is not True and synthetic_only is not True:
            raise ValueError("real optimizer steps require --formal-train-dev")
        self.device = torch.device(device)
        if synthetic_only and self.device.type != "cpu":
            raise ValueError("synthetic checks are CPU only")
        if not isinstance(model, TAV0) or model.config not in (
                TAV0Config(), TAV0Config(use_causal_history_gate=True)):
            raise ValueError("formal TA-v0 architecture is frozen")
        self.model = model.to(self.device)
        self.dataset_factory = dataset_factory
        self.formal_train_dev = formal_train_dev is True
        self.synthetic_only = synthetic_only is True
        self.data_identity = deepcopy(data_identity or {})
        self.git_head = git_head
        optimizer = PROTOCOL["optimizer"]
        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=optimizer["learning_rate"], betas=tuple(optimizer["betas"]),
            eps=optimizer["eps"], weight_decay=optimizer["weight_decay"], foreach=False,
        )
        self.selector = ValidationCheckpointSelector()
        self.optimizer_steps = 0
        self.epoch = 0
        self.best_checkpoint = None

    @property
    def identity(self):
        return {"schema": CHECKPOINT_SCHEMA, "model_config": self.model.config.checkpoint_config(),
                "label_vocab": list(LABEL_VOCAB), "seed": 100,
                "training_config": deepcopy(PROTOCOL), "data_identity": self.data_identity}

    def loader(self, split, *, epoch=1):
        require_development_split(split)
        dataset = self.dataset_factory(split)
        if dataset.split != split:
            raise ValueError("dataset split identity mismatch")
        generator = torch.Generator().manual_seed(100 + epoch - 1 if split == "train" else 100)
        return DataLoader(dataset, batch_size=16, shuffle=split == "train", num_workers=0,
                          collate_fn=ta_v0_collate, generator=generator)

    def forward_batch(self, batch, *, split):
        require_development_split(split)
        if set(batch) != BATCH_FIELDS or batch["split"] != split:
            raise ValueError("TA-only batch fields/split identity mismatch")
        batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}
        self.model.train(split == "train")
        output = self.model(**{key: batch[key] for key in MODEL_INPUTS})
        if not torch.all(batch["labels"][~output.mask] == -100):
            raise ValueError("invalid padding labels")
        loss, _ = masked_cross_entropy(output.logits_ta, batch["labels"], output.mask)
        if any(not torch.isfinite(value).all() for value in
               (output.logits_ta, output.prob_ta, output.h_ta)):
            raise ValueError("nonfinite TA output")
        return output, loss

    def train_step(self, batch):
        self.optimizer.zero_grad(set_to_none=True)
        output, loss = self.forward_batch(batch, split="train")
        loss.backward()
        parameters = [p for p in self.model.parameters() if p.requires_grad]
        if any(p.grad is None or not torch.isfinite(p.grad).all() for p in parameters):
            raise ValueError("missing/nonfinite gradients; optimizer step refused")
        norm = torch.nn.utils.clip_grad_norm_(parameters, 5.0, error_if_nonfinite=True)
        if not torch.isfinite(norm):
            raise ValueError("nonfinite gradient norm")
        self.optimizer.step()
        self.optimizer_steps += 1
        return {"loss": float(loss.detach()), "valid_target_count": int(output.mask.sum()),
                "gradient_norm": float(norm), "optimizer_steps": self.optimizer_steps}

    def evaluate(self, batches=None, *, split="dev"):
        if split != "dev":
            raise ValueError("Dev evaluator only; Test evaluator forbidden")
        if batches is None:
            batches = self.loader("dev")
        labels, predictions, numerator, count = [], [], 0.0, 0
        was_training = self.model.training
        try:
            with collect_gate_diagnostics(self.model) as diagnostics, torch.no_grad():
                for batch in batches:
                    output, loss = self.forward_batch(batch, split="dev")
                    mask = output.mask.cpu()
                    y = batch["labels"].cpu()[mask].tolist()
                    labels.extend(y)
                    predictions.extend(output.logits_ta[output.mask].argmax(-1).cpu().tolist())
                    numerator += float(loss) * len(y)
                    count += len(y)
        finally:
            self.model.train(was_training)
        if count == 0:
            raise ValueError("empty Dev evaluation")
        report = classification_report(labels, predictions, loss=numerator / count)
        if self.model.config.use_causal_history_gate:
            report["chg_gate_diagnostics"] = diagnostics.summary()
        return report

    @property
    def test_loader(self):
        raise ValueError("Test loader forbidden")

    @property
    def test_predictions(self):
        raise ValueError("Test predictions forbidden")

    def evaluate_test(self, *args, **kwargs):
        raise ValueError("Test evaluator forbidden")

    def checkpoint_payload(self, epoch, *, epoch_complete=True):
        rng = {"python": random.getstate(), "numpy": np.random.get_state(),
               "torch": torch.get_rng_state()}
        if self.device.type == "cuda":
            rng["cuda"] = torch.cuda.get_rng_state_all()
        return cpu_snapshot({
            **self.identity, "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(), "epoch": epoch,
            "epoch_complete": epoch_complete,
            "global_step": self.optimizer_steps, "optimizer_steps": self.optimizer_steps,
            "best_dev_weighted_f1": self.selector.best.val_weighted_f1,
            "best_dev_loss": self.selector.best.val_loss,
            "selector": self.selector.state_dict(), "rng_state": rng,
            "git_head": self.git_head, "test_loader_built": 0, "test_evaluation_count": 0,
        })

    def save_epoch(self, directory, *, epoch, dev, epoch_complete=True):
        metrics = dev["metrics"]
        if (not all(math.isfinite(metrics[k]) for k in ("weighted_f1", "loss"))
                or not 0 <= metrics["weighted_f1"] <= 1):
            raise ValueError("nonfinite/invalid checkpoint metric")
        improved = self.selector.update(epoch=epoch, val_weighted_f1=metrics["weighted_f1"],
                                        val_loss=metrics["loss"], global_step=self.optimizer_steps)
        self.epoch = epoch
        current = self.checkpoint_payload(epoch, epoch_complete=epoch_complete)
        if improved:
            self.best_checkpoint = current
            save_checkpoint_atomic(directory / "best_model.pt", current)
            save_checkpoint_atomic(directory / "best_model_weights_only.pt", current["model_state_dict"])
        # Self-contained last checkpoint preserves the actual historical best,
        # including its own optimizer/RNG state; there is no checkpoint search.
        save_checkpoint_atomic(directory / "last_model.pt",
                               dict(current, best_checkpoint=self.best_checkpoint))
        return improved

    def validate_checkpoint(self, checkpoint):
        for key, value in self.identity.items():
            if checkpoint.get(key) != value:
                raise ValueError(f"resume {key} identity mismatch")
        required = {"model_state_dict", "optimizer_state_dict", "epoch", "global_step",
                    "optimizer_steps", "selector", "rng_state", "best_dev_weighted_f1",
                    "best_dev_loss", "test_loader_built", "test_evaluation_count"}
        if not required <= set(checkpoint):
            raise ValueError("incomplete structured resume checkpoint")
        if checkpoint["test_loader_built"] != 0 or checkpoint["test_evaluation_count"] != 0:
            raise ValueError("resume Test isolation mismatch")
        if (type(checkpoint["epoch"]) is not int or not 1 <= checkpoint["epoch"] <= 30
                or checkpoint["global_step"] != checkpoint["optimizer_steps"]
                or checkpoint["global_step"] < 1):
            raise ValueError("invalid resume epoch/global_step")
        if checkpoint["selector"]["locked"]:
            raise ValueError("locked selector cannot resume")
        # Check optimizer values as well as the declared training identity.
        groups = checkpoint["optimizer_state_dict"]["param_groups"]
        if len(groups) != 1:
            raise ValueError("resume optimizer group mismatch")
        expected = self.optimizer.state_dict()["param_groups"][0]
        if groups[0] != expected:
            raise ValueError("resume optimizer config mismatch")

    def resume(self, path):
        checkpoint = load_checkpoint(path, torch.device("cpu"))
        self.validate_checkpoint(checkpoint)
        if checkpoint.get("epoch_complete") is not True:
            raise ValueError("resume requires a completed epoch; one-step smoke is not resumable")
        best = checkpoint.get("best_checkpoint", checkpoint)
        self.validate_checkpoint(best)
        selector = ValidationCheckpointSelector()
        selector.load_state_dict(checkpoint["selector"])
        if (selector.best is None or selector.best.epoch != best["epoch"]
                or selector.best.val_weighted_f1 != best["best_dev_weighted_f1"]
                or selector.best.val_loss != best["best_dev_loss"]):
            raise ValueError("resume historical best identity mismatch")
        self.model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.selector = selector
        self.optimizer_steps = checkpoint["global_step"]
        self.epoch = checkpoint["epoch"]
        self.best_checkpoint = best
        random.setstate(checkpoint["rng_state"]["python"])
        np.random.set_state(checkpoint["rng_state"]["numpy"])
        torch.set_rng_state(checkpoint["rng_state"]["torch"])
        if self.device.type == "cuda":
            if "cuda" not in checkpoint["rng_state"]:
                raise ValueError("CUDA resume requires CUDA RNG state")
            torch.cuda.set_rng_state_all(checkpoint["rng_state"]["cuda"])
        return checkpoint

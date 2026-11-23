"""Reuse frozen F0 pairing and apply an objective only on Train batches."""

from contextlib import contextmanager
from itertools import chain, combinations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from models.multidag_cl.ta_v0 import LABEL_VOCAB
from scripts.runtime.multidag_cl_fusion.runtime import FusionAuditRuntime, paired_model, training_rng_identity
from scripts.runtime.multidag_cl_fusion.spec import F0_PARAMETER_COUNT
from .objectives import alpha_values, class_anchor_ce, new_prototypes
from .prefix_diagnostic import exposure_rows
from .spec import A1, ANCHOR_WEIGHT, B1, CTRL, UNIFORM_EPOCH, baseline, validate_arm


@contextmanager
def capture_pre_dropout(model):
    classifier = model.classifier
    if not (isinstance(classifier[3], nn.ReLU) and isinstance(classifier[4], nn.Dropout)):
        raise ValueError("frozen second ReLU / dropout classifier interface changed")
    activations = []
    hook = classifier[3].register_forward_hook(lambda module, args, output: activations.append(output))
    try:
        yield activations
    finally:
        hook.remove()


def prototype_rows(prototypes, epoch):
    vectors = prototypes.detach().cpu()
    normalized = F.normalize(vectors, dim=-1)
    cosines = normalized @ normalized.T
    norms = [{"epoch": epoch, "label": label, "prototype_l2_norm": float(value)}
             for label, value in zip(LABEL_VOCAB, vectors.norm(dim=-1))]
    pairs = [{"epoch": epoch, "label_a": LABEL_VOCAB[i], "label_b": LABEL_VOCAB[j],
              "prototype_cosine": float(cosines[i, j])}
             for i, j in combinations(range(len(LABEL_VOCAB)), 2)]
    return norms, pairs


class FastKillRuntime(FusionAuditRuntime):
    def __init__(self, model, dataset_factory, *, config, prefix_difficulties=None, **options):
        super().__init__(model, dataset_factory, **options)
        self.config = config
        self.arm = config["arm"]
        if self.arm == A1 and prefix_difficulties is None:
            raise ValueError("A1 requires the passed A0 prefix scores")
        self.prefix_difficulties = prefix_difficulties
        self.prototypes = new_prototypes(self.device) if self.arm == B1 else None
        self.epoch_objectives, self.prototype_norms, self.prototype_cosines = [], [], []
        self.alpha_history, self.class_exposure = [], []
        self._ce_terms, self._anchor_terms, self._weights, self._labels = [], [], [], []
        if self.prototypes is not None:
            # Keep AdamW's original options and one parameter group.
            self.optimizer.param_groups[0]["params"].append(self.prototypes)
            self.prototypes.register_hook(self._validate_prototype_gradient)

    @staticmethod
    def _validate_prototype_gradient(gradient):
        if not torch.isfinite(gradient).all():
            raise ValueError("nonfinite class prototype gradient")

    @property
    def identity(self):
        frozen = super().identity
        if self.arm == CTRL:
            return frozen
        return {**frozen, "fastkill": self.config}

    def training_parameters(self):
        parameters = super().training_parameters()
        return parameters if self.prototypes is None else chain(parameters, (self.prototypes,))

    def checkpoint_payload(self, epoch, *, epoch_complete=True):
        payload = super().checkpoint_payload(epoch, epoch_complete=epoch_complete)
        if self.prototypes is not None:
            payload["training_only_class_prototypes"] = self.prototypes.detach().cpu().clone()
        return payload

    def validate_checkpoint(self, checkpoint):
        super().validate_checkpoint(checkpoint)
        prototypes = checkpoint.get("training_only_class_prototypes")
        if self.arm == B1 and (not isinstance(prototypes, torch.Tensor) or prototypes.shape != (6, 300)
                               or not torch.isfinite(prototypes).all()):
            raise ValueError("B1 checkpoint lacks valid training-only prototypes")

    def forward_batch(self, batch, *, split, active_modalities=None):
        if split != "train" or self.arm == CTRL:
            return super().forward_batch(batch, split=split, active_modalities=active_modalities)
        if self.arm == B1:
            with capture_pre_dropout(self.model) as activations:
                output, ce = super().forward_batch(batch, split=split, active_modalities=active_modalities)
            if len(activations) != 1 or activations[0].shape != (*output.mask.shape, 300):
                raise ValueError("missing B1 classifier activation")
            labels = batch["labels"].to(self.device)[output.mask]
            anchor = class_anchor_ce(activations[0][output.mask], labels, self.prototypes)
            total = ce + ANCHOR_WEIGHT * anchor
            if not torch.isfinite(total):
                raise ValueError("nonfinite B1 total loss")
            self._ce_terms.append((float(ce.detach()), len(labels)))
            self._anchor_terms.append((float(anchor.detach()), len(labels)))
            return output, total

        output, ce = super().forward_batch(batch, split=split, active_modalities=active_modalities)
        difficulties = [self.prefix_difficulties[did] for did in batch["dialogue_ids"]]
        if tuple(map(len, difficulties)) != tuple(map(len, batch["utterance_ids"])):
            raise ValueError("A1 scores do not align with the Train batch")
        epoch = self.epoch + 1
        alpha = alpha_values([value for dialogue in difficulties for value in dialogue], epoch)
        targets = batch["labels"][batch["attention_mask"]]
        if len(alpha) != len(targets):
            raise ValueError("A1 weight count differs from valid target count")
        self._labels.extend(targets.tolist())
        self._weights.extend(alpha)
        if epoch >= UNIFORM_EPOCH:
            return output, ce
        weights = output.logits_ta.new_tensor(alpha)
        per_target = F.cross_entropy(output.logits_ta[output.mask], targets.to(self.device), reduction="none")
        return output, (weights * per_target).mean()

    def save_epoch(self, directory, *, epoch, dev, epoch_complete=True):
        if self.arm == B1:
            count = sum(size for _, size in self._ce_terms)
            ce = sum(value * size for value, size in self._ce_terms) / count
            anchor = sum(value * size for value, size in self._anchor_terms) / count
            self.epoch_objectives.append({"epoch": epoch, "valid_targets": count, "L_CE": ce,
                                          "L_anchor": anchor, "L_total": ce + ANCHOR_WEIGHT * anchor})
            norms, cosines = prototype_rows(self.prototypes, epoch)
            self.prototype_norms.extend(norms)
            self.prototype_cosines.extend(cosines)
            self._ce_terms.clear()
            self._anchor_terms.clear()
        if self.arm == A1:
            alpha = np.asarray(self._weights)
            self.alpha_history.append({"epoch": epoch, "valid_targets": len(alpha), "mean": float(alpha.mean()),
                                       "min": float(alpha.min()), "max": float(alpha.max()),
                                       "q05": float(np.quantile(alpha, 0.05)), "median": float(np.median(alpha)),
                                       "q95": float(np.quantile(alpha, 0.95))})
            self.class_exposure.extend(exposure_rows(self._labels, alpha, epoch))
            self._weights.clear()
            self._labels.clear()
        return super().save_epoch(directory, epoch=epoch, dev=dev, epoch_complete=epoch_complete)


def build_runtime(config, datasets, *, device, reference=None, pairing=None, **options):
    config = validate_arm(config)
    model, reference, initialization = paired_model(baseline(), device=device, reference=reference)
    runtime = FastKillRuntime(model, datasets.__getitem__, config=config, initialization=initialization,
                              f0_pairing=pairing, device=device, **options)
    initialization["training_rng_identity"] = training_rng_identity(device)
    if sum(parameter.numel() for parameter in model.parameters()) != F0_PARAMETER_COUNT:
        raise ValueError("frozen inference parameter count changed")
    return runtime, reference

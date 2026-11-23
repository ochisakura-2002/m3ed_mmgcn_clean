"""Prefix scoring and the minimal train-only cosine prototype objective."""

import math

import torch
from torch.nn import functional as F

from models.multidag_cl.ta_v0 import LABEL_VOCAB
from .spec import SEED, TAU, UNIFORM_EPOCH, WEIGHT_STRENGTH, WES_B, WES_K

# Six IEMOCAP points from the LSDGNN authors' emotion wheel. The angle
# numerators are in units of pi/20. Neutral is the origin (zero valence).
ANGLE = {"hap": 3, "sad": 29, "neu": None, "ang": 11, "exc": 7, "fru": 21}
POINTS = tuple((0.0, 0.0) if ANGLE[label] is None else
               (math.cos(ANGLE[label] * math.pi / 20), math.sin(ANGLE[label] * math.pi / 20))
               for label in LABEL_VOCAB)


def wheel_similarity(left, right):
    (v_left, a_left), (v_right, a_right) = POINTS[left], POINTS[right]
    valence_product = v_left * v_right
    if valence_product == 0:
        return 1 / len(LABEL_VOCAB)
    if valence_product < 0:
        return 0.0
    return max(valence_product + a_left * a_right, 0.0)


def prefix_scores(labels, speakers):
    """Return per-position equal-shift score, WES score and weighted increment."""
    if not len(labels) or len(labels) != len(speakers):
        raise ValueError("aligned nonempty labels/speakers required")
    last_label, scores = {}, []
    equal_shifts, weighted_shifts = 0, 0.0
    for position, (label, speaker) in enumerate(zip(labels, speakers), start=1):
        if type(label) is not int or not 0 <= label < len(LABEL_VOCAB) or type(speaker) is not int or speaker < 0:
            raise ValueError("prefix scoring requires valid integer targets and unpadded speakers")
        previous = last_label.get(speaker, label)
        shifted = previous != label
        increment = WES_K * wheel_similarity(previous, label) + WES_B if shifted else 0.0
        equal_shifts += shifted
        weighted_shifts += increment
        last_label[speaker] = label
        speakers_seen = len(last_label)
        denominator = position + speakers_seen
        scores.append(((equal_shifts + speakers_seen) / denominator,
                       (weighted_shifts + speakers_seen) / denominator, increment))
    return scores


def alpha_values(difficulties, epoch):
    if type(epoch) is not int or epoch < 1:
        raise ValueError("weight schedule requires a positive integer epoch")
    decay = max(0.0, (UNIFORM_EPOCH - epoch) / (UNIFORM_EPOCH - 1))
    limit = max(1.0, WES_K + WES_B)
    return [1.0 - WEIGHT_STRENGTH * decay * difficulty / limit for difficulty in difficulties]


def new_prototypes(device):
    generator = torch.Generator(device="cpu").manual_seed(SEED + 1)
    values = F.normalize(torch.randn(len(LABEL_VOCAB), 300, generator=generator), dim=1)
    return torch.nn.Parameter(values.to(device))


def class_anchor_ce(representation, targets, prototypes):
    cosine = F.normalize(representation, dim=-1) @ F.normalize(prototypes, dim=-1).T
    return F.cross_entropy(cosine / TAU, targets)

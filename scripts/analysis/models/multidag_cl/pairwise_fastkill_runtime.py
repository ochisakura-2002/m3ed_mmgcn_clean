"""Frozen-feature training, label-free inference, and causal probes."""

from dataclasses import dataclass
import time

import numpy as np
import torch
from torch.nn import functional as F

from models.multidag_cl.pairwise_refinement import (
    CLASS_COUNT, PARAMETER_COUNT, STATE_WIDTH, PairwiseRefinementHead, Refinement, bounded_writeback,
)
from scripts.runtime.multidag_cl_ta_v0.formal import json_identity
from scripts.runtime.multidag_cl_ta_v0.training import seed_everything

from .pairwise_preflight_inputs import FEATURE_DIMENSIONS, SEED
from .pairwise_preflight_metrics import gate, ranked_candidates


EPOCHS = 20
BATCH_SIZE = 128
LR = 3e-3
BETAS = (0.9, 0.999)
TRAINING = {"optimizer": "Adam", "lr": LR, "betas": BETAS, "weight_decay": 0,
            "batch_size": BATCH_SIZE, "epochs": EPOCHS, "shuffle": True, "seed": SEED,
            "loss": "mean_softplus_negative_signed_margin"}
RTOL, ATOL = 1e-5, 1e-6


def cache_identity(split):
    h0, h4 = split.layer_states
    labels = split.labels
    arrays = {"h0": h0, "h4": h4, "logits": split.logits, "labels": labels}
    rows = len(split.population)
    for name, values in arrays.items():
        width = {"h0": STATE_WIDTH, "h4": STATE_WIDTH, "logits": CLASS_COUNT}.get(name)
        shape, dtype = ((rows,), np.int64) if width is None else ((rows, width), np.float32)
        if values.shape != shape or values.dtype != dtype or not np.isfinite(values).all():
            raise ValueError(f"invalid frozen cache: {name}")
    return {"population": rows, "population_sha256": json_identity(split.population),
            "tensors": {name: {"shape": array.shape, "dtype": str(array.dtype), "bytes": array.nbytes}
                        for name, array in arrays.items()},
            "tensor_bytes": sum(array.nbytes for array in arrays.values())}


def supervised_rows(labels, top1, top2, selected):
    return selected & ((labels == top1) | (labels == top2))


def ranking_loss(margin, delta, target):
    return F.softplus(-target * (margin + delta)).mean()


def sync(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def train_head(train, scores, selected, *, device):
    """The only optimization loop. It has no Dev or backbone argument."""
    supervised = supervised_rows(scores.labels, scores.top1, scores.top2, selected)
    support = int(supervised.sum())
    targets = np.where(scores.labels[supervised] == scores.top1[supervised], 1, -1)
    if support == 0:
        raise ValueError("empty Train supervision set")
    h0, h4, a, b, margin, target = (
        torch.as_tensor(values, device=device) for values in
        (*(state[supervised] for state in train.layer_states), scores.top1[supervised],
         scores.top2[supervised], scores.margin[supervised], targets)
    )
    seed_everything(device, SEED)
    shuffle = torch.Generator().manual_seed(SEED)
    head = PairwiseRefinementHead().to(device)
    if sum(p.numel() for p in head.parameters() if p.requires_grad) != PARAMETER_COUNT:
        raise ValueError("incorrect trainable Pairwise Head parameter count")
    optimizer = torch.optim.Adam(head.parameters(), lr=LR, betas=BETAS, weight_decay=0)
    history = []
    sync(device)
    start = time.perf_counter()
    for epoch in range(1, EPOCHS + 1):
        loss_sum = 0.0
        for indices in torch.randperm(support, generator=shuffle).split(BATCH_SIZE):
            indices = indices.to(device)
            optimizer.zero_grad(set_to_none=True)
            delta = head(h0[indices], h4[indices], a[indices], b[indices])
            loss = ranking_loss(margin[indices], delta, target[indices])
            if not torch.isfinite(loss):
                raise ValueError(f"nonfinite Train loss at epoch {epoch}")
            loss.backward()
            if any(p.grad is None or not torch.isfinite(p.grad).all() for p in head.parameters()):
                raise ValueError(f"missing or nonfinite head gradients at epoch {epoch}")
            optimizer.step()
            loss_sum += loss.item() * len(indices)
        with torch.no_grad():
            delta = head(h0, h4, a, b)
            if (any(not torch.isfinite(p).all() for p in head.parameters())
                    or not (delta.abs() < 1).all()):
                raise ValueError(f"nonfinite or unbounded head at epoch {epoch}")
            history.append({"epoch": epoch, "train_loss": loss_sum / support, "n_s": support,
                            "mean_abs_delta": delta.abs().mean().item(),
                            "fraction_delta_positive": (delta > 0).float().mean().item(),
                            "fraction_delta_negative": (delta < 0).float().mean().item()})
    sync(device)
    return head.eval(), history, time.perf_counter() - start


@dataclass(frozen=True)
class Inference:
    top1: torch.Tensor
    top2: torch.Tensor
    mad: torch.Tensor
    margin: torch.Tensor
    selected: torch.Tensor
    output: Refinement


@torch.no_grad()
def infer(head, logits, h0, h4, valid, threshold):
    """Rank only valid rows using Preflight arithmetic. Gold is absent."""
    ranked, mad, margin = ranked_candidates(logits[valid].cpu().numpy())
    selected = torch.zeros_like(valid)
    delta = torch.zeros_like(valid, dtype=logits.dtype)
    top1 = torch.zeros_like(valid, dtype=torch.long)
    top2 = torch.zeros_like(top1)
    full_mad = torch.zeros_like(valid, dtype=torch.float64)
    full_margin = torch.zeros_like(full_mad)
    top1[valid] = torch.as_tensor(ranked[:, 0], device=valid.device)
    top2[valid] = torch.as_tensor(ranked[:, 1], device=valid.device)
    full_mad[valid] = torch.as_tensor(mad, device=valid.device)
    full_margin[valid] = torch.as_tensor(margin, device=valid.device)
    selected[valid] = torch.as_tensor(gate(mad, margin, True, threshold), device=valid.device)
    if head is None or not selected.any():
        output = Refinement(logits, delta, full_margin, torch.zeros_like(selected))
    else:
        delta[selected] = head(h0[selected], h4[selected], top1[selected], top2[selected])
        output = bounded_writeback(logits, top1, top2, selected, full_margin, delta)
    return Inference(top1, top2, full_mad, full_margin, selected, output)


def verify_writeback(logits, valid, result):
    output = result.output
    outside = ~output.written
    classes = torch.arange(logits.shape[-1], device=logits.device)
    non_candidates = (classes != result.top1[..., None]) & (classes != result.top2[..., None])
    expected = torch.where(output.written & (output.corrected_margin < 0), result.top2, result.top1)
    if (not torch.isfinite(output.logits[valid]).all() or not (output.delta.abs() < 1).all()
            or not torch.isfinite(output.corrected_margin).all()):
        raise ValueError("nonfinite or unbounded Pairwise inference")
    if ((result.selected & ~valid).any() or (output.written & ~result.selected).any()
            or (output.delta[~result.selected] != 0).any()
            or not torch.equal(output.logits[outside].view(torch.uint8), logits[outside].view(torch.uint8))
            or not torch.equal(output.logits[non_candidates].view(torch.uint8), logits[non_candidates].view(torch.uint8))
            or not torch.equal(output.logits[valid].argmax(-1), expected[valid])):
        raise ValueError("Pairwise bypass, top-two writeback, or winner identity failed")


@torch.no_grad()
def collect_causal_probes(model, device):
    """Exercise the loaded F0 on independent synthetic inputs only."""
    generator = torch.Generator().manual_seed(SEED)
    lengths = torch.tensor([5, 3])
    mask = torch.arange(5)[None, :] < lengths[:, None]
    features = {f"{name}_features": torch.randn(2, 5, dim, generator=generator).masked_fill(~mask[..., None], 0)
                for name, dim in FEATURE_DIMENSIONS.items()}
    speaker = torch.arange(5).remainder(2).expand(2, -1).masked_fill(~mask, -1)
    batch = {**features, "lengths": lengths, "attention_mask": mask, "speaker_ids_int": speaker}
    prefix = {key: value[:, :2] for key, value in batch.items() if key != "lengths"}
    prefix["lengths"] = lengths.clamp_max(2)
    future = {**batch, **{key: torch.where(mask[..., None] & (torch.arange(5)[None, :, None] >= 2),
                                        value * -4 + 5, value)
                         for key, value in features.items()}}
    future["speaker_ids_int"] = torch.where(mask & (torch.arange(5) >= 2), 1 - speaker, speaker)
    reordered = {key: value.flip(0) for key, value in batch.items()}
    padded = {**batch, **{key: F.pad(value, (0, 0, 0, 2)) for key, value in features.items()},
              "attention_mask": F.pad(mask, (0, 2)), "speaker_ids_int": F.pad(speaker, (0, 2), value=-1)}
    probes = {}
    for name, inputs in (("baseline", batch), ("prefix", prefix), ("future", future),
                         ("batch_order", reordered), ("padding", padded)):
        output = model(**{key: value.to(device) for key, value in inputs.items()})
        if not torch.equal(output.mask, inputs["attention_mask"].to(device)) or torch.triu(output.adjacency).any():
            raise ValueError(f"causal probe failed mask or past-only adjacency: {name}")
        probes[name] = (output.logits_ta, output.layer_states[0], output.layer_states[4], output.mask)
    if model.training or any(p.requires_grad or p.grad is not None for p in model.parameters()):
        raise ValueError("F0 freeze or gradient contract failed")
    return probes


def verify_causal_probes(head, probes, threshold, active_row):
    """Compare prefixes and exercise an active gate even on synthetic F0 probes."""
    comparisons = ("prefix", "future", "batch_order", "padding")
    if set(probes) != {"baseline", *comparisons}:
        raise ValueError("incomplete causal/invariance probes")

    def evaluate(inputs):
        logits, h0, h4, valid = inputs
        result = infer(head, logits, h0, h4, valid, threshold)
        forced = active_row.expand_as(logits)
        gated_result = infer(head, forced, h0, h4, valid, threshold)
        verify_writeback(logits, valid, result)
        verify_writeback(forced, valid, gated_result)
        if not torch.equal(gated_result.selected, valid):
            raise ValueError("active-gate causal probe did not exercise every valid row")
        return (logits, h0, h4, result.mad, result.margin, result.output.delta, result.output.logits,
                gated_result.output.delta, gated_result.output.logits), (
                    valid, result.top1, result.top2, result.selected, result.output.written,
                    result.output.logits.argmax(-1), gated_result.output.written, gated_result.output.logits.argmax(-1))

    reference_values, reference_exact = evaluate(probes["baseline"])
    for name in comparisons:
        values, exact = evaluate(probes[name])
        width = 2 if name in ("prefix", "future") else reference_exact[0].shape[1]
        if name == "batch_order":
            values, exact = (tuple(value.flip(0) for value in series) for series in (values, exact))
        if (any(not torch.allclose(base[:, :width], value[:, :width], rtol=RTOL, atol=ATOL)
                for base, value in zip(reference_values, values))
                or any(not torch.equal(base[:, :width], value[:, :width])
                       for base, value in zip(reference_exact, exact))):
            raise ValueError(f"causal/invariance probe failed: {name}")
    return {"status": "PASS", "comparisons": list(comparisons),
            "forced_active_gate": True, "synthetic_f0_forwards": len(probes), "rtol": RTOL, "atol": ATOL}

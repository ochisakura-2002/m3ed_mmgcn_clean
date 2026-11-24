"""Train two H1 scalars and verify their fixed-endpoint causal execution."""

from collections import Counter

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from datasets.iemocap.tav_full_audit_dataset import collate, require_split
from models.multidag_cl.history_attenuation import HistoryAttenuation, PARAMETER_COUNT
from models.multidag_cl.paper_reimplementation.contracts import SpeakerRelations
from scripts.runtime.multidag_cl_ta_v0.formal import json_identity
from scripts.runtime.multidag_cl_ta_v0.training import masked_cross_entropy, seed_everything
from scripts.runtime.multidag_cl_tav_full_audit_v1.runtime import MODEL_INPUTS

from .pairwise_preflight_inputs import FEATURE_DIMENSIONS, SEED, FrozenSplit


EPOCHS = 20
ADAM = {"lr": 3e-3, "betas": (0.9, 0.999), "weight_decay": 0}
PARAMETERS = ("raw_lambda_same", "raw_lambda_diff")
RTOL, ATOL = 1e-5, 1e-6


def lambda_values(attenuation):
    return dict(zip(("lambda_same", "lambda_diff"), (value.item() for value in attenuation.lambdas())))


def optimizer_step_gate(device):
    """Check PyTorch's boundary derivative using the H1 module itself."""
    attenuation = HistoryAttenuation().to(device)
    operator = torch.ones((1, 3, 3), device=device)
    edges = torch.zeros_like(operator, dtype=torch.bool)
    edges[0, 1, 0] = edges[0, 2, 0] = True
    same = edges.clone()
    same[0, 2, 0] = False
    relations = SpeakerRelations(same, edges & ~same)
    h0 = torch.tensor([[[1., 0.], [-1., 0.], [0., 1.]]], device=device)
    optimizer = torch.optim.Adam(attenuation.parameters(), **ADAM)
    attenuation(operator, h0, edges, relations)[edges].sum().backward()
    optimizer.step()
    values = lambda_values(attenuation)
    if not all(0 < value <= 0.5 for value in values.values()):
        raise ValueError("H1_PARAMETERIZATION = BLOCKED: clamp boundary Adam step did not move both scalars")
    return values


def freeze_snapshot(model):
    return {name: tensor.detach().cpu().clone() for name, tensor in model.f0.state_dict().items()
            if torch.is_tensor(tensor)}


def check_frozen(model, snapshot=None):
    trainable = tuple((name, p.numel()) for name, p in model.named_parameters() if p.requires_grad)
    if (trainable != tuple((f"attenuation.{name}", 1) for name in PARAMETERS)
            or len(trainable) != PARAMETER_COUNT
            or any(module.training for module in model.f0.modules())
            or any(p.grad is not None for p in model.f0.parameters())):
        raise ValueError("F0 must stay in eval mode with frozen parameters and no gradients")
    if snapshot is not None:
        state = {name: value for name, value in model.f0.state_dict().items() if torch.is_tensor(value)}
        if state.keys() != snapshot.keys() or any(not torch.equal(value.cpu(), snapshot[name])
                                                  for name, value in state.items()):
            raise ValueError("F0 parameters or buffers changed")


def model_inputs(batch, device):
    return {key: batch[key].to(device) for key in MODEL_INPUTS}


def check_output(output, inputs):
    mask = inputs["attention_mask"]
    width = mask.shape[1]
    positions = torch.arange(width, device=mask.device)
    expected = (positions[:, None] > positions[None, :]) & mask[:, :, None] & mask[:, None, :]
    if (not torch.equal(output.mask, mask) or not torch.equal(output.adjacency, expected)
            or torch.count_nonzero(output.logits_ta[~mask])):
        raise ValueError("strict past-all DAG, padding or valid mask changed")
    if not torch.isfinite(output.logits_ta).all():
        raise ValueError("nonfinite logits")


def require_exact(f0, candidate, label):
    if (not torch.equal(f0.logits_ta, candidate.logits_ta)
            or not torch.equal(f0.logits_ta.argmax(-1), candidate.logits_ta.argmax(-1))
            or not torch.equal(f0.adjacency, candidate.adjacency)
            or not torch.equal(f0.mask, candidate.mask)):
        raise ValueError(f"{label} failed exact F0 identity")


def batch_identity(batch, split, *, certified_rows=None):
    require_split(split)
    examples = tuple(zip(batch["dialogue_ids"], batch["utterance_ids"]))
    mask = batch["attention_mask"]
    labels = batch["labels"][mask].tolist()
    keys = [(uid, did) for did, utterances in examples for uid in utterances]
    if (batch["split"] != split or len(keys) != len(labels) or (batch["labels"][~mask] != -100).any()
            or tuple(map(len, batch["utterance_ids"])) != tuple(batch["lengths"].tolist())):
        raise ValueError(f"{split} split, population or padding mismatch")
    population = [(uid, did, label) for (uid, did), label in zip(keys, labels)]
    if certified_rows is not None:
        for uid, did, label in population:
            row = certified_rows[uid]
            if row["dialogue_id"] != did or int(row["benchmark_label_id"]) != label:
                raise ValueError("Train gold/canonical dialogue identity changed")
    return examples, population


def loader(dataset, split, batch_size, *, generator=None):
    require_split(split)
    if dataset.split != split:
        raise ValueError("dataset split differs from requested Train/Dev")
    return DataLoader(dataset, batch_size=batch_size, shuffle=generator is not None,
                      generator=generator, collate_fn=collate, num_workers=0)


def train_fixed(model, train, *, device, batch_size):
    """The only formal optimization loop; its input is exclusively Train."""
    check_frozen(model)
    if not model.history_attenuation_enabled or any(parameter.item() != 0 for parameter in model.attenuation.parameters()):
        raise ValueError("formal H1 must start enabled at zero")
    seed_everything(device, SEED)
    generator = torch.Generator().manual_seed(SEED)
    iterator = loader(train, "train", batch_size, generator=generator)
    expected = Counter()
    for did in train.dialogue_ids:
        ids = tuple(item[2] for item in train.ordered[did])
        expected[train.rows[ids[0]]["dialogue_id"], ids] += 1
    expected_targets = sum(len(ids) for _, ids in expected.elements())
    parameters = tuple(model.attenuation.parameters())
    optimizer = torch.optim.Adam(parameters, **ADAM)
    history = []
    for epoch in range(1, EPOCHS + 1):
        tape, seen, total, loss_sum = [], Counter(), 0, 0.0
        for batch in iterator:
            examples, _ = batch_identity(batch, "train", certified_rows=train.rows)
            tape.append(json_identity(examples))
            seen.update(examples)
            inputs = model_inputs(batch, device)
            optimizer.zero_grad(set_to_none=True)
            output = model(**inputs)
            check_output(output, inputs)
            loss, count = masked_cross_entropy(output.logits_ta, batch["labels"].to(device), output.mask)
            loss.backward()
            check_frozen(model)
            if any(value.grad is None or not torch.isfinite(value.grad).all() for value in parameters):
                raise ValueError(f"missing or nonfinite H1 scalar gradient at epoch {epoch}")
            optimizer.step()
            if any(not torch.isfinite(value) for value in parameters):
                raise ValueError(f"nonfinite raw lambda at epoch {epoch}")
            total += count
            loss_sum += loss.item() * count
        if seen != expected or total != expected_targets:
            raise ValueError(f"Train population mismatch at epoch {epoch}")
        history.append({"epoch": epoch, "train_loss": loss_sum / total, "train_targets": total,
                        "train_batch_tape_sha256": json_identity(tape), "optimizer_steps": len(tape),
                        **dict(zip(PARAMETERS, (value.item() for value in parameters))),
                        **lambda_values(model.attenuation)})
    return history


@torch.no_grad()
def collect_dev(model, dev, *, source, device, batch_size, identity=False, observe=None):
    logits, population, batch_hashes = [], [], []
    expected_batches = source.manifest["batch_pairing"]["dev"]["batch_sha256"]
    for batch in loader(dev, "dev", batch_size):
        examples, rows = batch_identity(batch, "dev")
        digest = json_identity(examples)
        if len(batch_hashes) >= len(expected_batches) or digest != expected_batches[len(batch_hashes)]:
            raise ValueError("Dev batch tape differs from the historical F0")
        batch_hashes.append(digest)
        population.extend(rows)
        inputs = model_inputs(batch, device)
        if identity:
            output = model.f0(**inputs)
            require_exact(output, model(history_attenuation_enabled=False, **inputs), "H1 bypass")
            require_exact(output, model(**inputs), "H1 zero-lambda")
        else:
            output = model(observe=observe, **inputs)
        check_output(output, inputs)
        logits.append(output.logits_ta[output.mask].cpu().numpy())
    if batch_hashes != expected_batches or tuple(population) != source.population:
        raise ValueError("incomplete or changed Dev tape or gold population")
    return FrozenSplit(np.concatenate(logits), tuple(population))


@torch.no_grad()
def verify_causal(model, device):
    """Exercise H1 at positive lambdas on synthetic dialogues and restore them."""
    generator = torch.Generator().manual_seed(SEED)
    lengths = torch.tensor([5, 3, 1], device=device)
    positions = torch.arange(5, device=device)
    mask = positions < lengths[:, None]
    speakers = positions.remainder(2).expand(3, -1).masked_fill(~mask, -1)
    features = {f"{name}_features": torch.randn(3, 5, width, generator=generator).to(device)
                .masked_fill(~mask[..., None], 0) for name, width in FEATURE_DIMENSIONS.items()}
    inputs = {**features, "lengths": lengths, "attention_mask": mask, "speaker_ids_int": speakers}
    future_mask = (positions >= 2) & mask
    future = {**inputs, **{name: torch.where(future_mask[..., None], -4 * value + 5, value)
                          for name, value in features.items()},
              "speaker_ids_int": torch.where(future_mask, 1 - speakers, speakers)}
    prefix = {**{key: value[:, :2] for key, value in inputs.items() if key != "lengths"},
              "lengths": lengths.clamp_max(2)}
    padded = {**inputs, **{key: F.pad(value, (0, 0, 0, 2)) for key, value in features.items()},
              "attention_mask": F.pad(mask, (0, 2)), "speaker_ids_int": F.pad(speakers, (0, 2), value=-1)}
    single = {key: value[-1:, :1] if key != "lengths" else value[-1:] for key, value in inputs.items()}
    params = tuple(model.attenuation.parameters())
    saved = tuple(parameter.clone() for parameter in params)
    try:
        for parameter, value in zip(params, (0.25, 0.5)):
            parameter.fill_(value)
        base = model(**inputs)
        check_output(base, inputs)
        require_exact(model.f0(**inputs), model(history_attenuation_enabled=False, **inputs), "active H1 bypass")
        require_exact(model.f0(**single), model(**single), "zero-history H1")
        for name, batch, width in (("future", future, 2), ("prefix", prefix, 2), ("padding", padded, 5)):
            actual = model(**batch)
            check_output(actual, batch)
            logits, reference = actual.logits_ta[:, :width], base.logits_ta[:, :width]
            if (not torch.allclose(logits, reference, rtol=RTOL, atol=ATOL)
                    or not torch.equal(logits.argmax(-1), reference.argmax(-1))
                    or not torch.equal(actual.mask[:, :width], base.mask[:, :width])):
                raise ValueError(f"H1 {name} causal invariance failed")
    finally:
        for parameter, value in zip(params, saved):
            parameter.copy_(value)
    return {"status": "PASS", "probes": ["future", "prefix", "padding", "zero_history", "active_bypass"],
            "rtol": RTOL, "atol": ATOL}

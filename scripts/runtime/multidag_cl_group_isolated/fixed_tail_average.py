"""Authenticate and average the four fixed CE checkpoints of one Protocol B fold."""

from collections.abc import Mapping
from copy import deepcopy
from hashlib import sha256
from io import BytesIO
import json
import re

import torch

from scripts.models.multidag_cl.group_isolated import run_identity, steps_per_epoch
from scripts.runtime.multidag_cl_group_isolated import protocol
from scripts.runtime.multidag_cl_group_isolated.evaluation import checkpoint_path
from scripts.runtime.multidag_cl_group_isolated.protocol import (
    ENDPOINT, G1_EPSILON, G1_SCHEMA, HOLD, PARAMETERS, PROJECTION_FIELDS, SCHEMA, SPLITS, validate_hold,
)
from scripts.runtime.multidag_cl_ta_v0.formal import json_identity


AVERAGING_EPOCHS = (21, 24, 27, ENDPOINT)
EXTRA_STATE = "_extra_state"


def authenticated_run(run_dir, config, frozen, raw):
    """Validate the manifest against the canonical recipe and frozen fold, without opening weights."""
    path = run_dir / "manifests" / "run_manifest.json"
    payload = path.read_bytes()
    manifest = json.loads(payload)
    validate_hold(manifest)
    session = config["held_out_session"]
    fold = frozen["folds"][session]
    formal = frozen["formal_f0"]
    projection = {
        **frozen["certified_projection"], **HOLD, "splits": list(SPLITS),
        "split_counts": protocol.TRAIN_DEV_COUNTS, "source_data_identity": formal["data_identity"],
    }
    # This is the exact identity constructed by load_fold, also checked against its real output at inference.
    data = {
        "schema": SCHEMA, "held_out_session": session, "fold": fold,
        "projection": {key: projection[key] for key in PROJECTION_FIELDS},
        "formal_f0_population": formal["population"],
        "source_protocol_b_sha256": frozen["source_protocol_b_sha256"], **HOLD,
    }
    steps = steps_per_epoch(fold["fit"], raw)
    expected = {
        "status": "completed", "last_completed_epoch": ENDPOINT, "endpoint_epoch": ENDPOINT,
        "checkpoint_selection": "fixed_epoch", "synthetic_only": False, "seed": config["seed"],
        "run_identity": run_identity(config, raw, data), "run_id": config["run_id"],
        "steps_per_epoch": steps, "optimizer_steps": ENDPOINT * steps, "f0_recipe": raw,
        "data_identity": data, "parameter_count": PARAMETERS, "TEST_TOUCHED": "NO",
    }
    if config["schema"] == G1_SCHEMA:
        expected.update(fit_objective={"name": "uniform_label_smoothed_ce", "label_smoothing": G1_EPSILON},
                        evaluation_objective="hard_label_nll")
    elif config["schema"] != SCHEMA or manifest.get("fit_objective") is not None:
        raise ValueError(f"expected the canonical hard-label CE arm for {session}")
    if run_dir.name != config["run_id"] or any(manifest.get(k) != v for k, v in expected.items()):
        raise ValueError(f"run_manifest differs from the completed frozen {session} arm")
    hashes = manifest["checkpoint_sha256"]
    if (set(hashes) != set(map(str, range(1, ENDPOINT + 1)))
            or any(not isinstance(digest, str) or not re.fullmatch(r"[a-f0-9]{64}", digest)
                   for digest in hashes.values())):
        raise ValueError(f"incomplete or invalid checkpoint SHA-256 inventory for {session}")
    return manifest, sha256(payload).hexdigest()


def parameter_schema(model):
    state = model.state_dict()
    parameters = dict(model.named_parameters())
    buffers = list(dict(model.named_buffers()))
    if buffers or any(isinstance(module, torch.nn.modules.batchnorm._BatchNorm) for module in model.modules()):
        raise ValueError(f"unhandled model buffers or BatchNorm state: {buffers}")
    if set(state) != {*parameters, EXTRA_STATE} or not isinstance(state[EXTRA_STATE], dict):
        raise ValueError("unexpected model state outside the formal parameters and _extra_state")
    if any(value.dtype != torch.float32 or value.layout != torch.strided for value in parameters.values()):
        raise ValueError("unexpected dtype or layout in the frozen FP32 parameters")
    if sum(value.numel() for value in parameters.values()) != PARAMETERS:
        raise ValueError("formal F0 parameter count changed")
    return state


def validate_state(state, reference):
    if not isinstance(state, Mapping) or state.keys() != reference.keys():
        raise ValueError("checkpoint parameter keys differ from the model schema")
    if EXTRA_STATE in reference and (
        not isinstance(state[EXTRA_STATE], dict)
        or json_identity(state[EXTRA_STATE]) != json_identity(reference[EXTRA_STATE])
    ):
        raise ValueError("checkpoint _extra_state differs from the frozen model schema or recipe")
    for name, expected in reference.items():
        if name == EXTRA_STATE:
            continue
        value = state[name]
        if (not isinstance(value, torch.Tensor) or not value.is_floating_point() or value.device.type != "cpu"
                or value.layout != torch.strided or value.shape != expected.shape or value.dtype != expected.dtype):
            raise ValueError(f"unexpected checkpoint tensor schema: {name}")
        if not torch.isfinite(value).all():
            raise ValueError(f"nonfinite checkpoint parameter: {name}")


@torch.no_grad()
def equal_average(states, reference):
    """Stream one input at a time and retain exact single/identical-input behavior."""
    keys = [name for name in reference if name != EXTRA_STATE]
    accumulator = {}
    identical = True
    count = 0
    for count, state in enumerate(states, start=1):
        validate_state(state, reference)
        if count == 1:
            accumulator = {name: state[name].clone() for name in keys}
        elif not identical or any(not torch.equal(state[name], accumulator[name]) for name in keys):
            if identical:
                # Expand the identical leading states once, on the first differing input.
                accumulator = {name: value.to(torch.float32).mul_(count - 1)
                               for name, value in accumulator.items()}
                identical = False
            for name in keys:
                accumulator[name].add_(state[name].to(torch.float32))
        del state
    if not count:
        raise ValueError("no states to average")
    if not identical:
        accumulator = {name: value.div_(count).to(reference[name].dtype) for name, value in accumulator.items()}
    if EXTRA_STATE in reference:
        accumulator[EXTRA_STATE] = deepcopy(reference[EXTRA_STATE])
    validate_state(accumulator, reference)
    return accumulator


def source_inventory(run_dir, manifest):
    return [{"epoch": epoch, "path": checkpoint_path(run_dir, epoch).relative_to(run_dir).as_posix(),
             "sha256": manifest["checkpoint_sha256"][str(epoch)],
             "optimizer_steps": epoch * manifest["steps_per_epoch"]} for epoch in AVERAGING_EPOCHS]


def read_checkpoint(run_dir, source, identity):
    path = run_dir / source["path"]
    if not path.resolve().is_relative_to(run_dir.resolve()):
        raise ValueError(f"checkpoint path escapes its source run: {path}")
    payload = path.read_bytes()
    epoch = source["epoch"]
    if sha256(payload).hexdigest() != source["sha256"]:
        raise ValueError(f"checkpoint SHA-256 mismatch at epoch {epoch}: {path}")
    checkpoint = torch.load(BytesIO(payload), map_location="cpu", weights_only=True)
    if (checkpoint["identity"] != identity or checkpoint["epoch"] != epoch
            or checkpoint["optimizer_steps"] != source["optimizer_steps"]):
        raise ValueError(f"checkpoint run, epoch or optimizer-step identity mismatch at epoch {epoch}")
    return checkpoint["model_state_dict"]


def load_fixed_tail(run_dir, manifest, model):
    if manifest["run_identity"]["schema"] != SCHEMA:
        raise ValueError("fixed tail average accepts only one CE trajectory")
    reference = parameter_schema(model)
    sources = source_inventory(run_dir, manifest)
    state = equal_average((read_checkpoint(run_dir, source, manifest["run_identity"]) for source in sources), reference)
    model.load_state_dict(state, strict=True)
    return state, sources

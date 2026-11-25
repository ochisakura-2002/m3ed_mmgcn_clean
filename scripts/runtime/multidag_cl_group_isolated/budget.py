"""G3 fit-side splits, fixed optimizer trajectories and hard-NLL selection."""

from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
from itertools import combinations
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from scripts.models.multidag_cl.group_isolated import build_runtime, steps_per_epoch
from scripts.runtime.multidag_cl_fusion.runtime import state_digest, training_rng_identity
from scripts.runtime.multidag_cl_group_isolated.evaluation import hard_label_scores, validate_predictions
from scripts.runtime.multidag_cl_group_isolated.fixed_tail_average import parameter_schema, validate_state
from scripts.runtime.multidag_cl_group_isolated.protocol import FoldView, HOLD, SEED, SESSIONS, describe
from scripts.runtime.multidag_cl_paper_reimplementation.checkpoint import save_checkpoint_atomic
from scripts.runtime.multidag_cl_ta_v0.formal import json_identity, sha256_file
from scripts.runtime.multidag_cl_ta_v0.training import PROTOCOL, cpu_snapshot


GROUP = "g3_fit_side_budget_ct_multidag"
SCHEMA = "ct_multidag_g3_fit_side_budget_v1"
HOLD_RECORD = {**HOLD, "TEST_TOUCHED": "NO"}


def candidates(outer):
    if outer not in SESSIONS:
        raise ValueError("invalid G3 outer session")
    increment = 24 if outer == "Ses04" else 18
    return tuple(increment * i for i in range(1, 6))


def assignments():
    for outer in SESSIONS:
        fit = tuple(s for s in SESSIONS if s != outer)
        for validation in fit:
            yield outer, validation, tuple(s for s in fit if s != validation)


def plan(frozen, raw):
    sizes = {s: frozen["folds"][s]["held_out"]["dialogues"] for s in SESSIONS}
    steps = {pair: steps_per_epoch({"dialogues": sum(sizes[s] for s in pair)}, raw)
             for pair in combinations(SESSIONS, 2)}
    records = []
    for outer, validation, pair in assignments():
        grid = candidates(outer)
        if grid[-1] != 30 * min(value for key, value in steps.items() if outer not in key):
            raise ValueError("G3 frozen candidate grid differs from the certified dialogue counts")
        records.append({"outer": outer, "validation": validation, "fit": list(pair),
                        "steps_per_epoch": steps[pair], "candidates": list(grid)})
    return records


def subset(fit, entries, sessions, split):
    selected = set(sessions)
    rows = [row for row in entries if row[1][:5] in selected]
    view = FoldView((item for item in fit if item["dialogue_id"][:5] in selected), split)
    realized = [(uid, item["dialogue_id"], label) for item in view
                for uid, label in zip(item["utterance_ids"], item["labels"].tolist())]
    if realized != rows or set(describe(rows)["sessions"]) != selected:
        raise ValueError("G3 subset failed the authenticated population/order check")
    return view, rows


def qualify_shared(members):
    """Group only plans with identical fit identity and a matching budget prefix."""
    by_pair = {}
    for member in members:
        by_pair.setdefault(tuple(member["spec"]["fit"]), []).append(member)
    groups = list(by_pair.values())
    if len(groups) != 6 or any(len(group) != 2 for group in groups):
        raise ValueError("expected six G3 pairs with two outer consumers each")
    for first, second in groups:
        shorter = min(len(first["tape"]), len(second["tape"]))
        if first["identity"] != second["identity"] or first["tape"][:shorter] != second["tape"][:shorter]:
            return [[member] for member in members], "NOT_SUPPORTED"
    return groups, "VERIFIED"


def identity_collate(dialogues):
    return {"dialogue_ids": [item["dialogue_id"] for item in dialogues],
            "utterance_ids": [item["utterance_ids"] for item in dialogues]}


def epoch_batches(runtime, rows, epoch, *, identity_only=False):
    expected = {}
    for uid, did, _ in rows:
        expected.setdefault(did, []).append(uid)
    seen = set()
    loader = runtime.loader("train", epoch=epoch)
    if identity_only:
        # Keep the real DataLoader, generator and sampler. Skip feature padding for preflight.
        loader.collate_fn = identity_collate
    for batch in loader:
        order = list(zip(batch["dialogue_ids"], batch["utterance_ids"]))
        for did, ids in order:
            if did in seen or tuple(expected.get(did, ())) != ids:
                raise ValueError("G3 fit tape contains a duplicate, incomplete or foreign dialogue")
            seen.add(did)
        yield batch, {"dialogue_ids": [did for did, _ in order], "dialogue_exposure": len(order),
                      "valid_targets": sum(len(ids) for _, ids in order), "batch_sha256": json_identity(order)}
    if seen != expected.keys():
        raise ValueError("G3 epoch is missing fit dialogues")


def batch_tape(runtime, rows, steps_per_epoch, stop):
    tape = []
    for epoch in range(1, (stop - 1) // steps_per_epoch + 2):
        batches = list(epoch_batches(runtime, rows, epoch, identity_only=True))
        if len(batches) != steps_per_epoch:
            raise ValueError("G3 optimizer steps per epoch changed")
        tape.extend(record for _, record in batches)
    return tape[:stop]


def position(tape, step, steps_per_epoch):
    epoch, batch = divmod(step - 1, steps_per_epoch)
    prefix = tape[:step]
    if step <= 0 or len(prefix) != step:
        raise ValueError("G3 checkpoint step is outside the fit tape")
    return {
        "optimizer_steps": step, "epoch": epoch + 1, "batch_in_epoch": batch + 1,
        "end_of_epoch": batch + 1 == steps_per_epoch,
        "dialogue_exposure": sum(batch["dialogue_exposure"] for batch in prefix),
        "valid_target_exposure": sum(batch["valid_targets"] for batch in prefix),
        "batch_tape_sha256": json_identity(prefix),
    }


@dataclass
class Prepared:
    runtime: object
    rows: list
    identity: dict
    tape: list
    initial_rng: dict
    initialization: dict


def prepare(raw, fit, rows, projection, source, stop, *, smoothing=0.0, device, synthetic=False):
    runtime, initialization = build_runtime(raw, {"train": fit}, device, synthetic=synthetic,
                                            train_label_smoothing=smoothing)
    rng = training_rng_identity(device)
    parameters = parameter_schema(runtime.model)
    full_init = state_digest({key: value.detach().cpu() for key, value in parameters.items()
                              if isinstance(value, torch.Tensor)})
    population = describe(rows)
    steps = steps_per_epoch(population, raw)
    tape = batch_tape(runtime, rows, steps, stop)
    if training_rng_identity(device) != rng:
        raise ValueError("G3 fit tape construction consumed global training RNG")
    identity = {
        "schema": SCHEMA, "seed": SEED, "population": population, "projection": projection,
        "recipe_sha256": json_identity(raw), "initial_state_sha256": full_init,
        "initial_rng": rng, "steps_per_epoch": steps, "source": source,
        "optimizer": {"name": type(runtime.optimizer).__name__,
                      "groups": [{key: value for key, value in group.items() if key != "params"}
                                 for group in runtime.optimizer.param_groups]},
        "gradient_clip_norm": PROTOCOL["gradient_clip_norm"], "scheduler": None,
        "label_smoothing": smoothing, "validation_controls_updates": False,
    }
    return Prepared(runtime, rows, identity, tape, rng, initialization)


def sync(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def train_fixed(directory, prepared, save_steps, *, verify_epoch=None):
    """Fit only. The runtime must be freshly prepared immediately before this call."""
    runtime, tape, identity = prepared.runtime, prepared.tape, prepared.identity
    save_steps = set(save_steps)
    stop = len(tape)
    if (not save_steps or min(save_steps) < 1 or max(save_steps) != stop or runtime.optimizer_steps
            or training_rng_identity(runtime.device) != prepared.initial_rng):
        raise ValueError("G3 training requires a fresh runtime and a fixed final save point")
    directory.mkdir(parents=True, exist_ok=False)
    steps = identity["steps_per_epoch"]
    checkpoints, update_seconds = {}, 0.0
    sync(runtime.device)
    start = perf_counter()
    for epoch in range(1, (stop - 1) // steps + 2):
        for batch, record in epoch_batches(runtime, prepared.rows, epoch):
            step = runtime.optimizer_steps + 1
            if record != tape[step - 1]:
                raise ValueError("observed G3 batch differs from the frozen batch tape")
            tick = perf_counter()
            result = runtime.train_step(batch)
            sync(runtime.device)
            update_seconds += perf_counter() - tick
            if result["optimizer_steps"] != step or result["valid_target_count"] != record["valid_targets"]:
                raise ValueError("G3 optimizer update/exposure mismatch")
            if verify_epoch and step % steps == 0:
                verify_epoch(epoch, runtime.model)
            if step in save_steps:
                path = directory / f"update_{step:03d}.pt"
                state = cpu_snapshot(runtime.model.state_dict())
                validate_state(state, runtime.model.state_dict())
                save_checkpoint_atomic(path, {"trajectory_sha256": json_identity(identity),
                                              "optimizer_steps": step, "model_state_dict": state})
                checkpoints[str(step)] = {"path": path.name, "sha256": sha256_file(path),
                                          **position(tape, step, steps)}
            if step == stop:
                return {**HOLD_RECORD, "identity": identity, "updates": stop, "checkpoints": checkpoints,
                        "fit_wall_seconds": perf_counter() - start, "update_wall_seconds": update_seconds,
                        "evaluation_calls_during_training": 0, "batch_tape": tape}
    raise ValueError("G3 training ended before its fixed update budget")


def score_state(runtime, state, validation, entries):
    """Score a separate validation-only dataset with the inherited hard-label evaluator."""
    validate_state(state, runtime.model.state_dict())
    runtime.model.load_state_dict(state, strict=True)
    runtime.dataset_factory = {"dev": validation}.__getitem__
    with torch.inference_mode():
        report = runtime.evaluate(include_predictions=True, prediction_details=True)
    if runtime.optimizer_steps or runtime.optimizer.state or any(p.grad is not None for p in runtime.model.parameters()):
        raise ValueError("G3 scoring performed a training update")
    rows = report["predictions"]
    validate_predictions(rows, entries)
    return rows


def load_state(directory, identity, record):
    path = Path(directory) / record["path"]
    if path.parent != Path(directory) or not path.resolve().is_relative_to(Path(directory).resolve()):
        raise ValueError("G3 checkpoint escapes its trajectory")
    payload = path.read_bytes()
    if sha256(payload).hexdigest() != record["sha256"]:
        raise ValueError("G3 checkpoint SHA-256 mismatch")
    checkpoint = torch.load(BytesIO(payload), map_location="cpu", weights_only=True)
    if (checkpoint["trajectory_sha256"] != json_identity(identity)
            or checkpoint["optimizer_steps"] != record["optimizer_steps"]):
        raise ValueError("G3 checkpoint trajectory or optimizer-step identity changed")
    return checkpoint["model_state_dict"]


def hard_nll(rows):
    logits = np.asarray([row["logits"] for row in rows], dtype=np.float64)
    labels = np.asarray([row["true_label"] for row in rows], dtype=np.int64)
    _, scores = hard_label_scores(logits, labels)
    return float(scores["nll"].mean())


def select(outer, tables, frozen):
    grid = candidates(outer)
    expected = set(SESSIONS) - {outer}
    by_session = {table["validation"]: table for table in tables}
    if len(tables) != 3 or by_session.keys() != expected:
        raise ValueError("G3 requires exactly three independent inner validation sessions")
    for session, table in by_session.items():
        if (table["outer"] != outer or table["fit"] != sorted(expected - {session})
                or table["validation_population"] != frozen["folds"][session]["held_out"]
                or [row["optimizer_steps"] for row in table["scores"]] != list(grid)
                or any(row["valid_targets"] != table["validation_population"]["utterances"]
                       or not np.isfinite(row["hard_nll"]) or row["hard_nll"] < 0 for row in table["scores"])):
            raise ValueError("G3 inner table violates its frozen population or scoring grid")
    means = []
    for index, step in enumerate(grid):
        nll = {s: by_session[s]["scores"][index]["hard_nll"] for s in sorted(expected)}
        means.append({"optimizer_steps": step, "session_nll": nll, "mean_session_nll": sum(nll.values()) / 3})
    return {"candidates": means, "selected_updates": min(means, key=lambda row: (
        row["mean_session_nll"], row["optimizer_steps"]))["optimizer_steps"]}

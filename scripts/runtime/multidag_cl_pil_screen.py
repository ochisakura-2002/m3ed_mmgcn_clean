"""Train-only targets, paired runtime and epoch transactions for PIL_SCREEN_001."""

from collections import defaultdict
from copy import deepcopy
import os
import random
import time

import torch
from torch import nn
from torch.nn import functional as F

from datasets.iemocap.tav_full_audit_dataset import collate
from models.multidag_cl.fusion.model import FusionTAV
from models.experimental.multidag_cl_pil_screen import C01FusionTAV, predecessors
from scripts.analysis.models.multidag_cl.original_dev_analysis import FitView
from scripts.runtime.multidag_cl_fusion.original_dev_alignment import STEPS
from scripts.runtime.multidag_cl_fusion.runtime import (
    FusionAuditRuntime, paired_model, shared_state, state_digest, training_rng_identity,
)
from scripts.runtime.multidag_cl_group_isolated.budget import identity_collate
from scripts.runtime.multidag_cl_group_isolated.protocol import ENDPOINT, HOLD, SEED, read_json
from scripts.runtime.multidag_cl_paper_reimplementation.checkpoint import load_checkpoint, save_checkpoint_atomic
from scripts.runtime.multidag_cl_ta_v0.formal import json_identity, sha256_file, write_json_atomic
from scripts.runtime.multidag_cl_ta_v0.training import cpu_snapshot
from scripts.runtime.multidag_cl_tav_full_audit_v1.runtime import AuditRuntime, MODEL_INPUTS, SyntheticDialogues


ARMS = ("C01-CURRENT", "C01-PREV", "C07-NULL", "C07-TRUE")
CONTROL = "PIL-CTRL"
BASE_PARAMETERS = 1_638_306
EXPECTED_PARAMETERS = dict(zip((CONTROL, *ARMS), (BASE_PARAMETERS, 1_698_306, 1_698_306, 1_639_807, 1_639_807)))
AUX_WEIGHT = 0.1


class InvalidRun(ValueError):
    """An experimental identity or numerical gate failed."""


def dialogue_history(item):
    ids = item["utterance_ids"]
    speakers = item["speaker_ids_int"].tolist()
    labels = item["labels"].tolist()
    if (not ids or len(ids) != len(speakers) or len(ids) != len(labels)
            or any(type(uid) is not str or not uid for uid in ids)
            or any(type(label) is not int or not 0 <= label < 6 for label in labels)
            or any(type(speaker) is not int or speaker < 0 for speaker in speakers)):
        raise InvalidRun("invalid dialogue ID, turn order, labels or speaker metadata")
    last = {}
    for i, (uid, speaker, label) in enumerate(zip(ids, speakers, labels)):
        previous = last.get(speaker)
        yield {"utterance_id": uid, "dialogue_id": item["dialogue_id"], "speaker": speaker, "position": i,
               "true_label": label, "previous": previous}
        last[speaker] = (i, uid, label)


def build_target_tape(train, expected):
    if train.split != "train":
        raise InvalidRun("auxiliary targets require the original Train population")
    population, dialogues, targets = [], set(), {}
    strata = defaultdict(list)
    for index in range(len(train)):
        item = train[index]
        did = item["dialogue_id"]
        if item["split"] != "train" or did in dialogues:
            raise InvalidRun("wrong split or repeated Train dialogue")
        dialogues.add(did)
        for row in dialogue_history(item):
            uid, label, previous = row["utterance_id"], row["true_label"], row["previous"]
            population.append((uid, did, label))
            targets[uid] = [did, row["speaker"], row["position"], label, None, None, None]
            if previous is not None:
                z = int(label != previous[2])
                targets[uid][-3:] = [previous[1], z, z]
                strata[label].append(uid)
    if population != list(expected) or len(population) != len(targets):
        raise InvalidRun("wrong, duplicate or missing Train target IDs")
    rng = random.Random(10007)
    distribution, changed = [], 0
    for label in range(6):
        ids = sorted(strata[label])
        real = [targets[uid][-2] for uid in ids]
        null = real.copy()
        rng.shuffle(null)
        for uid, z in zip(ids, null):
            targets[uid][-1] = z
        changed += sum(a != b for a, b in zip(real, null))
        distribution.append({"label": label, "positives": sum(real), "negatives": len(real) - sum(real)})
    eligible = sum(len(ids) for ids in strata.values())
    return {"schema": "pil_c07_train_targets_v1_python_random_shuffle", "split": "train", "seed": 10007,
            "implementation": "ascending label then utterance_id, Python Random.shuffle once per stratum",
            "population_sha256": json_identity(population), "targets": targets,
            "mapping_sha256": json_identity(targets), "distribution_each_arm": distribution,
            "utterances": len(population), "eligible": eligible, "changed_count": changed,
            "changed_fraction": changed / eligible if eligible else 0.0}


def validate_target_tape(saved, expected, expected_sha):
    if json_identity(saved) != expected_sha or saved != expected or saved["split"] != "train":
        raise InvalidRun("Train auxiliary tape split, IDs, mapping or SHA mismatch")
    if not saved["changed_count"]:
        raise InvalidRun("fixed C07-NULL shuffle does not change any target")


class PILRuntime(FusionAuditRuntime):
    def __init__(self, raw, datasets, arm, *, experiment, targets=None, pairing=None, **options):
        if arm not in EXPECTED_PARAMETERS:
            raise InvalidRun("unapproved PIL arm")
        device = options["device"]
        model, reference, initialization = paired_model(raw, device=device)
        rng = training_rng_identity(device)
        if arm.startswith("C01"):
            with torch.random.fork_rng(devices=[]):
                model = C01FusionTAV(raw, arm)
            missing, unexpected = model.load_state_dict(
                {**reference.shared, "_extra_state": model.get_extra_state()}, strict=False)
            if set(missing) != {"side_weight"} or unexpected:
                raise InvalidRun("C01 cannot reload the frozen shared initialization")
            shared = {key: value for key, value in shared_state(model).items() if key != "side_weight"}
            if state_digest(shared) != initialization["shared_initial_state_sha256"]:
                raise InvalidRun("C01 shared initial tensors differ from F0")
        self.aux = None
        if arm.startswith("C07"):
            if (targets is None or json_identity(targets) != experiment["target_tape_sha256"]
                    or targets["split"] != "train" or not targets["changed_count"]):
                raise InvalidRun("missing or invalid Train-only auxiliary target tape")
            self.aux = nn.ParameterDict({"weight": nn.Parameter(torch.zeros(1500)),
                                         "bias": nn.Parameter(torch.zeros(1))}).to(device)
        super().__init__(model, datasets.__getitem__, initialization=initialization, f0_pairing=pairing, **options)
        if self.aux is not None:
            self.optimizer.param_groups[0]["params"].extend(self.aux.parameters())
        initialization["training_rng_identity"] = training_rng_identity(device)
        if rng != initialization["training_rng_identity"]:
            raise InvalidRun("candidate construction changed the shared training RNG")
        self.arm, self.experiment = arm, deepcopy(experiment)
        self.targets = targets["targets"] if self.aux is not None else None
        self.parameter_count = sum(p.numel() for p in self.training_parameters())
        if self.parameter_count != EXPECTED_PARAMETERS[arm]:
            raise InvalidRun("unexpected PIL parameter count")
        self.steps_per_epoch = (len(datasets["train"]) + 15) // 16
        self.history = []

    @property
    def identity(self):
        return {**super().identity, "pil_experiment": self.experiment}

    def training_parameters(self):
        return iter(self.optimizer.param_groups[0]["params"])

    def forward_batch(self, batch, **options):
        output, loss = super().forward_batch(batch, **options)
        if not torch.equal(output.mask, batch["attention_mask"].to(self.device)):
            raise InvalidRun("canonical target mask changed")
        return output, loss

    def auxiliary_targets(self, batch, previous):
        if batch["split"] != "train":
            raise InvalidRun("auxiliary loss requested outside Train")
        if (len(batch["dialogue_ids"]) != len(batch["utterance_ids"])
                or [len(ids) for ids in batch["utterance_ids"]] != batch["lengths"].tolist()):
            raise InvalidRun("incomplete C07 batch utterance ID coverage")
        targets, seen = [], set()
        for b, (did, ids) in enumerate(zip(batch["dialogue_ids"], batch["utterance_ids"])):
            positions = previous[b, :len(ids)].tolist()
            for i, (uid, p) in enumerate(zip(ids, positions)):
                row = self.targets.get(uid)
                prev_id = ids[p] if 0 <= p < i else None
                expected = [did, int(batch["speaker_ids_int"][b, i]), i, int(batch["labels"][b, i]), prev_id]
                if uid in seen or row is None or row[:5] != expected or not -1 <= p < i:
                    raise InvalidRun(f"invalid C07 Train target binding: {uid}")
                seen.add(uid)
                if p >= 0:
                    targets.append(row[-1 if self.arm == "C07-NULL" else -2])
        return targets

    def auxiliary_loss(self, batch, output):
        previous = predecessors(output.adjacency, batch["speaker_ids_int"].to(self.device), output.mask)
        targets = self.auxiliary_targets(batch, previous.cpu())
        rows, positions = torch.where(previous >= 0)
        if len(rows) != len(targets):
            raise InvalidRun("C07 eligibility and target population differ")
        if not targets:
            return output.h_ta.new_zeros(()), 0, 0
        difference = (output.h_ta[rows, positions] - output.h_ta[rows, previous[rows, positions]]).abs()
        logits = F.linear(difference, self.aux["weight"][None], self.aux["bias"]).squeeze(-1)
        loss = F.binary_cross_entropy_with_logits(logits, logits.new_tensor(targets), reduction="sum")
        return loss / output.mask.sum(), len(targets), sum(targets)

    def train_step(self, batch):
        start = time.perf_counter()
        self.optimizer.zero_grad(set_to_none=True)
        output, ce = self.forward_batch(batch, split="train")
        auxiliary, eligible, positive = self.auxiliary_loss(batch, output) if self.aux is not None else (None, 0, 0)
        loss = ce if auxiliary is None else ce + AUX_WEIGHT * auxiliary
        if not torch.isfinite(loss):
            raise InvalidRun("nonfinite PIL training objective")
        loss.backward()
        params = list(self.training_parameters())
        grads = [p.grad for p in params if p.grad is not None]
        if not grads or any(not torch.isfinite(g).all() for g in grads):
            raise InvalidRun("nonfinite/missing PIL gradient")
        added = list(self.aux.parameters()) if self.aux is not None else []
        if isinstance(self.model, C01FusionTAV):
            added = [self.model.side_weight]
        extra_ids = {id(p) for p in added}
        shared_gradients = [p.grad for p in params if id(p) not in extra_ids and p.grad is not None]
        added_gradients = [p.grad for p in added if p.grad is not None]

        def group_norm(gradients):
            return float(torch.linalg.vector_norm(torch.stack([g.norm() for g in gradients]))) if gradients else 0.0

        shared_norm, added_norm = group_norm(shared_gradients), group_norm(added_gradients)
        norm = float(torch.nn.utils.clip_grad_norm_(params, 5.0, error_if_nonfinite=True))
        self.optimizer.step()
        self.optimizer_steps += 1
        return {"step": self.optimizer_steps, "valid_targets": int(output.mask.sum()), "hard_ce": float(ce.detach()),
                "objective": float(loss.detach()), "aux_bce_per_valid": float(auxiliary.detach()) if auxiliary is not None else None,
                "eligible": eligible, "aux_positive": positive, "gradient_norm": norm, "clipped": norm > 5.0,
                "shared_gradient_norm": shared_norm, "added_gradient_norm": added_norm,
                "update_seconds": time.perf_counter() - start}

    def checkpoint_payload(self, epoch, *, epoch_complete=True):
        return {**super().checkpoint_payload(epoch, epoch_complete=epoch_complete),
                "aux_state": cpu_snapshot(self.aux.state_dict()) if self.aux is not None else None,
                "history": deepcopy(self.history), "batch_pairing": deepcopy(self.pairing),
                "initialization": deepcopy(self.initialization)}

    def validate_checkpoint(self, checkpoint):
        super().validate_checkpoint(checkpoint)
        epoch = checkpoint["epoch"]
        aux = checkpoint["aux_state"]
        if ((aux is None) != (self.aux is None)
                or checkpoint["git_head"] != self.git_head
                or checkpoint["initialization"] != self.initialization
                or checkpoint["optimizer_steps"] != self.steps_per_epoch * epoch
                or [record["epoch"] for record in checkpoint["history"]] != list(range(1, epoch + 1))
                or set(checkpoint["batch_pairing"]["train"]) != {str(i) for i in range(1, epoch + 1)}
                or checkpoint["batch_pairing"]["population"] != self.pairing["population"]):
            raise InvalidRun("PIL checkpoint progress, initialization or auxiliary identity mismatch")
        if aux is not None and (aux.keys() != self.aux.state_dict().keys()
                or any(value.shape != self.aux[key].shape or value.dtype != self.aux[key].dtype
                       or not torch.isfinite(value).all() for key, value in aux.items())):
            raise InvalidRun("invalid C07 auxiliary checkpoint tensors")
        if self.f0_pairing is not None:
            tape = checkpoint["batch_pairing"]
            if (tape["dev"] != self.f0_pairing["dev"]
                    or any(value != self.f0_pairing["train"][key] for key, value in tape["train"].items())):
                raise InvalidRun("PIL checkpoint does not match the original batch/RNG tape")

    def resume(self, path):
        checkpoint = super().resume(path)
        if self.aux is not None:
            self.aux.load_state_dict(checkpoint["aux_state"], strict=True)
        self.history = checkpoint["history"]
        self.pairing = checkpoint["batch_pairing"]
        return checkpoint

    def fit_metrics(self):
        train = FitView(self.dataset("train"))
        fit = torch.utils.data.DataLoader(train, batch_size=16, collate_fn=collate, num_workers=0,
                                         generator=torch.Generator().manual_seed(SEED))
        return self.evaluate(fit)["metrics"]

    def audit_sampler(self, expected):
        tape = {"train": {}, "dev": {}}
        for split, epochs in (("train", range(1, ENDPOINT + 1)), ("dev", (1,))):
            for epoch in epochs:
                loader = AuditRuntime.loader(self, split, epoch=epoch)
                # Iterating the DataLoader consumes its base seed before shuffling.
                loader.collate_fn = identity_collate
                orders = [list(zip(b["dialogue_ids"], b["utterance_ids"])) for b in loader]
                hashes = [json_identity(order) for order in orders]
                if expected is not None:
                    ref = expected["train"][str(epoch)] if split == "train" else expected["dev"]
                    if hashes != ref["batch_sha256"]:
                        raise InvalidRun(f"{split} epoch {epoch} sampler differs from the historical tape")
                if split == "train" and (len(hashes) != STEPS or [len(batch) for batch in orders] != [16] * 6 + [4]):
                    raise InvalidRun("frozen Train batch schedule changed")
                tape[split][str(epoch)] = [
                    {"dialogue_ids": [did for did, _ in order], "valid_targets": sum(len(ids) for _, ids in order),
                     "dialogue_utterance_sha256": digest} for order, digest in zip(orders, hashes)]
        return tape

    def export_deploy(self, path):
        state = cpu_snapshot(self.model.state_dict())
        with torch.random.fork_rng(devices=[]):
            deployed = C01FusionTAV(self.model.audit_config, self.arm) if self.arm.startswith("C01") else FusionTAV(self.model.audit_config)
        deployed.load_state_dict(state, strict=True)
        return save_checkpoint_atomic(path, state)


@torch.no_grad()
def device_parity(original, candidates):
    """Check zero-init equality on actual device kernels using only synthetic data."""
    item = dict(SyntheticDialogues("dev")[0])
    item["speaker_ids_int"] = torch.zeros_like(item["speaker_ids_int"])
    batch = collate([item])
    args = {key: batch[key].to(original.device) for key in MODEL_INPUTS}
    original.model.eval()
    canonical = original.model(**args)
    for runtime in candidates:
        runtime.model.eval()
        actual = runtime.model(**args)
        if any(not torch.equal(getattr(actual, key), getattr(canonical, key)) for key in
               ("mask", "adjacency", "h_ta", "logits_ta", "prob_ta")):
            raise InvalidRun(f"{runtime.arm} zero-init supplied-feature parity failed on {runtime.device}")
    return {"same_shape_zero_init": "EXACT_VALUES", "dtype": str(canonical.logits_ta.dtype),
            "device": str(original.device), "synthetic_only": True}


def train_arm(runtime, run, *, epochs=ENDPOINT, resume=False):
    """Only the atomic last checkpoint commits an epoch and its evidence."""
    if not runtime.synthetic_only and (os.name != "posix" or runtime.device.type != "cuda"
                                       or runtime.steps_per_epoch != STEPS or epochs != ENDPOINT):
        raise InvalidRun("invalid formal PIL schedule")
    checkpoints = run / "checkpoints"
    last = checkpoints / "last_model.pt"
    result_file = run / "result.json"
    if resume:
        if result_file.exists() or not last.is_file():
            raise InvalidRun("resume needs an unfinished run and a committed epoch checkpoint")
        runtime.resume(last)
    else:
        run.mkdir(parents=True, exist_ok=False)
    if runtime.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for epoch in range(runtime.epoch + 1, epochs + 1):
        start = time.perf_counter()
        steps = [runtime.train_step(batch) for batch in runtime.loader("train", epoch=epoch)]
        if (len(steps) != runtime.steps_per_epoch or runtime.optimizer_steps != epoch * runtime.steps_per_epoch
                or sum(row["valid_targets"] for row in steps) != runtime.pairing["population"]["train"]["utterances"]):
            raise InvalidRun("actual Train update schedule or target count differs")
        rng = training_rng_identity(runtime.device)
        # The observed Dev tape is included in the epoch transaction.
        dev = runtime.evaluate(include_predictions=epoch == epochs)
        fit = runtime.fit_metrics()
        if (training_rng_identity(runtime.device) != rng
                or dev["metrics"]["valid_target_count"] != runtime.pairing["population"]["dev"]["utterances"]
                or fit["valid_target_count"] != runtime.pairing["population"]["train"]["utterances"]):
            raise InvalidRun("evaluation changed shared RNG or saw the wrong population")
        runtime.history.append({"epoch": epoch, "dev": dev, "fit": fit, "steps": steps,
                                "seconds": time.perf_counter() - start,
                                "gpu_peak_bytes": torch.cuda.max_memory_allocated() if runtime.device.type == "cuda" else None})
        runtime.save_epoch(checkpoints, epoch=epoch, dev=dev)
        print(f"{runtime.arm} E{epoch:02d} Dev W-F1={dev['metrics']['weighted_f1']:.9f}", flush=True)
    # Use the best embedded in the committed last epoch, even after an interrupted save.
    best = runtime.best_checkpoint
    runtime.validate_checkpoint(best)
    runtime.model.load_state_dict(best["model_state_dict"], strict=True)
    best_report = runtime.evaluate(include_predictions=True)
    if best_report["metrics"] != runtime.history[best["epoch"] - 1]["dev"]["metrics"]:
        raise InvalidRun("Best checkpoint reload changed Dev metrics")
    best_path = checkpoints / "best_model.pt"
    save_checkpoint_atomic(best_path, best)
    deploy_path = checkpoints / "deploy_state_dict.pt"
    runtime.export_deploy(deploy_path)
    result = {**HOLD, "experiment": runtime.experiment, "parameter_count": runtime.parameter_count,
              "initialization": runtime.initialization, "optimizer_steps": runtime.optimizer_steps,
              "batch_pairing": runtime.pairing, "curriculum_membership_sha256": runtime.curriculum.manifest.membership_sha256,
              "best_epoch": best["epoch"], "Best": best_report, "E30": runtime.history[-1]["dev"],
              "history": [{**row, "dev": row["dev"]["metrics"]} for row in runtime.history],
              "checkpoints": {path.name: sha256_file(path) for path in (best_path, last, deploy_path)}}
    write_json_atomic(result_file, result)
    return result


def read_completed_arm(runtime, run):
    result = read_json(run / "result.json")
    if (result["experiment"] != runtime.experiment or result["initialization"] != runtime.initialization
            or result["optimizer_steps"] != ENDPOINT * STEPS or result["batch_pairing"] != runtime.f0_pairing
            or any(result.get(key) != value for key, value in HOLD.items())
            or set(result["checkpoints"]) != {"last_model.pt", "best_model.pt", "deploy_state_dict.pt"}
            or any(sha256_file(run / "checkpoints" / name) != digest for name, digest in result["checkpoints"].items())):
        raise InvalidRun("completed PIL result/checkpoint identity differs")
    checkpoint = load_checkpoint(run / "checkpoints/last_model.pt", torch.device("cpu"))
    runtime.validate_checkpoint(checkpoint)
    if ([{**row, "dev": row["dev"]["metrics"]} for row in checkpoint["history"]] != result["history"]
            or checkpoint["selector"]["best"]["epoch"] != result["best_epoch"]
            or checkpoint["history"][-1]["dev"] != result["E30"]):
        raise InvalidRun("completed PIL metrics or progress differ from the committed trajectory")
    return result

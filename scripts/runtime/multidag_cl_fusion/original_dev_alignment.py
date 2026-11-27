"""Additional evidence for the frozen F0 Train/Dev runtime."""

import csv
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import yaml

from scripts.analysis.models.multidag_cl.group_isolated_comparison import read_export
from scripts.analysis.models.multidag_cl.pairwise_preflight_inputs import (
    F0_IDENTITY, HOLD, PROJECTION_MANIFEST, SPLITS, TRAIN_DEV_COUNTS,
    load_projected_train_dev, validate_hold, validate_population,
)
from scripts.models.multidag_cl.group_isolated import build_runtime
from scripts.runtime.multidag_cl_fusion.runtime import (
    FusionAuditRuntime, paired_model, state_digest, training_rng_identity,
)
from scripts.runtime.multidag_cl_group_isolated.budget import identity_collate, score_state
from scripts.runtime.multidag_cl_group_isolated.evaluation import checkpoint_path, hard_label_scores, validate_predictions
from scripts.runtime.multidag_cl_group_isolated.fixed_tail_average import (
    AVERAGING_EPOCHS, authenticated_run, parameter_schema, read_checkpoint, validate_state,
)
from scripts.runtime.multidag_cl_group_isolated.protocol import (
    ENDPOINT, FoldView, G1_EPSILON, PROJECTION_FIELDS, SEED, SESSIONS,
    describe, fold_config, population, read_json,
)
from scripts.runtime.multidag_cl_paper_reimplementation.checkpoint import save_checkpoint_atomic
from scripts.runtime.multidag_cl_ta_v0.formal import json_identity, sha256_file
from scripts.runtime.multidag_cl_ta_v0.training import cpu_snapshot
from scripts.runtime.multidag_cl_tav_full_audit_v1.confirm import checked_completion, checked_epochs
from scripts.runtime.multidag_cl_tav_full_audit_v1.runtime import AuditRuntime


ROOT = Path(__file__).resolve().parents[3]
CONFIG = ROOT / "configs/multidag_cl/original_dev_alignment/seed100.yaml"
PROTOCOL = "ORIGINAL_CT_MULTIDAG_DEV"
GROUP = "original_dev_alignment"
STEPS = 7
CLASSIFICATION = ("accuracy", "weighted_f1", "macro_f1", "uar")
REFERENCE_FILES = {
    "resolved_config.yaml": "cff1d9509267ce344f4fc54627d3e7d547de6823ed45e7b3ec96f8107e81c1dd",
    "manifests/run_manifest.json": "19ec7cd2bd1c53415ea00918440c3a97e17bbcedb50ca5032b597ecd044be2eb",
    "logs/epoch_metrics.tsv": "8cf872043f9c69547458dac06752afa5922affc99abe834dfa733c4b17578150",
    "reports/validation_best_dev_evidence.json": "d1c1ece87462c8c7bce5f2ea1e3da5077eed62efb58e9ff79c34ca5685755f7e",
}
BUDGET_EXPORT_SHA256 = {
    # Original G1 review bundle inventory, CE arms only.
    "Ses01": ("7bc6d3091c2319a9169b3f932155e7fe6f819462fc393839e5bec15eeb725046",
              "6f3579c775922793584a233be997ca6e2a9efe2e8dc731454bb2af59bd31cdcc",
              "ec3b1e30985e9e2cc02896dbb64ae86a46253816b5fd842f390977b941e449f3",
              "65cb0ca93ddaaadaa2e93e61e07a2c2f24b32453a0da3ce67ef2cd9760aa94cf",
              "7bf22235ecd93d801e06c3dd06fdd9d5e8a6fa985fdbdfb4cc744836e7b038db"),
    "Ses02": ("85e5de7523bc1c3478dd2a459e8896ee8eea524b65fdf4ebb5a2b2dbb9122535",
              "7398d4ee051ed037eee1ef7244afb0cffef96488cecd6a83cb6460bcb02fadfc",
              "adc0aa38bf161733412da5316914f98d4018f773a3415e3d6cfb55d3c94ca87a",
              "24ae4e804053a81e11a5285cfb2b9019895f52468699e534fd90c07c82151052",
              "59cdfb1ed75a7d7394f87ace66ac06a1d63fc17434d2cf1f5c5930b832c7c61b"),
    "Ses03": ("63bb41b237e49d77b47a336c7a297e3ff2cf96d51e3873480880752d3e6c59f9",
              "43d585f6841dc9302a6a8a418f5f50c1d403b8c703eeff045a6d055e85f112eb",
              "4104618f6e46109dd0f72901335704e0b7d054947c21005dcb632bb7c1d6deb9",
              "2766787baafa28db5a4aafc5792d16473ce9daaf9a13afb9d8578766a5bf2f98",
              "0579cb950ccb5661be235f2185e632099604f220410579b4df2b23a0425e443a"),
    "Ses04": ("cf9877b1bae8fd53b6916aeb60ebd9d9ac71444e6e11285834d1539928903767",
              "eed2a6e1709ccc08422f75c37b95f60e6909e5e00c4a4bb9a0499d02b238552b",
              "952b591d5937c890285b9da998794b7ae28b353fbd8a7ac0ea0f677c281c1116",
              "0c1f86e22ae75e4aa83438e7c32a4da235863c6cf261d3fc7b2f0f4b982a1de4",
              "350b646b294c751c7878d6bdc7dd3758beb653cb2194f85c3701ee546110a23f"),
}


def read_config(frozen):
    config = yaml.safe_load(CONFIG.read_bytes())
    fit_steps = [(frozen["folds"][session]["fit"]["dialogues"] + 15) // 16 for session in SESSIONS]
    common_steps = [step for step in range(1, min(fit_steps) * ENDPOINT + 1)
                    if all(step % length == 0 for length in fit_steps)]
    expected = {
        "protocol": PROTOCOL, "seed": SEED, "train_label_smoothing": {"D0": 0.0, "D1": G1_EPSILON},
        "average_epochs": list(AVERAGING_EPOCHS), "budget_candidate_updates": common_steps,
        "output": {"root": "outputs", "experiment_group": GROUP, "experiment_date": None}, "test": "HOLD",
    }
    if config != expected or not common_steps:
        raise ValueError("D0–D4 recipe or frozen common update grid changed")
    return config


def epoch_rows(run):
    with (run / "logs/epoch_metrics.tsv").open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def read_reference(run, frozen, raw):
    if any(sha256_file(run / path) != digest for path, digest in REFERENCE_FILES.items()):
        raise ValueError("historical F0 reference file SHA-256 differs from the audited archive")
    manifest = read_json(run / "manifests/run_manifest.json")
    resolved = yaml.safe_load((run / "resolved_config.yaml").read_bytes())
    evidence = read_json(run / "reports/validation_best_dev_evidence.json")
    checked_completion(run, manifest)
    validate_hold(evidence, legacy=True)
    formal = frozen["formal_f0"]
    if (resolved["audit_config"] != raw or manifest["config_sha256"] != json_identity(raw)
            or manifest["data_identity"] != formal["data_identity"]
            or manifest["architecture_identity"] != formal["architecture_sha256"]
            or manifest["batch_pairing"]["population"] != formal["population"]
            or manifest["git_head"] != F0_IDENTITY["git_head"]
            or manifest["best_checkpoint_sha256"] != F0_IDENTITY["checkpoint_sha256"]
            or evidence["checkpoint_sha256"] != F0_IDENTITY["checkpoint_sha256"]
            or evidence["metrics"] != manifest["best_dev_metrics"]
            or evidence["split"] != "dev" or evidence["optimizer_steps_performed"] != 0):
        raise ValueError("F0 reference protocol or checkpoint identity mismatch")
    checked_epochs(run, F0_IDENTITY["epoch"])
    return manifest, evidence, epoch_rows(run)


def load_inputs(directory, frozen, historical, evidence):
    certificate = read_json(directory / PROJECTION_MANIFEST)
    validate_hold(certificate)
    expected = {**frozen["certified_projection"], **HOLD, "splits": list(SPLITS),
                "split_counts": TRAIN_DEV_COUNTS, "source_data_identity": historical["data_identity"]}
    if {key: certificate.get(key) for key in PROJECTION_FIELDS} != expected:
        raise ValueError("projection differs from the authenticated POST-PH J/K Train/Dev asset")
    dev_rows = tuple((r["utterance_id"], r["dialogue_id"], r["true_label"]) for r in evidence["predictions"])
    source = SimpleNamespace(manifest=historical, population=dev_rows)
    datasets, projection = load_projected_train_dev(directory, source)
    if projection != expected:
        raise ValueError("loaded projection differs from the authenticated POST-PH J/K Train/Dev asset")
    entries = {split: population(datasets[split]) for split in SPLITS}
    for group, index in (("utterance", 0), ("dialogue", 1)):
        if {row[index] for row in entries["train"]} & {row[index] for row in entries["dev"]}:
            raise ValueError(f"F0 Train/Dev {group} IDs overlap")

    summaries = {split: describe(rows) for split, rows in entries.items()}
    # These identities are shared by the archived F0 Dev predictions and the frozen formal Train folds.
    for key, expected_overlap in (("sessions", {"Ses04"}), ("actors", {"Ses04F", "Ses04M"})):
        dev_groups = set(summaries["dev"][key])
        overlap = set(summaries["train"][key]) & dev_groups
        if overlap != expected_overlap:
            raise ValueError(
                f"F0 Train/Dev shared {key} mismatch: expected {sorted(expected_overlap)}, got {sorted(overlap)}"
            )
        if dev_groups != expected_overlap:
            raise ValueError(f"F0 Dev {key} mismatch: expected {sorted(expected_overlap)}, got {sorted(dev_groups)}")

    validate_population(datasets, source)
    for split in SPLITS:
        if summaries[split]["labeled_tape_sha256"] != frozen["formal_f0"]["labeled_tape_sha256"][split]:
            raise ValueError(f"F0 {split} labeled population SHA-256 mismatch")
    return datasets, entries, projection, summaries


def replay_budget_weights(run, manifest, frozen_fold, raw, train, epochs, device):
    """Read historical G0 weights and score only one held-out formal Train session."""
    held_ids = set(frozen_fold["held_out"]["dialogue_ids"])
    fit, held = [], []
    for index in range(len(train)):
        item = train[index]
        (held if item["dialogue_id"] in held_ids else fit).append(item)
    rows = [row for row in population(train) if row[1] in held_ids]
    if describe(rows) != frozen_fold["held_out"] or len(fit) != frozen_fold["fit"]["dialogues"]:
        raise ValueError("historical budget checkpoint replay population differs from the frozen fold")
    runtime, _ = build_runtime(raw, {"train": FoldView(fit, "train")}, device)
    validation = FoldView(held, "dev")
    curves, hashes = {}, {}
    for epoch in epochs:
        digest = manifest["checkpoint_sha256"][str(epoch)]
        relative = checkpoint_path(run, epoch).relative_to(run).as_posix()
        source = {"epoch": epoch, "path": relative, "sha256": digest,
                  "optimizer_steps": epoch * manifest["steps_per_epoch"]}
        state = read_checkpoint(run, source, manifest["run_identity"])
        curves[epoch] = score_state(runtime, state, validation, rows)
        hashes[relative] = digest
    return curves, hashes


def select_budget(config, frozen, raw, ce_root, *, train=None, device="cuda"):
    """Score only the four historical holdouts drawn from formal Train."""
    grid = config["budget_candidate_updates"]
    base = {**HOLD, "candidate_updates": grid, "criterion": "mean_session_hard_nll_then_smallest_updates"}
    if ce_root is None:
        return {**base, "status": "UNAVAILABLE", "reason": "No historical G0 CE export root supplied"}
    try:
        sources, scores = {}, []
        for session in SESSIONS:
            fold = fold_config(session)
            run = ce_root / "runs" / fold["run_id"]
            manifest, digest = authenticated_run(run, fold, frozen, raw)
            pinned = BUDGET_EXPORT_SHA256[session]
            if digest != pinned[0]:
                raise ValueError(f"{session} CE source differs from the original G1 review inventory")
            epochs = []
            for update in grid:
                epoch, remainder = divmod(update, manifest["steps_per_epoch"])
                if remainder or not 1 <= epoch <= ENDPOINT:
                    raise ValueError(f"{session} has no frozen epoch for update {update}")
                epochs.append(epoch)
            try:
                _, curves, exported, _, proof = read_export(ce_root, fold, frozen, raw)
                prediction_hashes = proof["prediction_files_sha256"]
                actual = (proof["run_manifest_sha256"], proof["evaluation_manifest_sha256"],
                          proof["fit_updates_sha256"], prediction_hashes["trajectory_predictions.csv.gz"],
                          prediction_hashes["held_out_epoch30_predictions.csv"])
                if exported != manifest or actual != pinned:
                    raise ValueError(f"{session} CE export SHA-256 differs from the original G1 review inventory")
                sources[session] = {"mode": "historical_predictions", **proof}
            except FileNotFoundError:
                if train is None:
                    raise
                curves, hashes = replay_budget_weights(run, manifest, frozen["folds"][session], raw, train, epochs, device)
                sources[session] = {"mode": "historical_checkpoint_replay", "run_manifest_sha256": digest,
                                    "checkpoint_files_sha256": hashes}
            for update, epoch in zip(grid, epochs):
                rows = curves[epoch]
                logits = np.array([r["logits"] for r in rows], dtype=np.float64)
                labels = np.array([r["true_label"] for r in rows])
                if not np.array_equal(logits.argmax(1), [r["predicted_label"] for r in rows]):
                    raise ValueError(f"{session} replay predictions disagree with logits")
                _, details = hard_label_scores(logits, labels)
                scores.append({"session": session, "updates": update, "epoch": epoch, "n": len(rows),
                               "checkpoint_sha256": manifest["checkpoint_sha256"][str(epoch)],
                               "hard_nll": float(details["nll"].mean())})
        means = [{"updates": update, "hard_nll": sum(r["hard_nll"] for r in scores
                  if r["updates"] == update) / len(SESSIONS)} for update in grid]
        choice = min(means, key=lambda row: (row["hard_nll"], row["updates"]))
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, EOFError, csv.Error) as error:
        return {**base, "status": "UNAVAILABLE", "reason": f"{type(error).__name__}: {error}"}
    return {**base, "status": "LOCKED", "selected_updates": choice["updates"], "scores": scores,
            "means": means, "sources": sources, "scope": "original_train_ses01_through_ses04_only"}


class AlignmentRuntime(FusionAuditRuntime):
    """Keep the original loop and selector, and save model-only states alongside them."""

    def __init__(self, raw, datasets, run_dir, experiment, *, options, pairing=None, historical_epochs=None):
        device = options["device"]
        model, _, initialization = paired_model(raw, device=device)
        super().__init__(model, datasets.__getitem__, initialization=initialization, f0_pairing=pairing,
                         train_label_smoothing=experiment["train_label_smoothing"], **options)
        parameter_schema(self.model)
        initialization["training_rng_identity"] = training_rng_identity(device)
        self.run_dir, self.experiment = run_dir, experiment
        self.steps_per_epoch = (len(datasets["train"]) + 15) // 16
        self.historical_epochs = historical_epochs
        self.snapshots = {}

    def forward_batch(self, batch, **options):
        output, loss = super().forward_batch(batch, **options)
        if not torch.equal(output.mask, batch["attention_mask"].to(output.mask.device)):
            raise ValueError("model mask differs from the canonical valid targets")
        return output, loss

    def snapshot(self, name, epoch, epoch_complete):
        path = self.run_dir / "checkpoints" / "alignment" / f"{name}.pt"
        record = {"epoch": epoch, "optimizer_steps": self.optimizer_steps, "epoch_complete": epoch_complete}
        save_checkpoint_atomic(path, {**record, "identity": self.experiment,
                                      "model_state_dict": cpu_snapshot(self.model.state_dict())})
        self.snapshots[name] = {**record, "path": path.relative_to(self.run_dir).as_posix(),
                                "sha256": sha256_file(path)}

    def train_step(self, batch):
        step = super().train_step(batch)
        if self.optimizer_steps == self.experiment["budget_updates"]:
            self.snapshot("u_star", self.epoch + 1, self.optimizer_steps % self.steps_per_epoch == 0)
        return step

    def save_epoch(self, directory, *, epoch, dev, epoch_complete=True):
        if self.optimizer_steps != epoch * self.steps_per_epoch:
            raise ValueError("optimizer steps differ from the frozen epoch schedule")
        improved = super().save_epoch(directory, epoch=epoch, dev=dev, epoch_complete=epoch_complete)
        self.snapshot(f"epoch_{epoch:02d}", epoch, True)
        if self.historical_epochs is not None:
            expected = self.historical_epochs[epoch - 1]
            drift = {key: {"historical": float(expected[f"dev_{key}"]), "actual": dev["metrics"][key]}
                     for key in CLASSIFICATION if float(expected[f"dev_{key}"]) != dev["metrics"][key]}
            if drift:
                raise ValueError(f"Dev performance drift at epoch {epoch}: {drift}")
        return improved


def read_state(run, name, record, identity, model):
    expected_path = f"checkpoints/alignment/{name}.pt"
    if record["path"] != expected_path:
        raise ValueError(f"unexpected path for {name}")
    payload = (run / expected_path).read_bytes()
    if sha256(payload).hexdigest() != record["sha256"]:
        raise ValueError(f"checkpoint SHA-256 mismatch for {name}")
    state = torch.load(BytesIO(payload), weights_only=True, map_location="cpu")
    for key in ("epoch", "optimizer_steps", "epoch_complete"):
        if state[key] != record[key]:
            raise ValueError(f"checkpoint {key} mismatch for {name}")
    if state["identity"] != identity:
        raise ValueError(f"checkpoint experiment mismatch for {name}")
    validate_state(state["model_state_dict"], parameter_schema(model))
    return state["model_state_dict"]


def preflight(runtime, expected_pairing, entries, *, epochs=ENDPOINT):
    """Audit actual sampler order and model inputs before the first update."""
    start_rng = training_rng_identity(runtime.device)
    for split in SPLITS:
        for epoch in range(1, (epochs if split == "train" else 1) + 1):
            loader = AuditRuntime.loader(runtime, split, epoch=epoch)
            loader.collate_fn = identity_collate
            batches = [json_identity(list(zip(b["dialogue_ids"], b["utterance_ids"]))) for b in loader]
            expected = expected_pairing["train"][str(epoch)] if split == "train" else expected_pairing["dev"]
            if batches != expected["batch_sha256"]:
                raise ValueError(f"{split} epoch {epoch} preflight batch tape differs")
    batch = next(iter(AuditRuntime.loader(runtime, "dev")))
    valid = batch["attention_mask"]
    perturbed = {**batch, "labels": torch.where(valid, (batch["labels"] + 1) % 6, batch["labels"])}
    mode = runtime.model.training
    before = state_digest({k: v.detach().cpu() for k, v in runtime.model.named_parameters()})
    optimizer = cpu_snapshot(runtime.optimizer.state_dict())
    report = runtime.evaluate([batch], include_predictions=True)
    alternate = runtime.evaluate([perturbed], include_predictions=True)
    validate_predictions(report["predictions"], entries["dev"][:int(valid.sum())])
    if any(a[key] != b[key] for a, b in zip(report["predictions"], alternate["predictions"])
           for key in ("logits", "probabilities", "predicted_label")):
        raise ValueError("gold-label perturbation changed inference")
    # Keep shape, speaker layout and the entire supplied prefix fixed.
    tail = torch.arange(valid.shape[1])[None] >= 2
    changed = dict(batch)
    for feature in ("text_features", "audio_features", "visual_features"):
        changed[feature] = batch[feature] + (tail & valid)[..., None] * 7
    with torch.no_grad():
        output, _ = runtime.forward_batch(batch, split="dev")
        comparison, _ = runtime.forward_batch(changed, split="dev")
        prefix = (valid & ~tail).to(runtime.device)
        if not torch.equal(output.logits_ta[prefix], comparison.logits_ta[prefix]):
            raise ValueError("supplied future feature perturbation changed prefix logits")
    runtime.model.train(mode)
    after = state_digest({k: v.detach().cpu() for k, v in runtime.model.named_parameters()})
    if (before != after or start_rng != training_rng_identity(runtime.device)
            or runtime.optimizer_steps != 0 or runtime.epoch != 0 or runtime.selector.best is not None
            or runtime.optimizer.state_dict() != optimizer or runtime.model.training != mode
            or any(parameter.grad is not None for parameter in runtime.model.parameters())):
        raise ValueError("Train/Dev evaluation changed model, optimizer, selector, mode or RNG")
    return {"batch_tape": "MATCH", "label_perturbation": "PASS", "causal_prefix": "PASS",
            "evaluation_state_isolation": "PASS", "canonical_valid_mask": "MATCH"}

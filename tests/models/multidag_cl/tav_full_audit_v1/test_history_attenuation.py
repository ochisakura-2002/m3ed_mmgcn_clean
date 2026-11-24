import csv
import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace
from zipfile import ZipFile

import numpy as np
import pytest
import torch

from datasets.iemocap.tav_full_audit_dataset import DIMS, collate
from models.multidag_cl.fusion.model import FusionTAV
from models.multidag_cl.graph_core_variants.common import row_operator
from models.multidag_cl.graph_core_variants.track import TrackCore
from models.multidag_cl.history_attenuation import FrozenHistoryModel, HistoryAttenuation
from models.multidag_cl.paper_reimplementation.contracts import SpeakerRelations
from scripts.analysis.models.multidag_cl import history_fastkill_provenance as provenance
from scripts.analysis.models.multidag_cl import history_fastkill_runtime as runtime
from scripts.analysis.models.multidag_cl import run_history_fastkill as entry
from scripts.analysis.models.multidag_cl.failure_audit_metrics import metadata
from scripts.analysis.models.multidag_cl.history_fastkill_metrics import decision
from scripts.analysis.models.multidag_cl.pairwise_preflight_inputs import CLASS_ORDER, F0_IDENTITY, HOLD, PROJECTION_MANIFEST
from scripts.runtime.multidag_cl_fusion.spec import make_final_config
from scripts.runtime.multidag_cl_ta_v0.formal import json_identity, sha256_file


def make_model():
    torch.manual_seed(100)
    return FusionTAV(make_final_config("F0", 100)).eval()


class SyntheticDataset:
    def __init__(self, split, lengths):
        self.split = split
        self.samples, self.dialogue_ids, self.ordered, self.rows = [], [], {}, {}
        generator = torch.Generator().manual_seed(6 if split == "train" else 7)
        for index, length in enumerate(lengths):
            did = f"{split}_{index}"
            ids = tuple(f"{did}_{i}" for i in range(length))
            labels = torch.arange(length).add(index).remainder(6)
            self.samples.append({"split": split, "dialogue_id": did, "utterance_ids": ids,
                                 "speaker_ids_int": torch.arange(length).remainder(2), "labels": labels,
                                 **{key: torch.randn(length, dim, generator=generator) for key, dim in DIMS.items()}})
            self.dialogue_ids.append(did)
            self.ordered[did] = [(i, i, uid) for i, uid in enumerate(ids)]
            self.rows.update({uid: {"dialogue_id": did, "benchmark_label_id": label}
                              for uid, label in zip(ids, labels.tolist())})

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]


def test_two_scalars_stay_bounded_and_only_weight_legal_history():
    h0 = torch.tensor([[[1., 0.], [-1., 0.], [0., 1.], [0., 0.]]], requires_grad=True)
    valid = torch.tensor([[True, True, True, False]])
    adjacency = torch.zeros((1, 4, 4), dtype=torch.bool)
    adjacency[0, 1, 0] = adjacency[0, 2, 0] = adjacency[0, 2, 1] = True
    affiliations = torch.tensor([[0, 0, 1, -1]])
    pairs = valid[:, :, None] & valid[:, None, :]
    same = (affiliations[:, :, None] == affiliations[:, None, :]) & pairs
    relations = SpeakerRelations(same, pairs & ~same)
    baseline = row_operator(adjacency, valid, self_terms=True)
    module = HistoryAttenuation()
    with torch.no_grad():
        module.raw_lambda_same.fill_(2.0)
        module.raw_lambda_diff.fill_(0.25)
    assert [p.numel() for p in module.parameters() if p.requires_grad] == [1, 1]
    assert runtime.lambda_values(module) == {"lambda_same": 0.5, "lambda_diff": 0.25}
    measured = []
    result = module(baseline, h0, adjacency, relations, observe=lambda *values: measured.extend(values))
    relation, factor, similarity, distance = measured
    assert relation.tolist() == [True, False, False]
    assert torch.equal(factor, torch.tensor([0.5, 0.875, 0.875]))
    assert factor.min() >= 0.5 and factor.max() <= 1.0
    assert torch.equal(similarity, torch.tensor([-1., 0., 0.]))
    assert torch.equal(distance, torch.tensor([1., 0.5, 0.5]))
    assert torch.equal(result[~adjacency], baseline[~adjacency])
    assert torch.equal(result[adjacency], baseline[adjacency] * factor)
    assert result[0, 1].sum() < 1  # Original denominator and unchanged diagonal.
    result.sum().backward()
    assert h0.grad is None
    with torch.no_grad():
        module.raw_lambda_same.fill_(-2.)
    assert runtime.lambda_values(module)["lambda_same"] == 0.
    assert not torch.triu(adjacency).any()
    outside = adjacency.clone()
    outside[0, 0, 1] = True
    with pytest.raises(ValueError, match="future"):
        TrackCore(2)(h0, outside, relations, valid, history_attenuation=module)


def test_clamp_optimizer_updates_from_zero_and_handles_no_edges():
    assert all(0 < value <= 0.5 for value in runtime.optimizer_step_gate("cpu").values())
    model = FrozenHistoryModel(make_model()).eval()
    dataset = SyntheticDataset("train", [1, 1])
    batch = collate(dataset.samples)
    output = model(**runtime.model_inputs(batch, "cpu"))
    output.logits_ta.sum().backward()
    for parameter in model.attenuation.parameters():
        assert parameter.grad is not None
        assert parameter.grad.item() == 0


def test_end_to_end_f0_exact_bypass_zero_lambdas_and_freeze():
    model = FrozenHistoryModel(make_model(), history_attenuation_enabled=False).eval()
    snapshot = runtime.freeze_snapshot(model)
    batch = collate(SyntheticDataset("train", [5, 3, 1]).samples)
    inputs = runtime.model_inputs(batch, "cpu")
    original = model.f0(**inputs)
    disabled = model(**inputs)
    zero = model(history_attenuation_enabled=True, **inputs)
    for output in (disabled, zero):
        runtime.require_exact(original, output, "synthetic identity")
        assert torch.equal(output.logits_ta.view(torch.uint8), original.logits_ta.view(torch.uint8))
        assert all(torch.equal(a, b) for a, b in zip(output.layer_states, original.layer_states))
    zero.logits_ta[zero.mask].square().mean().backward()
    runtime.check_frozen(model, snapshot)
    assert all(p.grad is not None and torch.isfinite(p.grad) for p in model.attenuation.parameters())
    assert all(not p.requires_grad and p.grad is None for p in model.f0.parameters())


def test_positive_attenuation_is_causal_padding_and_zero_history_safe():
    model = FrozenHistoryModel(make_model()).eval()
    probe = runtime.verify_causal(model, "cpu")
    assert set(probe["probes"]) == {"future", "prefix", "padding", "zero_history", "active_bypass"}
    assert runtime.lambda_values(model.attenuation) == {"lambda_same": 0, "lambda_diff": 0}


def test_train_shuffle_and_adam_trajectory_reproduce_at_seed100():
    train = SyntheticDataset("train", [2, 3, 4, 2])
    first, second = (FrozenHistoryModel(make_model()).eval() for _ in range(2))
    history = runtime.train_fixed(first, train, device="cpu", batch_size=2)
    assert history == runtime.train_fixed(second, train, device="cpu", batch_size=2)
    assert [row["epoch"] for row in history] == list(range(1, 21))
    assert all(row["train_targets"] == 11 and row["optimizer_steps"] == 2 for row in history)
    assert len({row["train_batch_tape_sha256"] for row in history}) > 1


@pytest.mark.parametrize(("changes", "status"), [
    ({}, "GO"),
    ({"delta_weighted_f1_pp": 0.20}, "REVIEW_SIGNAL"),
    ({"delta_weighted_f1_pp": 0.1999}, "STOP"),
    ({"net": 0}, "STOP"),
    ({"net": 3}, "REVIEW_SIGNAL"),
    ({"h_top25_net": 0}, "STOP"),
    ({"h_top25_net": 2}, "REVIEW_SIGNAL"),
    ({"shift_net": -1}, "REVIEW_SIGNAL"),
    ({"delta_macro_f1_pp": -1.0}, "REVIEW_SIGNAL"),
    ({"delta_macro_f1_pp": -1.0001}, "STOP"),
    ({"delta_uar_pp": -1.0}, "REVIEW_SIGNAL"),
    ({"delta_uar_pp": -1.0001}, "STOP"),
])
def test_decision_boundaries(changes, status):
    summary = {"delta_weighted_f1_pp": 0.50, "net": 4, "h_top25_net": 3, "shift_net": 0,
               "delta_macro_f1_pp": -0.30, "delta_uar_pp": -0.30, **changes}
    result = decision(summary)
    assert result["FASTKILL"] == status
    assert result["five_seed_eligible"] is (status == "GO")


def csv_bytes(rows):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


@pytest.fixture
def recovery_case(artifact_dir):
    population = []
    previous = {}
    remaining = 273
    for dialogue in range(20):
        did = f"Ses04F_impro{dialogue:02}"
        for index in range(49):
            speaker = "FM"[index % 2]
            key = did, speaker
            label = previous.get(key, 0)
            if key in previous and remaining:
                label = 1 - label
                remaining -= 1
            previous[key] = label
            population.append((f"{did}_{speaker}{index:03}", did, label))
    error_indices = {*range(102), *range(300, 485)}
    h_indices = [*range(102), 300, 301, *range(300 - 141, 300)]
    u_indices = set(range(245))
    uh, h, shifts, history = [], [], [], []
    for index, row in enumerate(metadata(population)):
        gold = population[index][2]
        prediction = CLASS_ORDER[1 - gold if index in error_indices else gold]
        base = {"seed": 100, "dev_index": index, **row, "gold": row["true_label"]}
        in_u, in_h = index in u_indices, index in h_indices
        name = {(True, True): "U_INTERSECT_H", (True, False): "U_ONLY",
                (False, True): "H_ONLY", (False, False): "NEITHER"}[in_u, in_h]
        shifts.append(base)
        uh.append({**base, "full_prediction": prediction, "is_u_top25": in_u, "is_h_top25": in_h, "subgroup": name})
        if in_h:
            h.append({**base, "full_prediction": prediction, "rank": h_indices.index(index) + 1, "is_h_top25": True})
        # An artificial tied trace makes recomputation observably different.
        history.extend({**base, "condition": condition, "row_type": "sample", "prediction": prediction}
                       for condition in provenance.HISTORY)
    directories = SimpleNamespace(folder=artifact_dir / "recovery", archive=artifact_dir / "recovery.zip")
    directories.folder.mkdir()
    return SimpleNamespace(**vars(directories), h=sorted(h, key=lambda row: row["rank"]), uh=uh, shifts=shifts,
                           history=history, population=tuple(population))


def publish(case, monkeypatch, **changes):
    data = {provenance.H_CSV: csv_bytes(case.h), provenance.UH_CSV: csv_bytes(case.uh),
            provenance.SHIFT_CSV: csv_bytes(case.shifts), provenance.HISTORY_CSV: csv_bytes(case.history),
            provenance.REPORT: b"synthetic recovery"}
    manifest = {**HOLD, **provenance.ANCHORS, "schema": "ct_multidag_history_top25_recovery_v1",
                "experiment_group": provenance.GROUP, "batch_id": provenance.RECOVERY_BATCH, "split": "dev", "seed": 100,
                "f0_checkpoint": F0_IDENTITY, "history_conditions": list(provenance.HISTORY),
                "FORMAL_TRAIN_RUNS": 0, "OPTIMIZER_STEPS": 0, "WORKSTREAM_CHECK": "PASS", "F0_IDENTITY": "PASS",
                "PROVENANCE_RECOVERY": "PASS", "H_TOP25_PROVENANCE": "PASS", "H_SUBGROUP_PROVENANCE": "PASS",
                "TEST_TOUCHED": "NO", "train_dev_projection": HOLD,
                "dev_gold_population_sha256": json_identity(case.population),
                "uh_subgroup_counts": {name: sum(row["subgroup"] == name for row in case.uh)
                                       for name in provenance.H_GROUPS},
                "outputs_sha256": {name: hashlib.sha256(value).hexdigest() for name, value in data.items()}, **changes}
    data[provenance.MANIFEST] = json.dumps(manifest).encode("utf-8")
    with ZipFile(case.archive, "w") as zipped:
        for name, value in data.items():
            zipped.writestr(name, value)
            (case.folder / name).write_bytes(value)
    monkeypatch.setattr(provenance, "RECOVERY_SHA256", sha256_file(case.archive))


def test_reuse_frozen_h245_and_shift_membership_without_reranking(recovery_case, monkeypatch):
    publish(recovery_case, monkeypatch)
    result = provenance.read_recovery(recovery_case.folder, recovery_case.archive)
    assert result.population == recovery_case.population
    assert result.groups["H_TOP25"].tolist() == [row["is_h_top25"] for row in recovery_case.uh]
    assert result.groups["H_TOP25"].sum() == 245
    labels = np.array([row[2] for row in result.population])
    assert ((labels != result.predictions) & result.groups["H_TOP25"]).sum() == 104
    assert {name: mask.sum() for name, mask in result.groups.items() if name in provenance.GROUPS} == {
        "SHIFT": 273, "STABLE": 667, "NO_HISTORY": 40}
    assert np.all(sum(result.groups[name] for name in provenance.H_GROUPS) == 1)


def test_recovery_must_bind_to_archived_logits_projection_and_populations(recovery_case, monkeypatch):
    logits = np.eye(len(CLASS_ORDER))[[CLASS_ORDER.index(row["full_prediction"]) for row in recovery_case.uh]]
    source = SimpleNamespace(population=recovery_case.population, identity=F0_IDENTITY,
                             config={"experiment_id": "frozen-synthetic"}, logits=logits,
                             manifest={"best_dev_metrics": {"weighted_f1": .69}})
    dev_population = {"dialogues": 20, "utterances": 980}
    publish(recovery_case, monkeypatch, f0_run_id=source.config["experiment_id"],
            f0_dev_metrics=source.manifest["best_dev_metrics"], dev_population=dev_population,
            archived_dev_logits_sha256=json_identity(logits.tolist()))
    result = provenance.read_recovery(recovery_case.folder, recovery_case.archive)
    result.verify_source(source, HOLD, dev_population)
    with pytest.raises(ValueError, match="projection"):
        result.verify_source(source, {**HOLD, "schema": "wrong"}, dev_population)
    source.logits = logits + 10  # Same predictions, different archived logits identity.
    with pytest.raises(ValueError, match="authenticated F0"):
        result.verify_source(source, HOLD, dev_population)


@pytest.mark.parametrize("counter", sorted(HOLD.keys() - {"test_policy"}))
def test_recovery_rejects_nonzero_hold_counter(recovery_case, monkeypatch, counter):
    publish(recovery_case, monkeypatch, **{counter: 1})
    with pytest.raises(ValueError, match=counter):
        provenance.read_recovery(recovery_case.folder, recovery_case.archive)


def test_recovery_rejects_changed_membership_file_and_archive(recovery_case, monkeypatch):
    publish(recovery_case, monkeypatch)
    (recovery_case.folder / provenance.H_CSV).write_bytes(b"changed")
    with pytest.raises(ValueError, match="directory"):
        provenance.read_recovery(recovery_case.folder, recovery_case.archive)
    recovery_case.archive.write_bytes(b"changed")
    with pytest.raises(ValueError, match="SHA-256"):
        provenance.read_recovery(recovery_case.folder, recovery_case.archive)


def test_recovery_rejects_validly_packaged_but_wrong_anchors(recovery_case, monkeypatch):
    publish(recovery_case, monkeypatch, H_TOP25_F0_ERRORS=105)
    with pytest.raises(ValueError, match="anchors"):
        provenance.read_recovery(recovery_case.folder, recovery_case.archive)


def test_epoch20_once_no_selection_test_hold_and_lightweight_outputs(artifact_dir, monkeypatch):
    assert entry.read_enabled(entry.CONFIG) is True
    train, dev = SyntheticDataset("train", [4, 3, 2]), SyntheticDataset("dev", [3, 2])
    f0 = make_model().requires_grad_(False)
    checkpoint = artifact_dir / "f0.bin"
    checkpoint.write_bytes(b"synthetic F0 checkpoint")
    certificate = artifact_dir / PROJECTION_MANIFEST
    certificate.write_text("{}", encoding="utf-8")
    dev_batch = collate(dev.samples)
    examples, population = runtime.batch_identity(dev_batch, "dev")
    actual = f0(**runtime.model_inputs(dev_batch, "cpu"))
    logits = actual.logits_ta[actual.mask].detach().numpy().copy()
    source = SimpleNamespace(identity={**F0_IDENTITY, "checkpoint_sha256": sha256_file(checkpoint)},
                             config={"training": {"batch_size": 2}}, checkpoint=checkpoint,
                             population=tuple(population), logits=logits,
                             manifest={"config_sha256": "synthetic", "curriculum_membership_sha256": "synthetic",
                                       "batch_pairing": {"dev": {"batch_sha256": [json_identity(examples)]}}})
    groups = {name: np.zeros(len(population), dtype=bool) for name in (*provenance.GROUPS, *provenance.H_GROUPS)}
    groups["H_TOP25"] = np.array([True, False, False, False, False])
    groups["SHIFT"][:] = groups["NEITHER"][:] = True
    events = []
    verified = SimpleNamespace(verify_source=lambda *args: events.append("provenance"), groups=groups,
                               manifest={"batch_id": provenance.RECOVERY_BATCH, "outputs_sha256": {}},
                               manifest_sha256="synthetic", predictions=tuple(logits.argmax(axis=1)))
    monkeypatch.setattr(entry, "read_recovery", lambda *_: verified)
    monkeypatch.setattr(entry, "read_f0", lambda _: source)
    monkeypatch.setattr(entry, "load_projected_train_dev", lambda *_: ({"train": train, "dev": dev}, HOLD))
    monkeypatch.setattr(entry, "validate_population", lambda *_: {"dev": {}, "train": {}})
    monkeypatch.setattr(entry, "membership", lambda ds: "synthetic" if ds is train else pytest.fail("wrong split"))
    monkeypatch.setattr(entry, "frozen_model", lambda *args, **kwargs: f0)
    analyze = entry.analyze

    def tracked_train(model, dataset, **kwargs):
        assert dataset is train
        history = runtime.train_fixed(model, dataset, **kwargs)
        events.extend([row["epoch"] for row in history])
        return history

    def tracked_dev(model, dataset, **kwargs):
        assert dataset is dev
        event = "ctrl" if kwargs.get("identity") else "primary"
        events.append(event)
        return runtime.collect_dev(model, dataset, **kwargs)

    def tracked_analyze(*args):
        events.append("score")
        return analyze(*args)

    monkeypatch.setattr(entry, "train_fixed", tracked_train)
    monkeypatch.setattr(entry, "collect_dev", tracked_dev)
    monkeypatch.setattr(entry, "analyze", tracked_analyze)
    output, archive = artifact_dir / "output", artifact_dir / "output.zip"
    args = SimpleNamespace(recovery_dir=artifact_dir, recovery_archive=archive, frozen_runs=artifact_dir,
                           train_dev_package=artifact_dir, batch_id="synthetic")
    result = entry.execute(args, output, archive, device="cpu", git={}, experiment_date="20261124")
    manifest = json.loads((output / entry.MANIFEST).read_text(encoding="utf-8"))
    assert result == 0, manifest
    assert events == ["provenance", "ctrl", *range(1, 21), "primary", "score"]
    assert manifest["primary_epoch"] == 20
    assert manifest["optimizer_steps"] == 40
    assert manifest["primary_dev_evaluation_count"] == 1
    assert not manifest["dev_used_for_model_selection"]
    assert manifest["TEST_TOUCHED"] == "NO" and all(manifest[k] == val for k, val in HOLD.items())
    assert all(manifest[key] == "PASS" for key in ("F0_IDENTITY", "F0_BYPASS_IDENTITY", "LAMBDA_ZERO_IDENTITY",
                                                   "CAUSAL_INVARIANCE", "F0_PARAMETERS_UNCHANGED",
                                                   "F0_GRADIENTS_ABSENT"))
    state = torch.load(output / entry.CHECKPOINT, weights_only=True)
    assert state["epoch"] == 20 and tuple(state["state_dict"]) == runtime.PARAMETERS
    assert set(output.glob("*.pt")) == {output / entry.CHECKPOINT}
    with ZipFile(archive) as zipped:
        assert zipped.testzip() is None
        assert set(zipped.namelist()) == {*manifest["outputs_sha256"], entry.MANIFEST}
        assert all(Path(name).suffix in {".csv", ".json", ".md"} for name in zipped.namelist())
        rows = list(csv.DictReader(zipped.read("ATTENUATION_DISTRIBUTION.csv").decode().splitlines()))
        assert [int(row["count"]) for row in rows] == [4, 1, 3]
        assert all(hashlib.sha256(zipped.read(name)).hexdigest() == digest
                   for name, digest in manifest["outputs_sha256"].items())
    with pytest.raises(FileExistsError):
        entry.execute(args, output, archive, device="cpu", git={}, experiment_date="20261124")


def test_invalid_recovery_stops_before_f0_or_any_training(artifact_dir, monkeypatch):
    args = SimpleNamespace(recovery_dir=artifact_dir, recovery_archive=artifact_dir / "missing.zip", batch_id="blocked")
    monkeypatch.setattr(entry, "read_f0", lambda *_: pytest.fail("opened F0 after missing provenance"))
    monkeypatch.setattr(entry, "train_fixed", lambda *_: pytest.fail("trained after missing provenance"))
    output, archive = artifact_dir / "output", artifact_dir / "output.zip"
    assert entry.execute(args, output, archive, device="cpu", git={}, experiment_date="20261124") == 2
    data = json.loads((output / entry.MANIFEST).read_text(encoding="utf-8"))
    assert data["FASTKILL"] == "REVIEW_INVALID" and data["failed_stage"] == "RECOVERY_ARTIFACT"
    assert data["formal_train_runs"] == data["primary_dev_evaluation_count"] == 0
    assert not archive.exists()

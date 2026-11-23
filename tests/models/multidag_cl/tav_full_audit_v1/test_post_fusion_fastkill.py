"""Matched training-only objectives, isolation, and plot-ready export."""

import csv
from copy import deepcopy
import hashlib
import io
import json
import math
import tarfile

import numpy as np
import pytest
import torch

from datasets.iemocap.tav_full_audit_dataset import COUNTS, MANIFEST_FIELDS, SCHEMA, collate, identity_sha
from scripts.analysis.models.multidag_cl.export_fastkill_bundle import GATE_PAIRS, route_decisions
from scripts.analysis.models.multidag_cl.fastkill_geometry import represent
from scripts.runtime.multidag_cl_fastkill.objectives import alpha_values, prefix_scores, wheel_similarity
from scripts.runtime.multidag_cl_fastkill.prefix_diagnostic import from_dataset, from_manifest, require_suffix_invariance
from scripts.runtime.multidag_cl_fastkill.runtime import build_runtime
from scripts.runtime.multidag_cl_fastkill.spec import A1, ARMS, B1, CTRL, GROUP, HOLD, baseline, load_configs, validate_arm
from scripts.runtime.multidag_cl_fusion.runtime import paired_runtime
from scripts.runtime.multidag_cl_fusion.spec import F0_PARAMETER_COUNT
from scripts.runtime.multidag_cl_ta_v0.formal import sha256_file
from scripts.runtime.multidag_cl_tav_full_audit_v1 import confirm
from scripts.runtime.multidag_cl_tav_full_audit_v1.runtime import SyntheticDialogues
from scripts.workflows.benchmarks import run_ct_multidag_post_fusion_fastkill as launcher
from utils.output_paths import resolve_output_paths


@pytest.fixture
def datasets():
    return {split: SyntheticDialogues(split) for split in ("train", "dev")}


def assert_models_equal(model, reference):
    assert model.state_dict().keys() == reference.state_dict().keys()
    for key, value in model.state_dict().items():
        expected = reference.state_dict()[key]
        if isinstance(value, torch.Tensor):
            assert torch.equal(value, expected), key
        else:
            assert value == expected, key


def test_ctrl_matches_frozen_loss_predictions_parameters_rng_and_updates(datasets):
    options = {"device": "cpu", "synthetic_only": True}

    def take_steps(runtime):
        before = runtime.initialization
        steps = []
        for epoch in range(1, 3):
            batches = runtime.loader("train", epoch=epoch)
            for batch in batches:
                rng = torch.get_rng_state()
                output, loss = runtime.forward_batch(batch, split="train")
                torch.set_rng_state(rng)
                step = runtime.train_step(batch)
                steps.append((output.logits_ta.detach(), output.prob_ta.detach(), float(loss), step))
        return before, steps, runtime.evaluate(include_predictions=True)

    frozen, _ = paired_runtime(baseline(), datasets, **options)
    old_init, old_steps, old_eval = take_steps(frozen)
    ctrl, _ = build_runtime(load_configs()[CTRL], datasets, **options)
    new_init, new_steps, new_eval = take_steps(ctrl)
    assert new_init == old_init and new_eval == old_eval and ctrl.pairing == frozen.pairing
    assert len(new_steps) == len(old_steps) == ctrl.optimizer_steps == frozen.optimizer_steps == 2
    for (actual, probabilities, loss, step), (expected, expected_probabilities, expected_loss, expected_step) in zip(
            new_steps, old_steps):
        assert torch.equal(actual, expected) and torch.equal(probabilities, expected_probabilities)
        assert (loss, step) == (expected_loss, expected_step)
    assert ctrl.identity == frozen.identity
    assert_models_equal(ctrl.model, frozen.model)
    assert confirm.frozen_model_source()


def test_b1_gradient_loss_parameter_count_and_inference_bypass(datasets, monkeypatch):
    from scripts.runtime.multidag_cl_fastkill import runtime as runtime_module

    ctrl, reference = build_runtime(load_configs()[CTRL], datasets, device="cpu", synthetic_only=True)
    arm, _ = build_runtime(load_configs()[B1], datasets, device="cpu", reference=reference, synthetic_only=True)
    assert_models_equal(arm.model, ctrl.model)
    assert sum(p.numel() for p in ctrl.model.parameters()) == sum(p.numel() for p in arm.model.parameters())
    assert sum(p.numel() for p in arm.model.parameters()) == F0_PARAMETER_COUNT
    assert sum(p.numel() for p in arm.training_parameters()) == F0_PARAMETER_COUNT + 1800
    assert arm.prototypes.shape == (6, 300)

    def forbidden(*args):
        raise AssertionError("Dev requested the training auxiliary objective")

    with monkeypatch.context() as patch:
        patch.setattr(runtime_module, "class_anchor_ce", forbidden)
        assert arm.evaluate(include_predictions=True) == ctrl.evaluate(include_predictions=True)
    before = arm.prototypes.detach().clone()
    step = arm.train_step(collate([datasets["train"][0], datasets["train"][1]]))
    assert math.isfinite(step["loss"]) and math.isfinite(arm._anchor_terms[0][0])
    assert torch.isfinite(arm.prototypes.grad).all() and torch.count_nonzero(arm.prototypes.grad)
    assert not torch.equal(before, arm.prototypes)
    assert (not arm.model.classifier[3]._forward_hooks
            and "training_only_class_prototypes" not in arm.model.state_dict())


def test_paper_similarity_prefix_only_suffix_mutation_and_convergence(datasets):
    assert wheel_similarity(0, 4) == pytest.approx(math.cos(math.pi / 5))
    assert wheel_similarity(0, 3) == 0 and wheel_similarity(2, 3) == 1 / 6
    assert wheel_similarity(3, 5) == pytest.approx(0.0, abs=1e-14)
    sample = list(from_dataset(datasets["train"]))[-1]
    labels, speakers = (0, 2, 4, 3, 3), (0, 1, 0, 1, 1)
    scores = prefix_scores(labels, speakers)
    assert scores[0] == scores[1] == (0.5, 0.5, 0.0)
    assert scores[2][2] == pytest.approx(math.cos(math.pi / 5) + 0.4)
    assert scores[3][2] == pytest.approx(1 / 6 + 0.4)
    assert scores[-1][2] == 0
    assert prefix_scores(labels[:3], speakers[:3]) == scores[:3]
    assert require_suffix_invariance(sample, prefix_scores(sample.labels, sample.speakers)) == len(sample.labels) - 1
    difficulty = [0.1, 0.5, 1.4]
    weights = [alpha_values(difficulty, epoch) for epoch in range(1, 31)]
    assert weights[0][0] > weights[0][1] > weights[0][2] == pytest.approx(0.9)
    assert all(previous <= current for previous, current in zip(weights, weights[1:]))
    assert weights[7:] == [[1.0] * len(difficulty)] * 23


def test_a1_loss_reweights_only_valid_targets_and_rejoins_exact_ce(datasets):
    dialogues = list(from_dataset(datasets["train"]))
    difficulties = {d.dialogue_id: tuple(row[1] for row in prefix_scores(d.labels, d.speakers)) for d in dialogues}
    arm, _ = build_runtime(load_configs()[A1], datasets, device="cpu", synthetic_only=True,
                           prefix_difficulties=difficulties)
    batch = collate([datasets["train"][0], datasets["train"][2]])
    from scripts.runtime.multidag_cl_ta_v0.training import masked_cross_entropy

    output, actual = arm.forward_batch(batch, split="train")
    targets = batch["labels"][output.mask]
    alpha = [weight for did in batch["dialogue_ids"] for weight in alpha_values(difficulties[did], 1)]
    ce = torch.nn.functional.cross_entropy(output.logits_ta[output.mask], targets, reduction="none")
    assert torch.equal(actual, (torch.tensor(alpha) * ce).mean())
    arm.epoch = 7
    output, actual = arm.forward_batch(batch, split="train")
    expected, _ = masked_cross_entropy(output.logits_ta, batch["labels"], output.mask)
    assert torch.equal(actual, expected)


def test_config_combo_and_no_test_access(datasets, monkeypatch):
    configs = load_configs()
    changed = deepcopy(configs[B1])
    changed["prefix_wes"]["enabled"] = True
    with pytest.raises(ValueError, match="UNAUTHORIZED_COMBINATION"):
        validate_arm(changed)
    with pytest.raises(ValueError, match="A1 requires"):
        build_runtime(configs[A1], datasets, device="cpu", synthetic_only=True)
    arm, _ = build_runtime(configs[B1], datasets, device="cpu", synthetic_only=True)
    for evaluate in (lambda: arm.loader("test"), lambda: arm.dataset("test"),
                     lambda: arm.evaluate(split="test"), lambda: represent(arm, "test"),
                     lambda: arm.forward_batch({}, split="test"), arm.evaluate_test):
        with pytest.raises(ValueError, match="Test"):
            evaluate()
    with pytest.raises(ValueError, match="Train"):
        list(from_dataset(datasets["dev"]))
    monkeypatch.setattr(launcher.sys, "platform", "win32")
    monkeypatch.setattr(launcher, "load_train_dev", lambda *_: pytest.fail("Windows opened formal data"))
    with pytest.raises(SystemExit, match="2"):
        launcher.main(["--execute"])


def test_a0_certified_manifest_does_not_consume_held_labels(artifact_dir, monkeypatch):
    # Each held split contains deliberately unparsable targets. A0 needs only
    # identity fields from those rows to verify the shared manifest certificate.
    from scripts.runtime.multidag_cl_fastkill import prefix_diagnostic

    rows = [
        {"utterance_id": f"{split}{idx}", "dialogue_id": split, "split": split, "speaker": "F",
         "canonical_dialogue_order": str(order), "asset_dialogue_id": split, "asset_row_index": str(idx),
         "benchmark_label": label, "benchmark_label_id": target, "audio_sha256": "unused"}
        for idx, (split, order, label, target) in enumerate(
            (("train", 2, "exc", "4"), ("test", 1, "hold", "invalid"), ("train", 1, "hap", "0"),
             ("dev", 1, "hold", "invalid")))
    ]
    monkeypatch.setattr(prefix_diagnostic, "COUNTS", {**COUNTS, "train": {"dialogues": 1, "utterances": 2}})
    manifest, provenance = artifact_dir / "manifest.csv", artifact_dir / "provenance.json"
    with manifest.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    provenance.write_text(json.dumps({"schema": SCHEMA, "status": "complete", "test_labels_inspected": 0,
                                      "model_manifest_sha256": sha256_file(manifest),
                                      "row_order_identity_sha256": identity_sha(rows),
                                      "split_counts": prefix_diagnostic.COUNTS}), encoding="utf-8")
    dialogue, = from_manifest(manifest, provenance)
    assert dialogue.labels == (0, 4) and dialogue.speakers == (0, 0) and dialogue.utterance_ids == ("train2", "train0")
    manifest.write_text("tampered", encoding="utf-8")
    with pytest.raises(ValueError, match="certified"):
        list(from_manifest(manifest, provenance))


def test_synthetic_batch_matches_steps_targets_tape_and_exports(artifact_dir, datasets):
    # Exercise multiple batches and A1's first uniform epoch.
    train = datasets["train"]
    for index in range(10):
        item = train.items[index % 7].copy()
        item["dialogue_id"] = f"extra_{index}"
        item["utterance_ids"] = tuple(f"extra_{index}_{j}" for j in range(len(item["utterance_ids"])))
        train.items.append(item)
    paths = resolve_output_paths(output_base=artifact_dir, experiment_date="20261123",
                                 experiment_group=GROUP, batch_id="synthetic")
    archive = artifact_dir / "fastkill.tar.gz"
    state = launcher.run_batch(paths, load_configs(), lambda: (datasets, {"id": "synthetic"}, {}),
                               {"git_head": "synthetic"}, device="cpu", archive=archive, synthetic_epochs=8)
    assert state["status"] == state["ANALYSIS_BUNDLE"] == "PASS"
    assert state["A_ROUTE"] == state["B_ROUTE"] == "NOT_EVALUATED_SYNTHETIC"
    first = state["jobs"][CTRL]["result"]
    assert first["optimizer_steps"] == 16
    for arm in ARMS:
        result = state["jobs"][arm]["result"]
        assert result["batch_pairing"] == first["batch_pairing"]
        assert result["optimizer_steps"] == first["optimizer_steps"]
    with tarfile.open(archive, "r:gz") as bundle:
        def read(name):
            return bundle.extractfile(name).read()

        manifest = json.loads(read("analysis_bundle_manifest.json"))
        assert all(hashlib.sha256(read(name)).hexdigest() == sha for name, sha in manifest["files_sha256"].items())
        assert json.loads(read("test_hold_status.json")) == HOLD
        target_count = sum(len(item["labels"]) for item in train)
        for arm in ARMS:
            rows = list(csv.DictReader(read(f"epoch_history_{arm}_seed100.csv").decode().splitlines()))
            assert [int(row["optimizer_steps"]) for row in rows] == list(range(2, 17, 2))
            assert {int(row["train_targets"]) for row in rows} == {target_count}
        alpha = list(csv.DictReader(read(f"{A1}/alpha_distribution.csv").decode().splitlines()))
        assert float(alpha[0]["max"]) < 1 and float(alpha[-1]["min"]) == 1
        with np.load(io.BytesIO(read(f"{B1}/geometry/dev_geometry.npz")), allow_pickle=False) as geometry:
            assert geometry["representation"].shape[1] == 300
            assert geometry["centroid"].shape == (6, 300)
            assert geometry["nearest_centroid_confusion"].shape == (6, 6)
        assert json.loads(read("A0/a0_manifest.json"))["PREFIX_SUFFIX_INVARIANCE"] == "PASS"
    with pytest.raises(FileExistsError):
        launcher.run_batch(paths, load_configs(), lambda: pytest.fail("collision opened data"), {},
                           device="cpu", archive=archive, synthetic_epochs=1)


def test_route_gates_avoid_opposing_pair_regression_and_require_final_a1():
    from scripts.runtime.multidag_cl_ta_v0.training import classification_report

    ctrl = dict.fromkeys(("accuracy", "weighted_f1", "macro_f1", "uar"), 0.5)
    b1 = {**ctrl, "weighted_f1": 0.503, "macro_f1": 0.502}
    a1 = {**ctrl, "weighted_f1": 0.503}
    selected, final = {CTRL: ctrl, B1: b1, A1: a1}, {CTRL: ctrl, B1: b1, A1: ctrl}
    hap, exc = 0, 4
    targets = [hap] * 100 + [exc] * 100
    baseline_predictions = [exc] * 20 + [hap] * 80 + [hap] * 20 + [exc] * 80
    one_way = [exc] * 2 + [hap] * 98 + [hap] * 25 + [exc] * 75
    both_ways = [exc] * 17 + [hap] * 83 + [hap] * 18 + [exc] * 82
    control = classification_report(targets, baseline_predictions, loss=1.0)
    evidence = {CTRL: control, B1: classification_report(targets, one_way, loss=1.0)}
    routes, _, pairs = route_decisions(selected, final, evidence, pairing_ok=True, suffix_ok=True, synthetic=False)
    assert len(pairs) == len(GATE_PAIRS) and not pairs[0]["pair_gate"]
    assert routes == {"B_ROUTE": "STOP", "A_ROUTE": "STOP"}
    evidence[B1] = classification_report(targets, both_ways, loss=1.0)
    routes, _, _ = route_decisions(selected, {**final, A1: a1}, evidence, pairing_ok=True,
                                   suffix_ok=True, synthetic=False)
    assert routes == {"B_ROUTE": "KEEP_FOR_CONFIRMATION", "A_ROUTE": "KEEP_FOR_CONFIRMATION"}
    routes, _, _ = route_decisions(selected, {**final, A1: a1}, evidence, pairing_ok=False,
                                   suffix_ok=True, synthetic=False)
    assert routes["A_ROUTE"] == "INVALID"

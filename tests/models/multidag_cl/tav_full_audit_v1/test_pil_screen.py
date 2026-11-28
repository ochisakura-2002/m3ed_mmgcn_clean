from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import torch

from datasets.iemocap.tav_full_audit_dataset import collate
from models.multidag_cl.fusion.model import FusionTAV
from models.experimental.multidag_cl_pil_screen import C01FusionTAV, predecessors
from scripts.analysis.models.multidag_cl.pil_screen_analysis import export_analysis, plot_tables, verdict
from scripts.runtime.multidag_cl_pil_screen import (
    ARMS, CONTROL, EXPECTED_PARAMETERS, InvalidRun, PILRuntime, build_target_tape,
    device_parity, train_arm, validate_target_tape,
)
from scripts.runtime.multidag_cl_fusion.runtime import paired_runtime, training_rng_identity
from scripts.runtime.multidag_cl_fusion.spec import make_final_config
from scripts.runtime.multidag_cl_paper_reimplementation.checkpoint import load_checkpoint
from scripts.runtime.multidag_cl_ta_v0.formal import json_identity, sha256_file
from scripts.runtime.multidag_cl_ta_v0.training import classification_report
from scripts.runtime.multidag_cl_tav_full_audit_v1.runtime import MODEL_INPUTS, SyntheticDialogues
from scripts.workflows.benchmarks import run_pil_screen
from utils.output_paths import resolve_output_paths


def datasets():
    result = {split: SyntheticDialogues(split) for split in ("train", "dev")}
    for i, item in enumerate(result["train"].items):
        n = len(item["labels"])
        item["speaker_ids_int"] = torch.zeros(n, dtype=torch.long)
        item["labels"] = torch.tensor([i % 2] + [0] * (n - 1))
    return result


def train_population(ds):
    return [(uid, item["dialogue_id"], int(label))
            for item in ds for uid, label in zip(item["utterance_ids"], item["labels"])]


def runtime(ds, arm, *, tape=None, pairing=None):
    raw = make_final_config("F0", 100)
    return PILRuntime(raw, ds, arm, experiment={"arm": arm, "target_tape_sha256": json_identity(tape)},
                      targets=tape, pairing=pairing, synthetic_only=True, device="cpu")


def args(batch):
    return {key: batch[key] for key in MODEL_INPUTS}


def shorten(item, length):
    return {key: value[:length] if isinstance(value, (torch.Tensor, tuple)) else value for key, value in item.items()}


def test_predecessor_is_strict_past_and_dialogue_local():
    mask = torch.tensor([[True, True, True, True, True], [True, True, True, False, False]])
    speaker = torch.tensor([[0, 0, 1, 0, 1], [0, 1, 1, 0, 0]])
    all_edges = torch.ones(2, 5, 5, dtype=torch.bool)
    expected = torch.tensor([[-1, 0, -1, 1, 2], [-1, -1, 1, -1, -1]])
    assert torch.equal(predecessors(all_edges, speaker, mask), expected)
    assert torch.equal(predecessors(all_edges, 17 - speaker, mask), expected)


def test_initialization_full_f0_path_and_both_candidate_contracts():
    ds = datasets()
    tape = build_target_tape(ds["train"], train_population(ds["train"]))
    canonical, _ = paired_runtime(make_final_config("F0", 100), ds, device="cpu", synthetic_only=True)
    original = runtime(ds, CONTROL)
    all_arms = [runtime(ds, arm, tape=tape) for arm in ARMS]
    assert type(original.model) is FusionTAV
    assert original.initialization == canonical.initialization
    assert device_parity(original, all_arms)["same_shape_zero_init"] == "EXACT_VALUES"
    for instance in all_arms:
        assert instance.initialization == original.initialization
        assert instance.parameter_count == EXPECTED_PARAMETERS[instance.arm]
        if instance.aux is not None:
            assert type(instance.model) is FusionTAV
            assert not any(parameter.count_nonzero() for parameter in instance.aux.parameters())
        else:
            assert instance.model.side_weight.count_nonzero() == 0
        assert instance.optimizer.state_dict()["param_groups"][0]["foreach"] is False


def test_shared_off_optimizer_update_matches_original():
    ds = datasets()
    canonical, _ = paired_runtime(make_final_config("F0", 100), ds, device="cpu", synthetic_only=True)
    old_tape = list(canonical.loader("train", epoch=1))
    expected = canonical.train_step(old_tape[0])
    original_rng = training_rng_identity("cpu")
    off = runtime(ds, CONTROL)
    actual_tape = list(off.loader("train", epoch=1))
    actual = off.train_step(actual_tape[0])
    assert actual["objective"] == actual["hard_ce"] == expected["loss"]
    assert actual["gradient_norm"] == expected["gradient_norm"]
    assert training_rng_identity("cpu") == original_rng
    assert all(torch.equal(a, b) for a, b in zip(canonical.model.parameters(), off.model.parameters()))


def test_full_sampler_tape_uses_real_dataloader_iteration():
    ds = datasets()
    template = ds["train"][0]
    ds["train"].items = [{**template, "dialogue_id": f"d{i}", "utterance_ids": (f"u{i}_0", f"u{i}_1")}
                         for i in range(100)]
    off = runtime(ds, CONTROL)
    for epoch in range(1, 31):
        list(off.loader("train", epoch=epoch))
    list(off.loader("dev"))
    rng = training_rng_identity("cpu")
    tape = off.audit_sampler(off.pairing)
    assert training_rng_identity("cpu") == rng
    assert len(tape["train"]) == 30
    assert [len(batch["dialogue_ids"]) for batch in tape["train"]["30"]] == [16] * 6 + [4]
    invalid = deepcopy(off.pairing)
    invalid["train"]["1"]["batch_sha256"][0] = "0" * 64
    with pytest.raises(InvalidRun, match="historical tape"):
        off.audit_sampler(invalid)


@torch.no_grad()
def test_c01_source_missing_prefix_and_padding():
    ds = datasets()
    item = dict(ds["train"][2])
    item["speaker_ids_int"] = torch.tensor([0, 0, 1, 0])
    current, prev = [runtime(ds, arm).model.eval() for arm in ARMS[:2]]
    original = runtime(ds, CONTROL).model.eval()
    second = shorten(item, 2)
    second["dialogue_id"] = "independent_dialogue"
    second["utterance_ids"] = ("independent_0", "independent_1")
    batch = collate([item, second])
    for model in (current, prev):
        model.side_weight[:, :].copy_(torch.linspace(-.03, .03, 200).expand(300, -1))
    supplied = args(batch)
    for model in (current, prev):
        first_inputs = args(collate([shorten(item, 3), second]))
        original_full = original(**supplied).logits_ta[:, :3]
        original_prefix = original(**first_inputs).logits_ta
        calibrated = max(float((original_full - original_prefix).abs().max()),
                         8 * torch.finfo(original_prefix.dtype).eps)
        full = model(**supplied)
        assert torch.allclose(full.logits_ta[:, :3], model(**first_inputs).logits_ta, atol=calibrated, rtol=0)
        future = {key: tensor.clone() for key, tensor in supplied.items()}
        for key in ("audio_features", "visual_features", "text_features"):
            future[key][0, 3] += 19
        assert torch.equal(full.logits_ta[:, :3], model(**future).logits_ta[:, :3])
        assert torch.allclose(full.logits_ta[1, :2], model(**args(collate([second]))).logits_ta[0],
                              atol=calibrated, rtol=0)
        assert not full.logits_ta[~full.mask].count_nonzero()
        visible = torch.ones((*full.mask.shape, 3), dtype=torch.bool)
        visible[:, 0, 0] = False
        masked = model(**supplied, availability=visible).logits_ta
        baseline = original(**supplied, availability=visible).logits_ta
        assert torch.equal(masked[:, :3], baseline[:, :3])
        visible[:, 0, 0] = True
        visible[:, 1, 1] = False
        assert torch.equal(model(**supplied, availability=visible).logits_ta,
                           original(**supplied, availability=visible).logits_ta)
        with pytest.raises(ValueError, match="complete TAV"):
            model.train()(**supplied, availability=visible)
        model.eval()
    assert not torch.equal(current(**supplied).logits_ta, prev(**supplied).logits_ta)


def test_c07_target_provenance_and_binding():
    ds = datasets()
    expected = train_population(ds["train"])
    tape = build_target_tape(ds["train"], expected)
    digest = json_identity(tape)
    assert tape["changed_count"] > 0
    validate_target_tape(deepcopy(tape), tape, digest)
    for label in range(6):
        values = [row for row in tape["targets"].values() if row[3] == label and row[4] is not None]
        assert sorted(row[-1] for row in values) == sorted(row[-2] for row in values)
    for bad in ({**tape, "split": "dev"}, {**tape, "mapping_sha256": "0" * 64},
                {**tape, "targets": dict(list(tape["targets"].items())[1:])}):
        with pytest.raises(InvalidRun):
            validate_target_tape(bad, tape, digest)
    with pytest.raises(InvalidRun):
        validate_target_tape(tape, tape, "0" * 64)
    with pytest.raises(InvalidRun):
        build_target_tape(ds["dev"], expected)
    with pytest.raises(InvalidRun):
        build_target_tape(ds["train"], expected[:-1])
    instance = runtime(ds, "C07-TRUE", tape=tape)
    batch = collate([ds["train"][2]])
    actual, _ = instance.forward_batch(batch, split="train")
    with pytest.raises(InvalidRun, match="outside Train"):
        instance.auxiliary_loss({**batch, "split": "dev"}, actual)
    ids = batch["utterance_ids"][0]
    for invalid_ids in ([ids[1:]], [(ids[0], ids[0], *ids[2:])], [("unexpected", *ids[1:])]):
        with pytest.raises(InvalidRun):
            instance.auxiliary_loss({**batch, "utterance_ids": invalid_ids}, actual)
    before = training_rng_identity("cpu")
    dev = collate([ds["dev"][1]])
    changed = {**dev, "labels": (dev["labels"] + 1) % 6}
    r1 = instance.evaluate([dev], include_predictions=True)
    r2 = instance.evaluate([changed], include_predictions=True)
    assert [row["logits"] for row in r1["predictions"]] == [row["logits"] for row in r2["predictions"]]
    assert json_identity(tape) == digest and training_rng_identity("cpu") == before


def test_c07_aux_denominator_zero_init_empty_and_gather_gradients():
    ds = datasets()
    tape = build_target_tape(ds["train"], train_population(ds["train"]))
    instance = runtime(ds, "C07-TRUE", tape=tape)
    batch = collate([ds["train"][2], shorten(ds["train"][1], 1)])
    output, _ = instance.forward_batch(batch, split="train")
    aux, count, _ = instance.auxiliary_loss(batch, output)
    assert count == 3 and float(aux) == pytest.approx(3 * np.log(2) / 5)
    parameters = tuple(instance.model.parameters())
    shared = torch.autograd.grad(aux, parameters, allow_unused=True, retain_graph=True)
    assert all(g is None or not g.count_nonzero() for g in shared)
    aux_grads = torch.autograd.grad(aux, tuple(instance.aux.parameters()), retain_graph=True)
    assert any(grad.count_nonzero() for grad in aux_grads)
    with torch.no_grad():
        instance.aux["weight"].fill_(0.001)
    aux, _, _ = instance.auxiliary_loss(batch, output)
    gradient = torch.autograd.grad(aux, output.h_ta)[0]
    assert gradient[0, 0].count_nonzero() and gradient[0, 3].count_nonzero()
    assert not gradient[1].count_nonzero()
    assert torch.allclose(gradient[0].sum(0), torch.zeros_like(gradient[0, 0]), atol=1e-9, rtol=0)
    empty_batch = collate([shorten(ds["train"][1], 1)])
    empty, _ = instance.forward_batch(empty_batch, split="train")
    loss, n, positive = instance.auxiliary_loss(empty_batch, empty)
    assert float(loss) == n == positive == 0 and not loss.requires_grad


def test_same_run_epoch_resume_aux_state_and_strict_export(artifact_dir, monkeypatch):
    ds = datasets()
    tape = build_target_tape(ds["train"], train_population(ds["train"]))
    control = train_arm(runtime(ds, CONTROL), artifact_dir / "off", epochs=2)
    uninterrupted = runtime(ds, "C07-TRUE", tape=tape, pairing=control["batch_pairing"])
    result = train_arm(uninterrupted, artifact_dir / "uninterrupted", epochs=2)
    interrupted = runtime(ds, "C07-TRUE", tape=tape, pairing=control["batch_pairing"])
    perform = interrupted.train_step

    def interrupt(batch):
        if interrupted.optimizer_steps == 1:
            raise KeyboardInterrupt
        return perform(batch)

    monkeypatch.setattr(interrupted, "train_step", interrupt)
    retry = artifact_dir / "retry"
    with pytest.raises(KeyboardInterrupt):
        train_arm(interrupted, retry, epochs=2)
    resumed = runtime(ds, "C07-TRUE", tape=tape, pairing=control["batch_pairing"])
    actual = train_arm(resumed, retry, epochs=2, resume=True)
    assert actual["Best"] == result["Best"] and actual["E30"] == result["E30"]
    assert actual["batch_pairing"] == result["batch_pairing"]
    a, b = [load_checkpoint(p / "checkpoints/last_model.pt", torch.device("cpu"))
            for p in (retry, artifact_dir / "uninterrupted")]
    for key in ("model_state_dict", "aux_state"):
        assert a[key].keys() == b[key].keys()
        assert all(torch.equal(v, b[key][name]) if isinstance(v, torch.Tensor) else v == b[key][name]
                   for name, v in a[key].items())
    deploy = torch.load(retry / "checkpoints/deploy_state_dict.pt", weights_only=True)
    assert FusionTAV(resumed.model.audit_config).load_state_dict(deploy, strict=True).missing_keys == []
    with pytest.raises(InvalidRun, match="unfinished"):
        train_arm(runtime(ds, "C07-TRUE", tape=tape), retry, epochs=2, resume=True)


@pytest.mark.parametrize("arm", ARMS[:2] + ("C07-NULL",))
def test_matched_arms_synthetic_training_and_reload(arm, artifact_dir):
    ds = datasets()
    tape = build_target_tape(ds["train"], train_population(ds["train"]))
    candidate = runtime(ds, arm, tape=tape)
    result = train_arm(candidate, artifact_dir / "run", epochs=1)
    assert result["optimizer_steps"] == 1
    assert result["history"][0]["steps"][0]["added_gradient_norm"] > 0
    state = torch.load(artifact_dir / "run/checkpoints/deploy_state_dict.pt", weights_only=True)
    if arm.startswith("C01"):
        C01FusionTAV(candidate.model.audit_config, arm).load_state_dict(state, strict=True)
        other_arm = next(name for name in ARMS[:2] if name != arm)
        with pytest.raises(ValueError, match="identity mismatch"):
            C01FusionTAV(candidate.model.audit_config, other_arm).load_state_dict(state, strict=True)
    else:
        FusionTAV(candidate.model.audit_config).load_state_dict(state, strict=True)


def test_analysis_csv_figures_and_sealed_publication(artifact_dir):
    dev = []
    for index in range(20):
        dialogue = f"Ses04F_impro{index:02d}"
        dev.append({"dialogue_id": dialogue, "utterance_ids": [f"{dialogue}_{'FM'[i % 2]}{i:03d}" for i in range(49)],
                    "speaker_ids_int": torch.arange(49) % 2, "labels": (torch.arange(49) + index) % 6})
    expected = train_population(dev)
    report = classification_report([r[2] for r in expected], [r[2] for r in expected], loss=.3)
    probabilities = torch.tensor([2.0, 0, 0, 0, 0, 0]).softmax(0)
    report["predictions"] = [
        {"utterance_id": uid, "dialogue_id": did, "true_label": gold, "predicted_label": gold,
         "logits": [2.0 if k == gold else 0.0 for k in range(6)],
         "probabilities": [float(probabilities[0 if k == gold else 1]) for k in range(6)]}
        for uid, did, gold in expected]
    paths = resolve_output_paths(output_base=artifact_dir, experiment_date="20261128",
                                 experiment_group="pil_screen_001", batch_id="synthetic")
    stage = paths.analysis_root.parent / ".synthetic.build"
    stage.mkdir(parents=True)
    state = {"batch_id": "synthetic", "identity": {"baseline": {"arm": "D0"}}, "arms": {}}
    results = {}
    for arm in ("D0", *ARMS):
        step = {"valid_targets": 4830, "hard_ce": .3, "objective": .3, "aux_bce_per_valid": .1,
                "eligible": 4500, "aux_positive": 2000, "clipped": False, "update_seconds": .04}
        history = [{"epoch": i, "fit": {**report["metrics"], "valid_target_count": 4830}, "dev": report["metrics"],
                    "steps": [] if arm == "D0" else [step], "historical_train_update_loss": .3,
                    "seconds": .2, "gpu_peak_bytes": 42} for i in range(1, 31)]
        results[arm] = {"history": history, "Best": report, "E30": report, "best_epoch": 21,
                        "parameter_count": EXPECTED_PARAMETERS.get(arm, EXPECTED_PARAMETERS[CONTROL])}
    export_analysis(stage, results, expected, dev, state)
    plot_tables(stage)
    assert len(list((stage / "figures").glob("*.svg"))) == len(list((stage / "figures").glob("*.png"))) == 10
    assert "STOP" in (stage / "tables/verdicts.csv").read_text(encoding="utf-8")
    package = stage.with_name(stage.name + ".tar.gz")
    digest = run_pil_screen.d.ship(stage, package, "PASS")
    state["publication"] = {"staging_name": stage.name, "archive_sha256": digest,
                            "bundle_manifest_sha256": sha256_file(stage / run_pil_screen.d.MANIFEST_NAME)}
    paths.manifest_root.mkdir(parents=True)
    from os import link
    archive = stage.parent / "MERC_PIL_SCREEN_001_synthetic_analysis.tar.gz"
    link(package, archive)  # Simulate an interruption after linking the sealed package.
    run_pil_screen.publish_analysis(state, paths, paths.manifest_root / "state.json")
    assert state["phase"] == "complete" and sha256_file(Path(state["archive"])) == digest
    assert state["plot_ready"] == "YES" and (paths.analysis_root / "RESULTS.md").is_file()


@pytest.mark.parametrize(("f1", "acc", "decision"),
                         [(0.50, 0, "GO"), (0.20, 0, "REVIEW"), (0.49, .1, "REVIEW"),
                          (0.19, 0, "STOP"), (0.7, -.1, "REVIEW"), (-.01, 0, "STOP")])
def test_registered_classification_thresholds(f1, acc, decision):
    assert verdict(f1, acc) == decision

"""Synthetic probability files only; no feature caches, experts or real data."""

import csv
import json
from pathlib import Path
import subprocess
import sys
from uuid import uuid4

import numpy as np
import pytest
import torch
import yaml

from models.asymmetric_merc.dynamic_decision_gate import (
    ConstantDecisionGate, DynamicDecisionGate, build_model, gate_input,
)
from scripts.analysis.models.asymmetric_merc.dynamic_decision_report import (
    scientific_verdict, write_decision_report,
)
from scripts.analysis.models.asymmetric_merc.vrg_report import assign_quadrants
from scripts.analysis.models.asymmetric_merc.weighted_fusion import (
    AuditError, exact_join, fusion_scores, load_predictions, metrics,
)
from scripts.models.asymmetric_merc import train_dynamic_decision_gate as runtime

ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture
def tmp_path():
    # Workspace-temp convention avoids managed Windows temporary-directory ACLs.
    path = ROOT / "tmp/assistant_work/decision_gate_checks" / uuid4().hex
    path.mkdir(parents=True)
    return path


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _write(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t" if path.suffix == ".tsv" else ",")
        writer.writeheader()
        writer.writerows(rows)
    return path


def _rows(split, visual=False):
    rows = []
    for i in range(12):
        true = i % 6
        # Each group of four has Q11, Q10, Q01, Q00 respectively.
        correct = i % 4 in ({0, 2} if visual else {0, 1})
        pred = true if correct else (true + 1 + int(visual)) % 6
        probs = [.04] * 6
        probs[pred] = .80
        rows.append({"utterance_id": f"{split}-u{i:02d}", "true_label": true,
                     "pred_label" if visual else "predicted_label": pred, "split": split,
                     **{f"prob_{k}": value for k, value in enumerate(probs)}})
    return rows


def _inputs(path):
    result = {}
    for split in ("train", "dev"):
        for expert in ("ta", "v"):
            rows = _rows(split, visual=expert == "v")
            if expert == "ta":
                extra = dict(rows[0], utterance_id=f"{split}-ta-only")
                rows.append(extra)
            result[f"{expert}_{split}"] = _write(path / f"{expert}_{split}.{'tsv' if expert == 'ta' else 'csv'}", rows)
    return result


def _args(paths, output, *, synthetic=True):
    args = ["--output-dir", str(output)]
    for name, path in paths.items():
        args.extend(["--" + name.replace("_", "-"), str(path)])
    if synthetic:
        args.extend(["--synthetic", "--smoke-epochs", "2"])
    return args


def _join(paths, split="dev"):
    return exact_join(load_predictions(paths[f"ta_{split}"], split=split),
                      load_predictions(paths[f"v_{split}"], split=split))


def _json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _reference(paths, directory):
    joined = _join(paths)
    scores = fusion_scores(joined, .88)
    rows = [{"utterance_id": uid, "true_label": int(joined.true[i]),
             "predicted_label": int(scores[i].argmax()), "alpha": .88,
             **{f"score_{k}": float(scores[i, k]) for k in range(6)}} for i, uid in enumerate(joined.ids)]
    pred = _write(directory / "b0_predictions.csv", rows)
    metric = directory / "b0_metrics.json"
    metric.write_text(json.dumps({"alpha": .88, "common_rows": len(rows), "synthetic": True,
                                  "fusion": metrics(joined.true, scores.argmax(1))}), encoding="utf-8")
    return {"b0_predictions": pred, "b0_metrics": metric}


def test_exact_join_shuffling_duplicates_label_mismatch_and_split(tmp_path):
    paths = _inputs(tmp_path)
    expected = _join(paths)
    assert len(expected.ids) == 12 and expected.audit["ta_only_ids"] == ["dev-ta-only"]
    for split in ("train", "dev"):
        for expert in ("ta", "v"):
            rows = _rows(split, visual=expert == "v")
            _write(paths[f"{expert}_{split}"], rows[::-1])
    actual = _join(paths)
    assert expected.ids == actual.ids
    assert np.array_equal(expected.ta_probabilities, actual.ta_probabilities)
    rows = _rows("dev", visual=True)
    _write(paths["v_dev"], rows + rows[:1])
    with pytest.raises(AuditError, match="duplicate IDs"):
        _join(paths)
    rows[0]["true_label"] = 5
    _write(paths["v_dev"], rows)
    with pytest.raises(AuditError, match="label mismatch"):
        _join(paths)
    with pytest.raises(AuditError, match="non-Dev split"):
        load_predictions(paths["ta_train"])
    assert load_predictions(paths["ta_train"], split="train").manifest["split_evidence"] == "validated_column"


@pytest.mark.parametrize("damage", ["negative", "nan", "inf", "sum", "argmax", "label"])
def test_probability_validation(tmp_path, damage):
    rows = _rows("train")
    if damage == "negative":
        rows[0].update(prob_1=-.01, prob_0=.85)
    elif damage in {"nan", "inf"}:
        rows[0]["prob_1"] = float(damage)
    elif damage == "sum":
        rows[0]["prob_0"] = .9
    elif damage == "argmax":
        rows[0]["predicted_label"] = 1
    else:
        rows[0]["true_label"] = 6
    with pytest.raises(AuditError):
        load_predictions(_write(tmp_path / "invalid.csv", rows), split="train")


def test_shapes_parameters_init_clamp_no_renormalization_and_frozen_experts(tmp_path):
    data = _join(_inputs(tmp_path))
    ta = torch.from_numpy(data.ta_probabilities).requires_grad_(True)
    visual = torch.from_numpy(data.v_probabilities).requires_grad_(True)
    ta.data[0] = torch.tensor([1., 0., 0., 0., 0., 0.], dtype=torch.float64)
    ta.data[1] *= 1 + 1e-6  # Legal reader tolerance; never renormalize it.
    model = DynamicDecisionGate()
    result = model(ta, visual, return_details=True)
    assert result["q"].shape == (12, 24) and result["g"].shape == (12, 1)
    assert sum(p.numel() for p in model.parameters()) == 833
    assert torch.equal(model.final_gate_layer.weight, torch.zeros_like(model.final_gate_layer.weight))
    assert model.final_gate_layer.bias.item() == pytest.approx(-1.9924301646902063, abs=1e-15)
    assert torch.equal(result["g"], torch.full((12, 1), .12, dtype=torch.float64))
    assert torch.equal(result["q"][:, :6], ta.detach().clamp(min=1e-12))
    assert torch.equal(result["q"][:, 12:18], (ta.detach().clamp(min=1e-12) - visual.detach()).abs())
    assert torch.equal(result["q"][:, 18:], ta.detach().clamp(min=1e-12) * visual.detach())
    loss = torch.nn.functional.cross_entropy(result["scores"], torch.from_numpy(data.true))
    loss.backward()
    assert torch.isfinite(loss)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert model.final_gate_layer.weight.grad.abs().sum() > 0
    assert ta.grad is None and visual.grad is None
    # Zero final weights correctly give the hidden layer zero gradient on the first step.
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.01)
    optimizer.step()
    optimizer.zero_grad()
    torch.nn.functional.cross_entropy(model(ta, visual), torch.from_numpy(data.true)).backward()
    assert model.hidden.weight.grad.abs().sum() > 0
    with pytest.raises(ValueError, match="shape"):
        gate_input(ta[:, :5], visual)


def test_epoch0_constant_and_fixed_alpha_scores_predictions_metrics_identical(tmp_path):
    data = _join(_inputs(tmp_path))
    ta, visual = torch.from_numpy(data.ta_probabilities), torch.from_numpy(data.v_probabilities)
    initial = DynamicDecisionGate()(ta, visual).detach().numpy()
    constant = ConstantDecisionGate()(ta, visual).numpy()
    fixed = fusion_scores(data, .88)
    assert np.array_equal(initial, constant)
    assert np.allclose(initial, fixed, atol=1e-14, rtol=0)
    assert np.array_equal(initial.argmax(1), fixed.argmax(1))
    assert metrics(data.true, initial.argmax(1)) == metrics(data.true, fixed.argmax(1))
    assert sum(p.numel() for p in build_model(enabled=False).parameters()) == 0


def test_strict_selection_and_patience():
    select = runtime.CheckpointSelection(0, .57)
    assert not select.consider(1, .57) and select.best_epoch == 0
    for epoch in range(2, 10):
        assert not select.consider(epoch, .56)
    assert not select.should_stop()
    assert not select.consider(10, .57) and select.should_stop() and select.best_epoch == 0
    assert select.consider(11, .5700000000000001)  # Strict >, without a tolerance tie.
    assert select.best_epoch == 11 and select.waiting == 0
    assert not select.consider(12, .57) and select.best_epoch == 11


def test_quadrants_and_mechanism(tmp_path):
    data = _join(_inputs(tmp_path))
    assert list(assign_quadrants(data.true, data.ta_predicted, data.v_predicted)) == ["Q11", "Q10", "Q01", "Q00"] * 3
    gates = np.array([.12, .1, .3, .2] * 3)
    output = tmp_path / "report"
    output.mkdir()
    report = write_decision_report(output, data, data.ta_predicted, gates,
                                   best_metrics={"weighted_f1": .6}, baseline={"weighted_f1": .57}, synthetic=False)
    assert report["delta_rescue_harm"] == pytest.approx(.2)
    assert report["all"]["min"] == .1 and report["all"]["max"] == .3
    assert report["all"]["std"] == pytest.approx(gates.std())
    assert all(v["count"] == 3 for v in report["quadrants"].values())
    assert sum(v["count"] for v in report["true_classes"].values()) == 12
    assert report["mean_abs_deviation_from_0_12"] == pytest.approx(np.abs(gates - .12).mean())
    assert report["fraction_gate_gt_0_12"] == .5 and report["fraction_gate_lt_0_12"] == .25
    assert report["verdict"] == "STRONG_POSITIVE"
    with (output / "dev_gates.tsv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    assert list(rows[0]) == ["utterance_id", "true_label", "ta_pred", "v_pred", "dynamic_pred", "visual_gate", "quadrant"]
    assert [r["quadrant"] for r in rows] == list(assign_quadrants(data.true, data.ta_predicted, data.v_predicted))


@pytest.mark.parametrize("delta,direction,expected", [(0, .2, "NEGATIVE"), (-.001, .2, "NEGATIVE"),
    (.004999, .2, "WEAK"), (.005, .1, "STRONG_POSITIVE"), (.005, 0, "PERFORMANCE_ONLY"),
    (.006, -.1, "PERFORMANCE_ONLY"), (.006, None, "UNDETERMINED_MISSING_QUADRANT")])
def test_predeclared_scientific_criteria(delta, direction, expected):
    assert scientific_verdict(delta, direction) == expected


@pytest.mark.parametrize("slot", [*runtime.INPUTS, "config", "output_dir"])
@pytest.mark.parametrize("forbidden", ["test.csv", "testing/data.csv", "validation_test.csv", "TESTING.csv"])
def test_test_named_paths_fail_before_any_input_io(tmp_path, monkeypatch, slot, forbidden):
    args = runtime.build_parser().parse_args(_args(_inputs(tmp_path), tmp_path / "never"))
    setattr(args, slot, tmp_path / forbidden)
    calls = []

    def forbidden_io(*a, **k):
        calls.append(a)
        raise AssertionError("input IO occurred")

    monkeypatch.setattr(Path, "open", forbidden_io)
    monkeypatch.setattr(Path, "read_bytes", forbidden_io)
    monkeypatch.setattr(Path, "read_text", forbidden_io)
    with pytest.raises(ValueError, match="forbidden before IO"):
        runtime.run(args)
    assert not calls


@pytest.mark.parametrize("split", ["test", "testing", "validation_test", "train"])
def test_bad_split_fail_before_io(tmp_path, monkeypatch, split):
    args = runtime.build_parser().parse_args(_args(_inputs(tmp_path), tmp_path / "never"))
    args.split = split
    monkeypatch.setattr(Path, "open", lambda *a, **k: pytest.fail("IO before split rejection"))
    with pytest.raises(ValueError, match="before IO"):
        runtime.run(args)


def test_symlink_target_is_checked_before_reading(tmp_path):
    target = tmp_path / "testing"
    target.mkdir()
    link = tmp_path / "innocent"
    try:
        link.symlink_to(target, target_is_directory=True)
    except OSError as error:
        pytest.skip(f"OS does not permit creating symlinks: {error}")
    with pytest.raises(ValueError, match="before IO"):
        runtime.guard_path(link / "predictions.csv")


def test_synthetic_train_checkpoint_reload_outputs_and_no_feature_requirements(tmp_path):
    paths = _inputs(tmp_path)
    output = tmp_path / "smoke"
    assert runtime.main(_args(paths, output)) == 0
    assert {p.name for p in output.iterdir()} == {
        "best_model.pt", "epoch_metrics.csv", "best_dev_predictions.csv", "dev_gates.tsv", "best_dev_metrics.json",
        "gate_mechanism.json", "gate_mechanism.md", "resolved_config.yaml", "run_manifest.json", "epoch0_reproduction.json"}
    state = _json(output / "run_manifest.json")
    audit = _json(output / "epoch0_reproduction.json")
    assert state["status"] == "complete" and state["parameter_count"] == 833 and state["trained_epochs"] == 2
    assert state["base_train_predictions"] == "in_sample" and not state["cross_fitted"]
    assert state["base_prediction_training_status"] == {"TA": "in_sample", "V": "in_sample"}
    assert state["best_dev"]["weighted_f1"] >= audit["fixed_alpha_metrics"]["weighted_f1"]
    assert state["test_access"] is False and state["test_loader_built"] == state["test_evaluation_count"] == 0
    assert audit["status"] == "PASS" and all(audit["checks"].values())
    assert audit["gate_mean"] == pytest.approx(.12, abs=1e-15) and audit["gate_std"] <= 1e-15
    assert state["verdict"] == "SYNTHETIC_ONLY" and state["git_head"]
    checkpoint = torch.load(output / "best_model.pt", weights_only=True)
    assert checkpoint["epoch"] == state["best_epoch"]
    selected = load_predictions(output / "best_dev_predictions.csv")
    assert selected.ids == _join(paths).ids
    assert metrics(selected.true, selected.predicted)["weighted_f1"] == state["best_dev"]["weighted_f1"]
    report = (output / "gate_mechanism.md").read_text(encoding="utf-8")
    assert "in-sample rather than out-of-fold" in report and "SYNTHETIC ONLY" in report


def test_all_training_ties_preserve_epoch0_even_if_loss_improves(tmp_path, monkeypatch):
    paths = _inputs(tmp_path)
    actual = runtime.evaluate
    calls = []

    def improve_loss_only(model, loader, device):
        values, scores, gates = actual(model, loader, device)
        calls.append(values["weighted_f1"])
        values["loss"] = 1 / len(calls)
        return values, scores, gates

    monkeypatch.setattr(runtime, "evaluate", improve_loss_only)
    output = tmp_path / "ties"
    assert runtime.main(_args(paths, output)) == 0
    assert len(set(calls)) == 1
    assert _json(output / "run_manifest.json")["best_epoch"] == 0
    assert torch.load(output / "best_model.pt", weights_only=True)["epoch"] == 0


def test_disabled_gate_is_untrained_constant_sanity_control(tmp_path, monkeypatch):
    config = yaml.safe_load(runtime.DEFAULT_CONFIG.read_text(encoding="utf-8"))
    config["dynamic_decision_gate"]["enabled"] = False
    path = tmp_path / "constant.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    monkeypatch.setattr(torch.optim, "AdamW", lambda *a, **k: pytest.fail("Constant control must never construct optimizer"))
    output = tmp_path / "constant"
    assert runtime.main([*_args(_inputs(tmp_path), output), "--config", str(path)]) == 0
    result = _json(output / "run_manifest.json")
    assert result["best_epoch"] == result["trained_epochs"] == result["parameter_count"] == 0
    assert not result["optimizer_steps_allowed"]


def test_full_precision_reference_artifact_and_shuffle_reproduction(tmp_path):
    paths = _inputs(tmp_path)
    references = _reference(paths, tmp_path)
    output = tmp_path / "referenced"
    assert runtime.main(_args({**paths, **references}, output)) == 0
    audit = _json(output / "epoch0_reproduction.json")
    assert audit["reference_mode"] == "frozen_Q2_2_artifacts" and audit["checks"]["reference_predictions_exact"]
    for name, path in paths.items():
        rows = _rows("train" if name.endswith("train") else "dev", visual=name.startswith("v"))
        _write(path, rows[::-1])
    second = tmp_path / "reordered"
    assert runtime.main(_args({**paths, **references}, second)) == 0
    for name in ("epoch_metrics.csv", "best_dev_metrics.json", "best_dev_predictions.csv", "dev_gates.tsv", "gate_mechanism.json"):
        assert (output / name).read_bytes() == (second / name).read_bytes()


@pytest.mark.parametrize("damage", ["rounded_metric", "wrong_prediction", "id", "label", "missing_metric"])
def test_reference_mismatch_aborts_before_optimizer_and_checkpoint(tmp_path, monkeypatch, damage):
    paths = _inputs(tmp_path)
    if damage == "rounded_metric":
        # Make the artificial metric non-terminating; 0.5 would survive rounding.
        for expert in ("ta", "v"):
            path = paths[f"{expert}_dev"]
            with path.open(encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle, delimiter="\t" if path.suffix == ".tsv" else ","))
            rows[0]["true_label"] = 5
            _write(path, rows)
    references = _reference(paths, tmp_path)
    if damage in {"rounded_metric", "missing_metric"}:
        artifact = _json(references["b0_metrics"])
        if damage == "missing_metric":
            del artifact["fusion"]["uar"]
        else:
            artifact["fusion"]["weighted_f1"] = round(artifact["fusion"]["weighted_f1"], 6)
        references["b0_metrics"].write_text(json.dumps(artifact), encoding="utf-8")
    else:
        with references["b0_predictions"].open(encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        if damage == "id":
            rows[0]["utterance_id"] = "wrong-id"
        elif damage == "label":
            rows[0]["true_label"] = 5
        else:
            rows[0]["predicted_label"] = 1
            rows[0]["score_1"] = 0  # Valid scored artifact with a genuinely different decision.
        _write(references["b0_predictions"], rows)
    monkeypatch.setattr(torch.optim, "AdamW", lambda *a, **k: pytest.fail("Reproduction failure must precede optimizer"))
    output = tmp_path / "abort"
    assert runtime.main(_args({**paths, **references}, output)) == 1
    assert not (output / "best_model.pt").exists() and not (output / "epoch_metrics.csv").exists()
    if damage != "missing_metric":
        audit = _json(output / "epoch0_reproduction.json")
        assert audit["status"] == "ABORT_BEFORE_TRAINING" and not all(audit["checks"].values())


@pytest.mark.parametrize("difference,passed", [(5e-11, True), (2e-10, False)])
def test_reproduction_metric_tolerance_is_at_most_1e_minus_10(tmp_path, difference, passed):
    paths = _inputs(tmp_path)
    references = _reference(paths, tmp_path)
    data = _join(paths)
    ta, visual = torch.from_numpy(data.ta_probabilities), torch.from_numpy(data.v_probabilities)
    details = DynamicDecisionGate()(ta, visual, return_details=True)
    scores, gates = details["scores"].detach().numpy(), details["g"].detach().numpy().reshape(-1)
    observed = {**metrics(data.true, scores.argmax(1)), "loss": 1.}
    expected = dict(observed, weighted_f1=observed["weighted_f1"] + difference)
    report = runtime.epoch0_reproduction(data, observed, scores, gates, scores,
        load_predictions(references["b0_predictions"], score_columns=True), expected, synthetic=True)
    assert (report["status"] == "PASS") is passed
    assert report["metric_atol"] == 1e-10


def test_formal_gate_never_accepts_synthetic_count_or_absent_references(tmp_path):
    paths = _inputs(tmp_path)
    assert runtime.main(_args(paths, tmp_path / "missing", synthetic=False)) == 1
    references = _reference(paths, tmp_path)
    assert runtime.main(_args({**paths, **references}, tmp_path / "counts", synthetic=False)) == 1
    assert not (tmp_path / "counts").exists()


def test_protocol_is_frozen_and_no_expert_cache_cli(tmp_path):
    paths = _inputs(tmp_path)
    for key, value in (("seed", 101), ("lr", .01), ("selection", "dev_loss")):
        config = yaml.safe_load(runtime.DEFAULT_CONFIG.read_text(encoding="utf-8"))
        config["train"][key] = value
        path = tmp_path / f"changed_{key}.yaml"
        path.write_text(yaml.safe_dump(config), encoding="utf-8")
        assert runtime.main([*_args(paths, tmp_path / f"never_{key}"), "--config", str(path)]) == 1
    flags = {flag for action in runtime.build_parser()._actions for flag in action.option_strings}
    assert not flags & {"--h-ta", "--h-v", "--cache-dir", "--checkpoint", "--test", "--seed", "--lr"}
    code = ("import sys; from scripts.models.asymmetric_merc import train_dynamic_decision_gate; "
            "assert not any(n.startswith(('models.s2d','models.multidag_cl','datasets.',"
            "'scripts.models.asymmetric_merc.frozen_tav_cache')) for n in sys.modules)")
    result = subprocess.run([sys.executable, "-B", "-c", code], cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_existing_output_and_noncanonical_layout_fail_without_overwrite(tmp_path, monkeypatch):
    paths = _inputs(tmp_path)
    output = tmp_path / "existing"
    output.mkdir()
    sentinel = output / "keep.txt"
    sentinel.write_text("preserve", encoding="utf-8")
    assert runtime.main(_args(paths, output)) == 1
    assert list(output.iterdir()) == [sentinel] and sentinel.read_text(encoding="utf-8") == "preserve"
    monkeypatch.setattr(runtime, "PROJECT_ROOT", tmp_path)
    with pytest.raises(ValueError, match="outputs/<date>/<group>/runs"):
        runtime.reserve_output(tmp_path / "outputs/old_layout/run")
    canonical = tmp_path / "outputs/20261112/dynamic_decision_gate/runs/synthetic"
    assert runtime.reserve_output(canonical) == canonical

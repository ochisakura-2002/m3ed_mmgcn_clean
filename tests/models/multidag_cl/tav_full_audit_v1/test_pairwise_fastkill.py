from itertools import combinations
from pathlib import Path
from types import SimpleNamespace
from zipfile import ZipFile

import numpy as np
import pytest
import torch

from models.multidag_cl.fusion.model import FusionTAV
from models.multidag_cl.pairwise_refinement import PairwiseRefinementHead, bounded_writeback, pair_index_and_sign
from scripts.analysis.models.multidag_cl import pairwise_fastkill_metrics as metrics
from scripts.analysis.models.multidag_cl import pairwise_fastkill_runtime as runtime
from scripts.analysis.models.multidag_cl import run_pairwise_fastkill as runner
from scripts.analysis.models.multidag_cl.pairwise_preflight_inputs import (
    HOLD, F0_IDENTITY, FrozenSplit, TestHoldViolation as HoldViolation, frozen_forward, read_json,
)
from scripts.analysis.models.multidag_cl.pairwise_preflight_metrics import gate, scores, train_threshold
from scripts.runtime.multidag_cl_fusion.spec import make_final_config
from scripts.runtime.multidag_cl_ta_v0.formal import sha256_file
from scripts.runtime.multidag_cl_tav_full_audit_v1.runtime import SyntheticDialogues
from scripts.runtime.multidag_cl_tav_full_audit_v1.spec import read_csv


def tensors(rows=4):
    generator = torch.Generator().manual_seed(71)
    return tuple(torch.randn(rows, 300, generator=generator) for _ in range(2))


def sample_split(logits, labels):
    n = len(labels)
    return FrozenSplit(np.asarray(logits, dtype=np.float32),
                       tuple((f"utt{i}", f"dialogue{i // 3}", y) for i, y in enumerate(labels)),
                       tuple(t.numpy() for t in tensors(n)))


def test_parameters_and_fixed_pair_order():
    head = PairwiseRefinementHead()
    params = dict(head.named_parameters())
    assert {k: tuple(v.shape) for k, v in params.items()} == {
        "pair_embeddings": (15, 16), "projection.weight": (16, 600), "projection.bias": (16,),
    }
    assert sum(p.numel() for p in params.values() if p.requires_grad) == 9856
    projection = params["projection.weight"]
    assert projection.abs().max() <= (6 / (600 + 16)) ** 0.5
    assert projection.count_nonzero() > 0
    assert not params["projection.bias"].count_nonzero() and not params["pair_embeddings"].count_nonzero()
    assert head.normalize.eps == 1e-5 and not head.normalize.elementwise_affine
    a, b = torch.tensor(list(combinations(range(6), 2))).T
    for first, second, expected_sign in ((a, b, 1), (b, a, -1)):
        ids, sign = pair_index_and_sign(first, second)
        assert torch.equal(ids, torch.arange(15)) and (sign == expected_sign).all()


def test_direction_antisymmetry_and_strict_floating_point_bound():
    head = PairwiseRefinementHead()
    h0, h4 = tensors(15)
    a, b = torch.tensor(list(combinations(range(6), 2))).T
    with torch.no_grad():
        head.pair_embeddings.copy_(torch.arange(240).reshape(15, 16) / 50 + .05)
        projected = torch.tanh(head.projection(torch.cat((head.normalize(h0), head.normalize(h4)), dim=-1)))
        expected = torch.tanh((head.pair_embeddings * projected).sum(-1))
        torch.testing.assert_close(head(h0, h4, a, b), expected)
        assert torch.equal(head(h0, h4, a, b), -head(h0, h4, b, a))
        head.pair_embeddings.mul_(1e6)
        delta = head(h0, h4, a, b)
        assert delta.abs().max() == 1 - torch.finfo(delta.dtype).eps / 2
        assert ((delta > -1) & (delta < 1)).all()


def test_initialization_identity_disabled_and_padding():
    head = PairwiseRefinementHead()
    h0, h4 = tensors()
    logits = torch.tensor([[.3, 0, -.1, -1, -2, -0.0],
                           [0, 2, 0, 0, 0, -0.0],
                           [0, 0, 0, 0, 0, -0.0],
                           [float("nan"), 0, 0, 0, 0, -0.0]])
    valid = torch.tensor([True, True, True, False])
    for adapter in (head, None):
        result = runtime.infer(adapter, logits, h0, h4, valid, 0.0)
        assert result.selected.tolist() == [True, False, False, False]
        assert not result.output.delta.count_nonzero()
        assert not result.output.written.any()
        assert torch.equal(result.output.logits[valid].argmax(-1), logits[valid].argmax(-1))
        assert torch.equal(result.output.logits.view(torch.uint8), logits.view(torch.uint8))
        runtime.verify_writeback(logits, valid, result)
    result = runtime.infer(head, logits[3:], h0[3:], h4[3:], valid[3:], 0.0)
    assert not result.selected.any() and not result.output.written.any()


def test_writeback_flip_and_tie_bypasses_with_third_class_adjacent():
    base = float(2**22)
    logits = torch.tensor([[.3, 0, 0, -.2, -1, -0.0],
                           [-1, -.2, 0, .3, 0, -.2],
                           [.5, 0, -.1, -1, -1, -1],
                           [base + .5, base, base, base - .5, base - 1, base - 1],
                           [-0.0, .5, .25, 0, 0, 0]])
    a = torch.tensor([0, 3, 0, 0, 1])
    b = torch.tensor([1, 2, 1, 1, 2])
    selected = torch.tensor([True, True, True, True, False])
    row = torch.arange(5)
    margin = logits[row, a].double() - logits[row, b].double()
    delta = torch.tensor([-.5, -.1, -.5, -.625, 0])
    result = bounded_writeback(logits, a, b, selected, margin, delta)
    assert result.written.tolist() == [True, True, False, False, False]
    assert result.logits.argmax(-1).tolist() == [1, 3, 0, 0, 1]
    assert result.corrected_margin[0].item() == pytest.approx(-.2)
    assert result.logits[0, :2].tolist() == pytest.approx([0., .2])
    non_candidates = (torch.arange(6) != a[:, None]) & (torch.arange(6) != b[:, None])
    assert torch.equal(result.logits[non_candidates].view(torch.uint8), logits[non_candidates].view(torch.uint8))
    assert torch.equal(result.logits[2:].view(torch.uint8), logits[2:].view(torch.uint8))


def test_gate_and_s_are_fixed_and_label_free():
    z = np.array([[.5, 0, 0, 0, 0, 0], [1., 0, 0, 0, 0, 0], [.2, 0, 0, 0, 0, 0],
                  [.2, 0, 0, 0, 0, 0], [0., 0, 0, 0, 0, 0]])
    labels = np.array([0, 1, 2, -100, 0])
    valid = labels >= 0
    data = scores(z[valid], labels[valid])
    selected = gate(data.mad, data.margin, True, data.mad[0])
    assert selected.tolist() == [True, False, True, False]
    assert data.top2.tolist() == [1, 1, 1, 1]
    assert runtime.supervised_rows(data.labels, data.top1, data.top2, selected).tolist() == [True, False, False, False]
    assert train_threshold(np.array([.8, .3, .6, .9, .2])) == .8
    result = runtime.infer(None, torch.from_numpy(z), *tensors(5), torch.from_numpy(valid), data.mad[0])
    assert result.selected.tolist() == [True, False, True, False, False]
    imagined_other_gold = np.array([1, 3, 0, 5])
    swapped_gold = scores(z[valid], imagined_other_gold)
    assert np.array_equal(gate(swapped_gold.mad, swapped_gold.margin, True, data.mad[0]), selected)


def test_gradient_scope_and_tiny_synthetic_optimization():
    torch.manual_seed(29)
    h0, h4 = tensors()
    backbone = torch.nn.Linear(300, 300).eval().requires_grad_(False)
    # Even a misconfigured caller's grad-bearing graph is detached by the head.
    h0 = backbone(h0.requires_grad_())
    h4 = backbone(h4.requires_grad_())
    h0.retain_grad()
    h4.retain_grad()
    head = PairwiseRefinementHead()
    optimizer = torch.optim.Adam(head.parameters(), lr=runtime.LR, betas=runtime.BETAS, weight_decay=0)
    a, b, margin, target = torch.zeros(4, dtype=torch.long), torch.ones(4, dtype=torch.long), torch.full((4,), .3), -torch.ones(4)

    def loss():
        return runtime.ranking_loss(margin, head(h0, h4, a, b), target)

    initial = loss().item()
    for _ in range(5):
        optimizer.zero_grad(set_to_none=True)
        loss().backward()
        optimizer.step()
    assert loss().item() < initial
    assert h0.grad is None and h4.grad is None
    assert all(p.grad is None for p in backbone.parameters())
    assert all(p.grad is not None and p.grad.count_nonzero() for p in head.parameters())


def test_frozen_forward_cache_and_real_f0_causal_interface():
    torch.manual_seed(100)
    f0 = FusionTAV(make_final_config("F0", 100)).eval().requires_grad_(False)
    assert sum(p.numel() for p in f0.parameters()) == 1638306
    data = SyntheticDialogues("dev")
    ordinary = frozen_forward(f0, data, "dev", device="cpu", batch_size=16)
    cached = frozen_forward(f0, data, "dev", device="cpu", batch_size=16, capture_states=True)
    assert cached.population == ordinary.population
    np.testing.assert_array_equal(cached.logits, ordinary.logits)
    assert ordinary.layer_states is None
    assert runtime.cache_identity(cached)["tensors"]["h4"]["shape"] == (len(cached.population), 300)
    head = PairwiseRefinementHead().eval()
    with torch.no_grad():
        head.pair_embeddings.normal_(std=.1)
    probes = runtime.collect_causal_probes(f0, "cpu")
    active = torch.tensor([.25, 0., -.1, -.2, -.3, -.4])
    proof = runtime.verify_causal_probes(head, probes, 0.0, active)
    assert proof["comparisons"] == ["prefix", "future", "batch_order", "padding"] and proof["forced_active_gate"]
    assert all(p.grad is None and not p.requires_grad for p in f0.parameters())
    bad_logits, *rest = probes["prefix"]
    with pytest.raises(ValueError, match="causal/invariance"):
        runtime.verify_causal_probes(head, {**probes, "prefix": (bad_logits + 1, *rest)}, 0.0, active)


def verdict(**changes):
    summary = {"delta_weighted_f1_pp": .50, "delta_macro_f1_pp": -.30, "delta_uar_pp": -.30, "net": 4, "q4_net": 3}
    pairs = [{"pair": "/".join(pair), "pair_net": changes.pop("pair_net", -1) if index == 0 else 0,
              "difference": changes.pop("difference", 2) if index == 0 else 0}
             for index, pair in enumerate(pair for pair in metrics.PAIRS for _ in range(2))]
    return metrics.decision({**summary, **changes}, pairs)["FASTKILL"]


@pytest.mark.parametrize("change, expected", [
    ({}, "GO"), ({"delta_weighted_f1_pp": .499}, "REVIEW_SIGNAL"),
    ({"delta_weighted_f1_pp": .20}, "REVIEW_SIGNAL"), ({"delta_weighted_f1_pp": .199}, "STOP"),
    ({"net": 0}, "STOP"), ({"q4_net": 0}, "STOP"), ({"net": 3}, "REVIEW_SIGNAL"),
    ({"q4_net": 2}, "REVIEW_SIGNAL"), ({"delta_macro_f1_pp": -1.0}, "REVIEW_SIGNAL"),
    ({"delta_macro_f1_pp": -1.0001}, "STOP"), ({"delta_uar_pp": -1.0001}, "STOP"),
    ({"pair_net": -4}, "STOP"), ({"pair_net": -3}, "REVIEW_SIGNAL"), ({"difference": 3}, "REVIEW_SIGNAL"),
    ({"net": 0, "difference": 3}, "STOP"),
])
def test_frozen_stop_go_and_review_boundaries(change, expected):
    assert verdict(**change) == expected


def test_pair_confusion_net_separates_wrong_to_wrong_and_has_all_directions():
    gold, ctrl, updated = (np.array(row) for row in ([0, 0, 0, 4, 4], [4, 4, 0, 0, 4], [0, 1, 4, 4, 0]))
    pairs = metrics.pair_confusions(gold, ctrl, updated)
    assert len(pairs) == 30
    forward, reverse = [row for row in pairs if row["pair"] == "hap/exc"]
    assert (forward["ctrl_count"], forward["primary_count"], forward["difference"]) == (2, 1, -1)
    assert (forward["direction_r"], forward["direction_d"], forward["pair_net"]) == (1, 1, 0)
    assert forward["true_class_denominator"] == 3 and forward["primary_directional_error_rate"] == 1 / 3
    assert reverse["true_class_denominator"] == 2 and reverse["pair_net"] == 0


def test_calibration_and_dialogue_bootstrap():
    labels = np.array([0, 0, 1])
    ctrl, primary = np.array([1, 1, 1]), np.array([0, 0, 1])
    uniform = metrics.calibration(np.zeros((3, 6)), labels)
    assert uniform == pytest.approx({"nll": np.log(6), "brier": 5 / 6, "ece": .5})
    confidence_one = metrics.calibration(np.array([[1000., 0, 0, 0, 0, 0]]), np.array([0]))
    assert confidence_one == {"nll": 0, "brier": 0, "ece": 0}
    intervals = metrics.bootstrap(labels, ctrl, primary, ["large", "large", "small"])
    assert [r["metric"] for r in intervals] == ["weighted_f1", "accuracy"]
    for row in intervals:
        assert (row["seed"], row["resamples"], row["unit"], row["dialogues"]) == (7100, 2000, "dialogue", 2)
        assert row["lower_95_pp"] == 0 and row["upper_95_pp"] == 100


def test_single_yaml_switch_and_locked_training_contract(tmp_path):
    assert runner.enabled(runner.CONFIG) is True
    config = tmp_path / "pairwise.yaml"
    config.write_text("pairwise_refinement:\n  enabled: false\n", encoding="utf-8")
    assert runner.enabled(config) is False
    config.write_text("pairwise_refinement:\n  enabled: true\n  lr: 0.003\n", encoding="utf-8")
    with pytest.raises(ValueError, match="only the boolean"):
        runner.enabled(config)
    assert runtime.TRAINING == {
        "optimizer": "Adam", "lr": .003, "betas": (.9, .999), "weight_decay": 0,
        "batch_size": 128, "epochs": 20, "shuffle": True, "seed": 100,
        "loss": "mean_softplus_negative_signed_margin",
    }


@pytest.fixture
def synthetic_run(monkeypatch, tmp_path):
    args = SimpleNamespace(train_dev_package=tmp_path, frozen_runs=tmp_path, round1_dir=tmp_path, batch_id="synthetic")
    (tmp_path / runner.PROJECTION_MANIFEST).write_text("{}", encoding="utf-8")
    train = sample_split([[.25, 0., -.1, -.2, -.3, -.4]] * 8, [1, 1, 0, 2] * 2)
    dev = sample_split(train.logits[:6], [0, 1, 0, 1, 2, 1])
    source = SimpleNamespace(identity=F0_IDENTITY, logits=dev.logits,
                             manifest={"architecture_identity": "arch", "config_sha256": "config"})
    monkeypatch.setattr(runner, "read_f0", lambda _: source)
    monkeypatch.setattr(runner, "load_projected_train_dev", lambda *_: ({"train": train, "dev": dev}, {}))
    monkeypatch.setattr(runner, "validate_population", lambda *_: {"train": {}, "dev": {}})
    q4 = np.arange(6) >= 4
    groups = {**{name: np.arange(6) % 3 == i for i, name in enumerate(metrics.GROUPS)},
              **dict.fromkeys(metrics.H_GROUPS)}
    trace = {"Q4_SOURCE": "VERIFIED_ROUND1_TRACE", "SHIFT_SUBGROUP_STATUS": "AVAILABLE", "H_SUBGROUP_STATUS": "NOT_AVAILABLE"}
    monkeypatch.setattr(runner, "audit_subgroups", lambda *_: (q4, groups, trace))
    monkeypatch.setattr(runner, "TRAIN_DEV_COUNTS", {"train": {"utterances": 8}, "dev": {"utterances": 6}})
    monkeypatch.setattr(runner, "Q4_SUPPORT", 2)
    monkeypatch.setattr(runner, "verify_preflight", lambda *_: {"train_supervised_support": 6})
    events = []

    def forward(_, datasets, device, *, capture_states, inspect_model):
        assert set(datasets) == {"train", "dev"} and capture_states
        events.append("forward")
        model = FusionTAV(make_final_config("F0", 100)).eval().requires_grad_(False)
        inspect_model(model, device)
        return datasets["train"], datasets["dev"]

    monkeypatch.setattr(runner, "forward_f0", forward)
    return args, tmp_path / "result", tmp_path / "result.zip", events


def test_fixed_epoch_single_dev_scoring_and_lightweight_handoff(monkeypatch, synthetic_run):
    args, output, archive, events = synthetic_run
    original_train, original_score = runner.train_head, runner.analyze

    def train(*args, **kwargs):
        events.append("train")
        return original_train(*args, **kwargs)

    def score(*args, **kwargs):
        events.append("score")
        return original_score(*args, **kwargs)

    monkeypatch.setattr(runner, "train_head", train)
    monkeypatch.setattr(runner, "analyze", score)
    assert runner.execute(args, output, archive, device="cpu", git={}, experiment_date="20261124") == 0
    manifest = read_json(output / runner.MANIFEST)
    assert events == ["forward", "train", "score"]
    history = read_csv(output / "TRAIN_HISTORY.csv")
    assert [int(row["epoch"]) for row in history] == list(range(1, 21))
    assert {int(row["n_s"]) for row in history} == {6}
    assert float(history[-1]["train_loss"]) < float(history[0]["train_loss"])
    assert manifest["optimizer_steps"] == 20 and manifest["primary_dev_evaluation_count"] == 1
    assert manifest["f0_dev_dataset_passes"] == manifest["f0_train_dataset_passes"] == 1
    assert read_json(output / runner.HOLD_STATUS) == {**HOLD, "TEST_TOUCHED": "NO", "TEST_HOLD": "PASS"}
    ckpt = output / runner.CHECKPOINT
    assert [path.name for path in output.glob("*.pt")] == [runner.CHECKPOINT]
    payload = torch.load(ckpt, map_location="cpu", weights_only=True)
    PairwiseRefinementHead().load_state_dict(payload["state_dict"], strict=True)
    assert payload["epoch"] == 20 and payload["parameter_count"] == 9856
    assert payload["f0_checkpoint_sha256"] == F0_IDENTITY["checkpoint_sha256"]
    assert payload["train_mad_threshold"] == manifest["train_mad_threshold"]
    assert manifest["head_checkpoint"]["sha256"] == sha256_file(ckpt)
    assert len(read_csv(output / "DEV_SAMPLE_TRANSITIONS.csv")) == 6
    assert len(read_csv(output / "PAIR_CONFUSIONS.csv")) == 30
    assert all(row["status"] == "NOT_AVAILABLE" for row in read_csv(output / "UH_SUBGROUP_RECOVERY.csv"))
    with ZipFile(archive) as pack:
        assert set(pack.namelist()) == {*manifest["outputs_sha256"], runner.MANIFEST}
        assert pack.testzip() is None and all(Path(name).suffix in (".csv", ".md", ".json") for name in pack.namelist())
        assert all(sha256_file(output / name) == digest for name, digest in manifest["outputs_sha256"].items())
    with pytest.raises(FileExistsError):
        runner.execute(args, output, archive, device="cpu", git={}, experiment_date="20261124")


@pytest.mark.parametrize("site, error", [
    ("read_f0", ValueError("identity mismatch")),
    ("validate_population", ValueError("population mismatch")),
    ("cache_identity", ValueError("nonfinite frozen state")),
    ("verify_preflight", ValueError("frozen gate mismatch")),
    ("load_projected_train_dev", HoldViolation("counter", field="test_loader_built", value=1)),
    ("verify_causal_probes", ValueError("causal/invariance")),
    ("verify_writeback", ValueError("bypass")),
])
def test_invalid_takes_precedence_and_never_publishes_head(monkeypatch, synthetic_run, site, error):
    args, output, archive, _ = synthetic_run

    def fail(*_args, **_kwargs):
        raise error

    monkeypatch.setattr(runner, site, fail)
    assert runner.execute(args, output, archive, device="cpu", git={}, experiment_date="20261124") == 2
    assert read_json(output / runner.DECISION)["FASTKILL"] == "REVIEW_INVALID"
    manifest = read_json(output / runner.MANIFEST)
    assert manifest["primary_dev_evaluation_count"] == 0
    assert manifest["formal_train_runs"] == int(site in {"verify_causal_probes", "verify_writeback"})
    hold = read_json(output / runner.HOLD_STATUS)
    assert hold["TEST_TOUCHED"] == "NO"
    assert all(hold[key] == value for key, value in HOLD.items())
    assert hold["TEST_HOLD"] == ("FAIL" if isinstance(error, HoldViolation) else "PASS")
    assert not (output / runner.CHECKPOINT).exists() and not archive.exists()

"""Small CHG contracts and a synthetic Train/Dev checkpoint/export integration."""

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path

import pytest
import torch
import yaml

from models.multidag_cl.paper_reimplementation.attention import RelationAwareAttention
from models.multidag_cl.ta_v0 import TAV0, TAV0Config
from models.multidag_cl.ta_v0.causal_history_gate import (
    CausalHistoryGate, collect_gate_diagnostics, distance_bucket,
)
from scripts.models.multidag_cl.ta_v0.smoke_chg import CHG_CONFIG, smoke
from scripts.models.multidag_cl.ta_v0.train import synthetic_runner
from scripts.runtime.multidag_cl_ta_v0 import formal
from scripts.runtime.multidag_cl_ta_v0.training import TrainDevRuntime

BASE_CONFIG = CHG_CONFIG.replace("formal_train_dev_chg_v0.yaml", "formal_train_dev.yaml")


@pytest.fixture(autouse=True, scope="module")
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def test_distance_buckets_and_nonpast_rejection():
    assert distance_bucket(torch.tensor([1, 2, 3, 4, 5, 6, 10, 11, 99])).tolist() == [
        0, 1, 2, 3, 3, 4, 4, 5, 5]
    for distance in (0, -1):
        with pytest.raises(ValueError, match="j < i"):
            distance_bucket(torch.tensor([distance]))
    with pytest.raises(TypeError):
        distance_bucket(torch.tensor([1.0]))


def test_gate_semantic_product_normalization_and_group_moments():
    torch.manual_seed(100)
    reference = RelationAwareAttention(7).eval()
    attention = deepcopy(reference)
    attention.history_gate = CausalHistoryGate(7).eval()
    query, history = torch.randn(2, 7), torch.randn(2, 12, 7)
    mask = torch.tensor([[True] * 12, [False] * 12])
    same = torch.arange(12).remainder(2).bool().expand(2, -1) & mask
    seen = []
    hook = attention.history_gate.register_forward_hook(lambda m, a, g: seen.append(g.detach()))
    with collect_gate_diagnostics(attention) as diagnostics, torch.no_grad():
        base = reference(query, history, mask, same)
        result = attention(query, history, mask, same)
    hook.remove()
    gates = seen[0]
    assert gates.shape == (12,) and torch.isfinite(gates).all()
    assert torch.all((gates > 0) & (gates < 1))
    expected = base.weights[0] * gates
    expected = expected / (expected.sum() + 1e-8)
    torch.testing.assert_close(result.weights[0], expected)
    torch.testing.assert_close(result.weights.sum(-1), torch.tensor([1., 0.]))
    torch.testing.assert_close(result.logits, base.logits, rtol=0, atol=0)
    values = torch.where(same.unsqueeze(-1), reference.W_same(history), reference.W_different(history))
    torch.testing.assert_close(result.message, (result.weights.unsqueeze(-1) * values).sum(1))
    assert not result.message[1].any() and not result.weights[~mask].any()
    groups = diagnostics.summary()
    bucket = distance_bucket(torch.arange(12, 0, -1))
    subsets = {"all": torch.ones(12, dtype=torch.bool), "same_speaker": same[0],
               "different_speaker": ~same[0],
               **{f"distance_bucket_{i}": bucket == i for i in range(6)}}
    for name, selected in subsets.items():
        assert groups[name]["edge_count"] == int(selected.sum())
        assert groups[name]["mean"] == pytest.approx(float(gates[selected].double().mean()))
        assert groups[name]["std"] == pytest.approx(float(gates[selected].double().std(correction=0)))
    assert attention.history_gate.diagnostics is None
    # No legal predecessor: do not call the gate, return the baseline zero message.
    hook = attention.history_gate.register_forward_hook(lambda *args: pytest.fail("empty edge gated"))
    empty = attention(query, history, torch.zeros_like(mask), same & False)
    hook.remove()
    assert not empty.message.any() and not empty.weights.any()


def model_inputs(length=5):
    generator = torch.Generator().manual_seed(100)
    return {"text_features": torch.randn(1, length, 768, generator=generator),
            "audio_features": torch.randn(1, length, 1582, generator=generator),
            "speaker_ids_int": torch.tensor([[17, 4, 17, 4, 99]])[:, :length],
            "attention_mask": torch.ones(1, length, dtype=torch.bool),
            "lengths": torch.tensor([length])}


def test_gate_sees_exact_existing_edges_and_model_is_prefix_causal():
    model = TAV0(TAV0Config(use_causal_history_gate=True)).eval()
    value = model_inputs()
    adjacency = model.predecessor_builder.build(value["speaker_ids_int"], value["attention_mask"])
    seen = [[] for _ in model.graph_layers]
    hooks = [layer.attention.history_gate.register_forward_pre_hook(
        lambda m, args, index=index: seen[index].append((args[2].clone(), args[3].clone())))
        for index, layer in enumerate(model.graph_layers)]
    with torch.no_grad():
        reference = model(**value)
    for hook in hooks:
        hook.remove()
    for rows in seen:
        assert len(rows) == 4
        for target, (same, distances) in enumerate(rows, 1):
            sources = adjacency[0, target].nonzero().flatten()
            assert (distances > 0).all()
            assert distances.tolist() == (target - sources).tolist()
            assert same.tolist() == (value["speaker_ids_int"][0, sources] ==
                                      value["speaker_ids_int"][0, target]).tolist()
    changed = {k: v.clone() for k, v in value.items()}
    changed["text_features"][:, 3:] *= -50
    changed["audio_features"][:, 3:] += 100
    changed["speaker_ids_int"][:, 3:] = 500
    prefix = {k: v[:, :3] for k, v in value.items() if k != "lengths"}
    prefix["lengths"] = torch.tensor([3])
    with torch.no_grad():
        perturbed, truncated = model(**changed), model(**prefix)
    for name in ("h_ta", "logits_ta", "prob_ta"):
        torch.testing.assert_close(getattr(reference, name)[:, :3], getattr(perturbed, name)[:, :3])
        torch.testing.assert_close(getattr(reference, name)[:, :3], getattr(truncated, name))
    # Invalid self/future adjacency is rejected by the unchanged DAG before gating.
    relations = model.relation_builder.build(value["speaker_ids_int"], value["attention_mask"])
    bad = adjacency.clone()
    bad[0, 1, 3] = True
    hook = model.graph_layers[0].attention.history_gate.register_forward_hook(
        lambda *args: pytest.fail("future edge reached CHG"))
    with pytest.raises(ValueError, match="self or future"):
        model.graph_layers[0](torch.zeros(1, 5, 300), bad, relations, value["attention_mask"])
    hook.remove()


def test_disabled_identity_original_initialization_and_independent_layers():
    baseline = synthetic_runner().model.eval()
    explicit_off = TAV0(TAV0Config(use_causal_history_gate=False)).eval()
    explicit_off.load_state_dict(baseline.state_dict())
    raw = yaml.safe_load(Path(BASE_CONFIG).read_text(encoding="utf-8"))
    assert baseline.state_dict()["_extra_state"] == {
        "lineage": "ta_v0", "config": raw["model"], "label_vocab": baseline.get_extra_state()["label_vocab"]}
    assert sum(p.numel() for p in baseline.parameters()) == 5975510
    assert not any("history_gate" in name for name in baseline.state_dict())
    with torch.no_grad():
        for name in ("h_ta", "logits_ta", "prob_ta", "mask"):
            torch.testing.assert_close(getattr(baseline(**model_inputs()), name),
                                       getattr(explicit_off(**model_inputs()), name), rtol=0, atol=0)
    enabled = synthetic_runner(TAV0Config(use_causal_history_gate=True)).model
    old_parameters = dict(baseline.named_parameters())
    for name, parameter in enabled.named_parameters():
        if name in old_parameters:
            torch.testing.assert_close(parameter, old_parameters[name], rtol=0, atol=0)
    gates = [layer.attention.history_gate for layer in enabled.graph_layers]
    assert len({p.data_ptr() for gate in gates for p in gate.parameters()}) == 4 * 6
    assert [sum(p.numel() for p in gate.parameters()) for gate in gates] == [39617] * 4
    with pytest.raises(ValueError, match="checkpoint identity"):
        enabled.load_state_dict(baseline.state_dict(), strict=False)
    with pytest.raises(ValueError, match="checkpoint identity"):
        baseline.load_state_dict(enabled.state_dict(), strict=False)


def test_config_is_minimal_and_protocol_stays_frozen():
    raw = yaml.safe_load(Path(CHG_CONFIG).read_text(encoding="utf-8"))
    config = formal.validate_formal_config(raw)
    assert config == TAV0Config(use_causal_history_gate=True)
    baseline = yaml.safe_load(Path(BASE_CONFIG).read_text(encoding="utf-8"))
    stripped = deepcopy(raw)
    for field in ("use_causal_history_gate", "chg_speaker_embedding_dim",
                  "chg_distance_embedding_dim", "chg_hidden_dim"):
        stripped["model"].pop(field)
    stripped["output"]["experiment_group"] = baseline["output"]["experiment_group"]
    assert stripped == baseline
    invalid = deepcopy(raw)
    invalid["training"]["optimizer"]["learning_rate"] *= 2
    with pytest.raises(ValueError, match="frozen"):
        formal.validate_formal_config(invalid)
    with pytest.raises(ValueError, match="freezes"):
        replace(config, dag_hidden_dim=100)
    with pytest.raises(TypeError):
        replace(config, use_causal_history_gate=1)


def test_synthetic_backward_and_dev_diagnostics():
    result = smoke(TAV0Config(use_causal_history_gate=True))
    assert result["status"] == "PASS" and result["added_parameters"] == 158468
    assert result["test_evaluation_count"] == result["optimizer_steps"] == 0
    assert result["gate_diagnostics"]["all"]["edge_count"] == 12


def test_synthetic_best_checkpoint_export_and_epoch_diagnostics(tmp_path):
    runner = synthetic_runner(TAV0Config(use_causal_history_gate=True))
    runtime = TrainDevRuntime(runner.model, runner.dataset_factory, synthetic_only=True)
    ids = [uid for batch in runtime.loader("dev") for dialogue in batch["utterance_ids"] for uid in dialogue]
    directory = tmp_path / "synthetic_chg"
    result = formal.fit(runtime, directory, resolved_config={"synthetic": True}, manifest={},
                        expected_dev_ids=ids, synthetic_epochs=1)
    assert result["best_checkpoint_reloaded"]
    assert result["test_loader_built"] == result["test_evaluation_count"] == 0
    diagnostics = json.loads((directory / "chg_gate_diagnostics.json").read_text())
    assert diagnostics["split"] == "dev"
    assert diagnostics["checkpoint_identity"] == formal.sha256_file(directory / "checkpoints/best_model.pt")
    assert diagnostics["groups"] == runtime.evaluate()["chg_gate_diagnostics"]
    epoch = json.loads((directory / "logs/chg_gate_diagnostics.jsonl").read_text())
    assert epoch["epoch"] == 1 and epoch["groups"]["all"]["edge_count"] == 12
    assert all(layer.attention.history_gate.diagnostics is None for layer in runner.model.graph_layers)
    # No collection during Train, even if a caller accidentally leaves a scope open.
    with collect_gate_diagnostics(runner.model) as collector:
        runtime.forward_batch(next(iter(runtime.loader("train"))), split="train")
    assert collector.summary()["all"]["edge_count"] == 0

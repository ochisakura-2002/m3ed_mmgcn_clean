from copy import deepcopy
from dataclasses import replace
import inspect
from pathlib import Path

import pytest
import torch
import yaml

from datasets.iemocap.ta_v0_dataset import TAV0Dataset, ta_v0_collate
from models.multidag_cl.paper_reimplementation.attention import RelationAwareAttention
from models.multidag_cl.paper_reimplementation.dag_layer import DualGRUNodeUpdate, MultiDAGLayer
from models.multidag_cl.ta_v0 import LABEL_VOCAB, TAV0, TAV0Config
from scripts.models.multidag_cl.ta_v0.train import (
    DEFAULT_CONFIG, DevCheckpointSelector, DevelopmentRunner, MODEL_INPUTS,
    main, synthetic_dry_run, synthetic_package, synthetic_runner, validate_config, weighted_f1,
)


@pytest.fixture(autouse=True, scope="module")
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def batch():
    package = synthetic_package()
    dataset = TAV0Dataset(*package, split="train", expected_rows=len(package[1]))
    return ta_v0_collate([dataset[1], dataset[0]])


def inputs(value):
    return {key: value[key] for key in MODEL_INPUTS}


def test_ta_only_shapes_probabilities_and_direct_classifier_input():
    model = TAV0().eval()
    value = batch()
    seen = []
    handle = model.classifier.register_forward_pre_hook(lambda module, args: seen.append(args[0]))
    states = []
    handles = [layer.register_forward_hook(lambda module, args, out: states.append(out.state))
               for layer in model.graph_layers]
    with torch.no_grad():
        output = model(**inputs(value))
    handle.remove()
    for handle in handles:
        handle.remove()
    assert output.h_ta.shape == (2, 3, 1500)
    assert output.logits_ta.shape == output.prob_ta.shape == (2, 3, 6)
    assert seen[0] is output.h_ta
    torch.testing.assert_close(output.h_ta[..., 300:], torch.cat(states, -1))
    assert torch.count_nonzero(output.h_ta[~output.mask]) == 0
    for tensor in (output.h_ta, output.logits_ta, output.prob_ta):
        assert torch.isfinite(tensor).all() and tensor.device.type == "cpu"
    torch.testing.assert_close(output.prob_ta.sum(-1), torch.ones(2, 3))
    torch.testing.assert_close(output.prob_ta, output.logits_ta.softmax(-1))
    assert set(inspect.signature(model.forward).parameters) == set(MODEL_INPUTS)
    assert not any("visual" in name for name, _ in model.named_parameters())
    assert not any("visual" in name for name, _ in model.named_modules())
    assert model.ta_fusion_projection.in_features == 200
    assert model.ta_fusion_projection.out_features == 300
    assert model.encoders.text_encoder.hidden_size == 100
    assert model.encoders.text_encoder.num_layers == 1
    assert not model.encoders.text_encoder.bidirectional
    with pytest.raises(TypeError):
        model(**inputs(value), visual_features=torch.zeros(2, 3, 342))


def test_dag_reuses_frozen_components_and_forbids_future_and_self():
    model = TAV0()
    value = batch()
    adjacency = model.predecessor_builder.build(value["speaker_ids_int"], value["attention_mask"])
    assert not torch.triu(adjacency, diagonal=0).any()
    for layer in model.graph_layers:
        assert type(layer) is MultiDAGLayer
        assert type(layer.attention) is RelationAwareAttention
        assert type(layer.node_update) is DualGRUNodeUpdate


@pytest.mark.parametrize("k", [1, 2])
def test_future_perturbation_and_prefix_equivalence(k):
    torch.manual_seed(100)
    model = TAV0().eval()
    value = inputs(batch())
    # Use one dialogue of length 3, then perturb only future T/A/speaker.
    full = {name: tensor[1:2].clone() for name, tensor in value.items()}
    changed = {name: tensor.clone() for name, tensor in full.items()}
    changed["text_features"][:, k:] = torch.randn_like(changed["text_features"][:, k:]) * 50
    changed["audio_features"][:, k:] = torch.randn_like(changed["audio_features"][:, k:]) * 50
    changed["speaker_ids_int"][:, k:] = 9
    prefix = {name: tensor[:, :k].clone() for name, tensor in full.items() if name != "lengths"}
    prefix["lengths"] = torch.tensor([k])
    with torch.no_grad():
        reference, perturbed, truncated = model(**full), model(**changed), model(**prefix)
    for name in ("h_ta", "logits_ta", "prob_ta"):
        torch.testing.assert_close(getattr(reference, name)[:, :k], getattr(perturbed, name)[:, :k],
                                   atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(getattr(reference, name)[:, :k], getattr(truncated, name),
                                   atol=1e-6, rtol=1e-5)


def test_synthetic_backward_reaches_text_audio_fusion_and_dag():
    runner = synthetic_runner()
    output, loss = runner.forward_batch(batch(), split="train")
    loss.backward()
    for name, parameter in runner.model.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
    for module in (runner.model.encoders.text_projection, runner.model.encoders.text_encoder,
                   runner.model.encoders.audio_projection, runner.model.ta_fusion_projection,
                   runner.model.graph_layers[0], runner.model.classifier):
        assert sum(float(p.grad.abs().sum()) for p in module.parameters()) > 0


@pytest.mark.parametrize("case", ["nonprefix", "bad_lengths", "nan", "padding", "speaker"])
def test_model_rejects_invalid_inputs(case):
    value = inputs(batch())
    if case == "nonprefix":
        value["attention_mask"][0] = torch.tensor([True, False, True])
    elif case == "bad_lengths":
        value["lengths"][0] = 0
    elif case == "nan":
        value["text_features"][1, 0, 0] = float("nan")
    elif case == "padding":
        value["audio_features"][0, 2, 0] = 1
    elif case == "speaker":
        value["speaker_ids_int"][1, 0] = -1
    with pytest.raises(ValueError):
        TAV0()(**value)


def test_test_loader_fails_before_factory_and_selection_is_dev_only():
    calls = []
    runner = DevelopmentRunner(TAV0(), lambda split: calls.append(split))
    for split in ("test", "Test", "validation", ""):
        with pytest.raises(ValueError, match="Test loader forbidden"):
            runner.loader(split)
    assert calls == []
    selector = DevCheckpointSelector()
    assert selector.consider(split="dev", dev_weighted_f1=0.5, checkpoint_identity="epoch1")
    assert not selector.consider(split="dev", dev_weighted_f1=0.5, checkpoint_identity="epoch2")
    with pytest.raises(ValueError):
        selector.consider(split="test", dev_weighted_f1=1, checkpoint_identity="leak")
    assert selector.identity == "epoch1"
    assert weighted_f1(torch.tensor([0, 0, 1]), torch.tensor([0, 1, 1])) == pytest.approx(2 / 3)
    value = batch()
    value["split"] = "test"
    with pytest.raises(ValueError):
        synthetic_runner().forward_batch(value, split="dev")
    value = batch()
    value["visual_features"] = None
    with pytest.raises(ValueError):
        synthetic_runner().forward_batch(value, split="train")


def test_config_frozen_protocol_dry_run_and_checkpoint_identity():
    root = Path(__file__).resolve().parents[4]
    raw = yaml.safe_load((root / DEFAULT_CONFIG).read_text(encoding="utf-8"))
    config = validate_config(raw)
    result = synthetic_dry_run(config)
    assert result["status"] == "PASS"
    assert result["test_evaluation_count"] == result["optimizer_steps"] == 0
    assert result["training_started"] is False
    assert LABEL_VOCAB == ("hap", "sad", "neu", "ang", "exc", "fru")
    for section, key, changed in (("runtime", "seed", 42), ("runtime", "curriculum", True),
                                   ("runtime", "device", "cuda"),
                                   ("checkpoint", "test_evaluation_count", 1)):
        invalid = deepcopy(raw)
        invalid[section][key] = changed
        with pytest.raises(ValueError):
            validate_config(invalid)
    with pytest.raises(ValueError):
        replace(config, dag_hidden_dim=200)
    with pytest.raises(SystemExit):
        main(["--config", str(root / DEFAULT_CONFIG)])
    model, restored = TAV0(config).eval(), TAV0(config).eval()
    restored.load_state_dict(model.state_dict())
    with torch.no_grad():
        torch.testing.assert_close(model(**inputs(batch())).h_ta, restored(**inputs(batch())).h_ta)
    wrong = deepcopy(model.state_dict())
    wrong["_extra_state"]["label_vocab"] = tuple(reversed(LABEL_VOCAB))
    with pytest.raises(ValueError):
        restored.load_state_dict(wrong)


def test_ta_encoder_math_matches_historical_unilstm_and_audio_with_identical_weights():
    from models.multidag_cl.paper_reimplementation.config import MultiDAGCLConfig
    from models.multidag_cl.paper_reimplementation.encoders import PaperFormulaModalityEncoder

    root = Path(__file__).resolve().parents[4]
    path = root / "configs/multidag_cl/paper_reimplementation/iemocap/full_context/clean_roberta_features/project_fair.yaml"
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))["model_core"]
    raw["encoder"]["text_bidirectional"] = False
    raw["encoder"]["causal_text_ablation"] = True
    old = PaperFormulaModalityEncoder(MultiDAGCLConfig.from_mapping(raw)).eval()
    current = TAV0().encoders.eval()
    old.text_projection.load_state_dict(current.text_projection.state_dict())
    old.text_encoder.load_state_dict(current.text_encoder.state_dict())
    old.audio_projection.load_state_dict(current.audio_projection.state_dict())
    value = batch()
    with torch.no_grad():
        # The historical encoder alone retains its historical TAV API.
        # The TA-v0 path has no such argument or slot.
        historical = old(
            text_features=value["text_features"], audio_features=value["audio_features"],
            visual_features=torch.zeros(2, 3, 342), lengths=value["lengths"],
            attention_mask=value["attention_mask"],
        )
        text, audio = current(value["text_features"], value["audio_features"],
                              value["lengths"], value["attention_mask"])
    torch.testing.assert_close(text, historical.text, rtol=0, atol=0)
    torch.testing.assert_close(audio, historical.audio, rtol=0, atol=0)

import pytest
import torch

from models.multidag_cl.graph_core_variants import GraphCoreConfig, GraphCoreTA, VARIANTS
from models.multidag_cl.graph_core_variants.config import baseline_config
from models.multidag_cl.ta_v0 import TAV0
from scripts.models.multidag_cl.ta_v0.train import MODEL_INPUTS
from scripts.runtime.multidag_cl_graph_core_variants.formal import synthetic_runtime


@pytest.mark.parametrize("variant", VARIANTS)
def test_off_is_bitwise_baseline_including_state_dict_and_training_dropout(variant):
    torch.manual_seed(100)
    parent = TAV0(baseline_config())
    torch.manual_seed(100)
    model = GraphCoreTA(GraphCoreConfig(variant, enabled=False))
    assert model.state_dict().keys() == parent.state_dict().keys()
    for key, value in parent.state_dict().items():
        if isinstance(value, torch.Tensor):
            torch.testing.assert_close(model.state_dict()[key], value, rtol=0, atol=0)
        else:
            assert model.state_dict()[key] == value
    batch = next(iter(synthetic_runtime(GraphCoreConfig(variant)).loader("dev")))
    inputs = {key: batch[key] for key in MODEL_INPUTS}
    for training in (False, True):
        model.train(training)
        parent.train(training)
        torch.manual_seed(42)
        actual = model(**inputs)
        torch.manual_seed(42)
        expected = parent(**inputs)
        torch.testing.assert_close(actual.h_ta, expected.h_ta, rtol=0, atol=0)
        torch.testing.assert_close(actual.logits_ta, expected.logits_ta, rtol=0, atol=0)


@pytest.mark.parametrize("variant", VARIANTS)
def test_frozen_modules_h0_and_end_to_end_causal_boundary(variant):
    torch.manual_seed(100)
    parent = TAV0(baseline_config()).eval()
    torch.manual_seed(100)
    model = GraphCoreTA(GraphCoreConfig(variant)).eval()
    assert not hasattr(model, "graph_layers")
    for name in ("encoders", "ta_fusion_projection", "classifier"):
        assert type(getattr(model, name)) is type(getattr(parent, name))
        for key, value in getattr(parent, name).state_dict().items():
            torch.testing.assert_close(getattr(model, name).state_dict()[key], value, rtol=0, atol=0)
    runtime = synthetic_runtime(GraphCoreConfig(variant))
    batch = next(iter(runtime.loader("dev")))
    inputs = {key: batch[key].clone() for key in MODEL_INPUTS}
    with torch.no_grad():
        output, reference = model(**inputs), parent(**inputs)
    assert output.h_ta.shape == (2, 5, 1500)
    assert output.logits_ta.shape == output.prob_ta.shape == (2, 5, 6)
    torch.testing.assert_close(output.h_ta[..., :300], reference.h_ta[..., :300], rtol=0, atol=0)
    assert torch.count_nonzero(output.h_ta[~output.mask]) == 0
    assert torch.count_nonzero(output.logits_ta[~output.mask]) == 0
    torch.testing.assert_close(output.prob_ta.sum(-1), torch.ones(2, 5))
    inputs["text_features"][0, 3:] += 100
    inputs["audio_features"][0, 3:] -= 100
    with torch.no_grad():
        future = model(**inputs)
    torch.testing.assert_close(output.h_ta[0, :3], future.h_ta[0, :3], rtol=0, atol=0)
    torch.testing.assert_close(output.logits_ta[0, :3], future.logits_ta[0, :3], rtol=0, atol=0)


@pytest.mark.parametrize("variant", VARIANTS)
def test_checkpoint_cross_variant_and_parent_loads_fail_before_copy(variant):
    model = GraphCoreTA(GraphCoreConfig(variant))
    state = model.state_dict()
    GraphCoreTA(GraphCoreConfig(variant)).load_state_dict(state, strict=True)
    targets = [TAV0(baseline_config())] + [GraphCoreTA(GraphCoreConfig(v)) for v in VARIANTS if v != variant]
    for target in targets:
        before = target.classifier[0].weight.detach().clone()
        with pytest.raises(ValueError, match="identity"):
            target.load_state_dict(state, strict=False)
        torch.testing.assert_close(before, target.classifier[0].weight, rtol=0, atol=0)
    with pytest.raises(ValueError, match="identity"):
        model.load_state_dict(TAV0(baseline_config()).state_dict(), strict=False)


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("field", ["padding", "lengths", "dtype", "speaker", "mask", "nonfinite"])
def test_model_input_boundary_rejects_invalid_inputs(variant, field):
    runtime = synthetic_runtime(GraphCoreConfig(variant))
    batch = next(iter(runtime.loader("dev")))
    inputs = {key: batch[key].clone() for key in MODEL_INPUTS}
    if field == "padding":
        inputs["text_features"][1, 4] = 1
    elif field == "lengths":
        inputs["lengths"][0] -= 1
    elif field == "dtype":
        inputs["audio_features"] = inputs["audio_features"].double()
    elif field == "speaker":
        inputs["speaker_ids_int"][0, 0] = -1
    elif field == "mask":
        inputs["attention_mask"][0, 0] = False
    else:
        inputs["audio_features"][0, 0, 0] = float("nan")
    with pytest.raises((TypeError, ValueError)):
        runtime.model(**inputs)

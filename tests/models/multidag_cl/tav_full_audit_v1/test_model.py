from copy import deepcopy

import pytest
import torch

from datasets.iemocap.tav_full_audit_dataset import collate
from models.multidag_cl.graph_core_variants.coop import CoopCore
from models.multidag_cl.graph_core_variants.track import TrackCore
from models.multidag_cl.graph_core_variants.dcn import DCNCore
from models.multidag_cl.paper_reimplementation.dag_layer import MultiDAGLayer
from models.multidag_cl.paper_reimplementation.encoders import PaperFormulaModalityEncoder
from models.multidag_cl.tav_full_audit_v1 import TAVFullAudit
from scripts.runtime.multidag_cl_tav_full_audit_v1.runtime import MODEL_INPUTS, SyntheticDialogues
from scripts.runtime.multidag_cl_tav_full_audit_v1.spec import CORES, CONTEXTS, MODALITIES, make_config


def inputs():
    data = SyntheticDialogues("dev")
    batch = collate([data[1], data[0]])
    return {key: batch[key] for key in MODEL_INPUTS}


@pytest.mark.parametrize("core", CORES)
def test_historical_frontend_and_shared_core_contract(core):
    model = TAVFullAudit(make_config(core)).eval()
    assert isinstance(model.paper_encoder, PaperFormulaModalityEncoder)
    assert model.paper_encoder.text_encoder.bidirectional is False
    assert model.classifier[0].in_features == 1500
    if core == "original":
        assert len(model.graph_layers) == 4 and all(type(layer) is MultiDAGLayer for layer in model.graph_layers)
    else:
        assert type(model.graph_core) is {"coop": CoopCore, "track": TrackCore, "dcn": DCNCore}[core]
        assert not hasattr(model, "graph_layers")
    batch = inputs()
    with torch.no_grad():
        output = model(**batch)
        encoded = model.paper_encoder(**{key: batch[key] for key in
                                        ("text_features", "audio_features", "visual_features", "lengths", "attention_mask")})
    assert len(output.layer_states) == 5
    assert all(state.shape == (2, 3, 300) for state in output.layer_states)
    torch.testing.assert_close(output.layer_states[0], encoded.fused, rtol=0, atol=0)
    assert output.h_ta.shape == (2, 3, 1500) and output.logits_ta.shape == (2, 3, 6)
    assert not torch.count_nonzero(output.h_ta[~output.mask])
    assert not torch.triu(output.adjacency).any()


@pytest.mark.parametrize("core", CORES)
def test_future_perturbation_and_prefix_invariance(core):
    model = TAVFullAudit(make_config(core)).eval()
    batch = inputs()
    changed = {key: value.clone() for key, value in batch.items()}
    for key in ("text_features", "audio_features", "visual_features"):
        changed[key][0, 2:] += 100
    changed["speaker_ids_int"][0, 2:] = 1
    prefix = {key: value[:, :2].clone() if value.ndim >= 2 else value.clamp(max=2) for key, value in batch.items()}
    with torch.no_grad():
        full, perturbed, short = model(**batch), model(**changed), model(**prefix)
    torch.testing.assert_close(full.h_ta[:, :2], perturbed.h_ta[:, :2], rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(full.h_ta[:, :2], short.h_ta, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(full.logits_ta[:, :2], short.logits_ta, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("context,window", CONTEXTS.items())
def test_existing_utterance_window_semantics(context, window):
    model = TAVFullAudit(make_config(context=context)).eval()
    with torch.no_grad():
        result = model(**inputs())
    for b, length in enumerate((3, 2)):
        for target in range(3):
            for source in range(3):
                expected = target < length and source < target and (window is None or target - source <= window)
                assert bool(result.adjacency[b, target, source]) == expected


@pytest.mark.parametrize("active", MODALITIES)
def test_encoded_zero_mask_is_deterministic_and_ignores_inactive_features(active):
    model = TAVFullAudit(make_config()).eval()
    batch, changed = inputs(), inputs()
    for letter, key in (("T", "text_features"), ("A", "audio_features"), ("V", "visual_features")):
        if letter not in active:
            changed[key] = changed[key] + 99 * changed["attention_mask"][..., None]
    identity = deepcopy(model.get_extra_state())
    with torch.no_grad():
        output = model(**batch, active_modalities=active)
        other = model(**changed, active_modalities=active)
    for letter, value in output.encoded_modalities.items():
        if letter not in active:
            assert torch.count_nonzero(value) == 0
    torch.testing.assert_close(output.logits_ta, other.logits_ta, rtol=0, atol=0)
    assert model.get_extra_state() == identity


def test_seven_models_keep_parameter_shapes_initialization_and_architecture():
    reference = None
    counts = []
    for active in MODALITIES:
        torch.manual_seed(100)
        model = TAVFullAudit(make_config(modalities=active))
        state = {key: value for key, value in model.state_dict().items() if isinstance(value, torch.Tensor)}
        counts.append(sum(parameter.numel() for parameter in model.parameters()))
        if reference is None:
            reference = state
        else:
            assert state.keys() == reference.keys()
            assert all(torch.equal(value, reference[key]) for key, value in state.items())
    assert set(counts) == {5975110}


@pytest.mark.parametrize("dimension,key", [(1023, "text_features"), (1581, "audio_features"), (341, "visual_features")])
def test_wrong_dimensions_rejected(dimension, key):
    batch = inputs()
    batch[key] = batch[key][..., :dimension]
    with pytest.raises(ValueError, match="dimension"):
        TAVFullAudit(make_config())(**batch)


def test_checkpoint_mask_and_core_lineages_cannot_mix():
    model = TAVFullAudit(make_config())
    for raw in (make_config(modalities="TA"), make_config("track")):
        with pytest.raises(ValueError, match="identity"):
            TAVFullAudit(raw).load_state_dict(model.state_dict(), strict=False)
    with pytest.raises(ValueError, match="Dev-only"):
        model(**inputs(), active_modalities="T")

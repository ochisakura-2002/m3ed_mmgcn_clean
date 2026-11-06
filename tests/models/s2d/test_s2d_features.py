"""Synthetic interface checks; canonical alignment NEVER uses a model double.

Real-core tests skip explicitly if timm/einops are absent. They construct the
full original 12-layer core and original MobileFaceNet, without checkpoints.
The small contract doubles below only test guards, FX slicing and hook cleanup.
"""

import hashlib
import importlib.util
import os

import pytest
import torch
from torch import nn

from datasets.iemocap.label_vocab import LabelVocab
from models.s2d.iemocap.features import (
    _mcp0_static_graph, extract_contextual_slots, extract_static_features,
)
from models.s2d.iemocap.model import S2DIEMOCAP, build_s2d


STATIC_KEYS = ("a0_rgb_mean", "c0_mcp_static_mean", "d0_static_residual_mean")
VOCAB = LabelVocab(("hap", "sad", "neu", "ang", "exc", "fru"), "TEST_SYNTHETIC", True)


@pytest.fixture(scope="module", autouse=True)
def limited_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


class _ForbiddenTemporal(nn.Module):
    def forward(self, value):
        raise AssertionError("Static extraction must not execute TMA")


class _GraphContract(nn.Module):
    """FX slicing harness, intentionally not an implementation of S2D MCP."""

    def __init__(self):
        super().__init__()
        self.conv0_0 = nn.Conv2d(768, 2, 1)
        self.conv0_1 = nn.Conv2d(768, 2, 1)
        self.fovea = nn.Identity()
        self.conv1x1 = nn.Conv2d(2, 768, 1)
        self.TMA = _ForbiddenTemporal()

    def forward(self, value):
        b, c, h, w = value.shape
        left, right = value[:, :int(c / 2)], value[:, int(c / 2):]
        temporal = self.TMA(left.reshape(b, c // 2, h * w))
        spatial = self.conv1x1(self.conv0_0(left) - self.fovea(self.conv0_1(right)))
        return spatial, temporal


def test_contract_only_fx_view_shares_modules_and_excludes_temporal():
    prompt = _GraphContract().eval()
    original_modules = dict(prompt.named_modules())
    view = _mcp0_static_graph(prompt)
    assert not hasattr(view, "TMA")
    for name in ("conv0_0", "conv0_1", "fovea", "conv1x1"):
        assert view.get_submodule(name) is original_modules[name]
    assert dict(prompt.named_modules()) == original_modules
    with torch.inference_mode():
        # Dynamic BT works independently of the trace's spatial specialization.
        for count in (1, 3):
            assert view(torch.zeros(count, 1536, 14, 14)).shape == (count, 768, 14, 14)


class _ContractHead(nn.Module):
    def forward(self, value):
        pooled = value.mean(2).flatten(1)
        return pooled[:, :6], pooled


class _ContractCore(nn.Module):
    """Head-hook lifecycle double; no canonical numerical alignment claim."""

    def __init__(self):
        super().__init__()
        self.num_frames, self.embed_dim = 16, 768
        self.prompt_type, self.global_pool = "deep", "token"
        self.head = _ContractHead()
        self.prompt_parameter = nn.Parameter(torch.zeros(1))

    def forward(self, video, landmark):
        assert torch.is_inference_mode_enabled()
        value = torch.arange(video.shape[0] * 768 * 16, device=video.device,
                             dtype=video.dtype).reshape(video.shape[0], 768, 16, 1, 1)
        return self.head(value)


class _ContractLandmark(nn.Module):
    def forward(self, video):
        return video.new_zeros(video.shape[0] * 16, 128, 14, 14)


@pytest.fixture
def contract_model():
    return S2DIEMOCAP(_ContractCore(), _ContractLandmark(), VOCAB).eval()


@pytest.mark.parametrize("extract", [extract_static_features, extract_contextual_slots])
def test_requires_existing_canonical_wrapper(extract):
    with pytest.raises(TypeError, match="canonical S2DIEMOCAP"):
        extract(nn.Identity(), torch.empty(1, 3, 16, 224, 224, device="meta"))


@pytest.mark.parametrize("extract", [extract_static_features, extract_contextual_slots])
def test_rejects_training_without_changing_flags(contract_model, extract):
    contract_model.train()
    flags = [module.training for module in contract_model.modules()]
    with pytest.raises(ValueError, match=r"model.eval\(\)"):
        extract(contract_model, torch.empty(1, 3, 16, 224, 224, device="meta"))
    assert [module.training for module in contract_model.modules()] == flags


def test_rejects_mixed_training_modes(contract_model):
    contract_model.core.head.train()
    with pytest.raises(ValueError, match=r"model.eval\(\)"):
        extract_contextual_slots(contract_model, torch.empty(1, 3, 16, 224, 224, device="meta"))
    assert contract_model.core.head.training


@pytest.mark.parametrize("shape", [
    (3, 16, 224, 224), (1, 16, 3, 224, 224), (1, 3, 16, 112, 112),
    (0, 3, 16, 224, 224), (1, 3, 0, 224, 224),
])
@pytest.mark.parametrize("extract", [extract_static_features, extract_contextual_slots])
def test_rejects_invalid_pixel_contract(contract_model, shape, extract):
    with pytest.raises(ValueError, match="transformed RGB crops"):
        extract(contract_model, torch.empty(shape, device="meta"))


@pytest.mark.parametrize("extract", [extract_static_features, extract_contextual_slots])
def test_rejects_uint8_input(contract_model, extract):
    with pytest.raises(ValueError, match="floating-point"):
        extract(contract_model, torch.empty(1, 3, 16, 224, 224, dtype=torch.uint8, device="meta"))


@pytest.mark.parametrize("count", [1, 8, 32])
def test_contextual_guard_keeps_t16(contract_model, count):
    with pytest.raises(ValueError, match="arbitrary N is static-only"):
        extract_contextual_slots(contract_model, torch.empty(1, 3, count, 224, 224, device="meta"))
    assert not contract_model.core.head._forward_pre_hooks


def test_contract_only_context_hook_preserves_user_hooks(contract_model):
    seen = []
    user_hook = contract_model.core.head.register_forward_pre_hook(lambda *args: seen.append(1))
    try:
        result = extract_contextual_slots(contract_model, torch.zeros(2, 3, 16, 224, 224))
        expected = torch.arange(2 * 768 * 16).reshape(2, 768, 16).transpose(1, 2).float()
        torch.testing.assert_close(result["s2d_contextual_slots"], expected)
        assert len(contract_model.core.head._forward_pre_hooks) == 1 and seen == [1]
    finally:
        user_hook.remove()


def test_contract_only_hook_cleanup_on_forward_error(contract_model, monkeypatch):
    def fail(value):
        raise RuntimeError("TEST_SYNTHETIC_HEAD_FAILURE")

    monkeypatch.setattr(contract_model.core.head, "forward", fail)
    with pytest.raises(RuntimeError, match="TEST_SYNTHETIC_HEAD_FAILURE"):
        extract_contextual_slots(contract_model, torch.zeros(1, 3, 16, 224, 224))
    assert not contract_model.core.head._forward_pre_hooks


def _state_digest(model):
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _capture_boundaries(model, video, call):
    captured = {}

    def capture(name):
        def hook(module, inputs, output):
            captured[name] = output.detach().clone()
        return hook

    def capture_prompt(module, inputs, output):
        captured["mcp_return0"] = output[0].detach().clone()

    handles = [
        model.core.patch_embed.register_forward_hook(capture("a0")),
        model.core.prompt_blocks[0].conv1x1.register_forward_hook(capture("c0")),
        model.core.prompt_blocks[0].register_forward_hook(capture_prompt),
        model.core.fc_norm.register_forward_hook(capture("b_slots")),
    ]
    try:
        with torch.inference_mode():
            result = call(model, video)
    finally:
        for handle in handles:
            handle.remove()
    return result, captured


@pytest.fixture(scope="module")
def canonical_results():
    missing = [name for name in ("timm", "einops") if importlib.util.find_spec(name) is None]
    if missing:
        pytest.skip("REAL CANONICAL CORE BLOCKED: missing " + ", ".join(missing)
                    + "; run this file in Ubuntu s2d_iemocap; no mock alignment fallback")
    device = torch.device(os.environ.get("S2D_FEATURE_TEST_DEVICE", "cpu"))
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(206)
        # No checkpoint paths: real full-size classes with random test weights.
        model = build_s2d(VOCAB, {}).to(device).eval()
    generator = torch.Generator().manual_seed(207)
    video = torch.randn(1, 3, 16, 224, 224, generator=generator).to(device)
    before = _state_digest(model)
    flags_before = {name: value.requires_grad for name, value in model.named_parameters()}
    modules_before = {name: id(value) for name, value in model.named_modules()}
    schema_before = {name: tuple(value.shape) for name, value in model.state_dict().items()}
    modes_before = [value.training for value in model.modules()]
    forward_before = model.core.forward.__func__

    full, full_trace = _capture_boundaries(model, video, lambda instance, pixels: instance(pixels))
    # A forbidden hook proves the independent extractor never calls any TMA,
    # ViT block, temporal head, or complete Prompt_block.forward.
    forbidden_calls = []

    def forbidden(module, inputs):
        forbidden_calls.append(type(module).__name__)
        raise AssertionError("Static extraction executed a contextual module")

    handles = [module.register_forward_pre_hook(forbidden) for module in (
        *model.core.blocks, *(prompt.TMA for prompt in model.core.prompt_blocks),
        model.core.prompt_blocks[0], model.core.head,
    )]
    try:
        static, static_trace = _capture_boundaries(model, video, extract_static_features)
        repeat_static = extract_static_features(model, video)
        changed = video.clone()
        changed[:, :, 7] += 0.75  # Only frame j changes; all i != j remain identical.
        changed_static = extract_static_features(model, changed)
        dense = torch.cat((video, changed), dim=2)  # Synthetic only, no dense sampling.
        n32 = extract_static_features(model, dense)
        regrouped = extract_static_features(model, dense.reshape(1, 3, 2, 16, 224, 224)
                                            .permute(0, 2, 1, 3, 4, 5).reshape(2, 3, 16, 224, 224))
        single = extract_static_features(model, video[:, :, 3:4])
    finally:
        for handle in handles:
            handle.remove()
    contextual = extract_contextual_slots(model, video)["s2d_contextual_slots"]
    repeat_contextual = extract_contextual_slots(model, video)["s2d_contextual_slots"]
    return dict(
        full=full, full_trace=full_trace, static=static, static_trace=static_trace,
        repeat_static=repeat_static, changed=changed_static, n32=n32, regrouped=regrouped,
        single=single, contextual=contextual, repeat_contextual=repeat_contextual,
        forbidden_calls=forbidden_calls, state_before=before, state_after=_state_digest(model),
        flags_before=flags_before,
        flags_after={name: value.requires_grad for name, value in model.named_parameters()},
        modules_before=modules_before,
        modules_after={name: id(value) for name, value in model.named_modules()},
        schema_before=schema_before,
        schema_after={name: tuple(value.shape) for name, value in model.state_dict().items()},
        modes_before=modes_before, modes_after=[value.training for value in model.modules()],
        forward_before=forward_before, forward_after=model.core.forward.__func__,
        num_frames=model.core.num_frames,
        tma_frames=[prompt.TMA.num_frames for prompt in model.core.prompt_blocks],
        temporal_embedding_shape=tuple(model.core.temporal_embedding.shape),
    )


def test_canonical_a0_c0_d0_shapes_and_numeric_alignment(canonical_results):
    result = canonical_results
    full, static = result["full_trace"], result["static_trace"]
    assert full["a0"].shape == (16, 196, 768)
    assert full["c0"].shape == (16, 768, 14, 14)
    # Compare raw boundaries too, not merely shapes or only reduced vectors.
    torch.testing.assert_close(full["c0"], full["mcp_return0"], rtol=1e-5, atol=1e-6)
    for key in ("a0", "c0"):
        torch.testing.assert_close(static[key], full[key], rtol=1e-5, atol=1e-6)
    c0_tokens = full["c0"].flatten(2).transpose(1, 2).contiguous()
    expected = (full["a0"].mean(1), c0_tokens.mean(1), (full["a0"] + c0_tokens).mean(1))
    assert set(result["static"]) == set(STATIC_KEYS)
    for key, value in zip(STATIC_KEYS, expected):
        assert result["static"][key].shape == (1, 16, 768)
        torch.testing.assert_close(result["static"][key], value.reshape(1, 16, 768), rtol=1e-5, atol=1e-6)
    assert result["forbidden_calls"] == []


def test_canonical_frame_independence(canonical_results):
    result = canonical_results
    unchanged = [index for index in range(16) if index != 7]
    for key in STATIC_KEYS:
        torch.testing.assert_close(result["static"][key][:, unchanged],
                                   result["changed"][key][:, unchanged], rtol=1e-5, atol=1e-6)
    # The test input change is non-vacuous at the RGB representation boundary.
    assert not torch.allclose(result["static"]["a0_rgb_mean"][:, 7],
                              result["changed"]["a0_rgb_mean"][:, 7])


def test_canonical_n32_static_and_frame_batch_order(canonical_results):
    result = canonical_results
    for key in STATIC_KEYS:
        assert result["n32"][key].shape == (1, 32, 768)
        torch.testing.assert_close(result["n32"][key][:, :16], result["static"][key], rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(result["n32"][key][:, 16:], result["changed"][key], rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(result["n32"][key].reshape(2, 16, 768),
                                   result["regrouped"][key], rtol=1e-5, atol=1e-6)
        assert result["single"][key].shape == (1, 1, 768)
        torch.testing.assert_close(result["single"][key], result["static"][key][:, 3:4], rtol=1e-5, atol=1e-6)
    assert result["num_frames"] == 16 and result["tma_frames"] == [16] * 12
    assert result["temporal_embedding_shape"] == (1, 16, 768)


def test_canonical_b_shape_and_mean_equals_eval_hv(canonical_results):
    result = canonical_results
    assert result["contextual"].shape == (1, 16, 768)
    torch.testing.assert_close(result["contextual"], result["full_trace"]["b_slots"].reshape(1, 16, 768),
                               rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(result["contextual"].mean(1), result["full"]["h_v"], rtol=1e-5, atol=1e-6)


def test_canonical_eval_determinism_and_inference_outputs(canonical_results):
    result = canonical_results
    for key in STATIC_KEYS:
        torch.testing.assert_close(result["static"][key], result["repeat_static"][key], rtol=1e-5, atol=1e-6)
        assert not result["static"][key].requires_grad and result["static"][key].grad_fn is None
    torch.testing.assert_close(result["contextual"], result["repeat_contextual"], rtol=1e-5, atol=1e-6)
    assert not result["contextual"].requires_grad and result["contextual"].grad_fn is None


def test_canonical_no_parameter_buffer_or_requires_grad_mutation(canonical_results):
    result = canonical_results
    assert result["state_before"] == result["state_after"]
    assert result["flags_before"] == result["flags_after"]
    assert result["modes_before"] == result["modes_after"]
    assert result["modules_before"] == result["modules_after"]
    assert result["schema_before"] == result["schema_after"]
    assert result["forward_before"] is result["forward_after"]

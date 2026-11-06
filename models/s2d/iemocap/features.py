"""Read-only S2D representation interfaces for already transformed RGB crops.

D0 is the S2D MCP0 static residual feature, not the full S2D representation.
No videos, checkpoints, caches or feature banks are opened or written here.
"""

from __future__ import annotations

import torch
from torch import fx, nn

from .model import S2DIEMOCAP


class _MCP0Proxy(fx.Proxy):
    @property
    def shape(self):
        # Only the MCP input's spatial/channel dimensions are specialized.
        # BT stays symbolic; no T=16 assumption enters the static graph.
        return (self.size(0), 1536, 14, 14)


class _MCP0Tracer(fx.Tracer):
    def proxy(self, node):
        return _MCP0Proxy(node, self)

    def is_leaf_module(self, module, module_qualified_name):
        # Trace the original Prompt_block.forward, never execute/trace TMA.
        # Fovea and all convolutions remain the exact original module objects.
        return True


def _mcp0_static_graph(prompt: nn.Module) -> fx.GraphModule:
    """Slice return value 0 from canonical MCP code without rewriting its math.

    The resulting graph is a call view sharing the source's leaf modules and
    parameters. Only ancestors of the static output survive; TMA has no node
    or registered submodule in this view. Nothing is attached to the model.
    """
    traced = _MCP0Tracer().trace(prompt)
    output = next(node for node in traced.nodes if node.op == "output")
    values = output.args[0]
    if not isinstance(values, tuple) or len(values) != 2 or not isinstance(values[0], fx.Node):
        raise ValueError("Expected canonical MCP (static_map, temporal_tokens) output")

    required = set()

    def visit(node):
        if node not in required:
            required.add(node)
            for dependency in node.all_input_nodes:
                visit(dependency)

    visit(values[0])
    graph, remap = fx.Graph(), {}
    for node in traced.nodes:
        if node in required:
            remap[node] = graph.node_copy(node, lambda dependency: remap[dependency])
    graph.output(remap[values[0]])
    targets = {node.target for node in graph.nodes if node.op == "call_module"}
    if targets != {"conv0_0", "conv0_1", "fovea", "conv1x1"}:
        raise ValueError(f"Unsupported MCP0 static dependencies: {sorted(targets)}")
    graph.lint()
    return fx.GraphModule(prompt, graph)


def _validate(model: S2DIEMOCAP, video: torch.Tensor, *, contextual: bool) -> None:
    if not isinstance(model, S2DIEMOCAP):
        raise TypeError("Expected the existing canonical S2DIEMOCAP instance")
    if any(module.training for module in model.modules()):
        raise ValueError("Feature extraction requires model.eval() on all modules")
    if (not isinstance(video, torch.Tensor) or video.ndim != 5
            or video.shape[1] != 3 or tuple(video.shape[-2:]) != (224, 224)
            or video.shape[0] < 1 or video.shape[2] < 1):
        raise ValueError("Expected transformed RGB crops [B,3,N,224,224], B,N > 0")
    if not video.is_floating_point():
        raise ValueError("Expected floating-point, ImageNet-normalized RGB crops")
    if contextual and video.shape[2] != 16:
        raise ValueError("Contextual S2D requires [B,3,16,224,224]; arbitrary N is static-only")
    core = model.core
    if (core.num_frames != 16 or core.embed_dim != 768
            or core.prompt_type not in ("shallow", "deep") or core.global_pool != "token"):
        raise ValueError("Expected the canonical S2D-V0 MCP/CLS configuration")


@torch.inference_mode()
def extract_static_features(model: S2DIEMOCAP, video: torch.Tensor) -> dict[str, torch.Tensor]:
    """Return A0/C0/D0 means [B,N,768] for arbitrary positive N.

    ``video`` contains fixed, already cropped/resized/normalized pixels in
    [B,3,N,224,224] order. Call ``model.eval()`` first; this function never
    changes training flags, weights, requires_grad or the checkpoint schema.
    All modules belong to this same model instance. TMA, temporal embedding,
    ViT blocks and the classification head are absent from this computation.
    Outputs are inference tensors, intended for read-only feature extraction.
    """
    _validate(model, video, contextual=False)
    b, _, n, _, _ = video.shape
    core = model.core
    frames = video.permute(0, 2, 1, 3, 4).reshape(b * n, 3, 224, 224)
    a0 = core.patch_embed(frames)
    landmark = model.landmark(video)
    if tuple(landmark.shape) != (b * n, 128, 14, 14):
        raise ValueError("Expected canonical landmark maps [B*N,128,14,14]")
    prompt_tokens = core.patch_embed_prompt(landmark)
    if tuple(a0.shape) != (b * n, 196, 768) or prompt_tokens.shape != a0.shape:
        raise ValueError("Expected canonical RGB/landmark tokens [B*N,196,768]")
    norm = core.prompt_norms[0]
    rgb_map = norm(a0).transpose(1, 2).reshape(b * n, 768, 14, 14).contiguous()
    landmark_map = norm(prompt_tokens).transpose(1, 2).reshape(b * n, 768, 14, 14).contiguous()
    static = _mcp0_static_graph(core.prompt_blocks[0])
    c0 = static(torch.cat((rgb_map, landmark_map), dim=1))
    if tuple(c0.shape) != (b * n, 768, 14, 14):
        raise ValueError("Expected MCP0 static maps [B*N,768,14,14]")
    c0_tokens = c0.flatten(2).transpose(1, 2).contiguous()
    # Deliberately add tokens before reduction, preserving the D0 definition.
    d0_tokens = a0 + c0_tokens
    return {
        "a0_rgb_mean": a0.mean(dim=1).reshape(b, n, 768),
        "c0_mcp_static_mean": c0_tokens.mean(dim=1).reshape(b, n, 768),
        "d0_static_residual_mean": d0_tokens.mean(dim=1).reshape(b, n, 768),
    }


@torch.inference_mode()
def extract_contextual_slots(model: S2DIEMOCAP, video: torch.Tensor) -> dict[str, torch.Tensor]:
    """Capture B [B,16,768] from the unchanged full canonical forward.

    The head input is post-ln_post/post-CLS, pre-temporal-mean and pre-dropout.
    Its slots depend on the complete 16-frame context. Their temporal mean
    equals eval h_v within floating-point tolerance. Use this model instance
    serially while the temporary hook is installed; cleanup also runs on error.
    """
    _validate(model, video, contextual=True)
    captured = []

    def capture_head_input(module, inputs):
        value = inputs[0]
        if tuple(value.shape) != (video.shape[0], 768, 16, 1, 1):
            raise ValueError("Expected canonical pre-head tensor [B,768,16,1,1]")
        captured.append(value[..., 0, 0].transpose(1, 2).contiguous().clone())

    handle = model.core.head.register_forward_pre_hook(capture_head_input)
    try:
        model(video)
    finally:
        handle.remove()
    if len(captured) != 1:
        raise RuntimeError(f"Expected one canonical head call, got {len(captured)}")
    return {"s2d_contextual_slots": captured[0]}

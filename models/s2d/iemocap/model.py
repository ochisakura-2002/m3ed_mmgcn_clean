"""IEMOCAP wrapper around the unmodified official S2D MCP/TMA core."""

from __future__ import annotations

from functools import lru_cache
import hashlib
import importlib.util
from pathlib import Path
import sys

import torch
from torch import nn
from torch.nn import functional as F

from datasets.iemocap.label_vocab import LabelVocab


UPSTREAM_ROOT = Path(__file__).resolve().parents[3] / "third_party" / "S2D"


@lru_cache(maxsize=None)
def _upstream_module(relative_path: str):
    path = UPSTREAM_ROOT / relative_path
    name = "_merc_official_s2d_" + path.stem
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load upstream source: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(name, None)
        raise
    return module


def load_weights(module: nn.Module, checkpoint_path: str | Path, *, strict: bool,
                 drop_classifier: bool = False, s2d_pretrained: bool = False,
                 expected_sha256: str | None = None,
                 trusted_legacy_pickle: bool = False) -> dict:
    """Load local weights safely by default; never download or auto-fallback.

    strict=True enforces all names and shapes. Non-strict reports missing and
    unexpected keys but still rejects shape mismatches. Explicit classifier
    replacement is allowed only with strict=False. All other mismatches fail.
    Full-pickle loading requires explicit legacy trust and a matching SHA256.
    S2D pretrained weights always use weights_only=True.
    """
    if strict and drop_classifier:
        raise ValueError("Classifier replacement requires strict=False")
    if not isinstance(trusted_legacy_pickle, bool):
        raise ValueError("trusted_legacy_pickle must be a boolean")
    if s2d_pretrained and trusted_legacy_pickle:
        raise ValueError("S2D pretrained checkpoints require weights_only=True")
    if trusted_legacy_pickle and not expected_sha256:
        raise ValueError("trusted_legacy_pickle=True requires a nonempty expected_sha256")
    weights_only = not trusted_legacy_pickle
    sha256_verified = False
    with Path(checkpoint_path).open("rb") as stream:
        if expected_sha256:
            digest = hashlib.sha256()
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
            actual_sha256 = digest.hexdigest()
            if actual_sha256 != expected_sha256:
                raise ValueError(f"Checkpoint SHA256 mismatch for {checkpoint_path}: "
                                 f"expected {expected_sha256}, got {actual_sha256}")
            sha256_verified = True
            stream.seek(0)
        checkpoint = torch.load(stream, map_location="cpu", weights_only=weights_only)
    state = checkpoint
    state_source = "raw_state_dict"
    for key in ("model_state_dict", "model", "state_dict", "module"):
        if isinstance(checkpoint, dict) and isinstance(checkpoint.get(key), dict):
            state = checkpoint[key]
            state_source = key
            break
    if not isinstance(state, dict) or not state or not all(isinstance(v, torch.Tensor) for v in state.values()):
        raise ValueError("Expected a nonempty tensor state_dict")
    state = {k.removeprefix("module."): v for k, v in state.items()}
    dropped = []
    if s2d_pretrained:
        # These are the upstream _create_vision_transformer compatibility rules.
        for key in list(state):
            if "landmark" in key:
                dropped.append(key)
                del state[key]
        for suffix in ("weight", "bias"):
            key = f"norm.{suffix}"
            if key in state and f"ln_post.{suffix}" not in state:
                state[f"ln_post.{suffix}"] = state.pop(key)
    if drop_classifier:
        for key in ("head.fc_cls.weight", "head.fc_cls.bias"):
            if key in state:
                dropped.append(key)
                del state[key]
    own = module.state_dict()
    mismatches = [k for k in state if k in own and state[k].shape != own[k].shape]
    if mismatches:
        raise ValueError(f"Checkpoint shape mismatches (not silently skipped): {mismatches}")
    compatible = {k for k in state if k in own and state[k].shape == own[k].shape}
    if not compatible:
        raise ValueError("Checkpoint has no compatible parameter names")
    result = module.load_state_dict(state, strict=strict)
    return {"path": str(checkpoint_path), "strict": strict,
            "weights_only": weights_only, "sha256_verified": sha256_verified,
            "trusted_legacy_pickle": trusted_legacy_pickle, "state_source": state_source,
            "missing_keys": list(result.missing_keys), "unexpected_keys": list(result.unexpected_keys),
            "dropped_keys": dropped, "loaded_key_count": len(compatible)}


class FrozenLandmark(nn.Module):
    """Official MobileFaceNet + official Landmark.forward preprocessing.

    Constructed here solely to make its checkpoint path configurable and to
    avoid upstream Landmark's hard-coded checkpoint/CUDA path. Its output is
    the 128x14x14 feature map, never landmark coordinate concatenation.
    """

    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.requires_grad_(False)
        self.eval()

    def train(self, mode: bool = True):
        super().train(False)  # Also freeze BatchNorm statistics.
        return self

    @torch.no_grad()
    def forward(self, video):
        b, c, t, h, w = video.shape
        frames = video.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
        frames = F.interpolate(frames, size=(112, 112), mode="bilinear", align_corners=True)
        _, features = self.backbone(frames)
        return features


class S2DIEMOCAP(nn.Module):
    def __init__(self, core: nn.Module, landmark: nn.Module, vocab: LabelVocab,
                 num_frames: int = 16, input_size: int = 224):
        super().__init__()
        if (num_frames, input_size) != (16, 224):
            raise ValueError("S2D-V0 freezes 16 frames at 224x224")
        self.core, self.landmark, self.vocab = core, landmark, vocab
        self.num_frames, self.input_size = num_frames, input_size
        self.load_reports = {}
        # Same trainable-name policy as official run_class_finetuning.py.
        trainable = ("temporal_embedding", "temporal_attn", "cls_token_t", "ln_post",
                     "cls_head", "Adapter", "head", "prompt")
        for name, param in self.core.named_parameters():
            param.requires_grad = any(token in name for token in trainable)
        self.landmark.requires_grad_(False)
        self.landmark.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        self.landmark.eval()
        return self

    def forward(self, video):
        if video.ndim != 5 or tuple(video.shape[1:]) != (3, 16, 224, 224):
            raise ValueError("S2D expects video [B,3,16,224,224]")
        with torch.no_grad():
            landmark_features = self.landmark(video)
        expected = (video.shape[0] * self.num_frames, 128, 14, 14)
        if tuple(landmark_features.shape) != expected:
            raise ValueError(f"Official MCP expects landmark feature maps {expected}")
        # Official forward returns head logits and its pooled pre-classifier feature.
        logits, features = self.core(video, landmark_features)
        if logits.shape != (video.shape[0], self.vocab.num_classes):
            raise ValueError("S2D classifier output disagrees with shared label vocabulary")
        return {"h_v": features, "logits_v": logits, "prob_v": logits.softmax(dim=-1)}


def build_s2d(vocab: LabelVocab, config: dict) -> S2DIEMOCAP:
    if not config.get("enabled", True):
        raise ValueError("S2D branch disabled; existing TA baseline entrypoints remain unchanged")
    if config.get("sdl", False):
        raise ValueError("SDL must remain OFF for S2D-V0")
    if (config.get("num_frames", 16), config.get("input_size", 224)) != (16, 224):
        raise ValueError("S2D-V0 requires 16 frames at 224x224")
    if config.get("landmark_strict", True) is not True:
        raise ValueError("S2D-V0 requires landmark_strict=True")
    try:
        source = _upstream_module("model_finetuning.py")
    except ImportError as error:
        raise ImportError("Official S2D requires remote timm==0.6.13 and einops; no auto-install") from error
    # pretrained=False prevents all implicit loading/downloading.
    core = source.s2d_base_patch16_224(
        pretrained=False, num_classes=vocab.num_classes,
        num_frames=config.get("num_frames", 16), img_size=config.get("input_size", 224),
        in_chans_l=128, adapter_scale=config.get("adapter_scale", 0.25),
        head_dropout_ratio=config.get("head_dropout_ratio", 0.5))
    mobile = _upstream_module("landmark/mobilefacenet.py").MobileFaceNet([112, 112], 136)
    reports = {}
    if config.get("pretrained_checkpoint"):
        reports["s2d"] = load_weights(core, config["pretrained_checkpoint"],
                                     strict=config.get("pretrained_strict", False),
                                     drop_classifier=config.get("replace_fer_classifier", True),
                                     s2d_pretrained=True,
                                     expected_sha256=config.get("pretrained_checkpoint_sha256"))
    if config.get("landmark_checkpoint"):
        reports["landmark"] = load_weights(mobile, config["landmark_checkpoint"],
                                          strict=True,
                                          expected_sha256=config.get("landmark_checkpoint_sha256"),
                                          trusted_legacy_pickle=config.get("landmark_trusted_legacy_pickle", False))
    model = S2DIEMOCAP(core, FrozenLandmark(mobile), vocab,
                      config.get("num_frames", 16), config.get("input_size", 224))
    if "s2d" in reports:
        missing = set(reports["s2d"]["missing_keys"])
        uninitialized = [n for n, p in core.named_parameters() if not p.requires_grad and n in missing]
        if uninitialized:
            raise ValueError(f"Pretrained checkpoint leaves frozen ViT parameters uninitialized: {uninitialized}")
    model.load_reports = reports
    return model

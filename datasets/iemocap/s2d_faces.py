"""Replaceable face backend contract; no detector installation or face cache."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import Protocol
import warnings

from PIL import Image


@dataclass(frozen=True)
class FaceResult:
    face: Image.Image | None  # RGB crop/aligned face, before normalization
    face_valid: bool
    reason: str = ""


class FaceBackend(Protocol):
    test_only: bool

    def __call__(self, frame: Image.Image) -> FaceResult:
        """Detect/crop/align the target speaker in a speaker-specific RGB frame."""
        ...


class MockFullFrameBackend:
    """TEST ONLY: treats the whole synthetic frame as a face. Not a detector."""

    test_only = True

    def __call__(self, frame: Image.Image) -> FaceResult:
        return FaceResult(frame.copy(), True)


class FaceProcessor:
    def __init__(self, backend: FaceBackend, input_size: int = 224):
        if input_size != 224:
            raise ValueError("S2D-V0 freezes input_size=224")
        self.backend = backend
        self.input_size = input_size
        self.test_only = bool(getattr(backend, "test_only", False))

    def __call__(self, frame: Image.Image) -> FaceResult:
        if not isinstance(frame, Image.Image) or frame.mode != "RGB":
            raise ValueError("FaceProcessor expects a PIL RGB frame")
        result = self.backend(frame)
        if not isinstance(result, FaceResult):
            raise TypeError("Face backend must return FaceResult")
        if not result.face_valid:
            if not result.reason:
                raise ValueError("Failed face detection requires an explicit reason")
            warnings.warn(f"FaceProcessor: {result.reason}", RuntimeWarning, stacklevel=2)
            return FaceResult(None, False, result.reason)
        if not isinstance(result.face, Image.Image) or result.face.mode != "RGB":
            raise ValueError("Successful backend must return an RGB crop/aligned face")
        face = result.face.resize((self.input_size, self.input_size), Image.Resampling.BILINEAR)
        return FaceResult(face, True)


def build_face_processor(config: dict, *, allow_test_backend: bool = False) -> FaceProcessor:
    """Remote backend is an explicit 'module:factory' with optional kwargs."""
    name = config.get("backend")
    if not name:
        raise ValueError("Configure face.backend after remote detector selection/validation")
    if name == "mock_full_frame":
        backend = MockFullFrameBackend()
    else:
        module_name, factory_name = name.split(":", 1)
        backend = getattr(import_module(module_name), factory_name)(**config.get("kwargs", {}))
    processor = FaceProcessor(backend, input_size=config.get("input_size", 224))
    if processor.test_only and not allow_test_backend:
        raise ValueError("TEST/mock face backend is forbidden for real-data training/evaluation")
    return processor

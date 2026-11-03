"""Video decoding, face processing and S2D tensor adaptation (lazy video IO)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

from .s2d_faces import (CachedBBoxFaceProcessor, FaceBBoxCache, FaceCacheEntry,
                        FaceProcessor, PREPROCESSING_VERSION)
from .s2d_manifest import S2DManifest
from .s2d_sampling import Uniform16Sampler


class S2DTransform:
    """Official RGB/ImageNet normalization and C,T,H,W layout.

    S2D-V0 uses deterministic preprocessing for every split. The upstream
    optional RandAugment/spatial crop/flip/erasing policies are disabled.
    """

    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    def __init__(self, input_size: int = 224, num_frames: int = 16):
        if (input_size, num_frames) != (224, 16):
            raise ValueError("S2D-V0 requires 16 frames at 224x224")
        self.input_size, self.num_frames = input_size, num_frames

    def __call__(self, faces: list[Image.Image]) -> torch.Tensor:
        if len(faces) != self.num_frames:
            raise ValueError("Expected exactly 16 processed face frames")
        for face in faces:
            if face.mode != "RGB" or face.size != (self.input_size, self.input_size):
                raise ValueError("Transform expects RGB 224x224 faces from FaceProcessor")
        # THWC uint8 -> CTHW float32, matching official ClipToTensor.
        clip = torch.from_numpy(np.stack([np.asarray(f) for f in faces])).permute(3, 0, 1, 2)
        clip = clip.to(dtype=torch.float32).div(255)
        mean = clip.new_tensor(self.mean).view(3, 1, 1, 1)
        std = clip.new_tensor(self.std).view(3, 1, 1, 1)
        return ((clip - mean) / std).contiguous()


def decode_uniform_clip(path: Path, sampler: Uniform16Sampler):
    """Decode only selected RGB frames. Never resample another utterance on error."""
    try:
        from decord import VideoReader, cpu
    except ImportError as error:
        raise ImportError("Install decord in the remote S2D environment for video decoding") from error
    if not path.is_file():
        raise FileNotFoundError(f"Available video is missing: {path}")
    reader = VideoReader(str(path), ctx=cpu(0), num_threads=1)
    indices = sampler(len(reader))
    frames = reader.get_batch(list(indices)).asnumpy()
    return [Image.fromarray(frame).convert("RGB") for frame in frames], indices


class IEMOCAPS2DDataset(Dataset):
    def __init__(self, manifest: S2DManifest, split: str, face_processor: FaceProcessor,
                 sampler: Uniform16Sampler | None = None, transform: S2DTransform | None = None,
                 decoder=decode_uniform_clip, *, allow_test_backend: bool = False,
                 face_cache: FaceBBoxCache | str | Path | None = None,
                 require_face_cache_for_training: bool = False, remote_preprocessing: bool = False):
        self.manifest = manifest
        self.records = manifest.for_split(split, available_only=True)
        self.split = split
        self.face_processor = face_processor
        if face_processor.test_only and not allow_test_backend:
            raise ValueError("TEST/mock face backend requires explicit test-only use")
        if remote_preprocessing and face_cache is not None:
            raise ValueError("Remote preprocessing generates a new cache; do not supply an input cache")
        self.remote_preprocessing = remote_preprocessing
        self.face_cache = FaceBBoxCache.load(face_cache) if isinstance(face_cache, (str, Path)) else face_cache
        if self.face_cache is not None:
            self.face_cache.require_ids(r.utterance_id for r in self.records)
            # A supplied cache always bypasses the configured detector, including MTCNN.
            self.face_processor = CachedBBoxFaceProcessor()
        elif not remote_preprocessing:
            if face_processor.online_detection:
                raise ValueError("Online MTCNN is forbidden in training; use remote_preprocessing=True to build cache")
            if isinstance(face_processor, CachedBBoxFaceProcessor) or require_face_cache_for_training:
                raise ValueError("Training requires a face bbox cache")
        elif not face_processor.online_detection:
            raise ValueError("Remote preprocessing requires the mtcnn_cpu face processor")
        self.excluded_face_utterance_ids = tuple(
            r.utterance_id for r in self.records
            if self.face_cache is not None and not self.face_cache.entries[r.utterance_id].face_usable)
        self.records = tuple(r for r in self.records if r.utterance_id not in self.excluded_face_utterance_ids)
        self.sampler = sampler or Uniform16Sampler()
        self.transform = transform or S2DTransform()
        self.decoder = decoder

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        try:
            frames, indices = self.decoder(record.video_path, self.sampler)
            if len(frames) != 16 or len(indices) != 16:
                raise ValueError("Decoder must return exactly 16 sampled frames/indices")
            if self.face_cache is not None:
                clip = self.face_processor.process_cached_clip(
                    frames, indices, self.face_cache.entries[record.utterance_id])
            else:
                clip = self.face_processor.process_clip(frames)
            if self.remote_preprocessing:
                # Iterate sequentially, not via training DataLoader. Failed clips are also retained.
                return FaceCacheEntry.from_clip(record.utterance_id, indices, frames, clip).as_dict()
            faces = []
            for position, result in zip(indices, clip.frames):
                if not result.face_valid:
                    raise ValueError(f"face_valid=False at frame {position}: {result.reason}")
                faces.append(result.face)
            video = self.transform(faces)
        except Exception as error:
            raise RuntimeError(f"{record.utterance_id}: visual preprocessing failed: {error}") from error
        return {"utterance_id": record.utterance_id, "video_path": str(record.video_path),
                "benchmark_label": record.benchmark_label, "label_index": record.label_index,
                "split": record.split, "v_available": True, "v_valid": True,
                "frame_indices": torch.tensor(indices), "video": video,
                "face_usable": clip.face_usable, "valid_detection_count": clip.valid_detection_count,
                "detection_valid": torch.tensor([r.detection_valid for r in clip.frames]),
                "fallback_used": torch.tensor([r.fallback_used for r in clip.frames]),
                "selected_bbox": torch.tensor([r.selected_bbox if r.selected_bbox is not None else (-1.,) * 4
                                                for r in clip.frames], dtype=torch.float32),
                "detection_probability": torch.tensor([
                    r.detection_probability if r.detection_probability is not None else -1.
                    for r in clip.frames], dtype=torch.float32),
                "face_reason": [r.reason for r in clip.frames],
                "preprocessing_version": PREPROCESSING_VERSION if self.face_cache is not None else "uncached"}

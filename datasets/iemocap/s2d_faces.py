"""S2D V0 CPU face tracking and auditable, versioned bbox cache contract."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import json
import math
from pathlib import Path
from typing import Protocol
import warnings

from PIL import Image


PREPROCESSING_VERSION = "s2d_iemocap_v0_uniform16_mtcnn_cpu_v1"
BBox = tuple[float, float, float, float]  # original-frame xyxy, before margin


def validate_face_config(config: dict) -> None:
    if config.get("backend") not in ("mtcnn_cpu", "cached_bbox"):
        raise ValueError("S2D V0 face.backend must be mtcnn_cpu or cached_bbox")
    frozen = {"device": "cpu", "min_size": 20, "prob_threshold": 0.90,
              "margin": 1.40, "alignment": False, "full_frame_fallback": False,
              "input_size": 224}
    for key, expected in frozen.items():
        value = config.get(key, expected)
        if value != expected or (isinstance(expected, bool) and type(value) is not bool):
            raise ValueError(f"S2D V0 freezes face.{key}={expected}")
    if not isinstance(config.get("require_cache_for_training", True), bool):
        raise ValueError("face.require_cache_for_training must be boolean")


def _bbox(value) -> BBox:
    if len(value) != 4 or not all(math.isfinite(float(v)) for v in value):
        raise ValueError("bbox must contain four finite xyxy coordinates")
    box = tuple(float(v) for v in value)
    if box[0] < 0 or box[1] < 0 or box[2] <= box[0] or box[3] <= box[1]:
        raise ValueError("bbox must have nonnegative origin and positive area")
    return box


def square_crop_box(box: BBox, image_size: tuple[int, int], margin: float = 1.40):
    """Expand max(width,height) by margin, then shift/cap a square inside the image."""
    x1, y1, x2, y2 = _bbox(box)
    width, height = image_size
    if width < 1 or height < 1 or x2 > width or y2 > height or margin != 1.40:
        raise ValueError("Invalid image/bbox or non-V0 margin (requires 1.40)")
    side = min(math.ceil(max(x2 - x1, y2 - y1) * margin), width, height)
    left = max(0, min(math.floor((x1 + x2 - side) / 2), width - side))
    top = max(0, min(math.floor((y1 + y2 - side) / 2), height - side))
    return left, top, left + side, top + side


@dataclass(frozen=True)
class FaceResult:
    face: Image.Image | None  # RGB 224x224 after processor; None if unusable
    face_valid: bool
    reason: str = ""
    detection_valid: bool = False
    fallback_used: bool = False
    selected_bbox: BBox | None = None
    detection_probability: float | None = None  # None on detection failure, even with fallback
    crop_bbox: tuple[int, int, int, int] | None = None


@dataclass(frozen=True)
class FaceClipResult:
    frames: tuple[FaceResult, ...]
    face_usable: bool
    valid_detection_count: int


@dataclass(frozen=True)
class FaceCacheEntry:
    utterance_id: str
    frame_indices: tuple[int, ...]
    frame_sizes: tuple[tuple[int, int], ...]
    selected_bboxes: tuple[BBox | None, ...]
    detection_probabilities: tuple[float | None, ...]
    detection_valid: tuple[bool, ...]
    fallback_used: tuple[bool, ...]
    reasons: tuple[str, ...]
    valid_detection_count: int
    face_usable: bool
    preprocessing_version: str = PREPROCESSING_VERSION

    def __post_init__(self):
        if not isinstance(self.utterance_id, str) or not self.utterance_id.strip() or self.preprocessing_version != PREPROCESSING_VERSION:
            raise ValueError("Missing utterance_id or incompatible face preprocessing_version")
        arrays = (self.frame_indices, self.frame_sizes, self.selected_bboxes,
                  self.detection_probabilities, self.detection_valid, self.fallback_used, self.reasons)
        if any(len(v) != 16 for v in arrays):
            raise ValueError("Face cache requires exactly 16 entries per field")
        if any(type(i) is not int or i < 0 for i in self.frame_indices) or list(self.frame_indices) != sorted(self.frame_indices):
            raise ValueError("Face cache frame_indices must be sorted nonnegative integers")
        if any(type(v) is not bool for v in (*self.detection_valid, *self.fallback_used, self.face_usable)):
            raise ValueError("Face cache flags must be boolean")
        if type(self.valid_detection_count) is not int or self.valid_detection_count != sum(self.detection_valid):
            raise ValueError("Face cache valid_detection_count disagrees with flags")
        if self.face_usable != (self.valid_detection_count > 0):
            raise ValueError("face_usable must reflect at least one valid detection")
        previous = None
        first = next((b for b, valid in zip(self.selected_bboxes, self.detection_valid) if valid), None)
        for size, box, prob, valid, fallback, reason in zip(
                self.frame_sizes, self.selected_bboxes, self.detection_probabilities,
                self.detection_valid, self.fallback_used, self.reasons):
            if len(size) != 2 or any(type(v) is not int or v < 1 for v in size):
                raise ValueError("Face cache requires positive integer frame_sizes")
            if not isinstance(reason, str) or not reason:
                raise ValueError("Every cache frame requires an auditable reason")
            if valid:
                if box is None or fallback or prob is None or not math.isfinite(prob) or not 0.90 <= prob <= 1:
                    raise ValueError("Invalid detected bbox/probability/fallback contract")
                previous = box
            elif prob is not None or fallback != self.face_usable or (box is None) == self.face_usable:
                raise ValueError("Invalid failed-detection/fallback contract")
            elif self.face_usable and tuple(box) != tuple(previous if previous is not None else first):
                raise ValueError("Fallback bbox must reuse nearest previous/first valid bbox")
            if box is not None:
                square_crop_box(box, size)  # Also rejects out-of-frame cache coordinates.

    @classmethod
    def from_clip(cls, utterance_id, frame_indices, frames, result: FaceClipResult):
        return cls(utterance_id, tuple(int(i) for i in frame_indices), tuple(f.size for f in frames),
                   tuple(r.selected_bbox for r in result.frames),
                   tuple(r.detection_probability for r in result.frames),
                   tuple(r.detection_valid for r in result.frames),
                   tuple(r.fallback_used for r in result.frames), tuple(r.reason for r in result.frames),
                   result.valid_detection_count, result.face_usable)

    @classmethod
    def from_dict(cls, value):
        data = dict(value)
        if data.get("preprocessing_version") != PREPROCESSING_VERSION:
            raise ValueError("Missing/incompatible face entry preprocessing_version")
        for key in ("frame_indices", "frame_sizes", "selected_bboxes", "detection_probabilities",
                    "detection_valid", "fallback_used", "reasons"):
            data[key] = tuple(tuple(v) if isinstance(v, list) else v for v in data[key])
        return cls(**data)

    def as_dict(self):
        return asdict(self)


class FaceBBoxCache:
    """JSON envelope keyed by utterance_id. Save is explicit and never overwrites."""

    def __init__(self, entries: dict[str, FaceCacheEntry]):
        self.entries = dict(entries)
        for uid, entry in self.entries.items():
            if not isinstance(entry, FaceCacheEntry) or uid != entry.utterance_id:
                raise ValueError("Cache key must equal entry.utterance_id")

    @classmethod
    def load(cls, path: str | Path):
        def unique_keys(pairs):
            result = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError(f"Duplicate face cache key: {key}")
                result[key] = value
            return result
        with Path(path).open(encoding="utf-8") as stream:
            payload = json.load(stream, object_pairs_hook=unique_keys)
        if payload.get("preprocessing_version") != PREPROCESSING_VERSION:
            raise ValueError("Incompatible face cache preprocessing_version")
        return cls({uid: FaceCacheEntry.from_dict(v) for uid, v in payload["entries"].items()})

    def save(self, path: str | Path):
        payload = {"preprocessing_version": PREPROCESSING_VERSION,
                   "entries": {uid: v.as_dict() for uid, v in self.entries.items()}}
        with Path(path).open("x", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, allow_nan=False)
            stream.write("\n")

    def require_ids(self, utterance_ids):
        missing = set(utterance_ids) - self.entries.keys()
        if missing:
            raise ValueError(f"Missing face cache entries: {sorted(missing)}")


class FaceBackend(Protocol):
    test_only: bool

    def __call__(self, frame: Image.Image) -> FaceResult:
        """Crop the target speaker in a speaker-specific RGB frame."""
        ...


class MockFullFrameBackend:
    """TEST ONLY: treats the whole synthetic frame as a face. Not a detector."""

    test_only = True

    def __call__(self, frame: Image.Image) -> FaceResult:
        return FaceResult(frame.copy(), True, "test_only_full_frame", detection_valid=True)


class FaceProcessor:
    online_detection = False

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
            return replace(result, face=None)
        if not isinstance(result.face, Image.Image) or result.face.mode != "RGB":
            raise ValueError("Successful backend must return an RGB face crop")
        face = result.face.resize((self.input_size, self.input_size), Image.Resampling.BILINEAR)
        return replace(result, face=face)

    def process_clip(self, frames: list[Image.Image]) -> FaceClipResult:
        results = tuple(self(frame) for frame in frames)
        return FaceClipResult(results, all(r.face_valid for r in results),
                              sum(r.detection_valid for r in results))


class CachedBBoxFaceProcessor(FaceProcessor):
    test_only = False

    def __init__(self):
        self.input_size = 224

    def __call__(self, frame):
        raise ValueError("cached_bbox requires an utterance face cache entry")

    def process_cached_clip(self, frames, indices, entry: FaceCacheEntry) -> FaceClipResult:
        if tuple(indices) != entry.frame_indices or tuple(f.size for f in frames) != entry.frame_sizes:
            raise ValueError("Decoded frame indices/sizes disagree with face cache")
        results = []
        for frame, box, prob, valid, fallback, reason in zip(
                frames, entry.selected_bboxes, entry.detection_probabilities,
                entry.detection_valid, entry.fallback_used, entry.reasons):
            crop_box = square_crop_box(box, frame.size) if box is not None else None
            face = self._crop(frame, crop_box) if crop_box is not None else None
            results.append(FaceResult(face, face is not None, reason, valid, fallback, box, prob, crop_box))
        return FaceClipResult(tuple(results), entry.face_usable, entry.valid_detection_count)

    @staticmethod
    def _crop(frame, crop_box):
        if not isinstance(frame, Image.Image) or frame.mode != "RGB":
            raise ValueError("FaceProcessor expects a PIL RGB frame")
        return frame.crop(crop_box).resize((224, 224), Image.Resampling.BILINEAR)


class MTCNNFaceProcessor(CachedBBoxFaceProcessor):
    """CPU detect only (no alignment). Tracking state is local to process_clip."""

    online_detection = True

    def __init__(self, *, device="cpu", min_face_size=20, prob_threshold=0.90,
                 margin=1.40, input_size=224, detector=None):
        validate_face_config({"backend": "mtcnn_cpu", "device": device, "min_size": min_face_size,
                              "prob_threshold": prob_threshold, "margin": margin, "input_size": input_size})
        super().__init__()
        self._detector = detector  # Injection for mock tests; real dependency is lazy.

    def _candidates(self, frame):
        if not isinstance(frame, Image.Image) or frame.mode != "RGB":
            raise ValueError("FaceProcessor expects a PIL RGB frame")
        if self._detector is None:
            try:
                from facenet_pytorch import MTCNN
            except ImportError as error:
                raise ImportError("mtcnn_cpu requires facenet-pytorch==2.5.3 in the remote S2D environment") from error
            self._detector = MTCNN(device="cpu", keep_all=True, min_face_size=20)
        boxes, probabilities = self._detector.detect(frame, landmarks=False)
        candidates = []
        if boxes is None or probabilities is None:
            return candidates
        for raw, probability in zip(boxes, probabilities):
            if probability is None or not math.isfinite(float(probability)) or not .90 <= float(probability) <= 1:
                continue
            try:
                raw = tuple(float(v) for v in raw)
                if len(raw) != 4 or not all(math.isfinite(v) for v in raw):
                    continue
                box = _bbox((max(0, raw[0]), max(0, raw[1]),
                             min(frame.width, raw[2]), min(frame.height, raw[3])))
            except (TypeError, ValueError):
                continue
            candidates.append((box, float(probability)))
        return candidates

    def __call__(self, frame):
        return self.process_clip([frame]).frames[0]

    def process_clip(self, frames) -> FaceClipResult:
        previous, selected = None, []
        area = lambda b: (b[2] - b[0]) * (b[3] - b[1])
        for frame in frames:
            candidates = self._candidates(frame)
            if not candidates:
                selected.append(FaceResult(None, False, "no_valid_detection", selected_bbox=previous))
                continue
            if previous is None:
                box, prob = max(candidates, key=lambda c: area(c[0]))
                reason = "largest_face"
            else:
                eligible = [c for c in candidates if area(c[0]) >= .35 * area(previous)]
                reason = "nearest_center" if eligible else "nearest_center_area_filter_relaxed"
                box, prob = min(eligible or candidates, key=lambda c:
                                (c[0][0] + c[0][2] - previous[0] - previous[2]) ** 2 +
                                (c[0][1] + c[0][3] - previous[1] - previous[3]) ** 2)
            selected.append(FaceResult(None, False, reason, True, False, box, prob))
            previous = box
        first = next((r.selected_bbox for r in selected if r.detection_valid), None)
        results = []
        for frame, result in zip(frames, selected):
            if first is None:
                results.append(replace(result, reason="no_valid_detection_in_clip"))
                continue
            box = result.selected_bbox or first
            crop_box = square_crop_box(box, frame.size)
            reason = result.reason if result.detection_valid else (
                "no_valid_detection_reuse_previous_bbox" if result.selected_bbox is not None else
                "no_valid_detection_reuse_first_future_bbox")
            results.append(replace(result, face=self._crop(frame, crop_box), face_valid=True,
                                   reason=reason, fallback_used=not result.detection_valid,
                                   selected_bbox=box, crop_bbox=crop_box))
        return FaceClipResult(tuple(results), first is not None, sum(r.detection_valid for r in results))


def build_face_processor(config: dict, *, allow_test_backend: bool = False,
                         remote_preprocessing: bool = False) -> FaceProcessor:
    """Training uses bbox cache only. MTCNN is opt-in remote preprocessing."""
    name = config.get("backend")
    if not name:
        raise ValueError("Configure face.backend as cached_bbox or mtcnn_cpu")
    if name == "mock_full_frame":
        processor = FaceProcessor(MockFullFrameBackend(), input_size=config.get("input_size", 224))
    else:
        validate_face_config(config)
        if config.get("cache_path") and not remote_preprocessing:
            return CachedBBoxFaceProcessor()
        if name == "cached_bbox":
            return CachedBBoxFaceProcessor()
        if not remote_preprocessing:
            raise ValueError("mtcnn_cpu is restricted to remote_preprocessing=True; training requires face cache")
        return MTCNNFaceProcessor()
    if processor.test_only and not allow_test_backend:
        raise ValueError("TEST/mock face backend is forbidden for real-data training/evaluation")
    return processor

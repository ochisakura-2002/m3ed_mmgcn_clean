"""Synthetic/mock CPU face pipeline tests. Never instantiate a real detector."""

from dataclasses import replace
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import uuid

from PIL import Image
import pytest
from torch.utils.data import DataLoader

from datasets.iemocap.label_vocab import load_label_vocab
from datasets.iemocap.s2d_dataset import IEMOCAPS2DDataset
from datasets.iemocap.s2d_faces import (
    CachedBBoxFaceProcessor, FaceBBoxCache, FaceCacheEntry, MTCNNFaceProcessor,
    PREPROCESSING_VERSION, build_face_processor, square_crop_box, validate_face_config,
)
from datasets.iemocap.s2d_manifest import S2DManifest
from datasets.iemocap.s2d_sampling import Uniform16Sampler
from models.s2d.iemocap.config import load_config


ROOT = Path(__file__).resolve().parents[3]
BOX = (20., 20., 60., 60.)
GOOD = ([BOX], [.99])
MISSING = (None, None)


@pytest.fixture
def tmp_path():
    # Same ordinary-directory workaround as existing S2D tests for managed Windows ACLs.
    base = (ROOT / "tmp/assistant_work/s2d_face_tests").resolve()
    path = base / uuid.uuid4().hex
    path.mkdir(parents=True)
    try:
        yield path
    finally:
        assert path.resolve().parent == base
        for child in path.iterdir():
            child.unlink()
        path.rmdir()


@pytest.fixture
def manifest(tmp_path):
    vocab = load_label_vocab(ROOT / "configs/_shared/data/iemocap/benchmark_label_vocab_v1.yaml")
    return S2DManifest(Path(__file__).parent / "fixtures/synthetic_manifest.csv", tmp_path, vocab)


class MockDetector:
    def __init__(self, detections):
        self.detections = iter(detections)

    def detect(self, frame, *, landmarks):
        assert landmarks is False
        return next(self.detections)


def frames():
    return [Image.new("RGB", (100, 80), (i + 20, 40, 60)) for i in range(16)]


def process(detections):
    return MTCNNFaceProcessor(detector=MockDetector(detections)).process_clip(frames())


def entry(uid="TEST_SYNTHETIC_face", detections=None):
    clip = process([GOOD] * 16 if detections is None else detections)
    return FaceCacheEntry.from_clip(uid, Uniform16Sampler()(49), frames(), clip)


def decoder(path, sampler):
    return frames(), sampler(49)


def test_first_valid_frame_selects_largest_face():
    clip = process([([(1, 1, 11, 11), BOX], [.999, .91])] + [GOOD] * 15)
    assert clip.frames[0].selected_bbox == BOX
    assert clip.frames[0].detection_probability == .91
    assert clip.frames[0].reason == "largest_face"


def test_tracking_filters_tiny_background_face_then_uses_nearest_center():
    target = (36., 20., 76., 60.)
    tiny_nearby = (30., 30., 34., 34.)
    large_distant = (70., 0., 100., 80.)
    clip = process([GOOD, ([tiny_nearby, large_distant, target], [.99] * 3)] + [GOOD] * 14)
    assert clip.frames[1].selected_bbox == target
    assert clip.frames[1].reason == "nearest_center"


def test_area_filter_relaxes_only_if_no_candidate_passes():
    small = (30., 30., 34., 34.)
    clip = process([GOOD, ([small], [.99])] + [GOOD] * 14)
    assert clip.frames[1].selected_bbox == small
    assert clip.frames[1].reason == "nearest_center_area_filter_relaxed"


def test_probability_threshold_is_inclusive_and_rejects_invalid_detections():
    clip = process([([BOX] * 5, [.89999, float("nan"), None, 1.1, .90])] + [GOOD] * 15)
    assert clip.frames[0].detection_valid
    assert clip.frames[0].detection_probability == .90
    clip = process([([BOX, (3, 3, 2, 2)], [.89999, .99])] + [GOOD] * 15)
    assert not clip.frames[0].detection_valid and clip.frames[0].fallback_used


def test_failed_frame_reuses_bbox_on_current_frame_and_audits_failure():
    clip = process([GOOD, MISSING] + [GOOD] * 14)
    result = clip.frames[1]
    assert clip.face_usable and clip.valid_detection_count == 15
    assert result.face_valid and not result.detection_valid and result.fallback_used
    assert result.selected_bbox == BOX and result.detection_probability is None
    assert result.reason == "no_valid_detection_reuse_previous_bbox"
    assert result.face.getpixel((0, 0)) == (21, 40, 60)  # Current raw frame, not old/black pixels.


def test_leading_failures_use_first_future_bbox_with_explicit_reason():
    clip = process([MISSING, MISSING] + [GOOD] * 14)
    assert clip.valid_detection_count == 14 and clip.face_usable
    assert all(r.fallback_used and r.selected_bbox == BOX for r in clip.frames[:2])
    assert clip.frames[0].reason == "no_valid_detection_reuse_first_future_bbox"
    entry(detections=[MISSING, MISSING] + [GOOD] * 14)  # Same rule validates in cache contract.


def test_all_16_failures_never_produce_fake_faces_or_full_frame_fallback():
    clip = process([MISSING] * 16)
    assert not clip.face_usable and clip.valid_detection_count == 0
    assert all(r.face is None and r.selected_bbox is None and not r.face_valid and
               not r.detection_valid and not r.fallback_used for r in clip.frames)
    assert not entry(detections=[MISSING] * 16).face_usable


def test_tracking_is_reset_between_utterances():
    detector = MockDetector([GOOD] * 16 + [MISSING] * 16)
    processor = MTCNNFaceProcessor(detector=detector)
    assert processor.process_clip(frames()).face_usable
    assert not processor.process_clip(frames()).face_usable


@pytest.mark.parametrize("box,size,expected", [
    ((50, 50, 90, 90), (100, 100), (42, 42, 98, 98)),
    ((0, 0, 40, 40), (100, 80), (0, 0, 56, 56)),
    ((60, 40, 100, 80), (100, 80), (44, 24, 100, 80)),
    ((10, 10, 50, 30), (100, 80), (2, 0, 58, 56)),
    ((0, 0, 100, 80), (100, 80), (10, 0, 90, 80)),
])
def test_margin_140_square_crop_stays_in_bounds(box, size, expected):
    crop = square_crop_box(box, size)
    assert crop == expected
    assert crop[2] - crop[0] == crop[3] - crop[1]
    assert 0 <= crop[0] < crop[2] <= size[0] and 0 <= crop[1] < crop[3] <= size[1]


def test_detector_coordinates_clipped_and_output_is_rgb_224():
    clip = process([([(-2, -3, 40, 40)], [.99])] + [GOOD] * 15)
    assert clip.frames[0].selected_bbox == (0., 0., 40., 40.)
    assert all(r.face.mode == "RGB" and r.face.size == (224, 224) for r in clip.frames)


def test_lazy_import_and_clear_remote_dependency_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "facenet_pytorch", None)
    assert isinstance(build_face_processor({"backend": "cached_bbox"}), CachedBBoxFaceProcessor)
    mock = build_face_processor({"backend": "mock_full_frame"}, allow_test_backend=True)
    assert mock(frames()[0]).face.size == (224, 224)
    processor = build_face_processor({"backend": "mtcnn_cpu"}, remote_preprocessing=True)
    with pytest.raises(ImportError, match="facenet-pytorch==2.5.3.*remote"):
        processor.process_clip(frames())


def test_real_backend_constructor_is_cpu_keep_all_min20_without_alignment(monkeypatch):
    kwargs_seen = []
    def factory(**kwargs):
        kwargs_seen.append(kwargs)
        return MockDetector([GOOD] * 16)
    monkeypatch.setitem(sys.modules, "facenet_pytorch", SimpleNamespace(MTCNN=factory))
    processor = build_face_processor({"backend": "mtcnn_cpu"}, remote_preprocessing=True)
    assert not kwargs_seen  # No import/constructor/weights until explicit preprocessing.
    assert processor.process_clip(frames()).valid_detection_count == 16
    assert kwargs_seen == [{"device": "cpu", "keep_all": True, "min_face_size": 20}]


def test_cache_roundtrip_never_overwrites(tmp_path):
    cached = entry(detections=[GOOD, MISSING] + [GOOD] * 14)
    cache = FaceBBoxCache({cached.utterance_id: cached})
    path = tmp_path / "TEST_SYNTHETIC_bbox.json"
    cache.save(path)
    assert FaceBBoxCache.load(path).entries == cache.entries
    with pytest.raises(FileExistsError):
        cache.save(path)


@pytest.mark.parametrize("change,match", [
    ({"frame_indices": (0,)}, "exactly 16"),
    ({"preprocessing_version": "old"}, "preprocessing_version"),
    ({"valid_detection_count": 0}, "count"),
    ({"face_usable": False}, "face_usable"),
    ({"detection_probabilities": (.89,) * 16}, "probability"),
    ({"fallback_used": (True,) * 16}, "fallback"),
    ({"selected_bboxes": ((20, 20, 101, 60),) * 16}, "bbox"),
])
def test_cache_contract_rejects_corrupt_entries(change, match):
    with pytest.raises(ValueError, match=match):
        replace(entry(), **change)


def test_cache_loader_rejects_duplicate_keys_and_wrong_envelope_version(tmp_path):
    path = tmp_path / "TEST_SYNTHETIC_bad_cache.json"
    path.write_text('{"entries":{},"entries":{}}', encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate"):
        FaceBBoxCache.load(path)
    path.write_text(json.dumps({"preprocessing_version": "old", "entries": {}}), encoding="utf-8")
    with pytest.raises(ValueError, match="preprocessing_version"):
        FaceBBoxCache.load(path)


def test_cache_entry_requires_explicit_version_and_consistent_fallback_bbox():
    value = entry().as_dict()
    value.pop("preprocessing_version")
    with pytest.raises(ValueError, match="preprocessing_version"):
        FaceCacheEntry.from_dict(value)
    cached = entry(detections=[GOOD, MISSING] + [GOOD] * 14)
    with pytest.raises(ValueError, match="Fallback bbox"):
        replace(cached, selected_bboxes=(BOX, (1., 1., 11., 11.)) + (BOX,) * 14)


def test_cached_dataset_bypasses_mtcnn_and_collates_audit_fields(manifest, tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "facenet_pytorch", None)
    uid = "SYNTHETIC_test_001"
    path = tmp_path / "TEST_SYNTHETIC_cached.json"
    FaceBBoxCache({uid: entry(uid, [GOOD, MISSING] + [GOOD] * 14)}).save(path)
    # A direct MTCNN instance has no detector/dependency. Supplying cache must bypass it.
    dataset = IEMOCAPS2DDataset(manifest, "test", MTCNNFaceProcessor(), decoder=decoder,
                               face_cache=path, require_face_cache_for_training=True)
    batch = next(iter(DataLoader(dataset, batch_size=1, num_workers=0)))
    assert batch["video"].shape == (1, 3, 16, 224, 224)
    assert batch["detection_valid"].sum() == 15 and batch["fallback_used"].sum() == 1
    assert batch["detection_probability"][0, 1] == -1
    assert batch["preprocessing_version"] == [PREPROCESSING_VERSION]
    assert isinstance(build_face_processor({"backend": "mtcnn_cpu", "cache_path": str(path)}),
                      CachedBBoxFaceProcessor)


def test_cache_filters_face_unusable_without_deleting_global_manifest(manifest):
    uid = "SYNTHETIC_train_001"
    cache = FaceBBoxCache({uid: entry(uid, [MISSING] * 16)})
    dataset = IEMOCAPS2DDataset(manifest, "train", CachedBBoxFaceProcessor(), face_cache=cache)
    assert len(dataset) == 0 and dataset.excluded_face_utterance_ids == (uid,)
    assert len(manifest.records) == 4 and manifest.by_id[uid].v_available


def test_remote_preprocessing_keeps_all_failed_cache_record(manifest):
    processor = MTCNNFaceProcessor(detector=MockDetector([MISSING] * 16))
    dataset = IEMOCAPS2DDataset(manifest, "train", processor, decoder=decoder, remote_preprocessing=True)
    cached = FaceCacheEntry.from_dict(dataset[0])
    assert not cached.face_usable and cached.valid_detection_count == 0
    assert len(dataset) == 1 and len(manifest.records) == 4


def test_missing_cache_and_online_training_are_rejected(manifest):
    with pytest.raises(ValueError, match="remote_preprocessing"):
        build_face_processor({"backend": "mtcnn_cpu"})
    with pytest.raises(ValueError, match="Online MTCNN"):
        IEMOCAPS2DDataset(manifest, "train", MTCNNFaceProcessor())
    with pytest.raises(ValueError, match="requires a face bbox cache"):
        IEMOCAPS2DDataset(manifest, "train", CachedBBoxFaceProcessor())
    with pytest.raises(ValueError, match="Missing face cache entries"):
        IEMOCAPS2DDataset(manifest, "train", CachedBBoxFaceProcessor(), face_cache=FaceBBoxCache({}))


@pytest.mark.parametrize("indices,sizes", [(Uniform16Sampler()(50), frames()),
                                          (Uniform16Sampler()(49), [Image.new("RGB", (101, 80))] * 16)])
def test_cache_decoder_indices_and_size_must_match(indices, sizes):
    with pytest.raises(ValueError, match="indices/sizes"):
        CachedBBoxFaceProcessor().process_cached_clip(sizes, indices, entry())


@pytest.mark.parametrize("key,value", [("device", "cuda"), ("min_size", 21),
                                       ("prob_threshold", .8), ("margin", 1.0),
                                       ("alignment", True), ("full_frame_fallback", True),
                                       ("backend", "mock_full_frame")])
def test_config_freezes_v0_face_settings(key, value):
    config, _ = load_config(ROOT / "configs/s2d/iemocap/v0.yaml")
    config["face"][key] = value
    with pytest.raises(ValueError, match="S2D V0"):
        validate_face_config(config["face"])


def test_uniform16_matches_remote_pilot():
    assert Uniform16Sampler()(49) == (0, 3, 6, 9, 12, 16, 19, 22, 25, 28, 32, 35, 38, 41, 44, 48)

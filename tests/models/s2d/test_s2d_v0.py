"""SYNTHETIC ONLY. No IEMOCAP data, real detector, S2D weights or training."""

import csv
import hashlib
import json
from pathlib import Path
import uuid

import pytest
import torch
from torch import nn
from PIL import Image

from datasets.iemocap.label_vocab import load_label_vocab
from datasets.iemocap.s2d_dataset import IEMOCAPS2DDataset, S2DTransform
from datasets.iemocap.s2d_faces import FaceProcessor, FaceResult, MockFullFrameBackend, build_face_processor
from datasets.iemocap.s2d_manifest import S2DManifest
from datasets.iemocap.s2d_sampling import Uniform16Sampler
from models.s2d.iemocap.config import config_summary, load_config
from models.s2d.iemocap.model import S2DIEMOCAP, load_weights
from scripts.models.s2d.iemocap.train import classification_report, export_predictions


ROOT = Path(__file__).resolve().parents[3]
FIXTURE = Path(__file__).parent / "fixtures" / "synthetic_manifest.csv"
VOCAB = ROOT / "configs/_shared/data/iemocap/benchmark_label_vocab_v1.yaml"


@pytest.fixture
def tmp_path():
    # pytest's mode=0700 temp directories are inaccessible under this Windows
    # managed ACL. Use a fresh ordinary directory inside the authorized workspace.
    base = (ROOT / "tmp/assistant_work/s2d_tests").resolve()
    path = base / uuid.uuid4().hex
    path.mkdir(parents=True)
    try:
        yield path
    finally:
        # All files here are tiny synthetic fixtures owned by this test.
        assert path.resolve().parent == base
        for child in path.iterdir():
            child.unlink()
        path.rmdir()


@pytest.fixture
def vocab():
    return load_label_vocab(VOCAB)


def test_vocab_matches_existing_ta_contract(vocab):
    from datasets.iemocap.official_feature_dataset import IEMOCAPOfficialFeatureDataset

    # Explicit aliases of the tracked Candidate A/C loader names, in its order.
    aliases = {"Happy": "hap", "Sad": "sad", "Neutral": "neu", "Angry": "ang",
               "Excited": "exc", "Frustrated": "fru"}
    existing = IEMOCAPOfficialFeatureDataset.DEFAULT_LABEL_LIST
    assert vocab.index_to_label == tuple(aliases[label] for label in existing)
    assert vocab.num_classes == 6 and vocab.ta_mapping_verified
    assert vocab.encode("hap") == 0 and vocab.encode("ang") == 3
    with pytest.raises(ValueError, match="Unknown benchmark_label"):
        vocab.encode("canonical_emotion")


@pytest.mark.parametrize("n", [1, 5, 16, 31, 100])
def test_uniform16(n):
    sampler = Uniform16Sampler()
    indices = sampler(n)
    assert indices == sampler(n)
    assert len(indices) == 16 and indices[0] == 0 and indices[-1] == n - 1
    assert list(indices) == sorted(indices) and all(0 <= i < n for i in indices)
    if n == 16:
        assert indices == tuple(range(16))
    if n == 5:
        assert len(set(indices)) == 5


@pytest.mark.parametrize("n", [0, -1, 2.5, True])
def test_uniform16_rejects_empty_or_invalid_clip(n):
    with pytest.raises(ValueError):
        Uniform16Sampler()(n)


def test_parser_preserves_universe_splits_and_target(vocab, tmp_path):
    manifest = S2DManifest(FIXTURE, tmp_path, vocab)
    assert len(manifest.records) == 4
    assert manifest.counts()["test"] == {"universe": 2, "v_available": 1, "v_unavailable": 1}
    missing = manifest.by_id["SYNTHETIC_test_no_video"]
    assert not missing.v_available and missing.video_path is None
    assert manifest.for_split("train")[0].video_path == tmp_path / "TEST_SYNTHETIC/train.mp4"
    # Conflicting canonical_emotion is deliberately ignored.
    assert manifest.for_split("train")[0].benchmark_label == "hap"
    assert manifest.for_split("train")[0].label_index == 0
    assert len(manifest.for_split("test", available_only=False)) == 2


def changed_fixture(tmp_path, change):
    with FIXTURE.open(newline="") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
        headers = list(reader.fieldnames)
    change(rows)
    path = tmp_path / "TEST_SYNTHETIC_changed.csv"
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, headers)
        writer.writeheader()
        writer.writerows(rows)
    return path


@pytest.mark.parametrize("field,value,match", [
    ("split", "val", "train/dev/test"),
    ("label_index", "3", "label_index disagrees"),
    ("v_available", "unknown", "invalid v_available"),
    ("video_path", "", "requires video_path"),
    ("utterance_id", "SYNTHETIC_dev_001", "duplicate utterance_id"),
])
def test_bad_manifest_fails(vocab, tmp_path, field, value, match):
    path = changed_fixture(tmp_path, lambda rows: rows[0].update({field: value}))
    with pytest.raises(ValueError, match=match):
        S2DManifest(path, tmp_path, vocab)


def test_explicit_column_alias_and_derived_label(vocab, tmp_path):
    path = tmp_path / "TEST_SYNTHETIC_alias.csv"
    path.write_text("utterance_id,video_clip_path,benchmark_label,split,v_available\n"
                    "TEST_SYNTHETIC_alias,clips/one.mp4,fru,dev,1\n", encoding="utf-8")
    manifest = S2DManifest(path, tmp_path, vocab, {"video_path": "video_clip_path"})
    assert manifest.records[0].label_index == 5
    assert manifest.records[0].video_path == tmp_path / "clips/one.mp4"


def synthetic_decoder(path, sampler):
    return [Image.new("RGB", (32, 24), (255, 0, 0)) for _ in range(16)], sampler(5)


def test_synthetic_dataset_face_transform_chain(vocab, tmp_path):
    manifest = S2DManifest(FIXTURE, tmp_path, vocab)
    processor = FaceProcessor(MockFullFrameBackend())
    dataset = IEMOCAPS2DDataset(manifest, "test", processor,
                               decoder=synthetic_decoder, allow_test_backend=True)
    assert len(dataset) == 1
    item = dataset[0]
    assert item["utterance_id"] == "SYNTHETIC_test_001" and item["v_valid"]
    assert item["video"].shape == (3, 16, 224, 224)
    assert item["video"].dtype == torch.float32
    assert item["video"][0, 0, 0, 0].item() == pytest.approx((1 - .485) / .229)
    assert item["video"][1, 0, 0, 0].item() == pytest.approx(-.456 / .224)
    assert tuple(item["frame_indices"].tolist()) == Uniform16Sampler()(5)


def test_mock_forbidden_in_production(vocab, tmp_path):
    with pytest.raises(ValueError, match="mock face backend"):
        build_face_processor({"backend": "mock_full_frame"})
    with pytest.raises(ValueError, match="test-only"):
        IEMOCAPS2DDataset(S2DManifest(FIXTURE, tmp_path, vocab), "train",
                         FaceProcessor(MockFullFrameBackend()))


def test_failed_face_never_silently_replaces_sample(vocab, tmp_path):
    def failed_backend(frame):
        return FaceResult(None, False, "TEST_NO_FACE")
    dataset = IEMOCAPS2DDataset(S2DManifest(FIXTURE, tmp_path, vocab), "train",
                               FaceProcessor(failed_backend), decoder=synthetic_decoder)
    with pytest.warns(RuntimeWarning, match="TEST_NO_FACE"):
        with pytest.raises(RuntimeError, match="SYNTHETIC_train_001.*face_valid=False"):
            dataset[0]


class SyntheticLandmark(nn.Module):
    def __init__(self):
        super().__init__()
        self.bn = nn.BatchNorm1d(1)

    def forward(self, video):
        return video.new_zeros((video.shape[0] * 16, 128, 14, 14))


class SyntheticCore(nn.Module):
    """TEST contract double; not an S2D implementation or evidence of S2D forward."""
    def __init__(self, k):
        super().__init__()
        self.patch_embed = nn.Linear(1, 1)
        self.prompt_blocks = nn.Linear(1, 8)
        self.head = nn.Module()
        self.head.fc_cls = nn.Linear(8, k)

    def forward(self, video, features):
        h = self.prompt_blocks(video.new_ones((len(video), 1)))
        return self.head.fc_cls(h), h


def test_wrapper_contract_and_freezing_with_test_doubles(vocab):
    model = S2DIEMOCAP(SyntheticCore(6), SyntheticLandmark(), vocab)
    model.train()
    assert not model.landmark.training
    assert not any(p.requires_grad for p in model.landmark.parameters())
    assert not model.core.patch_embed.weight.requires_grad
    assert model.core.prompt_blocks.weight.requires_grad and model.core.head.fc_cls.weight.requires_grad
    # A small interface smoke only; no real S2D core and no backward.
    result = model(torch.zeros(1, 3, 16, 224, 224))
    assert set(result) == {"h_v", "logits_v", "prob_v"}
    assert result["h_v"].shape == (1, 8) and result["logits_v"].shape == (1, 6)
    assert torch.allclose(result["prob_v"].sum(-1), torch.ones(1))


def test_checkpoint_modes(vocab, tmp_path):
    core = SyntheticCore(6)
    path = tmp_path / "TEST_SYNTHETIC_checkpoint.pt"
    torch.save({"model": core.state_dict()}, path)
    report = load_weights(core, path, strict=True)
    assert not report["missing_keys"] and not report["unexpected_keys"]
    report = load_weights(core, path, strict=False, drop_classifier=True)
    assert set(report["missing_keys"]) == {"head.fc_cls.weight", "head.fc_cls.bias"}
    with pytest.raises(ValueError, match="strict=False"):
        load_weights(core, path, strict=True, drop_classifier=True)
    wrong = core.state_dict()
    wrong["prompt_blocks.weight"] = torch.zeros(3, 3)
    torch.save({"model": wrong}, path)
    with pytest.raises(ValueError, match="shape mismatches"):
        load_weights(core, path, strict=False)


def test_config_is_data_free():
    config, vocab = load_config(ROOT / "configs/s2d/iemocap/v0.yaml")
    summary = config_summary(config, vocab)
    assert summary["num_classes"] == 6 and not summary["data_or_weights_opened"]
    assert set(summary["REMOTE_REQUIRED_FIELDS"]) == {
        "data_root", "manifest_path", "face.cache_path", "pretrained_checkpoint", "landmark_checkpoint"}
    assert summary["REMOTE_GPU_SMOKE_REQUIRED"] == "YES"


def test_metrics_and_identity_export(vocab, tmp_path):
    report = classification_report([0, 1, 1], [0, 0, 1], vocab)
    assert report["acc"] == pytest.approx(2 / 3)
    assert len(report["per_class"]) == 6 and len(report["confusion_matrix"]) == 6
    batches = [{"utterance_id": ["TEST_SYNTHETIC_prediction"], "label_index": torch.tensor([0]),
                "h_v": torch.zeros(1, 8), "logits_v": torch.ones(1, 6),
                "prob_v": torch.ones(1, 6) / 6}]
    path = tmp_path / "TEST_SYNTHETIC_prediction.csv"
    export_predictions(batches, path, vocab)
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert rows[0]["utterance_id"] == "TEST_SYNTHETIC_prediction"
    assert rows[0]["benchmark_label"] == "hap" and "prob_fru" in rows[0]


def test_upstream_snapshot_is_source_only_and_unchanged():
    root = ROOT / "third_party/S2D"
    metadata = json.loads((root / "UPSTREAM.json").read_text())
    assert metadata["commit"] == "e1b5e5dfb57ab15f70103c933b346bfa62943159"
    assert metadata["license"] == "Apache-2.0"
    assert not (root / ".git").exists()
    for name, digest in metadata["files_sha256"].items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == digest
    assert not list(root.rglob("*.pth*")) and not list(root.rglob("*.mp4"))

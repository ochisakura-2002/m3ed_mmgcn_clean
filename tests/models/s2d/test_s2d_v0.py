"""SYNTHETIC ONLY. No IEMOCAP data, real detector, S2D weights or training."""

import csv
import hashlib
import json
from pathlib import Path
import pickle
import shutil
from types import SimpleNamespace
from unittest.mock import Mock
import uuid

import pytest
import torch
from torch import nn
from PIL import Image
import yaml

from datasets.iemocap.label_vocab import load_label_vocab
from datasets.iemocap.s2d_dataset import IEMOCAPS2DDataset, S2DTransform
from datasets.iemocap.s2d_faces import FaceProcessor, FaceResult, MockFullFrameBackend, build_face_processor
from datasets.iemocap.s2d_manifest import S2DManifest
from datasets.iemocap.s2d_sampling import Uniform16Sampler
from models.s2d.iemocap.config import RUNTIME_PATH_FIELDS, config_summary, load_config, validate_runtime_paths
from models.s2d.iemocap.model import S2DIEMOCAP, build_s2d, load_weights
from scripts.models.s2d.iemocap.train import classification_report, export_predictions
import scripts.models.s2d.iemocap.train as training_entrypoint


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
        shutil.rmtree(path)


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


@pytest.mark.parametrize("container", [None, "model_state_dict", "model", "state_dict", "module"])
def test_checkpoint_defaults_to_weights_only(tmp_path, monkeypatch, container):
    module = nn.Linear(2, 2)
    state = module.state_dict()
    path = tmp_path / "TEST_SYNTHETIC_safe.pt"
    torch.save(state if container is None else {container: state}, path)
    loader = Mock(wraps=torch.load)
    monkeypatch.setattr(torch, "load", loader)
    report = load_weights(module, path, strict=True)
    loader.assert_called_once()
    assert loader.call_args.kwargs == {"map_location": "cpu", "weights_only": True}
    assert report["weights_only"] and not report["trusted_legacy_pickle"]
    assert not report["sha256_verified"] and report["path"] == str(path)
    assert report["state_source"] == (container or "raw_state_dict")


@pytest.mark.parametrize("container", [None, "model_state_dict"])
def test_s2d_checkpoint_adaptation_preserved(tmp_path, container):
    core = SyntheticCore(6)
    core.ln_post = nn.LayerNorm(8)
    state = {"module." + k.replace("ln_post.", "norm."): v.clone()
             for k, v in core.state_dict().items()}
    state["module.norm.weight"] = torch.full((8,), 2.)
    state["module.head.fc_cls.weight"] = torch.zeros(7, 8)
    state["module.head.fc_cls.bias"] = torch.zeros(7)
    state["module.landmark.TEST_SYNTHETIC.weight"] = torch.zeros(1)
    path = tmp_path / "TEST_SYNTHETIC_s2d_adaptation.pt"
    torch.save(state if container is None else {container: state}, path)
    report = load_weights(core, path, strict=False, drop_classifier=True, s2d_pretrained=True,
                          expected_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    assert report["weights_only"] and report["sha256_verified"]
    assert not report["trusted_legacy_pickle"]
    assert report["state_source"] == (container or "raw_state_dict")
    assert set(report["dropped_keys"]) == {
        "landmark.TEST_SYNTHETIC.weight", "head.fc_cls.weight", "head.fc_cls.bias"}
    assert set(report["missing_keys"]) == {"head.fc_cls.weight", "head.fc_cls.bias"}
    assert not report["unexpected_keys"]
    assert torch.equal(core.ln_post.weight, torch.full((8,), 2.))
    assert core.head.fc_cls.out_features == 6


def test_untrusted_legacy_never_falls_back(tmp_path, monkeypatch):
    path = tmp_path / "TEST_SYNTHETIC_legacy.pth.tar"
    path.write_bytes(b"TEST_SYNTHETIC_mock_legacy_container")
    loader = Mock(side_effect=pickle.UnpicklingError("Weights only load failed"))
    monkeypatch.setattr(torch, "load", loader)
    with pytest.raises(pickle.UnpicklingError, match="Weights only load failed"):
        load_weights(nn.Linear(2, 2), path, strict=True, trusted_legacy_pickle=False,
                     expected_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    loader.assert_called_once()
    assert loader.call_args.kwargs["weights_only"] is True


@pytest.mark.parametrize("digest", [None, ""])
def test_trusted_legacy_requires_hash_before_load(tmp_path, monkeypatch, digest):
    loader = Mock()
    monkeypatch.setattr(torch, "load", loader)
    with pytest.raises(ValueError, match="requires a nonempty expected_sha256"):
        load_weights(nn.Linear(2, 2), tmp_path / "TEST_SYNTHETIC_missing.pt", strict=True,
                     trusted_legacy_pickle=True, expected_sha256=digest)
    loader.assert_not_called()


@pytest.mark.parametrize("trusted", [False, True])
def test_checkpoint_wrong_hash_blocks_load(tmp_path, monkeypatch, trusted):
    path = tmp_path / "TEST_SYNTHETIC_wrong_hash.pt"
    path.write_bytes(b"TEST_SYNTHETIC_mock_checkpoint")
    loader = Mock()
    monkeypatch.setattr(torch, "load", loader)
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        load_weights(nn.Linear(2, 2), path, strict=True, expected_sha256="0" * 64,
                     trusted_legacy_pickle=trusted)
    loader.assert_not_called()


def test_trusted_legacy_hash_verified_before_full_pickle(tmp_path, monkeypatch):
    module = nn.Linear(2, 2)
    path = tmp_path / "TEST_SYNTHETIC_legacy.pth.tar"
    content = b"TEST_SYNTHETIC_mock_legacy_container"
    path.write_bytes(content)
    state = {"module." + k: v.clone() for k, v in module.state_dict().items()}

    def mock_load(stream, *, map_location, weights_only):
        assert map_location == "cpu" and weights_only is False
        assert stream.tell() == 0 and stream.read() == content
        return {"epoch": 1, "state_dict": state, "acc": 0., "best_acc": 0., "optimizer": {}}

    loader = Mock(side_effect=mock_load)
    monkeypatch.setattr(torch, "load", loader)
    report = load_weights(module, path, strict=True, trusted_legacy_pickle=True,
                          expected_sha256=hashlib.sha256(content).hexdigest())
    loader.assert_called_once()
    assert not report["weights_only"] and report["sha256_verified"]
    assert report["trusted_legacy_pickle"] and report["state_source"] == "state_dict"
    assert report["strict"] and report["loaded_key_count"] == 2
    assert not report["missing_keys"] and not report["unexpected_keys"]


@pytest.mark.parametrize("trusted", [False, True])
def test_build_s2d_load_modes(vocab, tmp_path, monkeypatch, trusted):
    core, mobile = SyntheticCore(6), nn.Linear(2, 2)
    source = SimpleNamespace(s2d_base_patch16_224=lambda **kwargs: core)
    landmark_source = SimpleNamespace(MobileFaceNet=lambda *args: mobile)
    monkeypatch.setattr("models.s2d.iemocap.model._upstream_module",
                        lambda name: source if name == "model_finetuning.py" else landmark_source)
    s2d_path = tmp_path / "TEST_SYNTHETIC_s2d.pt"
    landmark_path = tmp_path / "TEST_SYNTHETIC_mobile.pth.tar"
    torch.save({"model_state_dict": core.state_dict()}, s2d_path)
    torch.save({"epoch": 1, "state_dict": mobile.state_dict(), "optimizer": {}}, landmark_path)
    loader = Mock(wraps=torch.load)
    monkeypatch.setattr(torch, "load", loader)
    model = build_s2d(vocab, {
        "pretrained_checkpoint": s2d_path,
        "pretrained_checkpoint_sha256": hashlib.sha256(s2d_path.read_bytes()).hexdigest(),
        "landmark_checkpoint": landmark_path,
        "landmark_checkpoint_sha256": hashlib.sha256(landmark_path.read_bytes()).hexdigest(),
        "landmark_trusted_legacy_pickle": trusted,
    })
    assert [call.kwargs["weights_only"] for call in loader.call_args_list] == [True, not trusted]
    s2d, landmark = model.load_reports["s2d"], model.load_reports["landmark"]
    assert s2d["weights_only"] and s2d["sha256_verified"] and not s2d["trusted_legacy_pickle"]
    assert s2d["state_source"] == "model_state_dict"
    assert landmark["strict"] and landmark["sha256_verified"]
    assert landmark["weights_only"] is not trusted and landmark["trusted_legacy_pickle"] is trusted
    assert landmark["state_source"] == "state_dict"


@pytest.mark.parametrize("mismatch,exception,match", [
    ("missing", RuntimeError, "Missing key"),
    ("unexpected", RuntimeError, "Unexpected key"),
    ("shape", ValueError, "shape mismatches"),
])
def test_legacy_landmark_strict_mismatch_fails(tmp_path, mismatch, exception, match):
    module = nn.Linear(2, 2)
    state = module.state_dict()
    if mismatch == "missing":
        del state["bias"]
    elif mismatch == "unexpected":
        state["TEST_SYNTHETIC_extra"] = torch.zeros(1)
    else:
        state["weight"] = torch.zeros(3, 2)
    path = tmp_path / "TEST_SYNTHETIC_bad_mobile.pth.tar"
    torch.save({"state_dict": state}, path)
    with pytest.raises(exception, match=match):
        load_weights(module, path, strict=True, trusted_legacy_pickle=True,
                     expected_sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def test_s2d_rejects_legacy_full_pickle(tmp_path, monkeypatch):
    loader = Mock()
    monkeypatch.setattr(torch, "load", loader)
    with pytest.raises(ValueError, match="S2D pretrained checkpoints require weights_only=True"):
        load_weights(SyntheticCore(6), tmp_path / "TEST_SYNTHETIC_missing.pt", strict=False,
                     s2d_pretrained=True, trusted_legacy_pickle=True, expected_sha256="0" * 64)
    loader.assert_not_called()


@pytest.mark.parametrize("key,value,match", [
    ("landmark_trusted_legacy_pickle", "false", "must be a boolean"),
    ("landmark_checkpoint_sha256", None, "requires landmark_checkpoint_sha256"),
    ("landmark_checkpoint_sha256", "", "lowercase SHA256"),
    ("pretrained_checkpoint_sha256", "bad", "lowercase SHA256"),
    ("landmark_checkpoint_sha256", "g" * 64, "lowercase SHA256"),
    ("landmark_checkpoint_sha256", 123, "lowercase SHA256"),
    ("landmark_strict", False, "landmark_strict=True"),
])
def test_config_checkpoint_safety(tmp_path, key, value, match):
    config, _ = load_config(ROOT / "configs/s2d/iemocap/v0.yaml")
    config["model"][key] = value
    path = tmp_path / "TEST_SYNTHETIC_config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(ValueError, match=match):
        load_config(path)


def test_build_s2d_rejects_non_strict_landmark(vocab, monkeypatch):
    source = Mock()
    monkeypatch.setattr("models.s2d.iemocap.model._upstream_module", source)
    with pytest.raises(ValueError, match="landmark_strict=True"):
        build_s2d(vocab, {"landmark_strict": False})
    source.assert_not_called()


def test_config_is_data_free():
    config, vocab = load_config(ROOT / "configs/s2d/iemocap/v0.yaml")
    summary = config_summary(config, vocab)
    assert summary["num_classes"] == 6 and not summary["data_or_weights_opened"]
    assert set(summary["REMOTE_REQUIRED_FIELDS"]) == {
        "data_root", "manifest_path", "face.cache_path", "pretrained_checkpoint", "landmark_checkpoint"}
    assert summary["REMOTE_GPU_SMOKE_REQUIRED"] == "NO"


@pytest.mark.parametrize("name", list(RUNTIME_PATH_FIELDS))
def test_cli_runtime_path_overrides_yaml(tmp_path, name):
    original, _ = load_config(ROOT / "configs/s2d/iemocap/v0.yaml")
    for key, (section, field) in RUNTIME_PATH_FIELDS.items():
        original[section][field] = f"TEST_SYNTHETIC_yaml_{key}"
    path = tmp_path / "TEST_SYNTHETIC_config.yaml"
    path.write_text(yaml.safe_dump(original), encoding="utf-8")
    args = training_entrypoint.parse_args(["--config", str(path),
                                          f"--{name.replace('_', '-')}", "TEST_SYNTHETIC_cli"])
    config, _ = load_config(args.config, overrides=vars(args))
    section, field = RUNTIME_PATH_FIELDS[name]
    original[section][field] = "TEST_SYNTHETIC_cli"
    assert config == original  # All research settings and other YAML paths survive.
    default_args = training_entrypoint.parse_args(["--config", str(path)])
    unchanged, _ = load_config(default_args.config, overrides=vars(default_args))
    assert unchanged == yaml.safe_load(path.read_text(encoding="utf-8"))


def test_explicit_empty_cli_path_does_not_fall_back_to_yaml(tmp_path):
    config, _ = load_config(ROOT / "configs/s2d/iemocap/v0.yaml", overrides={"data_root": ""})
    config["dataset"]["data_root"] = "TEST_SYNTHETIC_yaml_data"
    path = tmp_path / "TEST_SYNTHETIC_config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    args = training_entrypoint.parse_args(["--data-root", ""])
    config, _ = load_config(path, overrides=vars(args))
    assert config["dataset"]["data_root"] == ""
    with pytest.raises(ValueError, match="dataset.data_root is empty"):
        validate_runtime_paths(config)


def test_check_config_with_nonexistent_cli_inputs_is_read_only(tmp_path, monkeypatch, capsys):
    validate = Mock(side_effect=AssertionError("Static check must not validate files"))
    trainer = Mock(side_effect=AssertionError("Static check must not train"))
    monkeypatch.setattr(training_entrypoint, "validate_runtime_paths", validate)
    monkeypatch.setattr(training_entrypoint, "train", trainer)
    argv = ["--check-config"]
    expected = {}
    for name in RUNTIME_PATH_FIELDS:
        expected[name] = str(tmp_path / f"TEST_SYNTHETIC_missing_{name}")
        argv.extend([f"--{name.replace('_', '-')}", expected[name]])
    training_entrypoint.main(argv)
    summary = json.loads(capsys.readouterr().out)
    assert summary["runtime_paths"] == expected
    assert summary["status"] == "CONFIG_VALID_ONLY"
    assert not summary["data_or_weights_opened"] and not summary["REMOTE_REQUIRED_FIELDS"]
    assert not list(tmp_path.iterdir())
    validate.assert_not_called()
    trainer.assert_not_called()


@pytest.fixture
def runtime_config(tmp_path):
    """Existence-only input files; never decoded or loaded as real checkpoints."""
    config, _ = load_config(ROOT / "configs/s2d/iemocap/v0.yaml")
    root = tmp_path / "TEST_SYNTHETIC_data"
    root.mkdir()
    manifest = root / "TEST_SYNTHETIC_manifest.csv"
    manifest.touch()
    config["dataset"].update(data_root=str(root.relative_to(ROOT)), manifest_path=manifest.name)
    for name in ("face_cache", "pretrained_checkpoint", "landmark_checkpoint"):
        path = tmp_path / f"TEST_SYNTHETIC_{name}"
        path.touch()
        section, field = RUNTIME_PATH_FIELDS[name]
        config[section][field] = str(path.relative_to(ROOT))
    config["system"]["output_dir"] = str(tmp_path / "outputs/s2d/iemocap/v0")
    return config


@pytest.mark.parametrize("name", [key for key in RUNTIME_PATH_FIELDS if key != "output_dir"])
@pytest.mark.parametrize("failure", ["empty", "missing", "wrong_kind"])
def test_formal_runtime_paths_fail_before_cuda_or_epochs(runtime_config, vocab, tmp_path,
                                                        monkeypatch, name, failure):
    section, field = RUNTIME_PATH_FIELDS[name]
    path = tmp_path / "TEST_SYNTHETIC_invalid"
    if failure == "wrong_kind":
        path.touch() if name == "data_root" else path.mkdir()
    runtime_config[section][field] = " " if failure == "empty" else str(path)
    epoch = Mock(side_effect=AssertionError("Invalid paths reached epoch loop"))
    cuda = Mock(side_effect=AssertionError("Invalid paths reached CUDA setup"))
    monkeypatch.setattr(training_entrypoint, "run_epoch", epoch)
    monkeypatch.setattr(torch.cuda, "is_available", cuda)
    error = ValueError if failure == "empty" else FileNotFoundError
    with pytest.raises(error, match=f"{section}.{field}"):
        training_entrypoint.train(runtime_config, vocab)
    epoch.assert_not_called()
    cuda.assert_not_called()
    assert not Path(runtime_config["system"]["output_dir"]).exists()


def test_runtime_resolves_paths_without_opening_inputs(runtime_config, tmp_path):
    validate_runtime_paths(runtime_config)
    assert Path(runtime_config["dataset"]["manifest_path"]) == (
        Path(runtime_config["dataset"]["data_root"]) / "TEST_SYNTHETIC_manifest.csv")
    for name, (section, field) in RUNTIME_PATH_FIELDS.items():
        assert Path(runtime_config[section][field]).is_absolute()
    assert not Path(runtime_config["system"]["output_dir"]).exists()
    runtime_config["system"]["output_dir"] = ""
    with pytest.raises(ValueError, match="system.output_dir is empty"):
        validate_runtime_paths(runtime_config)


@pytest.mark.parametrize("export", [False, True])
def test_s2d_layout_resolved_overrides_and_provenance(runtime_config, vocab, tmp_path, export):
    overrides = {name: runtime_config[section][field]
                 for name, (section, field) in RUNTIME_PATH_FIELDS.items()}
    argv = ["--experiment-date", "20261104", "--experiment-group", "TEST_SYNTHETIC_group",
            "--run-id", "TEST_SYNTHETIC_seed100"]
    for name, value in overrides.items():
        argv.extend([f"--{name.replace('_', '-')}", value])
    args = training_entrypoint.parse_args(argv)
    config, _ = load_config(ROOT / "configs/s2d/iemocap/v0.yaml", overrides=vars(args))
    config["export"]["representations"] = export
    validate_runtime_paths(config)
    run = training_entrypoint.prepare_s2d_run_environment(config)
    assert run["run_dir"] == tmp_path / "outputs/s2d/iemocap/v0/20261104/TEST_SYNTHETIC_seed100"
    directories = {"checkpoints", "logs", "predictions", "metrics"}
    assert {p.name for p in run["run_dir"].iterdir() if p.is_dir()} == (
        directories | ({"representations"} if export else set()))
    saved = yaml.safe_load((run["run_dir"] / "resolved_config.yaml").read_text(encoding="utf-8"))
    assert saved == config
    for name, (section, field) in RUNTIME_PATH_FIELDS.items():
        assert saved[section][field] == config[section][field]
        assert Path(saved[section][field]).is_absolute()
    assert saved["output"]["run_id"] == "TEST_SYNTHETIC_seed100"
    assert saved["output"]["experiment_group"] == "TEST_SYNTHETIC_group"
    reports = {name: {"path": config["model"][field], "sha256_verified": True,
                      "state_source": "TEST_SYNTHETIC_mock_report"}
               for name, field in (("s2d", "pretrained_checkpoint"), ("landmark", "landmark_checkpoint"))}
    training_entrypoint.write_s2d_run_provenance(config, vocab, run, reports)
    metadata = json.loads((run["run_dir"] / "run_metadata.json").read_text(encoding="utf-8"))
    assert len(metadata["git_commit"]) == 40 and "git_branch" in metadata
    assert metadata["upstream_s2d_commit"] == "e1b5e5dfb57ab15f70103c933b346bfa62943159"
    assert metadata["seed"] == 100 and metadata["label_vocab"] == vocab.as_dict()
    assert metadata["run_id"] == saved["output"]["run_id"]
    assert metadata["manifest_path"] == saved["dataset"]["manifest_path"]
    assert metadata["face_cache_path"] == saved["face"]["cache_path"]
    assert metadata["checkpoint_selection_metric"] == "val_weighted_f1"
    assert metadata["test_split_used_for_selection"] is False
    for name, field in (("s2d", "pretrained_checkpoint"), ("landmark", "landmark_checkpoint")):
        assert metadata["checkpoints"][name] == {
            "path": saved["model"][field], "expected_sha256": saved["model"][f"{field}_sha256"],
            "sha256_verified": True}
    assert json.loads((run["logs_dir"] / "pretrained_load_report.json").read_text()) == reports


def test_fixed_run_collision_preserves_existing_artifacts(runtime_config):
    runtime_config["output"].update(experiment_date="20261104", run_id="TEST_SYNTHETIC_fixed")
    run = training_entrypoint.prepare_s2d_run_environment(runtime_config)
    sentinel = run["logs_dir"] / "epochs.jsonl"
    sentinel.write_text("TEST_SYNTHETIC_do_not_overwrite", encoding="utf-8")
    before = (run["run_dir"] / "resolved_config.yaml").read_bytes()
    with pytest.raises(FileExistsError):
        training_entrypoint.prepare_s2d_run_environment(runtime_config)
    assert sentinel.read_text(encoding="utf-8") == "TEST_SYNTHETIC_do_not_overwrite"
    assert (run["run_dir"] / "resolved_config.yaml").read_bytes() == before


def test_generated_runs_retry_collisions(runtime_config, monkeypatch):
    import copy
    import utils.io

    monkeypatch.setattr(utils.io, "make_run_id", lambda name: "TEST_SYNTHETIC_generated")
    suffixes = iter(["aaa", "aaa", "bbb"])
    monkeypatch.setattr(training_entrypoint.secrets, "token_hex", lambda size: next(suffixes))
    first = training_entrypoint.prepare_s2d_run_environment(copy.deepcopy(runtime_config), "20261104")
    second = training_entrypoint.prepare_s2d_run_environment(copy.deepcopy(runtime_config), "20261104")
    assert first["run_dir"] != second["run_dir"]
    assert first["run_dir"].is_dir() and second["run_dir"].is_dir()


@pytest.mark.parametrize("date,run_id", [("20260230", "TEST_SYNTHETIC_ok"),
                                        ("20261104", "../TEST_SYNTHETIC_escape")])
def test_output_identity_rejected_before_creation(runtime_config, date, run_id):
    runtime_config["output"].update(experiment_date=date, run_id=run_id)
    with pytest.raises(ValueError):
        training_entrypoint.prepare_s2d_run_environment(runtime_config)
    assert not Path(runtime_config["system"]["output_dir"]).exists()


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

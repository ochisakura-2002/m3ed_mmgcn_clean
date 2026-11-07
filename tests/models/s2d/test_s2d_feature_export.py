"""Synthetic bank IO/guards only; canonical numerical tests live in test_s2d_features."""

import copy
import json
from pathlib import Path
import shutil
from types import SimpleNamespace
from unittest.mock import Mock
import uuid

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import yaml

from datasets.iemocap.s2d_faces import FaceBBoxCache, FaceCacheEntry, PREPROCESSING_VERSION
from datasets.iemocap.s2d_manifest import S2DManifest
from models.s2d.iemocap.config import load_config
from scripts.models.s2d.iemocap import export_features as entry


ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture
def tmp_path():
    # Match existing S2D IO tests: pytest's mode=0700 temporary directories
    # are inaccessible in the managed Windows sandbox. Only our UUID is removed.
    base = (ROOT / "tmp/assistant_work/s2d_feature_export_tests").resolve()
    path = base / uuid.uuid4().hex
    path.mkdir(parents=True)
    try:
        yield path
    finally:
        assert path.resolve().parent == base
        shutil.rmtree(path)


@pytest.fixture(scope="module", autouse=True)
def limited_cpu_threads():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(before)


def forbidden(*args, **kwargs):
    raise AssertionError("TEST_SYNTHETIC_FORBIDDEN_OPERATION")


@pytest.mark.parametrize("splits", [("test",), ("train", "dev", "test"), (), ("Train",),
                                    ("train", "train"), "train,dev,test", "all", "train,"])
def test_split_hard_rejection_before_any_io(tmp_path, monkeypatch, splits):
    with monkeypatch.context() as guard:
        for name in ("load_frozen_run", "load_frozen_model", "prepare_datasets"):
            guard.setattr(entry, name, forbidden)
        with pytest.raises(ValueError, match="train/dev"):
            entry.export_run("TEST_SYNTHETIC_MISSING_RUN", tmp_path / "bank", splits=splits)
    with pytest.raises(ValueError, match="train/dev"):
        entry.prepare_datasets({}, None, splits)
    with pytest.raises(ValueError, match="train/dev"):
        entry.export_feature_bank(None, {}, {}, tmp_path / "bank", splits=splits, provenance={})
    with pytest.raises(ValueError, match="train/dev"):
        entry.write_split(None, None, tmp_path / "bank", "test", shard_size=2, device="cpu")
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("options", [["--split", "test"], ["--splits", "train,dev,test"],
                                    ["--splits", "test"], ["--splits", ""], []])
def test_cli_test_rejection(options, monkeypatch):
    monkeypatch.setattr(entry, "export_run", forbidden)
    with pytest.raises(SystemExit) as error:
        entry.main(["--run-dir", "TEST_SYNTHETIC", "--output-dir", "TEST_SYNTHETIC", *options])
    assert error.value.code == 2


def test_cli_preserves_explicit_split_request(monkeypatch):
    run = Mock()
    monkeypatch.setattr(entry, "export_run", run)
    entry.main(["--run-dir", "TEST_SYNTHETIC", "--output-dir", "TEST_SYNTHETIC", "--split", "dev"])
    assert run.call_args.kwargs["splits"] == ("dev",)
    assert run.call_args.kwargs["shard_size"] == 256
    assert run.call_args.kwargs["expected_sha256"] == entry.FROZEN_CHECKPOINT_SHA256


def cache_entry(uid, usable=True):
    return FaceCacheEntry(
        uid, tuple(range(16)), ((48, 36),) * 16,
        ((2., 2., 20., 20.) if usable else None,) * 16,
        ((.99 if usable else None),) * 16,
        (usable,) * 16, (False,) * 16,
        (("largest_face" if usable else "no_valid_detection_in_clip"),) * 16,
        16 if usable else 0, usable)


@pytest.fixture
def sources(tmp_path):
    config, vocab = load_config(ROOT / "configs/s2d/iemocap/v0.yaml")
    path = tmp_path / "TEST_SYNTHETIC_manifest.csv"
    path.write_text(
        "utterance_id,video_relative_path,benchmark_label,split,v_available\n"
        "TEST_SYNTHETIC_z,z.mp4,hap,train,1\n"
        "TEST_SYNTHETIC_a,a.mp4,sad,train,1\n"
        "TEST_SYNTHETIC_no_v,,neu,train,0\n"
        "TEST_SYNTHETIC_no_face,f.mp4,ang,train,1\n"
        "TEST_SYNTHETIC_dev,d.mp4,fru,dev,1\n", encoding="utf-8")
    cache_path = tmp_path / "TEST_SYNTHETIC_cache.json"
    entries = {uid: cache_entry(uid, uid != "TEST_SYNTHETIC_no_face") for uid in (
        "TEST_SYNTHETIC_z", "TEST_SYNTHETIC_a", "TEST_SYNTHETIC_no_face", "TEST_SYNTHETIC_dev")}
    FaceBBoxCache(entries).save(cache_path)
    config["dataset"].update(data_root=str(tmp_path), manifest_path=path.name)
    config["face"]["cache_path"] = str(cache_path)
    config["system"]["device"] = "cpu"
    config["train"].update(num_workers=0, batch_size=3)
    return SimpleNamespace(config=config, vocab=vocab, manifest=path, cache=cache_path)


@pytest.mark.parametrize("requested", [("train",), ("dev",), ("dev", "train")])
def test_prepare_reuses_canonical_dataset_only_requested_splits(sources, monkeypatch, requested):
    from datasets.iemocap.s2d_dataset import IEMOCAPS2DDataset, S2DTransform, decode_uniform_clip
    from datasets.iemocap.s2d_sampling import Uniform16Sampler
    from datasets.iemocap.s2d_faces import CachedBBoxFaceProcessor

    calls = []
    original = S2DManifest.for_split

    def checked(manifest, split, **kwargs):
        assert split in requested
        calls.append(split)
        return original(manifest, split, **kwargs)

    monkeypatch.setattr(S2DManifest, "for_split", checked)
    monkeypatch.setattr(S2DManifest, "counts", forbidden)
    datasets, coverage, identities = entry.prepare_datasets(sources.config, sources.vocab, requested)
    assert tuple(datasets) == tuple(coverage) == requested
    assert set(calls) == set(requested)
    for split, dataset in datasets.items():
        assert isinstance(dataset, IEMOCAPS2DDataset)
        assert isinstance(dataset.sampler, Uniform16Sampler) and isinstance(dataset.transform, S2DTransform)
        assert isinstance(dataset.face_processor, CachedBBoxFaceProcessor)
        assert dataset.decoder is decode_uniform_clip and not dataset.remote_preprocessing
        assert len(dataset) == coverage[split]["face_cache_usable"]
    assert identities["manifest"]["sha256"] == entry.file_identity(sources.manifest)["sha256"]
    assert identities["face_cache"]["preprocessing_version"] == PREPROCESSING_VERSION
    if "train" in requested:
        row = coverage["train"]
        assert (row["manifest_total"], row["v_available"], row["face_cache_usable"]) == (4, 3, 2)
        assert row["exclusions"] == {"v_unavailable": ["TEST_SYNTHETIC_no_v"],
                                     "cached_face_unusable": ["TEST_SYNTHETIC_no_face"]}
        assert row["face_unusable_reasons"] == {"TEST_SYNTHETIC_no_face": ["no_valid_detection_in_clip"]}
        assert row["unknown_exclusion_count"] == 0


def test_missing_cache_aborts_instead_of_inventing_exclusion(sources):
    FaceBBoxCache({}).save(sources.cache.with_name("TEST_SYNTHETIC_empty.json"))
    sources.config["face"]["cache_path"] = str(sources.cache.with_name("TEST_SYNTHETIC_empty.json"))
    with pytest.raises(ValueError, match="Missing face cache"):
        entry.prepare_datasets(sources.config, sources.vocab, ("train",))


class SyntheticDataset(Dataset):
    """Writer harness with scalar pixels; NEVER a S2D numerical substitute."""

    def __init__(self, split, count=5):
        self.split = split
        # Deliberately nonalphabetic: preserve canonical manifest order.
        self.records = tuple(SimpleNamespace(utterance_id=f"TEST_SYNTHETIC_{split}_{i:02d}", label_index=i % 6)
                             for i in reversed(range(count)))

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        row = self.records[index]
        return {"utterance_id": row.utterance_id, "label_index": row.label_index,
                "split": self.split, "v_valid": True, "face_usable": True,
                "frame_indices": torch.arange(16) * (index + 1), "video": torch.tensor([float(index)])}


@pytest.fixture
def bank_inputs(monkeypatch):
    model = nn.Linear(1, 1).eval()
    with torch.no_grad():
        model.weight.fill_(2)
        model.bias.fill_(1)
    calls = []

    @torch.inference_mode()
    def synthetic_bundle(current, video):
        assert current is model and not current.training
        calls.append(len(video))
        value = model(video).flatten().double()  # Verify explicit float32 persistence.
        return {key: value.reshape(-1, *([1] * len(shape))).expand(-1, *shape)
                for key, shape in entry.FEATURE_SHAPES.items()}

    monkeypatch.setattr(entry, "extract_uniform16_feature_bundle", synthetic_bundle)
    datasets = {split: SyntheticDataset(split) for split in ("train", "dev")}
    coverage = {split: {"manifest_total": 7, "v_available": 6, "face_cache_usable": 5,
                        "exclusions": {"v_unavailable": ["TEST_SYNTHETIC_MISSING_V"],
                                       "cached_face_unusable": ["TEST_SYNTHETIC_MISSING_FACE"]},
                        "unknown_exclusion_count": 0} for split in datasets}
    provenance = entry.build_provenance(
        {"source_run_id": "TEST_SYNTHETIC", "selected_epoch": 7, "selected_dev_weighted_f1": .34,
         "checkpoint": {"filename": "TEST_SYNTHETIC.pt", "sha256": "a" * 64}, "label_vocab": {"TEST_SYNTHETIC": True}},
        {"manifest": {"path": "TEST_SYNTHETIC.csv"}, "face_cache": {"preprocessing_version": PREPROCESSING_VERSION}},
        ("train", "dev"), 2, git_head="TEST_SYNTHETIC_COMMIT")
    return SimpleNamespace(model=model, datasets=datasets, coverage=coverage, provenance=provenance, calls=calls)


def write_bank(inputs, destination, **kwargs):
    return entry.export_feature_bank(inputs.model, inputs.datasets, inputs.coverage, destination,
                                     splits=("train", "dev"), provenance=inputs.provenance,
                                     shard_size=2, batch_size=3, **kwargs)


def test_shards_index_order_dtype_coverage_provenance_and_no_training(tmp_path, bank_inputs, monkeypatch):
    h = bank_inputs
    state = copy.deepcopy(h.model.state_dict())
    flags = [p.requires_grad for p in h.model.parameters()]
    monkeypatch.setattr(h.model, "train", forbidden)
    monkeypatch.setattr(torch.optim.AdamW, "__init__", forbidden)
    first, second = tmp_path / "bank1", tmp_path / "bank2"
    completed = write_bank(h, first)
    write_bank(h, second)
    assert h.calls == [3, 2] * 4  # One bundle per inference batch; no hidden extra pass.
    assert {p.name for p in first.iterdir()} == {"train", "dev", "provenance.json", "coverage.json", "complete.json"}
    for split, dataset in h.datasets.items():
        index = entry.read_json(first / split / "index.json")
        assert index == entry.read_json(second / split / "index.json")
        assert [r["utterance_id"] for r in index["rows"]] == [r.utterance_id for r in dataset.records]
        assert [r["row_count"] for r in index["shards"]] == [2, 2, 1]
        assert index["row_count"] == 5 and len(index["rows"]) == 5
        for row in index["rows"]:
            shard = torch.load(first / split / row["shard"], weights_only=True)
            other = torch.load(second / split / row["shard"], weights_only=True)
            i, n = row["row_in_shard"], len(shard["utterance_ids"])
            assert shard["schema_version"] == entry.SCHEMA_VERSION and shard["split"] == split
            assert shard["utterance_ids"][i] == row["utterance_id"]
            assert shard["labels"].dtype == shard["frame_indices"].dtype == torch.long
            expected = dataset[row["global_row"]]
            assert shard["labels"][i] == expected["label_index"]
            assert torch.equal(shard["frame_indices"][i], expected["frame_indices"])
            for key, shape in entry.FEATURE_SHAPES.items():
                assert shard[key].shape == (n, *shape) and shard[key].dtype == torch.float32
                assert torch.equal(shard[key], other[key])
                assert bool((shard[key][i] == 2 * row["global_row"] + 1).all())
        assert completed[split]["exported_count"] == 5 and completed[split]["excluded_count"] == 2
    assert set(entry.read_json(first / "coverage.json")["splits"]) == {"train", "dev"}
    provenance = entry.read_json(first / "provenance.json")
    assert {"schema_version", "git_head", "source_run_id", "selected_epoch", "selected_dev_weighted_f1",
            "checkpoint", "label_vocab", "sampling_policy", "num_frames", "feature_fields", "dtype",
            "feature_contract_version", "manifest", "face_cache", "shard_size"} <= provenance.keys()
    assert provenance["dtype"] == "float32" and provenance["sampling_policy"] == "Uniform16"
    assert entry.read_json(first / "complete.json")["status"] == "complete"
    assert all(torch.equal(value, h.model.state_dict()[key]) for key, value in state.items())
    assert [p.requires_grad for p in h.model.parameters()] == flags
    assert h.model.training is False
    assert not list(first.rglob("*.tmp"))


def test_engine_rejects_unrequested_collection(tmp_path, bank_inputs):
    h = bank_inputs
    with pytest.raises(ValueError, match="exactly the requested"):
        entry.export_feature_bank(h.model, h.datasets, h.coverage, tmp_path / "bank",
                                  splits=("train",), provenance=h.provenance)
    assert not (tmp_path / "bank").exists()
    with pytest.raises(ValueError, match="exactly the requested"):
        entry.export_feature_bank(h.model, {**h.datasets, "test": object()}, h.coverage, tmp_path / "bank",
                                  splits=("train", "dev"), provenance=h.provenance)
    assert not (tmp_path / "bank").exists()


def test_dev_only_engine_creates_only_dev(tmp_path, bank_inputs):
    h = bank_inputs
    provenance = dict(h.provenance, splits=["dev"])
    entry.export_feature_bank(h.model, {"dev": h.datasets["dev"]}, {"dev": h.coverage["dev"]},
                              tmp_path / "bank", splits=("dev",), provenance=provenance, shard_size=2)
    bank = tmp_path / "bank"
    assert not (bank / "train").exists()
    assert set(entry.read_json(bank / "coverage.json")["splits"]) == {"dev"}


@pytest.mark.parametrize("existing_content", [False, True])
def test_existing_output_refusal_before_forward(tmp_path, bank_inputs, existing_content):
    bank = tmp_path / "bank"
    bank.mkdir()
    if existing_content:
        (bank / "provenance.json").write_text("TEST_SYNTHETIC_PRESERVED", encoding="utf-8")
    before = {p.name: p.read_bytes() for p in bank.iterdir()}
    with pytest.raises(FileExistsError):
        write_bank(bank_inputs, bank)
    assert not bank_inputs.calls
    assert before == {p.name: p.read_bytes() for p in bank.iterdir()}


def test_atomic_shards_and_json_published_only_after_close(tmp_path, bank_inputs, monkeypatch):
    original = entry.os.replace
    destinations = []

    def inspect(source, destination):
        source, destination = Path(source), Path(destination)
        assert source.suffix == ".tmp" and source.parent == destination.parent and not destination.exists()
        # On Windows this rename also proves the writer's handle is closed.
        if destination.suffix == ".pt":
            assert torch.load(source, weights_only=True)["split"] in entry.ALLOWED_SPLITS
        else:
            json.loads(source.read_text(encoding="utf-8"))
        destinations.append(destination.name)
        original(source, destination)

    monkeypatch.setattr(entry.os, "replace", inspect)
    write_bank(bank_inputs, tmp_path / "bank")
    assert len(destinations) == 11  # 6 shards + 2 indices + provenance/coverage/completion.


@pytest.mark.parametrize("failure", ["serialization", "replace"])
def test_atomic_failure_leaves_no_final_shard_or_temp(tmp_path, bank_inputs, monkeypatch, failure):
    if failure == "serialization":
        def fail(value, stream):
            stream.write(b"TEST_SYNTHETIC_PARTIAL")
            raise OSError("TEST_SYNTHETIC_WRITE_FAILURE")
        monkeypatch.setattr(entry.torch, "save", fail)
    else:
        original = entry.os.replace

        def fail(source, destination):
            if Path(destination).suffix == ".pt":
                raise OSError("TEST_SYNTHETIC_WRITE_FAILURE")
            original(source, destination)
        monkeypatch.setattr(entry.os, "replace", fail)
    bank = tmp_path / "bank"
    with pytest.raises(OSError, match="TEST_SYNTHETIC_WRITE_FAILURE"):
        write_bank(bank_inputs, bank)
    assert not list(bank.rglob("*.tmp")) and not list(bank.rglob("*.pt"))
    assert not (bank / "complete.json").exists() and not (bank / "coverage.json").exists()
    with pytest.raises(FileExistsError):
        write_bank(bank_inputs, bank)


def test_atomic_json_failure_and_existing_file_preserved(tmp_path, monkeypatch):
    path = tmp_path / "index.json"
    with pytest.raises(ValueError):
        entry.atomic_json(path, {"TEST_SYNTHETIC": float("nan")})
    assert list(tmp_path.iterdir()) == []
    path.write_text("TEST_SYNTHETIC_PRESERVED", encoding="utf-8")
    with pytest.raises(FileExistsError):
        entry.atomic_json(path, {})
    assert path.read_text(encoding="utf-8") == "TEST_SYNTHETIC_PRESERVED"


def test_shuffled_loader_and_mixed_split_rows_rejected(tmp_path, bank_inputs):
    h = bank_inputs
    loader = DataLoader(h.datasets["train"], batch_size=2, shuffle=True)
    with pytest.raises(ValueError, match="shuffle=False"):
        entry.write_split(h.model, loader, tmp_path / "shuffled", "train", shard_size=2, device="cpu")
    loader = DataLoader(h.datasets["dev"], batch_size=2, shuffle=False)
    with pytest.raises(ValueError, match="requested split"):
        entry.write_split(h.model, loader, tmp_path / "mixed", "train", shard_size=2, device="cpu")
    assert list(tmp_path.iterdir()) == []


@pytest.fixture
def frozen_run(tmp_path, sources):
    run = tmp_path / "TEST_SYNTHETIC_frozen_run"
    (run / "logs").mkdir(parents=True)
    (run / "checkpoints").mkdir()
    config = sources.config
    config["output"]["run_id"] = run.name
    (run / "resolved_config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    metadata = {"run_id": run.name, "git_commit": "TEST_SYNTHETIC_SOURCE_COMMIT", "label_vocab": sources.vocab.as_dict(),
                "checkpoint_selection_metric": "val_weighted_f1", "test_split_used_for_selection": False,
                "seed": 100, "manifest_path": config["dataset"]["manifest_path"],
                "data_root": config["dataset"]["data_root"], "face_cache_path": config["face"]["cache_path"]}
    (run / "run_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    (run / "logs/label_vocab.json").write_text(json.dumps(sources.vocab.as_dict()), encoding="utf-8")
    rows = [{"epoch": epoch, "best_epoch": min(epoch, 7), "best_val_weighted_f1": min(epoch, 7) / 20,
             "epochs_without_improvement": max(epoch - 7, 0),
             "dev": {"weighted_f1": epoch / 20 if epoch <= 7 else .1}} for epoch in range(1, 16)]
    (run / "logs/epochs.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    checkpoint = run / "checkpoints/best_model_weights_only.pt"
    model = nn.Linear(1, 1)
    torch.save(model.state_dict(), checkpoint)
    return SimpleNamespace(run=run, checkpoint=checkpoint, model=model,
                           sha256=entry.file_identity(checkpoint)["sha256"])


def test_frozen_run_provenance_and_strict_weights_only_reload(frozen_run, monkeypatch):
    h = frozen_run
    before = {p.relative_to(h.run): p.read_bytes() for p in h.run.rglob("*") if p.is_file()}
    config, vocab, checkpoint, evidence = entry.load_frozen_run(h.run, expected_sha256=h.sha256)
    assert checkpoint == h.checkpoint and evidence["selected_epoch"] == 7
    assert evidence["selected_dev_weighted_f1"] == .35
    assert evidence["checkpoint"]["sha256"] == h.sha256
    calls, build_configs = [], []
    original_load = torch.load

    def checked_load(path, **kwargs):
        calls.append((path, kwargs))
        assert kwargs == {"map_location": "cpu", "weights_only": True}
        return original_load(path, **kwargs)

    class Target(nn.Linear):
        def load_state_dict(self, state, strict=True):
            assert strict is True
            return super().load_state_dict(state, strict=strict)

    def build(current_vocab, current_config):
        assert current_vocab == vocab
        build_configs.append(current_config)
        assert "pretrained_checkpoint" not in current_config and "landmark_checkpoint" not in current_config
        return Target(1, 1)

    monkeypatch.setattr(entry, "build_s2d", build)
    monkeypatch.setattr(torch, "load", checked_load)
    loaded = entry.load_frozen_model(config, vocab, checkpoint)
    assert len(calls) == len(build_configs) == 1 and not loaded.training
    assert all(torch.equal(value, loaded.state_dict()[key]) for key, value in h.model.state_dict().items())
    assert before == {p.relative_to(h.run): p.read_bytes() for p in h.run.rglob("*") if p.is_file()}


@pytest.mark.parametrize("failure", ["restricted_load", "strict_keys"])
def test_reload_errors_never_fallback_to_pickle_or_sibling(frozen_run, monkeypatch, failure):
    h = frozen_run
    calls = []

    def fail(path, **kwargs):
        calls.append((path, kwargs))
        assert path == h.checkpoint and kwargs["weights_only"] is True
        if failure == "restricted_load":
            raise RuntimeError("TEST_SYNTHETIC_RESTRICTED_FAILURE")
        return {"TEST_SYNTHETIC_BAD_KEY": torch.zeros(1)}

    monkeypatch.setattr(torch, "load", fail)
    monkeypatch.setattr(entry, "build_s2d", lambda *args: nn.Linear(1, 1))
    config, vocab, checkpoint, _ = entry.load_frozen_run(h.run, expected_sha256=h.sha256)
    with pytest.raises(RuntimeError):
        entry.load_frozen_model(config, vocab, checkpoint)
    assert len(calls) == 1


def test_missing_default_checkpoint_and_bad_hash_fail_closed(frozen_run, monkeypatch):
    h = frozen_run
    monkeypatch.setattr(entry, "build_s2d", forbidden)
    with pytest.raises(ValueError, match="SHA256"):
        entry.load_frozen_run(h.run)  # Real synthetic digest is not the frozen production digest.
    h.checkpoint.rename(h.checkpoint.with_name("best_model.pt"))  # Synthetic file only.
    with pytest.raises(FileNotFoundError):
        entry.load_frozen_run(h.run, expected_sha256=h.sha256)


def test_incomplete_training_or_wrong_vocab_rejected(frozen_run):
    h = frozen_run
    logs = h.run / "logs/epochs.jsonl"
    logs.write_text("\n".join(logs.read_text().splitlines()[:8]) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="stopping boundary"):
        entry.load_frozen_run(h.run, expected_sha256=h.sha256)
    (h.run / "logs/label_vocab.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="label vocab"):
        entry.load_frozen_run(h.run, expected_sha256=h.sha256)


def test_export_run_synthetic_integration_and_original_run_unchanged(frozen_run, bank_inputs, monkeypatch, tmp_path):
    h, inputs = frozen_run, bank_inputs
    before = {p.relative_to(h.run): p.read_bytes() for p in h.run.rglob("*") if p.is_file()}
    monkeypatch.setattr(entry, "build_s2d", lambda *args: inputs.model)
    calls = []

    def prepare(config, vocab, splits):
        calls.append(splits)
        assert splits == ("dev",)
        assert all(torch.equal(value, inputs.model.state_dict()[key]) for key, value in h.model.state_dict().items())
        return {"dev": inputs.datasets["dev"]}, {"dev": inputs.coverage["dev"]}, {"manifest": {"path": "TEST_SYNTHETIC"}}

    monkeypatch.setattr(entry, "prepare_datasets", prepare)
    monkeypatch.setattr(entry.subprocess, "run", Mock(return_value=SimpleNamespace(stdout="TEST_SYNTHETIC_HEAD\n")))
    destination = tmp_path / "bank"
    entry.main(["--run-dir", str(h.run), "--output-dir", str(destination), "--split", "dev",
                "--expected-checkpoint-sha256", h.sha256, "--shard-size", "2"])
    assert calls == [("dev",)]
    provenance = entry.read_json(destination / "provenance.json")
    assert provenance["selected_epoch"] == 7 and provenance["checkpoint"]["sha256"] == h.sha256
    assert provenance["git_head"] == "TEST_SYNTHETIC_HEAD"
    assert provenance["splits"] == ["dev"]
    assert set(entry.read_json(destination / "coverage.json")["splits"]) == {"dev"}
    assert before == {p.relative_to(h.run): p.read_bytes() for p in h.run.rglob("*") if p.is_file()}


@pytest.mark.parametrize("count", [0, 4])
def test_empty_and_exact_shard_boundaries(tmp_path, bank_inputs, count):
    h = bank_inputs
    dataset = SyntheticDataset("dev", count)
    coverage = {"manifest_total": count, "v_available": count, "face_cache_usable": count,
                "exclusions": {"v_unavailable": [], "cached_face_unusable": []}}
    entry.export_feature_bank(h.model, {"dev": dataset}, {"dev": coverage}, tmp_path / "bank",
                              splits=("dev",), provenance=dict(h.provenance, splits=["dev"]),
                              shard_size=2, batch_size=3)
    index = entry.read_json(tmp_path / "bank/dev/index.json")
    assert index["row_count"] == count
    assert [row["row_count"] for row in index["shards"]] == [2] * (count // 2)


def test_writer_rejects_train_mode_without_changing_flags(tmp_path, bank_inputs):
    h = bank_inputs
    h.model.train()
    with pytest.raises(ValueError, match=r"model.eval"):
        write_bank(h, tmp_path / "bank")
    assert h.model.training and not (tmp_path / "bank").exists() and not h.calls

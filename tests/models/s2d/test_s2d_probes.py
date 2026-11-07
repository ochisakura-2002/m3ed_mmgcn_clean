"""Synthetic-only Q1 runner, statistics, archive and fail-closed contracts."""

import copy
import csv
import json
from pathlib import Path
import shutil
import uuid

import pytest
import torch
from torch import nn

from models.s2d.iemocap.probes import LinearProbe, TemporalProbe, mean_pool, seed_probe, SEEDS
from scripts.models.s2d.iemocap import probe_bank as banks
from scripts.models.s2d.iemocap import probe_reports as reports
from scripts.models.s2d.iemocap import run_probes as entry


ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture
def tmp_path():
    # Existing S2D IO convention: sandbox cannot access pytest mode=0700 dirs.
    base = (ROOT / "tmp/assistant_work/s2d_probe_tests").resolve()
    path = base / uuid.uuid4().hex
    path.mkdir(parents=True)
    try:
        yield path
    finally:
        assert path.resolve().parent == base
        shutil.rmtree(path)


@pytest.fixture(scope="module", autouse=True)
def limited_threads():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(before)


def write_json(path, data):
    path.write_text(json.dumps(data), encoding="utf-8")


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def forbidden(*args, **kwargs):
    raise AssertionError("TEST_SYNTHETIC_FORBIDDEN")


@pytest.fixture
def config():
    value = entry.load_config("configs/s2d/iemocap/q1_probe.yaml")
    value["train"].update(max_epochs=2, patience=1, batch_size=6)
    return value


@pytest.fixture
def bank(tmp_path, monkeypatch):
    """Same canonical export schema, tiny generated tensors, no real assets.

    A synthetic score (1.0) is substituted ONLY in this pytest process. The
    production CLI/config has no expected-score or gate-bypass option.
    """
    monkeypatch.setattr(banks, "EXPECTED_DEV_WEIGHTED_F1", 1.0)
    root = tmp_path / "TEST_SYNTHETIC_BANK"
    root.mkdir()
    vocab = {"target": "benchmark_label", "index_to_label": list(banks.LABELS),
             "label_to_index": {label: i for i, label in enumerate(banks.LABELS)},
             "source": "TEST_SYNTHETIC", "ta_mapping_verified": True}
    write_json(root / "provenance.json", {
        "schema_version": banks.SCHEMA, "feature_contract_version": banks.CONTRACT,
        "splits": list(banks.SPLITS), "dtype": "float32", "num_frames": 16,
        "sampling_policy": "Uniform16", "shuffle": False,
        "feature_fields": {k: list(v) for k, v in banks.SHAPES.items()},
        "label_vocab": vocab, "selected_epoch": 7, "selected_dev_weighted_f1": 1.0,
        # Must never be followed; this runner reads only the supplied bank.
        "source_run_path": "TEST_SYNTHETIC_DO_NOT_OPEN_SOURCE_RUN"})
    write_json(root / "complete.json", {"schema_version": banks.SCHEMA, "status": "complete", "splits": list(banks.SPLITS)})
    coverage = {}
    generator = torch.Generator().manual_seed(900)
    for split, count in (("train", 12), ("dev", 6)):
        (root / split).mkdir()
        ids = [f"TEST_SYNTHETIC_{split}_{i}" for i in range(count)]
        labels = torch.arange(count) % 6
        a0 = torch.randn(count, 16, 768, generator=generator)
        c0 = torch.randn(count, 16, 768, generator=generator) * .2
        b = torch.randn(count, 16, 768, generator=generator)
        payload = {"schema_version": banks.SCHEMA, "split": split, "utterance_ids": ids,
                   "labels": labels, "frame_indices": torch.arange(16).repeat(count, 1),
                   banks.FEATURES["a0"]: a0, banks.FEATURES["c0"]: c0,
                   banks.FEATURES["d0"]: a0 + c0, banks.FEATURES["b"]: b,
                   "h_v": b.mean(1), "logits_v": torch.nn.functional.one_hot(labels, 6).float() * 2}
        shards, rows = [], []
        for part, start in enumerate(range(0, count, 5)):
            end = min(start + 5, count)
            name = f"part-{part:05d}.pt"
            torch.save({k: v[start:end] if isinstance(v, (list, torch.Tensor)) else v for k, v in payload.items()}, root / split / name)
            shards.append({"filename": name, "row_count": end - start})
            rows.extend(dict(global_row=i, utterance_id=ids[i], shard=name, row_in_shard=i-start) for i in range(start, end))
        write_json(root / split / "index.json", {"schema_version": banks.SCHEMA, "split": split,
                                                "row_count": count, "shards": shards, "rows": rows})
        coverage[split] = {"manifest_total": count + 1, "v_available": count, "face_cache_usable": count,
                           "exported_count": count, "excluded_count": 1, "unknown_exclusion_count": 0,
                           "exclusions": {"v_unavailable": [f"TEST_SYNTHETIC_EXCLUDED_{split}"], "cached_face_unusable": []}}
    write_json(root / "coverage.json", {"schema_version": banks.SCHEMA, "splits": coverage})
    return root


def mutate_json(path, change):
    value = banks.read_json(path)
    change(value)
    write_json(path, value)


def mutate_shard(bank, split, change):
    path = bank / split / "part-00000.pt"
    payload = torch.load(path, weights_only=True)
    change(payload)
    torch.save(payload, path)


def test_fixed_production_protocol_and_reference():
    assert banks.EXPECTED_EPOCH == 7
    assert banks.EXPECTED_DEV_WEIGHTED_F1 == 0.34355913724532466
    assert banks.BASELINE_TOLERANCE == 1e-10
    config = entry.load_config("configs/s2d/iemocap/q1_probe.yaml")
    assert config["seeds"] == [100, 101, 102, 103, 104]
    assert config["train"] == dict(learning_rate=.001, weight_decay=.01, max_epochs=100,
                                   patience=10, batch_size=128, cpu_threads=1)


def test_bank_schema_pooling_and_read_only(bank):
    before = {p.relative_to(bank).as_posix(): banks.identity(p) for p in bank.rglob("*") if p.is_file()}
    loaded = banks.FeatureBank(bank)
    assert loaded.ids["train"] == [f"TEST_SYNTHETIC_train_{i}" for i in range(12)]
    assert torch.equal(loaded.values("dev", banks.FEATURES["b"], pooled=True), loaded.values("dev", "h_v"))
    for feature in banks.FEATURES.values():
        values = loaded.values("train", feature)
        assert values.shape == (12, 16, 768)
        assert torch.equal(mean_pool(values), loaded.values("train", feature, pooled=True))
    loaded.assert_unchanged()
    assert before == {p.relative_to(bank).as_posix(): banks.identity(p) for p in bank.rglob("*") if p.is_file()}


@pytest.mark.parametrize("name", ["provenance.json", "coverage.json", "complete.json"])
def test_extra_test_split_rejected_before_shard_io(bank, monkeypatch, name):
    def change(value):
        if isinstance(value["splits"], list):
            value["splits"].append("test")
        else:
            value["splits"]["test"] = {}
    mutate_json(bank / name, change)
    monkeypatch.setattr(torch, "load", forbidden)
    with pytest.raises(ValueError, match="exactly train/dev"):
        banks.FeatureBank(bank)


@pytest.mark.parametrize("change,match", [
    (lambda v: v.update(schema_version="unknown"), "schema"),
    (lambda v: v.update(selected_epoch=8), "provenance"),
    (lambda v: v.update(selected_dev_weighted_f1=float("nan")), "provenance"),
    (lambda v: v.update(dtype="float16"), "provenance"),
    (lambda v: v["label_vocab"]["index_to_label"].reverse(), "provenance"),
])
def test_invalid_metadata(bank, change, match):
    mutate_json(bank / "provenance.json", change)
    with pytest.raises(ValueError, match=match):
        banks.FeatureBank(bank)


@pytest.mark.parametrize("change,match", [
    (lambda v: v.update(split="test"), "schema"),
    (lambda v: v.update(labels=v["labels"].float()), "dtype"),
    (lambda v: v.update(labels=torch.full_like(v["labels"], 6)), "labels"),
    (lambda v: v.update(h_v=v["h_v"] + 1), "mean_t"),
    (lambda v: v.update(logits_v=v["logits_v"].double()), "dtype"),
    (lambda v: v["a0_rgb_mean"].fill_(float("nan")), "finiteness"),
    (lambda v: v.update(a0_rgb_mean=v["a0_rgb_mean"][:, :15]), "shape"),
    (lambda v: v["frame_indices"].fill_(-1), "frame indices"),
    (lambda v: v["utterance_ids"].reverse(), "IDs"),
])
def test_invalid_shards(bank, change, match):
    mutate_shard(bank, "train", change)
    with pytest.raises(ValueError, match=match):
        banks.FeatureBank(bank)


@pytest.mark.parametrize("filename", ["../test/part-00000.pt", "test.pt", "/test/part-00000.pt", "part-00001.pt"])
def test_test_or_traversal_filename_not_opened(bank, monkeypatch, filename):
    mutate_json(bank / "train/index.json", lambda v: v["shards"][0].update(filename=filename))
    monkeypatch.setattr(torch, "load", forbidden)
    with pytest.raises(ValueError, match="filenames"):
        banks.FeatureBank(bank)


def test_missing_complete_coverage_duplicate_and_mutation(bank):
    complete = (bank / "complete.json").read_bytes()
    (bank / "complete.json").unlink()
    with pytest.raises(FileNotFoundError):
        banks.FeatureBank(bank)
    (bank / "complete.json").write_bytes(complete)
    coverage = (bank / "coverage.json").read_bytes()
    mutate_json(bank / "coverage.json", lambda v: v["splits"]["train"].update(unknown_exclusion_count=1))
    with pytest.raises(ValueError, match="accounting"):
        banks.FeatureBank(bank)
    (bank / "coverage.json").write_bytes(coverage)
    loaded = banks.FeatureBank(bank)
    mutate_shard(bank, "dev", lambda v: v["logits_v"].zero_())
    with pytest.raises(ValueError, match="changed"):
        loaded.values("dev", "logits_v")
    with pytest.raises(ValueError, match="changed"):
        loaded.assert_unchanged()
    mutate_json(bank / "train/index.json", lambda v: v["rows"][1].update(utterance_id=v["rows"][0]["utterance_id"]))
    with pytest.raises(ValueError, match="unique"):
        banks.FeatureBank(bank)


def test_no_discovery_no_source_no_test_paths(bank, monkeypatch):
    open_before = Path.open
    def checked(path, *args, **kwargs):
        assert path.resolve().is_relative_to(bank.resolve())
        assert "test" not in path.relative_to(bank).parts
        return open_before(path, *args, **kwargs)
    monkeypatch.setattr(Path, "open", checked)
    monkeypatch.setattr(Path, "glob", forbidden)
    monkeypatch.setattr(Path, "rglob", forbidden)
    monkeypatch.setattr(Path, "iterdir", forbidden)
    loaded = banks.FeatureBank(bank)
    for split in ("test", "all", "Train", ""):
        with pytest.raises(ValueError, match="train/dev"):
            loaded.values(split, "logits_v")
    loaded.normalization("b")


def test_train_only_normalization_and_shared_pool_transform(bank, monkeypatch):
    loaded = banks.FeatureBank(bank)
    original = loaded._read_shard
    calls = []
    def train_only(split, *args, **kwargs):
        assert split == "train"
        calls.append(split)
        return original(split, *args, **kwargs)
    with monkeypatch.context() as guard:
        guard.setattr(loaded, "_read_shard", train_only)
        stats = {feature: loaded.normalization(feature) for feature in banks.FEATURES}
    assert calls
    for feature, stat in stats.items():
        train = loaded.values("train", banks.FEATURES[feature])
        assert stat["fit_split"] == "train" and stat["count"] == 12 * 16
        torch.testing.assert_close(stat["mean"], train.double().mean((0, 1)).float())
        torch.testing.assert_close(stat["std"], train.double().std((0, 1), unbiased=False).float())
        normalized = (train - stat["mean"]) / stat["std"]
        torch.testing.assert_close(normalized.mean((0, 1)), torch.zeros(768), atol=1e-6, rtol=0)
        torch.testing.assert_close(normalized.mean(1), (train.mean(1) - stat["mean"]) / stat["std"], atol=1e-6, rtol=1e-5)
    assert not torch.equal(stats["a0"]["std"], stats["c0"]["std"])


def test_normalization_constant_channels_and_dev_cannot_change_fit(bank):
    for shard in (bank / "train").glob("part-*.pt"):
        payload = torch.load(shard, weights_only=True)
        payload[banks.FEATURES["a0"]][:, :, 0] = 1e6
        torch.save(payload, shard)
    first = banks.FeatureBank(bank).normalization("a0")
    assert first["mean"][0] == 1e6
    assert float(first["std"][0]) == pytest.approx(1e-6)
    mutate_shard(bank, "dev", lambda v: v[banks.FEATURES["a0"]].add_(5000))
    second = banks.FeatureBank(bank).normalization("a0")
    assert torch.equal(first["mean"], second["mean"])
    assert torch.equal(first["std"], second["std"])


@pytest.mark.parametrize("target", ["provenance.json", "train/index.json", "train/part-00000.pt"])
def test_symlink_metadata_and_shards_refused_before_open(bank, monkeypatch, target):
    original = Path.is_symlink
    monkeypatch.setattr(Path, "is_symlink", lambda path: path == bank / target or original(path))
    with pytest.raises(ValueError, match="symlink"):
        banks.FeatureBank(bank)


def test_shapes_and_causal_convolution_boundary():
    seed_probe(100)
    x = torch.randn(3, 16, 768)
    assert LinearProbe()(mean_pool(x)).shape == (3, 6)
    tcn = TemporalProbe().eval()
    assert tcn(x).shape == (3, 6)
    original = tcn.encode_slots(x)
    changed = x.clone()
    changed[:, 9:] += 100
    torch.testing.assert_close(original[:, :, :9], tcn.encode_slots(changed)[:, :, :9], rtol=0, atol=0)
    assert sum(p.numel() for p in tcn.parameters()) < 250000
    assert all(not isinstance(m, (nn.MultiheadAttention, nn.GRU, nn.LSTM, nn.Transformer)) for m in tcn.modules())
    with pytest.raises(ValueError):
        LinearProbe()(x)
    with pytest.raises(ValueError):
        tcn(x[:, :8])
    with pytest.raises(ValueError):
        mean_pool(x[:, :8])


def test_metric_definition_hand_calculation():
    report = reports.classification_report(torch.tensor([0, 0, 1, 2, 2, 2]), torch.tensor([0, 1, 1, 2, 0, 2]))
    assert report["acc"] == pytest.approx(4 / 6)
    assert report["weighted_f1"] == pytest.approx((2 * .5 + 2 / 3 + 3 * .8) / 6)
    assert report["macro_f1"] == pytest.approx((.5 + 2 / 3 + .8) / 6)
    assert report["uar"] == pytest.approx((.5 + 1 + 2 / 3) / 6)
    assert report["confusion_matrix"][0] == [1, 1, 0, 0, 0, 0]
    assert report["per_class"][5]["support"] == 0


def test_gate_tolerance_fixed_without_relaxation():
    expected = banks.EXPECTED_DEV_WEIGHTED_F1
    provenance = {"selected_epoch": 7}
    assert entry.baseline_gate({"weighted_f1": expected + 5e-11}, provenance)["status"] == "PASS"
    assert entry.baseline_gate({"weighted_f1": expected + 2e-10}, provenance)["status"] == "FAIL"
    assert entry.baseline_gate({"weighted_f1": float("nan")}, provenance)["status"] == "FAIL"


def test_baseline_mismatch_stops_before_fitting(bank, config, tmp_path, monkeypatch):
    mutate_shard(bank, "dev", lambda v: v["logits_v"].zero_())
    monkeypatch.setattr(entry, "fit_probe", forbidden)
    output = tmp_path / "FAILURE_ARCHIVE"
    with pytest.raises(entry.BaselineMismatch, match="PROBE_TRAINING_ALLOWED=NO"):
        entry.run_probes(bank, output, config)
    gate = banks.read_json(output / "01_reference/baseline_reproduction_gate.json")
    assert gate["status"] == "FAIL" and gate["absolute_difference"] > 1e-10
    assert not (output / "complete.json").exists()
    assert not (output / "02_dummy").exists()
    assert not (output / "03_linear").exists()


@pytest.mark.parametrize("kind", ["linear", "tcn"])
def test_seed_determinism_and_real_probe_checkpoint_roundtrip(bank, config, kind, tmp_path):
    loaded = banks.FeatureBank(bank)
    train = loaded.values("train", banks.FEATURES["d0"], pooled=kind == "linear")
    dev = loaded.values("dev", banks.FEATURES["d0"], pooled=kind == "linear")
    arguments = dict(kind=kind, seed=100, config=config, device="cpu")
    first = entry.fit_probe(train, loaded.labels["train"], dev, loaded.labels["dev"], **arguments)
    second = entry.fit_probe(train, loaded.labels["train"], dev, loaded.labels["dev"], **arguments)
    assert first[1] == second[1] and first[3] == second[3]
    assert torch.equal(first[2], second[2])
    assert all(torch.equal(first[0][k], second[0][k]) for k in first[0])
    seed_probe(101)
    different = (LinearProbe() if kind == "linear" else TemporalProbe()).state_dict()
    assert any(not torch.equal(first[0][k], different[k]) for k in different)
    path = tmp_path / "TEST_SYNTHETIC_PROBE.pt"
    reports.atomic_write(path, lambda stream: torch.save(first[0], stream))
    restored = LinearProbe() if kind == "linear" else TemporalProbe()
    restored.load_state_dict(torch.load(path, weights_only=True), strict=True)
    report, probs = entry.evaluate(restored, dev, loaded.labels["dev"], 6, "cpu")
    assert report["weighted_f1"] == first[1]["weighted_f1"]
    assert torch.equal(probs, first[2])


def test_early_stopping_ties_keep_earliest_weights(bank, config, monkeypatch):
    loaded = banks.FeatureBank(bank)
    train = loaded.values("train", banks.FEATURES["a0"], pooled=True)
    dev = loaded.values("dev", banks.FEATURES["a0"], pooled=True)
    original = entry.evaluate
    def constant_score(*args, **kwargs):
        report, probabilities = original(*args, **kwargs)
        report["weighted_f1"] = .25
        return report, probabilities
    monkeypatch.setattr(entry, "evaluate", constant_score)
    config["train"].update(max_epochs=1, patience=2)
    first = entry.fit_probe(train, loaded.labels["train"], dev, loaded.labels["dev"], kind="linear", seed=100, config=config, device="cpu")
    config["train"]["max_epochs"] = 6
    later = entry.fit_probe(train, loaded.labels["train"], dev, loaded.labels["dev"], kind="linear", seed=100, config=config, device="cpu")
    assert later[1]["best_epoch"] == 1 and later[1]["stopped_epoch"] == 3
    assert later[1]["stop_reason"] == "early_stopping"
    assert all(torch.equal(first[0][key], later[0][key]) for key in first[0])


def test_summary_paired_std_not_unpaired():
    rows = []
    for model, values in (("L-A0", [.1, .3]), ("L-D0", [.2, .4]), ("L-C0", [.5, .6]),
                          ("L-B", [.4, .7]), ("T-D0", [.3, .8]), ("T-B", [.9, 1.0])):
        for seed, value in zip((100, 101), values):
            rows.append(dict(model=model, feature=model[-1], probe_type="linear", seed=seed,
                             best_epoch=1, **{k: value for k in reports.METRICS}))
    rows.append(dict(model="OriginalS2D", feature="h_v", probe_type="reference", seed="", best_epoch=7,
                     **{k: .25 for k in reports.METRICS}))
    summary = reports.summarize(rows)
    assert summary[0]["weighted_f1_mean"] == pytest.approx(.2)
    assert summary[0]["weighted_f1_std"] == pytest.approx(.02 ** .5)
    deltas = reports.pairwise_deltas(rows)
    assert len(deltas) == 6 * 4 * 3
    correction = next(r for r in deltas if r["comparison"] == "D0 - A0" and r["metric"] == "weighted_f1" and r["row_type"] == "summary")
    assert correction["delta_mean"] == pytest.approx(.1) and correction["delta_std"] == pytest.approx(0, abs=1e-15)
    utilization = next(r for r in deltas if r["comparison"] == "LinearB - OriginalS2D" and r["row_type"] == "summary")
    assert utilization["delta_mean"] == pytest.approx(.3)


def test_atomic_writes_cleanup_no_clobber_and_concurrent_target(tmp_path, monkeypatch):
    path = tmp_path / "atomic.json"
    def crash(stream):
        stream.write(b"partial")
        raise RuntimeError("injected failure")
    with pytest.raises(RuntimeError):
        reports.atomic_write(path, crash)
    assert list(tmp_path.iterdir()) == []
    original_link = reports.os.link
    def race(source, target):
        Path(target).write_bytes(b"concurrent owner")
        original_link(source, target)
    with monkeypatch.context() as guard:
        guard.setattr(reports.os, "link", race)
        with pytest.raises(FileExistsError):
            reports.atomic_json(path, {"new": True})
    assert path.read_bytes() == b"concurrent owner"
    assert len(list(tmp_path.iterdir())) == 1
    with pytest.raises(FileExistsError):
        reports.atomic_json(path, {})
    good = tmp_path / "good.json"
    reports.atomic_json(good, {"valid": True})
    assert banks.read_json(good) == {"valid": True}


def test_existing_output_refused_before_bank_read(bank, config, tmp_path, monkeypatch):
    output = tmp_path / "existing"
    output.mkdir()
    monkeypatch.setattr(banks, "FeatureBank", forbidden)
    with pytest.raises(FileExistsError):
        entry.run_probes(bank, output, config)
    assert not list(output.iterdir())


@pytest.mark.parametrize("options", [["--split", "test"], ["--splits", "train,dev,test"],
                                    ["--expected-score", "0"], ["--seeds", "100"], ["--skip-baseline"]])
def test_cli_no_split_or_gate_bypass(options, monkeypatch):
    monkeypatch.setattr(entry, "run_probes", forbidden)
    with pytest.raises(SystemExit) as error:
        entry.main(["--feature-bank", "TEST_SYNTHETIC", "--output-dir", "TEST_SYNTHETIC", *options])
    assert error.value.code == 2


def test_unknown_config_or_feature_tuning_before_bank_read(bank, config, tmp_path, monkeypatch):
    monkeypatch.setattr(banks, "FeatureBank", forbidden)
    for change in (lambda v: v.update(split="test"), lambda v: v.update(seeds=[100]),
                   lambda v: v["train"].update(a0_learning_rate=.1)):
        wrong = copy.deepcopy(config)
        change(wrong)
        with pytest.raises(ValueError):
            entry.run_probes(bank, tmp_path / "unused", wrong)


@pytest.mark.parametrize("disabled", [False, True])
def test_preflight_only_and_disabled_no_fitting(bank, config, tmp_path, monkeypatch, disabled):
    output = tmp_path / "preflight"
    config["enabled"] = not disabled
    monkeypatch.setattr(entry, "fit_probe", forbidden)
    result = entry.run_probes(bank, output, config, preflight_only=not disabled)
    assert result["status"] == "PASS"
    complete = banks.read_json(output / "complete.json")
    assert complete["trainable_runs"] == 0 and complete["mode"] == "preflight_only"
    assert len(read_csv(output / "05_summary/metrics_long.csv")) == 2
    assert read_csv(output / "05_summary/pairwise_deltas.csv") == []
    dummy = banks.read_json(output / "02_dummy/config.json")
    assert dummy["class_index"] == 0 and dummy["fit_split"] == "train"


def test_full_matrix_archive_and_complete_marker_last(bank, config, tmp_path, monkeypatch):
    output = tmp_path / "full_archive"
    writes = []
    original_write = reports.atomic_write
    def observed(path, writer):
        writes.append(Path(path).relative_to(output).as_posix())
        if Path(path).name == "complete.json":
            assert (output / "05_summary/metrics_summary.csv").is_file()
            assert len(list(output.glob("03_linear/*/seed*/best_model.pt"))) == 20
            assert len(list(output.glob("04_tcn/*/seed*/best_model.pt"))) == 10
        return original_write(path, writer)
    monkeypatch.setattr(reports, "atomic_write", observed)
    entry.run_probes(bank, output, config)
    assert writes[-1] == "complete.json"
    complete = banks.read_json(output / "complete.json")
    assert complete["trainable_runs"] == 30 and complete["test_accessed"] is False
    assert len(complete["artifacts"]) == len(writes) - 1
    for path, expected in complete["artifacts"].items():
        assert banks.identity(output / path) == expected
    long = read_csv(output / "05_summary/metrics_long.csv")
    assert len(long) == 32 and set(reports.LONG_FIELDS) <= set(long[0])
    summary = read_csv(output / "05_summary/metrics_summary.csv")
    assert len(summary) == 8
    assert all(int(r["n_seeds"]) == 5 for r in summary if r["probe_type"] in {"linear", "tcn"})
    assert len(read_csv(output / "05_summary/pairwise_deltas.csv")) == 6 * 4 * 6
    assert len(read_csv(output / "05_summary/per_class_metrics.csv")) == 32 * 6
    assert len(read_csv(output / "05_summary/training_curves/metrics_by_epoch.csv")) >= 30
    for kind, features in (("03_linear", ("a0", "c0", "d0", "b")), ("04_tcn", ("d0", "b"))):
        for feature in features:
            for seed in SEEDS:
                folder = output / kind / feature / f"seed{seed}"
                assert {p.name for p in folder.iterdir()} == {"config.json", "metrics_best.json", "metrics_history.csv",
                    "predictions_dev.csv", "per_class_metrics.csv", "confusion_matrix.csv", "best_model.pt"}
                predictions = read_csv(folder / "predictions_dev.csv")
                assert list(predictions[0]) == list(reports.PREDICTION_FIELDS)
                assert [r["utterance_id"] for r in predictions] == [f"TEST_SYNTHETIC_dev_{i}" for i in range(6)]
                assert all(sum(float(r[f"prob_{i}"]) for i in range(6)) == pytest.approx(1) for r in predictions)
                matrix = read_csv(folder / "confusion_matrix.csv")
                assert len(matrix) == 6 and list(matrix[0]) == ["true_label", "true_label_name", *[f"pred_{i}" for i in range(6)]]
    assert not list(output.rglob("*.tmp"))


def test_failure_during_training_never_completes_or_resumes(bank, config, tmp_path, monkeypatch):
    output = tmp_path / "interrupted"
    monkeypatch.setattr(entry, "fit_probe", lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("injected training failure")))
    with pytest.raises(RuntimeError, match="injected"):
        entry.run_probes(bank, output, config)
    assert not (output / "complete.json").exists()
    assert not list(output.rglob("*.tmp"))
    with pytest.raises(FileExistsError):
        entry.run_probes(bank, output, config)


def test_completion_publication_failure_leaves_no_marker(bank, config, tmp_path, monkeypatch):
    output = tmp_path / "completion_failure"
    original = reports.atomic_write
    def fail_completion(path, writer):
        if Path(path).name == "complete.json":
            def crash(stream):
                stream.write(b"unfinished marker")
                raise OSError("injected marker failure")
            return original(path, crash)
        return original(path, writer)
    monkeypatch.setattr(reports, "atomic_write", fail_completion)
    with pytest.raises(OSError, match="marker failure"):
        entry.run_probes(bank, output, config, preflight_only=True)
    assert (output / "05_summary/metrics_summary.csv").exists()
    assert not (output / "complete.json").exists()
    assert not list(output.rglob("*.tmp"))

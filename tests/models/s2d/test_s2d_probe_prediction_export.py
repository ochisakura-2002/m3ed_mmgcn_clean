"""Synthetic frozen Linear-B export; no real IEMOCAP, training or Test assets."""

import csv
import json
from pathlib import Path
import shutil
import uuid

import pytest
import torch

from models.s2d.iemocap.probes import LinearProbe
from scripts.models.s2d.iemocap import export_probe_predictions as export
from scripts.models.s2d.iemocap import probe_bank as banks
from scripts.models.s2d.iemocap import probe_reports as reports
from scripts.models.s2d.iemocap import run_probes as runner

ROOT = Path(__file__).resolve().parents[3]


def forbidden(*args, **kwargs):
    raise AssertionError("Synthetic export reached forbidden IO or fitting")


@pytest.fixture
def tmp_path():
    # Match existing S2D tests: avoid inaccessible pytest mode=0700 directories.
    base = (ROOT / "tmp/assistant_work/linear_b_prediction_export").resolve()
    directory = base / uuid.uuid4().hex
    directory.mkdir(parents=True)
    try:
        yield directory
    finally:
        assert directory.resolve().parent == base
        shutil.rmtree(directory)


@pytest.fixture(scope="module", autouse=True)
def limited_threads():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(before)


def write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def seal(case):
    # Synthetic archive only: recompute its explicit six-artifact inventory.
    write_json(case["archive"] / "complete.json", {
        "schema_version": "s2d_q1_probe_archive_v1", "status": "complete",
        "mode": "all_30_probes", "trainable_runs": 30, "test_accessed": False,
        "artifacts": {p.relative_to(case["archive"]).as_posix(): banks.identity(p)
                      for p in case["sources"]},
    })


@pytest.fixture
def frozen(tmp_path, monkeypatch):
    # Tiny row counts substituted only in pytest, never a production CLI bypass.
    monkeypatch.setattr(export, "EXPECTED_ROWS", {"train": 7, "dev": 4})
    bank_dir, archive = tmp_path / "synthetic_bank", tmp_path / "synthetic_q1"
    bank_dir.mkdir()
    run = archive / "03_linear/b/seed100"
    norm_dir = archive / "00_provenance/normalization"
    run.mkdir(parents=True)
    norm_dir.mkdir(parents=True)
    provenance = {
        "schema_version": banks.SCHEMA, "feature_contract_version": banks.CONTRACT,
        "splits": ["train", "dev"], "dtype": "float32", "num_frames": 16,
        "sampling_policy": "Uniform16", "shuffle": False,
        "feature_fields": {k: list(v) for k, v in banks.SHAPES.items()},
        "label_vocab": {"target": "benchmark_label", "index_to_label": list(banks.LABELS),
                        "label_to_index": {label: i for i, label in enumerate(banks.LABELS)}},
        "selected_epoch": 7, "selected_dev_weighted_f1": banks.EXPECTED_DEV_WEIGHTED_F1,
    }
    write_json(bank_dir / "provenance.json", provenance)
    write_json(bank_dir / "complete.json", {
        "schema_version": banks.SCHEMA, "status": "complete", "splits": ["train", "dev"]})
    generator = torch.Generator().manual_seed(789)
    coverage = {}
    for split, count in export.EXPECTED_ROWS.items():
        directory = bank_dir / split
        directory.mkdir()
        # Deliberately non-lexicographic order, spanning multiple shards.
        ids = [f"synthetic_{split}_{count-i}" for i in range(count)]
        labels = (torch.arange(count) + 1) % 6
        slots = torch.randn(count, 16, 768, generator=generator) * 2 + 3
        payload = {
            "schema_version": banks.SCHEMA, "split": split, "utterance_ids": ids,
            "labels": labels, "frame_indices": torch.arange(16).repeat(count, 1),
            **{field: slots.clone() for field in banks.FEATURES.values()},
            "h_v": slots.mean(1), "logits_v": torch.zeros(count, 6),
        }
        shards, rows = [], []
        for part, start in enumerate(range(0, count, 3)):
            end, name = min(start + 3, count), f"part-{part:05d}.pt"
            torch.save({key: value[start:end] if isinstance(value, (list, torch.Tensor)) else value
                        for key, value in payload.items()}, directory / name)
            shards.append({"filename": name, "row_count": end - start})
            rows.extend(dict(global_row=i, utterance_id=ids[i], shard=name, row_in_shard=i-start)
                        for i in range(start, end))
        write_json(directory / "index.json", {
            "schema_version": banks.SCHEMA, "split": split, "row_count": count,
            "shards": shards, "rows": rows})
        coverage[split] = {
            "manifest_total": count, "v_available": count, "face_cache_usable": count,
            "exported_count": count, "excluded_count": 0, "unknown_exclusion_count": 0,
            "exclusions": {"v_unavailable": [], "cached_face_unusable": []},
        }
    write_json(bank_dir / "coverage.json", {"schema_version": banks.SCHEMA, "splits": coverage})
    bank = banks.FeatureBank(bank_dir)
    # Frozen arbitrary stats differ visibly from these synthetic Train moments.
    mean, std = torch.linspace(-1, 1, 768), torch.linspace(.5, 2, 768)
    norm = norm_dir / "b.json"
    write_json(norm, {
        "feature": "b", "field": banks.FEATURES["b"], "fit_split": "train",
        "axes": ["utterance", "slot"], "count": 7 * 16, "ddof": 0, "epsilon": 1e-6,
        "mean": mean.tolist(), "std": std.tolist(),
        "train_index_identity": bank.metadata_identities["train/index.json"],
    })
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(100)
        model = LinearProbe()
    with torch.no_grad():
        model.classifier.weight.mul_(.1)
        model.classifier.bias.copy_(torch.arange(6) * .15)
    torch.save(model.state_dict(), run / "best_model.pt")
    config = runner.load_config("configs/s2d/iemocap/q1_probe.yaml")
    config["train"]["batch_size"] = 3
    write_json(run / "config.json", {
        "model": "L-B", "feature": "b", "probe_type": "linear", "seed": 100,
        "normalization": export.NORMALIZATION, "normalization_identity": banks.identity(norm),
        "protocol": config, "selection_split": "dev", "selection_metric": "weighted_f1",
    })
    provenance_dir = archive / "00_provenance"
    write_json(provenance_dir / "provenance.json", {
        "git_head": "synthetic_frozen_q1", "device": "cpu", "test_accessed": False})
    write_json(provenance_dir / "feature_bank_reference.json", {
        "metadata_identities": bank.metadata_identities, "shard_identities": bank.shard_identities})
    _, probabilities = runner.evaluate(
        model, (bank.values("dev", banks.FEATURES["b"], pooled=True)-mean)/std,
        bank.labels["dev"], 3, "cpu")
    rows = [dict(utterance_id=uid, true_label=int(bank.labels["dev"][i]),
                 pred_label=int(probabilities[i].argmax()),
                 **{f"prob_{c}": float(probabilities[i, c]) for c in range(6)})
            for i, uid in enumerate(bank.ids["dev"])]
    reports.atomic_csv(run / "predictions_dev.csv", rows, reports.PREDICTION_FIELDS)
    case = {
        "bank_dir": bank_dir, "bank": bank, "archive": archive, "run": run,
        "norm": norm, "mean": mean, "std": std, "model": model,
        "reference": run / "predictions_dev.csv",
        "sources": [run / "config.json", run / "best_model.pt", run / "predictions_dev.csv",
                    norm, provenance_dir / "provenance.json", provenance_dir / "feature_bank_reference.json"],
    }
    seal(case)
    return case


def run_export(case, output, split="train"):
    return export.export_predictions(case["bank_dir"], case["run"], split, output)


def test_frozen_normalization_math_strict_load_schema_and_unchanged_bytes(frozen, tmp_path, monkeypatch):
    monkeypatch.setattr(banks.FeatureBank, "normalization", forbidden)
    monkeypatch.setattr(runner, "fit_probe", forbidden)
    monkeypatch.setattr(torch.optim, "AdamW", forbidden)
    before = {p: banks.identity(p) for p in [*frozen["sources"], frozen["archive"] / "complete.json"]}
    real_load, real_strict = torch.load, LinearProbe.load_state_dict
    checkpoint_loads, strict_calls = [], []

    def checked_load(path, *args, **kwargs):
        assert kwargs["weights_only"] is True
        if Path(path) == frozen["run"] / "best_model.pt":
            checkpoint_loads.append(path)
        return real_load(path, *args, **kwargs)

    def checked_strict(self, state, *args, **kwargs):
        strict_calls.append(kwargs.get("strict"))
        return real_strict(self, state, *args, **kwargs)

    monkeypatch.setattr(torch, "load", checked_load)
    monkeypatch.setattr(LinearProbe, "load_state_dict", checked_strict)
    output = tmp_path / "train_export"
    manifest = run_export(frozen, output)
    rows = read_csv(output / "predictions_train.csv")
    bank = frozen["bank"]
    assert list(rows[0]) == list(reports.PREDICTION_FIELDS)
    assert [row["utterance_id"] for row in rows] == bank.ids["train"]
    assert [int(row["true_label"]) for row in rows] == bank.labels["train"].tolist()
    # Independent affine/linear calculation with the deliberately frozen stats.
    slots = bank.values("train", banks.FEATURES["b"])
    expected_x = (slots.mean(1) - frozen["mean"]) / frozen["std"]
    torch.testing.assert_close(((slots-frozen["mean"])/frozen["std"]).mean(1), expected_x)
    expected = torch.nn.functional.linear(
        expected_x, frozen["model"].classifier.weight, frozen["model"].classifier.bias).softmax(-1)
    actual = torch.tensor([[float(row[f"prob_{c}"]) for c in range(6)] for row in rows])
    torch.testing.assert_close(actual, expected, rtol=0, atol=1e-6)
    assert checkpoint_loads and strict_calls == [True]
    assert before == {p: banks.identity(p) for p in before}
    bank.assert_unchanged()
    assert manifest["schema"] == export.SCHEMA and manifest["row_count"] == 7
    assert manifest["probe"] == "Linear-B" and manifest["seed"] == 100
    assert manifest["dev_reproduction_status"] == "PASS" and manifest["dev_max_abs_prob_diff"] == 0
    assert manifest["normalization_sha256"] == before[frozen["norm"]]["sha256"]
    assert manifest["checkpoint_sha256"] == before[frozen["run"] / "best_model.pt"]["sha256"]
    assert manifest["in_sample_base_prediction"] is True and manifest["test_access"] is False
    assert "not claimed to be unbiased out-of-fold" in manifest["train_prediction_limitation"]
    assert manifest["normalization_contract"]["refitted"] is False
    assert banks.read_json(output / "prediction_export_manifest.json") == manifest
    assert manifest["prediction_identity"] == banks.identity(output / "predictions_train.csv")
    assert sorted(p.name for p in output.iterdir()) == [
        "prediction_export_manifest.json", "predictions_train.csv"]


def test_dev_cli_reproduces_reference_exactly(frozen, tmp_path, capsys):
    output = tmp_path / "dev_export"
    export.main(["--bank-dir", str(frozen["bank_dir"]), "--probe-run-dir", str(frozen["run"]),
                 "--split", "dev", "--output-dir", str(output)])
    assert "max_abs_prob_diff=0" in capsys.readouterr().out
    assert (output / "predictions_dev.csv").read_bytes() == frozen["reference"].read_bytes()
    assert banks.read_json(output / "prediction_export_manifest.json")["in_sample_base_prediction"] is False


@pytest.mark.parametrize("change,match", [
    ("duplicate", "unique"), ("id", "utterance_id"), ("order", "utterance_id"),
    ("label", "true_label"), ("prediction", "pred_label"), ("schema", "schema"),
    ("row_count", "row_count"), ("extra_cell", "schema"), ("missing_cell", "schema"),
    ("nan", "probabilities"), ("negative", "probabilities"),
    ("sum", "probabilities"), ("probability_difference", "probability mismatch"),
])
def test_dev_gate_aborts_before_train_inference_or_publication(frozen, tmp_path, monkeypatch, change, match):
    rows = read_csv(frozen["reference"])
    fields = reports.PREDICTION_FIELDS
    if change == "duplicate":
        rows[1]["utterance_id"] = rows[0]["utterance_id"]
    elif change == "id":
        rows[0]["utterance_id"] = "different_id"
    elif change == "order":
        rows[0], rows[1] = rows[1], rows[0]
    elif change == "label":
        rows[0]["true_label"] = (int(rows[0]["true_label"]) + 1) % 6
    elif change == "prediction":
        rows[0]["pred_label"] = (int(rows[0]["pred_label"]) + 1) % 6
    elif change == "schema":
        fields = fields[:-1]
        for row in rows:
            del row["prob_5"]
    elif change == "row_count":
        rows.pop()
    elif change == "nan":
        rows[0]["prob_0"] = "nan"
    elif change == "negative":
        rows[0]["prob_0"] = "-.01"
    elif change == "sum":
        rows[0]["prob_0"] = ".99"
    elif change == "probability_difference":
        rows[0]["prob_0"] = float(rows[0]["prob_0"]) + 1e-4
        rows[0]["prob_1"] = float(rows[0]["prob_1"]) - 1e-4
    frozen["reference"].unlink()
    reports.atomic_csv(frozen["reference"], rows, fields)
    if change in {"extra_cell", "missing_cell"}:
        lines = frozen["reference"].read_text(encoding="utf-8").splitlines()
        lines[1] = lines[1] + ",extra" if change == "extra_cell" else lines[1].rsplit(",", 1)[0]
        frozen["reference"].write_text("\n".join(lines) + "\n", encoding="utf-8")
    seal(frozen)
    real_values = banks.FeatureBank.values

    def dev_only(self, split, *args, **kwargs):
        assert split == "dev", "Train inference occurred after failed Dev gate"
        return real_values(self, split, *args, **kwargs)

    monkeypatch.setattr(banks.FeatureBank, "values", dev_only)
    output = tmp_path / "rejected_export"
    with pytest.raises(ValueError, match=match):
        run_export(frozen, output)
    assert not output.exists()


@pytest.mark.parametrize("change", ["extra", "missing", "wrong_shape"])
def test_checkpoint_strict_schema(frozen, tmp_path, change):
    path = frozen["run"] / "best_model.pt"
    state = torch.load(path, weights_only=True)
    if change == "extra":
        state["hidden.weight"] = torch.zeros(2, 2)
    elif change == "missing":
        del state["classifier.bias"]
    else:
        state["classifier.weight"] = torch.zeros(6, 767)
    torch.save(state, path)
    seal(frozen)
    with pytest.raises(RuntimeError):
        run_export(frozen, tmp_path / "strict_rejected")
    assert not (tmp_path / "strict_rejected").exists()


@pytest.mark.parametrize("field,value", [
    ("fit_split", "dev"), ("ddof", 1), ("epsilon", 1e-5), ("axes", ["utterance"]),
    ("field", banks.FEATURES["d0"]), ("count", 7), ("train_index_identity", {}),
    ("mean", [float("inf")] * 768), ("std", [0.] * 768),
])
def test_normalization_contract_fail_closed(frozen, tmp_path, field, value):
    stats = banks.read_json(frozen["norm"])
    stats[field] = value
    write_json(frozen["norm"], stats)
    config = banks.read_json(frozen["run"] / "config.json")
    config["normalization_identity"] = banks.identity(frozen["norm"])
    write_json(frozen["run"] / "config.json", config)
    seal(frozen)
    with pytest.raises(ValueError, match="normalization"):
        run_export(frozen, tmp_path / "bad_norm")


def test_missing_frozen_normalization_never_refits(frozen, tmp_path, monkeypatch):
    frozen["norm"].unlink()
    monkeypatch.setattr(banks.FeatureBank, "normalization", forbidden)
    with pytest.raises(FileNotFoundError):
        run_export(frozen, tmp_path / "missing_norm")


@pytest.mark.parametrize("name", ["checkpoint", "normalization", "bank"])
def test_frozen_artifact_identity_rejected(frozen, tmp_path, name):
    if name == "checkpoint":
        with (frozen["run"] / "best_model.pt").open("ab") as stream:
            stream.write(b"changed")
    elif name == "normalization":
        with frozen["norm"].open("a", encoding="utf-8") as stream:
            stream.write(" ")
    else:
        with (frozen["bank_dir"] / "provenance.json").open("a", encoding="utf-8") as stream:
            stream.write(" ")
    with pytest.raises(ValueError, match="identity"):
        run_export(frozen, tmp_path / "identity_rejected")


@pytest.mark.parametrize("split", ["test", "all", "validation_test", "Test", "train,dev"])
def test_split_fails_before_any_input_io(tmp_path, monkeypatch, split):
    monkeypatch.setattr(export, "resolve_path", forbidden)
    monkeypatch.setattr(export, "load_artifacts", forbidden)
    monkeypatch.setattr(banks, "FeatureBank", forbidden)
    monkeypatch.setattr(torch, "load", forbidden)
    monkeypatch.setattr(Path, "open", forbidden)
    with pytest.raises(ValueError, match="Only train/dev"):
        export.export_predictions("missing_bank", "missing_run", split, tmp_path / "absent")


@pytest.mark.parametrize("split", ["test", "all", "validation_test"])
def test_cli_split_fails_before_exporter(monkeypatch, split):
    monkeypatch.setattr(export, "export_predictions", forbidden)
    with pytest.raises(SystemExit) as error:
        export.main(["--bank-dir", "absent_bank", "--probe-run-dir", "absent_run",
                     "--output-dir", "absent_output", "--split", split])
    assert error.value.code == 2


@pytest.mark.parametrize("position", range(3))
@pytest.mark.parametrize("name", ["test", "bank_Test", "validation_test", "testing-data", r"X:\Test\bank"])
def test_test_named_paths_fail_before_io(monkeypatch, position, name):
    arguments = ["bank", "archive/03_linear/b/seed100", "output"]
    arguments[position] = name
    monkeypatch.setattr(export, "resolve_path", forbidden)
    monkeypatch.setattr(export, "load_artifacts", forbidden)
    monkeypatch.setattr(banks, "FeatureBank", forbidden)
    monkeypatch.setattr(Path, "open", forbidden)
    with pytest.raises(ValueError, match="Test-named"):
        export.export_predictions(arguments[0], arguments[1], "train", arguments[2])


def test_no_discovery_no_loaders_only_explicit_train_dev(frozen, tmp_path, monkeypatch):
    real_open, opened = Path.open, []

    def checked_open(path, *args, **kwargs):
        export.reject_test_path(path)
        opened.append(path)
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", checked_open)
    monkeypatch.setattr(Path, "glob", forbidden)
    monkeypatch.setattr(Path, "rglob", forbidden)
    monkeypatch.setattr(Path, "iterdir", forbidden)
    monkeypatch.setattr(torch.utils.data, "DataLoader", forbidden)
    run_export(frozen, tmp_path / "explicit_export")
    bank_opens = [p.relative_to(frozen["bank_dir"]) for p in opened if p.is_relative_to(frozen["bank_dir"])]
    assert any(p.parts[0] == "train" for p in bank_opens)
    assert any(p.parts[0] == "dev" for p in bank_opens)
    assert all(p.parts[0] in {"train", "dev", "complete.json", "coverage.json", "provenance.json"}
               for p in bank_opens)


def test_existing_output_and_overlapping_sources_refused_before_input_io(frozen, tmp_path, monkeypatch):
    existing = tmp_path / "existing"
    existing.mkdir()
    monkeypatch.setattr(export, "load_artifacts", forbidden)
    with pytest.raises(FileExistsError):
        run_export(frozen, existing)
    for directory in (frozen["bank_dir"], frozen["bank_dir"] / "new", frozen["run"] / "new", tmp_path):
        with pytest.raises(ValueError, match="disjoint"):
            run_export(frozen, directory)


def test_row_counts_are_fixed_and_wrong_bank_aborts(frozen, tmp_path, monkeypatch):
    monkeypatch.setattr(export, "EXPECTED_ROWS", {"train": 4818, "dev": 978})
    with pytest.raises(ValueError, match="row_count"):
        run_export(frozen, tmp_path / "wrong_rows")


def test_production_constants_no_cli_bypass():
    assert export.EXPECTED_ROWS == {"train": 4818, "dev": 978}
    assert export.PROBABILITY_TOLERANCE == 1e-6
    with pytest.raises(SystemExit):
        export.parse_args(["--bank-dir", "b", "--probe-run-dir", "r", "--split", "train",
                           "--output-dir", "o", "--skip-dev-gate"])


def test_duplicate_bank_ids_rejected(frozen, tmp_path):
    path = frozen["bank_dir"] / "train/index.json"
    index = banks.read_json(path)
    index["rows"][1]["utterance_id"] = index["rows"][0]["utterance_id"]
    write_json(path, index)
    with pytest.raises(ValueError, match="unique"):
        run_export(frozen, tmp_path / "duplicate_ids")


def test_checkpoint_change_during_inference_aborts_before_publication(frozen, tmp_path, monkeypatch):
    real_evaluate = export.evaluate

    def change_after_eval(*args, **kwargs):
        result = real_evaluate(*args, **kwargs)
        with (frozen["run"] / "best_model.pt").open("ab") as stream:
            stream.write(b"concurrent_change")
        return result

    monkeypatch.setattr(export, "evaluate", change_after_eval)
    output = tmp_path / "concurrent_rejected"
    with pytest.raises(ValueError, match="changed during export"):
        run_export(frozen, output)
    assert not output.exists()


def test_train_row_order_is_repeatable(frozen, tmp_path):
    first, second = tmp_path / "first", tmp_path / "second"
    run_export(frozen, first)
    run_export(frozen, second)
    assert (first / "predictions_train.csv").read_bytes() == (second / "predictions_train.csv").read_bytes()

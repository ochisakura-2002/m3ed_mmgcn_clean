"""Synthetic-only frozen TA/V cache, B1/B2 and Dev mechanism regression checks."""

import csv
import json
from pathlib import Path
import subprocess
import sys
from uuid import uuid4

import numpy as np
import pytest
import torch
import yaml

from models.asymmetric_merc.feature_fusion import ConcatLinear, VisualReliabilityGate, build_model
from scripts.analysis.models.asymmetric_merc.vrg_report import assign_quadrants, summarize_gates
from scripts.analysis.models.asymmetric_merc.weighted_fusion import (
    Joined, LABEL_NAMES, complementarity, load_predictions,
)
from scripts.models.asymmetric_merc import frozen_tav_cache as cache
from scripts.models.asymmetric_merc import train_frozen_tav as runtime
from scripts.models.asymmetric_merc.build_frozen_tav_cache import main as build_main

ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture
def tmp_path():
    # Use the existing workspace-temp convention to avoid managed Windows 0700 ACLs.
    path = ROOT / "tmp/assistant_work/frozen_tav_tests" / uuid4().hex
    path.mkdir(parents=True)
    return path


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _save_json(path, value):
    Path(path).write_text(json.dumps(value), encoding="utf-8")


def _rewrite_npz(path, **updates):
    with np.load(path, allow_pickle=False) as archive:
        data = dict(archive)
    data.update(updates)
    np.savez(path, **data)


def _fixture(tmp_path, *, shuffled=False):
    rng = np.random.default_rng(100)
    bank = tmp_path / "bank"
    bank.mkdir()
    provenance = {"schema_version": cache.BANK_SCHEMA, "splits": ["train", "dev"],
                  "feature_contract_version": "s2d_q1_a0_c0_d0_b_v1", "selected_epoch": 7,
                  "num_frames": 16, "dtype": "float32", "sampling_policy": "Uniform16", "shuffle": False,
                  "label_vocab": {"index_to_label": list(LABEL_NAMES), "target": "benchmark_label",
                                  "label_to_index": {name: k for k, name in enumerate(LABEL_NAMES)}}}
    _save_json(bank / "provenance.json", provenance)
    _save_json(bank / "complete.json", {"schema_version": cache.BANK_SCHEMA, "splits": ["train", "dev"],
                                       "status": "complete"})
    coverage, ta_paths, raw = {}, {}, {}
    for split, n in (("train", 9), ("dev", 7)):
        ids = np.array([f"{split}-{k:02d}" for k in range(n)])
        labels = np.arange(n) % 6
        pred = labels.copy()
        if split == "dev":
            pred[:6] = [0, 1, 3, 4, 4, 0]
        prob = np.full((n, 6), .05, dtype=np.float32)
        prob[np.arange(n), pred] = .75
        h_ta = rng.normal(size=(n, 1500)).astype(np.float32) + (50 if split == "dev" else 0)
        h_ta[-1] = 10000  # Excluded Train TA-only row must not enter normalization.
        values = {"utterance_id": ids, "dialogue_id": np.array([f"{split}-dialogue"] * n),
                  "label": labels, "h_ta": h_ta, "prob_ta": prob, "logits_ta": np.log(prob)}
        raw[split] = values
        order = rng.permutation(n) if shuffled else np.arange(n)
        path = tmp_path / f"{split}_h_ta.npz"
        np.savez(path, **{key: value[order] for key, value in values.items()}, split=np.array(split))
        ta_paths[split] = path
        directory = bank / split
        directory.mkdir()
        m = n - 1
        slots = torch.from_numpy(rng.normal(size=(m, 16, 768)).astype(np.float32))
        # Deliberately disagree with stored h_v: only B.mean is allowed to feed VRG.
        payload = {"schema_version": cache.BANK_SCHEMA, "split": split,
                   "utterance_ids": ids[:m].tolist(), "labels": torch.tensor(labels[:m]),
                   "s2d_contextual_slots": slots, "h_v": slots.mean(1) + 123,
                   "logits_v": torch.zeros(m, 6), "frame_indices": torch.arange(16).expand(m, 16)}
        for key in ("a0_rgb_mean", "c0_mcp_static_mean", "d0_static_residual_mean"):
            payload[key] = torch.zeros_like(slots)
        if shuffled:
            permutation = rng.permutation(m)
            payload = {key: ([value[k] for k in permutation] if key == "utterance_ids"
                             else value[permutation] if isinstance(value, torch.Tensor) else value)
                       for key, value in payload.items()}
        torch.save(payload, directory / "part-00000.pt")
        rows = [{"global_row": k, "utterance_id": uid, "shard": "part-00000.pt", "row_in_shard": k}
                for k, uid in enumerate(payload["utterance_ids"])]
        _save_json(directory / "index.json", {"schema_version": cache.BANK_SCHEMA, "split": split,
                                              "row_count": m, "rows": rows,
                                              "shards": [{"filename": "part-00000.pt", "row_count": m}]})
        coverage[split] = {"manifest_total": n, "v_available": m, "face_cache_usable": m,
                           "exported_count": m, "excluded_count": 1, "unknown_exclusion_count": 0,
                           "exclusions": {"v_unavailable": [str(ids[-1])], "cached_face_unusable": []}}
    _save_json(bank / "coverage.json", {"schema_version": cache.BANK_SCHEMA, "splits": coverage})
    probe_path = tmp_path / "linear_b_seed100.csv"
    with probe_path.open("w", encoding="utf-8", newline="") as handle:
        fields = ["utterance_id", "true_label", "pred_label", "split", *(f"prob_{k}" for k in range(6))]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        indices = list(reversed(range(6))) if shuffled else list(range(6))
        for i in indices:
            pred = [0, 2, 2, 4, 4, 5][i]
            writer.writerow({"utterance_id": f"dev-{i:02d}", "true_label": i, "pred_label": pred, "split": "dev",
                             **{f"prob_{k}": .75 if k == pred else .05 for k in range(6)}})
    return ta_paths, bank, probe_path, raw


def _build(tmp_path, *, with_probe=True, shuffled=False):
    ta, bank, probe, raw = _fixture(tmp_path, shuffled=shuffled)
    output = tmp_path / "cache"
    manifest = cache.build_cache(ta["train"], ta["dev"], bank, output,
                                 v_quadrant_predictions=probe if with_probe else None, synthetic=True)
    return output, manifest, (ta, bank, probe, raw)


def _args(root, output, model="vrg", extra=()):
    return runtime.build_parser().parse_args(["--cache-dir", str(root), "--output-dir", str(output),
                                              "--model", model, "--synthetic", "--smoke-epochs", "2", *extra])


def test_exact_id_join_shuffled_schema_and_b_mean_source(tmp_path):
    output, manifest, (_, bank, _, raw) = _build(tmp_path, shuffled=True)
    common, _, _, _ = cache.load_cache(output, synthetic=True)
    for split, n in (("train", 8), ("dev", 6)):
        data = common[split]
        np.testing.assert_array_equal(data["utterance_id"], [f"{split}-{i:02d}" for i in range(n)])
        np.testing.assert_array_equal(data["h_ta"], raw[split]["h_ta"][:n])
        payload = torch.load(bank / split / "part-00000.pt", weights_only=True)
        by_id = dict(zip(payload["utterance_ids"], payload["s2d_contextual_slots"].mean(1).numpy()))
        np.testing.assert_array_equal(data["h_v"], np.stack([by_id[uid] for uid in data["utterance_id"]]))
        assert manifest["joins"][split]["ta_only_ids"] == [f"{split}-{n:02d}"]
        assert manifest["joins"][split]["unknown_v_exclusion"] == 0
        assert data["h_ta"].shape == (n, 1500) and data["h_v"].shape == (n, 768)
    assert manifest["source_v"] == "S2D-B mean contextual slots"
    assert common["dev"]["v_probe_pred"].tolist() == [0, 2, 2, 4, 4, 5]
    assert "v_probe_pred" not in common["train"]


def test_join_is_invariant_to_independent_row_permutations():
    ta = {"utterance_id": np.array(["c", "a", "b", "missing"]), "label": np.array([2, 0, 1, 3]),
          "h_ta": np.array([[2], [0], [1], [99]])}
    visual = {"utterance_id": np.array(["b", "c", "a"]), "label": np.array([1, 2, 0]),
              "h_v": np.array([[11], [12], [10]])}
    first, _ = cache.join_features(ta, visual, ["missing"])
    ta = {key: value[[2, 0, 3, 1]] for key, value in ta.items()}
    visual = {key: value[[2, 1, 0]] for key, value in visual.items()}
    second, _ = cache.join_features(ta, visual, ["missing"])
    for key in first:
        np.testing.assert_array_equal(first[key], second[key])


@pytest.mark.parametrize("damage,message", [("duplicate", "duplicate IDs"), ("label", "label mismatch"),
                                            ("unknown", "Unknown V exclusion"), ("excluded", "unknown V exclusion")])
def test_invalid_join_fails_closed(tmp_path, damage, message):
    ta, bank, _, _ = _fixture(tmp_path)
    if damage == "duplicate":
        with np.load(ta["train"]) as archive:
            ids = archive["utterance_id"].copy()
        ids[1] = ids[0]
        _rewrite_npz(ta["train"], utterance_id=ids)
    elif damage == "label":
        path = bank / "train/part-00000.pt"
        payload = torch.load(path, weights_only=True)
        payload["labels"][0] = 5
        torch.save(payload, path)
    else:
        path = bank / "coverage.json"
        coverage = _json(path)
        if damage == "unknown":
            coverage["splits"]["train"]["unknown_exclusion_count"] = 1
        else:
            coverage["splits"]["train"]["exclusions"]["v_unavailable"] = ["unaccounted"]
        _save_json(path, coverage)
    with pytest.raises(ValueError, match=message):
        cache.build_cache(ta["train"], ta["dev"], bank, tmp_path / "failed", synthetic=True)
    assert not (tmp_path / "failed").exists()


def test_normalization_uses_only_train_common_and_dev_cannot_change_stats(tmp_path):
    output, manifest, (ta, bank, probe, raw) = _build(tmp_path)
    common, norm, _, _ = cache.load_cache(output, synthetic=True)
    np.testing.assert_allclose(norm["mu_ta"], raw["train"]["h_ta"][:-1].astype(np.float64).mean(0))
    np.testing.assert_allclose(norm["sigma_ta"], raw["train"]["h_ta"][:-1].astype(np.float64).std(0, ddof=0))
    assert manifest["normalization"]["fit_count"] == 8
    for z in cache.standardize(common["train"], norm):
        np.testing.assert_allclose(z.mean(0), 0, atol=1e-6)
        np.testing.assert_allclose(z.std(0), 1, atol=1e-6)
    _rewrite_npz(ta["dev"], h_ta=raw["dev"]["h_ta"] + 1000)
    shifted = tmp_path / "shifted"
    cache.build_cache(ta["train"], ta["dev"], bank, shifted, v_quadrant_predictions=probe, synthetic=True)
    _, new_norm, _, _ = cache.load_cache(shifted, synthetic=True)
    for key in norm:
        np.testing.assert_array_equal(norm[key], new_norm[key])
    constant = cache.fit_normalization({"h_ta": np.ones((2, 1500)), "h_v": np.ones((2, 768))})
    np.testing.assert_array_equal(constant["sigma_ta"], np.full(1500, 1e-6))


@pytest.mark.parametrize("cls,params", [(ConcatLinear, 13614), (VisualReliabilityGate, 759735)])
def test_models_shapes_finite_backward_and_exact_parameter_counts(cls, params):
    torch.manual_seed(100)
    model = cls()
    ta, visual = torch.randn(8, 1500), torch.randn(8, 768)
    if cls is VisualReliabilityGate:
        result = model(ta, visual, return_details=True)
        assert result["t"].shape == result["v"].shape == (8, 300)
        assert result["r"].shape == (8, 1)
        assert bool(torch.isfinite(result["r"]).all())
        assert bool(((result["r"] > 0) & (result["r"] < 1)).all())
        expected_r = model.gate(torch.cat((result["t"], result["v"], torch.abs(result["t"] - result["v"]),
                                           result["t"] * result["v"]), dim=-1))
        torch.testing.assert_close(result["logits"], model.classifier(result["t"] + expected_r * result["v"]))
        logits = result["logits"]
    else:
        logits = model(ta, visual)
        assert list(model.children()) == [model.classifier]
    assert logits.shape == (8, 6) and sum(p.numel() for p in model.parameters()) == params
    loss = torch.nn.functional.cross_entropy(logits, torch.arange(8) % 6)
    assert bool(torch.isfinite(loss))
    loss.backward()
    for parameter in model.parameters():
        assert parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
        assert float(parameter.grad.abs().sum()) > 0


def test_yaml_disabled_path_is_exact_b1():
    torch.manual_seed(100)
    baseline = ConcatLinear()
    torch.manual_seed(100)
    disabled = build_model(vrg_enabled=False)
    assert disabled.state_dict().keys() == baseline.state_dict().keys()
    for key, value in baseline.state_dict().items():
        torch.testing.assert_close(disabled.state_dict()[key], value, rtol=0, atol=0)
    assert isinstance(build_model(vrg_enabled=True), VisualReliabilityGate)


def test_quadrants_match_frozen_q2_and_known_gate_statistics():
    true = np.array([0, 1, 2, 3])
    ta, visual = np.array([0, 1, 0, 0]), np.array([0, 0, 2, 0])
    assert assign_quadrants(true, ta, visual).tolist() == ["Q11", "Q10", "Q01", "Q00"]
    joined = Joined(tuple("abcd"), true, ta, visual, np.eye(6)[ta], np.eye(6)[visual], {})
    counts, _ = complementarity(joined)
    report = summarize_gates(true, ta, visual, [.5, .2, .8, .4])
    assert report["all"]["mean"] == pytest.approx(.475)
    assert report["all"]["std"] == pytest.approx(np.std([.5, .2, .8, .4]))
    assert report["delta_rescue_harm"] == pytest.approx(.6)
    assert all(report["quadrants"][key]["count"] == counts[key.lower()] for key in report["quadrants"])
    assert report["true_classes"]["exc"] == {"count": 0, "mean": None, "std": None}
    empty = summarize_gates([0], [0], [0], [.3])
    assert empty["delta_rescue_harm"] is None
    negative = summarize_gates(true, ta, visual, [.5, .8, .2, .4])
    assert negative["delta_rescue_harm"] < 0 and "pass" not in negative
    saturated = summarize_gates(true, ta, visual, [0., 1., .2, .4])
    assert saturated["finite_precision_endpoint_count"] == 2
    assert saturated["quadrants"]["Q10"]["mean"] == 1.  # Preserve, never clip the observed gate.


def test_checkpoint_selection_ties():
    rows = [{"dev_weighted_f1": .5, "dev_loss": 2., "epoch": 1},
            {"dev_weighted_f1": .5, "dev_loss": 1., "epoch": 2},
            {"dev_weighted_f1": .5, "dev_loss": 1., "epoch": 3},
            {"dev_weighted_f1": .4, "dev_loss": .1, "epoch": 4}]
    assert min(rows, key=runtime.selection_key) is rows[1]


@pytest.mark.parametrize("model", ["concat_linear", "vrg"])
def test_synthetic_train_best_checkpoint_reload_and_artifacts(tmp_path, model):
    root, _, _ = _build(tmp_path)
    output = tmp_path / model
    manifest = runtime.run(_args(root, output, model))
    assert manifest["status"] == "complete" and manifest["completed_epochs"] == 2
    assert manifest["test_access"] is False and manifest["test_loader_built"] == manifest["test_evaluation_count"] == 0
    assert {"best_model.pt", "epoch_metrics.csv", "best_dev_predictions.csv", "best_dev_metrics.json",
            "resolved_config.yaml", "run_manifest.json"} <= {p.name for p in output.iterdir()}
    checkpoint = torch.load(output / "best_model.pt", weights_only=True)
    assert checkpoint["epoch"] == manifest["best_epoch"]
    predictions = load_predictions(output / "best_dev_predictions.csv")
    assert len(predictions.ids) == 6
    assert len(manifest["best_dev"]["per_class"]) == 6
    config = yaml.safe_load((output / "resolved_config.yaml").read_text(encoding="utf-8"))
    assert config["train"]["lr"] == 1e-3 and config["train"]["selection"] == runtime.FROZEN_TRAIN["selection"]
    common, norm, _, _ = cache.load_cache(root, synthetic=True)
    restored = build_model(vrg_enabled=model == "vrg")
    restored.load_state_dict(checkpoint["model"], strict=True)
    ta, visual = (torch.from_numpy(z) for z in cache.standardize(common["dev"], norm))
    with torch.no_grad():
        expected = torch.softmax(restored(ta, visual), dim=1).numpy()
    np.testing.assert_array_equal(predictions.probabilities, expected)
    if model == "vrg":
        with (output / "dev_gates.tsv").open(encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle, delimiter="\t"))
        assert len(rows) == 6 and [row["quadrant"] for row in rows] == ["Q11", "Q10", "Q01", "Q00", "Q11", "Q01"]
        with torch.no_grad():
            gates = restored(ta, visual, return_details=True)["r"].numpy().reshape(-1)
        np.testing.assert_array_equal([float(row["gate"]) for row in rows], gates)
        assert _json(output / "gate_mechanism.json")["all"]["count"] == 6
    else:
        assert not (output / "dev_gates.tsv").exists()
    with pytest.raises(FileExistsError):
        runtime.run(_args(root, output, model))


def test_no_original_logits_substitution_missing_probe_fails_before_training(tmp_path, monkeypatch):
    root, manifest, (_, _, probe, _) = _build(tmp_path, with_probe=False)
    assert manifest["v_quadrant_predictions"] is None
    monkeypatch.setattr(runtime, "build_model", lambda **kwargs: pytest.fail("Training/model creation must not happen"))
    with pytest.raises(ValueError, match="requires --v-quadrant-predictions"):
        runtime.run(_args(root, tmp_path / "missing_probe"))
    assert not (tmp_path / "missing_probe").exists()
    # The external reader is usable on the same common IDs without rebuilding experts.
    common, _, _, _ = cache.load_cache(root, synthetic=True)
    dev, provenance = cache.attach_probe(common["dev"], probe)
    assert provenance["source"] == cache.V_PRED_SOURCE and dev["v_probe_pred"].shape == (6,)


def test_patience_uses_full_selection_order_and_stops_on_epoch_ties(tmp_path, monkeypatch):
    root, _, _ = _build(tmp_path)
    actual_evaluate = runtime.evaluate
    calls = []

    def fixed_evaluate(model, loader, device):
        result, prob, gate = actual_evaluate(model, loader, device)
        # Epoch2 improves the tie-breaking loss; subsequent epochs tie exactly.
        calls.append(1)
        result.update(weighted_f1=.5, loss=2. if len(calls) == 1 else 1.)
        return result, prob, gate

    monkeypatch.setattr(runtime, "evaluate", fixed_evaluate)
    args = _args(root, tmp_path / "patience", "concat_linear")
    args.smoke_epochs = None
    manifest = runtime.run(args)
    assert manifest["best_epoch"] == 2 and manifest["completed_epochs"] == 12
    assert manifest["stopped_early"] is True


@pytest.mark.parametrize("damage", ["missing", "label", "test_split", "duplicate"])
def test_linear_b_predictions_require_exact_ids_labels_dev_and_uniqueness(tmp_path, damage):
    root, _, (_, _, probe, _) = _build(tmp_path, with_probe=False)
    common, _, _, _ = cache.load_cache(root, synthetic=True)
    with probe.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if damage == "missing":
        rows.pop()
    elif damage == "label":
        rows[0]["true_label"] = "5"
    elif damage == "test_split":
        rows[0]["split"] = "test"
    else:
        rows.append(rows[0].copy())
    with probe.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError):
        cache.attach_probe(common["dev"], probe)


@pytest.mark.parametrize("metadata,value", [("split", "test"), ("test_features", np.ones(1))])
def test_ta_forbidden_metadata_fail_closed(tmp_path, metadata, value):
    ta, _, _, _ = _fixture(tmp_path)
    _rewrite_npz(ta["dev"], **{metadata: np.array(value)})
    with pytest.raises(ValueError, match="Test|split"):
        cache.load_ta(ta["dev"], "dev")


def test_test_paths_and_split_rejected_before_any_input_open(tmp_path, monkeypatch):
    monkeypatch.setattr(np, "load", lambda *a, **k: pytest.fail("No NPZ IO allowed"))
    monkeypatch.setattr(torch, "load", lambda *a, **k: pytest.fail("No shard IO allowed"))
    for path in (tmp_path / "test_h_ta.npz", tmp_path / "test/bank", tmp_path / "TA-Test.csv"):
        with pytest.raises(ValueError, match="Test input"):
            cache.build_cache(tmp_path / "train.npz", path, tmp_path / "bank", tmp_path / "unused", synthetic=True)
    with pytest.raises(ValueError, match="Test is forbidden"):
        cache.load_ta(tmp_path / "nonexistent.npz", "test")
    with pytest.raises(SystemExit) as caught:
        runtime.build_parser().parse_args(["--cache-dir", "x", "--output-dir", "y", "--split", "test"])
    assert caught.value.code == 2
    assert not (tmp_path / "unused").exists()


@pytest.mark.parametrize("damage", ["splits", "index", "shard_path"])
def test_bank_test_references_fail_before_any_shard_open(tmp_path, monkeypatch, damage):
    _, bank, _, _ = _fixture(tmp_path)
    if damage == "splits":
        path = bank / "provenance.json"
        value = _json(path)
        value["splits"].append("test")
    else:
        path = bank / "dev/index.json"
        value = _json(path)
        if damage == "index":
            value["split"] = "test"
        else:
            value["shards"][0]["filename"] = "../test/part-00000.pt"
    _save_json(path, value)
    monkeypatch.setattr(torch, "load", lambda *a, **k: pytest.fail("No shard IO before both split indices validate"))
    with pytest.raises(ValueError, match="Test|shard filename"):
        cache.load_visual_bank(bank)


def test_cache_tamper_and_dev_statistics_rejected(tmp_path):
    root, _, _ = _build(tmp_path)
    stats = root / "normalization.npz"
    with np.load(stats) as archive:
        normalization = dict(archive)
    normalization["mu_ta"] += 1
    np.savez(stats, **normalization)
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        cache.load_cache(root, synthetic=True)
    manifest_path = root / "cache_manifest.json"
    manifest = _json(manifest_path)
    manifest["artifacts"][stats.name] = cache.identity(stats)["sha256"]
    _save_json(manifest_path, manifest)
    with pytest.raises(ValueError, match="Train-common-only"):
        cache.load_cache(root, synthetic=True)


@pytest.mark.parametrize("damage", ["test_field", "test_split", "source"])
def test_cache_rejects_test_payload_and_wrong_frozen_ta_source(tmp_path, damage):
    root, _, _ = _build(tmp_path)
    manifest_path = root / "cache_manifest.json"
    manifest = _json(manifest_path)
    path = root / "dev_common.npz"
    if damage == "test_field":
        _rewrite_npz(path, test_features=np.ones((2, 1500)))
    elif damage == "test_split":
        _rewrite_npz(path, split=np.array("test"))
    else:
        manifest["source_ta"]["model"] = "CHG-v0"
    manifest["artifacts"][path.name] = cache.identity(path)["sha256"]
    _save_json(manifest_path, manifest)
    with pytest.raises(ValueError, match="Test|source"):
        cache.load_cache(root, synthetic=True)


def test_runner_external_linear_b_predictions_and_protocol_rejection(tmp_path):
    root, _, (_, _, probe, _) = _build(tmp_path, with_probe=False)
    args = _args(root, tmp_path / "external_probe", extra=("--v-quadrant-predictions", str(probe)))
    assert runtime.run(args)["status"] == "complete"
    config = yaml.safe_load(runtime.DEFAULT_CONFIG.read_text(encoding="utf-8"))
    config["splits"] = ["train", "dev", "test"]
    path = tmp_path / "forbidden_config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    args.config = path
    with pytest.raises(ValueError, match="protocol mismatch"):
        runtime.load_protocol(args)
    config["splits"] = ["train", "dev"]
    config["train"]["lr"] = .01
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(ValueError, match="protocol mismatch"):
        runtime.load_protocol(args)


def test_real_counts_and_synthetic_protocol_isolation(tmp_path):
    ta, bank, probe, _ = _fixture(tmp_path)
    with pytest.raises(ValueError, match="coverage differs"):
        cache.build_cache(ta["train"], ta["dev"], bank, tmp_path / "real", v_quadrant_predictions=probe)
    root = tmp_path / "synthetic_cache"
    cache.build_cache(ta["train"], ta["dev"], bank, root, synthetic=True)
    with pytest.raises(ValueError, match="evidence-mode"):
        cache.load_cache(root, synthetic=False)
    args = _args(root, tmp_path / "invalid")
    args.synthetic = False
    with pytest.raises(ValueError, match="smoke-epochs"):
        runtime.load_protocol(args)


def test_cli_builder_and_train_accept_only_named_inputs(tmp_path):
    ta, bank, probe, _ = _fixture(tmp_path)
    output = tmp_path / "cli_cache"
    assert build_main(["--ta-train-npz", str(ta["train"]), "--ta-dev-npz", str(ta["dev"]),
                       "--s2d-bank-dir", str(bank), "--output-dir", str(output),
                       "--v-quadrant-predictions", str(probe), "--synthetic"]) == 0
    result = subprocess.run([sys.executable, str(ROOT / "scripts/models/asymmetric_merc/train_frozen_tav.py"),
                             "--cache-dir", str(output), "--output-dir", str(tmp_path / "cli_run"),
                             "--model", "vrg", "--seed", "100", "--synthetic", "--smoke-epochs", "1"],
                            cwd=ROOT, capture_output=True, text=True, encoding="utf-8")
    assert result.returncode == 0, result.stdout + result.stderr
    assert _json(tmp_path / "cli_run/run_manifest.json")["status"] == "complete"


def test_canonical_output_paths_without_real_outputs(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, "ROOT", tmp_path)
    with pytest.raises(ValueError, match="Run output"):
        cache.reserve_output(tmp_path / "outputs/runs/legacy", "run")
    with pytest.raises(ValueError, match="Cache output"):
        cache.reserve_output(tmp_path / "outputs/analysis/legacy", "cache")
    run = tmp_path / "outputs/20261112/frozen_tav/runs/synthetic_b1"
    assert cache.reserve_output(run, "run") == run
    batch = tmp_path / "outputs/20261112/frozen_tav/analysis/batches/synthetic/frozen_tav_cache"
    assert cache.reserve_output(batch, "cache") == batch

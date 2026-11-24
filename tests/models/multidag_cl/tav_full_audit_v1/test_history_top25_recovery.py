import csv
import json
from types import SimpleNamespace
from zipfile import ZipFile

import numpy as np
import pytest
import torch

from datasets.iemocap.tav_full_audit_dataset import DIMS
from models.multidag_cl.fusion.model import FusionTAV
from scripts.analysis.models.multidag_cl import failure_audit_inference as inference
from scripts.analysis.models.multidag_cl import pairwise_preflight_inputs as inputs
from scripts.analysis.models.multidag_cl import run_history_top25_recovery as recovery
from scripts.runtime.multidag_cl_fusion.spec import make_final_config
from scripts.runtime.multidag_cl_ta_v0.formal import sha256_file


@pytest.fixture
def case(tmp_path, monkeypatch):
    population, previous = [], {}
    shifts = 273
    for dialogue in range(20):
        did = f"Ses04F_impro{dialogue:02}"
        for position in range(49):
            speaker = "FM"[position % 2]
            key = did, speaker
            label = previous.get(key, 0)
            if key in previous and shifts:
                label = 1 - label
                shifts -= 1
            previous[key] = label
            population.append((f"{did}_{speaker}{position:03}", did, label))

    labels = np.array([label for _, _, label in population])
    full = labels.copy()
    errors = np.r_[0:70, 200:232, 245:317, 867:980]
    full[errors] = 1 - full[errors]
    # The first 245 rows have identical MAD and are the frozen U-top25.
    scale = np.where(np.arange(980) < 245, 1., 4.)
    logits = np.eye(6)[full] * scale[:, None]
    measured = {"FULL_HISTORY": logits.copy()}
    for offset, condition in enumerate(inference.HISTORY[1:]):
        restricted = full.copy()
        selected = np.arange(200 + offset, 500, 5)
        restricted[selected] = 1 - full[selected]
        if offset == 1:
            restricted[[275, 390]] = 1 - full[[275, 390]]
        measured[condition] = np.eye(6)[restricted] * scale[:, None]

    checkpoint = tmp_path / "f0-reference"
    checkpoint.write_bytes(b"synthetic reference")
    source = SimpleNamespace(population=tuple(population), logits=logits,
                             identity={**inputs.F0_IDENTITY, "checkpoint_sha256": sha256_file(checkpoint)},
                             config=make_final_config("F0", 100), checkpoint=checkpoint,
                             manifest={"best_dev_metrics": {}, "curriculum_membership_sha256": "frozen-curriculum",
                                       "batch_pairing": {"dev": {"batch_sha256": ["frozen-tape"]}}})
    state = SimpleNamespace(source=source, measured=measured, full=inputs.FrozenSplit(logits.copy(), source.population),
                            args=SimpleNamespace(frozen_runs=tmp_path, train_dev_package=tmp_path, round1_dir=tmp_path,
                                                 chunk_size=8, batch_id="synthetic"), calls=[])
    dataset, model = object(), object()

    def project(_, actual, *, splits):
        assert actual is source and splits == ("dev",)
        return {"dev": dataset}, {}

    def validate_population(datasets, actual, *, splits):
        assert actual is source and datasets == {"dev": dataset} and splits == ("dev",)
        return {"dev": {"utterances": 980}}

    def checkpoint_model(actual, curriculum, device, *, check_test_hold):
        assert actual is source and curriculum == "frozen-curriculum" and device == "cpu"
        check_test_hold(getattr(state, "checkpoint_hold", inputs.HOLD))
        return model

    def frozen_forward(actual, dev, split, *, device, batch_size, expected_batches):
        assert actual is model and dev is dataset and split == "dev"
        assert device == "cpu" and batch_size == 16 and expected_batches == ["frozen-tape"]
        state.calls.append("full")
        return state.full

    def evaluate(actual, dev, split, *, device, chunk_size, history_only, batch_size):
        assert actual is model and dev is dataset and split == "dev" and history_only
        assert device == "cpu" and chunk_size == 8 and batch_size == 16
        state.calls.append("history")
        return state.measured

    def forbidden(*_args, **_kwargs):
        pytest.fail("optimizer or training invoked")

    for name in ("Adam", "AdamW", "SGD"):
        monkeypatch.setattr(torch.optim, name, forbidden)
    monkeypatch.setattr(torch.nn.Module, "train", forbidden)
    monkeypatch.setattr(recovery, "read_f0", lambda _: source)
    monkeypatch.setattr(recovery, "load_projected_train_dev", project)
    monkeypatch.setattr(recovery, "validate_population", validate_population)
    monkeypatch.setattr(recovery, "read_round1_manifest", lambda *_: (
        {"outputs_sha256": {"AMBIGUITY_AUDIT.csv": "round1-mad"}}, "round1-manifest"))
    monkeypatch.setattr(recovery, "checked_round1_mad", lambda _path, _manifest, _meta, archived: (archived.mad, None))
    monkeypatch.setattr(recovery, "frozen_model", checkpoint_model)
    monkeypatch.setattr(recovery, "frozen_forward", frozen_forward)
    monkeypatch.setattr(recovery, "evaluate", evaluate)
    return state


def run(case, root):
    output, archive = root / "analysis", root / "analysis.zip"
    status = recovery.execute(case.args, output, archive, device="cpu", git={}, experiment_date="20261124")
    manifest = json.loads((output / recovery.MANIFEST).read_text(encoding="utf-8"))
    return status, output, archive, manifest


def read_rows(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def test_verified_bundle_preserves_frozen_rank_and_sample_schema(case, tmp_path):
    status, output, archive, manifest = run(case, tmp_path)
    assert status == 0 and case.calls == ["full", "history"]
    assert manifest["H_SUBGROUP_PROVENANCE"] == manifest["F0_IDENTITY"] == "PASS"
    assert all(manifest[key] == value for key, value in inputs.HOLD.items())
    assert manifest["TEST_TOUCHED"] == "NO"
    assert manifest["FORMAL_TRAIN_RUNS"] == manifest["OPTIMIZER_STEPS"] == 0
    assert {key: manifest[key] for key in recovery.ANCHORS} == recovery.ANCHORS
    with ZipFile(archive) as bundle:
        assert bundle.testzip() is None
        assert set(bundle.namelist()) == {recovery.HISTORY_CSV, recovery.H_CSV, recovery.UH_CSV,
                                        recovery.SHIFT_CSV, recovery.MANIFEST, recovery.REPORT}
    assert all(sha256_file(output / name) == digest for name, digest in manifest["outputs_sha256"].items())

    samples = read_rows(output / recovery.HISTORY_CSV)
    assert len(samples) == 5880
    assert {row["row_type"] for row in samples} == {"sample"}
    assert {(int(row["dev_index"]), row["condition"]) for row in samples} == {
        (i, condition) for i in range(980) for condition in inference.HISTORY}
    for row in samples:
        uid, did, label = case.source.population[int(row["dev_index"])]
        assert (row["utterance_id"], row["dialogue_id"], row["gold"]) == (uid, did, inputs.CLASS_ORDER[label])
    assert all(row["prediction"] == row["predicted_label"] and row["gold"] == row["true_label"] for row in samples)
    full = {row["dev_index"]: row["prediction"] for row in samples if row["is_full_history"] == "True"}
    assert len(full) == 980
    assert all((row["prediction"] != full[row["dev_index"]]) == (row["prediction_changed_vs_full"] == "True")
               for row in samples)

    h_rows = read_rows(output / recovery.H_CSV)
    assert len(h_rows) == 245
    assert [int(row["dev_index"]) for row in h_rows] == [275, 390, *[i for i in range(200, 445) if i not in (275, 390)]]
    assert [int(row["rank"]) for row in h_rows] == list(range(1, 246))
    assert all(int(row["history_change_count"]) == sum(row[f"{name}_changed"] == "True" for name in inference.HISTORY[1:])
               for row in h_rows)

    uh_rows = read_rows(output / recovery.UH_CSV)
    assert [int(row["dev_index"]) for row in uh_rows] == list(range(980))
    expected_groups = {(True, False): "U_ONLY", (False, True): "H_ONLY", (True, True): "U_INTERSECT_H",
                       (False, False): "NEITHER"}
    assert all(expected_groups[row["is_u_top25"] == "True", row["is_h_top25"] == "True"] == row["subgroup"]
               for row in uh_rows)
    assert manifest["uh_subgroup_counts"] == {"U_ONLY": 200, "H_ONLY": 200, "U_INTERSECT_H": 45, "NEITHER": 535}
    shift_rows = read_rows(output / recovery.SHIFT_CSV)
    assert [int(row["dev_index"]) for row in shift_rows] == list(range(980))
    assert {key: sum(row["shift_group"] == key for row in shift_rows) for key in ("SHIFT", "STABLE", "NO_HISTORY")} == {
        "SHIFT": 273, "STABLE": 667, "NO_HISTORY": 40}


@pytest.mark.parametrize("defect", ["logits", "argmax", "order", "utterance", "dialogue", "gold", "duplicate", "missing"])
def test_identity_rejection_stops_before_restricted_forward(case, tmp_path, defect):
    population, logits = list(case.full.population), case.full.logits.copy()
    uid, did, gold = population[0]
    if defect == "logits":
        logits[0] += 0.1
    elif defect == "argmax":
        case.source.logits[0] = [1e-5, 0., 0., 0., 0., 0.]
        logits[0] = [0., 1e-5, 0., 0., 0., 0.]
        assert np.allclose(logits, case.source.logits, rtol=1e-4, atol=5e-4)
    elif defect == "order":
        population[:2] = population[:2][::-1]
    elif defect == "utterance":
        population[0] = (uid + "0", did, gold)
    elif defect == "dialogue":
        population[0] = (uid, did + "0", gold)
    elif defect == "gold":
        population[0] = (uid, did, 1 - gold)
    elif defect == "duplicate":
        population[0] = population[1]
    else:
        population.pop()
    case.full = inputs.FrozenSplit(logits, tuple(population))
    status, output, archive, manifest = run(case, tmp_path)
    assert status == 2 and case.calls == ["full"]
    assert manifest["PROVENANCE_RECOVERY"] == "REVIEW_INVALID" and manifest["F0_IDENTITY"] == "FAIL"
    assert {path.name for path in output.iterdir()} == {recovery.MANIFEST, recovery.REPORT} and not archive.exists()


def test_restricted_pass_baseline_is_revalidated(case, tmp_path):
    case.measured["FULL_HISTORY"][0] += 0.1
    status, output, archive, manifest = run(case, tmp_path)
    assert status == 2 and case.calls == ["full", "history"]
    assert manifest["PROVENANCE_RECOVERY"] == "REVIEW_INVALID" and manifest["F0_IDENTITY"] == "FAIL"
    assert not (output / recovery.H_CSV).exists() and not archive.exists()


@pytest.mark.parametrize("defect", ["duplicate", "missing_row", "missing_condition"])
def test_history_sample_matrix_rejects_incomplete_or_duplicated_keys(case, defect):
    meta = recovery.metadata(case.source.population)
    samples = [{**row, "condition": condition, "predicted_label": "hap"}
               for row in meta for condition in inference.HISTORY]
    if defect == "duplicate":
        samples[0] = samples[len(inference.HISTORY)]
    elif defect == "missing_row":
        samples.pop()
    else:
        del case.measured[inference.HISTORY[-1]]
        with pytest.raises(ValueError, match="six history conditions"):
            recovery.recovery_tables(case.measured, meta, None, None)
        return
    with pytest.raises(ValueError, match="incomplete or duplicated|variants are incomplete"):
        inputs.history_predictions(samples, meta)


def test_anchor_mismatch_preserves_diagnostic_rows_and_blocks_membership(case, tmp_path):
    case.measured = dict.fromkeys(inference.HISTORY, case.source.logits)
    status, output, archive, manifest = run(case, tmp_path)
    assert status == 2 and manifest["F0_IDENTITY"] == "PASS"
    assert manifest["PROVENANCE_RECOVERY"] == manifest["H_TOP25_PROVENANCE"] == manifest["H_SUBGROUP_PROVENANCE"] == "FAIL"
    assert manifest["H_TOP25_F0_ERRORS"] == 102 and "expected 104" in manifest["reason"]
    assert manifest["history_change_count_histogram"] == [980, 0, 0, 0, 0, 0]
    assert (output / recovery.HISTORY_CSV).exists() and not (output / recovery.H_CSV).exists() and not archive.exists()


@pytest.mark.parametrize("counter", inputs.HOLD_COUNTERS)
def test_nonzero_hold_counter_blocks_any_forward(case, tmp_path, counter):
    case.checkpoint_hold = {**inputs.HOLD, counter: 1}
    status, output, archive, manifest = run(case, tmp_path)
    assert status == 2 and not case.calls
    assert manifest["PROVENANCE_RECOVERY"] == "REVIEW_INVALID" and manifest["TEST_TOUCHED"] == "YES"
    assert manifest["rejected_input_hold_evidence"] == {counter: 1}
    assert not (output / recovery.HISTORY_CSV).exists() and not archive.exists()


def test_certified_projection_constructs_only_dev(tmp_path, monkeypatch):
    identity = {"schema": inputs.SCHEMA, "split_counts": inputs.COUNTS,
                "feature_dimensions": inputs.FEATURE_DIMENSIONS, "sha256": {"asset": "0" * 64}}
    hashes = {inputs.ASSET: "1" * 64, inputs.MODEL_MANIFEST: "2" * 64}
    certificate = {**inputs.HOLD, "schema": inputs.PROJECTION_SCHEMA, "splits": list(inputs.SPLITS),
                   "source_data_identity": identity, "split_counts": inputs.TRAIN_DEV_COUNTS,
                   "sha256": hashes, "row_order_identity_sha256": "certified-order"}
    source = SimpleNamespace(manifest={"data_identity": identity})
    (tmp_path / inputs.ASSET).touch()
    asset = {"splits": {"train": ["a"], "dev": ["b"]}, "dialogues": {"a": {}, "b": {}}}
    calls = []

    def dataset(view, rows, *, split, expected_counts):
        assert view["splits"]["test"] == [] and expected_counts["test"]["utterances"] == 0
        assert all(row["split"] != "test" for row in rows)
        calls.append(split)
        return split

    monkeypatch.setattr(inputs, "read_json", lambda _: certificate)
    monkeypatch.setattr(inputs, "read_manifest", lambda _: [{"split": "train"}, {"split": "dev"}])
    monkeypatch.setattr(inputs, "identity_sha", lambda _: "certified-order")
    monkeypatch.setattr(inputs, "sha256_file", lambda path: hashes[path.name])
    monkeypatch.setattr(inputs.pickle, "load", lambda _: asset)
    monkeypatch.setattr(inputs, "TAVAuditDataset", dataset)
    datasets, _ = inputs.load_projected_train_dev(tmp_path, source, splits=("dev",))
    assert datasets == {"dev": "dev"} and calls == ["dev"]
    with pytest.raises(ValueError, match="projection views"):
        inputs.load_projected_train_dev(tmp_path, source, splits=("test",))
    assert calls == ["dev"]


def test_real_history_only_forward_calls_frozen_model_in_eval_mode(monkeypatch):
    def forbidden(*_args, **_kwargs):
        pytest.fail("modality masking invoked")

    monkeypatch.setattr(inference, "modality_logits", forbidden)
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        model = FusionTAV(make_final_config("F0", 100)).requires_grad_(False).eval()
        calls = []

        def check_mode(module, _args):
            assert torch.is_inference_mode_enabled() and not module.training
            assert all(not param.requires_grad for param in module.parameters())
            calls.append(True)

        model.register_forward_pre_hook(check_mode)
        sample = {"dialogue_id": "dev", "split": "dev", "labels": torch.tensor([0, 1, 0]),
                  "utterance_ids": ("dev_0", "dev_1", "dev_2"), "speaker_ids_int": torch.tensor([0, 1, 0]),
                  **{name: torch.zeros((3, dim)) for name, dim in DIMS.items()}}
        measured = inference.evaluate(model, [sample], "dev", device="cpu", chunk_size=2, history_only=True)
        assert tuple(measured) == inference.HISTORY and len(calls) == 1
        assert all(logits.shape == (3, 6) for logits in measured.values())
        with pytest.raises(ValueError, match="Test factory forbidden"):
            inference.evaluate(model, [], "test", device="cpu", chunk_size=2, history_only=True)
        assert len(calls) == 1
    finally:
        torch.set_num_threads(threads)

"""Small synthetic Dev files only; no checkpoints, features, or dataset loaders."""

import copy
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from uuid import uuid4

import numpy as np
import pytest
import yaml

from scripts.analysis.models.asymmetric_merc import run_q2_2_weighted_fusion as runtime
from scripts.analysis.models.asymmetric_merc.weighted_fusion import (
    ALPHA_GRID, EPSILON, LABEL_NAMES, SEEDS, AuditError, complementarity,
    evaluate_fixed_alpha, exact_join, fusion_scores, load_predictions,
    select_alpha, select_grid_row,
)

ROOT = Path(__file__).resolve().parents[3]
CLI = ROOT / "scripts/analysis/models/asymmetric_merc/run_q2_2_weighted_fusion.py"


@pytest.fixture
def tmp_path():
    # Ordinary workspace directories avoid the managed Windows ACL issue with
    # pytest's mode=0700 directories, and obey the task's only allowed temp root.
    path = ROOT / "tmp/assistant_work/q2_2_tests" / uuid4().hex
    path.mkdir(parents=True)
    return path


def _rows(predictions, *, alias="pred_label", extra=False):
    true = [0, 1, 2, 3, 4, 5, 0, 1]
    rows = []
    for i, (target, predicted) in enumerate(zip(true, predictions)):
        rows.append({"utterance_id": f"dev-{i:02}", "true_label": target,
                     alias: predicted,
                     **{f"prob_{k}": 0.75 if k == predicted else 0.05 for k in range(6)}})
    if extra:
        rows.append({**rows[2], "utterance_id": "ta-only"})
    return rows


def _write(path, rows, delimiter=","):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter=delimiter)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _inputs(tmp_path):
    ta = _write(tmp_path / "ta.tsv", _rows([0, 1, 0, 0, 0, 5, 2, 2], alias="predicted_label", extra=True), "\t")
    paths = {"ta-predictions": ta}
    for seed in SEEDS:
        paths[f"linear-b-seed{seed}"] = _write(tmp_path / f"v{seed}.csv", _rows([0, 2, 2, 0, 4, 5, 2, 1]))
    paths["original-s2d"] = _write(tmp_path / "original.csv", _rows([0, 2, 2, 0, 4, 5, 2, 1]))
    return paths


def _arguments(paths, output, synthetic=True):
    args = [part for flag, path in paths.items() for part in (f"--{flag}", str(path))]
    return [*args, "--output-dir", str(output), *(["--synthetic"] if synthetic else [])]


def _joined(tmp_path):
    paths = _inputs(tmp_path)
    return exact_join(load_predictions(paths["ta-predictions"]), load_predictions(paths["linear-b-seed100"]))


def _json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _csv(path):
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_exact_id_inner_join_not_row_alignment(tmp_path):
    paths = _inputs(tmp_path)
    visual = list(reversed(_csv(paths["linear-b-seed100"])))
    visual.append({**visual[0], "utterance_id": "v-only"})
    _write(paths["linear-b-seed100"], visual)
    joined = exact_join(load_predictions(paths["ta-predictions"]), load_predictions(paths["linear-b-seed100"]))
    assert joined.ids == tuple(f"dev-{i:02}" for i in range(8))
    assert joined.v_predicted.tolist() == [0, 2, 2, 0, 4, 5, 2, 1]
    assert joined.audit["ta_rows"] == joined.audit["v_rows"] == 9
    assert joined.audit["common_rows"] == 8
    assert joined.audit["ta_only_ids"] == ["ta-only"]
    assert joined.audit["v_only_ids"] == ["v-only"]
    assert joined.audit["duplicate_ids"] == {"ta": [], "v": []}
    assert joined.audit["label_mismatch"] == 0


def test_complementarity_known_quadrants_and_all_six_classes(tmp_path):
    summary, rows = complementarity(_joined(tmp_path))
    assert [summary[k] for k in ("q11", "q10", "q01", "q00")] == [2, 1, 3, 2]
    assert summary["rescue_rate"] == 3 / 5
    assert summary["oracle_accuracy"] == 6 / 8
    assert summary["oracle_gain"] == 3 / 8
    assert summary["disagreement_rate"] == 4 / 8
    assert summary["ta_accuracy"] == 3 / 8
    assert summary["v_accuracy"] == 5 / 8
    assert [row["class_name"] for row in rows] == list(LABEL_NAMES)
    assert rows[0]["support"] == 2 and rows[0]["rescued"] == 0
    assert rows[1]["ta_errors"] == rows[1]["rescued"] == 1
    assert rows[1]["ta_correct_v_wrong"] == 1
    assert rows[1]["rescue_rate"] == 1
    assert rows[5]["rescue_rate"] is None


@pytest.mark.parametrize("field,value,message", [
    ("utterance_id", "  ", "empty utterance_id"),
    ("true_label", 6, "invalid true_label"),
    ("true_label", -1, "invalid true_label"),
    ("true_label", "1.5", "invalid true_label"),
    ("pred_label", 8, "invalid pred_label"),
    ("pred_label", "NaN", "invalid pred_label"),
    ("prob_0", "nan", "non-finite"),
    ("prob_0", "inf", "non-finite"),
    ("prob_0", "-inf", "non-finite"),
    ("prob_0", "oops", "non-numeric"),
    ("prob_1", -0.01, "negative probability"),
    ("prob_0", 0.5, "probability sum"),
    ("pred_label", 1, "argmax != stored"),
])
def test_invalid_input_rejected(tmp_path, field, value, message):
    rows = _rows([0, 1, 2, 3, 4, 5, 0, 1])
    rows[0][field] = value
    path = _write(tmp_path / "invalid.csv", rows)
    with pytest.raises(AuditError, match=message):
        load_predictions(path)


def test_probability_sum_tolerance_is_absolute_and_not_normalization(tmp_path):
    rows = _rows([0, 1, 2, 3, 4, 5, 0, 1])
    rows[0]["prob_0"] += 0.000005
    table = load_predictions(_write(tmp_path / "within.csv", rows))
    assert table.probabilities[0].sum() == pytest.approx(1.000005)
    rows[0]["prob_0"] += 0.00002
    with pytest.raises(AuditError, match="probability sum"):
        load_predictions(_write(tmp_path / "outside.csv", rows))


def test_duplicate_ids_rejected_and_reported(tmp_path):
    rows = _rows([0, 1, 2, 3, 4, 5, 0, 1])
    rows.append(copy.deepcopy(rows[0]))
    with pytest.raises(AuditError, match="duplicate IDs") as caught:
        load_predictions(_write(tmp_path / "duplicate.csv", rows))
    assert caught.value.audit["duplicate_ids"] == ["dev-00"]


def test_label_mismatch_fails_closed_with_audit(tmp_path):
    paths = _inputs(tmp_path)
    rows = _csv(paths["linear-b-seed100"])
    rows[0]["true_label"] = 5
    _write(paths["linear-b-seed100"], rows)
    output = tmp_path / "result"
    state = runtime.run(runtime.build_parser().parse_args(_arguments(paths, output)))
    assert state["status"] == "FAIL"
    audit = _json(output / "provenance/exact_join_audit.json")["linear_b_seed100"]
    assert audit["label_mismatch"] == 1 and audit["label_mismatch_ids"] == ["dev-00"]
    assert not (output / "alpha_selection").exists()
    assert not (output / "fusion").exists()


def test_empty_intersection_rejected(tmp_path):
    paths = _inputs(tmp_path)
    rows = _csv(paths["linear-b-seed100"])
    for row in rows:
        row["utterance_id"] = "other-" + row["utterance_id"]
    _write(paths["linear-b-seed100"], rows)
    with pytest.raises(AuditError, match="empty overlap"):
        exact_join(load_predictions(paths["ta-predictions"]), load_predictions(paths["linear-b-seed100"]))


def test_schema_empty_file_alias_conflict_and_duplicate_header(tmp_path):
    path = tmp_path / "empty.csv"
    path.write_text("utterance_id,true_label,pred_label," + ",".join(f"prob_{k}" for k in range(6)) + "\n", encoding="utf-8")
    with pytest.raises(AuditError, match="empty prediction file"):
        load_predictions(path)
    rows = _rows([0, 1, 2, 3, 4, 5, 0, 1])
    for row in rows:
        row["predicted_label"] = (row["pred_label"] + 1) % 6
    with pytest.raises(AuditError, match="conflicting prediction aliases"):
        load_predictions(_write(path, rows))
    path.write_text("utterance_id,utterance_id\na,b\n", encoding="utf-8")
    with pytest.raises(AuditError, match="duplicate column"):
        load_predictions(path)
    path.write_text("utterance_id,true_label\na,0\n", encoding="utf-8")
    with pytest.raises(AuditError, match="missing required"):
        load_predictions(path)


def test_endpoint_equivalence_and_clipping(tmp_path):
    joined = _joined(tmp_path)
    # One-hot probabilities exercise log(0) protection and endpoint equivalence.
    joined.ta_probabilities[:] = np.eye(6)[joined.ta_predicted]
    joined.v_probabilities[:] = np.eye(6)[joined.v_predicted]
    assert np.isfinite(fusion_scores(joined, 0.5)).all()
    np.testing.assert_array_equal(fusion_scores(joined, 1).argmax(1), joined.ta_predicted)
    np.testing.assert_array_equal(fusion_scores(joined, 0).argmax(1), joined.v_predicted)
    for value in (-0.1, 1.1, float("nan")):
        with pytest.raises(ValueError, match="alpha"):
            fusion_scores(joined, value)


def test_weighted_log_formula_is_not_probability_average(tmp_path):
    joined = _joined(tmp_path)
    joined.ta_probabilities[0] = [.8, .18, .005, .005, .005, .005]
    joined.v_probabilities[0] = [.1, .7, .05, .05, .05, .05]
    actual = fusion_scores(joined, .4)[0]
    expected = .4 * np.log(joined.ta_probabilities[0]) + .6 * np.log(joined.v_probabilities[0])
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-15)
    assert not np.allclose(actual, np.log(.4 * joined.ta_probabilities[0] + .6 * joined.v_probabilities[0]))


def test_selection_deterministic_and_grid_fixed(tmp_path):
    joined = _joined(tmp_path)
    best, grid = select_alpha(joined)
    assert len(grid) == 101 and [row["alpha"] for row in grid] == list(ALPHA_GRID)
    assert (best, grid) == select_alpha(joined)
    maximum = max(row["weighted_f1"] for row in grid)
    assert maximum - best["weighted_f1"] <= 1e-12


def test_tie_prefers_ta_and_compares_to_global_max():
    assert select_grid_row([{"alpha": a, "weighted_f1": .6} for a in ALPHA_GRID])["alpha"] == 1
    # A chain of near-ties must not admit a candidate > tolerance from max.
    rows = [{"alpha": 0., "weighted_f1": 1.},
            {"alpha": .5, "weighted_f1": 1. - .8e-12},
            {"alpha": 1., "weighted_f1": 1. - 1.6e-12}]
    assert select_grid_row(rows)["alpha"] == .5


def test_no_rescue_denominator_and_absent_classes_remain_visible(tmp_path):
    rows = _rows([0, 1, 2, 3, 4, 5, 0, 1])[:1]
    table = load_predictions(_write(tmp_path / "one.csv", rows))
    joined = exact_join(table, table)
    summary, classes = complementarity(joined)
    assert summary["rescue_rate"] is None and classes[1]["support"] == 0
    result = evaluate_fixed_alpha(joined, 1)
    assert len(result["per_class"]) == 6
    assert result["per_class"][1]["fusion_f1"] == 0
    assert len(result["confusion_matrices"]) == 18
    assert result["metrics"]["fusion"]["macro_f1"] == pytest.approx(1 / 6)


def test_hap_with_zero_f1_is_still_present(tmp_path):
    table = load_predictions(_write(tmp_path / "no_hap_prediction.csv", _rows([1] * 8)))
    result = evaluate_fixed_alpha(exact_join(table, table), .5)
    hap = result["per_class"][0]
    assert hap["class_name"] == "hap" and hap["support"] == 2
    assert hap["ta_f1"] == hap["v_f1"] == hap["fusion_f1"] == 0


def test_cli_smoke_schema_provenance_single_selection_and_fixed_alpha(tmp_path, monkeypatch):
    paths = _inputs(tmp_path)
    calls = []
    actual_select = runtime.select_alpha

    def spy(joined):
        calls.append(joined)
        return actual_select(joined)

    monkeypatch.setattr(runtime, "select_alpha", spy)
    # Change another seed to ensure it is evaluated with the primary alpha.
    _write(paths["linear-b-seed101"], _rows([5, 5, 5, 5, 5, 5, 5, 5]))
    # Even a perfect reference must not influence alpha selection.
    _write(paths["original-s2d"], _rows([0, 1, 2, 3, 4, 5, 0, 1]))
    output = tmp_path / "smoke"
    assert runtime.main(_arguments(paths, output)) == 0
    assert len(calls) == 1
    selection = _json(output / "alpha_selection/alpha_selection.json")
    assert selection["source"] == "linear_b_seed100" and selection["selection_count"] == 1
    primary_join = exact_join(load_predictions(paths["ta-predictions"]), load_predictions(paths["linear-b-seed100"]))
    assert selection["alpha"] == actual_select(primary_join)[0]["alpha"]
    robust = _csv(output / "robustness/fixed_alpha_seed100_104.csv")
    assert [int(row["seed"]) for row in robust] == list(SEEDS)
    assert all(float(row["alpha"]) == selection["alpha"] for row in robust)
    for seed, row in zip(SEEDS, robust):
        joined = exact_join(load_predictions(paths["ta-predictions"]), load_predictions(paths[f"linear-b-seed{seed}"]))
        expected = evaluate_fixed_alpha(joined, selection["alpha"])["metrics"]
        for metric in ("accuracy", "weighted_f1", "macro_f1", "uar"):
            assert float(row[f"fusion_{metric}"]) == expected["fusion"][metric]
            assert float(row[f"delta_vs_ta_{metric}"]) == expected["delta_vs_ta"][metric]
    per_class = _csv(output / "fusion/seed100_per_class.csv")
    assert [row["class_name"] for row in per_class] == list(LABEL_NAMES)
    assert {f"{source}_{metric}" for source in ("ta", "v", "fusion")
            for metric in ("precision", "recall", "f1")} <= per_class[0].keys()
    assert len(_csv(output / "fusion/seed100_predictions.csv")) == 8
    assert len(_csv(output / "fusion/seed100_confusion_matrix.csv")) == 18
    assert len(_csv(output / "robustness/fixed_alpha_per_class.csv")) == 30
    assert len(_csv(output / "complementarity/original_s2d_reference_classwise.csv")) == 6
    manifest = _json(output / "provenance/input_manifest.json")
    assert manifest["git_head"] and isinstance(manifest["git_dirty_status"], str)
    assert manifest["runtime_timestamp_utc"] and manifest["runtime_source_sha256"]
    assert len(manifest["inputs"]) == 7
    for record in manifest["inputs"].values():
        assert record["sha256"] == hashlib.sha256(Path(record["path"]).read_bytes()).hexdigest()
    protocol = _json(output / "provenance/protocol.json")
    assert protocol["epsilon"] == EPSILON and protocol["alpha_grid"] == list(ALPHA_GRID)
    for artifact in (manifest, protocol, _json(output / "provenance/run_status.json")):
        assert artifact["test_access"] is False
        assert artifact["test_loader_built"] == artifact["test_evaluation_count"] == 0
    report = (output / "reports/Q2_2_WEIGHTED_FUSION_REPORT.md").read_text(encoding="utf-8")
    assert "SYNTHETIC ONLY" in report and "LIMITATION: Alpha was selected on Dev" in report
    assert "Q2_2_DEV_SIGNAL = POSITIVE" not in report


def test_row_permutations_leave_every_numeric_artifact_unchanged(tmp_path):
    paths = _inputs(tmp_path)
    first, second = tmp_path / "first", tmp_path / "second"
    assert runtime.main(_arguments(paths, first)) == 0
    for path in paths.values():
        rows = _csv(path) if path.suffix == ".csv" else None
        if rows is None:
            with path.open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle, delimiter="\t"))
        _write(path, list(reversed(rows)), "\t" if path.suffix == ".tsv" else ",")
    assert runtime.main(_arguments(paths, second)) == 0
    for category in ("complementarity", "alpha_selection", "fusion", "robustness"):
        for file in (first / category).iterdir():
            assert file.read_bytes() == (second / category / file.name).read_bytes(), file.name


def test_clipping_counts_full_input_and_joined_subset(tmp_path):
    paths = _inputs(tmp_path)
    for flag, path in paths.items():
        alias = "predicted_label" if flag == "ta-predictions" else "pred_label"
        rows = _rows([0, 1, 2, 3, 4, 5, 0, 1], alias=alias, extra=flag == "ta-predictions")
        for row in rows:
            for k in range(6):
                row[f"prob_{k}"] = 1 if k == row[alias] else 0
        _write(path, rows)
    output = tmp_path / "result"
    assert runtime.main(_arguments(paths, output)) == 0
    manifest = _json(output / "provenance/input_manifest.json")
    audit = _json(output / "provenance/exact_join_audit.json")["linear_b_seed100"]
    assert manifest["inputs"]["ta"]["probability_clipping_count"] == 45
    assert audit["ta_probability_clipping_count"] == audit["v_probability_clipping_count"] == 40


def test_real_mode_mismatch_stops_before_robustness_and_preserves_report(tmp_path):
    paths = _inputs(tmp_path)
    output = tmp_path / "mismatch"
    assert runtime.main(_arguments(paths, output, synthetic=False)) == 1
    state = _json(output / "provenance/run_status.json")
    assert state["status"] == "FAIL" and "MISMATCH" in state["error"]
    checks = _json(output / "provenance/regression_audit.json")["checks"]
    assert any(row["key"] == "selection.alpha" and not row["pass"] for row in checks)
    assert (output / "alpha_selection/alpha_grid_seed100.csv").exists()
    assert not (output / "robustness").exists() and not (output / "fusion").exists()
    assert "STOP / REPORT MISMATCH" in (output / "reports/Q2_2_WEIGHTED_FUSION_REPORT.md").read_text(encoding="utf-8")


def test_existing_output_is_never_overwritten(tmp_path):
    paths = _inputs(tmp_path)
    output = tmp_path / "existing"
    output.mkdir()
    sentinel = output / "keep.txt"
    sentinel.write_text("preserve", encoding="utf-8")
    assert runtime.main(_arguments(paths, output)) == 1
    assert sentinel.read_text(encoding="utf-8") == "preserve"
    assert list(output.iterdir()) == [sentinel]


def test_invalid_input_failure_keeps_manifest_and_no_metrics(tmp_path):
    paths = _inputs(tmp_path)
    rows = _csv(paths["linear-b-seed100"])
    rows.append(rows[0])
    _write(paths["linear-b-seed100"], rows)
    output = tmp_path / "failure"
    assert runtime.main(_arguments(paths, output)) == 1
    manifest = _json(output / "provenance/input_manifest.json")
    assert manifest["inputs"]["linear_b_seed100"]["duplicate_ids"] == ["dev-00"]
    assert manifest["inputs"]["linear_b_seed101"]["status"] == "NOT_READ"
    assert not (output / "fusion").exists()


def test_optional_split_column_must_affirm_dev(tmp_path):
    rows = _rows([0, 1, 2, 3, 4, 5, 0, 1])
    for row in rows:
        row["split"] = "train"  # Synthetic invalid metadata, never a real split.
    with pytest.raises(AuditError, match="non-Dev split"):
        load_predictions(_write(tmp_path / "invalid_split.csv", rows))
    for row in rows:
        row["split"] = "dev"
    assert load_predictions(_write(tmp_path / "valid_split.csv", rows)).manifest["split_evidence"] == "validated_column"


def test_cli_has_no_test_or_training_inputs_and_imports_no_model_stack(tmp_path):
    parser = runtime.build_parser()
    options = {flag for action in parser._actions for flag in action.option_strings}
    assert options == {"-h", "--help", "--ta-predictions", "--original-s2d", "--output-dir", "--synthetic",
                       *(f"--linear-b-seed{seed}" for seed in SEEDS)}
    with pytest.raises(SystemExit):
        parser.parse_args([*_arguments(_inputs(tmp_path), tmp_path / "never"), "--test-predictions", "forbidden"])
    code = ("import sys; from scripts.analysis.models.asymmetric_merc import run_q2_2_weighted_fusion; "
            "assert not any(n.split('.')[0] in {'torch','models','datasets','trainers'} for n in sys.modules)")
    result = subprocess.run([sys.executable, "-B", "-c", code], cwd=ROOT,
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_cli_subprocess_accepts_all_seven_explicit_inputs(tmp_path):
    paths = _inputs(tmp_path)
    result = subprocess.run([sys.executable, "-B", str(CLI), *_arguments(paths, tmp_path / "subprocess")],
                            cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "Q2_2_RUNTIME=PASS" in result.stdout


def test_default_reference_targets_are_checks_only():
    checks = runtime._regressions({"alpha": .9}, {"alpha": .88}, 5e-6)
    assert checks == [{"key": "alpha", "expected": .88, "actual": .9, "pass": False}]


def _matching_fixture_reference(paths):
    """Build post-hoc targets for artificial files, never real Dev inputs."""
    reference = yaml.safe_load(runtime.REFERENCE.read_text(encoding="utf-8"))
    ta = load_predictions(paths["ta-predictions"])
    joined = exact_join(ta, load_predictions(paths["linear-b-seed100"]))
    summary, classes = complementarity(joined)
    selection, _ = select_alpha(joined)
    primary = reference["primary"]
    primary["join"] = {key: joined.audit[key] for key in primary["join"]}
    primary["complementarity"] = {key: summary[key] for key in primary["complementarity"]}
    primary["classwise"] = {row["class_name"]: {key: row[key] for key in ("rescued", "ta_errors")}
                            for row in classes}
    primary["ta_full_dev"] = runtime.metrics(ta.true, ta.predicted)
    primary["ta_common_dev"] = runtime.metrics(joined.true, joined.ta_predicted)
    primary["selection"] = selection
    primary["delta_weighted_f1"] = selection["weighted_f1"] - summary["ta_weighted_f1"]
    # These fixtures use identical visual predictions for all five seeds.
    reference["robustness"]["positive_weighted_f1_seeds"] = (
        len(SEEDS) if primary["delta_weighted_f1"] > 0 else 0)
    return reference


def _use_fixture_reference(tmp_path, monkeypatch, reference):
    path = tmp_path / "fixture_reference.yaml"
    path.write_text(yaml.safe_dump(reference), encoding="utf-8")
    monkeypatch.setattr(runtime, "REFERENCE", path)


def test_frozen_reference_has_six_classes_fusion_macro_and_distinct_ta_populations():
    primary = yaml.safe_load(runtime.REFERENCE.read_text(encoding="utf-8"))["primary"]
    assert list(primary["classwise"]) == list(LABEL_NAMES)
    assert primary["classwise"] == {
        "hap": {"rescued": 0, "ta_errors": 60},
        "sad": {"rescued": 3, "ta_errors": 21},
        "neu": {"rescued": 25, "ta_errors": 82},
        "ang": {"rescued": 13, "ta_errors": 68},
        "exc": {"rescued": 22, "ta_errors": 62},
        "fru": {"rescued": 63, "ta_errors": 118},
    }
    assert primary["selection"] == {
        "alpha": .88, "weighted_f1": .570588, "macro_f1": .506756,
        "uar": .524482, "accuracy": .592025,
    }
    assert primary["join"]["ta_rows"] == 980
    assert primary["join"]["common_rows"] == 978
    assert primary["ta_full_dev"] == {
        "accuracy": .580612, "weighted_f1": .559320, "macro_f1": .494189, "uar": .513883,
    }
    assert primary["ta_common_dev"] == {
        "accuracy": .579755, "weighted_f1": .558461, "macro_f1": .493549, "uar": .513336,
    }


def test_formal_regression_audits_actual_six_classes_and_all_metrics(tmp_path, monkeypatch):
    paths = _inputs(tmp_path)
    reference = _matching_fixture_reference(paths)
    _use_fixture_reference(tmp_path, monkeypatch, reference)
    output = tmp_path / "matching"
    assert runtime.main(_arguments(paths, output, synthetic=False)) == 0
    audit = _json(output / "provenance/regression_audit.json")
    assert audit["status"] == "PASS"
    checks = {row["key"]: row for row in audit["checks"]}
    required = {
        "selection.alpha",
        *(f"classwise.{name}.{key}" for name in LABEL_NAMES for key in ("rescued", "ta_errors")),
        *(f"{group}.{key}" for group in ("selection", "ta_full_dev", "ta_common_dev")
          for key in ("accuracy", "weighted_f1", "macro_f1", "uar")),
    }
    assert required <= checks.keys()
    assert all(row["pass"] for row in checks.values())
    state = _json(output / "provenance/run_status.json")
    assert state["primary"]["support"] == 8
    assert checks["join.ta_rows"]["actual"] == 9
    for metric in ("accuracy", "weighted_f1", "macro_f1", "uar"):
        assert checks[f"ta_common_dev.{metric}"]["actual"] == state["primary"][f"ta_{metric}"]
        assert checks[f"ta_full_dev.{metric}"]["actual"] == state["ta_full_dev"][metric]
    assert reference["primary"]["ta_common_dev"] != reference["primary"]["ta_full_dev"]
    assert checks["selection.macro_f1"]["actual"] == state["selection"]["macro_f1"]


@pytest.mark.parametrize("key", [
    "classwise.neu.rescued", "classwise.neu.ta_errors", "selection.macro_f1",
    "ta_common_dev.accuracy", "ta_common_dev.weighted_f1",
    "ta_common_dev.macro_f1", "ta_common_dev.uar",
])
@pytest.mark.parametrize("damage", ["missing_reference", "mismatched_reference"])
def test_required_regression_reference_damage_fails_closed(tmp_path, monkeypatch, key, damage):
    paths = _inputs(tmp_path)
    reference = _matching_fixture_reference(paths)
    target = reference["primary"]
    parts = key.split(".")
    for part in parts[:-1]:
        target = target[part]
    original = target[parts[-1]]
    if damage == "missing_reference":
        del target[parts[-1]]
    else:
        target[parts[-1]] += 1
    _use_fixture_reference(tmp_path, monkeypatch, reference)
    output = tmp_path / "damaged"
    assert runtime.main(_arguments(paths, output, synthetic=False)) == 1
    audit = _json(output / "provenance/regression_audit.json")
    assert audit["status"] == "FAIL_OR_INCOMPLETE"
    failed = [row for row in audit["checks"] if row["key"] == key and not row["pass"]]
    assert len(failed) == 1
    if damage == "missing_reference":
        assert "missing required" in failed[0]["error"]
        assert failed[0]["actual"] == original
    assert (output / "alpha_selection/alpha_grid_seed100.csv").is_file()
    assert not (output / "robustness").exists() and not (output / "fusion").exists()


@pytest.mark.parametrize("group", ["classwise.neu", "ta_common_dev"])
def test_missing_required_regression_group_fails_audit(tmp_path, monkeypatch, group):
    paths = _inputs(tmp_path)
    reference = _matching_fixture_reference(paths)
    if group == "classwise.neu":
        del reference["primary"]["classwise"]["neu"]
    else:
        del reference["primary"][group]
    _use_fixture_reference(tmp_path, monkeypatch, reference)
    output = tmp_path / "missing_group"
    assert runtime.main(_arguments(paths, output, synthetic=False)) == 1
    audit = _json(output / "provenance/regression_audit.json")
    assert audit["status"] == "FAIL_OR_INCOMPLETE"
    assert audit["checks"] and all(not row["pass"] and row["key"].startswith(group + ".")
                                   for row in audit["checks"])


@pytest.mark.parametrize("metric", ["accuracy", "weighted_f1", "macro_f1", "uar"])
def test_missing_ta_common_observation_has_no_reference_fallback(tmp_path, metric):
    reference = _matching_fixture_reference(_inputs(tmp_path))
    actual = copy.deepcopy(reference["primary"])
    del actual["ta_common_dev"][metric]
    checks = runtime._primary_regressions(actual, reference["primary"], reference["metric_atol"])
    failed = [row for row in checks if not row["pass"]]
    assert failed == [{"key": f"ta_common_dev.{metric}", "expected": reference["primary"]["ta_common_dev"][metric],
                       "actual": None, "pass": False}]


@pytest.mark.parametrize("group,other", [("ta_full_dev", "ta_common_dev"), ("ta_common_dev", "ta_full_dev")])
def test_ta_full_and_common_regression_references_cannot_be_substituted(tmp_path, monkeypatch, group, other):
    paths = _inputs(tmp_path)
    reference = _matching_fixture_reference(paths)
    expected_failures = {f"{group}.{metric}" for metric in ("accuracy", "weighted_f1", "macro_f1", "uar")
                         if reference["primary"][group][metric] != reference["primary"][other][metric]}
    assert f"{group}.accuracy" in expected_failures
    reference["primary"][group] = copy.deepcopy(reference["primary"][other])
    _use_fixture_reference(tmp_path, monkeypatch, reference)
    output = tmp_path / "confused_populations"
    assert runtime.main(_arguments(paths, output, synthetic=False)) == 1
    checks = _json(output / "provenance/regression_audit.json")["checks"]
    assert {row["key"] for row in checks if not row["pass"]} == expected_failures


@pytest.mark.parametrize("expected_alpha", [0., .88, 1.])
def test_frozen_regression_alpha_never_drives_selection(tmp_path, monkeypatch, expected_alpha):
    paths = _inputs(tmp_path)
    reference = _matching_fixture_reference(paths)
    joined = exact_join(load_predictions(paths["ta-predictions"]), load_predictions(paths["linear-b-seed100"]))
    actual_selection, actual_grid = select_alpha(joined)
    assert actual_selection["alpha"] != expected_alpha
    reference["primary"]["selection"]["alpha"] = expected_alpha
    _use_fixture_reference(tmp_path, monkeypatch, reference)
    calls = []
    actual_select = runtime.select_alpha

    def spy(joined):
        calls.append(joined)
        return actual_select(joined)

    monkeypatch.setattr(runtime, "select_alpha", spy)
    output = tmp_path / "post_hoc_only"
    assert runtime.main(_arguments(paths, output, synthetic=False)) == 1
    assert len(calls) == 1
    assert _json(output / "alpha_selection/alpha_selection.json")["alpha"] == actual_selection["alpha"]
    saved_grid = _csv(output / "alpha_selection/alpha_grid_seed100.csv")
    assert [{key: float(value) for key, value in row.items()} for row in saved_grid] == actual_grid
    failed = [row for row in _json(output / "provenance/regression_audit.json")["checks"] if not row["pass"]]
    assert failed == [{"key": "selection.alpha", "expected": expected_alpha,
                       "actual": actual_selection["alpha"], "pass": False}]
    assert not (output / "robustness").exists()


def test_canonical_outputs_layout_without_writing_real_outputs(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime, "PROJECT_ROOT", tmp_path)
    with pytest.raises(ValueError, match="analysis/batches"):
        runtime._reserve_output(tmp_path / "outputs/analysis/old_layout")
    valid = tmp_path / "outputs/20261111/q2_2_weighted_fusion/analysis/batches/synthetic"
    assert runtime._reserve_output(valid) == valid

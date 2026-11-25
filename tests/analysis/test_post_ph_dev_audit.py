"""Regression tests for Dev-only frozen audit metrics and artifact joins."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from scripts.analysis.models.multidag_cl.post_ph import dev_audit, dev_inputs, metrics


def test_probability_scores():
    labels = np.array([0, 1])
    probabilities = np.array([[0.8, 0.2], [0.7, 0.3]])
    scores = metrics.probability_details(probabilities, labels)
    assert scores["nll"] == pytest.approx([-np.log(0.8), -np.log(0.3)])
    assert scores["brier"] == pytest.approx([0.08, 0.98])
    assert scores["margin"] == pytest.approx([0.6, 0.4])
    assert scores["mad"] == pytest.approx([0.4, 0.6])
    assert scores["confidence"] == pytest.approx([0.8, 0.7])
    assert scores["entropy"][1] > scores["entropy"][0]
    assert 1 - scores["confidence"][1] > 1 - scores["confidence"][0]
    assert -scores["margin"][1] > -scores["margin"][0]


def test_uniform_and_equal_frequency_ece():
    confidence = np.array([0.4, 0.5, 0.8, 1.0])
    correct = np.array([0, 1, 0, 1], dtype=bool)
    uniform, rows = metrics.reliability(confidence, correct, 2)
    adaptive, adaptive_rows = metrics.reliability(confidence, correct, 2, adaptive=True)
    assert uniform == pytest.approx(0.175)
    assert adaptive == pytest.approx(0.225)
    assert [row["support"] for row in rows] == [1, 3]
    assert [row["support"] for row in adaptive_rows] == [2, 2]
    assert sum(row["ece_contribution"] for row in rows) == uniform
    assert metrics.reliability(np.array([]), np.array([]), 2) == (None, [])


def test_error_detection_positive_class_is_error():
    wrong = np.array([False, True, True, False])
    risk = np.array([0.2, 0.8, 0.6, 0.1])
    values, curve = metrics.error_detection(wrong, risk)
    assert values == {"auroc": 1, "auprc": 1}
    inverse, _ = metrics.error_detection(wrong, -risk)
    assert inverse == pytest.approx({"auroc": 0, "auprc": (1 / 3 + 2 / 4) / 2})
    assert {point["curve"] for point in curve} == {"ROC", "PR"}
    assert metrics.error_detection(np.zeros(4, dtype=bool), risk)[0]["auroc"] is None


def test_coverage_oracle_excess_and_ties():
    labels = np.array([0, 1, 0, 1])
    predictions = np.array([0, 1, 1, 0])
    summary, raw, fixed = metrics.selective_curve(labels, predictions, np.arange(4), 2)
    assert [row["risk"] for row in raw] == pytest.approx([0, 0, 1 / 3, 1 / 2])
    assert summary["aurc"] == pytest.approx(5 / 24)
    assert summary["excess_aurc"] == 0
    assert fixed[0]["coverage"] == 1
    assert fixed[-1]["coverage"] == 0.5
    assert [row["accepted"] for row in fixed] == [4, 3, 3, 3, 2, 2, 2]
    tied, raw, _ = metrics.selective_curve(labels, predictions, np.zeros(4), 2)
    assert [row["tie_expected_risk"] for row in raw] == [0.5] * 4
    assert tied["tie_expected_excess_aurc"] == pytest.approx(1 / 2 - 5 / 24)


def test_seed_votes_predictive_entropy_and_majority_ties():
    labels = np.array([0, 1])
    predictions = np.array([[0, 0], [0, 2], [0, 3], [0, 4], [0, 5]])
    probabilities = np.eye(len(dev_inputs.CLASSES))[predictions]
    out = metrics.seed_stability(predictions, probabilities, labels)
    assert out["correct_count"].tolist() == [5, 0]
    assert out["prediction_agreement"] == pytest.approx([1, 0.2])
    assert out["seed_disagreement"] == pytest.approx([0, 0.8])
    assert out["majority_tie"].tolist() == [False, True]
    assert out["majority_class"].tolist() == [0, 0]
    assert out["prediction_entropy"] == pytest.approx([0, np.log(5)])
    assert out["seed_predictive_entropy"] == pytest.approx(out["prediction_entropy"])
    assert out["confidence_variance"].tolist() == [0, 0]


@pytest.fixture
def toy_dev():
    dialogue = "Ses04M_impro01"
    identity = tuple((f"{dialogue}_{speaker}{i:03}", dialogue, label)
                     for i, (speaker, label) in enumerate((("M", "hap"), ("F", "sad"), ("M", "hap"), ("F", "neu"))))
    labels = np.array([0, 1, 0, 2])
    predictions = np.tile(labels, (len(dev_inputs.SEEDS), 1))
    return dev_inputs.DevEvidence(tuple(row[0] for row in identity), tuple(row[1] for row in identity),
                                  labels, predictions, np.eye(len(dev_inputs.CLASSES))[predictions],
                                  dev_inputs.sample_metadata(identity), {})


def test_subgroup_support_and_previous_same_speaker(toy_dev):
    assert [row["shift_group"] for row in toy_dev.metadata] == ["NO_HISTORY", "NO_HISTORY", "STABLE", "SHIFT"]
    groups = dev_audit.group_map(toy_dev, None)
    assert groups["speaker"]["Ses04F"].sum() == groups["speaker"]["Ses04M"].sum() == 2
    details = metrics.probability_details(toy_dev.probabilities[0], toy_dev.labels)
    subgroups, high = dev_audit.subgroup_report(groups, np.zeros(4, dtype=bool), details)
    incoming = {(row["group_type"], row["group"]): row for row in subgroups}
    assert incoming["shift", "NO_HISTORY"]["support"] == 2
    assert incoming["emotion", "fru"]["support"] == 0
    assert incoming["uh", "U_ONLY"]["support"] is None
    assert sum(row["support"] for row in high if row["group_type"] == "speaker") == 12


def test_uh_enrichment_join_checks_prediction(toy_dev, monkeypatch):
    rows = [dict(utterance_id=uid, dialogue_id=dialogue, gold=dev_inputs.CLASSES[gold],
                 F0_prediction=dev_inputs.CLASSES[gold], U_TOP25=str(i == 0), H_TOP25=str(i == 1),
                 history_sensitivity=str(i))
            for i, (uid, dialogue, gold) in enumerate(zip(toy_dev.utterances, toy_dev.dialogues, toy_dev.labels))]
    monkeypatch.setattr(dev_inputs, "read_rows", lambda _: rows[::-1])
    enrichment = dev_inputs.read_enrichment(Path("history.csv"), toy_dev)
    assert [row["UH_group"] for row in enrichment] == ["U_ONLY", "H_ONLY", "NEITHER", "NEITHER"]
    assert [row["history_sensitivity"] for row in enrichment] == [0, 1, 2, 3]
    mismatched = replace(toy_dev, predictions=np.zeros_like(toy_dev.predictions))
    with pytest.raises(ValueError, match="disagrees with F0"):
        dev_inputs.read_enrichment(Path("history.csv"), mismatched)


@pytest.mark.parametrize("rows,match", [
    ([dict(utterance_id="X"), dict(utterance_id="X")], "duplicate"),
    ([dict(utterance_id="X")], "Missing or extra"),
    ([dict(utterance_id="X", split="test")], "Only Dev"),
    ([dict(utterance_id="X", split="train")], "Only Dev"),
])
def test_utterance_duplicates_missing_and_non_dev_rejected(rows, match):
    with pytest.raises(ValueError, match=match):
        dev_inputs.index_rows(rows, required_uids=("X", "Y"))


def test_hold_rejects_nonzero_and_selection():
    valid = dict(test_policy="HOLD", test_used_for_selection=False, **dict.fromkeys(dev_inputs.HOLD_COUNTERS, 0))
    dev_inputs.hold(valid)
    with pytest.raises(ValueError, match="zero-counter"):
        dev_inputs.hold(dict(valid, test_evaluation_count=1))
    with pytest.raises(ValueError, match="selection"):
        dev_inputs.hold(dict(valid, test_used_for_selection=True))


@pytest.fixture
def archive_meta(monkeypatch):
    manifest = dict(split="dev", test_policy="HOLD", git_head=dev_inputs.SOURCE_HEAD,
                    synthetic_only=False, files_sha256={}, batch_id="audit")
    hold = dict(test_policy="HOLD", **dict.fromkeys(dev_inputs.HOLD_COUNTERS, 0))
    checkpoints = [dict(seed=seed, variant="F0", split="dev", architecture_sha256=dev_inputs.ARCHITECTURE,
                        git_head=dev_inputs.SOURCE_HEAD, run_id=f"tav_track_fusion_final_f0_s{seed}",
                        sha256=dev_inputs.CONTROL_CHECKPOINT, best_epoch=21, dev_population_sha256="one")
                   for seed in dev_inputs.SEEDS]
    vocab = dict(itos=list(dev_inputs.CLASSES), stoi=dict(zip(dev_inputs.CLASSES, range(len(dev_inputs.CLASSES)))))
    files = {"analysis_bundle_manifest.json": manifest, "test_hold_status.json": hold,
             "checkpoint_identity.json": dict(checkpoints=checkpoints), "label_vocab.json": vocab}
    monkeypatch.setattr(dev_inputs, "read_json", lambda path: files[path.name])
    monkeypatch.setattr(dev_inputs, "verify_digest", lambda *args: "digest")
    monkeypatch.setattr(dev_inputs, "digest", lambda *args: "digest")
    return manifest, checkpoints


def test_wrong_frozen_seed_identity_stops_before_reading_predictions(archive_meta, monkeypatch):
    _, checkpoints = archive_meta
    checkpoints[1]["seed"] = 42
    monkeypatch.setattr(dev_inputs, "read_seed", lambda *args: pytest.fail("read unverified predictions"))
    with pytest.raises(ValueError, match="incomplete or duplicated"):
        dev_inputs.read_dev(Path("archive"))


def test_official_dev_rejects_reordered_ids_even_when_sets_equal(archive_meta, toy_dev, monkeypatch):
    gold = [dev_inputs.CLASSES[i] for i in toy_dev.labels]
    identity = tuple(zip(toy_dev.utterances, toy_dev.dialogues, gold))

    def read_seed(_path, seed):
        source = identity[::-1] if seed == 100 else identity
        return source, toy_dev.labels, toy_dev.predictions[0], toy_dev.probabilities[0]

    monkeypatch.setattr(dev_inputs, "read_seed", read_seed)
    with pytest.raises(ValueError, match="population order differs at seed 100"):
        dev_inputs.read_dev(Path("archive"))

from io import StringIO
import json
from types import SimpleNamespace

import pytest

from scripts.analysis.models.multidag_cl.pairwise_preflight_inputs import HOLD
from scripts.analysis.models.multidag_cl.post_ph import history_audit
from scripts.analysis.models.multidag_cl.post_ph.history_audit import compare, keeps, transition_map


@pytest.mark.parametrize(
    "full,restricted,gold,expected",
    [("hap", "hap", "sad", "PREDICTION_UNCHANGED"), ("hap", "sad", "hap", "FULL_BETTER"),
     ("hap", "sad", "sad", "RESTRICTED_BETTER"), ("hap", "sad", "neu", "CHANGED_BOTH_WRONG")],
)
def test_history_outcomes(full, restricted, gold, expected):
    assert compare(full, restricted, gold) == expected


def test_history_prefix_masks():
    assert keeps("RECENT_ONLY", 4, False)
    assert not keeps("RECENT_ONLY", 5, True)
    assert keeps("DISTANT_ONLY", 5, True)
    assert not keeps("DISTANT_ONLY", 4, True)
    assert keeps("SAME_SPEAKER_ONLY", 8, True)
    assert keeps("OTHER_SPEAKER_ONLY", 1, False)
    assert not keeps("CURRENT_ONLY", 1, True)


def test_transitions_fail_on_wrong_split_duplicate_or_missing():
    membership = [dict(utterance_id="a", dialogue_id="d", gold="hap", full_prediction="sad")]
    header = "utterance_id,dialogue_id,gold,base,candidate,split\n"
    row = "a,d,hap,sad,hap,dev\n"

    def transitions(content):
        path = SimpleNamespace(open=lambda **_: StringIO(header + content))
        return transition_map(path, membership, source_prediction="base", candidate_prediction="candidate")

    assert transitions(row) == {"a": "RECOVERED"}
    with pytest.raises(ValueError, match="align"):
        transitions(row + row)
    with pytest.raises(ValueError, match="align"):
        transitions(row.replace(",dev", ",test"))
    with pytest.raises(ValueError, match="missing"):
        transitions("")


def test_transition_manifest_holds_test_and_pins_artifact(monkeypatch):
    schema = "ct_multidag_history_fastkill_v1"
    source = SimpleNamespace(identity={"seed": 100, "epoch": 21})
    manifest = {**HOLD, "schema": schema, "status": "completed", "f0": source.identity,
                "outputs_sha256": {"DEV_SAMPLE_TRANSITIONS.csv": "expected"}}
    path = SimpleNamespace(with_name=lambda _: SimpleNamespace(read_text=lambda **_: json.dumps(manifest)))
    monkeypatch.setattr(history_audit, "sha256_file", lambda _: "expected")
    history_audit.verify_transition_source(path, source, schema)
    manifest["test_labels_inspected"] = 1
    with pytest.raises(ValueError, match="HOLD"):
        history_audit.verify_transition_source(path, source, schema)
    manifest["test_labels_inspected"] = 0
    manifest["outputs_sha256"]["DEV_SAMPLE_TRANSITIONS.csv"] = "wrong"
    with pytest.raises(ValueError, match="run evidence"):
        history_audit.verify_transition_source(path, source, schema)


def test_history_cosine_scores_directional_changes_only():
    rows = [{"condition": "CURRENT_ONLY", "outcome": outcome, "removed_mean_h0_cosine": cosine}
            for outcome, cosine in (("FULL_BETTER", .8), ("RESTRICTED_BETTER", .1),
                                    ("CHANGED_BOTH_WRONG", .95), ("PREDICTION_UNCHANGED", -.2))]
    out = history_audit.cosine_summary(rows)[0]
    assert out["support"] == 2
    assert out["auroc_higher_cosine_predicts_full_better"] == 1

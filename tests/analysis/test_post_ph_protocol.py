from collections import Counter

import pytest

from scripts.analysis.models.multidag_cl.post_ph.internal_protocol import (
    LABEL_VOCAB, audit_folds, metadata, protocol_a, protocol_b, validate_train,
)


def train_rows():
    rows = []
    for session in range(1, 5):
        for index in range(1, 6):
            did = f"Ses0{session}F_impro{index:02}"
            for pos, label in enumerate(LABEL_VOCAB, 1):
                uid = did + f"_F{pos:03}"
                rows.append({"split": "train", "utterance_id": uid, "dialogue_id": did,
                             "position": pos, "gold": label, **metadata(uid, did)})
    return rows


def test_dialogue_folds_are_reproducible_and_balanced():
    rows = train_rows()
    assignment, evidence = protocol_a(rows)
    assert protocol_a(list(reversed(rows))) == (assignment, evidence)
    audit = audit_folds(rows, assignment)
    assert len(audit) == 5
    assert sum(row["validation"]["utterances"] for row in audit) == len(rows)
    for row in rows:
        assert row["dialogue_id"] in assignment
    assert evidence["max_class_prevalence_delta"] < 1e-12


def test_session_protocol_isolates_both_speakers():
    rows = train_rows()
    result = audit_folds(rows, protocol_b(rows), session_holdout=True)
    assert [row["validation"]["sessions"] for row in result] == [[f"Ses0{n}"] for n in range(1, 5)]
    assert all(not (set(row["fit"]["speaker_ids"]) & set(row["validation"]["speaker_ids"])) for row in result)
    count = Counter(row["gold"] for row in rows)
    assert all(sum(row["validation"]["class_counts"][label] for row in result) == count[label]
               for label in LABEL_VOCAB)


def test_split_duplicates_and_missing_turn_are_rejected():
    rows = train_rows()
    for split in ("dev", "test"):
        with pytest.raises(ValueError, match="Train-only"):
            validate_train([{**rows[0], "split": split}, *rows[1:]])
    with pytest.raises(ValueError, match="duplicate Train"):
        validate_train([*rows, rows[0]])
    with pytest.raises(ValueError, match="incomplete"):
        validate_train(rows[1:])
    with pytest.raises(ValueError, match="invalid canonical"):
        metadata(rows[0]["utterance_id"], rows[-1]["dialogue_id"])

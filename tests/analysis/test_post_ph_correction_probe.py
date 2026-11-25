import numpy as np
import pytest

from scripts.analysis.models.multidag_cl.pairwise_preflight_inputs import FrozenSplit
from scripts.analysis.models.multidag_cl.post_ph.correction_probe import (
    VIEW_NAMES, fit_probes, prepare_frozen, rank_support, scoring, shift_rows,
)


def frozen_fixture(session, groups):
    population, logits, h0 = [], [], []
    for index in range(groups):
        did = f"Ses0{session}F_impro{index:02}"
        for pos in range(12):
            gold = pos % 6
            z = np.zeros(6)
            q = (gold + 1) % 6
            flip = pos % 2
            z[gold], z[q] = (2, 3) if flip else (5, 2)
            if pos == 0:
                z[(gold + 2) % 6] = 6
                z[q] = 5.5
            uid = f"{did}_F{pos:03}"
            population.append((uid, did, gold))
            logits.append(z)
            state = np.zeros(300)
            state[0] = flip
            h0.append(state)
    h0 = np.array(h0, dtype=np.float32)
    return FrozenSplit(np.array(logits), tuple(population), (h0, -h0))


def test_fixed_grouped_probes_use_gold_top_two_only():
    train = prepare_frozen(frozen_fixture(1, 5), "train")
    dev = prepare_frozen(frozen_fixture(2, 2), "dev")
    assignment = {did: f"A{index}" for index, did in enumerate(sorted({row["dialogue_id"] for row in train[0]}))}
    result = fit_probes(train, dev, assignment)
    assert len(result) == len(VIEW_NAMES) * 7
    confirmed = [row for row in result if row["evaluation"] == "dev_fixed_once"]
    assert [row["feature_view"] for row in confirmed] == list(VIEW_NAMES)
    assert all(row["eligible"] == 22 and row["flip_count"] == 12 for row in confirmed)
    for row in confirmed[2:]:
        assert row["auroc"] == 1
    assert rank_support(dev[1], split_name="dev")[0]["outside_top2"] == 2
    assert {row["population"] for row in shift_rows(train, dev)} == {"train_in_sample", "dev"}


def test_train_dev_overlap_test_and_alignment_rejected():
    split = frozen_fixture(1, 3)
    train, dev = prepare_frozen(split, "train"), prepare_frozen(split, "dev")
    with pytest.raises(ValueError, match="Train/Dev leakage"):
        shift_rows(train, dev)
    with pytest.raises(ValueError, match="only Train and Dev"):
        prepare_frozen(split, "test")
    states = (np.zeros((len(split.labels) - 1, 300)), split.layer_states[1])
    with pytest.raises(ValueError, match="aligned"):
        prepare_frozen(FrozenSplit(split.logits, split.population, states), "dev")


def test_positive_is_flippable_and_net_corrections_count_harm():
    result = scoring([False, True, True, False], [0.45, 0.65, 0.3, 0.9])
    assert result["tp"] == result["fp"] == 1
    assert result["eligible_net_corrections"] == 0
    assert result["flip_prevalence"] == 0.5
    assert result["auroc"] == 0.25

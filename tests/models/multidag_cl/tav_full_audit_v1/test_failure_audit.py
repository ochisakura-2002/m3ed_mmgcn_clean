from copy import deepcopy

import numpy as np
import pytest
import torch

from datasets.iemocap.tav_full_audit_dataset import DIMS, collate
from models.multidag_cl.fusion.model import FusionTAV
from scripts.analysis.models.multidag_cl.failure_audit_inference import (
    HISTORY, MASKS, REPEATS, context_logits, donor_plans, evaluate, keep_history, substituted_prefixes,
)
from scripts.analysis.models.multidag_cl.failure_audit_metrics import (
    ambiguity_audit, context_audit, decision, details, history_audit, metadata, modality_audit,
    prioritize, quartiles, seed_disagreement, selection_groups, shift_audit,
)
from scripts.runtime.multidag_cl_fusion.spec import make_final_config
from scripts.runtime.multidag_cl_tav_full_audit_v1.runtime import MODEL_INPUTS
from scripts.runtime.multidag_cl_tav_full_audit_v1.spec import CONFIRM_SEEDS


def sample(dialogue, split, labels):
    length = len(labels)
    generator = torch.Generator().manual_seed(length)
    return {"dialogue_id": dialogue, "split": split, "labels": torch.tensor(labels),
            "utterance_ids": tuple(f"{dialogue}_{i}" for i in range(length)),
            "speaker_ids_int": torch.arange(length) % 2,
            **{key: torch.randn(length, dim, generator=generator) for key, dim in DIMS.items()}}


@pytest.fixture
def frozen():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield FusionTAV(make_final_config("F0", 100)).requires_grad_(False).eval()
    finally:
        torch.set_num_threads(threads)


def test_prefix_donors_preserve_target_and_speaker_layout(frozen):
    train = [sample("train_a", "train", [label for i in range(6) for label in (i, i)])]
    dev = [sample("dev_a", "dev", [0, 1, 0, 2, 3, 1, 5, 0])]
    plans = donor_plans(train, dev)
    recipient = dev[0]
    for repeat in REPEATS:
        for condition in ("ECCS", "EICS"):
            donors = plans[repeat, "dev_a", condition]
            prefix = substituted_prefixes(recipient, donors, [3])[0]
            assert torch.equal(prefix["speaker_ids_int"], recipient["speaker_ids_int"][:4])
            for key in DIMS:
                torch.testing.assert_close(prefix[key][-1], recipient[key][3], rtol=0, atol=0)
                for j in range(3):
                    donor, k = donors[j]
                    assert donor["split"] == "train" and donor["dialogue_id"] != recipient["dialogue_id"]
                    assert k % 2 == j % 2
                    assert bool(donor["labels"][k] == recipient["labels"][j]) == (condition == "ECCS")
                    torch.testing.assert_close(prefix[key][j], donor[key][k], rtol=0, atol=0)
    actual = context_logits(frozen, dev, plans, device="cpu", chunk_size=3)
    original = evaluate(frozen, dev, "dev", device="cpu", chunk_size=3)["FULL_HISTORY"]
    assert len(actual) == 6
    for logits in actual.values():
        np.testing.assert_allclose(logits[0], original[0], atol=1e-6)
        assert np.any(np.abs(logits[1:] - original[1:]) > 1e-6)
    with pytest.raises(ValueError, match="donors must come from Train"):
        donor_plans(dev, dev)


@torch.inference_mode()
def test_history_mask_is_induced_and_modalities_use_production_mask(frozen):
    dev = [sample("dev_a", "dev", [0, 1, 0, 3, 4, 5, 2, 1])]
    batch = collate(dev)
    inputs = {key: batch[key] for key in MODEL_INPUTS}
    original = frozen(**inputs)
    measured = evaluate(frozen, dev, "dev", device="cpu", chunk_size=3)
    target = torch.tensor([7])
    keep = keep_history(inputs["speaker_ids_int"], original.mask, target, "RECENT_ONLY")
    assert keep.nonzero()[:, 1].tolist() == [3, 4, 5, 6, 7]
    graphs = []

    def capture(_module, args):
        assert args[0].shape[-1] == 300
        graphs.append((args[0].clone(), args[1].clone()))

    handle = frozen.graph_core.register_forward_pre_hook(capture)
    try:
        from scripts.analysis.models.multidag_cl.failure_audit_inference import counterfactual_history
        counterfactual_history(frozen, inputs, original, chunk_size=8)
    finally:
        handle.remove()
    recent_h0, recent_edges = graphs[2]
    assert torch.equal(recent_h0[7], original.layer_states[0][0])
    assert not recent_edges[7, :3].any() and not recent_edges[7, :, :3].any()
    assert recent_edges[7, 7, 3:7].all()
    assert not torch.triu(recent_edges).any()
    assert not graphs[-1][1].any()
    for name, mask in MASKS.items():
        masked = frozen(**inputs, active_modalities=mask)
        np.testing.assert_array_equal(measured[name], masked.logits_ta[masked.mask].numpy())
        for modality, encoded in masked.encoded_modalities.items():
            if modality not in mask:
                assert not torch.count_nonzero(encoded)
    for condition in HISTORY:
        np.testing.assert_allclose(measured[condition][0], measured["FULL_HISTORY"][0], atol=1e-6)
    with pytest.raises(ValueError, match="Test factory forbidden"):
        evaluate(frozen, [], "test", device="cpu", chunk_size=3)


def test_ambiguity_and_reports_use_aligned_utterances():
    labels = np.array([0, 1, 0, 2, 3, 1, 5, 0])
    logits = np.array([[0.] * 6, [0., 1000., 0., 0., 0., 0.]] + [
        [3.0, 1.0, level, 0., 0., 0.] for level in range(6)])
    data = details(logits, labels)
    assert data["mad"][:2].tolist() == pytest.approx([1, 0])
    assert data["normalized_entropy"][:2].tolist() == pytest.approx([1, 0])
    assert data["top2_margin"][:2].tolist() == [0, 1]
    assert quartiles(np.ones(8)).tolist() == [1, 1, 2, 2, 3, 3, 4, 4]
    meta = metadata([(f"u{i}", "d", item) for i, item in enumerate(labels)], [0, 1] * 4)
    assert [row["shift_group"] for row in meta[:4]] == ["NO_HISTORY", "NO_HISTORY", "STABLE", "SHIFT"]
    groups = selection_groups(meta)
    shift, _ = shift_audit(100, data, labels, groups)
    assert sum(row["support"] for row in shift if row["true_label"] == "ALL") == 8
    subs = {(condition, seed): data for condition in ("ECCS", "EICS") for seed in REPEATS}
    context, context_signal = context_audit(100, data, subs, labels, groups)
    assert len(context) == 27 and context_signal["contrast_pp"] == 0
    history, helped, harmed, _ = history_audit(100, dict.fromkeys(HISTORY, data), meta, groups)
    assert len([row for row in history if row["row_type"] == "sample"]) == 8 * len(HISTORY)
    assert helped == harmed == []
    mod, recovery, signal = modality_audit(100, dict.fromkeys(MASKS, data), labels, meta)
    assert len([row for row in mod if row["row_type"] == "per_class"]) == 6 * len(MASKS)
    assert signal["recoverable"] == 0
    assert next(row["support"] for row in recovery if row["group"] == "ALL_AGREE" and row["row_type"] == "summary") == 8
    votes = seed_disagreement([data["prediction"]] * 5)
    assert np.all(votes["seed_majority_ratio"] == 1)
    audit, summary, _ = ambiguity_audit(100, data, labels, meta, groups, votes, np.zeros(8, dtype=bool))
    assert len(audit) == 8 and len(summary) == 24
    q4 = next(row for row in summary if row["quartile"] == 4 and row["proxy"] == "mad" and row["group"] == "ALL")
    assert q4["support"] == 2
    assert q4["residual_error_rate"] == q4["error_rate"]


def test_route_priority_requires_independent_five_seed_evidence():
    evidence = {"shift": {"wf1_gap": .08, "error_gap": .10},
                "ambiguity": {"ratio_gate": True, "mad": .30, "normalized_entropy": .31, "top2_margin": .32,
                              "residual_gap": None}}
    partial = {seed: deepcopy(evidence) for seed in CONFIRM_SEEDS}
    outcome = decision(partial, complete=False)
    assert outcome["AMBIGUITY_SIGNAL"] == "STRONG"
    assert outcome["ROUTE_U"] == "PENDING" and outcome["NEXT_ROUTE"] == "UNRESOLVED"
    full = deepcopy(partial)
    for row in full.values():
        row["context"] = {"eccs_drop_pp": 1, "eics_drop_pp": 7, "contrast_pp": 6}
        row["history"] = dict(helpful=.40, harmful=.20, shift_vs_stable=.10, same_vs_other=.0,
                              helpful_flip=.08, harmful_flip=.05)
        row["ambiguity"]["residual_gap"] = .25
    outcome = decision(full, complete=True)
    assert outcome["LABEL_REPLICATION"] == outcome["SELECTIVE_HISTORY_SIGNAL"] == "STRONG"
    assert outcome["ROUTE_H"] == outcome["ROUTE_U"] == "ELIGIBLE"
    assert outcome["MODALITY_RELIABILITY_SIGNAL"] == "NOT_AVAILABLE"
    assert outcome["NEXT_ROUTE"] == "U"
    assert prioritize({"H": [10] * 5, "U": [11] * 5}, complete=True)[:3] == ("H", None, "NO")
    assert prioritize({"H": [5, 20, 5, 20, 5], "M": [16] * 5}, complete=True)[:3] == ("UNRESOLVED", "H", "YES")

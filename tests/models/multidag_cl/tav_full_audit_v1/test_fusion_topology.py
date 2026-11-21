import csv
import hashlib
import itertools
import json
from pathlib import Path
import tarfile

import pytest
import torch

from models.multidag_cl.fusion.registry import build_fusion
from models.multidag_cl.ta_v0 import LABEL_VOCAB
from scripts.analysis.models.multidag_cl.export_fusion_topology_bundle import ARCHIVE_NAME, compare, read_run
from scripts.runtime.multidag_cl_fusion.runtime import edge_state, paired_model, shared_state
from scripts.runtime.multidag_cl_fusion.spec import (
    TOPOLOGY_EDGES, TOPOLOGY_GROUP, load_screen_configs, make_screen_config, make_topology_config, validate_screen_config,
)
from scripts.workflows.benchmarks import run_ct_multidag_fusion_seed100 as launcher
from utils.output_paths import resolve_output_paths
from .test_fusion import synthetic_screen
from .test_model import inputs


TOTALS = dict(zip(TOPOLOGY_EDGES, (1_638_306, 1_691_106, 1_677_906, 1_664_706)))


def test_initialization_matches_legacy_hybrid_and_preserves_training_rng():
    old, _, _ = paired_model(make_screen_config("Hybrid"), device="cpu")
    old.eval()
    canonical = edge_state(old)
    rng = torch.get_rng_state()
    reference = None
    with torch.no_grad():
        expected = old(**inputs())
    for arm, edges in TOPOLOGY_EDGES.items():
        model, reference, audit = paired_model(make_topology_config(arm), device="cpu", reference=reference)
        assert torch.equal(torch.get_rng_state(), rng)
        assert audit["common_edge_init_max_abs_diff"] == audit["shared_parameter_max_abs_diff"] == 0
        assert sum(p.numel() for p in model.parameters()) == TOTALS[arm]
        assert all(torch.equal(value, shared_state(old)[key]) for key, value in shared_state(model).items())
        if edges:
            actual = edge_state(model)
            assert tuple(actual) == edges
            assert all(torch.equal(value, canonical[edge][key]) for edge, state in actual.items()
                       for key, value in state.items())
        model.eval()
        with torch.no_grad():
            output = model(**inputs())
        if arm == "H4_FULL_HYBRID":
            old_state, h4_state = old.state_dict(), model.state_dict()
            assert old_state.keys() == h4_state.keys()
            assert all(torch.equal(value, h4_state[key]) for key, value in old_state.items()
                       if isinstance(value, torch.Tensor))
            for key in ("h_ta", "logits_ta", "prob_ta", "mask", "adjacency"):
                assert torch.equal(getattr(expected, key), getattr(output, key))
            assert all(torch.equal(a, b) for a, b in zip(expected.layer_states, output.layer_states))
            for edge in edges:
                assert torch.equal(expected.fusion_diagnostics[edge], output.fusion_diagnostics[edge][:, :4])
        if arm == "H2_AUDIO_TEXT":
            assert torch.equal(output.encoded_modalities["V"], output.layer_states[0][..., 100:200])


def test_topology_configurations_reject_drift():
    for arm, (_, raw) in load_screen_configs(topology=True).items():
        expected = make_screen_config("Hybrid" if TOPOLOGY_EDGES[arm] else "F0")
        assert raw["training"] == expected["training"]
        assert raw["checkpoint"] == expected["checkpoint"]
        assert raw["dataset"] == expected["dataset"]
        assert raw["context_mode"] == expected["context_mode"]
    for edges in ([], ["T<-A", "T<-A"], ["A<-V"], "T<-A", None):
        with pytest.raises(ValueError, match="active_edges"):
            build_fusion({**make_screen_config("Hybrid")["fusion"], "active_edges": edges})
    raw = make_topology_config("H3_ASYMMETRIC")
    raw["fusion"]["active_edges"].reverse()
    with pytest.raises(ValueError, match="screening config"):
        validate_screen_config(raw)


def test_full_topology_train_dev_synthetic_bundle(artifact_dir):
    options = synthetic_screen(artifact_dir)
    date, jobs, configs = launcher.plan(output_root=artifact_dir, experiment_date=options["date"], topology=True)
    paths = resolve_output_paths(output_base=artifact_dir, experiment_date=date, experiment_group=TOPOLOGY_GROUP,
                                 batch_id="topology")
    archive = artifact_dir / ARCHIVE_NAME
    options.update(jobs=jobs, configs=configs, paths=paths, archive=archive)
    state = launcher.execute_screen(**options)
    assert state["status"] == state["ANALYSIS_BUNDLE"] == "PASS" and state["PLOT_READY"] == "YES"
    assert state["common_edge_init_max_abs_diff"] == state["shared_backbone_init_max_abs_diff"] == 0
    assert state["paired_shared_initialization"] and state["paired_train_dev_batches"]
    assert state["archive_sha256"] == hashlib.sha256(archive.read_bytes()).hexdigest()
    assert [job["variant"] for job in state["training_jobs"]] == list(TOPOLOGY_EDGES)
    for job in state["training_jobs"]:
        result = job["result"]
        assert result["optimizer_steps"] == 4
        assert result["active_edges"] == list(TOPOLOGY_EDGES[job["variant"]])
        assert result["total_parameter_count"] == TOTALS[job["variant"]]
        assert result["batch_pairing"] == state["f0_batch_pairing"]

    with tarfile.open(archive, "r:gz") as bundle:
        def read(name):
            with bundle.extractfile(name) as stream:
                return stream.read()

        def rows(name):
            return list(csv.DictReader(read(name).decode().splitlines()))

        manifest = json.loads(read("analysis_bundle_manifest.json"))
        assert len(bundle.getnames()) == 26
        assert all(hashlib.sha256(read(name)).hexdigest() == digest for name, digest in manifest["files_sha256"].items())
        hold = json.loads(read("test_hold_status.json"))
        assert hold["test_policy"] == "HOLD"
        assert hold["test_loader_built"] == hold["test_evaluation_count"] == hold["test_checkpoint_selection_count"] == 0
        assert len(rows("per_class_metrics.csv")) == 24
        assert len(rows("paired_comparison.csv")) == 5
        log = read("master.log").decode()
        assert all(f"{arm} PASS best_epoch=" in log for arm in TOPOLOGY_EDGES)
        for arm in TOPOLOGY_EDGES:
            predictions = rows(f"dev_predictions_{arm}_seed100.csv")
            assert len(predictions) == 5 and all(row["split"] == "dev" and row["dialogue_id"] for row in predictions)
            assert all(abs(sum(float(row[f"prob_{label}"]) for label in LABEL_VOCAB) - 1) < 1e-6 for row in predictions)
            assert all(f"logit_{label}" in predictions[0] for label in LABEL_VOCAB)
            counts = rows(f"confusion_counts_{arm}_seed100.csv")
            assert sum(int(row[label]) for row, label in itertools.product(counts, LABEL_VOCAB)) == len(predictions)
            assert len(rows(f"epoch_history_{arm}_seed100.csv")) == 2
        mechanism = rows("fusion_mechanism_stats.csv")
        assert len(mechanism) == 27
        for row in mechanism:
            assert row["edge"] in TOPOLOGY_EDGES[row["variant"]]
            assert float(row["applied_edge_residual_receiver_ratio_mean"]) == pytest.approx(
                0.1 * float(row["message_receiver_ratio_mean"]), rel=1e-6)

    bad_job = state["training_jobs"][0]
    evidence_path = Path(bad_job["run_dir"]) / "reports/validation_best_dev_evidence.json"
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    evidence["predictions"][0]["probabilities"][0] += 0.2
    evidence_path.write_text(json.dumps(evidence), encoding="utf-8")
    with pytest.raises(ValueError, match="probabilities differ"):
        read_run(bad_job, state)


def test_thresholds_and_secondary_regression():
    values = [(0.7, 0.65), (0.705, 0.648), (0.702, 0.66), (0.698, 0.65)]
    summary = [dict(variant=arm, seed=100, weighted_f1=wf1, macro_f1=macro, uar=0.68, accuracy=0.7)
               for arm, (wf1, macro) in zip(TOPOLOGY_EDGES, values)]
    h4, h3, h2 = compare(summary)[:3]
    assert h4["wf1_grade_vs_f0"] == "STRONG" and not h4["favorable_screen_vs_f0"]
    assert h3["wf1_grade_vs_f0"] == "PROMISING" and h3["favorable_screen_vs_f0"]
    assert h2["wf1_grade_vs_f0"] == "NEGATIVE"

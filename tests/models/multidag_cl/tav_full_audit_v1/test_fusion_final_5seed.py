import csv
from copy import deepcopy
import hashlib
import json
import tarfile

import pytest
import torch

from scripts.analysis.models.multidag_cl.export_fusion_final_5seed_bundle import archive_path, paired_statistics
from scripts.runtime.multidag_cl_fusion.runtime import edge_state, paired_model, shared_state
from scripts.runtime.multidag_cl_fusion.spec import (
    FINAL_ARMS, FINAL_GROUP, final_seed_source, make_final_config, make_topology_config, validate_screen_config,
)
from scripts.runtime.multidag_cl_tav_full_audit_v1 import confirm
from scripts.runtime.multidag_cl_tav_full_audit_v1.spec import CONFIRM_SEEDS
from scripts.workflows.benchmarks.run_ct_multidag_fusion_seed100 import execute_screen, plan_final
from utils.output_paths import resolve_output_paths
from .test_fusion import synthetic_screen


def test_seed_source_is_frozen_track_confirm_matrix(monkeypatch):
    assert final_seed_source()["seeds"] == list(CONFIRM_SEEDS)
    rows, configs, sources = confirm.load_design()
    rows = [row for row in rows if row["experiment_id"] != "confirm_tav_track_cloff_tav_past_all_causal_s42"]
    monkeypatch.setattr(confirm, "load_design", lambda: (rows, configs, sources))
    with pytest.raises(ValueError, match=r"FAIL-STOP:.*TAV_MULTISEED_CONFIRM_MATRIX.csv.*CONFIRM_SEEDS"):
        final_seed_source()


@pytest.mark.parametrize("seed", CONFIRM_SEEDS)
def test_exact_backbone_and_common_edge_initialization(seed):
    reference, backbone, common_edges = None, None, None
    for arm in FINAL_ARMS:
        raw = make_final_config(arm, seed)
        expected = make_topology_config(arm)
        expected["training"]["seed"] = seed
        assert all(raw[key] == value for key, value in expected.items() if key not in ("schema", "experiment_id", "output"))
        model, reference, audit = paired_model(raw, device="cpu", reference=reference)
        assert audit["common_edge_init_max_abs_diff"] == audit["shared_parameter_max_abs_diff"] == 0
        if arm == "F0":
            backbone = shared_state(model)
            rng = torch.get_rng_state()
        assert torch.equal(rng, torch.get_rng_state())
        assert all(torch.equal(value, backbone[name]) for name, value in shared_state(model).items())
        if arm == "H4_FULL_HYBRID":
            common_edges = edge_state(model)
        if arm == "H2_AUDIO_TEXT":
            assert set(edge_state(model)) == {"T<-A", "A<-T"}
            assert all(torch.equal(value, common_edges[edge][key]) for edge, weights in edge_state(model).items()
                       for key, value in weights.items())


def test_final_configs_reject_retired_arm_and_training_drift():
    with pytest.raises(ValueError, match="outside the frozen"):
        make_final_config("H3_ASYMMETRIC", CONFIRM_SEEDS[0])
    raw = make_final_config("H2_AUDIO_TEXT", CONFIRM_SEEDS[0])
    changed = deepcopy(raw)
    changed["training"]["optimizer"]["weight_decay"] = 0.1
    with pytest.raises(ValueError, match="config mismatch"):
        validate_screen_config(changed)


def test_all_15_synthetic_runs_package_plot_ready(artifact_dir):
    options = synthetic_screen(artifact_dir)
    date = options["date"]
    _, jobs, configs, seed_source = plan_final(output_root=artifact_dir, experiment_date=date)
    paths = resolve_output_paths(output_base=artifact_dir, experiment_date=date, experiment_group=FINAL_GROUP,
                                 batch_id="final")
    archive = artifact_dir / archive_path(date).name
    options.update(jobs=jobs, configs=configs, paths=paths, archive=archive, seed_source=seed_source)
    state = execute_screen(**options)
    assert state["status"] == state["ANALYSIS_BUNDLE"] == "PASS" and state["PLOT_READY"] == "YES"
    assert state["archive_sha256"] == hashlib.sha256(archive.read_bytes()).hexdigest()
    assert len(state["training_jobs"]) == 15
    for job in state["training_jobs"]:
        result = job["result"]
        f0 = state["pairing_by_seed"][str(job["seed"])]
        assert result["batch_pairing"] == f0["f0_batch_pairing"]
        assert result["shared_initialization"]["training_rng_identity"] == f0["training_rng_identity"]
        assert result["optimizer_steps"] == 4

    with tarfile.open(archive, "r:gz") as bundle:
        def read(name):
            with bundle.extractfile(name) as stream:
                return stream.read()

        def rows(name):
            return list(csv.DictReader(read(name).decode().splitlines()))

        manifest = json.loads(read("analysis_bundle_manifest.json"))
        assert manifest["run_count"] == len(rows("run_summary.csv")) == 15
        assert all(hashlib.sha256(read(name)).hexdigest() == digest for name, digest in manifest["files_sha256"].items())
        assert len(rows("multi_seed_summary.csv")) == 15
        assert len(rows("paired_seed_deltas.csv")) == 40
        assert len(rows("paired_seed_summary.csv")) == 8
        assert len(rows("per_class_metrics.csv")) == 90
        mechanisms = rows("fusion_mechanism_multiseed_summary.csv")
        assert len(mechanisms) == 72 and {row["n"] for row in mechanisms} == {"5"}
        assert len(json.loads(read("checkpoint_identity.json"))["checkpoints"]) == 15
        hold = json.loads(read("test_hold_status.json"))
        assert hold["test_policy"] == "HOLD" and hold["test_loader_built"] == hold["test_evaluation_count"] == 0
        log = read("master.log").decode()
        for job in jobs:
            arm, seed = job["variant"], job["seed"]
            suffix = f"{arm}_seed{seed}.csv"
            assert f"seed={seed} {arm} PASS" in log
            assert len(rows(f"epoch_history_{suffix}")) == 2
            predictions = rows(f"dev_predictions_{suffix}")
            assert len(predictions) == 5 and predictions[0]["logit_ang"] and predictions[0]["prob_ang"]
            assert len(rows(f"confusion_counts_{suffix}")) == len(rows(f"confusion_normalized_{suffix}")) == 6
            assert f"configs/{job['experiment_id']}.yaml" in manifest["files_sha256"]


def score_rows(h4, h2, *, balance_gain=0.001):
    return [dict(seed=seed, variant=arm, weighted_f1=values[index], macro_f1=0.6 + gain, uar=0.6 + gain,
                 accuracy=0.7, loss=0.5) for arm, values, gain in (
                     ("F0", [0.7] * 5, 0), ("H4_FULL_HYBRID", h4, balance_gain), ("H2_AUDIO_TEXT", h2, 0),
                 ) for index, seed in enumerate(CONFIRM_SEEDS)]


def test_decision_thresholds_and_overlap():
    rows = score_rows([0.704, 0.704, 0.704, 0.7, 0.698], [0.7] * 5)
    _, _, stats, decision = paired_statistics(rows)
    assert decision["route"] == "KEEP"
    assert stats[0]["mean"] == pytest.approx(0.20)
    assert (stats[0]["wins"], stats[0]["ties"], stats[0]["losses"]) == (3, 1, 1)
    assert decision["candidates"]["H4_FULL_HYBRID"]["strong_keep"]
    assert decision["candidates"]["H2_AUDIO_TEXT"]["stop"]
    *_, decision = paired_statistics(score_rows([0.7001] * 5, [0.7] * 5))
    assert decision["route"] == "TRADE_OFF"
    *_, decision = paired_statistics(score_rows([0.7] * 5, [0.7] * 5))
    assert decision["route"] == "STOP" and decision["candidates"]["H4_FULL_HYBRID"]["trade_off"]
    with pytest.raises(ValueError, match="15 distinct"):
        paired_statistics(rows[:-1])

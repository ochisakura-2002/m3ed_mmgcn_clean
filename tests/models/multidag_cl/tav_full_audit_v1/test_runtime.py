from copy import deepcopy
import json
from pathlib import Path

import pytest
import torch
import yaml

from scripts.runtime.multidag_cl_paper_reimplementation.checkpoint import load_checkpoint
from scripts.runtime.multidag_cl_ta_v0.formal import json_identity, sha256_file
from scripts.runtime.multidag_cl_tav_full_audit_v1 import formal
from scripts.runtime.multidag_cl_tav_full_audit_v1.evaluation import evaluate_masks
from scripts.runtime.multidag_cl_tav_full_audit_v1.runtime import AuditRuntime, synthetic_runtime
from scripts.runtime.multidag_cl_tav_full_audit_v1.spec import (
    CONFIG_ROOT, CORES, CURRICULA, MODALITIES, ROOT, design, evaluation_design, load_matrices, make_config, validate_config,
)
from scripts.workflows.benchmarks.run_tav_full_audit_v1 import plan
from scripts.runtime.multidag_cl_tav_full_audit_v1 import spec


@pytest.mark.parametrize("core", CORES)
@pytest.mark.parametrize("curriculum", CURRICULA)
def test_shared_cl7_math_train_step_reload_and_seven_same_checkpoint_masks(core, curriculum, artifact_dir):
    raw = make_config(core, curriculum=curriculum)
    runtime = synthetic_runtime(raw)
    assert runtime.curriculum.manifest.actual_bucket_count == 7
    assert len(runtime.curriculum.visible_indices(1)) == (1 if curriculum == "ORIGINAL_CL7" else 7)
    assert len(runtime.curriculum.visible_indices(7)) == 7
    manifest = formal.execute(runtime, raw, artifact_dir / "run", synthetic_epochs=1)
    assert manifest["test_loader_built"] == manifest["test_evaluation_count"] == 0
    best = artifact_dir / "run/checkpoints/best_model.pt"
    checkpoint = load_checkpoint(best, torch.device("cpu"))
    runtime.validate_checkpoint(checkpoint)
    rows = [row for row in evaluation_design() if row["source_experiment_id"] == raw["experiment_id"]]
    results = evaluate_masks(runtime, artifact_dir / "run", rows, artifact_dir / "evaluation",
                             expected_checkpoint_sha256=manifest["best_checkpoint_sha256"])
    assert len(results) == 7
    assert {value["checkpoint_sha256"] for value in results} == {sha256_file(best)}
    assert all(value["optimizer_steps_performed"] == value["test_evaluation_count"] == 0 for value in results)
    assert results[0]["metrics"] == manifest["best_dev_metrics"]
    with pytest.raises(FileExistsError):
        evaluate_masks(runtime, artifact_dir / "run", rows, artifact_dir / "evaluation",
                       expected_checkpoint_sha256=manifest["best_checkpoint_sha256"])


@pytest.mark.parametrize("active", MODALITIES)
def test_train_ablation_backward_keeps_all_encoders_and_zero_inactive_gradients(active):
    runtime = synthetic_runtime(make_config(modalities=active))
    batch = next(iter(runtime.loader("train")))
    runtime.train_step(batch)
    encoders = runtime.model.paper_encoder
    for letter, name in (("T", "text_projection"), ("A", "audio_projection"), ("V", "visual_projection")):
        grad = getattr(encoders, name).weight.grad
        assert grad is not None and torch.isfinite(grad).all()
        if letter not in active:
            assert torch.count_nonzero(grad) == 0


@pytest.mark.parametrize("context", ("w0", "w1", "w3", "w5"))
def test_context_train_step_allows_structurally_unused_predecessor_parameters(context):
    runtime = synthetic_runtime(make_config(context=context))
    assert runtime.train_step(next(iter(runtime.loader("train"))))["optimizer_steps"] == 1


@pytest.mark.parametrize("curriculum", CURRICULA)
def test_completed_epoch_resume_matches_continuous_training(curriculum, artifact_dir):
    raw = make_config(curriculum=curriculum)
    continuous = synthetic_runtime(raw)
    formal.execute(continuous, raw, artifact_dir / "continuous", synthetic_epochs=2)
    first = synthetic_runtime(raw)
    formal.execute(first, raw, artifact_dir / "resumed", synthetic_epochs=1)
    resumed = synthetic_runtime(raw)
    formal.execute(resumed, raw, artifact_dir / "resumed", resume=artifact_dir / "resumed/checkpoints/last_model.pt", synthetic_epochs=2)
    left = load_checkpoint(artifact_dir / "continuous/checkpoints/last_model.pt", torch.device("cpu"))
    right = load_checkpoint(artifact_dir / "resumed/checkpoints/last_model.pt", torch.device("cpu"))
    assert left["selector"] == right["selector"] and left["global_step"] == right["global_step"]
    for name, value in left["model_state_dict"].items():
        if isinstance(value, torch.Tensor):
            torch.testing.assert_close(value, right["model_state_dict"][name], rtol=0, atol=0)
    assert torch.equal(left["rng_state"]["torch"], right["rng_state"]["torch"])


def test_resume_can_finish_dev_export_after_final_epoch_without_retraining(monkeypatch, artifact_dir):
    raw = make_config()
    runtime = synthetic_runtime(raw)
    evaluate = runtime.evaluate
    calls = []
    def fail_final_export(*args, **kwargs):
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError("synthetic export failure")
        return evaluate(*args, **kwargs)
    monkeypatch.setattr(runtime, "evaluate", fail_final_export)
    with pytest.raises(RuntimeError, match="export failure"):
        formal.execute(runtime, raw, artifact_dir / "run", synthetic_epochs=1)
    resumed = synthetic_runtime(raw)
    monkeypatch.setattr(resumed, "train_step", lambda *args: pytest.fail("finalized epoch retrained"))
    result = formal.execute(resumed, raw, artifact_dir / "run", synthetic_epochs=1,
                            resume=artifact_dir / "run/checkpoints/last_model.pt")
    assert result["status"] == "synthetic_completed" and result["last_completed_epoch"] == 1


def test_no_test_factory_before_any_loader_or_evaluation(monkeypatch):
    runtime = synthetic_runtime(make_config())
    calls = []
    runtime.dataset_factory = lambda split: calls.append(split)
    for action in (lambda: runtime.loader("test"), lambda: runtime.dataset("test"),
                   lambda: runtime.evaluate(split="test"), lambda: runtime.forward_batch({}, split="test"),
                   lambda: runtime.evaluate_test(), lambda: runtime.test_loader):
        with pytest.raises(ValueError):
            action()
    assert calls == []


def test_collision_failed_run_and_resume_identity_are_fail_closed(monkeypatch, artifact_dir):
    raw = make_config()
    runtime = synthetic_runtime(raw)
    directory = artifact_dir / "existing"
    directory.mkdir()
    monkeypatch.setattr(runtime, "loader", lambda *a, **kw: pytest.fail("loader invoked after collision"))
    with pytest.raises(FileExistsError):
        formal.execute(runtime, raw, directory, synthetic_epochs=1)
    runtime = synthetic_runtime(raw)
    monkeypatch.setattr(runtime, "train_step", lambda *a: (_ for _ in ()).throw(RuntimeError("deliberate failure")))
    with pytest.raises(RuntimeError, match="deliberate failure"):
        formal.execute(runtime, raw, artifact_dir / "failed", synthetic_epochs=1)
    manifest = json.loads((artifact_dir / "failed/manifests/run_manifest.json").read_text())
    assert manifest["status"] == "failed" and manifest["test_evaluation_count"] == 0
    with pytest.raises(FileExistsError):
        formal.execute(synthetic_runtime(raw), raw, artifact_dir / "failed", synthetic_epochs=1)
    with pytest.raises(ValueError, match="original run"):
        formal.execute(synthetic_runtime(raw), raw, artifact_dir / "failed", resume=artifact_dir / "other.pt", synthetic_epochs=2)


def test_resume_rejects_moved_run_directory(artifact_dir):
    raw = make_config()
    formal.execute(synthetic_runtime(raw), raw, artifact_dir / "run", synthetic_epochs=1)
    path = artifact_dir / "run/resolved_config.yaml"
    resolved = yaml.safe_load(path.read_text(encoding="utf-8"))
    resolved["run_dir"] = str(artifact_dir / "another_run")
    path.write_text(yaml.safe_dump(resolved), encoding="utf-8")
    with pytest.raises(ValueError, match="resolved config identity"):
        formal.execute(synthetic_runtime(raw), raw, artifact_dir / "run", synthetic_epochs=2,
                       resume=artifact_dir / "run/checkpoints/last_model.pt")


@pytest.mark.parametrize("section,key,value", [("training", "seed", 101), ("training", "max_epochs", 31),
    ("training", "scheduler", "cosine"), ("training", "class_weight", [1] * 6),
    ("runtime", "splits", ["train", "dev", "test"]), ("runtime", "test_policy", "EVALUATE"),
    ("curriculum", "variant", "weighted_icl"), ("checkpoint", "test_evaluation_count", 1),
    ("graph_core", "enabled", True)])
def test_protocol_changes_rejected_before_data(section, key, value):
    raw = make_config()
    raw[section][key] = value
    with pytest.raises(ValueError):
        validate_config(raw)


def test_exact_matrices_order_dedup_and_cl_on_off_single_factor_identity():
    training, evaluation = load_matrices()
    assert len(training) == len({row["experiment_id"] for row in training}) == 24
    assert len(evaluation) == len({row["evaluation_id"] for row in evaluation}) == 56
    assert [int(row["phase"]) for row in training] == [1] * 2 + [2] * 3 + [3] * 3 + [4] * 12 + [5] * 4
    assert len([row for row in training if row["graph_core"] == "original" and row["context_mode"] == "past_all_causal"]) == 14
    for active in MODALITIES:
        off, on = make_config(modalities=active), make_config(modalities=active, curriculum="ORIGINAL_CL7")
        for config in (off, on):
            config.pop("experiment_id")
            config["curriculum"]["variant"] = "NORMALIZED"
            config["training"]["curriculum"] = "NORMALIZED"
            config["output"]["experiment_group"] = "NORMALIZED"
        assert off == on
    date, jobs, evaluations = plan(experiment_date="20261118")
    assert len(jobs) == 24 and len(evaluations) == 56
    assert all(f"outputs/{date}/tav_audit_" in row["run_dir"] for row in jobs)


def test_mutated_matrix_and_inference_source_rejected(artifact_dir):
    directory = ROOT / CONFIG_ROOT
    for name in ("TAV_FULL_AUDIT_MATRIX.csv", "TAV_MISSING_MODALITY_EVAL_MATRIX.csv"):
        (artifact_dir / name).write_bytes((directory / name).read_bytes())
    path = artifact_dir / "TAV_FULL_AUDIT_MATRIX.csv"
    path.write_text(path.read_text().replace(",100,30,", ",101,30,", 1))
    with pytest.raises(ValueError, match="24/56"):
        load_matrices(artifact_dir)


def test_frontend_source_hash_is_portable_between_crlf_and_lf(monkeypatch, artifact_dir):
    monkeypatch.setattr(spec, "ROOT", artifact_dir)
    monkeypatch.setattr(spec, "SOURCE", "source.yaml")
    path = artifact_dir / "source.yaml"
    path.write_bytes(b"encoder: UniLSTM\ntext_dim: 1024\n")
    linux_hash = spec.source_sha256()
    path.write_bytes(b"encoder: UniLSTM\r\ntext_dim: 1024\r\n")
    assert spec.source_sha256() == linux_hash

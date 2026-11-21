import csv
import itertools
import json
from copy import deepcopy
from pathlib import Path

import pytest
import torch

from models.multidag_cl.fusion.model import FusionTAV
from models.multidag_cl.fusion.registry import build_fusion
from models.multidag_cl.tav_full_audit_v1 import TAVFullAudit
from scripts.runtime.multidag_cl_fusion.runtime import aggregate_dev, paired_model, shared_state
from scripts.runtime.multidag_cl_fusion.spec import ARMS, GROUP, load_screen_configs, make_screen_config
from scripts.runtime.multidag_cl_tav_full_audit_v1.formal import validate_training_config
from scripts.runtime.multidag_cl_tav_full_audit_v1.runtime import SyntheticDialogues
from scripts.runtime.multidag_cl_tav_full_audit_v1.spec import MODALITIES, make_config
from scripts.workflows.benchmarks import run_ct_multidag_fusion_seed100 as batch_runner
from utils.output_paths import resolve_output_paths
from .test_model import inputs


COUNTS = {"F0": (0, 1638306), "Hybrid": (52800, 1691106), "Fusion-A": (80000, 1718306),
          "Fusion-C": (174127, 1812433)}


@pytest.mark.parametrize("active", MODALITIES)
def test_f0_exactly_matches_frozen_track(active):
    torch.manual_seed(100)
    baseline = TAVFullAudit(make_config("track")).eval()
    f0, _, _ = paired_model(make_screen_config("F0"), device="cpu")
    f0.eval()
    assert all(torch.equal(value, shared_state(f0)[name]) for name, value in shared_state(baseline).items())
    with torch.no_grad():
        expected = baseline(**inputs(), active_modalities=active)
        actual = f0(**inputs(), active_modalities=active)
    fields = ("h_ta", "logits_ta", "prob_ta", "mask", "adjacency")
    assert all(torch.equal(getattr(actual, field), getattr(expected, field)) for field in fields)
    assert all(torch.equal(a, b) for a, b in zip(actual.layer_states, expected.layer_states))
    assert all(torch.equal(t, expected.encoded_modalities[k]) for k, t in actual.encoded_modalities.items())
    with pytest.raises(ValueError, match="identity"):
        f0.load_state_dict(baseline.state_dict(), strict=False)


@pytest.mark.parametrize("arm", ARMS)
def test_production_model_contract_and_backward(arm):
    raw = make_screen_config(arm)
    model, reference, audit = paired_model(raw, device="cpu")
    assert audit["shared_parameter_max_abs_diff"] == 0
    assert all(torch.equal(t, reference.shared[name]) for name, t in shared_state(model).items())
    assert sum(p.numel() for p in model.fusion.parameters()) == COUNTS[arm][0]
    assert sum(p.numel() for p in model.parameters()) == COUNTS[arm][1]
    batch = inputs()
    output = model(**batch)
    assert output.h_ta.shape == (2, 3, 1500)
    assert output.logits_ta.shape == (2, 3, 6)
    assert len(output.layer_states) == 5
    for value in (*output.layer_states, output.logits_ta, output.h_ta):
        assert torch.isfinite(value).all()
        assert not value[~output.mask].count_nonzero()
    assert all(state.shape == (2, 3, 300) for state in output.layer_states)
    assert not torch.triu(output.adjacency).any()
    labels = torch.arange(int(output.mask.sum())) % 6
    torch.nn.functional.cross_entropy(output.logits_ta[output.mask], labels).backward()
    assert all(p.grad is not None and p.grad.isfinite().all() and p.grad.count_nonzero()
               for p in model.parameters())


@pytest.mark.parametrize("arm", ARMS)
def test_causal_prefix_and_future_perturbations(arm):
    model = FusionTAV(make_screen_config(arm)).eval()
    batch = inputs()
    changed = {name: tensor.clone() for name, tensor in batch.items()}
    for name in ("text_features", "audio_features", "visual_features"):
        changed[name][0, 2] += 50
    changed["speaker_ids_int"][0, 2] = 1
    prefix = {name: value[:, :2] if value.ndim > 1 else value.clamp_max(2) for name, value in batch.items()}
    with torch.no_grad():
        full, other, short = model(**batch), model(**changed), model(**prefix)
    for original, perturbed, truncated in zip((*full.layer_states, full.logits_ta),
                                              (*other.layer_states, other.logits_ta),
                                              (*short.layer_states, short.logits_ta)):
        torch.testing.assert_close(original[:, :2], perturbed[:, :2], atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(original[:, :2], truncated, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("arm", ARMS)
def test_missing_modality_per_utterance_and_empty_router(arm):
    options = list(itertools.product((False, True), repeat=3))
    valid = torch.tensor([[True] * 8, [True] * 3 + [False] * 5])
    availability = torch.tensor([options, options[::-1]])
    active = availability & valid[..., None]
    torch.manual_seed(44)
    blocks = torch.randn(2, 8, 3, 100, requires_grad=True)
    fusion = build_fusion(ARMS[arm])
    diagnostics = {}

    def forward(x):
        return fusion(text=x[..., 2, :], audio=x[..., 0, :], visual=x[..., 1, :],
                      attention_mask=valid, availability=availability, diagnostics=diagnostics)

    mixed = forward(blocks).reshape_as(blocks)
    changed = forward(blocks.detach() + (~active)[..., None] * 100).reshape_as(blocks)
    assert torch.equal(mixed, changed)
    assert mixed.isfinite().all() and not mixed[~active].count_nonzero()
    mixed[:, :4].sum().backward()
    assert not blocks.grad[:, 4:].count_nonzero()
    assert not blocks.grad[~active].count_nonzero()
    if arm == "Fusion-C":
        p = diagnostics["router"][..., :3]
        assert not p[~active].count_nonzero()
        torch.testing.assert_close(p.sum(-1), active.any(-1).float())
        single = active.sum(-1) == 1
        single_probability = p[single].max(-1).values
        assert torch.equal(single_probability, torch.ones_like(single_probability))


def test_strict_yaml_registry_and_near_identity_warning():
    configs = load_screen_configs()
    common = None
    for _, raw in configs.values():
        assert validate_training_config(raw) == raw
        assert raw["dataset"]["population"]["train"]["utterances"] == 4830
        assert raw["dataset"]["population"]["dev"]["utterances"] == 980
        assert raw["runtime"] == {"splits": ["train", "dev"], "test_policy": "HOLD"}
        rest = {key: value for key, value in raw.items() if key not in ("experiment_id", "fusion")}
        if common is None:
            common = rest
        assert rest == common
    hybrid = deepcopy(configs["Hybrid"][1])
    hybrid["fusion"]["rank"] = 17
    with pytest.raises(ValueError, match="screening config"):
        validate_training_config(hybrid)
    with pytest.raises(ValueError, match="rank"):
        build_fusion(hybrid["fusion"])
    hybrid["fusion"]["type"] = "unknown"
    with pytest.raises(ValueError, match="fusion type"):
        build_fusion(hybrid["fusion"])
    activity = aggregate_dev({"activity": [torch.tensor([1e-5, 0, 1e-4])]}, [], by_class=False)
    assert activity["activity"]["FUSION_ACTIVITY_WARNING"] == "NEAR_IDENTITY"


def synthetic_screen(artifact_dir):
    date, jobs, configs = batch_runner.plan(output_root=artifact_dir, experiment_date="20261121")
    paths = resolve_output_paths(output_base=artifact_dir, experiment_date=date, experiment_group=GROUP,
                                 batch_id="synthetic")
    datasets = {split: SyntheticDialogues(split) for split in ("train", "dev")}
    train = datasets["train"]
    samples = list(train.items)
    for index in range(len(train), 17):
        item = samples[index % len(samples)]
        train.items.append(dict(item, dialogue_id=f"synthetic_extra_{index}",
                                utterance_ids=tuple(f"synthetic_extra_{index}_{i}" for i, _ in enumerate(item["labels"]))))
    return dict(date=date, jobs=jobs, configs=configs, paths=paths,
                load_data=lambda: (datasets, {"kind": "synthetic_train_dev_only"}, {}), git={"git_head": "synthetic"},
                device="cpu", synthetic_epochs=2)


def test_master_reuses_runner_and_writes_paired_dev_outputs(artifact_dir):
    opts = synthetic_screen(artifact_dir)
    state = batch_runner.execute_screen(**opts)
    assert state["status"] == "PASS" and state["synthetic_only"]
    assert state["test_loader_built"] == state["test_evaluation_count"] == 0
    assert state["paired_shared_initialization"] and state["paired_train_dev_batches"]
    assert len({job["result"]["architecture_identity"] for job in state["training_jobs"]}) == 4
    for job in state["training_jobs"]:
        assert job["status"] == "PASS" and job["result"]["best_checkpoint_reloaded"]
        assert job["result"]["batch_pairing"] == state["f0_batch_pairing"]
        assert job["result"]["parameter_count"] == COUNTS[job["variant"]][1]
        assert job["result"]["optimizer_steps"] == 4
        with (Path(job["run_dir"]) / "logs" / "epoch_metrics.tsv").open(newline="") as stream:
            epoch_rows = list(csv.DictReader(stream, delimiter="\t"))
        assert len(epoch_rows) == 2
        assert all(row["dev_acc"] == row["dev_accuracy"] and row["dev_wf1"] == row["dev_weighted_f1"]
                   for row in epoch_rows)
    assert len(state["f0_batch_pairing"]["train"]["1"]["batch_sha256"]) == 2
    assert state["f0_batch_pairing"]["train"]["1"] != state["f0_batch_pairing"]["train"]["2"]
    report = opts["paths"].report_root
    with (report / "fusion_seed100_summary.csv").open(newline="") as stream:
        summary = list(csv.DictReader(stream))
    assert [row["variant"] for row in summary] == list(ARMS)
    assert float(summary[0]["delta_vs_F0"]) == 0
    with (report / "fusion_seed100_per_class.csv").open(newline="") as stream:
        classes = list(csv.DictReader(stream))
    assert len(classes) == 24
    assert {row["label"] for row in classes if row["variant"] == "Fusion-C"} == set("hap sad neu ang exc fru".split())
    assert any(row["p_A"] for row in classes if row["variant"] == "Fusion-C" and int(row["support"]))
    with (report / "fusion_seed100_mechanism_stats.csv").open(newline="") as stream:
        mechanism = list(csv.DictReader(stream))
    assert {(row["variant"], row["epoch"]) for row in mechanism} == set(itertools.product(list(ARMS)[1:], ("1", "2")))
    assert {"gate_q05", "message_l2_mean", "residual_receiver_ratio_median", "router_entropy_mean",
            "V_near_zero_rate", "expert_T_residual_norm"} <= {row["metric"] for row in mechanism}


def test_master_records_collision_and_never_enters_an_epoch(artifact_dir):
    opts = synthetic_screen(artifact_dir)
    occupied = Path(opts["jobs"][1]["run_dir"])
    occupied.mkdir(parents=True)
    with pytest.raises(FileExistsError, match="hybrid"):
        batch_runner.execute_screen(**opts)
    state = json.loads((opts["paths"].manifest_root / "fusion_seed100_run_manifest.json").read_text())
    assert state["status"] == "FAIL"
    assert [job["status"] for job in state["training_jobs"]] == ["PENDING", "FAIL", "PENDING", "PENDING"]
    assert not Path(opts["jobs"][0]["run_dir"]).exists()

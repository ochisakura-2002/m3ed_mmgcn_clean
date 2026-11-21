import json

import pytest
import torch
import yaml

from models.multidag_cl.fusion.model import FusionTAV
from models.multidag_cl.ta_v0 import LABEL_VOCAB
from scripts.diagnostics.models.multidag_cl import run_hybrid_directional_intervention as diagnostic
from scripts.runtime.multidag_cl_ta_v0.formal import json_identity, sha256_file
from scripts.runtime.multidag_cl_tav_full_audit_v1.runtime import SyntheticDialogues
from scripts.runtime.multidag_cl_fusion.spec import make_screen_config
from .test_model import inputs


def assert_same_forward(original, actual):
    for name in ("h_ta", "logits_ta", "prob_ta", "mask", "adjacency"):
        assert torch.equal(getattr(original, name), getattr(actual, name))
    assert all(torch.equal(a, b) for a, b in zip(original.layer_states, actual.layer_states))
    for field in ("encoded_modalities", "fusion_diagnostics"):
        expected, observed = getattr(original, field), getattr(actual, field)
        assert expected.keys() == observed.keys()
        assert all(torch.equal(tensor, observed[key]) for key, tensor in expected.items())


def test_default_forward_is_elementwise_identical_and_masks_are_scoped():
    model = FusionTAV(make_screen_config("Hybrid")).eval()
    batch = inputs()
    expected_edges = {
        "H0_FULL": (), "H1_DROP_T_FROM_A": ("T<-A",), "H2_DROP_T_FROM_V": ("T<-V",),
        "H3_DROP_A_FROM_T": ("A<-T",), "H4_DROP_V_FROM_T": ("V<-T",),
        "H5_DROP_VISUAL_PAIR": ("T<-V", "V<-T"), "H6_DROP_AUDIO_PAIR": ("T<-A", "A<-T"),
        "H7_ALL_FUSION_OFF": ("T<-A", "T<-V", "A<-T", "V<-T"),
    }
    state = {name: tensor.clone() for name, tensor in model.named_parameters()}
    assert diagnostic.INTERVENTIONS.keys() == expected_edges.keys()
    with torch.inference_mode():
        original = model(**batch)
        with diagnostic.mask_messages(model, diagnostic.INTERVENTIONS["H0_FULL"]):
            assert_same_forward(original, model(**batch))
        for name, dropped in diagnostic.INTERVENTIONS.items():
            with diagnostic.mask_messages(model, dropped):
                output = model(**batch)
            for edge, trace in original.fusion_diagnostics.items():
                if edge == "activity":
                    continue
                masked = output.fusion_diagnostics[edge]
                if edge in expected_edges[name]:
                    assert torch.equal(masked[:, (0, 2)], trace[:, (0, 2)])
                    assert trace[:, 1].count_nonzero() == len(trace)
                    assert masked[:, (1, 3)].count_nonzero() == 0
                else:
                    assert torch.equal(masked, trace)
            assert_same_forward(original, model(**batch))
        assert torch.equal(output.layer_states[0], torch.cat(
            [original.encoded_modalities[key] for key in "AVT"], dim=-1))
        assert output.fusion_diagnostics["activity"].count_nonzero() == 0
    assert all(torch.equal(state[key], tensor) for key, tensor in model.named_parameters())

    with pytest.raises(ValueError, match="inference mode"), diagnostic.mask_messages(model, ("text_from_audio",)):
        pass
    with torch.inference_mode():
        with pytest.raises(RuntimeError, match="forced"), diagnostic.mask_messages(model, ("text_from_audio",)):
            raise RuntimeError("forced")
        assert_same_forward(original, model(**batch))


def test_formal_checkpoint_binding_rejects_changed_manifest_and_export(artifact_dir):
    raw = make_screen_config("Hybrid")
    resolved = {"audit_config": raw, "data_identity": {"name": "fixture"}, "data_paths": {}}
    model = FusionTAV(raw)
    state = model.state_dict()
    checkpoint = {"schema": "tav_full_audit_checkpoint_v1", "audit_config": raw,
                  "data_identity": resolved["data_identity"], "epoch": 19, "epoch_complete": True,
                  "seed": 100, "git_head": "fixture", "test_loader_built": 0, "test_evaluation_count": 0,
                  "model_state_dict": state}
    (artifact_dir / "checkpoints").mkdir()
    (artifact_dir / "manifests").mkdir()
    structured_path = artifact_dir / "checkpoints/best_model.pt"
    weights_path = artifact_dir / "checkpoints/best_model_weights_only.pt"
    torch.save(checkpoint, structured_path)
    torch.save(state, weights_path)
    manifest = {"run_id": raw["experiment_id"], "experiment_id": raw["experiment_id"],
                "fusion": raw["fusion"], "seed": 100, "status": "completed", "synthetic_only": False,
                "last_completed_epoch": 30, "training_finished": True, "best_checkpoint_reloaded": True,
                "best_epoch": 19, "config_sha256": json_identity(raw), "data_identity": resolved["data_identity"],
                "data_paths": resolved["data_paths"], "git_head": "fixture", "test_loader_built": 0,
                "test_evaluation_count": 0, "best_checkpoint_sha256": sha256_file(structured_path)}
    (artifact_dir / "resolved_config.yaml").write_text(yaml.safe_dump(resolved), encoding="utf-8")
    (artifact_dir / "manifests/run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    verified_resolved, verified_manifest = diagnostic.read_run(artifact_dir)
    weights, identity = diagnostic.verified_weights(artifact_dir, verified_resolved, verified_manifest)
    diagnostic.verify_state(state, weights)
    assert identity["weights_only_sha256"] == sha256_file(weights_path)

    with pytest.raises(ValueError, match="SHA256 differs"):
        diagnostic.verified_weights(artifact_dir, resolved, {**manifest, "best_checkpoint_sha256": "wrong"})
    key = next(iter(dict(model.named_parameters())))
    torch.save({**state, key: state[key] + 1}, weights_path)
    with pytest.raises(ValueError, match="state differs"):
        diagnostic.verified_weights(artifact_dir, resolved, manifest)


def test_non_dev_dialogue_fields_are_never_indexed():
    class IdentityOnly(dict):
        def __getitem__(self, key):
            assert key in ("utterance_ids", "speakers")
            return super().__getitem__(key)

    dev = {"text_features": object(), "benchmark_label_ids": object()}
    held = IdentityOnly(utterance_ids=("held_0",), speakers=("F",), text_features=object(),
                        benchmark_label_ids=object())
    asset = dict(schema="fixture", dataset_version="fixture", feature_spec={}, label_vocab={},
                 splits={"dev": ["dev"], "test": ["held"]}, dialogues={"dev": dev, "held": held})
    assert diagnostic.dev_asset_view(asset)["dialogues"] == {
        "dev": dev, "held": {"utterance_ids": ("held_0",), "speakers": ("F",)},
    }


def test_paired_dev_metrics_include_every_case_and_class(monkeypatch):
    data = SyntheticDialogues("dev")
    raw = make_screen_config("Hybrid")
    population = {"dialogues": 2, "utterances": 5}
    monkeypatch.setattr(diagnostic, "COUNTS", {"dev": population})
    identities = [(item["dialogue_id"], item["utterance_ids"]) for item in data]
    digest = json_identity(identities)
    order = {"dialogue_order_sha256": digest, "batch_sha256": [digest],
             "batch_sequence_sha256": json_identity([digest])}
    pairing = {"population": {"dev": {"sha256": digest, **population}}, "dev": order}
    batches = diagnostic.frozen_dev_batches(data, raw, pairing)
    with pytest.raises(ValueError, match="batch order"):
        diagnostic.frozen_dev_batches(data, raw, {**pairing, "dev": {}})
    model = FusionTAV(raw).eval()
    reports = diagnostic.evaluate_masks(model, batches, device="cpu")
    summary, classes = diagnostic.result_rows(reports)
    assert [row["intervention"] for row in summary] == list(diagnostic.INTERVENTIONS)
    assert all(summary[0][f"delta_{metric}_pp"] == 0 for metric in diagnostic.METRICS)
    assert len(classes) == 48
    for index in range(0, len(classes), 6):
        assert [row["label"] for row in classes[index:index + 6]] == list(LABEL_VOCAB)
        assert sum(row["support"] for row in classes[index:index + 6]) == population["utterances"]
    row = summary[1]
    assert row["delta_weighted_f1_pp"] == pytest.approx(row["weighted_f1_pct"] - summary[0]["weighted_f1_pct"])
    baseline = reports["H0_FULL"]["fusion_mechanism"]
    rendered = diagnostic.render_report(summary, classes, baseline, baseline, "fixture")
    assert "H7_ALL_FUSION_OFF" in rendered and "applied residual/receiver" in rendered

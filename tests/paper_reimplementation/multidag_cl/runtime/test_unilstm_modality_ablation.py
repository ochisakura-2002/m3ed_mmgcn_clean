from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import torch
import yaml

from models.multidag_cl.paper_reimplementation.config import MultiDAGCLConfig
from models.registry.paper_reimplementation import build_paper_reimplementation_model
import scripts.models.multidag_cl.paper_reimplementation.train as cli
import scripts.runtime.multidag_cl_paper_reimplementation.trainer as trainer_module
import scripts.runtime.multidag_cl_paper_reimplementation.validation as validation_module
from scripts.runtime.multidag_cl_paper_reimplementation.adapter import FeatureRegistryMetadata


ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / (
    "configs/multidag_cl/paper_reimplementation/iemocap/full_context/"
    "paper_data_reproduction/causal_text_encoder_diagnostic/cl7_unilstm.yaml"
)
CONFIG_DIR = BASE.parent / "modality_ablation"
SUBSETS = {
    "tav": ("text", "audio", "visual"),
    "ta": ("text", "audio"),
    "tv": ("text", "visual"),
    "av": ("audio", "visual"),
    "t": ("text",),
    "a": ("audio",),
    "v": ("visual",),
}


def _load(path: Path) -> dict:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _config(key: str) -> dict:
    return _load(CONFIG_DIR / f"cl7_unilstm_{key}.yaml")


def _toy_inputs() -> dict[str, torch.Tensor]:
    torch.manual_seed(20260916)
    return {
        "text_features": torch.randn(1, 4, 1024),
        "audio_features": torch.randn(1, 4, 1582),
        "visual_features": torch.randn(1, 4, 342),
        "attention_mask": torch.ones(1, 4, dtype=torch.int64),
        "lengths": torch.tensor([4], dtype=torch.int64),
        "speaker_ids_int": torch.tensor([[0, 1, 0, 1]], dtype=torch.int64),
    }


@pytest.mark.parametrize("key,active", SUBSETS.items())
def test_seven_configs_change_only_subset_and_experiment_identity(key: str, active: tuple[str, ...]) -> None:
    original = _load(BASE)
    derived = _config(key)
    assert derived["model_core"]["modality_ablation"] == {
        "enabled": True,
        "active_modalities": list(active),
    }
    assert derived["run_name"] == f"multidag_cl_unilstm_modality_{key}_paper_data_cl7_seed100"
    assert derived["output"]["experiment_group"] == (
        "multidag_cl_unilstm_modality_ablation_paper_data_cl7"
    )
    expected_deviations = ["causal_unidirectional_text_encoder_ablation"]
    if key != "tav":
        expected_deviations.append("fixed_subset_encoded_modality_mask")
    assert derived["provenance"]["model_math_deviation_list"] == expected_deviations

    del derived["model_core"]["modality_ablation"]
    derived["run_name"] = original["run_name"]
    derived["output"]["experiment_group"] = original["output"]["experiment_group"]
    derived["provenance"]["model_math_deviation_list"] = original["provenance"]["model_math_deviation_list"]
    assert derived == original
    core = MultiDAGCLConfig.from_mapping(_config(key)["model_core"])
    assert core.to_mapping()["modality_ablation"] == {
        "enabled": True,
        "active_modalities": list(active),
    }
    assert MultiDAGCLConfig.from_mapping(core.to_mapping()) == core


@pytest.mark.parametrize("key,active", SUBSETS.items())
def test_formal_entry_check_constructs_each_fixed_subset_without_data_or_training(
    key: str,
    active: tuple[str, ...],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: list[tuple[str, ...]] = []
    original_builder = trainer_module.build_paper_reimplementation_model

    def observe_builder(config):
        model = original_builder(config)
        observed.append(model.config.active_modalities)
        assert model.config.modality_ablation_enabled
        return model

    def metadata(config, *, project_root, require_file, verify_checksum):
        del project_root, verify_checksum
        assert require_file is False
        assert config["dataset"]["feature_registry"] == "multidag_cl_official_2948_v1"
        return FeatureRegistryMetadata(
            registry_key="multidag_cl_official_2948_v1",
            feature_path=config["dataset"]["feature_path"],
            feature_sha256="0" * 64,
            text_dim=1024,
            audio_dim=1582,
            visual_dim=342,
        )

    def forbidden_dataset(*args, **kwargs):
        pytest.fail("check mode accessed a dataset")

    assert cli.run_runtime is trainer_module.run_runtime
    monkeypatch.setattr(validation_module, "resolve_feature_metadata", metadata)
    monkeypatch.setattr(trainer_module, "build_paper_reimplementation_model", observe_builder)
    monkeypatch.setattr(trainer_module, "_make_dataset", forbidden_dataset)
    result = cli.main(
        ["--mode", "check", "--config", str(CONFIG_DIR / f"cl7_unilstm_{key}.yaml"), "--device", "cpu"]
    )
    assert result["status"] == "PASS"
    assert result["mode"] == "check"
    assert result["model_parameter_count"] == 5_975_110
    assert result["optimizer_steps"] == 0
    assert "run_dir" not in result
    assert observed == [active]


@pytest.mark.parametrize("key,active", SUBSETS.items())
def test_mask_is_exactly_zero_after_encoding_and_fusion_stays_300d(key: str, active: tuple[str, ...]) -> None:
    model = build_paper_reimplementation_model(_config(key)).eval()
    assert sum(parameter.numel() for parameter in model.parameters()) == 5_975_110
    assert model.paper_encoder is not None
    with torch.no_grad():
        model.paper_encoder.audio_projection.bias.fill_(1.0)
        model.paper_encoder.visual_projection.bias.fill_(1.0)
        output = model(**_toy_inputs())
    assert output.encoded_modalities is not None
    encoded = output.encoded_modalities
    assert encoded.fused.shape == (1, 4, 300)
    assert output.encoded_state.shape == (1, 4, 300)
    assert output.logits.shape == (1, 4, 6)
    for name, start in (("audio", 0), ("visual", 100), ("text", 200)):
        representation = getattr(encoded, name)
        assert representation.shape == (1, 4, 100)
        torch.testing.assert_close(encoded.fused[..., start : start + 100], representation)
        if name not in active:
            assert torch.count_nonzero(representation).item() == 0
    expected_deviations = ("causal_unidirectional_text_encoder_ablation",)
    if key != "tav":
        expected_deviations += ("fixed_subset_encoded_modality_mask",)
    assert encoded.model_math_deviation_list == expected_deviations


def test_tav_matches_existing_unilstm_with_identical_initial_parameters() -> None:
    torch.manual_seed(104)
    original = build_paper_reimplementation_model(_load(BASE)).eval()
    torch.manual_seed(104)
    control = build_paper_reimplementation_model(_config("tav")).eval()
    original_params = dict(original.named_parameters())
    control_params = dict(control.named_parameters())
    assert original_params.keys() == control_params.keys()
    assert sum(value.numel() for value in original_params.values()) == 5_975_110
    for name in original_params:
        torch.testing.assert_close(original_params[name], control_params[name], rtol=0, atol=0)
    inputs = _toy_inputs()
    with torch.no_grad():
        before = original(**inputs)
        after = control(**inputs)
    assert before.logits.shape == after.logits.shape
    torch.testing.assert_close(before.logits, after.logits, rtol=0, atol=0)
    torch.testing.assert_close(before.encoded_state, after.encoded_state, rtol=0, atol=0)


@pytest.mark.parametrize(
    "section,expected_error",
    [
        ({"enabled": True, "active_modalities": []}, "nonempty"),
        ({"enabled": True, "active_modalities": ["text", "text"]}, "unique"),
        ({"enabled": True, "active_modalities": ["text", "depth"]}, "unknown modality"),
        ({"enabled": True}, "requires active_modalities"),
        ({"enabled": False, "active_modalities": ["text"]}, "requires all modalities"),
    ],
)
def test_invalid_modality_subset_fails_closed(section: dict, expected_error: str) -> None:
    mapping = deepcopy(_load(BASE)["model_core"])
    mapping["modality_ablation"] = section
    with pytest.raises((TypeError, ValueError), match=expected_error):
        MultiDAGCLConfig.from_mapping(mapping)

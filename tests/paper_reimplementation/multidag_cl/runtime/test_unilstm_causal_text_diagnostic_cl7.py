from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import torch
import yaml
from torch.nn.modules.module import register_module_module_registration_hook

from models.multidag_cl.paper_reimplementation.encoders import (
    PaperFormulaModalityEncoder,
)
from models.registry.paper_reimplementation import build_paper_reimplementation_model
import scripts.models.multidag_cl.paper_reimplementation.train as cli
import scripts.runtime.multidag_cl_paper_reimplementation.validation as validation_module
import scripts.runtime.multidag_cl_paper_reimplementation.trainer as trainer_module
from scripts.runtime.multidag_cl_paper_reimplementation.adapter import (
    FeatureRegistryMetadata,
)


ROOT = Path(__file__).resolve().parents[4]
BASELINE_CONFIG = (
    ROOT
    / "configs/multidag_cl/paper_reimplementation/iemocap/full_context/"
    "paper_data_reproduction/curriculum_comparison/cl7.yaml"
)
UNILSTM_CONFIG = (
    ROOT
    / "configs/multidag_cl/paper_reimplementation/iemocap/full_context/"
    "paper_data_reproduction/causal_text_encoder_diagnostic/cl7_unilstm.yaml"
)
DERIVED_MODEL_IDENTITY = (
    "multidag_cl_unilstm_causal_text_diagnostic_paper_data_cl7_seed100"
)


def _load(path: Path) -> dict:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _parameter_count(module: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def _without_controlled_differences(config: dict) -> dict:
    normalized = deepcopy(config)
    del normalized["run_name"]
    del normalized["model_core"]["encoder"]["text_bidirectional"]
    del normalized["model_core"]["encoder"]["causal_text_ablation"]
    del normalized["output"]["experiment_group"]
    del normalized["provenance"]["model_math_deviation_list"]
    return normalized


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


def test_formal_train_entry_constructs_frozen_bilstm_and_unilstm_without_training(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset_accesses: list[str] = []

    def fake_official_feature_metadata(
        config,
        *,
        project_root,
        require_file,
        verify_checksum,
    ) -> FeatureRegistryMetadata:
        del project_root, verify_checksum
        dataset = config["dataset"]
        assert require_file is False
        assert dataset["feature_registry"] == "multidag_cl_official_2948_v1"
        assert dataset["feature_sha256"] == "FROM_OFFICIAL_ASSET_MANIFEST"
        assert dataset["feature_dimensions"] == {
            "text": 1024,
            "audio": 1582,
            "visual": 342,
        }
        return FeatureRegistryMetadata(
            registry_key=dataset["feature_registry"],
            feature_path=dataset["feature_path"],
            feature_sha256="0" * 64,
            text_dim=1024,
            audio_dim=1582,
            visual_dim=342,
        )

    def forbidden_dataset_access(*args, **kwargs):
        dataset_accesses.append(str(kwargs.get("split", "unknown")))
        pytest.fail("check mode must not construct or access a real dataset")

    assert cli.run_runtime is trainer_module.run_runtime
    monkeypatch.setattr(
        validation_module,
        "resolve_feature_metadata",
        fake_official_feature_metadata,
    )
    monkeypatch.setattr(trainer_module, "_make_dataset", forbidden_dataset_access)

    cases = (
        ("A0", BASELINE_CONFIG, 50, True, 5_955_110),
        ("A1", UNILSTM_CONFIG, 100, False, 5_975_110),
    )
    for case_name, config_path, hidden_size, bidirectional, parameter_count in cases:
        observed_encoders: list[PaperFormulaModalityEncoder] = []
        observed_lstms: list[torch.nn.LSTM] = []

        def observe_module_registration(owner, name, module):
            del owner, name
            if isinstance(module, PaperFormulaModalityEncoder):
                observed_encoders.append(module)
            if isinstance(module, torch.nn.LSTM):
                observed_lstms.append(module)

        hook = register_module_module_registration_hook(observe_module_registration)
        try:
            result = cli.main(
                [
                    "--mode",
                    "check",
                    "--config",
                    str(config_path),
                    "--device",
                    "cpu",
                ]
            )
        finally:
            hook.remove()

        assert result["status"] == "PASS", case_name
        assert result["mode"] == "check", case_name
        assert result["model_parameter_count"] == parameter_count, case_name
        assert result["optimizer_steps"] == 0, case_name
        assert result["gradient_clip_count"] == 0, case_name
        assert "run_dir" not in result, case_name
        assert len(observed_encoders) == 1, case_name
        assert len(observed_lstms) == 1, case_name

        encoder = observed_encoders[0]
        text_lstm = observed_lstms[0]
        assert encoder.text_encoder is text_lstm, case_name
        assert encoder.config.text_output_dim == 100, case_name
        assert text_lstm.input_size == 100, case_name
        assert text_lstm.hidden_size == hidden_size, case_name
        assert text_lstm.bidirectional is bidirectional, case_name
        assert text_lstm.hidden_size * (2 if text_lstm.bidirectional else 1) == 100

        inputs = _toy_inputs()
        with torch.no_grad():
            encoded = encoder(
                text_features=inputs["text_features"],
                audio_features=inputs["audio_features"],
                visual_features=inputs["visual_features"],
                lengths=inputs["lengths"],
                attention_mask=inputs["attention_mask"],
            )
        assert encoded.text.shape == (1, 4, 100), case_name

    assert dataset_accesses == []


def test_frozen_cl7_baseline_remains_bilstm_with_expected_parameter_count() -> None:
    config = _load(BASELINE_CONFIG)
    model = build_paper_reimplementation_model(config)
    assert model.paper_encoder is not None
    text_lstm = model.paper_encoder.text_encoder

    assert config["run_name"] == (
        "multidag_cl_paper_reimplementation_paper_data_cl7_seed100"
    )
    assert config["model_core"]["encoder"]["causal_text_ablation"] is False
    assert text_lstm.input_size == 100
    assert text_lstm.hidden_size == 50
    assert text_lstm.num_layers == 1
    assert text_lstm.bidirectional is True
    assert text_lstm.dropout == 0.0
    assert text_lstm.batch_first is True
    assert model.config.text_output_dim == 100
    assert _parameter_count(text_lstm) == 60_800
    assert _parameter_count(model) == 5_955_110


def test_unilstm_cl7_has_independent_identity_and_only_controlled_changes() -> None:
    baseline_config = _load(BASELINE_CONFIG)
    unilstm_config = _load(UNILSTM_CONFIG)
    assert _without_controlled_differences(unilstm_config) == (
        _without_controlled_differences(baseline_config)
    )
    assert unilstm_config["run_name"] == DERIVED_MODEL_IDENTITY
    assert unilstm_config["run_name"] != baseline_config["run_name"]
    assert unilstm_config["output"]["experiment_group"] == (
        "multidag_cl_unilstm_causal_text_diagnostic_paper_data_cl7"
    )
    assert unilstm_config["provenance"]["model_math_deviation_list"] == [
        "causal_unidirectional_text_encoder_ablation"
    ]

    baseline = build_paper_reimplementation_model(baseline_config)
    unilstm = build_paper_reimplementation_model(unilstm_config)
    assert unilstm.paper_encoder is not None
    text_lstm = unilstm.paper_encoder.text_encoder
    assert text_lstm.input_size == 100
    assert text_lstm.hidden_size == 100
    assert text_lstm.num_layers == 1
    assert text_lstm.bidirectional is False
    assert text_lstm.dropout == 0.0
    assert text_lstm.batch_first is True
    assert unilstm.config.text_output_dim == 100
    assert _parameter_count(text_lstm) == 80_800
    assert _parameter_count(unilstm) == 5_975_110

    baseline_parameters = dict(baseline.named_parameters())
    unilstm_parameters = dict(unilstm.named_parameters())
    baseline_frozen_parameters = {
        name: parameter
        for name, parameter in baseline_parameters.items()
        if not name.startswith("paper_encoder.text_encoder.")
    }
    unilstm_frozen_parameters = {
        name: parameter
        for name, parameter in unilstm_parameters.items()
        if not name.startswith("paper_encoder.text_encoder.")
    }
    assert baseline_frozen_parameters.keys() == unilstm_frozen_parameters.keys()
    for name, parameter in baseline_frozen_parameters.items():
        assert parameter.shape == unilstm_frozen_parameters[name].shape

    with pytest.raises(ValueError, match="identity mismatch"):
        unilstm.load_state_dict(baseline.state_dict(), strict=True)


def test_unilstm_cl7_downstream_shapes_remain_compatible() -> None:
    model = build_paper_reimplementation_model(_load(UNILSTM_CONFIG)).eval()
    inputs = _toy_inputs()
    with torch.no_grad():
        output = model(**inputs)

    assert output.encoded_modalities is not None
    assert output.encoded_modalities.text.shape == (1, 4, 100)
    assert output.encoded_modalities.fused.shape == (1, 4, 300)
    assert output.encoded_state.shape == (1, 4, 300)
    assert output.layer_states[0].shape == (1, 4, 300)
    assert len(output.layer_states) == 5
    assert output.representation.shape == (1, 4, 1500)
    assert output.logits.shape == (1, 4, 6)


def test_unilstm_text_representation_is_invariant_to_future_text_perturbation() -> None:
    model = build_paper_reimplementation_model(_load(UNILSTM_CONFIG)).eval()
    assert model.paper_encoder is not None
    inputs = _toy_inputs()
    encoder_inputs = {
        name: inputs[name]
        for name in (
            "text_features",
            "audio_features",
            "visual_features",
            "lengths",
            "attention_mask",
        )
    }
    with torch.no_grad():
        baseline = model.paper_encoder(**encoder_inputs)
        perturbed_inputs = dict(encoder_inputs)
        perturbed_text = inputs["text_features"].clone()
        perturbed_text[:, 2:] = perturbed_text[:, 2:] * -7.0 + 123.0
        perturbed_inputs["text_features"] = perturbed_text
        perturbed = model.paper_encoder(**perturbed_inputs)

    torch.testing.assert_close(
        baseline.text[:, :2],
        perturbed.text[:, :2],
        rtol=0,
        atol=1e-6,
    )
    assert perturbed.context_visibility_identity.dag_topology_causal
    assert (
        perturbed.context_visibility_identity.end_to_end_causal_assuming_local_text_features
    )
    assert not perturbed.context_visibility_identity.end_to_end_causal

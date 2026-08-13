from __future__ import annotations

from dataclasses import FrozenInstanceError
from copy import deepcopy

import pytest
import torch

from models.multidag_cl.paper_reimplementation.config import (
    AblationProfile,
    ConformanceProfile,
    MultiDAGCLConfig,
)
from models.multidag_cl.paper_reimplementation.contracts import MultiDAGBatchContract

from _helpers import cloned_batch, config_mapping, make_batch, make_config


@pytest.mark.parametrize(
    "profile",
    ["paper_formula_behavior", "official_source_behavior"],
)
def test_config_round_trip_is_primitive_and_profile_fixed(profile: str) -> None:
    config = make_config(profile=profile)
    serialized = config.to_mapping()
    assert serialized["identity"]["conformance_profile"] == profile
    assert isinstance(serialized["encoder"]["modality_order"], list)
    assert MultiDAGCLConfig.from_mapping(serialized) == config
    with pytest.raises(FrozenInstanceError):
        config.hidden_dim = 12


def test_primary_and_source_profile_contracts_are_explicitly_distinct() -> None:
    primary = make_config()
    source = make_config(profile="official_source_behavior")
    assert primary.conformance_profile is ConformanceProfile.PAPER_FORMULA_BEHAVIOR
    assert primary.ablation_profile is AblationProfile.NONE
    assert primary.modality_order == ("audio", "visual", "text")
    assert not primary.raw_feature_skip
    assert primary.bucket_count == 5
    assert source.modality_order == ("text", "audio", "visual")
    assert source.single_projection and source.raw_feature_skip
    assert source.bucket_count == 12


@pytest.mark.parametrize("bucket_count", [4, 7, 10, 15])
def test_paper_curriculum_ablation_identity_only_unlocks_table4_buckets(
    bucket_count: int,
) -> None:
    mapping = config_mapping()
    mapping["identity"]["ablation_profile"] = "paper_curriculum_ablation"
    mapping["data"]["track"] = "paper_data"
    mapping["curriculum"]["bucket_count"] = bucket_count
    config = MultiDAGCLConfig.from_mapping(mapping)
    assert config.conformance_profile is ConformanceProfile.PAPER_FORMULA_BEHAVIOR
    assert config.ablation_profile is AblationProfile.PAPER_CURRICULUM_ABLATION
    assert config.bucket_count == bucket_count


@pytest.mark.parametrize("bucket_count", [3, 5, 8, 12])
def test_paper_curriculum_ablation_rejects_non_pending_table4_identity_buckets(
    bucket_count: int,
) -> None:
    mapping = config_mapping()
    mapping["identity"]["ablation_profile"] = "paper_curriculum_ablation"
    mapping["data"]["track"] = "paper_data"
    mapping["curriculum"]["bucket_count"] = bucket_count
    with pytest.raises(ValueError, match="paper_curriculum_ablation"):
        MultiDAGCLConfig.from_mapping(mapping)


@pytest.mark.parametrize(
    ("data_track", "curriculum_enabled"),
    [("project_fair", True), ("paper_data", False)],
)
def test_paper_curriculum_ablation_requires_enabled_paper_data(
    data_track: str,
    curriculum_enabled: bool,
) -> None:
    mapping = config_mapping(curriculum_enabled=curriculum_enabled)
    mapping["identity"]["ablation_profile"] = "paper_curriculum_ablation"
    mapping["data"]["track"] = data_track
    mapping["curriculum"]["bucket_count"] = 4
    with pytest.raises(ValueError, match="paper_curriculum_ablation"):
        MultiDAGCLConfig.from_mapping(mapping)


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("identity", "conformance_profile"), "unknown", "unknown"),
        (("graph", "window_future"), 1, "future"),
        (("graph", "allow_self_edge"), True, "self"),
        (("graph", "allow_future_edge"), True, "future"),
        (("graph", "global_nodal_attention"), True, "global"),
        (("attention", "dropout"), 0.1, "zero"),
        (("dag", "hidden_dim"), 299, "300"),
        (("dag", "raw_feature_skip"), True, "raw"),
        (("curriculum", "bucket_count"), 4, "5"),
        (("checkpoint", "test_split_used_for_selection"), True, "test"),
    ],
)
def test_invalid_primary_configuration_is_rejected(path, value, message: str) -> None:
    mapping = config_mapping()
    mapping[path[0]][path[1]] = value
    with pytest.raises(ValueError, match=message):
        MultiDAGCLConfig.from_mapping(mapping)


def test_implicit_modality_order_and_hidden_profile_switch_are_rejected() -> None:
    missing = config_mapping()
    del missing["encoder"]["modality_order"]
    with pytest.raises((TypeError, ValueError), match="modality_order"):
        MultiDAGCLConfig.from_mapping(missing)

    hidden_switch = config_mapping()
    hidden_switch["encoder"]["profile"] = "official_source_single_projection"
    with pytest.raises(ValueError, match="paper modality encoder"):
        MultiDAGCLConfig.from_mapping(hidden_switch)


def test_source_profile_rejects_paper_encoder_and_missing_raw_skip() -> None:
    mapping = config_mapping(profile="official_source_behavior")
    mapping["encoder"]["profile"] = "paper_modality_specific"
    with pytest.raises(ValueError, match="single-projection"):
        MultiDAGCLConfig.from_mapping(mapping)
    mapping = config_mapping(profile="official_source_behavior")
    mapping["dag"]["raw_feature_skip"] = False
    with pytest.raises(ValueError, match="raw feature skip"):
        MultiDAGCLConfig.from_mapping(mapping)


def test_author_official_identity_and_unknown_test_selection_field_are_rejected() -> None:
    mapping = config_mapping()
    mapping["identity"]["implementation_identity"] = "author_official"
    with pytest.raises(ValueError, match="forbidden"):
        MultiDAGCLConfig.from_mapping(mapping)
    mapping = config_mapping()
    mapping["checkpoint"]["select_by_test_f1"] = False
    with pytest.raises(ValueError, match="unknown checkpoint"):
        MultiDAGCLConfig.from_mapping(mapping)


def test_valid_batch_contract_accepts_train_and_label_free_inference() -> None:
    contract = MultiDAGBatchContract(make_config())
    batch = make_batch()
    batch["labels"][0, 1] = -1
    assert contract.validate(batch, require_labels=True, split="train") is None
    inference = dict(batch)
    del inference["labels"]
    assert contract.validate(inference, require_labels=False, split="inference") is None


@pytest.mark.parametrize(
    "mutator",
    [
        lambda batch: batch.__setitem__("text_features", batch["text_features"].double()),
        lambda batch: batch["audio_features"].__setitem__((1, 3, 0), 1.0),
        lambda batch: batch["labels"].__setitem__((1, 3), 0),
        lambda batch: batch["speaker_ids_int"].__setitem__((1, 3), 2),
        lambda batch: batch["labels"].__setitem__((0, 0), 3),
        lambda batch: batch["speaker_ids_int"].__setitem__((0, 0), -1),
        lambda batch: batch["attention_mask"].__setitem__((1, slice(None)), torch.tensor([1, 0, 1, 0])),
        lambda batch: batch["lengths"].__setitem__(1, 2),
    ],
)
def test_batch_contract_rejects_dtype_padding_labels_speakers_and_masks(mutator) -> None:
    batch = make_batch()
    mutator(batch)
    with pytest.raises((TypeError, ValueError)):
        MultiDAGBatchContract(make_config()).validate(
            batch,
            require_labels=True,
            split="train",
        )


def test_batch_contract_rejects_named_dimension_id_and_device_mismatches() -> None:
    contract = MultiDAGBatchContract(make_config())
    wrong_dim = make_batch()
    wrong_dim["visual_features"] = torch.zeros(2, 4, 3)
    with pytest.raises(ValueError, match="feature dimension"):
        contract.validate(wrong_dim, require_labels=True, split="train")

    duplicate_ids = make_batch()
    duplicate_ids["dialogue_ids"] = ["d0", "d0"]
    with pytest.raises(ValueError, match="unique"):
        contract.validate(duplicate_ids, require_labels=True, split="train")

    wrong_utterances = make_batch()
    wrong_utterances["utterance_ids"][1].append("padded")
    with pytest.raises(ValueError, match="lengths"):
        contract.validate(wrong_utterances, require_labels=True, split="train")

    wrong_device = make_batch()
    wrong_device["labels"] = torch.empty((2, 4), dtype=torch.int64, device="meta")
    with pytest.raises(ValueError, match="device"):
        contract.validate(wrong_device, require_labels=True, split="train")


def test_batch_validator_does_not_modify_or_replace_inputs() -> None:
    batch = make_batch()
    before = cloned_batch(batch)
    tensor_ids = {name: id(value) for name, value in batch.items() if isinstance(value, torch.Tensor)}
    MultiDAGBatchContract(make_config()).validate(
        batch,
        require_labels=True,
        split="train",
    )
    for name, expected in before.items():
        if isinstance(expected, torch.Tensor):
            assert id(batch[name]) == tensor_ids[name]
            assert torch.equal(batch[name], expected)
        else:
            assert batch[name] == expected

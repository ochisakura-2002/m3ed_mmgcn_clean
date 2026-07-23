from __future__ import annotations

from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import pickle
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from models.registry.causal import build_new_causal_baseline
import scripts.baselines.train_multidag_cl as train_multidag_module
import scripts.baselines.train_new_causal_graph_baseline as train_causal_module
import scripts.train_mmgcn as train_mmgcn_module
from scripts.diagnostics.data.audit_iemocap_feature_pkl import (
    audit_feature_pkls,
    write_audit_outputs,
)
from scripts.diagnostics.data.probe_iemocap_text_features import run_probes
from scripts.baselines.train_multidag_cl import build_model as build_multidag
from scripts.dev.validate_config_tree import (
    validate_clean_roberta_v1_formal_pipeline,
    validate_clean_roberta_v1_formal_training,
    validate_clean_roberta_v1_registry,
    validate_clean_roberta_v1_smoke_config,
    validate_clean_roberta_v1_tree,
)
from scripts.data.build.build_iemocap_clean_text_features import (
    FEATURE_SET_NAME,
    TEXT_FEATURE_DIM,
    build_clean_feature_pkl,
    mean_pool_without_padding_or_special,
    verify_sha256,
)
from scripts.run_experiment_pipeline import (
    PipelineLock,
    resolve_current_run_info,
)
from scripts.train_mmgcn import build_model as build_mmgcn
from utils.evaluation import build_prediction_row, compute_calibration_metrics
from utils.iemocap_features import (
    UNPINNED_SHA256,
    load_iemocap_feature_registry,
    validate_iemocap_feature_config,
)
from utils.run_metadata import compute_file_sha256


ROOT = Path(__file__).resolve().parents[1]
CLEAN_FEATURE_NAME = "iemocap_clean_roberta_base_utterance_mean_v1"
CLEAN_FEATURE_PATH = (
    "data/processed/iemocap/"
    "IEMOCAP_features_clean_roberta_base_c8b8a37_utterance_mean_v1.pkl"
)
CLEAN_FEATURE_SHA256 = "c604c557bc3fbb129ca02b2acd57166b669a89ef76ff0cea1e14f9cb206324bf"
CLEAN_CONFIGS = (
    ROOT / "configs/mmgcn/unified/iemocap/causal_context/clean_roberta_features/smoke_real_2epoch.yaml",
    ROOT / "configs/baselines/multidag_cl/iemocap/clean_roberta_v1/smoke_real_2epoch.yaml",
    ROOT / "configs/baselines/gsmcc/iemocap/clean_roberta_v1/smoke_real_2epoch.yaml",
    ROOT / "configs/baselines/dialoguegcn/iemocap/clean_roberta_v1/smoke_real_2epoch.yaml",
)
CLEAN_FORMAL_PIPELINE_CONFIGS = tuple(
    ROOT / f"configs/pipeline/{family}/iemocap/clean_roberta_v1/val_ses0{session}.yaml"
    for family in ("mmgcn", "multidag_cl")
    for session in range(1, 5)
)
CLEAN_FORMAL_TRAIN_CONFIGS = tuple(
    ROOT
    / (
        f"configs/mmgcn/unified/iemocap/causal_context/"
        f"clean_roberta_features/val_ses0{session}.yaml"
        if family == "mmgcn"
        else f"configs/baselines/{family}/iemocap/"
        f"clean_roberta_v1/val_ses0{session}.yaml"
    )
    for family in ("mmgcn", "multidag_cl")
    for session in range(1, 5)
)
CLEAN_FORMAL_MANIFEST = ROOT / "configs/experiments/iemocap_clean_roberta_v1_8run.yaml"
FORBIDDEN_FORMAL_CAPS = (
    "train_batch_cap",
    "val_batch_cap",
    "test_batch_cap",
    "max_train_batches",
    "max_val_batches",
    "max_test_batches",
)


def fake_nine_item_pkl() -> list[object]:
    dialogue_ids = ["Ses01F_impro01", "Ses05M_script01"]
    video_ids = {
        dialogue_ids[0]: ["Ses01F_impro01_F000", "Ses01F_impro01_F001"],
        dialogue_ids[1]: ["Ses05M_script01_M000"],
    }
    speakers = {dialogue_ids[0]: ["F", "F"], dialogue_ids[1]: ["M"]}
    labels = {dialogue_ids[0]: [0, 1], dialogue_ids[1]: [2]}
    text = {
        dialogue_ids[0]: np.arange(8, dtype=np.float32).reshape(2, 4),
        dialogue_ids[1]: np.arange(4, dtype=np.float32).reshape(1, 4),
    }
    audio = {
        dialogue_ids[0]: np.arange(6, dtype=np.float64).reshape(2, 3),
        dialogue_ids[1]: np.arange(3, dtype=np.float64).reshape(1, 3),
    }
    visual = {
        dialogue_ids[0]: np.arange(4, dtype=np.float32).reshape(2, 2),
        dialogue_ids[1]: np.arange(2, dtype=np.float32).reshape(1, 2),
    }
    sentences = {
        dialogue_ids[0]: ["hello there", "   "],
        dialogue_ids[1]: ["one two three four five six"],
    }
    return [
        video_ids,
        speakers,
        labels,
        text,
        audio,
        visual,
        sentences,
        [dialogue_ids[0]],
        [dialogue_ids[1]],
    ]


def dump_pickle(path: Path, value: object) -> str:
    with path.open("wb") as file:
        pickle.dump(value, file, protocol=pickle.HIGHEST_PROTOCOL)
    return compute_file_sha256(path)


class FakeTokenizer:
    unk_token = "[UNK]"
    pad_token_id = 0

    def __call__(
        self,
        texts,
        *,
        add_special_tokens=True,
        truncation=False,
        padding=False,
        max_length=None,
        return_special_tokens_mask=False,
        return_tensors=None,
    ):
        rows = []
        masks = []
        for text in texts:
            body = [10 + index for index, _ in enumerate(str(text).split())]
            ids = ([101] + body + [102]) if add_special_tokens else body
            if truncation and max_length is not None and len(ids) > max_length:
                ids = ids[:max_length]
                ids[-1] = 102
            special = [0] * len(ids)
            if add_special_tokens:
                special[0] = 1
                special[-1] = 1
            rows.append(ids)
            masks.append(special)
        if return_tensors is None:
            return {"input_ids": rows}
        width = max(len(row) for row in rows)
        padded_ids = []
        attention = []
        padded_special = []
        for ids, special in zip(rows, masks):
            pad = width - len(ids)
            padded_ids.append(ids + [0] * pad)
            attention.append([1] * len(ids) + [0] * pad)
            padded_special.append(special + [1] * pad)
        result = {
            "input_ids": torch.tensor(padded_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention, dtype=torch.long),
        }
        if return_special_tokens_mask:
            result["special_tokens_mask"] = torch.tensor(
                padded_special, dtype=torch.long
            )
        return result


class FakeModel(torch.nn.Module):
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        offsets = torch.arange(TEXT_FEATURE_DIM, device=input_ids.device).float() / 1000.0
        hidden = input_ids.float().unsqueeze(-1) + offsets
        return SimpleNamespace(last_hidden_state=hidden)


def test_mean_pool_excludes_padding_and_special_tokens() -> None:
    hidden = torch.tensor([[[100.0], [2.0], [4.0], [999.0], [777.0]]])
    attention = torch.tensor([[1, 1, 1, 1, 0]])
    special = torch.tensor([[1, 0, 0, 1, 1]])
    pooled = mean_pool_without_padding_or_special(hidden, attention, special)
    torch.testing.assert_close(pooled, torch.tensor([[3.0]]))

    with pytest.raises(ValueError, match="zero valid tokens"):
        mean_pool_without_padding_or_special(
            hidden[:, :2], torch.ones(1, 2), torch.ones(1, 2)
        )


def test_offline_builder_preserves_non_text_items_and_writes_metadata(
    tmp_path: Path,
) -> None:
    legacy_path = tmp_path / "legacy.pkl"
    legacy = fake_nine_item_pkl()
    legacy_sha = dump_pickle(legacy_path, legacy)
    output_path = tmp_path / "clean.pkl"
    metadata_path = tmp_path / "clean.metadata.json"
    sha_path = tmp_path / "clean.sha256"

    metadata = build_clean_feature_pkl(
        input_pkl=legacy_path,
        expected_input_sha256=legacy_sha,
        output_pkl=output_path,
        tokenizer=FakeTokenizer(),
        model=FakeModel(),
        model_local_path=tmp_path / "local_roberta",
        batch_size=2,
        max_length=5,
        device="cpu",
        seed=17,
        metadata_output=metadata_path,
        sha256_output=sha_path,
        transformers_version="fake",
    )

    with output_path.open("rb") as file:
        candidate = pickle.load(file)
    for index in (0, 1, 2, 4, 5, 6, 7, 8):
        if index in (4, 5):
            for dialogue_id in legacy[index]:
                np.testing.assert_array_equal(
                    candidate[index][dialogue_id], legacy[index][dialogue_id]
                )
                assert candidate[index][dialogue_id].dtype == legacy[index][dialogue_id].dtype
        else:
            assert candidate[index] == legacy[index]
    for value in candidate[3].values():
        assert value.dtype == np.float32
        assert value.shape[1] == 768
        assert np.isfinite(value).all()

    required = {
        "schema_version",
        "feature_set_name",
        "created_at",
        "source_pkl_path",
        "source_pkl_sha256",
        "output_pkl_sha256",
        "model_id",
        "model_revision",
        "model_local_path",
        "transformers_version",
        "torch_version",
        "python_version",
        "pooling",
        "max_length",
        "batch_size",
        "dtype",
        "text_feature_dim",
        "audio_feature_dim",
        "visual_feature_dim",
        "dialogue_count",
        "utterance_count",
        "empty_sentence_count",
        "truncated_sentence_count",
        "device_used",
        "seed",
        "causality_scope",
        "label_usage",
    }
    assert required <= set(metadata)
    assert metadata["feature_set_name"] == FEATURE_SET_NAME
    assert metadata["empty_sentence_count"] == 1
    assert metadata["truncated_sentence_count"] == 1
    assert json.loads(metadata_path.read_text(encoding="utf-8")) == metadata
    assert sha_path.read_text(encoding="utf-8").split()[0] == compute_file_sha256(
        output_path
    )

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        build_clean_feature_pkl(
            input_pkl=legacy_path,
            expected_input_sha256=legacy_sha,
            output_pkl=output_path,
            tokenizer=FakeTokenizer(),
            model=FakeModel(),
            model_local_path=tmp_path,
            device="cpu",
        )


def test_builder_rejects_legacy_sha_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "legacy.pkl"
    dump_pickle(path, fake_nine_item_pkl())
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        verify_sha256(path, "0" * 64)


def test_strict_audit_detects_nan_and_writes_all_outputs(tmp_path: Path) -> None:
    legacy_path = tmp_path / "legacy.pkl"
    candidate_path = tmp_path / "candidate.pkl"
    legacy = fake_nine_item_pkl()
    legacy_sha = dump_pickle(legacy_path, legacy)
    candidate = copy.deepcopy(legacy)
    candidate[3] = {
        key: np.ones((len(value), 768), dtype=np.float32)
        for key, value in legacy[0].items()
    }
    dump_pickle(candidate_path, candidate)
    summary, dialogue_rows, statistics_rows = audit_feature_pkls(
        legacy_path,
        candidate_path,
        expected_legacy_sha256=legacy_sha,
        expected_text_dim=768,
    )
    assert summary["passed"] is True
    assert summary["checks"]["only_videoText_changed"] is True
    assert summary["checks"]["videoAudio_values_identical"] is True
    assert summary["checks"]["videoVisual_values_identical"] is True

    candidate[3]["Ses01F_impro01"][0, 0] = np.nan
    dump_pickle(candidate_path, candidate)
    failed, dialogue_rows, statistics_rows = audit_feature_pkls(
        legacy_path,
        candidate_path,
        expected_legacy_sha256=legacy_sha,
        expected_text_dim=768,
    )
    assert failed["passed"] is False
    assert failed["checks"]["candidate_videoText_all_finite"] is False
    output_dir = tmp_path / "audit"
    write_audit_outputs(output_dir, failed, dialogue_rows, statistics_rows)
    assert {path.name for path in output_dir.iterdir()} == {
        "feature_audit_report.md",
        "feature_audit_summary.json",
        "dialogue_shape_audit.csv",
        "feature_statistics.csv",
    }


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def nested_values_for_key(value: object, wanted_key: str) -> list[object]:
    matches: list[object] = []
    if isinstance(value, dict):
        for key, child in value.items():
            if key == wanted_key:
                matches.append(child)
            matches.extend(nested_values_for_key(child, wanted_key))
    elif isinstance(value, list):
        for child in value:
            matches.extend(nested_values_for_key(child, wanted_key))
    return matches


def test_config_tree_accepts_frozen_registry_smoke_and_formal_configs() -> None:
    errors = []
    validate_clean_roberta_v1_tree(errors)
    assert errors == []

    explicit_false = load_yaml(CLEAN_CONFIGS[0])
    explicit_false["dataset"]["allow_unpinned_feature_for_smoke"] = False
    validate_clean_roberta_v1_smoke_config(
        CLEAN_CONFIGS[0], "mmgcn", explicit_false, errors
    )
    assert errors == []


def test_eight_formal_configs_and_manifest_lock_the_run_contract() -> None:
    assert len(CLEAN_FORMAL_PIPELINE_CONFIGS) == 8
    assert len(set(CLEAN_FORMAL_PIPELINE_CONFIGS)) == 8
    assert all(path.is_file() for path in CLEAN_FORMAL_PIPELINE_CONFIGS)
    assert all(path.is_file() for path in CLEAN_FORMAL_TRAIN_CONFIGS)

    model_session_pairs = []
    for pipeline_path, train_path in zip(
        CLEAN_FORMAL_PIPELINE_CONFIGS, CLEAN_FORMAL_TRAIN_CONFIGS
    ):
        pipeline = load_yaml(pipeline_path)
        training = load_yaml(train_path)
        dataset = pipeline["dataset"]
        model = pipeline["model"]
        protocol = pipeline["protocol"]
        expected_session = f"Ses0{int(pipeline_path.stem[-2:])}"

        assert dataset["feature_pkl_path"] == CLEAN_FEATURE_PATH
        assert dataset["feature_set_name"] == CLEAN_FEATURE_NAME
        assert dataset["feature_sha256"] == CLEAN_FEATURE_SHA256
        assert dataset["val_split_strategy"] == "session_holdout"
        assert dataset["val_session_id"] == expected_session
        assert dataset["test_session_id"] == "Ses05"
        assert model["text_feature_dim"] == 768
        assert model["audio_feature_dim"] == 1582
        assert model["visual_feature_dim"] == 342
        assert protocol["seed"] == 42
        assert protocol["epochs"] == 30
        assert protocol["checkpoint_selection_metric"] == "val_weighted_f1"
        assert protocol["test_split_used_for_selection"] is False
        assert pipeline["evaluation"]["checkpoint_name"] == "best_model.pt"
        assert pipeline["evaluation"]["checkpoint_selected_by"] == "val_weighted_f1"
        assert pipeline["evaluation"]["test_split_used_for_selection"] is False
        assert ROOT / pipeline["train"]["train_config_path"] == train_path

        assert training["system"]["seed"] == 42
        assert training["dataset"]["val_session_id"] == expected_session
        assert training["dataset"]["test_session_id"] == "Ses05"
        assert training["model"]["text_feature_dim"] == 768
        assert training["graph"]["window_future"] == 0
        assert training["protocol"]["test_split_used_for_selection"] is False
        if model["name"] == "MMGCN":
            assert training["train"]["max_epochs"] == 30
            assert training["logging"]["monitor_metric"] == "val_weighted_f1"
        else:
            assert training["training"]["epochs"] == 30
            assert training["training"]["select_best_by"] == "val_weighted_f1"

        for config in (pipeline, training):
            for cap_name in FORBIDDEN_FORMAL_CAPS:
                assert nested_values_for_key(config, cap_name) == []
            rendered = yaml.safe_dump(config).lower()
            assert "placeholder" not in rendered
            assert "to_be_filled" not in rendered
            assert "smoke" not in rendered
            assert "quick" not in rendered
            assert "debug" not in rendered
        model_session_pairs.append((model["name"], expected_session))

    assert Counter(model for model, _ in model_session_pairs) == Counter(
        {"MMGCN": 4, "MultiDAGCL": 4}
    )
    assert Counter(model_session_pairs) == Counter(
        (model, session)
        for model in ("MMGCN", "MultiDAGCL")
        for session in ("Ses01", "Ses02", "Ses03", "Ses04")
    )

    manifest = load_yaml(CLEAN_FORMAL_MANIFEST)
    assert manifest["expected_run_count"] == 8
    assert manifest["execution"] == {"mode": "sequential", "max_concurrent_runs": 1}
    assert manifest["feature"]["path"] == CLEAN_FEATURE_PATH
    assert manifest["feature"]["sha256"] == CLEAN_FEATURE_SHA256
    assert manifest["feature"]["text_dim"] == 768
    assert manifest["protocol"]["checkpoint_selection_split"] == "validation"
    assert manifest["protocol"]["test_split_used_for_selection"] is False
    assert manifest["protocol"]["test_session"] == "Ses05"
    assert [run["config_path"] for run in manifest["runs"]] == [
        path.relative_to(ROOT).as_posix() for path in CLEAN_FORMAL_PIPELINE_CONFIGS
    ]
    assert all(run["seed"] == 42 and run["epochs"] == 30 for run in manifest["runs"])
    assert all(run["feature_sha256"] == CLEAN_FEATURE_SHA256 for run in manifest["runs"])
    assert all(run["test_split_used_for_selection"] is False for run in manifest["runs"])


def test_formal_validator_rejects_drift_caps_and_test_selection() -> None:
    train_path = CLEAN_FORMAL_TRAIN_CONFIGS[0]
    training = load_yaml(train_path)
    training["dataset"]["feature_sha256"] = UNPINNED_SHA256
    training["model"]["text_feature_dim"] = 100
    training["system"]["seed"] = 7
    training["train"]["max_epochs"] = 29
    training["train"]["max_train_batches"] = 1
    training["protocol"]["test_split_used_for_selection"] = True
    errors = []
    validate_clean_roberta_v1_formal_training(
        train_path, "mmgcn", training, errors
    )
    assert any("clean feature SHA mismatch" in error for error in errors)
    assert any("text dim mismatch" in error for error in errors)
    assert any("seed mismatch" in error for error in errors)
    assert any("epochs mismatch" in error for error in errors)
    assert any("max_train_batches" in error for error in errors)
    assert any("exclude test from selection" in error for error in errors)

    pipeline_path = CLEAN_FORMAL_PIPELINE_CONFIGS[4]
    pipeline = load_yaml(pipeline_path)
    pipeline["evaluation"]["checkpoint_selected_by"] = "test_weighted_f1"
    pipeline["evaluation"]["max_test_batches"] = 1
    pipeline["protocol"]["test_split_used_for_selection"] = True
    errors = []
    validate_clean_roberta_v1_formal_pipeline(
        pipeline_path, "multidag_cl", pipeline, errors
    )
    assert any("validation-selected" in error for error in errors)
    assert any("max_test_batches" in error for error in errors)
    assert any("exclude test from selection" in error for error in errors)


def test_config_tree_rejects_wrong_registry_sha() -> None:
    registry = load_yaml(ROOT / "configs/_shared/data/iemocap/feature_sets.yaml")
    registry["clean_roberta_v1"]["sha256"] = "0" * 64
    errors = []
    validate_clean_roberta_v1_registry(registry, errors)
    assert "clean v1 registry SHA mismatch" in errors


@pytest.mark.parametrize("invalid_sha", ("0" * 64, UNPINNED_SHA256))
def test_config_tree_rejects_wrong_or_placeholder_smoke_sha(
    invalid_sha: str,
) -> None:
    config = load_yaml(CLEAN_CONFIGS[0])
    config["dataset"]["feature_sha256"] = invalid_sha
    errors = []
    validate_clean_roberta_v1_smoke_config(
        CLEAN_CONFIGS[0], "mmgcn", config, errors
    )
    assert any("clean feature SHA mismatch" in error for error in errors)


def test_config_tree_rejects_enabled_unpinned_smoke_flag() -> None:
    config = load_yaml(CLEAN_CONFIGS[0])
    config["dataset"]["allow_unpinned_feature_for_smoke"] = True
    errors = []
    validate_clean_roberta_v1_smoke_config(
        CLEAN_CONFIGS[0], "mmgcn", config, errors
    )
    assert any("must not enable unpinned clean features" in error for error in errors)


@pytest.mark.parametrize(
    ("family", "config_path"),
    tuple(
        zip(
            ("mmgcn", "multidag_cl", "gsmcc", "dialoguegcn"),
            CLEAN_CONFIGS,
        )
    ),
)
def test_config_tree_rejects_non_768_text_feature_dim(
    family: str, config_path: Path
) -> None:
    config = load_yaml(config_path)
    if family in {"mmgcn", "multidag_cl"}:
        config["model"]["text_feature_dim"] = 100
    else:
        config["model"]["text_dim"] = 100
    errors = []
    validate_clean_roberta_v1_smoke_config(config_path, family, config, errors)
    assert any("text" in error and "dim mismatch" in error for error in errors)


def test_registry_and_four_clean_configs_use_frozen_sha_and_768_dims() -> None:
    registry = load_iemocap_feature_registry(ROOT)
    clean_entry = registry["clean_roberta_v1"]
    assert clean_entry["feature_set_name"] == CLEAN_FEATURE_NAME
    assert clean_entry["path"] == CLEAN_FEATURE_PATH
    assert clean_entry["sha256"] == CLEAN_FEATURE_SHA256
    assert clean_entry["text_dim"] == 768
    assert clean_entry["status"] == "frozen_for_main_experiments"

    configs = [load_yaml(path) for path in CLEAN_CONFIGS]
    for config in configs:
        dataset = config["dataset"]
        assert dataset["feature_pkl_path"] == CLEAN_FEATURE_PATH
        assert dataset["feature_set_name"] == CLEAN_FEATURE_NAME
        assert dataset["feature_sha256"] == CLEAN_FEATURE_SHA256
        assert "allow_unpinned_feature_for_smoke" not in dataset
        validate_iemocap_feature_config(config, ROOT, require_file=False)

    mmgcn = build_mmgcn(configs[0])
    multidag = build_multidag(configs[1])
    gsmcc = build_new_causal_baseline(configs[2])
    dialoguegcn = build_new_causal_baseline(configs[3])
    assert mmgcn.text_fc.in_features == 768
    assert multidag.text_dim == 768
    assert gsmcc.config.text_dim == 768
    assert dialoguegcn.config.text_dim == 768

    unpinned = copy.deepcopy(configs[0])
    unpinned["dataset"]["feature_sha256"] = UNPINNED_SHA256
    with pytest.raises(ValueError, match="Refusing unpinned"):
        validate_iemocap_feature_config(unpinned, ROOT, require_file=False)


def test_smoke_and_formal_validation_both_hash_actual_pkl(tmp_path: Path) -> None:
    feature_path = tmp_path / CLEAN_FEATURE_PATH
    feature_path.parent.mkdir(parents=True)
    feature_path.write_bytes(b"audited-clean-feature-test-double")
    actual_sha256 = compute_file_sha256(feature_path)

    registry_path = tmp_path / "configs/_shared/data/iemocap/feature_sets.yaml"
    registry_path.parent.mkdir(parents=True)
    registry_path.write_text(
        yaml.safe_dump(
            {
                "clean_roberta_v1": {
                    "feature_set_name": CLEAN_FEATURE_NAME,
                    "path": CLEAN_FEATURE_PATH,
                    "sha256": actual_sha256,
                    "text_dim": 768,
                    "status": "frozen_for_main_experiments",
                }
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    base_config = {
        "dataset": {
            "name": "IEMOCAP",
            "feature_pkl_path": CLEAN_FEATURE_PATH,
            "feature_set_name": CLEAN_FEATURE_NAME,
            "feature_sha256": actual_sha256,
        },
        "model": {"text_feature_dim": 768},
    }
    for run_kind in ("smoke", "formal"):
        config = copy.deepcopy(base_config)
        config["project"] = {"experiment_name": f"{run_kind}_hash_validation"}
        assert validate_iemocap_feature_config(config, tmp_path) == actual_sha256

    feature_path.write_bytes(b"mutated-feature-test-double")
    for run_kind in ("smoke", "formal"):
        config = copy.deepcopy(base_config)
        config["project"] = {"experiment_name": f"{run_kind}_hash_validation"}
        with pytest.raises(ValueError, match="SHA256 mismatch"):
            validate_iemocap_feature_config(config, tmp_path)


@pytest.mark.parametrize(
    ("config_path", "trainer_kind"),
    (
        (CLEAN_CONFIGS[0], "mmgcn"),
        (CLEAN_CONFIGS[1], "multidag_cl"),
        (CLEAN_CONFIGS[2], "causal"),
        (CLEAN_CONFIGS[3], "causal"),
    ),
)
def test_sha_mismatch_fails_before_model_initialization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    config_path: Path,
    trainer_kind: str,
) -> None:
    feature_path = tmp_path / "mismatched.pkl"
    feature_path.write_bytes(b"not-the-frozen-clean-v1-pkl")
    config = load_yaml(config_path)
    config["dataset"]["feature_pkl_path"] = str(feature_path)
    config["dataset"].pop("feature_set_name", None)
    config["dataset"]["feature_sha256"] = CLEAN_FEATURE_SHA256
    temporary_config = tmp_path / f"{config_path.parent.parent.parent.parent.name}.yaml"
    temporary_config.write_text(
        yaml.safe_dump(config, sort_keys=False),
        encoding="utf-8",
    )

    model_initializations = []

    def forbidden_model_initialization(*args, **kwargs):
        model_initializations.append((args, kwargs))
        raise AssertionError("model initialization must not precede feature SHA validation")

    if trainer_kind == "mmgcn":
        monkeypatch.setattr(train_mmgcn_module, "build_model", forbidden_model_initialization)
        monkeypatch.setattr(sys, "argv", ["train_mmgcn.py", "--config", str(temporary_config)])
        invocation = train_mmgcn_module.main
    elif trainer_kind == "multidag_cl":
        monkeypatch.setattr(train_multidag_module, "build_model", forbidden_model_initialization)
        monkeypatch.setattr(
            sys,
            "argv",
            ["train_multidag_cl.py", "--config", str(temporary_config)],
        )
        invocation = train_multidag_module.main
    else:
        monkeypatch.setattr(
            train_causal_module,
            "build_new_causal_baseline",
            forbidden_model_initialization,
        )
        invocation = lambda: train_causal_module.run_training(temporary_config)

    with pytest.raises(ValueError, match="SHA256 mismatch"):
        invocation()
    assert model_initializations == []


def test_pipeline_lock_is_atomic_cleans_up_and_force_unlocks(tmp_path: Path) -> None:
    path = tmp_path / "benchmark.lock"
    first = PipelineLock(path, Path("batch.yaml"))
    second = PipelineLock(path, Path("batch.yaml"))
    first.acquire()
    assert path.is_file()
    with pytest.raises(RuntimeError, match="already locked"):
        second.acquire()
    second.acquire(force_unlock=True)
    second.release()
    assert not path.exists()
    first.release()


def test_pipeline_uses_exact_subprocess_run_info(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "run-a"
    run_dir.mkdir(parents=True)
    info = resolve_current_run_info(
        {},
        train_was_executed=True,
        dry_run=False,
        training_run_info={"run_id": "run-a", "run_dir": str(run_dir)},
    )
    assert info == {"run_id": "run-a", "run_dir": str(run_dir.resolve())}


def test_calibration_metrics_and_prediction_schema() -> None:
    probabilities = [[0.8, 0.2], [0.7, 0.3]]
    metrics = compute_calibration_metrics([0, 1], probabilities)
    assert metrics["ece_10"] == pytest.approx(0.45)
    assert metrics["nll"] == pytest.approx(-np.log([0.8, 0.3]).mean())
    assert metrics["brier_score"] == pytest.approx(0.53)
    assert metrics["mean_confidence"] == pytest.approx(0.75)
    assert metrics["mean_confidence_correct"] == pytest.approx(0.8)
    assert metrics["mean_confidence_incorrect"] == pytest.approx(0.7)

    row = build_prediction_row(
        split="test",
        dialogue_id="Ses05F_impro01",
        utterance_id="Ses05F_impro01_F000",
        utterance_index=0,
        true_label_id=1,
        predicted_label_id=0,
        probabilities=probabilities[1],
        label_list=["Happy", "Sad"],
    )
    required = {
        "confidence",
        "probability_0",
        "probability_1",
        "predicted_label",
        "true_label",
        "dialogue_id",
        "utterance_id",
        "utterance_index",
        "session_id",
        "dialogue_type",
    }
    assert required <= set(row)
    assert row["session_id"] == "Ses05"
    assert row["dialogue_type"] == "impro"


def test_session_probes_use_all_five_held_out_sessions() -> None:
    rows = []
    for session_index, session_id in enumerate(
        ("Ses01", "Ses02", "Ses03", "Ses04", "Ses05"), start=1
    ):
        for label_id, dialogue_type in ((0, "impro"), (1, "script")):
            feature = np.array(
                [float(label_id), float(session_index), float(label_id + session_index), 1.0],
                dtype=np.float32,
            )
            rows.append(
                {
                    "dialogue_id": f"{session_id}F_{dialogue_type}01",
                    "utterance_id": f"{session_id}_{label_id}",
                    "utterance_index": 0,
                    "session_id": session_id,
                    "dialogue_type": dialogue_type,
                    "label_id": label_id,
                    "feature": feature,
                }
            )
    metrics, distribution, duplicates = run_probes(
        pd.DataFrame(rows), seed=11, max_iter=200
    )
    assert set(metrics["held_out_session"]) == {
        "Ses01",
        "Ses02",
        "Ses03",
        "Ses04",
        "Ses05",
    }
    assert set(metrics["method"]) == {
        "linear_logistic_regression",
        "nearest_class_centroid",
        "cross_session_cosine_1nn",
    }
    assert set(metrics["dialogue_type"]) == {"all", "impro", "script"}
    assert len(distribution) == 10
    assert len(duplicates) == 5

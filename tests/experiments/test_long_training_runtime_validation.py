from __future__ import annotations

import copy
from collections import Counter
from pathlib import Path
import shutil
import tempfile

import pytest
import yaml

import scripts.models.mmgcn.unified.train as mmgcn_train
import scripts.models.multidag_cl.unified.train as multidag_train
import scripts.runtime.causal_graph as causal_runtime
import scripts.runtime.paper_aligned as paper_runtime
from scripts.workflows.benchmarks.prepare_long_training import (
    load_yaml,
    prepare_long_training_matrix,
)
from utils.iemocap_features import validate_iemocap_feature_config


ROOT = Path(__file__).resolve().parents[2]
MATRIX_PATH = Path(
    "configs/benchmarks/long_training/iemocap_clean/primary_seed42.yaml"
)
SESSION_TRACK = "clean_roberta_session_holdout_fair_comparison"
SESSION_STRATEGY = "session_holdout"
PROTOCOL_COMPARABILITY = "fair_comparison_not_paper_reproduction"
PROTOCOL_VERSION = "long_training_session_holdout_v1"


@pytest.fixture
def workspace_tmp_dir() -> Path:
    temp_parent = ROOT / "tmp"
    temp_parent.mkdir(parents=True, exist_ok=True)
    path = Path(tempfile.mkdtemp(prefix="pytest_long_training_", dir=temp_parent))
    try:
        yield path
    finally:
        shutil.rmtree(path)


def _load_resolved(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    assert isinstance(config, dict)
    return config


def _validate_for_entrypoint(
    config: dict,
    entrypoint: str,
    *,
    fake_feature_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    validate_iemocap_feature_config(config, ROOT, require_file=False)
    if entrypoint == "scripts/workflows/paper_aligned/train.py":
        normalized = paper_runtime.normalized_training_config(config)
        paper_runtime.validate_runtime_config(normalized)
        return
    if entrypoint == "scripts/workflows/causal_graph/train.py":
        normalized = causal_runtime.normalized_training_config(config)
        causal_runtime.validate_runtime_config(normalized)
        return
    if entrypoint == "scripts/models/multidag_cl/unified/train.py":
        monkeypatch.setattr(
            multidag_train,
            "validate_iemocap_feature_config",
            lambda value, project_root: validate_iemocap_feature_config(
                value, project_root, require_file=False
            ),
        )
        monkeypatch.setattr(
            multidag_train, "resolve_path", lambda path_text: fake_feature_path
        )
        multidag_train.validate_config(config)
        return
    if entrypoint == "scripts/models/mmgcn/unified/train.py":
        mmgcn_train.build_model(config)
        return
    raise AssertionError(f"Unhandled long-training entrypoint: {entrypoint}")


def test_primary_long32_prepare_and_runtime_validation(
    workspace_tmp_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    resolved_root = workspace_tmp_dir / "resolved_configs"
    result = prepare_long_training_matrix(
        MATRIX_PATH,
        "prepare",
        "20260725",
        root=ROOT,
        resolved_root=resolved_root,
    )

    records = result["records"]
    commands = result["commands"]
    resolved_paths = sorted(resolved_root.glob("*.yaml"))
    assert result["expanded_run_count"] == 32
    assert len(records) == 32
    assert len(resolved_paths) == 32
    assert len(commands) == 32
    assert len(set(commands)) == 32
    assert result["pair_key_count"] == 16
    assert result["unpaired_context_run_count"] == 0
    assert result["duplicate_pair_member_count"] == 0
    assert result["output_collision_count"] == 0
    assert result["context_counts"] == Counter(
        {"full_context": 16, "causal_context": 16}
    )
    assert result["model_counts"] == Counter(
        {
            "mmgcn": 8,
            "multidag_cl": 8,
            "dialoguegcn": 8,
            "gsmcc_project_variant": 8,
        }
    )

    run_ids: set[str] = set()
    output_roots: set[str] = set()
    pair_members: dict[tuple[str, str, str, str, str, int], Counter[str]] = {}
    validation_sessions: Counter[str] = Counter()
    entrypoint_counts: Counter[str] = Counter()
    validation_pass_count = 0
    fake_feature_path = workspace_tmp_dir / "remote_feature_placeholder.pkl"
    fake_feature_path.touch()

    for record, resolved_path in zip(records, resolved_paths):
        config = _load_resolved(resolved_path)
        dataset = config["dataset"]
        long_training = config["long_training"]
        protocol = config["protocol"]
        run_id = record["run_id"]
        output_root = config["output"]["root"]
        entrypoint = record["entrypoint"]
        test_session = dataset.get(
            "test_session_id", dataset.get("outer_test_session")
        )

        assert resolved_path == record["resolved_path"]
        assert run_id in resolved_path.name
        assert entrypoint in record["command"]
        assert resolved_path.as_posix() in record["command"]
        assert dataset["experiment_track"] == SESSION_TRACK
        assert dataset["val_split_strategy"] == SESSION_STRATEGY
        assert dataset["protocol_comparability"] == PROTOCOL_COMPARABILITY
        assert config["protocol_version"] == PROTOCOL_VERSION
        assert dataset["val_session_id"] in {"Ses01", "Ses02", "Ses03", "Ses04"}
        assert test_session == "Ses05"
        assert protocol["checkpoint_selection_metric"] == "val_weighted_f1"
        assert protocol["test_split_used_for_selection"] is False

        _validate_for_entrypoint(
            config,
            entrypoint,
            fake_feature_path=fake_feature_path,
            monkeypatch=monkeypatch,
        )
        validation_pass_count += 1

        run_ids.add(run_id)
        output_roots.add(output_root)
        validation_sessions[dataset["val_session_id"]] += 1
        entrypoint_counts[entrypoint] += 1
        pair_key = (
            long_training["model_family"],
            long_training["protocol_lineage"],
            dataset["feature_set_name"],
            dataset["val_session_id"],
            str(test_session),
            int(config["system"]["seed"]),
        )
        pair_members.setdefault(pair_key, Counter())[
            long_training["context_mode"]
        ] += 1

    assert validation_pass_count == 32
    assert len(run_ids) == 32
    assert len(output_roots) == 32
    assert validation_sessions == Counter(
        {"Ses01": 8, "Ses02": 8, "Ses03": 8, "Ses04": 8}
    )
    assert entrypoint_counts == Counter(
        {
            "scripts/workflows/paper_aligned/train.py": 16,
            "scripts/workflows/causal_graph/train.py": 8,
            "scripts/models/mmgcn/unified/train.py": 4,
            "scripts/models/multidag_cl/unified/train.py": 4,
        }
    )
    assert len(pair_members) == 16
    assert all(
        members == Counter({"full_context": 1, "causal_context": 1})
        for members in pair_members.values()
    )


def test_long_training_base_and_parameter_source_protocols_are_explicit() -> None:
    matrix = load_yaml(ROOT / MATRIX_PATH)
    assert matrix["protocol"]["experiment_track"] == SESSION_TRACK
    assert matrix["protocol"]["val_split_strategy"] == SESSION_STRATEGY
    assert matrix["protocol"]["protocol_version"] == PROTOCOL_VERSION
    assert (
        matrix["protocol"]["protocol_comparability"] == PROTOCOL_COMPARABILITY
    )

    source_protocols: Counter[tuple[object, object]] = Counter()
    for record in matrix["base_configs"]:
        base = load_yaml(ROOT / record["base_config"])
        source = load_yaml(ROOT / record["parameter_source"])
        dataset = base["dataset"]
        assert base["protocol_version"] == PROTOCOL_VERSION
        assert dataset["experiment_track"] == SESSION_TRACK
        assert dataset["val_split_strategy"] == SESSION_STRATEGY
        assert dataset["protocol_comparability"] == PROTOCOL_COMPARABILITY
        assert base["long_training"]["parameter_source"] == record["parameter_source"]
        assert base["long_training"]["entrypoint"] == record["entrypoint"]
        source_dataset = source["dataset"]
        source_protocols[
            (
                source_dataset.get("experiment_track"),
                source_dataset.get("val_split_strategy"),
            )
        ] += 1

    assert source_protocols == Counter(
        {
            ("clean_roberta_fivefold_fair_comparison", "outer_session_stratified"): 4,
            (None, "session_holdout"): 4,
        }
    )


def test_paper_runtime_track_split_allowlist_remains_strict() -> None:
    assert {
        name: policy["split_strategy"]
        for name, policy in paper_runtime.EXPERIMENT_TRACKS.items()
    } == {
        "legacy_official_split_safe_selection": "official_train_stratified",
        "legacy_fivefold_fair_comparison": "outer_session_stratified",
        "clean_roberta_fivefold_fair_comparison": "outer_session_stratified",
        SESSION_TRACK: SESSION_STRATEGY,
    }

    config = load_yaml(
        ROOT
        / "configs/benchmarks/long_training/iemocap_clean/base/mmgcn_full_context.yaml"
    )
    paper_runtime.validate_runtime_config(
        paper_runtime.normalized_training_config(config)
    )
    mismatched = copy.deepcopy(config)
    mismatched["dataset"][
        "experiment_track"
    ] = "clean_roberta_fivefold_fair_comparison"
    with pytest.raises(
        ValueError, match="experiment track and split strategy do not match"
    ):
        paper_runtime.validate_runtime_config(
            paper_runtime.normalized_training_config(mismatched)
        )


def test_paper_runtime_forwards_session_holdout_validation_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_yaml(
        ROOT
        / "configs/benchmarks/long_training/iemocap_clean/base/mmgcn_full_context.yaml"
    )
    captured: dict = {}

    def fake_build_iemocap_dataloader(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(
        paper_runtime, "build_iemocap_dataloader", fake_build_iemocap_dataloader
    )
    paper_runtime.build_dataloader(config, "train", shuffle=True)
    assert captured["val_split_strategy"] == SESSION_STRATEGY
    assert captured["val_session_id"] == "Ses01"
    assert captured["outer_test_session"] == "Ses05"

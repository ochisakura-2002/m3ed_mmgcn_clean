from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

import pytest
import torch
import yaml

from models.multidag_cl.paper_reimplementation.config import MultiDAGCLConfig
from scripts.runtime.multidag_cl_paper_reimplementation import CurriculumRuntime
from scripts.runtime.multidag_cl_paper_reimplementation.adapter import (
    FeatureRegistryMetadata,
)
from scripts.runtime.multidag_cl_paper_reimplementation.trainer import _check_mode
from scripts.runtime.multidag_cl_paper_reimplementation.validation import (
    validate_runtime_config,
)
from scripts.workflows.benchmarks import run_multidag_cl_paper_data_curriculum as launcher


ROOT = Path(__file__).resolve().parents[4]
BASE_CONFIG = (
    ROOT
    / "configs/multidag_cl/paper_reimplementation/iemocap/full_context/"
    "paper_data_reproduction/paper_data_reproduction.yaml"
)
CONFIG_ROOT = BASE_CONFIG.parent / "curriculum_comparison"
CONFIGS = {
    "CL4": CONFIG_ROOT / "cl4.yaml",
    "CL7": CONFIG_ROOT / "cl7.yaml",
    "CL10": CONFIG_ROOT / "cl10.yaml",
    "CL15": CONFIG_ROOT / "cl15.yaml",
}
EXPECTED = {
    "CL4": 4,
    "CL7": 7,
    "CL10": 10,
    "CL15": 15,
}
HISTORICAL_CONFIGS = tuple(CONFIG_ROOT / name for name in ("cl3.yaml", "cl8.yaml", "cl12.yaml"))
NOCL_CONFIG = CONFIG_ROOT / "nocl.yaml"
OFFICIAL_FEATURE_SHA256 = (
    "4b3320d1d7127609abec256052827740054694b9b973586daf22923ba818fe3c"
)


def _load(path: Path) -> dict:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _without_allowed_differences(config: dict) -> dict:
    value = deepcopy(config)
    value.pop("run_name")
    value["model_core"]["curriculum"].pop("bucket_count")
    value["model_core"]["identity"].pop("ablation_profile", None)
    value["output"].pop("experiment_group")
    return value


def _feature_metadata() -> FeatureRegistryMetadata:
    return FeatureRegistryMetadata(
        registry_key="multidag_cl_official_2948_v1",
        feature_path=(
            "data/processed/iemocap/multidag_cl_paper_data_reproduction/"
            "IEMOCAP_features_multidag_cl_official.pkl"
        ),
        feature_sha256=OFFICIAL_FEATURE_SHA256,
        text_dim=1024,
        audio_dim=1582,
        visual_dim=342,
    )


def test_configs_only_change_bucket_ablation_and_run_identity_from_existing_cl5() -> None:
    baseline = _load(BASE_CONFIG)
    expected_scientific_config = _without_allowed_differences(baseline)
    run_names: set[str] = set()
    for experiment, path in CONFIGS.items():
        config = _load(path)
        assert _without_allowed_differences(config) == expected_scientific_config
        buckets = EXPECTED[experiment]
        assert config["model_core"]["identity"]["conformance_profile"] == (
            "paper_formula_behavior"
        )
        assert config["model_core"]["identity"]["ablation_profile"] == (
            "paper_curriculum_ablation"
        )
        assert config["model_core"]["curriculum"] == {
            "enabled": True,
            "bucket_count": buckets,
            "partition": "balanced_stable_contiguous",
            "schedule": "official_one_bucket_per_epoch",
        }
        assert config["output"]["experiment_group"] == (
            "multidag_cl_paper_data_curriculum"
        )
        run_names.add(config["run_name"])
    assert len(run_names) == 4
    assert baseline["model_core"]["identity"].get("ablation_profile") is None
    assert baseline["model_core"]["curriculum"] == {
        "enabled": True,
        "bucket_count": 5,
        "partition": "balanced_stable_contiguous",
        "schedule": "official_one_bucket_per_epoch",
    }


def test_strict_paper_profile_keeps_bucket_five_and_rejects_deviations() -> None:
    baseline = _load(BASE_CONFIG)
    core = MultiDAGCLConfig.from_mapping(baseline["model_core"])
    assert core.bucket_count == 5
    assert core.conformance_profile.value == "paper_formula_behavior"
    for bucket_count in (4, 7, 10, 15):
        mutated = deepcopy(baseline["model_core"])
        mutated["curriculum"]["bucket_count"] = bucket_count
        with pytest.raises(ValueError, match="paper profile bucket_count must be 5"):
            MultiDAGCLConfig.from_mapping(mutated)


def test_new_configs_validate_and_check_without_optimizer_steps_or_test_selection(
    monkeypatch,
) -> None:
    feature = _feature_metadata()
    monkeypatch.setattr(
        "scripts.runtime.multidag_cl_paper_reimplementation.validation.resolve_feature_metadata",
        lambda *args, **kwargs: feature,
    )
    for experiment, path in CONFIGS.items():
        config = _load(path)
        core, resolved_feature = validate_runtime_config(
            config,
            mode="check",
            project_root=ROOT,
            verify_checksum=False,
        )
        assert (core.curriculum_enabled, core.bucket_count) == (
            True,
            EXPECTED[experiment],
        )
        assert core.test_split_used_for_selection is False
        assert config["checkpoint"]["primary_metric"] == "val_weighted_f1"
        assert config["checkpoint"]["test_split_used_for_selection"] is False
        result = _check_mode(
            config,
            core=core,
            feature=resolved_feature,
            device=torch.device("cpu"),
        )
        assert result["status"] == "PASS"
        assert result["mode"] == "check"
        assert result["optimizer_steps"] == 0
        assert result["gradient_clip_count"] == 0
        assert result["conformance_profile"] == "paper_formula_behavior"


def test_ablation_identity_cannot_change_other_scientific_fields() -> None:
    config = _load(BASE_CONFIG)
    config["model_core"]["identity"]["ablation_profile"] = (
        "paper_curriculum_ablation"
    )
    config["model_core"]["curriculum"]["bucket_count"] = 4
    config["model_core"]["training"]["seed"] = 101
    with pytest.raises(ValueError, match="training_seed"):
        MultiDAGCLConfig.from_mapping(config["model_core"])


def test_historical_non_table_matrix_configs_are_preserved_but_fail_closed() -> None:
    for path in HISTORICAL_CONFIGS:
        assert "not part of the current IEMOCAP paper Table 4 matrix" in (
            path.read_text(encoding="utf-8")
        )
        with pytest.raises(ValueError, match="paper profile bucket_count must be 5"):
            MultiDAGCLConfig.from_mapping(_load(path)["model_core"])


def test_launcher_plan_contains_only_pending_table4_runs() -> None:
    assert launcher.RUN_PLAN == tuple(
        (experiment, CONFIGS[experiment].relative_to(ROOT))
        for experiment in ("CL4", "CL7", "CL10", "CL15")
    )
    planned_paths = {ROOT / path for _, path in launcher.RUN_PLAN}
    assert NOCL_CONFIG not in planned_paths
    assert not set(HISTORICAL_CONFIGS) & planned_paths


class _TinyTrainDataset:
    def __len__(self) -> int:
        return 13

    def __getitem__(self, index: int) -> dict:
        return {
            "dialogue_id": f"dialogue_{index:02d}",
            "labels": [index % 6, (index + 1) % 6],
            "speaker_ids_int": [0, 1],
            "length": 2,
        }


def test_nocl_uses_the_complete_train_split_from_epoch_one() -> None:
    config = _load(NOCL_CONFIG)
    core = MultiDAGCLConfig.from_mapping(config["model_core"])
    dataset = _TinyTrainDataset()
    runtime = CurriculumRuntime.from_training_dataset(
        dataset,
        split="train",
        bucket_count=core.bucket_count,
        partition_profile=core.curriculum_partition,
        schedule_profile=core.curriculum_schedule,
        enabled=core.curriculum_enabled,
    )
    assert runtime.visible_bucket_count(1) == runtime.manifest.actual_bucket_count
    assert runtime.visible_indices(1) == list(range(len(dataset)))
    assert set(runtime.visible_dialogue_ids(1)) == {
        f"dialogue_{index:02d}" for index in range(len(dataset))
    }
    assert {row.visible_from_epoch for row in runtime.rows} == {1}


def test_launcher_is_serial_continues_after_failure_and_writes_summary(
    tmp_path: Path,
) -> None:
    calls: list[list[str]] = []

    def fake_runner(command, **kwargs):
        calls.append(command)
        config = Path(command[command.index("--config") + 1])
        experiment = config.stem
        failed = experiment == "cl7"
        payload = {
            "status": "FAILED" if failed else "PASS",
            "run_id": "" if failed else f"run_{experiment}",
        }
        kwargs["stdout"].write(
            launcher.RESULT_PREFIX + json.dumps(payload) + "\n"
        )
        kwargs["stdout"].flush()
        return subprocess.CompletedProcess(command, 9 if failed else 0)

    args = launcher.parse_args(
        [
            "--experiment-date",
            "20260813",
            "--output-root",
            str(tmp_path / "outputs"),
            "--batch-id",
            "pytest_curriculum",
            "--device",
            "cuda",
        ]
    )
    fixed_now = datetime(2026, 8, 13, 12, 0, tzinfo=timezone.utc)
    summary = launcher.launch_batch(
        args,
        runner=fake_runner,
        now_provider=lambda: fixed_now,
    )

    assert [Path(call[call.index("--config") + 1]).stem for call in calls] == [
        "cl4",
        "cl7",
        "cl10",
        "cl15",
    ]
    assert all(call[call.index("--mode") + 1] == "train" for call in calls)
    assert summary["status"] == "COMPLETED_WITH_FAILURES"
    assert summary["continue_on_error"] is True
    assert [row["experiment"] for row in summary["experiments"]] == [
        "CL4",
        "CL7",
        "CL10",
        "CL15",
    ]
    assert [row["exit_code"] for row in summary["experiments"]] == [0, 9, 0, 0]
    assert [row["status"] for row in summary["experiments"]] == [
        "PASS",
        "FAILED",
        "PASS",
        "PASS",
    ]

    batch_root = (
        tmp_path
        / "outputs/20260813/multidag_cl_paper_data_curriculum"
    )
    logs = sorted((batch_root / "logs/launcher/pytest_curriculum").glob("*.log"))
    assert [path.name for path in logs] == [
        "01_cl4.log",
        "02_cl7.log",
        "03_cl10.log",
        "04_cl15.log",
    ]
    manifest = batch_root / "manifests/batches/pytest_curriculum/run_manifest.tsv"
    report = batch_root / "reports/batches/pytest_curriculum/final_summary.json"
    assert manifest.is_file()
    assert report.is_file()
    assert "experiment\tconfig\texit_code\tstatus\trun_id" in manifest.read_text(
        encoding="utf-8"
    )

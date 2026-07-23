from __future__ import annotations

import ast
import copy
import hashlib
import importlib
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]


MIGRATED_SUPPORT = (
    (
        "scripts.analyze.audit_iemocap_feature_pkl",
        "scripts.diagnostics.data.audit_iemocap_feature_pkl",
        "scripts/analyze/audit_iemocap_feature_pkl.py",
        "scripts/diagnostics/data/audit_iemocap_feature_pkl.py",
        ("--legacy-pkl", "--candidate-pkl", "--expected-legacy-sha256", "--output-dir"),
    ),
    (
        "scripts.analyze.diagnose_iemocap_splits",
        "scripts.diagnostics.data.diagnose_iemocap_splits",
        "scripts/analyze/diagnose_iemocap_splits.py",
        "scripts/diagnostics/data/diagnose_iemocap_splits.py",
        ("--config", "--output-dir", "--experiment-date"),
    ),
    (
        "scripts.analyze.diagnose_loss_stability",
        "scripts.diagnostics.experiments.diagnose_loss_stability",
        "scripts/analyze/diagnose_loss_stability.py",
        "scripts/diagnostics/experiments/diagnose_loss_stability.py",
        ("--run-id", "--output-dir", "--experiment-date"),
    ),
    (
        "scripts.analyze.diagnose_multidag_cl_run",
        "scripts.diagnostics.experiments.diagnose_multidag_cl_run",
        "scripts/analyze/diagnose_multidag_cl_run.py",
        "scripts/diagnostics/experiments/diagnose_multidag_cl_run.py",
        ("--run-id",),
    ),
    (
        "scripts.analyze.probe_iemocap_text_features",
        "scripts.diagnostics.data.probe_iemocap_text_features",
        "scripts/analyze/probe_iemocap_text_features.py",
        "scripts/diagnostics/data/probe_iemocap_text_features.py",
        ("--input-pkl", "--output-dir", "--seed", "--max-iter"),
    ),
    (
        "scripts.baselines.debug_causal_dialoguegcn_forward",
        "scripts.diagnostics.models.dialoguegcn.debug_causal_dialoguegcn_forward",
        "scripts/baselines/debug_causal_dialoguegcn_forward.py",
        "scripts/diagnostics/models/dialoguegcn/debug_causal_dialoguegcn_forward.py",
        ("--config",),
    ),
    (
        "scripts.baselines.debug_causal_gsmcc_forward",
        "scripts.diagnostics.models.gsmcc.debug_causal_gsmcc_forward",
        "scripts/baselines/debug_causal_gsmcc_forward.py",
        "scripts/diagnostics/models/gsmcc/debug_causal_gsmcc_forward.py",
        ("--config",),
    ),
    (
        "scripts.baselines.debug_multidag_cl_forward",
        "scripts.diagnostics.models.multidag_cl.debug_multidag_cl_forward",
        "scripts/baselines/debug_multidag_cl_forward.py",
        "scripts/diagnostics/models/multidag_cl/debug_multidag_cl_forward.py",
        (),
    ),
    (
        "scripts.baselines.debug_multidag_cl_real_batch",
        "scripts.diagnostics.models.multidag_cl.debug_multidag_cl_real_batch",
        "scripts/baselines/debug_multidag_cl_real_batch.py",
        "scripts/diagnostics/models/multidag_cl/debug_multidag_cl_real_batch.py",
        ("--config", "--dataset", "--split", "--batch-size-override", "--device"),
    ),
    (
        "scripts.baselines.debug_new_causal_graph_real_batch",
        "scripts.diagnostics.models.debug_new_causal_graph_real_batch",
        "scripts/baselines/debug_new_causal_graph_real_batch.py",
        "scripts/diagnostics/models/debug_new_causal_graph_real_batch.py",
        ("--config", "--split", "--device"),
    ),
    (
        "scripts.baselines.debug_sdt_forward",
        "scripts.diagnostics.models.sdt.debug_sdt_forward",
        "scripts/baselines/debug_sdt_forward.py",
        "scripts/diagnostics/models/sdt/debug_sdt_forward.py",
        (),
    ),
    (
        "scripts.debug.debug_dialogue_feature_dataset",
        "scripts.diagnostics.data.debug_dialogue_feature_dataset",
        "scripts/debug/debug_dialogue_feature_dataset.py",
        "scripts/diagnostics/data/debug_dialogue_feature_dataset.py",
        (),
    ),
    (
        "scripts.debug.debug_iemocap_dataloader",
        "scripts.diagnostics.data.debug_iemocap_dataloader",
        "scripts/debug/debug_iemocap_dataloader.py",
        "scripts/diagnostics/data/debug_iemocap_dataloader.py",
        ("--feature-pkl-path", "--batch-size", "--val-split-strategy"),
    ),
    (
        "scripts.debug.debug_io",
        "scripts.diagnostics.experiments.debug_io",
        "scripts/debug/debug_io.py",
        "scripts/diagnostics/experiments/debug_io.py",
        (),
    ),
    (
        "scripts.debug.debug_m3ed_dataloader",
        "scripts.diagnostics.data.debug_m3ed_dataloader",
        "scripts/debug/debug_m3ed_dataloader.py",
        "scripts/diagnostics/data/debug_m3ed_dataloader.py",
        (),
    ),
    (
        "scripts.debug.debug_m3ed_dataset",
        "scripts.diagnostics.data.debug_m3ed_dataset",
        "scripts/debug/debug_m3ed_dataset.py",
        "scripts/diagnostics/data/debug_m3ed_dataset.py",
        (),
    ),
    (
        "scripts.debug.debug_m3ed_feature_dataset",
        "scripts.diagnostics.data.debug_m3ed_feature_dataset",
        "scripts/debug/debug_m3ed_feature_dataset.py",
        "scripts/diagnostics/data/debug_m3ed_feature_dataset.py",
        (),
    ),
    (
        "scripts.debug.debug_mmgcn_forward",
        "scripts.diagnostics.models.mmgcn.debug_mmgcn_forward",
        "scripts/debug/debug_mmgcn_forward.py",
        "scripts/diagnostics/models/mmgcn/debug_mmgcn_forward.py",
        (),
    ),
    (
        "scripts.debug.debug_simple_mlp_step",
        "scripts.diagnostics.models.simple_mlp.debug_simple_mlp_step",
        "scripts/debug/debug_simple_mlp_step.py",
        "scripts/diagnostics/models/simple_mlp/debug_simple_mlp_step.py",
        (),
    ),
    (
        "scripts.dev.diagnose_gsmcc_numerics",
        "scripts.diagnostics.models.gsmcc.diagnose_gsmcc_numerics",
        "scripts/dev/diagnose_gsmcc_numerics.py",
        "scripts/diagnostics/models/gsmcc/diagnose_gsmcc_numerics.py",
        ("--config", "--output-root", "--max-batches", "--contrastive-mode"),
    ),
    (
        "scripts.diagnose.diagnose_m3ed_label_alignment",
        "scripts.diagnostics.data.diagnose_m3ed_label_alignment",
        "scripts/diagnose/diagnose_m3ed_label_alignment.py",
        "scripts/diagnostics/data/diagnose_m3ed_label_alignment.py",
        (),
    ),
    (
        "scripts.features.build_iemocap_clean_text_features",
        "scripts.data.build.build_iemocap_clean_text_features",
        "scripts/features/build_iemocap_clean_text_features.py",
        "scripts/data/build/build_iemocap_clean_text_features.py",
        ("--input-pkl", "--model-dir", "--output-pkl", "--device"),
    ),
    (
        "scripts.inspect.extract_mmgcn_core_blocks",
        "scripts.diagnostics.models.mmgcn.extract_mmgcn_core_blocks",
        "scripts/inspect/extract_mmgcn_core_blocks.py",
        "scripts/diagnostics/models/mmgcn/extract_mmgcn_core_blocks.py",
        ("--output-dir", "--experiment-date"),
    ),
    (
        "scripts.inspect.inspect_m3ed_feature_files",
        "scripts.data.inspect.inspect_m3ed_feature_files",
        "scripts/inspect/inspect_m3ed_feature_files.py",
        "scripts/data/inspect/inspect_m3ed_feature_files.py",
        (),
    ),
    (
        "scripts.inspect.inspect_m3ed_official",
        "scripts.data.inspect.inspect_m3ed_official",
        "scripts/inspect/inspect_m3ed_official.py",
        "scripts/data/inspect/inspect_m3ed_official.py",
        (),
    ),
    (
        "scripts.inspect.inspect_mmgcn_feature_pkl",
        "scripts.data.inspect.inspect_mmgcn_feature_pkl",
        "scripts/inspect/inspect_mmgcn_feature_pkl.py",
        "scripts/data/inspect/inspect_mmgcn_feature_pkl.py",
        (),
    ),
    (
        "scripts.inspect.inspect_mmgcn_official_model",
        "scripts.diagnostics.models.mmgcn.inspect_mmgcn_official_model",
        "scripts/inspect/inspect_mmgcn_official_model.py",
        "scripts/diagnostics/models/mmgcn/inspect_mmgcn_official_model.py",
        (),
    ),
    (
        "scripts.maintenance.check_env",
        "scripts.dev.check_env",
        "scripts/maintenance/check_env.py",
        "scripts/dev/check_env.py",
        (),
    ),
    (
        "scripts.prepare.prepare_m3ed_metadata",
        "scripts.data.prepare.prepare_m3ed_metadata",
        "scripts/prepare/prepare_m3ed_metadata.py",
        "scripts/data/prepare/prepare_m3ed_metadata.py",
        (),
    ),
)


def _help(script: str, work_dir: Path) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["MPLCONFIGDIR"] = str(work_dir / "mplconfig")
    env["PYTHONPYCACHEPREFIX"] = str(work_dir / "pycache")
    return subprocess.run(
        [sys.executable, str(PROJECT_ROOT / script), "--help"],
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=90,
        check=False,
    )


@pytest.mark.parametrize(
    ("legacy_module", "canonical_module", "legacy_path", "canonical_path", "_flags"),
    MIGRATED_SUPPORT,
)
def test_migrated_support_imports_alias_canonical_module(
    legacy_module: str,
    canonical_module: str,
    legacy_path: str,
    canonical_path: str,
    _flags: tuple[str, ...],
) -> None:
    canonical = importlib.import_module(canonical_module)
    legacy = importlib.import_module(legacy_module)

    assert legacy is canonical
    assert legacy.main is canonical.main
    assert (PROJECT_ROOT / legacy_path).is_file()
    assert (PROJECT_ROOT / canonical_path).is_file()


@pytest.mark.parametrize(
    ("_legacy_module", "canonical_module", "legacy_path", "canonical_path", "_flags"),
    MIGRATED_SUPPORT,
)
def test_support_wrappers_are_one_hop_and_contain_no_core_implementation(
    _legacy_module: str,
    canonical_module: str,
    legacy_path: str,
    canonical_path: str,
    _flags: tuple[str, ...],
) -> None:
    wrapper_path = PROJECT_ROOT / legacy_path
    wrapper_source = wrapper_path.read_text(encoding="utf-8")
    wrapper_tree = ast.parse(wrapper_source, filename=str(wrapper_path))

    assert canonical_module in wrapper_source
    assert "models.baselines" not in wrapper_source
    assert not any(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        for node in ast.walk(wrapper_tree)
    )

    canonical_source = (PROJECT_ROOT / canonical_path).read_text(encoding="utf-8")
    legacy_module_text = legacy_path[:-3].replace("/", ".")
    assert legacy_module_text not in canonical_source


@pytest.mark.parametrize(
    ("_legacy_module", "_canonical_module", "legacy_path", "canonical_path", "flags"),
    tuple(item for item in MIGRATED_SUPPORT if item[4]),
)
def test_legacy_and_canonical_cli_help_keep_key_options(
    _legacy_module: str,
    _canonical_module: str,
    legacy_path: str,
    canonical_path: str,
    flags: tuple[str, ...],
    tmp_path: Path,
) -> None:
    legacy = _help(legacy_path, tmp_path)
    canonical = _help(canonical_path, tmp_path)

    assert legacy.returncode == canonical.returncode == 0
    legacy_help = legacy.stdout + legacy.stderr
    canonical_help = canonical.stdout + canonical.stderr
    for flag in flags:
        assert flag in legacy_help
        assert flag in canonical_help


def test_canonical_support_has_no_reverse_or_legacy_model_imports() -> None:
    forbidden_prefixes = (
        "models.baselines",
        "scripts.analyze",
        "scripts.baselines",
        "scripts.debug",
        "scripts.diagnose",
        "scripts.features",
        "scripts.inspect",
        "scripts.prepare",
    )
    canonical_paths = sorted({PROJECT_ROOT / item[3] for item in MIGRATED_SUPPORT})

    for path in canonical_paths:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        imported_modules: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_modules.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported_modules.append(node.module)
        assert not any(
            module == prefix or module.startswith(prefix + ".")
            for module in imported_modules
            for prefix in forbidden_prefixes
        ), path


def test_canonical_support_repo_roots_resolve_from_new_layout() -> None:
    for canonical_module in sorted({item[1] for item in MIGRATED_SUPPORT}):
        module = importlib.import_module(canonical_module)
        if hasattr(module, "PROJECT_ROOT"):
            assert Path(module.PROJECT_ROOT).resolve() == PROJECT_ROOT.resolve()


def _fake_nine_item_pkl() -> list[object]:
    dialogue_id = "Ses01F_impro01"
    return [
        {dialogue_id: ["Ses01F_impro01_F000", "Ses01F_impro01_F001"]},
        {dialogue_id: ["F", "F"]},
        {dialogue_id: [0, 1]},
        {dialogue_id: np.arange(8, dtype=np.float32).reshape(2, 4)},
        {dialogue_id: np.arange(6, dtype=np.float64).reshape(2, 3)},
        {dialogue_id: np.arange(4, dtype=np.float32).reshape(2, 2)},
        {dialogue_id: ["hello there", "small fixture"]},
        [dialogue_id],
        [],
    ]


def _dump_pickle(path: Path, value: object) -> str:
    with path.open("wb") as file:
        pickle.dump(value, file, protocol=pickle.HIGHEST_PROTOCOL)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _directory_bytes(path: Path) -> dict[str, bytes]:
    return {
        item.relative_to(path).as_posix(): item.read_bytes()
        for item in sorted(path.rglob("*"))
        if item.is_file()
    }


def test_feature_audit_pass_outputs_match_old_and_canonical(
    tmp_path: Path,
) -> None:
    legacy_module = importlib.import_module(
        "scripts.analyze.audit_iemocap_feature_pkl"
    )
    canonical_module = importlib.import_module(
        "scripts.diagnostics.data.audit_iemocap_feature_pkl"
    )
    legacy_pkl = tmp_path / "legacy.pkl"
    candidate_pkl = tmp_path / "candidate.pkl"
    expected_sha = _dump_pickle(legacy_pkl, _fake_nine_item_pkl())
    _dump_pickle(candidate_pkl, _fake_nine_item_pkl())

    legacy_result = legacy_module.audit_feature_pkls(
        legacy_pkl,
        candidate_pkl,
        expected_legacy_sha256=expected_sha,
        expected_text_dim=4,
    )
    canonical_result = canonical_module.audit_feature_pkls(
        legacy_pkl,
        candidate_pkl,
        expected_legacy_sha256=expected_sha,
        expected_text_dim=4,
    )
    assert legacy_result == canonical_result
    assert canonical_result[0]["passed"] is True

    legacy_output = tmp_path / "legacy-output"
    canonical_output = tmp_path / "canonical-output"
    legacy_module.write_audit_outputs(legacy_output, *legacy_result)
    canonical_module.write_audit_outputs(canonical_output, *canonical_result)
    assert _directory_bytes(legacy_output) == _directory_bytes(canonical_output)


def test_feature_audit_strict_failure_matches_old_and_canonical(
    tmp_path: Path,
) -> None:
    legacy_pkl = tmp_path / "legacy.pkl"
    candidate_pkl = tmp_path / "candidate.pkl"
    expected_sha = _dump_pickle(legacy_pkl, _fake_nine_item_pkl())
    candidate = copy.deepcopy(_fake_nine_item_pkl())
    candidate[3]["Ses01F_impro01"][0, 0] = np.nan
    _dump_pickle(candidate_pkl, candidate)

    results = []
    for script, name in (
        ("scripts/analyze/audit_iemocap_feature_pkl.py", "legacy"),
        (
            "scripts/diagnostics/data/audit_iemocap_feature_pkl.py",
            "canonical",
        ),
    ):
        output_dir = tmp_path / name
        result = subprocess.run(
            [
                sys.executable,
                str(PROJECT_ROOT / script),
                "--legacy-pkl",
                str(legacy_pkl),
                "--candidate-pkl",
                str(candidate_pkl),
                "--expected-legacy-sha256",
                expected_sha,
                "--expected-text-dim",
                "4",
                "--output-dir",
                str(output_dir),
                "--strict",
            ],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        results.append((result, output_dir))

    assert results[0][0].returncode == results[1][0].returncode == 1
    summaries = [
        json.loads(
            (output_dir / "feature_audit_summary.json").read_text(encoding="utf-8")
        )
        for _, output_dir in results
    ]
    assert summaries[0] == summaries[1]
    assert summaries[0]["passed"] is False
    assert "candidate_videoText_all_finite" in summaries[0]["failed_checks"]
    assert _directory_bytes(results[0][1]) == _directory_bytes(results[1][1])


def test_read_only_feature_inspection_matches_and_preserves_input(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    legacy_module = importlib.import_module(
        "scripts.inspect.inspect_mmgcn_feature_pkl"
    )
    canonical_module = importlib.import_module(
        "scripts.data.inspect.inspect_mmgcn_feature_pkl"
    )
    input_path = tmp_path / "feature.pkl"
    before = _dump_pickle(input_path, _fake_nine_item_pkl())

    legacy_module.inspect_one_pkl("fixture.pkl", input_path)
    legacy_output = capsys.readouterr().out
    canonical_module.inspect_one_pkl("fixture.pkl", input_path)
    canonical_output = capsys.readouterr().out

    assert legacy_output == canonical_output
    assert hashlib.sha256(input_path.read_bytes()).hexdigest() == before


class _FakeTokenizer:
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
        result = {
            "input_ids": torch.tensor(
                [row + [0] * (width - len(row)) for row in rows],
                dtype=torch.long,
            ),
            "attention_mask": torch.tensor(
                [[1] * len(row) + [0] * (width - len(row)) for row in rows],
                dtype=torch.long,
            ),
        }
        if return_special_tokens_mask:
            result["special_tokens_mask"] = torch.tensor(
                [
                    special + [1] * (width - len(special))
                    for special in masks
                ],
                dtype=torch.long,
            )
        return result


class _FakeModel(torch.nn.Module):
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        offsets = torch.arange(768, device=input_ids.device).float() / 1000.0
        hidden = input_ids.float().unsqueeze(-1) + offsets
        return SimpleNamespace(last_hidden_state=hidden)


def test_synthetic_feature_build_matches_old_and_canonical(
    tmp_path: Path,
) -> None:
    legacy_module = importlib.import_module(
        "scripts.features.build_iemocap_clean_text_features"
    )
    canonical_module = importlib.import_module(
        "scripts.data.build.build_iemocap_clean_text_features"
    )
    input_path = tmp_path / "legacy.pkl"
    input_sha = _dump_pickle(input_path, _fake_nine_item_pkl())
    outputs = []

    for module, directory_name in (
        (legacy_module, "legacy-build"),
        (canonical_module, "canonical-build"),
    ):
        output_dir = tmp_path / directory_name
        output_dir.mkdir()
        output_path = output_dir / "clean.pkl"
        metadata = module.build_clean_feature_pkl(
            input_pkl=input_path,
            expected_input_sha256=input_sha,
            output_pkl=output_path,
            tokenizer=_FakeTokenizer(),
            model=_FakeModel(),
            model_local_path=tmp_path / "offline-model",
            batch_size=2,
            max_length=32,
            device="cpu",
            transformers_version="synthetic",
        )
        outputs.append((output_path, metadata))

    assert outputs[0][0].name == outputs[1][0].name == "clean.pkl"
    assert outputs[0][0].read_bytes() == outputs[1][0].read_bytes()
    comparable_metadata = []
    for _, metadata in outputs:
        filtered = dict(metadata)
        filtered.pop("created_at")
        comparable_metadata.append(filtered)
    assert comparable_metadata[0] == comparable_metadata[1]
    assert hashlib.sha256(input_path.read_bytes()).hexdigest() == input_sha


def test_unsafe_support_tools_are_not_executed_by_compatibility_tests() -> None:
    risky_modules = (
        "scripts.data.prepare.prepare_m3ed_metadata",
        "scripts.diagnostics.experiments.debug_io",
        "scripts.diagnostics.models.multidag_cl.debug_multidag_cl_real_batch",
        "scripts.diagnostics.models.debug_new_causal_graph_real_batch",
        "scripts.diagnostics.models.gsmcc.diagnose_gsmcc_numerics",
        "scripts.diagnostics.models.mmgcn.debug_mmgcn_forward",
        "scripts.diagnostics.models.simple_mlp.debug_simple_mlp_step",
        "scripts.maintenance.collect_evaluation_summary",
        "scripts.maintenance.rebuild_experiment_summary",
    )
    for module_name in risky_modules:
        module = importlib.import_module(module_name)
        assert callable(module.main)

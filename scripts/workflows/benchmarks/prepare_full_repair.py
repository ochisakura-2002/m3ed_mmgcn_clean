"""Check or materialize the DialogueGCN/GS-MCC full repair matrix.

This command only resolves configurations and writes preparation manifests. It
never imports a training entrypoint or executes a training epoch.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import shlex
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

import yaml


PROJECT_ROOT = next(
    candidate
    for candidate in Path(__file__).resolve().parents
    if (candidate / "AGENTS.md").is_file() and (candidate / "scripts").is_dir()
)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.runtime.paper_aligned import (  # noqa: E402
    normalized_training_config,
    validate_runtime_config,
)
from utils.output_paths import (  # noqa: E402
    resolve_output_paths,
    validate_batch_id,
    validate_experiment_date,
    validate_experiment_group,
)


EXPERIMENT_GROUP = "dialoguegcn_gsmcc_full_repair"
ENTRYPOINT = "scripts/workflows/paper_aligned/train.py"
ALLOWED_MODEL_FAMILIES = {"dialoguegcn", "gsmcc_project_variant"}
ALLOWED_CANDIDATE_TYPES = {
    "original_diagnostic",
    "delayed_early_stop",
    "lr_candidate",
    "best_candidate_control",
}
REQUIRED_RECORD_FIELDS = {
    "run_id",
    "model_family",
    "implementation_identity",
    "context_mode",
    "validation_session",
    "test_session",
    "seed",
    "candidate_type",
    "changed_variables",
    "unchanged_variables",
    "control_group",
    "diagnostic_question",
    "base_config",
    "entrypoint",
    "overrides",
}


def load_yaml(path: Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as file:
        value = yaml.safe_load(file)
    if not isinstance(value, dict):
        raise TypeError(f"YAML root must be a mapping: {path}")
    return value


def _resolve(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def set_dotted(config: dict[str, Any], dotted: str, value: Any) -> None:
    keys = str(dotted).split(".")
    current = config
    for key in keys[:-1]:
        child = current.setdefault(key, {})
        if not isinstance(child, dict):
            raise TypeError(f"Cannot set {dotted!r}; {key!r} is not a mapping")
        current = child
    current[keys[-1]] = copy.deepcopy(value)


def _flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            child = f"{prefix}.{key}" if prefix else str(key)
            result.update(_flatten(item, child))
        return result
    return {prefix: value}


def _core_config(config: Mapping[str, Any]) -> dict[str, Any]:
    comparable = copy.deepcopy(dict(config))
    for key in ("run_name", "diagnostics", "output", "repair_experiment"):
        comparable.pop(key, None)
    return comparable


def _changed_core_paths(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
) -> set[str]:
    left = _flatten(_core_config(before))
    right = _flatten(_core_config(after))
    return {
        key
        for key in set(left) | set(right)
        if left.get(key, object()) != right.get(key, object())
    }


def _expected_core_changes(candidate_type: str) -> set[str]:
    if candidate_type == "original_diagnostic":
        return set()
    if candidate_type in {"delayed_early_stop", "best_candidate_control"}:
        return {
            "training.early_stopping_min_epochs",
            "training.early_stopping_patience",
        }
    if candidate_type == "lr_candidate":
        return {"optimizer.learning_rate"}
    raise ValueError(f"Unsupported candidate_type: {candidate_type!r}")


def _validate_human_csv(
    matrix: Mapping[str, Any],
    project_root: Path,
) -> None:
    csv_path = _resolve(project_root, Path(str(matrix["human_matrix_csv"])))
    with csv_path.open("r", encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    candidates = list(matrix["candidates"])
    if [row["run_id"] for row in rows] != [
        str(candidate["run_id"]) for candidate in candidates
    ]:
        raise ValueError("human matrix CSV run order does not match repair matrix YAML")
    for row, candidate in zip(rows, candidates):
        for key in (
            "model_family",
            "implementation_identity",
            "context_mode",
            "validation_session",
            "test_session",
            "candidate_type",
            "base_config",
            "entrypoint",
        ):
            if str(row[key]) != str(candidate[key]):
                raise ValueError(
                    f"human matrix CSV mismatch for {candidate['run_id']} field {key}"
                )


def prepare_full_repair_matrix(
    matrix_path: Path,
    mode: str,
    experiment_date: str,
    *,
    root: Path | None = None,
    resolved_root: Path | None = None,
    batch_id: str | None = None,
) -> dict[str, Any]:
    """Validate the matrix and optionally materialize configs and commands."""

    if mode not in {"check", "prepare"}:
        raise ValueError(f"Unsupported mode: {mode!r}")
    experiment_date = validate_experiment_date(experiment_date)
    project_root = Path.cwd() if root is None else Path(root)
    matrix_file = _resolve(project_root, Path(matrix_path))
    matrix = load_yaml(matrix_file)
    if str(matrix.get("experiment_group")) != EXPERIMENT_GROUP:
        raise ValueError(f"experiment_group must be {EXPERIMENT_GROUP}")
    experiment_group = validate_experiment_group(matrix["experiment_group"])
    expected_run_count = int(matrix["expected_run_count"])
    if expected_run_count <= 0 or expected_run_count > 12:
        raise ValueError("repair matrix run count must be between 1 and 12")
    candidates = list(matrix.get("candidates", []))
    if len(candidates) != expected_run_count:
        raise ValueError(
            f"repair matrix lists {len(candidates)} candidates; "
            f"expected {expected_run_count}"
        )
    protocol = matrix["protocol"]
    if protocol.get("checkpoint_selection_metric") != "val_weighted_f1":
        raise ValueError("checkpoint selection must use val_weighted_f1")
    if protocol.get("checkpoint_selection_split") != "validation":
        raise ValueError("checkpoint selection must be validation-only")
    if protocol.get("test_split_used_for_selection") is not False:
        raise ValueError("test split must not participate in selection")
    if str(protocol.get("test_session")) != "Ses05":
        raise ValueError("repair matrix must keep Ses05 as test")

    _validate_human_csv(matrix, project_root)
    output_base = Path(str(matrix.get("output_base", "outputs")))
    resolved_batch_id = validate_batch_id(
        batch_id or f"{experiment_group}_{experiment_date}_check"
    )
    batch_layout = resolve_output_paths(
        output_base=output_base,
        experiment_date=experiment_date,
        experiment_group=experiment_group,
        batch_id=resolved_batch_id,
    )
    resolved_dir = (
        batch_layout.manifest_root / "resolved_configs"
        if resolved_root is None
        else Path(resolved_root)
    )
    manifest_dir = resolved_dir.parent

    formal_source_paths = sorted(
        {
            _resolve(project_root, Path(str(candidate["base_config"])))
            for candidate in candidates
        }
    )
    formal_source_bytes = {
        path: path.read_bytes()
        for path in formal_source_paths
    }

    if mode == "prepare":
        if resolved_root is None:
            absolute_manifest = _resolve(project_root, batch_layout.manifest_root)
            absolute_manifest.mkdir(parents=True, exist_ok=False)
            _resolve(project_root, resolved_dir).mkdir(parents=False, exist_ok=False)
        else:
            _resolve(project_root, resolved_dir).mkdir(parents=True, exist_ok=False)

    run_ids: set[str] = set()
    output_roots: set[str] = set()
    records: list[dict[str, Any]] = []
    commands: list[str] = []
    model_counts: Counter[str] = Counter()
    candidate_counts: Counter[str] = Counter()
    runtime_validation_count = 0

    for order, candidate in enumerate(candidates, start=1):
        missing = REQUIRED_RECORD_FIELDS - set(candidate)
        if missing:
            raise ValueError(
                f"repair candidate {order} missing fields: {sorted(missing)}"
            )
        run_id = str(candidate["run_id"])
        if run_id in run_ids:
            raise ValueError(f"duplicate run_id: {run_id}")
        run_ids.add(run_id)
        model_family = str(candidate["model_family"])
        if model_family not in ALLOWED_MODEL_FAMILIES:
            raise ValueError(f"unsupported model_family: {model_family}")
        candidate_type = str(candidate["candidate_type"])
        if candidate_type not in ALLOWED_CANDIDATE_TYPES:
            raise ValueError(f"unsupported candidate_type: {candidate_type}")
        if str(candidate["context_mode"]) != "full_context":
            raise ValueError(f"{run_id} must remain full_context")
        validation_session = str(candidate["validation_session"])
        if validation_session not in {"Ses01", "Ses02", "Ses03", "Ses04"}:
            raise ValueError(f"{run_id} has invalid validation session")
        if str(candidate["test_session"]) != "Ses05":
            raise ValueError(f"{run_id} must keep Ses05 as test")
        if int(candidate["seed"]) != 42:
            raise ValueError(f"{run_id} must keep Seed 42")
        if str(candidate["entrypoint"]) != ENTRYPOINT:
            raise ValueError(f"{run_id} must use the canonical full-context entrypoint")

        base_path = _resolve(project_root, Path(str(candidate["base_config"])))
        entrypoint_path = _resolve(project_root, Path(str(candidate["entrypoint"])))
        if not base_path.is_file() or not entrypoint_path.is_file():
            raise FileNotFoundError(base_path if not base_path.is_file() else entrypoint_path)
        config = load_yaml(base_path)
        set_dotted(config, "dataset.val_session_id", validation_session)
        set_dotted(config, "dataset.outer_test_session", "Ses05")
        session_baseline = copy.deepcopy(config)
        for dotted, value in candidate["overrides"].items():
            set_dotted(config, str(dotted), value)
        diagnostics = copy.deepcopy(dict(matrix["diagnostics"]))
        diagnostics["enabled"] = True
        config["diagnostics"] = diagnostics

        changed_core = _changed_core_paths(session_baseline, config)
        expected_core = _expected_core_changes(candidate_type)
        if changed_core != expected_core:
            raise ValueError(
                f"{run_id} changes {sorted(changed_core)}; "
                f"expected exactly {sorted(expected_core)}"
            )
        declared_changed = set(str(value) for value in candidate["changed_variables"])
        if declared_changed != expected_core | {"diagnostics.enabled"}:
            raise ValueError(f"{run_id} changed_variables does not match its overrides")

        run_layout = resolve_output_paths(
            output_base=output_base,
            experiment_date=experiment_date,
            experiment_group=experiment_group,
            run_id=run_id,
        )
        assert run_layout.run_root is not None
        output_root = run_layout.run_root.as_posix()
        if output_root in output_roots:
            raise ValueError(f"duplicate output root: {output_root}")
        output_roots.add(output_root)
        canonical_resolved_path = (
            batch_layout.manifest_root / "resolved_configs" / f"{run_id}.yaml"
        )
        resolved_path = (
            canonical_resolved_path
            if resolved_root is None
            else resolved_dir / f"{run_id}.yaml"
        )

        config["run_name"] = run_id
        config.setdefault("output", {})
        config["output"].update(
            {
                "root": output_root,
                "output_base": output_base.as_posix(),
                "experiment_date": experiment_date,
                "experiment_group": experiment_group,
                "run_id": run_id,
                "run_root": output_root,
                "manifest_root": batch_layout.manifest_root.as_posix(),
                "review_root": batch_layout.review_root.as_posix(),
                "report_root": batch_layout.report_root.as_posix(),
                "analysis_root": batch_layout.analysis_root.as_posix(),
            }
        )
        config["repair_experiment"] = {
            "run_id": run_id,
            "model_family": model_family,
            "implementation_identity": candidate["implementation_identity"],
            "context_mode": candidate["context_mode"],
            "validation_session": validation_session,
            "test_session": "Ses05",
            "seed": 42,
            "candidate_type": candidate_type,
            "changed_variables": list(candidate["changed_variables"]),
            "unchanged_variables": list(candidate["unchanged_variables"]),
            "control_group": candidate["control_group"],
            "diagnostic_question": candidate["diagnostic_question"],
            "base_config": Path(str(candidate["base_config"])).as_posix(),
            "resolved_config": canonical_resolved_path.as_posix(),
            "entrypoint": ENTRYPOINT,
            "max_epochs": int(config["training"]["epochs"]),
            "min_epochs": int(
                config["training"].get("early_stopping_min_epochs", 0)
            ),
            "patience": int(config["training"].get("early_stopping_patience", 0)),
            "learning_rate": float(config["optimizer"]["learning_rate"]),
            "selection_metric": "val_weighted_f1",
            "test_selection_leakage": False,
            "experiment_group": experiment_group,
            "output_root": output_root,
        }

        normalized = normalized_training_config(config)
        validate_runtime_config(normalized)
        if normalized["training"]["select_best_by"] != "val_weighted_f1":
            raise ValueError(f"{run_id} changes the selection metric")
        if normalized.get("protocol", {}).get("test_split_used_for_selection") is not False:
            raise ValueError(f"{run_id} permits test selection leakage")
        runtime_validation_count += 1

        if mode == "prepare":
            resolved_file = _resolve(project_root, resolved_path)
            with resolved_file.open("x", encoding="utf-8", newline="\n") as file:
                yaml.safe_dump(config, file, sort_keys=False, allow_unicode=True)

        command_parts = [
            "python",
            "-u",
            ENTRYPOINT,
            "--config",
            resolved_path.as_posix(),
            "--experiment-date",
            experiment_date,
            "--experiment-group",
            experiment_group,
        ]
        command = " ".join(shlex.quote(part) for part in command_parts)
        commands.append(command)
        records.append(
            {
                **config["repair_experiment"],
                "order": order,
                "resolved_path": resolved_path,
                "canonical_resolved_path": canonical_resolved_path,
                "config": config,
                "command": command,
                "runtime_validation_status": "PASS",
            }
        )
        model_counts[model_family] += 1
        candidate_counts[candidate_type] += 1

    if model_counts != Counter({"dialoguegcn": 4, "gsmcc_project_variant": 8}):
        raise ValueError(f"unexpected model run counts: {dict(model_counts)}")
    if len(output_roots) != expected_run_count:
        raise ValueError("repair output roots must be unique")

    formal_sources_unchanged = all(
        path.read_bytes() == original
        for path, original in formal_source_bytes.items()
    )
    if not formal_sources_unchanged:
        raise RuntimeError("formal long32 base config changed during preparation")

    if mode == "prepare":
        absolute_manifest = _resolve(project_root, manifest_dir)
        with (absolute_manifest / "commands.txt").open(
            "x", encoding="utf-8", newline="\n"
        ) as file:
            file.write("\n".join(commands) + "\n")
        with (absolute_manifest / "matrix.yaml").open(
            "x", encoding="utf-8", newline="\n"
        ) as file:
            yaml.safe_dump(matrix, file, sort_keys=False, allow_unicode=True)
        with (absolute_manifest / "matrix.csv").open(
            "x", encoding="utf-8", newline=""
        ) as file:
            fieldnames = [
                "order",
                "run_id",
                "model_family",
                "implementation_identity",
                "context_mode",
                "validation_session",
                "test_session",
                "seed",
                "candidate_type",
                "changed_variables",
                "unchanged_variables",
                "control_group",
                "diagnostic_question",
                "base_config",
                "resolved_config",
                "entrypoint",
                "max_epochs",
                "min_epochs",
                "patience",
                "learning_rate",
                "selection_metric",
                "test_selection_leakage",
                "experiment_group",
                "output_root",
            ]
            writer = csv.DictWriter(file, fieldnames=fieldnames, lineterminator="\n")
            writer.writeheader()
            for record in records:
                row = {key: record[key] for key in fieldnames}
                row["changed_variables"] = json.dumps(
                    row["changed_variables"], ensure_ascii=False
                )
                row["unchanged_variables"] = json.dumps(
                    row["unchanged_variables"], ensure_ascii=False
                )
                writer.writerow(row)
        try:
            git_commit = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=project_root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            git_commit = "unavailable"
        (absolute_manifest / "git_commit.txt").write_text(
            git_commit + "\n", encoding="utf-8"
        )
        (absolute_manifest / "preparation_metadata.json").write_text(
            json.dumps(
                {
                    "matrix_name": matrix["matrix_name"],
                    "experiment_date": experiment_date,
                    "experiment_group": experiment_group,
                    "batch_id": resolved_batch_id,
                    "expanded_run_count": len(records),
                    "generated_command_count": len(commands),
                    "runtime_validation_pass_count": runtime_validation_count,
                    "test_selection_leakage_found": 0,
                    "formal_sources_unchanged": formal_sources_unchanged,
                    "formal_training_started": 0,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    return {
        "matrix": matrix,
        "experiment_date": experiment_date,
        "experiment_group": experiment_group,
        "batch_id": resolved_batch_id,
        "experiment_root": batch_layout.experiment_root,
        "launcher_log_root": batch_layout.launcher_log_root,
        "manifest_root": batch_layout.manifest_root,
        "review_root": batch_layout.review_root,
        "report_root": batch_layout.report_root,
        "analysis_root": batch_layout.analysis_root,
        "records": records,
        "commands": commands,
        "expanded_run_count": len(records),
        "model_counts": model_counts,
        "candidate_counts": candidate_counts,
        "output_collision_count": len(records) - len(output_roots),
        "runtime_validation_count": runtime_validation_count,
        "formal_sources_unchanged": formal_sources_unchanged,
        "formal_training_started": 0,
    }


def print_result(result: Mapping[str, Any], mode: str) -> None:
    print(f"MODE={mode}")
    print(f"MATRIX={result['matrix']['matrix_name']}")
    print(f"EXPERIMENT_GROUP={result['experiment_group']}")
    print(f"EXPANDED_RUN_COUNT={result['expanded_run_count']}")
    print(f"GENERATED_COMMAND_COUNT={len(result['commands'])}")
    print(f"RUNTIME_VALIDATION_COUNT={result['runtime_validation_count']}")
    print(f"OUTPUT_COLLISION_COUNT={result['output_collision_count']}")
    print(
        "FORMAL_SOURCES_UNCHANGED="
        + ("YES" if result["formal_sources_unchanged"] else "NO")
    )
    print("FORMAL_TRAINING_STARTED=0")
    for command in result["commands"]:
        print(f"COMMAND={command}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("matrix_path", type=Path)
    parser.add_argument("mode", choices=("check", "prepare"))
    parser.add_argument("--experiment-date", required=True)
    parser.add_argument("--batch-id", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = prepare_full_repair_matrix(
        args.matrix_path,
        args.mode,
        args.experiment_date,
        root=PROJECT_ROOT,
        batch_id=args.batch_id,
    )
    print_result(result, args.mode)


if __name__ == "__main__":
    main()

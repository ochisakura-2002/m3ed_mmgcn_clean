"""Validate or materialize the formal repaired full-context 16-run matrix.

This module only resolves configuration files and renders training commands.
It never imports a training entrypoint or starts training.
"""

from __future__ import annotations

import argparse
import copy
import shlex
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
    resolve_experiment_date,
    resolve_output_paths,
    validate_batch_id,
    validate_experiment_date,
    validate_experiment_group,
)


MATRIX_NAME = "formal_full16_repaired_seed42"
EXPERIMENT_GROUP = "formal_full16_repaired_seed42"
ENTRYPOINT = "scripts/workflows/paper_aligned/train.py"
VALIDATION_SESSIONS = ("Ses01", "Ses02", "Ses03", "Ses04")
TEST_SESSION = "Ses05"
SEED = 42
EXPECTED_MODELS = (
    "mmgcn",
    "multidag_cl",
    "dialoguegcn",
    "gsmcc_project_variant",
)
EXPECTED_IDENTITIES = {
    "mmgcn": "unified_project_implementation",
    "multidag_cl": "project_variant_not_author_official",
    "dialoguegcn": "paper_aligned_repaired_not_author_official",
    "gsmcc_project_variant": "project_variant_not_author_official",
}
REPAIR_OVERRIDE_PATHS = {
    "optimizer.learning_rate",
    "training.epochs",
    "training.early_stopping_min_epochs",
    "training.early_stopping_patience",
}
REPAIR_SIGNATURES = {
    "dialoguegcn": {
        "learning_rate": 3e-4,
        "max_epochs": 60,
        "early_stopping_min_epochs": 30,
        "early_stopping_patience": 20,
    },
    "gsmcc_project_variant": {
        "learning_rate": 1e-5,
        "max_epochs": 250,
        "early_stopping_min_epochs": 90,
        "early_stopping_patience": 40,
    },
}


def load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML mapping from *path*."""

    with Path(path).open("r", encoding="utf-8") as file:
        value = yaml.safe_load(file)
    if not isinstance(value, dict):
        raise TypeError(f"YAML root must be a mapping: {path}")
    return value


def _resolve(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def set_dotted(config: dict[str, Any], dotted: str, value: Any) -> None:
    """Set one dotted mapping path without guessing schema aliases."""

    keys = str(dotted).split(".")
    current = config
    for key in keys[:-1]:
        child = current.setdefault(key, {})
        if not isinstance(child, dict):
            raise TypeError(f"Cannot set {dotted!r}; {key!r} is not a mapping")
        current = child
    current[keys[-1]] = copy.deepcopy(value)


def _effective_signature(config: Mapping[str, Any]) -> dict[str, Any]:
    training = config["training"]
    optimizer = config["optimizer"]
    return {
        "learning_rate": float(optimizer["learning_rate"]),
        "max_epochs": int(training["epochs"]),
        "early_stopping_min_epochs": int(
            training.get("early_stopping_min_epochs", 0)
        ),
        "early_stopping_patience": int(
            training.get("early_stopping_patience", 0)
        ),
    }


def _validate_matrix_header(matrix: Mapping[str, Any]) -> None:
    if matrix.get("matrix_name") != MATRIX_NAME:
        raise ValueError(f"matrix_name must be {MATRIX_NAME}")
    if matrix.get("enabled") is not True:
        raise ValueError("formal full16 matrix must be enabled")
    if int(matrix.get("expected_run_count", -1)) != 16:
        raise ValueError("expected_run_count must be 16")
    if validate_experiment_group(matrix.get("experiment_group")) != EXPERIMENT_GROUP:
        raise ValueError(f"experiment_group must be {EXPERIMENT_GROUP}")

    protocol = matrix.get("protocol", {})
    if protocol.get("context_mode") != "full_context":
        raise ValueError("formal full16 matrix must be full_context only")
    if tuple(protocol.get("validation_sessions", ())) != VALIDATION_SESSIONS:
        raise ValueError("validation sessions must be Ses01-Ses04 in order")
    if protocol.get("test_session") != TEST_SESSION:
        raise ValueError("test session must be Ses05")
    if list(protocol.get("seed_values", ())) != [SEED]:
        raise ValueError("seed_values must be [42]")
    if protocol.get("checkpoint_selection_metric") != "val_weighted_f1":
        raise ValueError("checkpoint selection must use validation weighted F1")
    if protocol.get("checkpoint_selection_split") != "validation":
        raise ValueError("checkpoint selection must be validation-only")
    if protocol.get("test_split_used_for_selection") is not False:
        raise ValueError("test must not participate in selection")
    if int(protocol.get("test_selection_leakage_found", -1)) != 0:
        raise ValueError("test selection leakage must remain zero")
    if protocol.get("val_split_strategy") != "session_holdout":
        raise ValueError("validation must use session_holdout")


def _validate_feature_contract(
    matrix: Mapping[str, Any], project_root: Path
) -> None:
    feature = matrix["feature"]
    registry_path = _resolve(project_root, Path(str(feature["registry_path"])))
    registry = load_yaml(registry_path)
    registry_key = str(feature["registry_key"])
    if registry_key not in registry:
        raise KeyError(f"feature registry entry not found: {registry_key}")
    registry_entry = registry[registry_key]
    for key in ("feature_set_name", "path", "sha256", "text_dim"):
        if registry_entry.get(key) != feature.get(key):
            raise ValueError(f"feature registry mismatch for {key}")


def _validate_base_record(
    record: Mapping[str, Any],
    matrix: Mapping[str, Any],
    project_root: Path,
) -> tuple[Path, dict[str, Any]]:
    model_family = str(record.get("model_family"))
    if model_family not in EXPECTED_MODELS:
        raise ValueError(f"unsupported model_family: {model_family}")
    if record.get("implementation_identity") != EXPECTED_IDENTITIES[model_family]:
        raise ValueError(f"implementation identity mismatch for {model_family}")
    if record.get("context_mode") != "full_context":
        raise ValueError(f"{model_family} must remain full_context")
    if record.get("entrypoint") != ENTRYPOINT:
        raise ValueError(f"{model_family} must use {ENTRYPOINT}")

    base_path = _resolve(project_root, Path(str(record["base_config"])))
    parameter_source = _resolve(project_root, Path(str(record["parameter_source"])))
    entrypoint = _resolve(project_root, Path(str(record["entrypoint"])))
    for required in (base_path, parameter_source, entrypoint):
        if not required.is_file():
            raise FileNotFoundError(required)

    base = load_yaml(base_path)
    long_training = base.get("long_training", {})
    if long_training.get("model_family") != model_family:
        raise ValueError(f"model family mismatch in {base_path}")
    if long_training.get("context_mode") != "full_context":
        raise ValueError(f"context mode mismatch in {base_path}")
    if long_training.get("entrypoint") != record["entrypoint"]:
        raise ValueError(f"entrypoint mismatch in {base_path}")
    if long_training.get("parameter_source") != record["parameter_source"]:
        raise ValueError(f"parameter source mismatch in {base_path}")

    protocol = matrix["protocol"]
    dataset = base.get("dataset", {})
    if base.get("protocol_version") != protocol["protocol_version"]:
        raise ValueError(f"protocol version mismatch in {base_path}")
    for field in (
        "experiment_track",
        "val_split_strategy",
        "protocol_comparability",
    ):
        if dataset.get(field) != protocol[field]:
            raise ValueError(f"{field} mismatch in {base_path}")
    feature = matrix["feature"]
    expected_feature_fields = {
        "feature_set_name": feature["feature_set_name"],
        "feature_pkl_path": feature["path"],
        "feature_sha256": feature["sha256"],
    }
    for field, expected in expected_feature_fields.items():
        if dataset.get(field) != expected:
            raise ValueError(f"shared feature mismatch for {field} in {base_path}")

    overrides = record.get("overrides", {})
    if not isinstance(overrides, Mapping):
        raise TypeError(f"overrides must be a mapping for {model_family}")
    override_paths = set(str(path) for path in overrides)
    if model_family in {"mmgcn", "multidag_cl"}:
        if override_paths:
            raise ValueError(f"{model_family} must reuse the old full base unchanged")
    else:
        if override_paths != REPAIR_OVERRIDE_PATHS:
            raise ValueError(
                f"{model_family} repair overrides must be exactly "
                f"{sorted(REPAIR_OVERRIDE_PATHS)}"
            )
        repaired = copy.deepcopy(base)
        for dotted, value in overrides.items():
            set_dotted(repaired, str(dotted), value)
        if _effective_signature(repaired) != REPAIR_SIGNATURES[model_family]:
            raise ValueError(f"repair signature mismatch for {model_family}")
    return base_path, base


def prepare_formal_full16_repaired(
    matrix_path: Path,
    mode: str,
    experiment_date: str,
    *,
    root: Path | None = None,
    batch_id: str | None = None,
    output_base_override: Path | None = None,
) -> dict[str, Any]:
    """Expand, validate, and optionally materialize the formal full16 matrix."""

    if mode not in {"check", "prepare"}:
        raise ValueError(f"unsupported mode: {mode!r}")
    experiment_date = validate_experiment_date(experiment_date)
    project_root = Path.cwd() if root is None else Path(root)
    matrix_file = _resolve(project_root, Path(matrix_path))
    matrix = load_yaml(matrix_file)
    _validate_matrix_header(matrix)
    _validate_feature_contract(matrix, project_root)

    records_by_model = list(matrix.get("base_configs", ()))
    if [str(record.get("model_family")) for record in records_by_model] != list(
        EXPECTED_MODELS
    ):
        raise ValueError("base_configs must list the four formal models once, in order")

    output_base = (
        Path(str(matrix.get("output_base", "outputs")))
        if output_base_override is None
        else Path(output_base_override)
    )
    resolved_batch_id = validate_batch_id(
        batch_id or f"{EXPERIMENT_GROUP}_{experiment_date}_check"
    )
    batch_layout = resolve_output_paths(
        output_base=output_base,
        experiment_date=experiment_date,
        experiment_group=EXPERIMENT_GROUP,
        batch_id=resolved_batch_id,
    )

    source_bytes: dict[Path, bytes] = {}
    records: list[dict[str, Any]] = []
    commands: list[str] = []
    model_counts: Counter[str] = Counter()
    context_counts: Counter[str] = Counter()
    validation_counts: Counter[str] = Counter()
    test_sessions: set[str] = set()
    seed_values: set[int] = set()
    runtime_validation_count = 0

    for base_record in records_by_model:
        base_path, base = _validate_base_record(base_record, matrix, project_root)
        source_bytes[base_path] = base_path.read_bytes()
        model_family = str(base_record["model_family"])
        for validation_session in VALIDATION_SESSIONS:
            tokens = {
                "model_family": model_family,
                "session_slug": validation_session.lower(),
                "seed": SEED,
                "output_base": output_base.as_posix(),
                "experiment_date": experiment_date,
                "experiment_group": EXPERIMENT_GROUP,
                "batch_id": resolved_batch_id,
            }
            run_id = str(matrix["expansion"]["run_id_template"]).format(**tokens)
            tokens["run_id"] = run_id
            run_layout = resolve_output_paths(
                output_base=output_base,
                experiment_date=experiment_date,
                experiment_group=EXPERIMENT_GROUP,
                run_id=run_id,
            )
            assert run_layout.run_root is not None
            output_root = str(
                matrix["expansion"]["output_root_template"]
            ).format(**tokens)
            if Path(output_root) != run_layout.run_root:
                raise ValueError(
                    f"non-canonical output root for {run_id}: {output_root}"
                )
            canonical_resolved_path = Path(
                str(matrix["expansion"]["resolved_config_template"]).format(
                    **tokens
                )
            )
            expected_resolved_path = (
                batch_layout.manifest_root / "resolved_configs" / f"{run_id}.yaml"
            )
            if canonical_resolved_path != expected_resolved_path:
                raise ValueError(
                    f"non-canonical resolved config path for {run_id}: "
                    f"{canonical_resolved_path}"
                )

            config = copy.deepcopy(base)
            for dotted, value in base_record.get("overrides", {}).items():
                set_dotted(config, str(dotted), value)
            set_dotted(config, "dataset.val_session_id", validation_session)
            set_dotted(config, "dataset.outer_test_session", TEST_SESSION)
            set_dotted(config, "dataset.split_seed", SEED)
            set_dotted(config, "training.seed", SEED)
            set_dotted(config, "system.seed", SEED)
            config["run_name"] = run_id
            config.setdefault("protocol", {})["run_type"] = MATRIX_NAME
            config.setdefault("output", {}).update(
                {
                    "root": output_root,
                    "output_base": output_base.as_posix(),
                    "experiment_date": experiment_date,
                    "experiment_group": EXPERIMENT_GROUP,
                    "experiment_root": run_layout.experiment_root.as_posix(),
                    "run_id": run_id,
                    "run_root": output_root,
                    "manifest_root": batch_layout.manifest_root.as_posix(),
                    "review_root": batch_layout.review_root.as_posix(),
                    "report_root": batch_layout.report_root.as_posix(),
                    "analysis_root": batch_layout.analysis_root.as_posix(),
                }
            )
            config["formal_full16"] = {
                "matrix_name": MATRIX_NAME,
                "run_id": run_id,
                "model_family": model_family,
                "display_name": base_record["display_name"],
                "implementation_identity": base_record["implementation_identity"],
                "author_official_reproduction": False,
                "context_mode": "full_context",
                "validation_session": validation_session,
                "test_session": TEST_SESSION,
                "seed": SEED,
                "selection_metric": "val_weighted_f1",
                "selection_split": "validation",
                "test_selection_leakage": False,
                "base_config": Path(str(base_record["base_config"])).as_posix(),
                "parameter_source": Path(
                    str(base_record["parameter_source"])
                ).as_posix(),
                "resolved_config": canonical_resolved_path.as_posix(),
                "entrypoint": ENTRYPOINT,
                "training_signature": _effective_signature(config),
                "experiment_group": EXPERIMENT_GROUP,
                "output_root": output_root,
            }

            normalized = normalized_training_config(config)
            validate_runtime_config(normalized)
            if normalized["training"]["select_best_by"] != "val_weighted_f1":
                raise ValueError(f"{run_id} changes the selection metric")
            if normalized.get("protocol", {}).get(
                "test_split_used_for_selection"
            ) is not False:
                raise ValueError(f"{run_id} permits test selection leakage")
            runtime_validation_count += 1

            command_parts = [
                "python",
                "-u",
                ENTRYPOINT,
                "--config",
                canonical_resolved_path.as_posix(),
                "--experiment-date",
                experiment_date,
                "--experiment-group",
                EXPERIMENT_GROUP,
            ]
            command = " ".join(shlex.quote(part) for part in command_parts)
            records.append(
                {
                    "run_id": run_id,
                    "model_family": model_family,
                    "context_mode": "full_context",
                    "validation_session": validation_session,
                    "test_session": TEST_SESSION,
                    "seed": SEED,
                    "output_root": output_root,
                    "resolved_path": canonical_resolved_path,
                    "config": config,
                    "command": command,
                }
            )
            commands.append(command)
            model_counts[model_family] += 1
            context_counts["full_context"] += 1
            validation_counts[validation_session] += 1
            test_sessions.add(TEST_SESSION)
            seed_values.add(SEED)

    run_ids = [record["run_id"] for record in records]
    output_roots = [record["output_root"] for record in records]
    duplicate_run_id_count = len(run_ids) - len(set(run_ids))
    duplicate_output_root_count = len(output_roots) - len(set(output_roots))
    output_collision_count = sum(
        _resolve(project_root, Path(output_root)).exists()
        for output_root in output_roots
    )
    leakage_count = sum(
        record["config"]["formal_full16"]["test_selection_leakage"] is not False
        for record in records
    )

    if len(records) != 16 or len(commands) != 16:
        raise ValueError(
            f"matrix must produce exactly 16 runs and commands; got "
            f"{len(records)} runs and {len(commands)} commands"
        )
    if model_counts != Counter({model: 4 for model in EXPECTED_MODELS}):
        raise ValueError(f"unexpected model counts: {dict(model_counts)}")
    if context_counts != Counter({"full_context": 16}):
        raise ValueError(f"unexpected context counts: {dict(context_counts)}")
    if validation_counts != Counter({session: 4 for session in VALIDATION_SESSIONS}):
        raise ValueError(f"unexpected validation counts: {dict(validation_counts)}")
    if test_sessions != {TEST_SESSION} or seed_values != {SEED}:
        raise ValueError("test session or seed contract changed")
    if duplicate_run_id_count:
        raise ValueError(f"duplicate run IDs found: {duplicate_run_id_count}")
    if duplicate_output_root_count:
        raise ValueError(
            f"duplicate output roots found: {duplicate_output_root_count}"
        )
    if output_collision_count:
        raise ValueError(f"output collisions found: {output_collision_count}")
    if leakage_count:
        raise ValueError(f"test selection leakage found: {leakage_count}")
    if runtime_validation_count != 16:
        raise ValueError("all 16 configs must pass config-only runtime validation")

    source_files_unchanged = all(
        path.read_bytes() == original for path, original in source_bytes.items()
    )
    if not source_files_unchanged:
        raise RuntimeError("a source Long32 base changed during preparation")

    if mode == "prepare":
        manifest_root = _resolve(project_root, batch_layout.manifest_root)
        if manifest_root.exists():
            raise FileExistsError(f"target batch already exists: {manifest_root}")
        experiment_root = _resolve(project_root, batch_layout.experiment_root)
        for name in ("runs", "logs", "manifests", "review", "reports", "analysis"):
            (experiment_root / name).mkdir(parents=True, exist_ok=True)
        manifest_root.mkdir(parents=True, exist_ok=False)
        resolved_root = manifest_root / "resolved_configs"
        resolved_root.mkdir(parents=False, exist_ok=False)
        for record in records:
            destination = resolved_root / f"{record['run_id']}.yaml"
            with destination.open("x", encoding="utf-8", newline="\n") as file:
                yaml.safe_dump(
                    record["config"], file, sort_keys=False, allow_unicode=True
                )
        with (manifest_root / "commands.txt").open(
            "x", encoding="utf-8", newline="\n"
        ) as file:
            file.write("\n".join(commands) + "\n")

    return {
        "matrix": matrix,
        "experiment_date": experiment_date,
        "experiment_group": EXPERIMENT_GROUP,
        "batch_id": resolved_batch_id,
        "experiment_root": batch_layout.experiment_root,
        "manifest_root": batch_layout.manifest_root,
        "records": records,
        "commands": commands,
        "expanded_run_count": len(records),
        "context_counts": context_counts,
        "model_counts": model_counts,
        "validation_counts": validation_counts,
        "test_session": next(iter(test_sessions)),
        "seed_values": sorted(seed_values),
        "test_selection_leakage_found": leakage_count,
        "output_collision_count": output_collision_count,
        "duplicate_run_id_count": duplicate_run_id_count,
        "duplicate_output_root_count": duplicate_output_root_count,
        "runtime_validation_count": runtime_validation_count,
        "source_files_unchanged": source_files_unchanged,
        "training_started": 0,
    }


def print_result(result: Mapping[str, Any], mode: str) -> None:
    """Print the audit counters and one summary for every expanded run."""

    model_counts = result["model_counts"]
    validation_counts = result["validation_counts"]
    context_counts = result["context_counts"]
    print(f"MODE={mode}")
    print(f"MATRIX={result['matrix']['matrix_name']}")
    print(f"EXPERIMENT_DATE={result['experiment_date']}")
    print(f"EXPERIMENT_GROUP={result['experiment_group']}")
    print(f"BATCH_ID={result['batch_id']}")
    print(f"EXPANDED_RUN_COUNT={result['expanded_run_count']}")
    print(f"FULL_CONTEXT_RUN_COUNT={context_counts['full_context']}")
    print(f"CAUSAL_CONTEXT_RUN_COUNT={context_counts['causal_context']}")
    print(f"MMGCN_RUN_COUNT={model_counts['mmgcn']}")
    print(f"MULTIDAG_CL_RUN_COUNT={model_counts['multidag_cl']}")
    print(f"DIALOGUEGCN_RUN_COUNT={model_counts['dialoguegcn']}")
    print(
        "GSMCC_PROJECT_VARIANT_RUN_COUNT="
        f"{model_counts['gsmcc_project_variant']}"
    )
    for session in VALIDATION_SESSIONS:
        print(f"VALIDATION_{session.upper()}_COUNT={validation_counts[session]}")
    print(f"TEST_SESSION={result['test_session']}")
    print(f"SEED_VALUES={result['seed_values']}")
    print(
        "TEST_SELECTION_LEAKAGE_FOUND="
        f"{result['test_selection_leakage_found']}"
    )
    print(f"OUTPUT_COLLISION_COUNT={result['output_collision_count']}")
    print(f"DUPLICATE_RUN_ID_COUNT={result['duplicate_run_id_count']}")
    print(f"RUNTIME_VALIDATION_COUNT={result['runtime_validation_count']}")
    print(f"GENERATED_COMMAND_COUNT={len(result['commands'])}")
    print(f"TRAINING_STARTED={result['training_started']}")
    for record in result["records"]:
        print(
            "RUN_SUMMARY="
            f"{record['run_id']}|model={record['model_family']}|"
            f"context={record['context_mode']}|"
            f"validation={record['validation_session']}|"
            f"test={record['test_session']}|seed={record['seed']}|"
            f"output={record['output_root']}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--mode", choices=("check", "prepare"), required=True)
    parser.add_argument("--experiment-date", default=None)
    parser.add_argument("--batch-id", default=None)
    args = parser.parse_args()
    if args.mode == "prepare" and not args.experiment_date:
        parser.error("--experiment-date is required in prepare mode")
    if args.mode == "prepare" and not args.batch_id:
        parser.error("--batch-id is required in prepare mode")
    return args


def main() -> None:
    args = parse_args()
    experiment_date = resolve_experiment_date(cli_date=args.experiment_date)
    result = prepare_formal_full16_repaired(
        args.matrix,
        args.mode,
        experiment_date,
        root=PROJECT_ROOT,
        batch_id=args.batch_id,
    )
    print_result(result, args.mode)


if __name__ == "__main__":
    main()

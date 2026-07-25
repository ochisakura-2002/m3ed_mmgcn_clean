"""Materialize and audit the paired IEMOCAP long-training matrix without training."""

from __future__ import annotations

import argparse
import copy
import re
import shlex
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

import yaml


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        value = yaml.safe_load(file)
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain a YAML mapping")
    return value


def set_dotted(config: dict[str, Any], dotted: str, value: object) -> None:
    current = config
    parts = dotted.split(".")
    for part in parts[:-1]:
        child = current.setdefault(part, {})
        if not isinstance(child, dict):
            raise TypeError(f"Cannot set {dotted}: {part} is not a mapping")
        current = child
    current[parts[-1]] = value


def _resolve(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def prepare_long_training_matrix(
    matrix_path: Path,
    mode: str,
    experiment_date: str,
    *,
    root: Path | None = None,
    resolved_root: Path | None = None,
) -> dict[str, Any]:
    """Expand one matrix, enforce its protocol, and optionally write YAML files."""

    if mode not in {"check", "prepare"}:
        raise ValueError(f"Unsupported mode: {mode!r}")
    if not re.fullmatch(r"\d{8}", experiment_date):
        raise ValueError(f"Invalid experiment date: {experiment_date!r}")

    project_root = Path.cwd() if root is None else Path(root)
    matrix_file = _resolve(project_root, Path(matrix_path))
    matrix = load_yaml(matrix_file)
    feature = matrix["feature"]
    matrix_protocol = matrix["protocol"]
    if matrix_protocol["checkpoint_selection_metric"] != "val_weighted_f1":
        raise ValueError("Checkpoint selection must use val_weighted_f1")
    if int(matrix_protocol["test_selection_leakage_found"]) != 0:
        raise ValueError("Test selection leakage must remain zero")

    expected_track = str(matrix_protocol["experiment_track"])
    expected_strategy = str(matrix_protocol["val_split_strategy"])
    expected_comparability = str(matrix_protocol["protocol_comparability"])
    expected_protocol_version = str(matrix_protocol["protocol_version"])
    if expected_strategy != "session_holdout":
        raise ValueError("Long-training matrix must use session_holdout validation")

    registry = load_yaml(_resolve(project_root, Path(feature["registry_path"])))
    entry = registry[feature["registry_key"]]
    for key in ("feature_set_name", "path", "sha256", "text_dim"):
        if entry[key] != feature[key]:
            raise ValueError(f"Feature registry mismatch for {key}")

    if resolved_root is None:
        resolved_dir = (
            Path("outputs")
            / experiment_date
            / "manifests"
            / str(matrix["matrix_name"])
            / "resolved_configs"
        )
    else:
        resolved_dir = Path(resolved_root)
    if mode == "prepare":
        _resolve(project_root, resolved_dir).mkdir(parents=True, exist_ok=False)

    run_ids: set[str] = set()
    output_roots: set[str] = set()
    commands: list[str] = []
    records: list[dict[str, Any]] = []
    pair_members: dict[tuple[str, str, str, str, str, int], Counter[str]] = {}
    model_counts: Counter[str] = Counter()
    context_counts: Counter[str] = Counter()
    order = 0

    for base_record in matrix["base_configs"]:
        base_path = Path(base_record["base_config"])
        parameter_source = Path(base_record["parameter_source"])
        entrypoint = Path(base_record["entrypoint"])
        for required in (base_path, parameter_source, entrypoint):
            resolved_required = _resolve(project_root, required)
            if not resolved_required.is_file():
                raise FileNotFoundError(resolved_required)
        base = load_yaml(_resolve(project_root, base_path))
        long_training = base["long_training"]
        if str(base.get("protocol_version")) != expected_protocol_version:
            raise ValueError(f"Protocol version mismatch in {base_path}")
        if long_training["parameter_source"] != base_record["parameter_source"]:
            raise ValueError(f"Parameter source mismatch in {base_path}")
        if long_training["entrypoint"] != base_record["entrypoint"]:
            raise ValueError(f"Entrypoint mismatch in {base_path}")

        session_rule = matrix["session_rules"][base_record["session_rule"]]
        for session in session_rule["expansions"]:
            for seed in matrix_protocol["seed_values"]:
                order += 1
                tokens = {
                    "model_family": base_record["model_family"],
                    "context_mode": base_record["context_mode"],
                    "validation_id": session["validation_id"],
                    "seed": int(seed),
                }
                run_id = matrix["expansion"]["run_id_template"].format(**tokens)
                tokens["run_id"] = run_id
                output_root = matrix["expansion"]["output_root_template"].format(**tokens)
                if run_id in run_ids:
                    raise ValueError(f"Duplicate run_id: {run_id}")
                if output_root in output_roots:
                    raise ValueError(f"Duplicate output_root: {output_root}")
                run_ids.add(run_id)
                output_roots.add(output_root)

                config = copy.deepcopy(base)
                for dotted, value in session.get("overrides", {}).items():
                    set_dotted(config, str(dotted), value)
                dataset = config.get("dataset", {})
                validation_session = str(session["validation_session"])
                test_session = str(session["test_session"])
                if dataset.get("experiment_track") != expected_track:
                    raise ValueError(f"{run_id} experiment track mismatch")
                if dataset.get("val_split_strategy") != expected_strategy:
                    raise ValueError(f"{run_id} split strategy mismatch")
                if dataset.get("protocol_comparability") != expected_comparability:
                    raise ValueError(f"{run_id} protocol comparability mismatch")
                if str(dataset.get("val_session_id")) != validation_session:
                    raise ValueError(f"{run_id} validation session mismatch")
                configured_test_session = dataset.get(
                    "test_session_id", dataset.get("outer_test_session")
                )
                if str(configured_test_session) != test_session:
                    raise ValueError(f"{run_id} test session mismatch")
                if config.get("protocol", {}).get(
                    "checkpoint_selection_metric"
                ) != "val_weighted_f1":
                    raise ValueError(f"{run_id} selection metric mismatch")
                if (
                    config.get("protocol", {}).get("test_split_used_for_selection")
                    is not False
                ):
                    raise ValueError(f"{run_id} uses test split for selection")

                pairing_implementation = str(long_training["protocol_lineage"])
                feature_set_name = str(dataset["feature_set_name"])
                pair_key = (
                    str(base_record["model_family"]),
                    pairing_implementation,
                    feature_set_name,
                    validation_session,
                    test_session,
                    int(seed),
                )
                pair_members.setdefault(pair_key, Counter())[
                    str(base_record["context_mode"])
                ] += 1
                model_counts[str(base_record["model_family"])] += 1
                context_counts[str(base_record["context_mode"])] += 1

                if "run_name" in config:
                    config["run_name"] = run_id
                if isinstance(config.get("project"), dict):
                    config["project"]["experiment_name"] = run_id
                if isinstance(config.get("output"), dict):
                    config["output"]["root"] = output_root
                    config["output"]["experiment_date"] = experiment_date
                    if "experiment_name" in config["output"]:
                        config["output"]["experiment_name"] = run_id
                if isinstance(config.get("system"), dict):
                    config["system"]["seed"] = int(seed)
                    if "output_dir" in config["system"]:
                        config["system"]["output_dir"] = output_root
                if (
                    isinstance(config.get("training"), dict)
                    and "seed" in config["training"]
                ):
                    config["training"]["seed"] = int(seed)
                if (
                    isinstance(config.get("dataset"), dict)
                    and "split_seed" in config["dataset"]
                ):
                    config["dataset"]["split_seed"] = int(seed)

                resolved_path = resolved_dir / f"{order:03d}_{run_id}.yaml"
                if mode == "prepare":
                    resolved_file = _resolve(project_root, resolved_path)
                    with resolved_file.open("x", encoding="utf-8") as file:
                        yaml.safe_dump(
                            config, file, sort_keys=False, allow_unicode=True
                        )
                command = [
                    "python",
                    "-u",
                    entrypoint.as_posix(),
                    "--config",
                    resolved_path.as_posix(),
                    "--experiment-date",
                    experiment_date,
                ]
                rendered_command = " ".join(shlex.quote(part) for part in command)
                commands.append(rendered_command)
                records.append(
                    {
                        "run_id": run_id,
                        "output_root": output_root,
                        "entrypoint": entrypoint.as_posix(),
                        "resolved_path": resolved_path,
                        "config": config,
                        "command": rendered_command,
                    }
                )

    if order != int(matrix["expected_run_count"]):
        raise ValueError(f"Expanded {order} runs; expected {matrix['expected_run_count']}")

    required_contexts = {"full_context", "causal_context"}
    unpaired_context_run_count = sum(
        sum(members.values())
        for members in pair_members.values()
        if set(members) != required_contexts
        or any(members[context] != 1 for context in required_contexts)
    )
    duplicate_pair_member_count = sum(
        max(0, count - 1)
        for members in pair_members.values()
        for count in members.values()
    )
    if unpaired_context_run_count:
        raise ValueError(f"Unpaired context runs: {unpaired_context_run_count}")
    if duplicate_pair_member_count:
        raise ValueError(f"Duplicate pair members: {duplicate_pair_member_count}")

    return {
        "matrix": matrix,
        "records": records,
        "commands": commands,
        "expanded_run_count": order,
        "pair_key_count": len(pair_members),
        "unpaired_context_run_count": unpaired_context_run_count,
        "duplicate_pair_member_count": duplicate_pair_member_count,
        "context_counts": context_counts,
        "model_counts": model_counts,
        "output_collision_count": order - len(output_roots),
    }


def print_result(result: Mapping[str, Any], mode: str) -> None:
    matrix = result["matrix"]
    context_counts = result["context_counts"]
    model_counts = result["model_counts"]
    print(f"MATRIX={matrix['matrix_name']}")
    print(f"ENABLED={str(bool(matrix['enabled'])).lower()}")
    print(f"EXPANDED_RUN_COUNT={result['expanded_run_count']}")
    print(f"GENERATED_COMMAND_COUNT={len(result['commands'])}")
    print(f"PAIR_KEY_COUNT={result['pair_key_count']}")
    print(f"UNPAIRED_CONTEXT_RUN_COUNT={result['unpaired_context_run_count']}")
    print(f"DUPLICATE_PAIR_MEMBER_COUNT={result['duplicate_pair_member_count']}")
    print(f"FULL_CONTEXT_RUN_COUNT={context_counts['full_context']}")
    print(f"CAUSAL_CONTEXT_RUN_COUNT={context_counts['causal_context']}")
    for model_family in sorted(model_counts):
        print(f"{model_family.upper()}_RUN_COUNT={model_counts[model_family]}")
    print(f"OUTPUT_COLLISION_COUNT={result['output_collision_count']}")
    print(
        "TEST_SELECTION_LEAKAGE_FOUND="
        f"{matrix['protocol']['test_selection_leakage_found']}"
    )
    print(f"MODE={mode}")
    for command in result["commands"]:
        print(command)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check or materialize the long-training config matrix."
    )
    parser.add_argument("matrix_path", type=Path)
    parser.add_argument("mode", choices=("check", "prepare"))
    parser.add_argument("experiment_date")
    parser.add_argument(
        "--resolved-root",
        type=Path,
        default=None,
        help="Optional config-only output override, primarily for local regression tests.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = prepare_long_training_matrix(
        args.matrix_path,
        args.mode,
        args.experiment_date,
        resolved_root=args.resolved_root,
    )
    print_result(result, args.mode)


if __name__ == "__main__":
    main()

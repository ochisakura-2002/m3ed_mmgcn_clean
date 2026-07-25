#!/usr/bin/env bash
set -euo pipefail

# Config preparation and command generation only. This script never executes
# the generated training commands.
# The paired primary matrix expands to 32 runs (16 full + 16 causal); the
# disabled three-seed matrix expands the same 16 pair keys to 96 runs.

MATRIX_PATH="${1:-configs/benchmarks/long_training/iemocap_clean/primary_seed42.yaml}"
MODE="${2:-check}"
EXPERIMENT_DATE="${MERC_EXPERIMENT_DATE:-$(date +%Y%m%d)}"

if [[ "${MODE}" != "check" && "${MODE}" != "prepare" ]]; then
  echo "Usage: $0 [matrix.yaml] [check|prepare]" >&2
  exit 2
fi

python - "${MATRIX_PATH}" "${MODE}" "${EXPERIMENT_DATE}" <<'PY'
from __future__ import annotations

import copy
import re
import shlex
import sys
from collections import Counter
from pathlib import Path

import yaml


ROOT = Path.cwd()
matrix_path = Path(sys.argv[1])
mode = sys.argv[2]
experiment_date = sys.argv[3]

if not re.fullmatch(r"\d{8}", experiment_date):
    raise SystemExit(f"Invalid experiment date: {experiment_date!r}")


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        value = yaml.safe_load(file)
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain a YAML mapping")
    return value


def set_dotted(config: dict, dotted: str, value: object) -> None:
    current = config
    parts = dotted.split(".")
    for part in parts[:-1]:
        child = current.setdefault(part, {})
        if not isinstance(child, dict):
            raise TypeError(f"Cannot set {dotted}: {part} is not a mapping")
        current = child
    current[parts[-1]] = value


matrix = load_yaml(matrix_path)
feature = matrix["feature"]
if matrix["protocol"]["checkpoint_selection_metric"] != "val_weighted_f1":
    raise ValueError("Checkpoint selection must use val_weighted_f1")
if int(matrix["protocol"]["test_selection_leakage_found"]) != 0:
    raise ValueError("Test selection leakage must remain zero")
registry = load_yaml(Path(feature["registry_path"]))
entry = registry[feature["registry_key"]]
for key in ("feature_set_name", "path", "sha256", "text_dim"):
    if entry[key] != feature[key]:
        raise ValueError(f"Feature registry mismatch for {key}")

resolved_root = (
    Path("outputs")
    / experiment_date
    / "manifests"
    / str(matrix["matrix_name"])
    / "resolved_configs"
)
if mode == "prepare":
    resolved_root.mkdir(parents=True, exist_ok=False)

run_ids: set[str] = set()
output_roots: set[str] = set()
commands: list[str] = []
pair_members: dict[tuple[str, str, str, str, str, int], Counter[str]] = {}
model_counts: Counter[str] = Counter()
context_counts: Counter[str] = Counter()
order = 0

for base_record in matrix["base_configs"]:
    base_path = Path(base_record["base_config"])
    parameter_source = Path(base_record["parameter_source"])
    entrypoint = Path(base_record["entrypoint"])
    for required in (base_path, parameter_source, entrypoint):
        if not required.is_file():
            raise FileNotFoundError(required)
    base = load_yaml(base_path)
    if base["long_training"]["parameter_source"] != base_record["parameter_source"]:
        raise ValueError(f"Parameter source mismatch in {base_path}")

    session_rule = matrix["session_rules"][base_record["session_rule"]]
    for session in session_rule["expansions"]:
        for seed in matrix["protocol"]["seed_values"]:
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
            if dataset.get("val_split_strategy") != "session_holdout":
                raise ValueError(f"{run_id} must use session_holdout validation")
            if str(dataset.get("val_session_id")) != validation_session:
                raise ValueError(f"{run_id} validation session mismatch")
            configured_test_session = dataset.get(
                "test_session_id", dataset.get("outer_test_session")
            )
            if str(configured_test_session) != test_session:
                raise ValueError(f"{run_id} test session mismatch")
            if config.get("protocol", {}).get("checkpoint_selection_metric") != "val_weighted_f1":
                raise ValueError(f"{run_id} selection metric mismatch")
            if config.get("protocol", {}).get("test_split_used_for_selection") is not False:
                raise ValueError(f"{run_id} uses test split for selection")

            pairing_implementation = str(base["long_training"]["protocol_lineage"])
            feature_set_name = str(dataset["feature_set_name"])
            pair_key = (
                str(base_record["model_family"]),
                pairing_implementation,
                feature_set_name,
                validation_session,
                test_session,
                int(seed),
            )
            pair_members.setdefault(pair_key, Counter())[str(base_record["context_mode"])] += 1
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
            if isinstance(config.get("training"), dict) and "seed" in config["training"]:
                config["training"]["seed"] = int(seed)
            if isinstance(config.get("dataset"), dict) and "split_seed" in config["dataset"]:
                config["dataset"]["split_seed"] = int(seed)

            resolved_path = resolved_root / f"{order:03d}_{run_id}.yaml"
            if mode == "prepare":
                with resolved_path.open("x", encoding="utf-8") as file:
                    yaml.safe_dump(config, file, sort_keys=False, allow_unicode=True)
            command = [
                "python",
                "-u",
                str(entrypoint).replace("\\", "/"),
                "--config",
                str(resolved_path).replace("\\", "/"),
                "--experiment-date",
                experiment_date,
            ]
            commands.append(" ".join(shlex.quote(part) for part in command))

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

print(f"MATRIX={matrix['matrix_name']}")
print(f"ENABLED={str(bool(matrix['enabled'])).lower()}")
print(f"EXPANDED_RUN_COUNT={order}")
print(f"PAIR_KEY_COUNT={len(pair_members)}")
print(f"UNPAIRED_CONTEXT_RUN_COUNT={unpaired_context_run_count}")
print(f"DUPLICATE_PAIR_MEMBER_COUNT={duplicate_pair_member_count}")
print(f"FULL_CONTEXT_RUN_COUNT={context_counts['full_context']}")
print(f"CAUSAL_CONTEXT_RUN_COUNT={context_counts['causal_context']}")
for model_family in sorted(model_counts):
    print(f"{model_family.upper()}_RUN_COUNT={model_counts[model_family]}")
print(f"OUTPUT_COLLISION_COUNT={order - len(output_roots)}")
print(f"TEST_SELECTION_LEAKAGE_FOUND={matrix['protocol']['test_selection_leakage_found']}")
print(f"MODE={mode}")
for command in commands:
    print(command)
PY

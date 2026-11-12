"""Reproduce Q2-1 and Q2-2 from seven explicitly supplied frozen Dev files."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
import platform
import subprocess
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import yaml

from scripts.analysis.models.asymmetric_merc.fusion_report import write_csv, write_json, write_report
from scripts.analysis.models.asymmetric_merc.weighted_fusion import (
    ALPHA_GRID, EPSILON, ISOLATION, LABEL_NAMES, PROBABILITY_SUM_ATOL, SEEDS, TIE_ATOL,
    AuditError, complementarity, evaluate_robustness, exact_join, load_predictions,
    metrics, select_alpha,
)
from utils.output_paths import resolve_output_paths
from utils.run_metadata import compute_file_sha256

REFERENCE = PROJECT_ROOT / "configs/asymmetric_merc/analysis/q2_2_dev_regression.yaml"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--ta-predictions", type=Path, required=True, help="Frozen TA-v0 Dev CSV/TSV")
    for seed in SEEDS:
        parser.add_argument(f"--linear-b-seed{seed}", type=Path, required=True,
                            help=f"Frozen Linear-B seed{seed} Dev CSV/TSV")
    parser.add_argument("--original-s2d", type=Path, required=True, help="Original S2D Dev reference CSV/TSV")
    parser.add_argument("--output-dir", type=Path, required=True, help="New, non-existing analysis batch directory")
    parser.add_argument("--synthetic", action="store_true",
                        help="Synthetic fixtures only; do not apply real-Dev regression references or claim a Dev signal")
    return parser


def _reserve_output(path: Path) -> Path:
    resolved = path.resolve()
    # Explicit outputs overrides still follow the repository's canonical layout.
    try:
        parts = resolved.relative_to(PROJECT_ROOT / "outputs").parts
    except ValueError:
        parts = None
    if parts is not None:
        if len(parts) != 5 or parts[2:4] != ("analysis", "batches"):
            raise ValueError("--output-dir under outputs/ must be <date>/<group>/analysis/batches/<batch_id>")
        expected = resolve_output_paths(output_base=PROJECT_ROOT / "outputs",
                                        experiment_date=parts[0], experiment_group=parts[1],
                                        batch_id=parts[4]).analysis_root
        if expected != resolved:
            raise ValueError("non-canonical output path")
    resolved.mkdir(parents=True, exist_ok=False)
    return resolved


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=PROJECT_ROOT, check=True,
                          capture_output=True, text=True, encoding="utf-8").stdout.rstrip()


def _regressions(actual: dict, expected: dict, atol: float, prefix: str = "") -> list[dict]:
    checks = []
    for key, target in expected.items():
        observed = actual.get(key)
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(target, dict):
            checks.extend(_regressions(observed or {}, target, atol, name))
        else:
            matches = (observed is not None and abs(observed - target) <=
                       (0 if isinstance(target, int) else atol))
            checks.append({"key": name, "expected": target, "actual": observed, "pass": matches})
    return checks


def _primary_regressions(actual: dict, expected: dict, atol: float) -> list[dict]:
    # Required coverage is independent of which fields the reference contains.
    # Run only after observations and the seed100 alpha have been computed.
    metric_keys = ("accuracy", "weighted_f1", "macro_f1", "uar")
    required = {
        "selection": ("alpha", *metric_keys),
        "ta_full_dev": metric_keys,
        "ta_common_dev": metric_keys,
        **{f"classwise.{name}": ("rescued", "ta_errors") for name in LABEL_NAMES},
    }
    missing = []
    for group, keys in required.items():
        targets, observations = expected, actual
        for part in group.split("."):
            targets = targets.get(part) if isinstance(targets, dict) else None
            observations = observations.get(part) if isinstance(observations, dict) else None
        for key in keys:
            if not isinstance(targets, dict) or targets.get(key) is None:
                missing.append({"key": f"{group}.{key}", "expected": None,
                                "actual": observations.get(key) if isinstance(observations, dict) else None,
                                "pass": False,
                                "error": "missing required frozen regression reference"})
    if missing:
        return missing
    return _regressions(actual, expected, atol)


def run(args: argparse.Namespace) -> dict:
    output = _reserve_output(args.output_dir)
    state = {"status": "RUNNING", "synthetic": args.synthetic, **ISOLATION}
    manifest = {"inputs": {}, "git_head": None, "git_dirty_status": None,
                "runtime_timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "working_directory": Path.cwd().as_posix(), "output_dir": output.as_posix(),
                "synthetic": args.synthetic, **ISOLATION}
    audits, checks = {}, []
    protocol = {
        "protocol_version": "q2_2_weighted_log_probability_dev_v1",
        "label_vocabulary": dict(enumerate(LABEL_NAMES)), "split": "dev",
        "formula": "alpha * log(max(P_TA, epsilon)) + (1-alpha) * log(max(P_V, epsilon))",
        "epsilon": EPSILON, "probability_sum_atol": PROBABILITY_SUM_ATOL,
        "probability_sum_rtol": 0, "probabilities_renormalized": False,
        "probability_clipping_count_unit": "individual probability entries strictly below epsilon",
        "alpha_grid": list(ALPHA_GRID), "selection_source": "linear_b_seed100",
        "selection_metric": "weighted_f1", "tie_atol": TIE_ATOL,
        "tie_break": "largest alpha within tie_atol of global max Weighted-F1",
        "argmax_tie_break": "first class in fixed label order", "alpha_selection_count": 1,
        "original_s2d_role": "complementarity_reference_only",
        "robustness": "fixed alpha, each seed's exact common subset and matched TA baseline",
        "zero_division_metrics": 0, "zero_denominator_rescue_rate": None,
        "regression_reference": REFERENCE.relative_to(PROJECT_ROOT).as_posix(),
        "synthetic": args.synthetic, **ISOLATION,
    }
    try:
        manifest.update(git_head=_git("rev-parse", "HEAD"),
                        git_dirty_status=_git("status", "--porcelain=v1", "--untracked-files=all"),
                        runtime_versions={name: version(name) for name in ("numpy", "scikit-learn", "PyYAML")},
                        python_version=platform.python_version(), command=sys.argv,
                        runtime_source_sha256={p.relative_to(PROJECT_ROOT).as_posix(): compute_file_sha256(p)
                                               for p in [*Path(__file__).parent.glob("*.py"),
                                                         PROJECT_ROOT / "utils/metrics.py",
                                                         PROJECT_ROOT / "utils/run_metadata.py",
                                                         PROJECT_ROOT / "utils/output_paths.py"]})
        reference = yaml.safe_load(REFERENCE.read_text(encoding="utf-8"))
        manifest["regression_reference_sha256"] = compute_file_sha256(REFERENCE)
        protocol["frozen_source_declarations"] = reference["source_declarations"]
        paths = {"ta": args.ta_predictions,
                 **{f"linear_b_seed{seed}": getattr(args, f"linear_b_seed{seed}") for seed in SEEDS},
                 "original_s2d": args.original_s2d}
        # Record every requested path even if an earlier input fails validation.
        manifest["inputs"] = {name: {"path": path.as_posix(), "status": "NOT_READ"}
                              for name, path in paths.items()}
        tables = {}
        for name, path in paths.items():
            try:
                tables[name] = load_predictions(path)
            except (OSError, ValueError, csv.Error) as exc:
                manifest["inputs"][name].update(getattr(exc, "audit", {}), status="FAIL", error=str(exc))
                raise
            manifest["inputs"][name] = {**tables[name].manifest, "status": "PASS"}
        joins, summaries, classwise = {}, {}, {}
        for name in paths:
            if name == "ta":
                continue
            try:
                joined = exact_join(tables["ta"], tables[name])
            except AuditError as exc:
                audits[name] = {**exc.audit, "status": "FAIL"}
                raise
            audits[name] = {**joined.audit, "status": "PASS"}
            summaries[name], classwise[name] = complementarity(joined)
            if name.startswith("linear_b_seed"):
                joins[int(name.removeprefix("linear_b_seed"))] = joined
        primary = summaries["linear_b_seed100"]
        state["primary"] = primary
        state["ta_full_dev"] = metrics(tables["ta"].true, tables["ta"].predicted)
        write_csv(output / "complementarity/primary_seed100_summary.csv", [primary])
        write_csv(output / "complementarity/primary_seed100_classwise.csv", classwise["linear_b_seed100"])
        write_csv(output / "complementarity/linear_b_seed_robustness.csv",
                  [{"seed": seed, **summaries[f"linear_b_seed{seed}"]} for seed in SEEDS])
        write_csv(output / "complementarity/original_s2d_reference.csv", [summaries["original_s2d"]])
        write_csv(output / "complementarity/original_s2d_reference_classwise.csv", classwise["original_s2d"])
        selection, grid = select_alpha(joins[100])
        state["selection"] = selection
        write_csv(output / "alpha_selection/alpha_grid_seed100.csv", grid)
        write_json(output / "alpha_selection/alpha_selection.json", {
            **selection, "source": "linear_b_seed100", "selection_count": 1,
            "maximum_grid_weighted_f1": max(row["weighted_f1"] for row in grid),
            "tie_atol": TIE_ATOL, "tie_break": protocol["tie_break"],
        })
        if not args.synthetic:
            actual = {
                "join": {k: audits["linear_b_seed100"][k] for k in ("ta_rows", "v_rows", "common_rows", "label_mismatch")},
                "complementarity": primary,
                "classwise": {row["class_name"]: row for row in classwise["linear_b_seed100"]},
                "ta_common_dev": {key: primary[f"ta_{key}"]
                                  for key in ("accuracy", "weighted_f1", "macro_f1", "uar")},
                "ta_full_dev": state["ta_full_dev"], "selection": selection,
                "delta_weighted_f1": selection["weighted_f1"] - primary["ta_weighted_f1"],
            }
            checks.extend(_primary_regressions(actual, reference["primary"], reference["metric_atol"]))
            if any(not check["pass"] for check in checks):
                raise AuditError("frozen Dev regression MISMATCH; stop before robustness (protocol unchanged)",
                                 {"failed_checks": [check for check in checks if not check["pass"]]})
        rows, evaluations = evaluate_robustness(joins, selection["alpha"])
        state["robustness"] = rows
        write_csv(output / "robustness/fixed_alpha_seed100_104.csv", rows)
        write_csv(output / "robustness/fixed_alpha_per_class.csv",
                  [{"seed": seed, "alpha": selection["alpha"], **row}
                   for seed in SEEDS for row in evaluations[seed]["per_class"]])
        result = evaluations[100]
        write_json(output / "fusion/seed100_metrics.json", {
            **result["metrics"], "alpha": selection["alpha"], "common_rows": len(joins[100].ids),
            "ta_full_dev": state["ta_full_dev"], "synthetic": args.synthetic, **ISOLATION,
        })
        write_csv(output / "fusion/seed100_per_class.csv", result["per_class"])
        write_csv(output / "fusion/seed100_confusion_matrix.csv", result["confusion_matrices"])
        write_csv(output / "fusion/seed100_predictions.csv", [
            {"utterance_id": uid, "true_label": int(joins[100].true[i]),
             "ta_predicted_label": int(joins[100].ta_predicted[i]),
             "v_predicted_label": int(joins[100].v_predicted[i]),
             "predicted_label": int(result["predictions"][i]), "alpha": selection["alpha"],
             **{f"ta_prob_{k}": float(joins[100].ta_probabilities[i, k]) for k in range(6)},
             **{f"v_prob_{k}": float(joins[100].v_probabilities[i, k]) for k in range(6)},
             **{f"score_{k}": float(result["scores"][i, k]) for k in range(6)}}
            for i, uid in enumerate(joins[100].ids)
        ])
        if not args.synthetic:
            positive = sum(row["delta_vs_ta_weighted_f1"] > 0 for row in rows)
            checks.extend(_regressions({"positive_weighted_f1_seeds": positive},
                                       reference["robustness"], 0))
            if any(not check["pass"] for check in checks):
                raise AuditError("fixed-alpha robustness regression MISMATCH; protocol unchanged")
        state["status"] = "PASS"
    except (OSError, ValueError, csv.Error, subprocess.SubprocessError) as exc:
        state.update(status="FAIL", error=str(exc), error_details=getattr(exc, "audit", {}))
    finally:
        write_json(output / "provenance/input_manifest.json", manifest)
        write_json(output / "provenance/protocol.json", protocol)
        write_json(output / "provenance/exact_join_audit.json", audits)
        write_json(output / "provenance/regression_audit.json", {
            "status": "NOT_APPLICABLE_SYNTHETIC" if args.synthetic else
                      ("PASS" if checks and all(c["pass"] for c in checks) else "FAIL_OR_INCOMPLETE"),
            "checks": checks,
        })
        write_json(output / "provenance/run_status.json", state)
        state["report_path"] = str(write_report(output, state))
    return state


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        state = run(args)
    except (OSError, ValueError) as exc:
        print(f"Q2_2_RUNTIME=FAIL: {exc}", file=sys.stderr)
        return 1
    print(f"Q2_2_RUNTIME={state['status']}")
    print(f"REPORT={state['report_path']}")
    if "error" in state:
        print(f"STOP / REPORT MISMATCH: {state['error']}", file=sys.stderr)
    return 0 if state["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())

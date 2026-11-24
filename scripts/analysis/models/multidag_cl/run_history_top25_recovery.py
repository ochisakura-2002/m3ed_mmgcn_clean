"""Recover the frozen seed100 F0 history top-25% from Dev-only inference."""

import argparse
from functools import partial
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

import numpy as np
import torch

from scripts.analysis.models.multidag_cl.failure_audit_inference import HISTORY, evaluate, frozen_model
from scripts.analysis.models.multidag_cl.failure_audit_metrics import details, history_audit, metadata, selection_groups
from scripts.analysis.models.multidag_cl.pairwise_preflight_inputs import (
    CLASS_ORDER, HOLD, Q4_SUPPORT, SEED, TRAIN_DEV_COUNTS, FrozenSplit, TestHoldViolation,
    checked_round1_mad, frozen_forward, history_predictions, load_projected_train_dev, read_f0, read_round1_manifest,
    stable_top25, uh_partitions, validate_f0_dev, validate_hold, validate_population,
)
from scripts.analysis.models.multidag_cl.pairwise_preflight_metrics import scores
from scripts.analysis.models.multidag_cl.run_pairwise_preflight import bundle
from scripts.dev.check_workstream import check_workstream
from scripts.runtime.multidag_cl_ta_v0.formal import git_identity, json_identity, sha256_file, write_json_atomic
from scripts.runtime.multidag_cl_ta_v0.training import seed_everything
from scripts.runtime.multidag_cl_tav_full_audit_v1.spec import write_csv
from utils.output_paths import resolve_experiment_date, resolve_output_paths


GROUP = "history_top25_provenance_recovery"
HISTORY_CSV = "HISTORY_UTILITY_AUDIT_SEED100_RECOVERED.csv"
H_CSV = "H_TOP25_MEMBERSHIP_SEED100.csv"
UH_CSV = "UH_SUBGROUP_MEMBERSHIP_SEED100.csv"
SHIFT_CSV = "SHIFT_STABLE_MEMBERSHIP_SEED100.csv"
MANIFEST = "history_top25_recovery_manifest.json"
REPORT = "HISTORY_TOP25_RECOVERY_REPORT.md"
DEV_SIZE = TRAIN_DEV_COUNTS["dev"]["utterances"]
ANCHORS = {"DEV_POPULATION": DEV_SIZE, "HISTORY_SAMPLE_ROWS": DEV_SIZE * len(HISTORY),
           "H_TOP25_SUPPORT": Q4_SUPPORT, "H_TOP25_F0_ERRORS": 104, "F0_TOTAL_ERRORS": 287,
           "U_TOP25_SUPPORT": Q4_SUPPORT, "U_TOP25_F0_ERRORS": 102,
           "SHIFT": 273, "STABLE": 667, "NO_HISTORY": 40}


def recovery_tables(measured, meta, archived, mad):
    if set(measured) != set(HISTORY) or len(meta) != DEV_SIZE:
        raise ValueError("expected six history conditions on all 980 Dev utterances")
    shifts = selection_groups(meta)
    variants = {condition: details(measured[condition], archived.labels) for condition in HISTORY}
    rows, _, _, _ = history_audit(SEED, variants, meta, shifts)
    samples = [row for row in rows if row["row_type"] == "sample"]
    predictions = history_predictions(samples, meta)
    if not np.array_equal(predictions[0], archived.top1):
        raise ValueError("recovered FULL_HISTORY sample predictions differ from archived F0")

    changed = predictions != predictions[0]
    history_count = changed[1:].sum(axis=0)
    h_rank = stable_top25(history_count)
    indices = np.arange(DEV_SIZE)
    u = np.isin(indices, stable_top25(mad))
    h = np.isin(indices, h_rank)
    partitions = uh_partitions(u, h)
    if not np.all(np.stack(list(partitions.values())).sum(axis=0) == 1):
        raise ValueError("U/H membership must partition all Dev utterances exactly once")
    subgroup = np.select(list(partitions.values()), list(partitions), default="")

    dev_indices = {row["utterance_id"]: index for index, row in enumerate(meta)}
    condition_indices = {name: index for index, name in enumerate(HISTORY)}
    for row in samples:
        index = dev_indices[row["utterance_id"]]
        condition = row["condition"]
        row.update(dev_index=index, gold=row["true_label"], prediction=row["predicted_label"],
                   is_full_history=condition == "FULL_HISTORY",
                   prediction_changed_vs_full=bool(changed[condition_indices[condition], index]))

    dev_rows = [{"seed": SEED, "dev_index": index, **row, "gold": row["true_label"],
                 "full_prediction": CLASS_ORDER[predictions[0, index]], "history_change_count": int(history_count[index])}
                for index, row in enumerate(meta)]
    h_rows = [{**dev_rows[index], "rank": rank, "is_h_top25": True,
               **{f"{name}_changed": bool(changed[condition_indices[name], index]) for name in HISTORY[1:]}}
              for rank, index in enumerate(h_rank, 1)]
    uh_rows = [{**row, "is_u_top25": bool(u[index]), "is_h_top25": bool(h[index]),
                "subgroup": str(subgroup[index])} for index, row in enumerate(dev_rows)]
    shift_rows = [{"seed": SEED, "dev_index": index, **row, "gold": row["true_label"]}
                  for index, row in enumerate(meta)]
    observed = {"DEV_POPULATION": len(meta), "HISTORY_SAMPLE_ROWS": len(samples), "H_TOP25_SUPPORT": len(h_rows),
                "H_TOP25_F0_ERRORS": int((h & archived.wrong).sum()), "F0_TOTAL_ERRORS": int(archived.wrong.sum()),
                "U_TOP25_SUPPORT": int(u.sum()), "U_TOP25_F0_ERRORS": int((u & archived.wrong).sum()),
                **{name: int(shifts[name].sum()) for name in ("SHIFT", "STABLE", "NO_HISTORY")},
                "uh_subgroup_counts": {name: int(mask.sum()) for name, mask in partitions.items()},
                "history_change_count_histogram": np.bincount(history_count, minlength=len(HISTORY)).tolist(),
                "history_condition_flip_counts": dict(zip(HISTORY, changed.sum(axis=1).tolist())),
                "h_top25_cutoff_change_count": int(history_count[h_rank[-1]])}
    return {HISTORY_CSV: samples, H_CSV: h_rows, UH_CSV: uh_rows, SHIFT_CSV: shift_rows}, observed


def recover(args, manifest, device):
    source = read_f0(args.frozen_runs)
    datasets, projection = load_projected_train_dev(args.train_dev_package, source, splits=("dev",))
    populations = validate_population(datasets, source, splits=("dev",))
    meta = metadata(source.population)
    archived = scores(source.logits, [label for _, _, label in source.population])
    round1, round1_sha = read_round1_manifest(source, args.round1_dir)
    mad, _ = checked_round1_mad(args.round1_dir, round1, meta, archived)
    manifest.update(f0_run_id=source.config["experiment_id"], f0_checkpoint=source.identity,
                    f0_dev_metrics=source.manifest["best_dev_metrics"], train_dev_projection=projection,
                    dev_population=populations["dev"], dev_gold_population_sha256=json_identity(source.population),
                    archived_dev_logits_sha256=json_identity(source.logits.tolist()),
                    round1_manifest_sha256=round1_sha,
                    round1_ambiguity_sha256=round1["outputs_sha256"]["AMBIGUITY_AUDIT.csv"])

    seed_everything(device, SEED)
    model = frozen_model(source, source.manifest["curriculum_membership_sha256"], device,
                         check_test_hold=partial(validate_hold, legacy=True))
    batch_size = source.config["training"]["batch_size"]
    dev = datasets["dev"]
    # Close the full-population identity gate before running any restricted condition.
    full = frozen_forward(model, dev, "dev", device=device, batch_size=batch_size,
                          expected_batches=source.manifest["batch_pairing"]["dev"]["batch_sha256"])
    validate_f0_dev(source, full)
    manifest["dev_logits_max_abs_diff"] = float(np.max(np.abs(full.logits - source.logits)))
    measured = evaluate(model, dev, "dev", device=device, chunk_size=args.chunk_size,
                        history_only=True, batch_size=batch_size)
    # Recheck the baseline from the actual counterfactual batches before ranking.
    validate_f0_dev(source, FrozenSplit(measured["FULL_HISTORY"], full.population))
    if sha256_file(source.checkpoint) != source.identity["checkpoint_sha256"]:
        raise ValueError("checkpoint changed during history recovery")
    manifest["F0_IDENTITY"] = "PASS"
    return recovery_tables(measured, meta, archived, mad)


def write_report(output, manifest):
    lines = ["# Seed100 history top-25% provenance recovery", "",
             f'**{manifest["PROVENANCE_RECOVERY"]}**', "",
             "The score counts the five restricted predictions that differ from FULL_HISTORY. "
             "Ties follow frozen Dev order. The Dev index starts at zero and rank starts at one. "
             "U uses the frozen Round 1 MAD trace and the Round 2 stable descending order.", "",
             f'F0 identity `{manifest["F0_IDENTITY"]}`. Test touched `{manifest["TEST_TOUCHED"]}`. '
             f'Formal training runs `{manifest["FORMAL_TRAIN_RUNS"]}`. '
             f'Optimizer steps `{manifest["OPTIMIZER_STEPS"]}`.', "",
             "| Anchor | Observed | Expected |", "| --- | ---: | ---: |"]
    lines.extend(f'| `{key}` | {manifest.get(key, "unverified")} | {expected} |' for key, expected in ANCHORS.items())
    if manifest.get("F0_TOTAL_ERRORS"):
        capture = 100 * manifest["H_TOP25_F0_ERRORS"] / manifest["F0_TOTAL_ERRORS"]
        lines.extend(["", f"H-top25 captured {capture:.2f}% of F0 Dev errors."])
    if "reason" in manifest:
        lines.extend(["", manifest["reason"]])
    (output / REPORT).write_text("\n".join(lines) + "\n", encoding="utf-8")


def execute(args, output, archive, *, device, git, experiment_date):
    if archive.exists():
        raise FileExistsError("existing recovery archive cannot be overwritten")
    output.mkdir(parents=True, exist_ok=False)
    manifest = {**git, **HOLD, "schema": "ct_multidag_history_top25_recovery_v1", "WORKSTREAM_CHECK": "PASS",
                "experiment_date": experiment_date, "experiment_group": GROUP, "batch_id": args.batch_id,
                "device": device, "split": "dev", "seed": SEED, "history_conditions": list(HISTORY),
                "chunk_size": args.chunk_size, "F0_IDENTITY": "FAIL", "FORMAL_TRAIN_RUNS": 0, "OPTIMIZER_STEPS": 0,
                "TEST_TOUCHED": "NO", "PROVENANCE_RECOVERY": "REVIEW_INVALID",
                "H_TOP25_PROVENANCE": "FAIL", "H_SUBGROUP_PROVENANCE": "FAIL"}
    tables = {}
    try:
        validate_hold(manifest)
        tables, observed = recover(args, manifest, device)
        manifest.update(observed)
        differences = [f"{key} observed {observed[key]}, expected {expected}"
                       for key, expected in ANCHORS.items() if observed[key] != expected]
        if differences:
            manifest.update(PROVENANCE_RECOVERY="FAIL", reason="Frozen anchor mismatch. " + ". ".join(differences))
            tables = {HISTORY_CSV: tables[HISTORY_CSV]}
        else:
            manifest.update(PROVENANCE_RECOVERY="PASS", H_TOP25_PROVENANCE="PASS", H_SUBGROUP_PROVENANCE="PASS")
    except Exception as error:
        manifest.update(reason=f"{type(error).__name__} {error}")
        if isinstance(error, TestHoldViolation):
            manifest.update(TEST_TOUCHED="YES", rejected_input_hold_evidence=error.evidence)
        tables = {}

    for name, rows in tables.items():
        write_csv(output / name, rows)
    write_report(output, manifest)
    manifest["outputs_sha256"] = {name: sha256_file(output / name) for name in (*tables, REPORT)}
    write_json_atomic(output / MANIFEST, manifest)
    if manifest["PROVENANCE_RECOVERY"] == "PASS":
        try:
            digest = bundle(output, archive, (*tables, REPORT, MANIFEST))
        except (OSError, ValueError) as error:
            manifest.update(PROVENANCE_RECOVERY="REVIEW_INVALID", H_TOP25_PROVENANCE="FAIL",
                            H_SUBGROUP_PROVENANCE="FAIL", reason=f"Archive publication failed. {error}")
            write_report(output, manifest)
            manifest["outputs_sha256"][REPORT] = sha256_file(output / REPORT)
            write_json_atomic(output / MANIFEST, manifest)
        else:
            print(f"RECOVERY_ARCHIVE = {archive}\nRECOVERY_ARCHIVE_SHA256 = {digest}", flush=True)
    for key in ("PROVENANCE_RECOVERY", "F0_IDENTITY", *ANCHORS, "H_TOP25_PROVENANCE", "H_SUBGROUP_PROVENANCE",
                *HOLD, "TEST_TOUCHED", "FORMAL_TRAIN_RUNS", "OPTIMIZER_STEPS", "reason"):
        if key in manifest:
            print(f"{key} = {manifest[key]}", flush=True)
    return 0 if manifest["PROVENANCE_RECOVERY"] == "PASS" else 2


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-runs", required=True, type=Path, help="parent directory of the frozen seed100 F0 run")
    parser.add_argument("--train-dev-package", required=True, type=Path, help="certified Train/Dev-only TAV projection")
    parser.add_argument("--round1-dir", required=True, type=Path, help="verified Round 1 directory with the frozen MAD trace")
    parser.add_argument("--batch-id", required=True)
    parser.add_argument("--experiment-date")
    parser.add_argument("--output-root", default="outputs")
    parser.add_argument("--chunk-size", type=int, default=8)
    args = parser.parse_args(argv)
    errors = check_workstream("multidag")
    if errors:
        print("WORKSTREAM_CHECK = FAIL", flush=True)
        parser.error("\n".join(errors))
    print("WORKSTREAM_CHECK = PASS", flush=True)
    if args.chunk_size < 1:
        parser.error("--chunk-size must be positive")
    if not sys.platform.startswith("linux") or not torch.cuda.is_available():
        parser.error("formal frozen recovery requires the Ubuntu CUDA workspace")
    date = resolve_experiment_date(args.experiment_date)
    paths = resolve_output_paths(output_base=args.output_root, experiment_date=date, experiment_group=GROUP,
                                 batch_id=args.batch_id)
    output = paths.analysis_root
    archive = output.parent / f"{output.name}.zip"
    return execute(args, output, archive, device="cuda", git=git_identity(), experiment_date=date)


if __name__ == "__main__":
    raise SystemExit(main())

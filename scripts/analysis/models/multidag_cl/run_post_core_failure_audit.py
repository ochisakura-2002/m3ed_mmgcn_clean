"""Run POST_CORE_FAILURE_MECHANISM_AUDIT_001 or summarize its archived Dev baseline."""

import argparse
import csv
from pathlib import Path
import sys
from zipfile import ZIP_DEFLATED, ZipFile

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

import numpy as np
import torch

from scripts.analysis.models.multidag_cl.export_fusion_topology_bundle import check_hold, read_json
from scripts.analysis.models.multidag_cl.failure_audit_inference import (
    HISTORY, MASKS, REPEATS, context_logits, donor_plans, evaluate, frozen_model, membership, sources_from_runs,
)
from scripts.analysis.models.multidag_cl.failure_audit_metrics import (
    PAIRS, ambiguity_audit, average, context_audit, decision, details, history_audit, metadata,
    modality_audit, modality_group, seed_disagreement, selection_groups, shift_audit,
)
from scripts.dev.check_workstream import check_workstream
from scripts.runtime.multidag_cl_ta_v0.formal import git_identity, sha256_file, write_json_atomic
from scripts.runtime.multidag_cl_ta_v0.training import seed_everything
from scripts.runtime.multidag_cl_tav_full_audit_v1.runtime import load_train_dev
from scripts.runtime.multidag_cl_tav_full_audit_v1.spec import read_csv
from utils.output_paths import resolve_experiment_date, resolve_output_paths


GROUP = "post_core_failure_mechanism_audit_001"
RUNBOOK = Path("docs/research/post_fusion/FAILURE_AUDIT_RUNBOOK.md")
HOLD = {"test_policy": "HOLD", "test_loader_built": 0, "test_labels_inspected": 0,
        "test_evaluation_count": 0, "test_checkpoint_selection_count": 0}
TABLES = {
    "shift": "SHIFT_STABLE_PERFORMANCE.csv",
    "context": "CONTEXT_SUBSTITUTION_AUDIT.csv",
    "history": "HISTORY_UTILITY_AUDIT.csv",
    "helpful": "HISTORY_HELPFUL_SAMPLES.csv",
    "harmful": "HISTORY_HARMFUL_SAMPLES.csv",
    "ambiguity": "AMBIGUITY_AUDIT.csv",
    "quartile": "AMBIGUITY_QUARTILE_SUMMARY.csv",
    "modality": "MODALITY_RELIABILITY_AUDIT.csv",
    "recovery": "MODALITY_ERROR_RECOVERY.csv",
    "train_dev": "TRAIN_DEV_FAILURE_SUMMARY.csv",
    "matrix": "FAILURE_MECHANISM_EVIDENCE_MATRIX.csv",
}
ROUTES = {"H": "Causal Selective History", "U": "Ambiguity-aware", "M": "Dynamic modality reliability"}
DESIGNS = {
    "H": "Use `r_ij = sigmoid(f(h_i, h_j, relation_ij, delta_t))` and `m_ij = r_ij * message_ij` "
         "inside TrackCore aggregation for `j < i`. Preserve the self term. A disabled gate recovers F0.",
    "U": "Use UAF-lite with one MAD-guided auxiliary weighting term. Compute MAD from the "
         "six-class prediction distribution, detach it for weighting, and retain F0 inference.",
    "M": "Use a sample-conditioned T/A/V reliability gate over the three encoded blocks before "
         "TrackCore. Anchor the disabled path to F0 DirectConcat.",
}


def check_dev_alignment(samples, population):
    actual = [(uid, sample["dialogue_id"], int(label))
              for sample in samples for uid, label in zip(sample["utterance_ids"], sample["labels"])]
    if tuple(actual) != population:
        raise ValueError("Dev dataset order, IDs or targets differ from the archived F0 evidence")


def comparison(seed, labels, full, variants, split):
    result = {"seed": seed, "split": split, "support": len(labels), "source": "frozen_forward",
              "mean_mad": average(full["mad"]), "error_rate": average(~full["correct"])}
    if "CURRENT_ONLY" in variants:
        current = variants["CURRENT_ONLY"]
        result.update(mean_abs_history_logit_delta=average(np.abs(
            full["true_class_logit_margin"] - current["true_class_logit_margin"])),
                      history_prediction_flip_rate=average(full["prediction"] != current["prediction"]))
    if all(f"{modality}_ONLY" in variants for modality in "TAV"):
        result["modality_disagreement_rate"] = average(modality_group(variants) != "ALL_AGREE")
    return result


def b0_rows(directory, sources):
    if directory is None:
        return [{"source": "B0", "status": "NOT_PROVIDED"}]
    path = Path(directory)
    manifest = read_json(path / "b0_manifest.json")
    check_hold(manifest)
    identities = {row["seed"]: row for row in manifest["checkpoints"]}
    if (manifest["B0_DIAGNOSTIC"] != "PASS" or manifest["centroid_reference_split"] != "train"
            or manifest["optimizer_steps_performed"] != 0
            or manifest["data_identity"] != next(iter(sources.values())).manifest["data_identity"]
            or identities != {seed: source.identity for seed, source in sources.items()}):
        raise ValueError("B0 geometry identity differs from the frozen F0 checkpoints")
    rows = []
    for seed in sources:
        seed_path = path / f"seed{seed}"
        pairs = [row for row in read_csv(seed_path / "pair_geometry.csv")
                 if (row["true_label"], row["other_label"]) in PAIRS
                 or (row["other_label"], row["true_label"]) in PAIRS]
        classes = read_csv(seed_path / "class_geometry.csv")
        pair_keys = {(row["split"], row["true_label"], row["other_label"]) for row in pairs}
        expected = {(split, x, y) for split in ("train", "dev") for a, b in PAIRS for x, y in ((a, b), (b, a))}
        if len(pairs) != len(expected) or pair_keys != expected or any(int(row["seed"]) != seed for row in pairs + classes):
            raise ValueError("B0 geometry table identity mismatch")
        rows.extend({"source": "B0_pair_geometry", **row} for row in pairs)
        rows.extend({"source": "B0_class_geometry", **row} for row in classes)
    return rows


def collect(sources, *, datasets, device, chunk_size, geometry):
    tables = {key: [] for key in TABLES}
    tables["train_dev"].extend(b0_rows(geometry, sources))
    population = next(iter(sources.values())).population
    labels = np.asarray([label for _, _, label in population])
    meta = metadata(population)
    groups = selection_groups(meta)
    original = {seed: details(source.logits, labels) for seed, source in sources.items()}
    disagreement = seed_disagreement([row["prediction"] for row in original.values()])
    if datasets is not None:
        train, dev = (list(datasets[split]) for split in ("train", "dev"))
        check_dev_alignment(dev, population)
        train_labels = np.concatenate([sample["labels"].numpy() for sample in train])
        curriculum_sha = membership(datasets["train"])
        plans = donor_plans(train, dev)
    per_seed = {}
    for seed, source in sources.items():
        base = original[seed]
        record = per_seed[seed] = {}
        shift_rows, record["shift"] = shift_audit(seed, base, labels, groups)
        tables["shift"].extend(shift_rows)
        recovered = None
        variants = {}
        if datasets is not None:
            print(f"FROZEN_FORWARD_SEED = {seed}", flush=True)
            seed_everything(device, seed)
            model = frozen_model(source, curriculum_sha, device)
            measured = evaluate(model, dev, "dev", device=device, chunk_size=chunk_size)
            if (not np.allclose(measured["FULL_HISTORY"], source.logits, rtol=1e-4, atol=5e-4)
                    or not np.array_equal(measured["FULL_HISTORY"].argmax(axis=1), base["prediction"])):
                raise ValueError(f"seed {seed} live F0 logits/predictions differ from archived Dev evidence")
            variants = {name: details(value, labels) for name, value in measured.items()}
            base = variants["FULL_HISTORY"]
            substitutions = {key: details(value, labels) for key, value in
                             context_logits(model, dev, plans, device=device, chunk_size=chunk_size).items()}
            rows, record["context"] = context_audit(seed, base, substitutions, labels, groups)
            tables["context"].extend(rows)
            rows, helped, harmed, record["history"] = history_audit(
                seed, {name: variants[name] for name in HISTORY}, meta, groups)
            tables["history"].extend(rows)
            tables["helpful"].extend(helped)
            tables["harmful"].extend(harmed)
            recovered = np.logical_or.reduce([variants[name]["correct"] for name in HISTORY[1:]])
            recovered |= np.stack([substitutions["ECCS", repeat]["correct"] for repeat in REPEATS]).sum(axis=0) >= 2
            recovered &= ~base["correct"]
            if set(MASKS) <= measured.keys():
                rows, recovery, record["modality"] = modality_audit(seed, variants, labels, meta)
                tables["modality"].extend(rows)
                tables["recovery"].extend(recovery)
            train_variants = {name: details(values, train_labels) for name, values in
                              evaluate(model, train, "train", device=device, chunk_size=chunk_size).items()}
            train_row = comparison(seed, train_labels, train_variants["FULL_HISTORY"], train_variants, "train")
            tables["train_dev"].append(train_row)
            if sha256_file(source.checkpoint) != source.identity["checkpoint_sha256"]:
                raise ValueError(f"seed {seed} checkpoint changed during audit")
            del model
        dev_row = comparison(seed, labels, base, variants, "dev")
        if datasets is None:
            dev_row["source"] = "archived_dev_evidence"
        else:
            for key in ("mean_mad", "mean_abs_history_logit_delta", "history_prediction_flip_rate",
                        "modality_disagreement_rate"):
                if key in dev_row and key in train_row:
                    dev_row[f"dev_minus_train_{key}"] = dev_row[key] - train_row[key]
        tables["train_dev"].append(dev_row)
        samples, summaries, record["ambiguity"] = ambiguity_audit(
            seed, base, labels, meta, groups, disagreement, recovered)
        tables["ambiguity"].extend(samples)
        tables["quartile"].extend(summaries)
    if datasets is not None and len({("modality" in row) for row in per_seed.values()}) != 1:
        raise ValueError("modality mask availability changed across seeds")
    return tables, per_seed


def evidence_matrix(outcome):
    return [
        {"mechanism": "History / Context", "test_1": f'SHIFT [{outcome["H_SHIFT_SIGNAL"]}]',
         "test_2": f'ECCS/EICS [{outcome["LABEL_REPLICATION"]}]',
         "test_3": f'History utility [{outcome["SELECTIVE_HISTORY_SIGNAL"]}]',
         "evidence": outcome["history_total"], "route": outcome["ROUTE_H"]},
        {"mechanism": "Ambiguity", "test_1": f'MAD [{outcome["AMBIGUITY_SIGNAL"]}]',
         "test_2": f'{outcome["ambiguity_proxy_agreement"]}/3 concordant MAD / entropy / margin proxies',
         "test_3": "Five-seed disagreement by MAD quartile",
         "evidence": outcome["AMBIGUITY_SIGNAL"], "route": outcome["ROUTE_U"]},
        {"mechanism": "Modality reliability", "test_1": "Same-checkpoint single-modality masks",
         "test_2": "Single-modality oracle and fused-error recovery", "test_3": "Disagreement error gradient",
         "evidence": outcome["MODALITY_RELIABILITY_SIGNAL"], "route": outcome["ROUTE_M"]},
    ]


def write_table(path, rows):
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(key for row in rows for key in row)),
                                lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def round_two_plan(outcome):
    top = sorted(outcome["effect_pp"], key=lambda route: (
        -float(np.median(outcome["effect_pp"][route])), "HUM".index(route)))[:2]
    scores = {"H": "fraction of the five restricted-history predictions that differ from FULL_HISTORY",
              "U": "MAD from full TAV probabilities",
              "M": "fraction of the three single-modality predictions that differ from FULL_TAV"}
    candidates = "\n".join(f"- {route} ranks by the {scores[route]}." for route in top)
    return (f"# POST_CORE_FAILURE_MECHANISM_AUDIT_002_PLAN\n\n"
            f"Compare {ROUTES[top[0]]} and {ROUTES[top[1]]} in one paired selective-risk experiment "
            "on the same five frozen F0 Dev predictions. Use zero training. Test stays on HOLD.\n\n"
            "Rank utterances without gold labels and reserve exactly the top 25% per seed and route. "
            "Use frozen Dev order to break tied ranks. After selection, score the fraction of original F0 "
            "errors captured at this equal review budget. Reuse the Round 1 CSV predictions.\n\n"
            f"{candidates}\n\n"
            "Choose the route with the higher median paired benefit across seeds. "
            "Resolve an exact tie by H > U > M and advance only that route.\n")


def report_pair_errors(rows):
    selected = [row for row in rows if row["proxy"] == "mad" and row["group"] == "ALL"]
    lines = ["", "| Confusion direction | MAD Q1 errors / class support | MAD Q4 errors / class support |",
             "| --- | ---: | ---: |"]
    for a, b in PAIRS:
        for left, right in ((a, b), (b, a)):
            values = []
            for quartile in (1, 4):
                group = [row for row in selected if row["quartile"] == quartile]
                count = sum(row[f"{left}_to_{right}_count"] for row in group)
                support = sum(row[f"{left}_support"] for row in group)
                values.append(f"{count}/{support} ({count / support:.2%})" if support else "N/A")
            lines.append(f"| {left} → {right} | {values[0]} | {values[1]} |")
    lines += ["", "Counts are seed–utterance decisions. The denominator is the true-class support in each quartile."]
    return lines


def report_train_dev(rows):
    lines = ["", "| Measure | Mean Dev − Train | Dev higher |",
             "| --- | ---: | ---: |"]
    for key, label in (("mean_mad", "MAD"), ("mean_abs_history_logit_delta", "Absolute graph-history logit delta"),
                       ("history_prediction_flip_rate", "Graph-history prediction flip rate"),
                       ("modality_disagreement_rate", "Single-modality disagreement rate")):
        values = [row[f"dev_minus_train_{key}"] for row in rows if f"dev_minus_train_{key}" in row]
        if values:
            lines.append(f"| {label} | {average(values):+.4f} | {sum(value > 0 for value in values)}/{len(values)} |")
    for a, b in PAIRS:
        spread = []
        for split in ("train", "dev"):
            matching = [float(row["separation_ratio"]) for row in rows if row["source"] == "B0_pair_geometry"
                        and row["split"] == split and row["true_label"] == a and row["other_label"] == b]
            if matching:
                spread.append(average(matching))
        if len(spread) == 2:
            lines.append(f"| B0 {a}/{b} separation ratio | {spread[1] - spread[0]:+.3f} | — |")
    return lines if len(lines) > 3 else []


def write_reports(directory, tables, per_seed, manifest):
    tables["matrix"] = evidence_matrix(manifest)
    for key, path in TABLES.items():
        status = "PENDING" if manifest["mode"] == "archived_dev" else (
            "NOT_AVAILABLE" if key in ("modality", "recovery") else "NO_SAMPLES")
        write_table(directory / path, tables[key] or [{"status": status}])
    status_keys = ("WORKSTREAM_CHECK", "FORMAL_TRAIN_RUNS", "TEST_TOUCHED", "H_SHIFT_SIGNAL", "LABEL_REPLICATION",
                   "SELECTIVE_HISTORY_SIGNAL", "AMBIGUITY_SIGNAL", "MODALITY_RELIABILITY_SIGNAL", "ROUTE_H",
                   "ROUTE_U", "ROUTE_M", "NEXT_ROUTE", "ROUND_2_REQUIRED", "READY_FOR_NEXT_METHOD_FASTKILL",
                   "HANDOFF_READY")
    status = "\n".join(f"{key} = {manifest[key]}" for key in status_keys)
    route = manifest["NEXT_ROUTE"]
    next_report = f'# NEXT_ROUTE_DECISION\n\n{manifest["rationale"]}\n\n'
    if route in ROUTES:
        next_report += f"## {ROUTES[route]}\n\n{DESIGNS[route]}\n\n"
    next_report += f"```text\n{status}\n```\n"
    (directory / "NEXT_ROUTE_DECISION.md").write_text(next_report, encoding="utf-8")
    if manifest["ROUND_2_REQUIRED"] == "YES":
        (directory / "POST_CORE_FAILURE_MECHANISM_AUDIT_002_PLAN.md").write_text(round_two_plan(manifest), encoding="utf-8")
    samples = [row for row in tables["quartile"] if row["proxy"] == "mad" and row["group"] == "ALL"]
    report = ["# POST_CORE_FAILURE_MECHANISM_AUDIT_001", "",
              f'Execution mode **{manifest["mode"]}**. {manifest["rationale"]}', "",
              "| Seed | STABLE − SHIFT W-F1 (pp) | SHIFT − STABLE error (pp) | MAD Q1 error | MAD Q4 error |",
              "| --- | ---: | ---: | ---: | ---: |"]
    for seed, values in per_seed.items():
        q1, q4 = (next(row["error_rate"] for row in samples if row["seed"] == seed and row["quartile"] == q)
                  for q in (1, 4))
        shift = values["shift"]
        report.append(f'| {seed} | {shift["wf1_gap"] * 100:.2f} | {shift["error_gap"] * 100:.2f} '
                      f'| {q1:.2%} | {q4:.2%} |')
    records = list(per_seed.values())
    wf1 = average([record["shift"]["wf1_gap"] for record in records]) * 100
    error = average([record["shift"]["error_gap"] for record in records]) * 100
    low = average([record["ambiguity"]["q1_error"] for record in records])
    high = average([record["ambiguity"]["q4_error"] for record in records])
    report += ["", f"Across the five seeds, STABLE leads SHIFT by {wf1:+.2f} pp in W-F1 and SHIFT has "
               f"{error:+.2f} pp higher error. Mean MAD Q1 error is {low:.2%} and Q4 error is {high:.2%}."]
    report += report_pair_errors(tables["quartile"])
    if manifest["mode"] == "frozen_forward":
        report += ["", "| Seed | ECCS W-F1 drop (pp) | EICS W-F1 drop (pp) | History helpful | History harmful "
                   "| MAD residual Q4 − Q1 (pp) | Fused errors recoverable |",
                   "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
        for seed, row in per_seed.items():
            history, context = row["history"], row["context"]
            recovery = f'{row["modality"]["recoverable"]:.2%}' if "modality" in row else "N/A"
            report.append(f'| {seed} | {context["eccs_drop_pp"]:.2f} | {context["eics_drop_pp"]:.2f} '
                          f'| {history["helpful"]:.2%} | {history["harmful"]:.2%} '
                          f'| {row["ambiguity"]["residual_gap"] * 100:.2f} | {recovery} |')
    report += report_train_dev(tables["train_dev"])
    report += ["", "MAD measures proximity of model probabilities to uniform. Train/Dev comparisons "
               "are descriptive and may co-occur with the fixed session split.", "", f"```text\n{status}\n```", ""]
    (directory / "POST_CORE_FAILURE_MECHANISM_AUDIT_001.md").write_text("\n".join(report), encoding="utf-8")
    (directory / RUNBOOK.name).write_bytes(RUNBOOK.read_bytes())
    manifest["outputs_sha256"] = {path.name: sha256_file(path) for path in sorted(directory.iterdir())}
    write_json_atomic(directory / "failure_audit_manifest.json", manifest)
    return status


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-runs", type=Path, required=True)
    parser.add_argument("--batch-id", required=True)
    parser.add_argument("--experiment-date")
    parser.add_argument("--output-root", default="outputs")
    parser.add_argument("--b0-geometry", type=Path)
    parser.add_argument("--archived-dev", action="store_true")
    parser.add_argument("--asset-path", type=Path)
    parser.add_argument("--model-manifest-path", type=Path)
    parser.add_argument("--provenance-manifest-path", type=Path)
    parser.add_argument("--chunk-size", type=int, default=8)
    args = parser.parse_args(argv)
    errors = check_workstream("multidag")
    if errors:
        print("WORKSTREAM_CHECK = FAIL", flush=True)
        parser.error("\n".join(errors))
    print("WORKSTREAM_CHECK = PASS", flush=True)
    assets = (args.asset_path, args.model_manifest_path, args.provenance_manifest_path)
    if args.chunk_size < 1 or (args.archived_dev and any(assets)) or (not args.archived_dev and not all(assets)):
        parser.error("provide either --archived-dev or all three asset paths, and a positive chunk size")
    if not args.archived_dev and (not sys.platform.startswith("linux") or not torch.cuda.is_available()):
        parser.error("formal frozen analysis requires the Ubuntu CUDA workspace")
    paths = resolve_output_paths(output_base=args.output_root, experiment_date=resolve_experiment_date(args.experiment_date),
                                 experiment_group=GROUP, batch_id=args.batch_id)
    output = paths.analysis_root
    archive = output.parent / f"{output.name}.zip"
    if output.exists() or archive.exists():
        raise FileExistsError("existing audit batch cannot be overwritten")
    datasets, data_identity = (None, None) if args.archived_dev else load_train_dev(*assets)[:2]
    sources = sources_from_runs(args.frozen_runs, data_identity)
    output.mkdir(parents=True, exist_ok=False)
    try:
        tables, per_seed = collect(sources, datasets=datasets, device="cuda", chunk_size=args.chunk_size,
                                   geometry=args.b0_geometry)
        result = decision(per_seed, complete=not args.archived_dev)
        manifest = {**HOLD, **git_identity(), **result, "WORKSTREAM_CHECK": "PASS", "FORMAL_TRAIN_RUNS": 0,
                    "TEST_TOUCHED": "NO", "HANDOFF_READY": "NO" if args.archived_dev else "YES",
                    "READY_FOR_NEXT_METHOD_FASTKILL": "YES" if result["NEXT_ROUTE"] in ROUTES else "NO",
                    "mode": "archived_dev" if args.archived_dev else "frozen_forward", "per_seed": per_seed,
                    "data_identity": next(iter(sources.values())).manifest["data_identity"],
                    "sources": [source.identity for source in sources.values()]}
        status = write_reports(output, tables, per_seed, manifest)
        files = [*manifest["outputs_sha256"], "failure_audit_manifest.json"]
        with ZipFile(archive, "x", compression=ZIP_DEFLATED) as bundle:
            for name in files:
                bundle.write(output / name, arcname=name)
        with ZipFile(archive) as bundle:
            if bundle.testzip() is not None or set(bundle.namelist()) != set(files):
                raise ValueError("audit archive failed integrity check")
    except Exception as error:
        write_json_atomic(output / "failure.json", {**HOLD, "status": "INCOMPLETE", "error": str(error)})
        raise
    print(status)
    print(f"AUDIT_ARCHIVE = {archive}\nAUDIT_ARCHIVE_SHA256 = {sha256_file(archive)}")


if __name__ == "__main__":
    main()

"""Analyze authenticated seed100 Dev history interventions without running the model."""

import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
from statistics import mean
from zipfile import Path as ZipPath, ZipFile

import numpy as np
from sklearn.metrics import roc_auc_score

from scripts.analysis.models.multidag_cl.failure_audit_inference import HISTORY, RECENT, sources_from_runs
from scripts.analysis.models.multidag_cl.history_fastkill_provenance import (
    RECOVERY_SHA256, UH_CSV, csv_rows, read_recovery,
)
from scripts.analysis.models.multidag_cl.pairwise_preflight_inputs import FrozenSplit, validate_f0_dev, validate_hold
from scripts.analysis.models.multidag_cl.run_history_top25_recovery import HISTORY_CSV
from scripts.runtime.multidag_cl_ta_v0.formal import json_identity, sha256_file


def read_archive(archive, source):
    if sha256_file(archive) != RECOVERY_SHA256:
        raise ValueError("history ZIP differs from the authenticated seed100 archive")
    with ZipFile(archive) as zipped:
        membership = csv_rows(zipped.read(UH_CSV))
        trace = csv_rows(zipped.read(HISTORY_CSV))
        recovery = read_recovery(ZipPath(zipped), archive)
        recovery.verify_source(source, recovery.manifest["train_dev_projection"],
                               recovery.manifest["dev_population"])
        return recovery.manifest, membership, trace


def compare(full, restricted, gold):
    if full == restricted:
        return "PREDICTION_UNCHANGED"
    if full == gold:
        return "FULL_BETTER"
    if restricted == gold:
        return "RESTRICTED_BETTER"
    return "CHANGED_BOTH_WRONG"


def keeps(condition, distance, same_speaker):
    return {
        "CURRENT_ONLY": False,
        "SAME_SPEAKER_ONLY": same_speaker,
        "OTHER_SPEAKER_ONLY": not same_speaker,
        "RECENT_ONLY": distance <= RECENT,
        "DISTANT_ONLY": distance > RECENT,
    }[condition]


def load_h0(path, source):
    manifest = json.loads(path.with_name("CORRECTION_DIRECTION_MANIFEST.json").read_text(encoding="utf-8"))
    if (manifest["checkpoint"] != source.identity or manifest["TEST_TOUCHED"] != "NO"
            or manifest["dev_state_file_sha256"] != sha256_file(path)):
        raise ValueError("frozen state export does not match its correction diagnostic manifest")
    with np.load(path, allow_pickle=False) as arrays:
        id_columns = (arrays["utterance_id"], arrays["dialogue_id"], arrays["gold_label_id"])
        size = len(source.population)
        if any(column.shape != (size,) for column in id_columns):
            raise ValueError("h0 export has an invalid seed100 Dev population shape")
        ids = tuple(zip(*id_columns))
        states = arrays["h0"]
        validate_f0_dev(source, FrozenSplit(arrays["logits"], ids))
        if (arrays["checkpoint_sha256"].item() != source.identity["checkpoint_sha256"]
                or states.shape != (size, 300) or not np.isfinite(states).all()):
            raise ValueError("h0 export does not match the full frozen seed100 Dev population and logits")
    states = np.asarray(states, dtype=np.float64)
    norm = np.maximum(np.linalg.norm(states, axis=1, keepdims=True), 1e-8)
    return dict(zip((row[0] for row in ids), states / norm))


def analyze(membership, trace, h0=None):
    history = {(row["utterance_id"], row["condition"]): row for row in trace}
    prefixes = defaultdict(list)
    records = []
    for row in membership:
        uid, did = row["utterance_id"], row["dialogue_id"]
        gold, full = row["gold"], row["full_prediction"]
        prefix = prefixes[did]
        full_row = history[uid, HISTORY[0]]
        if len(prefix) != int(row["target_index"]):
            raise ValueError("history metadata order differs from the canonical dialogue prefix")
        for condition in HISTORY[1:]:
            restricted = history[uid, condition]
            pred = restricted["prediction"]
            removed = [(distance, prior) for distance, prior in enumerate(reversed(prefix), start=1)
                       if not keeps(condition, distance, prior["speaker"] == row["speaker"])]
            same_count = sum(prior["speaker"] == row["speaker"] for _, prior in removed)
            cosines = ([float(h0[uid] @ h0[prior["utterance_id"]]) for _, prior in removed]
                       if h0 is not None else [])
            records.append({
                "utterance_id": uid, "dialogue_id": did, "condition": condition, "gold": gold,
                "full_prediction": full, "restricted_prediction": pred,
                "outcome": compare(full, pred, gold), "shift_group": row["shift_group"],
                "speaker_transition": ("NO_PREVIOUS" if not prefix else "SAME" if
                                       prefix[-1]["speaker"] == row["speaker"] else "DIFFERENT"),
                "previous_same_speaker_label": row["previous_same_speaker_label"],
                "removed_target_edges": len(removed), "removed_same_speaker": same_count,
                "removed_other_speaker": len(removed) - same_count,
                "removed_recent": sum(distance <= RECENT for distance, _ in removed),
                "removed_mean_relative_distance": mean(distance for distance, _ in removed) if removed else None,
                "removed_nearest_distance": removed[0][0] if removed else None,
                "gold_label_match_count": sum(prior["gold"] == gold for _, prior in removed),
                "removed_mean_h0_cosine": mean(cosines) if cosines else None,
                "delta_true_class_logit_margin": float(full_row["true_class_logit_margin"]) -
                float(restricted["true_class_logit_margin"]),
            })
        prefix.append(row)
    return records


def summarize(records):
    groups = defaultdict(list)
    for row in records:
        groups[(row["condition"], "ALL")].append(row)
        groups[(row["condition"], row["shift_group"])].append(row)
        groups[(row["condition"], "PREVIOUS_SPEAKER_" + row["speaker_transition"])].append(row)
        groups[(row["condition"], "REMOVED_RECENT" if row["removed_recent"] else "NO_REMOVED_RECENT")].append(row)
        groups[(row["condition"], "FULL_PREDICTION_" + row["full_prediction"])].append(row)
    output = []
    for (condition, group), rows in sorted(groups.items()):
        counts = Counter(row["outcome"] for row in rows)
        flipped = len(rows) - counts["PREDICTION_UNCHANGED"]
        output.append({
            "condition": condition, "group": group, "support": len(rows),
            "full_better": counts["FULL_BETTER"], "restricted_better": counts["RESTRICTED_BETTER"],
            "unchanged": counts["PREDICTION_UNCHANGED"], "changed_both_wrong": counts["CHANGED_BOTH_WRONG"],
            "prediction_changed": flipped, "change_rate": flipped / len(rows),
            "full_advantage_pp": 100 * (counts["FULL_BETTER"] - counts["RESTRICTED_BETTER"]) / len(rows),
            "mean_delta_true_class_logit_margin": mean(row["delta_true_class_logit_margin"] for row in rows),
        })
    return output


def cosine_summary(records):
    """Compare directions only on contrasts where exactly one prediction is correct."""
    output = []
    for condition in HISTORY[1:]:
        comparable = [row for row in records if row["condition"] == condition
                      and row["outcome"] in {"FULL_BETTER", "RESTRICTED_BETTER"}
                      and row["removed_mean_h0_cosine"] is not None]
        helpful = [row["outcome"] == "FULL_BETTER" for row in comparable]
        cosine = [row["removed_mean_h0_cosine"] for row in comparable]
        output.append({"condition": condition, "support": len(helpful), "full_better": sum(helpful),
                       "restricted_better": len(helpful) - sum(helpful),
                       "auroc_higher_cosine_predicts_full_better": (
                           roc_auc_score(helpful, cosine) if len(set(helpful)) == 2 else None)})
    return output


def transition_map(path, membership, *, source_prediction, candidate_prediction):
    if path is None:
        return {}
    expected = {row["utterance_id"]: row for row in membership}
    observed = {}
    with path.open(newline="", encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream):
            uid = row["utterance_id"]
            baseline = expected.get(uid)
            candidate = row[candidate_prediction]
            if (uid in observed or baseline is None or row.get("split", "dev") != "dev"
                    or any(row[key] != baseline[key] for key in ("dialogue_id", "gold"))
                    or row[source_prediction] != baseline["full_prediction"]
                    or candidate not in {"hap", "sad", "neu", "ang", "exc", "fru"}):
                raise ValueError("candidate transition table does not align with frozen Dev")
            observed[uid] = {
                (True, True): "STABLE_CORRECT", (True, False): "DAMAGED",
                (False, True): "RECOVERED", (False, False): "STABLE_WRONG",
            }[(row[source_prediction] == row["gold"], candidate == row["gold"])]
    if set(observed) != set(expected):
        raise ValueError("candidate transition table is missing Dev samples")
    return observed


def verify_transition_source(path, source, schema):
    if path is None:
        return
    manifest = json.loads(path.with_name("run_manifest.json").read_text(encoding="utf-8"))
    validate_hold(manifest)
    if (manifest["status"] != "completed" or manifest["schema"] != schema
            or manifest["f0"] != source.identity
            or manifest["outputs_sha256"]["DEV_SAMPLE_TRANSITIONS.csv"] != sha256_file(path)):
        raise ValueError("candidate transition table does not match its frozen F0 run evidence")


def write_csv(path, records):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-runs", required=True, type=Path)
    parser.add_argument("--recovery-zip", required=True, type=Path)
    parser.add_argument("--dev-frozen-states", type=Path)
    parser.add_argument("--p-transitions", type=Path)
    parser.add_argument("--h1-transitions", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    source = sources_from_runs(args.frozen_runs, seeds=(100,))[100]
    manifest, membership, trace = read_archive(args.recovery_zip, source)
    verify_transition_source(args.p_transitions, source, "ct_multidag_pairwise_fastkill_v1")
    verify_transition_source(args.h1_transitions, source, "ct_multidag_history_fastkill_v1")
    h0 = load_h0(args.dev_frozen_states, source) if args.dev_frozen_states else None
    records = analyze(membership, trace, h0)
    summary = summarize(records)
    p = transition_map(args.p_transitions, membership, source_prediction="f0_top1", candidate_prediction="primary_prediction")
    h1 = transition_map(args.h1_transitions, membership, source_prediction="f0_prediction", candidate_prediction="h1_prediction")
    enrichment = [dict(utterance_id=row["utterance_id"], dialogue_id=row["dialogue_id"], gold=row["gold"],
                       F0_prediction=row["full_prediction"], U_TOP25=row["is_u_top25"], H_TOP25=row["is_h_top25"],
                       history_sensitivity=row["history_change_count"], P_transition=p.get(row["utterance_id"], ""),
                       H1_transition=h1.get(row["utterance_id"], ""))
                  for row in membership]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    products = {"HISTORY_RELEVANCE_DIAGNOSTICS.csv": summary, "HISTORY_CONTRASTS_PER_SAMPLE.csv": records,
                "HISTORY_ENRICHMENT.csv": enrichment}
    if h0 is not None:
        products["HISTORY_COSINE_SEPARABILITY.csv"] = cosine_summary(records)
    for name, data in products.items():
        write_csv(args.output_dir / name, data)
    provenance = {"TEST_TOUCHED": "NO", "FORMAL_TRAIN_RUNS": 0, "split": "dev", "seed": 100,
                  "f0_checkpoint": source.identity, "dev_population_sha256": json_identity(source.population),
                  "recovery_sha256": RECOVERY_SHA256, "recovery_source_head": manifest["git_head"],
                  "contrasts": len(records), "cosine": "MEASURED" if h0 is not None else "MISSING_H0",
                  "H1_PROXY_FAILURE": "UNRESOLVED",
                  "optional_inputs_sha256": {name: sha256_file(path) for name, path in
                                             (("dev_frozen_states", args.dev_frozen_states),
                                              ("p_transitions", args.p_transitions), ("h1_transitions", args.h1_transitions))
                                             if path is not None},
                  "outputs_sha256": {name: sha256_file(args.output_dir / name) for name in products}}
    (args.output_dir / "HISTORY_AUDIT_MANIFEST.json").write_text(json.dumps(provenance, indent=2) + "\n",
                                                                  encoding="utf-8")
    print(f"DEV_HISTORY_CONTRASTS = {len(records)}\nTEST_TOUCHED = NO")


if __name__ == "__main__":
    main()

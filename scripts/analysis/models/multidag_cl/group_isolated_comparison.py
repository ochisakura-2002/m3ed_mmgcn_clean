"""Aggregate and pair completed Protocol B endpoint exports without loading features."""

import argparse
import csv
import gzip
from pathlib import Path
import re
from tempfile import TemporaryDirectory

import numpy as np

from models.multidag_cl.ta_v0 import LABEL_VOCAB
from scripts.analysis.models.multidag_cl.failure_audit_metrics import PAIRS
from scripts.analysis.models.multidag_cl.pairwise_fastkill_metrics import bootstrap, pair_confusions, transitions
from scripts.analysis.models.multidag_cl.post_ph.internal_protocol import write_rows
from scripts.analysis.models.multidag_cl.post_ph.metrics import reliability
from scripts.models.multidag_cl.group_isolated import ROOT, run_identity, steps_per_epoch
from scripts.runtime.multidag_cl_group_isolated.evaluation import export_role, hard_label_scores, summarize
from scripts.runtime.multidag_cl_group_isolated.protocol import (
    ENDPOINT, G1_GROUP, SESSIONS, describe, fold_config, read_frozen, read_json, validate_hold,
)
from scripts.runtime.multidag_cl_ta_v0.formal import sha256_file, write_json_atomic
from scripts.runtime.multidag_cl_ta_v0.training import classification_report


def parse_prediction(row, *, trajectory=False):
    return {
        "utterance_id": row["utterance_id"], "dialogue_id": row["dialogue_id"],
        "true_label": int(row["true_label" if trajectory else "true_label_id"]),
        "predicted_label": int(row["predicted_label" if trajectory else "predicted_label_id"]),
        "logits": [float(row[f"logit_{label}"]) for label in LABEL_VOCAB],
    }


def population(rows):
    return [(row["utterance_id"], row["dialogue_id"], row["true_label"]) for row in rows]


def read_export(root, config, frozen, raw):
    run = root / "runs" / config["run_id"]
    runs = run / "manifests" / "run_manifest.json"
    manifest = read_json(runs)
    exported = run / "reports" / "evaluation"
    eval_manifest = exported / "evaluation_manifest.json"
    evaluation = read_json(eval_manifest)
    fitted = frozen["folds"][config["held_out_session"]]
    steps = steps_per_epoch(fitted["fit"], raw)
    identity = run_identity(config, raw, manifest["data_identity"])
    for document in (manifest, evaluation):
        validate_hold(document)
        if document["run_identity"] != identity or document["TEST_TOUCHED"] != "NO":
            raise ValueError("paired comparison requires exact arm and fold identities")
    if (manifest["f0_recipe"] != raw or manifest["status"] != "completed" or manifest["synthetic_only"]
            or manifest["seed"] != config["seed"] or manifest["steps_per_epoch"] != steps
            or manifest["optimizer_steps"] != ENDPOINT * steps or manifest["last_completed_epoch"] != ENDPOINT
            or evaluation["endpoint_epoch"] != ENDPOINT or evaluation["checkpoint_selection"] != "fixed_epoch"
            or evaluation["endpoint_sha256"] != manifest["checkpoint_sha256"][str(ENDPOINT)]
            or evaluation["optimizer_steps_performed"] != 0 or evaluation["calibration"] is not None):
        raise ValueError("paired comparison requires zero-update replay of both fixed epoch-30 arms")

    digest = {}
    for name in ("held_out_epoch30_predictions.csv", "trajectory_predictions.csv.gz"):
        digest[name] = sha256_file(exported / name)
        if digest[name] != evaluation["files_sha256"][name]:
            raise ValueError(f"evaluation export SHA-256 mismatch: {name}")
    updates = run / "logs" / "fit_updates.csv"
    with updates.open(encoding="utf-8", newline="") as stream:
        tape = [(int(row["epoch"]), int(row["optimizer_steps"]), row["batch_tape_sha256"])
                for row in csv.DictReader(stream)]
    if [(epoch, step) for epoch, step, _ in tape] != [(e, e * steps) for e in range(1, ENDPOINT + 1)]:
        raise ValueError("fit log does not cover the full frozen step schedule")
    with (exported / "held_out_epoch30_predictions.csv").open(encoding="utf-8", newline="") as stream:
        rows = [parse_prediction(row) for row in csv.DictReader(stream)]
    if describe(population(rows)) != fitted["held_out"]:
        raise ValueError("endpoint rows differ from frozen held-out IDs/order/gold")
    curves = {epoch: [] for epoch in range(1, ENDPOINT)}
    with gzip.open(exported / "trajectory_predictions.csv.gz", "rt", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            if row["population"] == "held_out":
                curves[int(row["epoch"])].append(parse_prediction(row, trajectory=True))
    if any(population(epoch_rows) != population(rows) for epoch_rows in curves.values()):
        raise ValueError("trajectory differs from frozen held-out IDs/order/gold")
    curves[ENDPOINT] = rows
    evidence = {"run_manifest_sha256": sha256_file(runs), "evaluation_manifest_sha256": sha256_file(eval_manifest),
                "fit_updates_sha256": sha256_file(updates), "prediction_files_sha256": digest}
    return rows, curves, manifest, tape, evidence


def curve_row(arm, epoch, rows):
    labels = np.array([row["true_label"] for row in rows])
    guesses = np.array([row["predicted_label"] for row in rows])
    logits = np.array([row["logits"] for row in rows], dtype=np.float64)
    _, scores = hard_label_scores(logits, labels)
    if not np.array_equal(logits.argmax(1), guesses):
        raise ValueError("trajectory predictions differ from logits")
    nll = float(scores["nll"].mean())
    metrics = classification_report(labels, guesses, loss=nll)["metrics"]
    metrics.pop("loss")
    ece, _ = reliability(scores["confidence"], labels == guesses, 10)
    return {"arm": arm, "epoch": epoch, **metrics, "hard_label_nll": nll,
            "brier": float(scores["brier"].mean()), "ece10": ece}


def paired_report(ctrl, g1):
    expected = population(ctrl)
    by_id = {row["utterance_id"]: row for row in g1}
    if len(by_id) != len(g1) or set(by_id) != {uid for uid, _, _ in expected} or len(by_id) != len(ctrl):
        raise ValueError("paired endpoint requires a one-to-one utterance ID join")
    aligned = [by_id[uid] for uid, _, _ in expected]
    before, metadata, *_ = summarize(ctrl, expected)
    after, *_ = summarize(aligned, expected)
    for arm in (before, after):
        confidence = arm["auxiliary"]["high_confidence"]
        confidence["error_rate"] = confidence["error_support"] / confidence["support"] if confidence["support"] else None
        confidence["error_share_all"] = confidence["error_support"] / len(expected)
    gold = np.array([target for _, _, target in expected])
    a, b = (np.array([row["predicted_label"] for row in rows]) for rows in (ctrl, aligned))
    groups = np.array([row["shift_group"] for row in metadata])
    pair_rates = [row for row in pair_confusions(gold, a, b)
                  if frozenset(row["pair"].split("/")) in {frozenset(pair) for pair in PAIRS}]
    details = [
        {"utterance_id": uid, "dialogue_id": did, "true_label_id": target, "group": group,
         "ctrl_predicted_label_id": int(old), "g1_predicted_label_id": int(new),
         "recovered": int(old != target and new == target), "damaged": int(old == target and new != target)}
        for (uid, did, target), group, old, new in zip(expected, groups, a, b)
    ]
    return {
        "ctrl": before, "g1": after,
        "delta_pp": {metric: 100 * (after["metrics"][metric] - before["metrics"][metric])
                     for metric in ("accuracy", "weighted_f1", "macro_f1", "uar")},
        "paired": {group: transitions(a != gold, b != gold, selected)
                   for group, selected in (("ALL", np.ones_like(gold, dtype=bool)),
                                           *((group, groups == group) for group in ("SHIFT", "STABLE", "NO_HISTORY")))},
        "pair_confusions": pair_rates,
        "paired_dialogue_bootstrap": bootstrap(gold, a, b, [did for _, did, _ in expected],
                                              strata=[did[:5] for _, did, _ in expected]),
    }, details


def compare(ctrl_root, g1_root, batch_id):
    if g1_root.name != G1_GROUP or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", batch_id):
        raise ValueError("expected the canonical G1 experiment root and a safe batch ID")
    dest = g1_root / "analysis" / "batches" / batch_id
    if dest.exists() or dest.is_symlink():
        raise FileExistsError(dest)
    frozen, raw = read_frozen(ROOT)
    pooled = {arm: {epoch: [] for epoch in range(1, ENDPOINT + 1)} for arm in ("ctrl", "g1")}
    summary, evidence = {}, {}
    for session in SESSIONS:
        a, a_curve, a_manifest, a_tape, a_evidence = read_export(ctrl_root, fold_config(session), frozen, raw)
        b, b_curve, b_manifest, b_tape, b_evidence = read_export(g1_root, fold_config(session, g1=True), frozen, raw)
        if (a_manifest["data_identity"] != b_manifest["data_identity"] or a_tape != b_tape
                or a_manifest["initialization"] != b_manifest["initialization"]):
            raise ValueError(f"paired fit identity differs for {session}")
        summary[session], _ = paired_report(a, b)
        evidence[session] = {"ctrl": a_evidence, "g1": b_evidence}
        for arm, curves in (("ctrl", a_curve), ("g1", b_curve)):
            for epoch, rows in curves.items():
                pooled[arm][epoch].extend(rows)
    a, b = pooled["ctrl"][ENDPOINT], pooled["g1"][ENDPOINT]
    summary["pooled_oof"], details = paired_report(a, b)
    curves = [curve_row(arm, epoch, rows) for arm, epochs in pooled.items() for epoch, rows in epochs.items()]
    dest.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".g1_comparison_", dir=dest.parent) as temporary:
        stage = Path(temporary)
        for arm, rows in (("ctrl_pooled_oof", a), ("g1_pooled_oof", b)):
            export_role(stage, arm, rows, population(rows))
        write_rows(stage / "paired_utterances.csv", details, tuple(details[0]))
        write_rows(stage / "pooled_oof_epoch_trends.csv", curves, tuple(curves[0]))
        write_json_atomic(stage / "paired_summary.json", summary)
        write_json_atomic(stage / "comparison_manifest.json", {
            "endpoint_epoch": ENDPOINT, "test_touched": "NO", "fit_identity": "DATA_INIT_BATCH_TAPE_MATCH",
            "historical_source_provenance": "REQUIRES_SEPARATE_REVIEW", "input_sha256": evidence,
            "output_sha256": {path.name: sha256_file(path) for path in sorted(stage.iterdir())},
        })
        stage.rename(dest)
    return dest


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--ctrl-root", type=Path, required=True)
    cli.add_argument("--g1-root", type=Path, required=True)
    cli.add_argument("--batch-id", required=True)
    args = cli.parse_args()
    print(compare(args.ctrl_root, args.g1_root, args.batch_id))


if __name__ == "__main__":
    main()

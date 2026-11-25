"""Authenticate the fixed G4 controls and report their paired factorial contrasts."""

import csv
from dataclasses import dataclass
from hashlib import sha256
from io import StringIO
import json
from pathlib import Path
import re

import numpy as np
import yaml

from models.multidag_cl.ta_v0 import LABEL_VOCAB
from scripts.analysis.models.multidag_cl.group_isolated_comparison import paired_report, parse_prediction, population
from scripts.analysis.models.multidag_cl.pairwise_fastkill_metrics import BOOTSTRAP_RESAMPLES, BOOTSTRAP_SEED
from scripts.analysis.models.multidag_cl.post_ph.internal_protocol import write_rows
from scripts.models.multidag_cl.group_isolated import ROOT
from scripts.models.multidag_cl.group_isolated_tail_average import (
    AVG, CE30, LS30, RELIABILITY, export_scoring, g2_run_id, read_average,
)
from scripts.runtime.multidag_cl_group_isolated.evaluation import ROLES, summarize
from scripts.runtime.multidag_cl_group_isolated.fixed_tail_average import AVERAGING_EPOCHS, authenticated_run
from scripts.runtime.multidag_cl_group_isolated.protocol import (
    CONFIG_DIR, ENDPOINT, G1_CONFIG_DIR, G1_EPSILON, G1_GROUP, G1_SCHEMA, SEED, SESSIONS,
    describe, fold_config, load_config, validate_hold,
)
from scripts.runtime.multidag_cl_ta_v0.formal import sha256_file


GROUP = "g4_fixed_ls_tail_average_ct_multidag"
SCHEMA = "ct_multidag_g4_fixed_ls_tail_v1"
CONFIG = G1_CONFIG_DIR.with_name("group_isolated_fixed_ls_average.yaml")
LS_AVG = "ls_avg"
ARMS = (CE30, AVG, LS30, LS_AVG)
CONTRASTS = ((LS30, LS_AVG), (CE30, AVG), (CE30, LS30), (AVG, LS_AVG))
PRIMARY = f"{LS_AVG}_vs_{LS30}"
CLASSIFICATION = ("accuracy", "weighted_f1", "macro_f1", "uar")
BOOTSTRAP = {"resamples": BOOTSTRAP_RESAMPLES, "seed": BOOTSTRAP_SEED, "strata": "session",
             "unit": "dialogue", "metrics": list(CLASSIFICATION)}
UNPINNED = object()


def recipe():
    config = yaml.safe_load((ROOT / CONFIG).read_text(encoding="utf-8"))
    expected = {"schema": SCHEMA, "source_group": G1_GROUP, "source_schema": G1_SCHEMA,
                "fit_objective": {"name": "uniform_label_smoothed_ce", "label_smoothing": G1_EPSILON},
                "seed": SEED, "averaging_epochs": list(AVERAGING_EPOCHS),
                "output": {"root": "outputs", "experiment_group": GROUP}}
    if config != expected:
        raise ValueError("G4 YAML differs from the fixed LS tail recipe")
    return config


def valid_sha(value):
    return isinstance(value, str) and re.fullmatch(r"[a-f0-9]{64}", value) is not None


def read_csv(path, expected_sha=UNPINNED):
    payload = path.read_bytes()
    digest = sha256(payload).hexdigest()
    if expected_sha is not UNPINNED and digest != expected_sha:
        raise ValueError(f"prediction SHA-256 mismatch: {path}")
    return list(csv.DictReader(StringIO(payload.decode("utf-8"), newline=""))), digest


def read_json_sha(path):
    payload = path.read_bytes()
    return json.loads(payload), sha256(payload).hexdigest()


@dataclass
class Endpoint:
    run: Path
    manifest: dict
    digest: str
    predictions: list
    tape: list
    evidence: dict


def read_endpoint(root, config, frozen, raw):
    run = root / "runs" / config["run_id"]
    if run.resolve() != root.resolve() / "runs" / config["run_id"]:
        raise ValueError(f"source run escapes its canonical path: {run}")
    manifest, digest = authenticated_run(run, config, frozen, raw)
    export = run / "reports" / "evaluation"
    eval_path = export / "evaluation_manifest.json"
    evaluation, eval_sha = read_json_sha(eval_path)
    validate_hold(evaluation)
    expected = {"run_id": config["run_id"], "run_identity": manifest["run_identity"], "endpoint_epoch": ENDPOINT,
                "checkpoint_selection": "fixed_epoch", "endpoint_sha256": manifest["checkpoint_sha256"][str(ENDPOINT)],
                "optimizer_steps_performed": 0, "calibration": None, "scored_populations": list(ROLES),
                "TEST_TOUCHED": "NO"}
    if any(key not in evaluation or evaluation[key] != value for key, value in expected.items()):
        raise ValueError(f"endpoint export does not match fixed E30: {run}")
    filename = "held_out_epoch30_predictions.csv"
    pred_sha = evaluation["files_sha256"][filename]
    if not valid_sha(pred_sha):
        raise ValueError(f"invalid endpoint prediction SHA-256: {run}")
    rows, _ = read_csv(export / filename, pred_sha)
    predictions = [parse_prediction(row) for row in rows]
    if describe(population(predictions)) != manifest["data_identity"]["fold"]["held_out"]:
        raise ValueError(f"endpoint rows differ from frozen held-out IDs/order/gold: {run}")
    updates, tape_sha = read_csv(run / "logs" / "fit_updates.csv")
    tape = [(int(row["epoch"]), int(row["optimizer_steps"]), row["batch_tape_sha256"]) for row in updates]
    steps = manifest["steps_per_epoch"]
    if ([(epoch, step) for epoch, step, _ in tape] != [(e, e * steps) for e in range(1, ENDPOINT + 1)]
            or not all(valid_sha(value) for _, _, value in tape)):
        raise ValueError(f"invalid epoch/update tape: {run}")
    return Endpoint(run, manifest, digest, predictions, tape, {
        "run_manifest_sha256": digest, "evaluation_manifest_sha256": eval_sha,
        "prediction_sha256": pred_sha, "fit_updates_sha256": tape_sha,
    })


def source_configs(session, raw):
    return (ROOT / G1_CONFIG_DIR / f"{session.lower()}_seed100.yaml", ROOT / CONFIG_DIR / "fold_identities.json",
            ROOT / raw["frontend_source"]["config_path"])


def authenticate_ls_source(ce, ls, session, raw, reference):
    config_path, *configs = source_configs(session, raw)
    if load_config(config_path) != fold_config(session, g1=True):
        raise ValueError(f"G1 fold YAML differs: {session}")
    left, right = ce.manifest, ls.manifest
    initialization = right["initialization"]
    rng = right["training_initial_rng"]
    required_rng = {"python", "numpy", "torch_cpu"} | ({"torch_cuda"} if right["device"] == "cuda" else set())
    if (left["initialization"] != initialization or left["data_identity"] != right["data_identity"]
            or ce.tape != ls.tape or population(ce.predictions) != population(ls.predictions)
            or not valid_sha(initialization["shared_initial_state_sha256"])
            or initialization["shared_parameter_max_abs_diff"] != 0 or not required_rng <= rng.keys()
            or not all(valid_sha(value) for value in rng.values())):
        raise ValueError(f"CE/LS fit identity, initialization, RNG or tape differs: {session}")
    provenance = right["runtime_provenance"]
    early, late = provenance["files_sha256"], provenance["late_imports_sha256"]
    head = provenance["git_head"]
    required = {**reference, **{path.relative_to(ROOT).as_posix(): sha256_file(path)
                               for path in (config_path, *configs)}}
    inventory = {**early, **late}
    if (not reference or early.keys() & late.keys() or head != right["git"]["git_head"]
            or not isinstance(head, str) or not re.fullmatch(r"[a-f0-9]{40}", head)
            or any(inventory.get(name) != digest for name, digest in required.items())):
        raise ValueError(f"incomplete or mismatched G1 source inventory: {session}")
    for name, digest in inventory.items():
        path = (ROOT / name).resolve()
        if (not path.is_relative_to(ROOT) or path.relative_to(ROOT).as_posix() != name
                or not valid_sha(digest) or sha256_file(path) != digest):
            raise ValueError(f"historical G1 source differs: {name}")
    return {"git_head": head, "files_sha256": inventory, "initialization": initialization,
            "training_initial_rng": rng, "ce_ls_fit_match": True}


def read_controls(ce_root, ls_root, g2_root, session, frozen, raw, reference):
    ce = read_endpoint(ce_root, fold_config(session), frozen, raw)
    ls = read_endpoint(ls_root, fold_config(session, g1=True), frozen, raw)
    proof = authenticate_ls_source(ce, ls, session, raw, reference)
    expected = population(ce.predictions)
    avg, g2_proof = read_average(g2_root / "runs" / g2_run_id(session), ce.run, ce.manifest,
                                 ce.digest, session, expected)
    arms = {CE30: ce.predictions, AVG: avg, LS30: ls.predictions}
    for rows in arms.values():
        summarize(rows, expected)
    return ls, arms, {CE30: ce.evidence, AVG: g2_proof, LS30: {**ls.evidence, **proof}}


def bootstrap_arms(arms, expected):
    """Draw whole dialogues once per scope and index every arm with the same draws."""
    dialogue_ids = [did for _, did, _ in expected]
    labels = np.array([target for _, _, target in expected])
    unique, inverse = np.unique(dialogue_ids, return_inverse=True)
    strata = np.array([did[:5] for did in unique])
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    groups = [np.flatnonzero(strata == session) for session in SESSIONS if np.any(strata == session)]
    draws = np.concatenate([rng.choice(group, size=(BOOTSTRAP_RESAMPLES, len(group))) for group in groups], axis=1)
    multiplicities = np.stack([(draws == index).sum(axis=1) for index in range(len(unique))], axis=1)
    classes = len(LABEL_VOCAB)
    scores = {}
    for arm in ARMS:
        guesses = np.array([row["predicted_label"] for row in arms[arm]])
        keys = inverse * classes**2 + labels * classes + guesses
        counts = np.bincount(keys, minlength=len(unique) * classes**2).reshape(len(unique), -1)
        matrices = (multiplicities @ counts).reshape(-1, classes, classes)
        support, predicted = matrices.sum(axis=2), matrices.sum(axis=1)
        tp, total = matrices.diagonal(axis1=1, axis2=2), support.sum(axis=1)
        denominator = support + predicted
        f1 = np.divide(2 * tp, denominator, out=np.zeros_like(tp, dtype=float), where=denominator != 0)
        recall = np.divide(tp, support, out=np.zeros_like(tp, dtype=float), where=support != 0)
        scores[arm] = np.array([tp.sum(axis=1) / total, (f1 * support).sum(axis=1) / total,
                                f1.mean(axis=1), recall.mean(axis=1)])
    return scores, sha256(np.asarray(draws, dtype="<i8").tobytes()).hexdigest()


def intervals(delta):
    return {key: {"lower_95_pp": float(lo), "upper_95_pp": float(hi)}
            for key, (lo, hi) in zip(CLASSIFICATION, np.quantile(100 * delta, [0.025, 0.975], axis=1).T)}


def reliability_metrics(summary):
    aux = summary["auxiliary"]
    return {**{key: aux[key] for key in RELIABILITY},
            **{key: aux["error_detection"][key] for key in ("auroc", "auprc")},
            **{key: aux["risk_coverage"][key] for key in ("aurc", "tie_expected_aurc")}}


def difference(after, before):
    return {key: after[key] - value if after[key] is not None and value is not None else None
            for key, value in before.items()}


def report_scope(directory, arms, expected):
    directory.mkdir()
    metrics, risks = {}, {}
    for arm in ARMS:
        metrics[arm], risks[arm] = export_scoring(directory, arm, arms[arm], expected)
    scores, draws_sha = bootstrap_arms(arms, expected)
    comparisons = {}
    for before, after in CONTRASTS:
        report, pairs = paired_report(arms[before], arms[after], include_bootstrap=False)
        del report["ctrl"], report["g1"]
        report["paired_dialogue_bootstrap"] = intervals(scores[after] - scores[before])
        report["reliability_delta"] = difference(reliability_metrics(metrics[after]), reliability_metrics(metrics[before]))
        accepted = metrics[before]["auxiliary"]["high_confidence"]["support"]
        report["risk_at_control_high_confidence_coverage"] = ({
            "accepted": accepted, "coverage": accepted / len(expected),
            before: risks[before][accepted - 1]["tie_expected_risk"],
            after: risks[after][accepted - 1]["tie_expected_risk"],
        } if accepted else None)
        report["per_class_r_d_net"] = {}
        for label_id, label in enumerate(LABEL_VOCAB):
            selected = [row for row in pairs if row["true_label_id"] == label_id]
            recovered, damaged = (sum(row[key] for row in selected) for key in ("recovered", "damaged"))
            report["per_class_r_d_net"][label] = {"support": len(selected), "r": recovered,
                                                  "d": damaged, "net": recovered - damaged}
        for row in pairs:
            row["control_predicted_label_id"] = row.pop("ctrl_predicted_label_id")
            row["candidate_predicted_label_id"] = row.pop("g1_predicted_label_id")
        key = f"{after}_vs_{before}"
        write_rows(directory / f"{key}_paired_utterances.csv", pairs, tuple(pairs[0]))
        comparisons[key] = report
    ls, ce = comparisons[PRIMARY], comparisons[f"{AVG}_vs_{CE30}"]
    return {"arms": metrics, "comparisons": comparisons, "bootstrap_draws_sha256": draws_sha,
            "interaction": {"contrast": "(LS AVG - LS E30) - (CE AVG - CE E30)",
                            "delta_pp": difference(ls["delta_pp"], ce["delta_pp"]),
                            "reliability_delta": difference(ls["reliability_delta"], ce["reliability_delta"]),
                            "paired_dialogue_bootstrap": intervals((scores[LS_AVG] - scores[LS30])
                                                                     - (scores[AVG] - scores[CE30]))}}


def decision(summary):
    primary = summary["pooled_oof"]["comparisons"][PRIMARY]
    deltas = primary["delta_pp"]
    folds = {s: summary[s]["comparisons"][PRIMARY]["delta_pp"]["weighted_f1"] for s in SESSIONS}
    gates = {"pooled_wf1_at_least_0_50pp": deltas["weighted_f1"] >= 0.50 - 1e-10,
             "at_least_three_positive_folds": sum(delta > 0 for delta in folds.values()) >= 3,
             "no_fold_below_minus_1pp": min(folds.values()) >= -1.00 - 1e-10,
             "macro_f1_at_least_minus_0_30pp": deltas["macro_f1"] >= -0.30 - 1e-10,
             "uar_at_least_minus_0_30pp": deltas["uar"] >= -0.30 - 1e-10}
    reliability = primary["reliability_delta"]
    return {"status": "DEVELOPMENT_GO" if all(gates.values()) else "NO_CLASSIFICATION_GO",
            "primary": PRIMARY, "classification_gates": gates, "fold_wf1_delta_pp": folds,
            "pooled_classification_delta_pp": deltas, "reliability_delta": reliability,
            "reliability_review_signal": reliability["nll"] <= -0.03 + 1e-10,
            "reliability_review_nll_threshold": -0.03}

"""Full-coverage metrics and auxiliary reliability from fixed-epoch G0 logits."""

import csv
import gzip
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

from datasets.iemocap.tav_full_audit_dataset import collate
from models.multidag_cl.ta_v0 import LABEL_VOCAB
from scripts.analysis.models.multidag_cl.failure_audit_metrics import metadata, logit_probabilities
from scripts.analysis.models.multidag_cl.post_ph.internal_protocol import write_rows
from scripts.analysis.models.multidag_cl.post_ph.metrics import (
    error_detection, probability_details, reliability, selective_curve,
)
from scripts.runtime.multidag_cl_ta_v0.formal import sha256_file, write_json_atomic
from scripts.runtime.multidag_cl_ta_v0.training import classification_report
from .protocol import ENDPOINT, FoldView, HOLD


HIGH_CONFIDENCE = 0.90
ROLES = ("fit", "held_out")
TRAJECTORY_FIELDS = ("epoch", "population", "utterance_id", "dialogue_id", "true_label", "predicted_label",
                     *(f"logit_{label}" for label in LABEL_VOCAB))


def checkpoint_path(run_dir, epoch):
    if epoch == ENDPOINT:
        return run_dir / "checkpoints" / "epoch30_model.pt"
    return run_dir / "checkpoints" / "trajectory" / f"epoch{epoch:02d}.pt"


def validate_predictions(predictions, expected):
    actual = [(row["utterance_id"], row["dialogue_id"], row["true_label"]) for row in predictions]
    if actual != expected:
        raise ValueError("scored population does not match the frozen G0 order, IDs and labels")
    return actual


def summarize(predictions, expected):
    actual = validate_predictions(predictions, expected)
    labels = np.asarray([row[2] for row in actual])
    logits = np.asarray([row["logits"] for row in predictions], dtype=np.float64)
    probabilities = logit_probabilities(logits)
    top1 = logits.argmax(axis=1)
    if not np.array_equal(top1, [row["predicted_label"] for row in predictions]):
        raise ValueError("exported predictions differ from the logits")
    scores = probability_details(probabilities, labels)
    # Stable log-sum-exp yields exact NLL even when a softmax entry underflows.
    scores["nll"] = np.logaddexp.reduce(logits, axis=1) - logits[np.arange(len(labels)), labels]
    confidence = scores["confidence"]
    wrong = top1 != labels
    ece, bins = reliability(confidence, ~wrong, 10)
    detection, detection_curve = error_detection(wrong, 1 - confidence)
    selective, risk_curve, _ = selective_curve(labels, top1, 1 - confidence, len(LABEL_VOCAB))
    classification = classification_report(labels, top1, loss=float(scores["nll"].mean()))
    summary = {**classification, "auxiliary": {
        "nll": float(scores["nll"].mean()), "brier": float(scores["brier"].mean()), "ece10": ece,
        "error_detection": {**detection, "risk_score": "1 - max softmax", "error_support": int(wrong.sum())},
        "risk_coverage": selective,
        "high_confidence": {"threshold": HIGH_CONFIDENCE,
                            "support": int((confidence >= HIGH_CONFIDENCE).sum()),
                            "error_support": int((wrong & (confidence >= HIGH_CONFIDENCE)).sum())},
    }}
    classified = metadata(actual)
    for group in ("SHIFT", "STABLE", "NO_HISTORY"):
        mask = np.array([row["shift_group"] == group for row in classified])
        if mask.any():
            subgroup_ece, _ = reliability(confidence[mask], ~wrong[mask], 10)
            summary.setdefault("shift_groups", {})[group] = {
                **classification_report(labels[mask], top1[mask], loss=float(scores["nll"][mask].mean())),
                "brier": float(scores["brier"][mask].mean()), "ece10": subgroup_ece,
            }
        else:
            summary.setdefault("shift_groups", {})[group] = {"metrics": {"valid_target_count": 0}}
    rows = []
    for idx, sample in enumerate(classified):
        did = sample["dialogue_id"]
        rows.append({
            **sample, "session": did[:5], "actor": did[:5] + sample["speaker"],
            "true_label_id": int(labels[idx]), "predicted_label_id": int(top1[idx]),
            "predicted_label": LABEL_VOCAB[top1[idx]], "is_error": bool(wrong[idx]),
            "confidence": float(confidence[idx]), "nll": float(scores["nll"][idx]),
            "brier": float(scores["brier"][idx]),
            **{f"logit_{name}": float(value) for name, value in zip(LABEL_VOCAB, logits[idx])},
            **{f"prob_{name}": float(value) for name, value in zip(LABEL_VOCAB, probabilities[idx])},
        })
    return summary, rows, bins, detection_curve, risk_curve


def export_role(directory, role, predictions, expected):
    summary, rows, bins, detection, risk = summarize(predictions, expected)
    write_json_atomic(directory / f"{role}_epoch30_metrics.json", summary)
    write_rows(directory / f"{role}_epoch30_predictions.csv", rows, tuple(rows[0]))
    write_rows(directory / f"{role}_high_confidence_errors.csv",
               (r for r in rows if r["is_error"] and r["confidence"] >= HIGH_CONFIDENCE), tuple(rows[0]))
    write_rows(directory / f"{role}_ece10_bins.csv", bins, tuple(bins[0]))
    write_rows(directory / f"{role}_error_detection.csv", detection,
               ("curve", "fpr", "tpr", "precision", "recall", "threshold"))
    write_rows(directory / f"{role}_risk_coverage.csv", risk, tuple(risk[0]))


def replay(run_dir, runtime, views, entries, manifest):
    destination = run_dir / "reports" / "evaluation"
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    destination.parent.mkdir(exist_ok=True)
    fit_batches = torch.utils.data.DataLoader(
        FoldView(views["fit"].items, "dev"), batch_size=manifest["f0_recipe"]["training"]["batch_size"],
        shuffle=False, num_workers=0, collate_fn=collate, generator=torch.Generator().manual_seed(manifest["seed"]),
    )
    with TemporaryDirectory(prefix=".g0_evaluation_", dir=destination.parent) as temporary:
        stage = Path(temporary)
        trends = []
        with gzip.open(stage / "trajectory_predictions.csv.gz", "wt", encoding="utf-8", newline="") as stream:
            trajectory = csv.writer(stream)
            trajectory.writerow(TRAJECTORY_FIELDS)
            for epoch in range(1, ENDPOINT + 1):
                payload = checkpoint_path(run_dir, epoch).read_bytes()
                if sha256(payload).hexdigest() != manifest["checkpoint_sha256"][str(epoch)]:
                    raise ValueError(f"G0 checkpoint SHA-256 mismatch at epoch {epoch}")
                checkpoint = torch.load(BytesIO(payload), map_location="cpu", weights_only=True)
                if (checkpoint["identity"] != manifest["run_identity"] or checkpoint["epoch"] != epoch
                        or checkpoint["optimizer_steps"] != epoch * manifest["steps_per_epoch"]):
                    raise ValueError("G0 checkpoint run, epoch or optimizer-step identity mismatch")
                runtime.model.load_state_dict(checkpoint["model_state_dict"], strict=True)
                for role in ROLES:
                    batches = runtime.loader("dev") if role == "held_out" else fit_batches
                    report = runtime.evaluate(batches, include_predictions=True, prediction_details=True)
                    trends.append({"epoch": epoch, "population": role, **report["metrics"]})
                    if epoch == ENDPOINT:
                        export_role(stage, role, report["predictions"], entries[role])
                    else:
                        validate_predictions(report["predictions"], entries[role])
                        trajectory.writerows(
                            (epoch, role, row["utterance_id"], row["dialogue_id"], row["true_label"],
                             row["predicted_label"], *row["logits"])
                            for row in report["predictions"]
                        )
        if runtime.optimizer_steps != 0:
            raise ValueError("G0 replay performed optimizer steps")
        write_rows(stage / "epoch_trends.csv", trends, tuple(trends[0]))
        write_json_atomic(stage / "evaluation_manifest.json", {
            **HOLD, "TEST_TOUCHED": "NO", "run_id": manifest["run_id"], "run_identity": manifest["run_identity"],
            "endpoint_epoch": ENDPOINT, "endpoint_sha256": manifest["checkpoint_sha256"][str(ENDPOINT)],
            "optimizer_steps_performed": 0, "scored_populations": list(ROLES),
            "checkpoint_selection": "fixed_epoch", "calibration": None,
            "files_sha256": {p.name: sha256_file(p) for p in sorted(stage.iterdir())},
        })
        stage.rename(destination)
    return destination

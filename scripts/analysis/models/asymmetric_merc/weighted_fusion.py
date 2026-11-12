"""Q2-2 Dev analysis. No models, feature files, loaders, or training imports."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import precision_recall_fscore_support

from utils.metrics import compute_classification_metrics, compute_confusion_matrix
from utils.run_metadata import compute_file_sha256


LABEL_NAMES = ("hap", "sad", "neu", "ang", "exc", "fru")
LABEL_IDS = list(range(len(LABEL_NAMES)))
SEEDS = (100, 101, 102, 103, 104)
PROBABILITY_SUM_ATOL = 1e-5
EPSILON = 1e-12
TIE_ATOL = 1e-12
ALPHA_GRID = tuple(index / 100 for index in range(101))
ISOLATION = {"test_access": False, "test_loader_built": 0, "test_evaluation_count": 0}


class AuditError(ValueError):
    """A closed failure whose details can be preserved by the CLI."""

    def __init__(self, message: str, audit: dict[str, Any] | None = None):
        super().__init__(message)
        self.audit = audit or {}


@dataclass(frozen=True)
class Predictions:
    ids: tuple[str, ...]
    true: np.ndarray
    predicted: np.ndarray
    probabilities: np.ndarray
    manifest: dict[str, Any]


@dataclass(frozen=True)
class Joined:
    ids: tuple[str, ...]
    true: np.ndarray
    ta_predicted: np.ndarray
    v_predicted: np.ndarray
    ta_probabilities: np.ndarray
    v_probabilities: np.ndarray
    audit: dict[str, Any]


def load_predictions(path: Path, *, split: str = "dev", score_columns: bool = False) -> Predictions:
    """One six-class reader for frozen probabilities and Q2-2 reference scores.

    Defaults preserve the Dev probability contract. Score columns are only for
    reading the existing Q2-2 fusion artifact, never for base-expert inputs.
    """
    if split not in {"train", "dev"}:
        raise AuditError("Only train/dev prediction splits allowed; Test is forbidden")
    path = Path(path)
    before = compute_file_sha256(path)
    audit: dict[str, Any] = {"path": path.as_posix(), "sha256": before}

    def fail(message: str, **details: Any) -> None:
        raise AuditError(f"{path}: {message}", {**audit, **details})

    def label(value: str, field: str, line: int) -> int:
        try:
            result = int(value)
        except (ValueError, TypeError):
            fail(f"invalid {field} at line {line}")
        if result not in LABEL_IDS:
            fail(f"invalid {field} at line {line}: expected 0..5")
        return result

    rows = []
    seen: set[str] = set()
    duplicates: set[str] = set()
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        first = handle.readline()
        delimiter = "\t" if "\t" in first else ","
        handle.seek(0)
        reader = csv.DictReader(handle, delimiter=delimiter, strict=True)
        fields = reader.fieldnames or []
        if len(fields) != len(set(fields)):
            fail("duplicate column names")
        prefix = "score" if score_columns else "prob"
        required = {"utterance_id", "true_label", *(f"{prefix}_{k}" for k in LABEL_IDS)}
        aliases = [name for name in ("predicted_label", "pred_label") if name in fields]
        if required - set(fields) or not aliases:
            fail("missing required columns", missing_columns=sorted(required - set(fields)))
        for line, row in enumerate(reader, 2):
            if None in row or any(value is None for value in row.values()):
                fail(f"malformed column count at line {line}")
            uid = row["utterance_id"]
            if not uid.strip():
                fail(f"empty utterance_id at line {line}")
            if uid in seen:
                duplicates.add(uid)
            seen.add(uid)
            aliases_for_split = {"train"} if split == "train" else {"dev", "val", "validation"}
            if "split" in fields and row["split"].strip().lower() not in aliases_for_split:
                fail(f"non-{'Dev' if split == 'dev' else 'Train'} split at line {line}")
            target = label(row["true_label"], "true_label", line)
            predictions = [label(row[name], name, line) for name in aliases]
            if len(set(predictions)) != 1:
                fail(f"conflicting prediction aliases at line {line}")
            try:
                probs = np.array([float(row[f"{prefix}_{k}"]) for k in LABEL_IDS])
            except ValueError:
                fail(f"non-numeric probabilities at line {line}")
            if not np.isfinite(probs).all():
                fail(f"non-finite probabilities (NaN/Inf) at line {line}")
            if score_columns:
                # Q2-2 stores weighted log-probability scores. Softmax is only
                # for this reference format; legal base probabilities are untouched.
                probs = np.exp(probs - probs.max())
                probs = probs / probs.sum()
            if (probs < 0).any():
                fail(f"negative probability at line {line}")
            if not np.isclose(probs.sum(), 1, atol=PROBABILITY_SUM_ATOL, rtol=0):
                fail(f"probability sum is not 1 at line {line}")
            if int(probs.argmax()) != predictions[0]:
                fail(f"argmax != stored predicted label at line {line}")
            rows.append((uid, target, predictions[0], probs))
    audit.update(rows=len(rows), duplicate_ids=sorted(duplicates), delimiter=delimiter)
    if duplicates:
        fail("duplicate IDs", duplicate_ids=sorted(duplicates))
    if not rows:
        fail("empty prediction file")
    if compute_file_sha256(path) != before:
        fail("input changed while reading")
    rows.sort(key=lambda row: row[0])
    probabilities = np.stack([row[3] for row in rows])
    audit.update(
        predicted_label_field=aliases,
        split_evidence="validated_column" if "split" in fields else f"caller_declared_{split}",
        probability_clipping_count=int((probabilities < EPSILON).sum()),
    )
    return Predictions(
        tuple(row[0] for row in rows),
        np.array([row[1] for row in rows], dtype=np.int64),
        np.array([row[2] for row in rows], dtype=np.int64),
        probabilities,
        audit,
    )


def exact_join(ta: Predictions, visual: Predictions) -> Joined:
    """Sorted exact-ID inner join; labels must agree before any analysis."""
    ta_index = {uid: index for index, uid in enumerate(ta.ids)}
    v_index = {uid: index for index, uid in enumerate(visual.ids)}
    ids = tuple(sorted(ta_index.keys() & v_index.keys()))
    ti = np.array([ta_index[uid] for uid in ids], dtype=np.int64)
    vi = np.array([v_index[uid] for uid in ids], dtype=np.int64)
    mismatch = [uid for uid, i, j in zip(ids, ti, vi) if ta.true[i] != visual.true[j]]
    audit = {
        "key": "utterance_id", "join": "inner", "order": "lexicographic_id",
        "ta_rows": len(ta.ids), "v_rows": len(visual.ids), "common_rows": len(ids),
        "ta_only_ids": sorted(ta_index.keys() - v_index.keys()),
        "v_only_ids": sorted(v_index.keys() - ta_index.keys()),
        "duplicate_ids": {"ta": [], "v": []},
        "label_mismatch": len(mismatch), "label_mismatch_ids": mismatch,
        "ta_probability_clipping_count": int((ta.probabilities[ti] < EPSILON).sum()),
        "v_probability_clipping_count": int((visual.probabilities[vi] < EPSILON).sum()),
    }
    if not ids or mismatch:
        raise AuditError("exact join failed: empty overlap or label mismatch", audit)
    return Joined(ids, ta.true[ti], ta.predicted[ti], visual.predicted[vi],
                  ta.probabilities[ti], visual.probabilities[vi], audit)


def metrics(true: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    values = compute_classification_metrics(true, predicted, labels=LABEL_IDS)
    return {"accuracy" if key == "acc" else key: value for key, value in values.items()}


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def complementarity(joined: Joined) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    ta_ok = joined.ta_predicted == joined.true
    v_ok = joined.v_predicted == joined.true

    def counts(mask: np.ndarray) -> dict[str, Any]:
        tc, vc = ta_ok[mask], v_ok[mask]
        q11, q10 = int((tc & vc).sum()), int((tc & ~vc).sum())
        q01, q00 = int((~tc & vc).sum()), int((~tc & ~vc).sum())
        return {
            "support": int(mask.sum()), "q11": q11, "q10": q10,
            "q01": q01, "q00": q00, "ta_correct": q11 + q10,
            "ta_errors": q01 + q00, "v_correct": q11 + q01,
            "rescued": q01, "rescue_rate": _rate(q01, q01 + q00),
            "ta_correct_v_wrong": q10, "both_correct": q11, "both_wrong": q00,
        }

    summary = counts(np.ones(len(joined.ids), dtype=bool))
    summary["disagreement_rate"] = float((joined.ta_predicted != joined.v_predicted).mean())
    summary["oracle_accuracy"] = (summary["q11"] + summary["q10"] + summary["q01"]) / len(joined.ids)
    for prefix, predicted in (("ta", joined.ta_predicted), ("v", joined.v_predicted)):
        summary.update({f"{prefix}_{k}": v for k, v in metrics(joined.true, predicted).items()})
    summary["oracle_gain"] = summary["oracle_accuracy"] - summary["ta_accuracy"]
    classwise = [
        {"class_id": k, "class_name": name, **counts(joined.true == k)}
        for k, name in enumerate(LABEL_NAMES)
    ]
    return summary, classwise


def fusion_scores(joined: Joined, alpha: float) -> np.ndarray:
    if not np.isfinite(alpha) or not 0 <= alpha <= 1:
        raise ValueError("alpha must lie in [0, 1]")
    return (alpha * np.log(np.maximum(joined.ta_probabilities, EPSILON))
            + (1 - alpha) * np.log(np.maximum(joined.v_probabilities, EPSILON)))


def select_grid_row(grid: list[dict[str, float]]) -> dict[str, float]:
    """Compare every candidate to the global maximum (avoid chained ties)."""
    best = max(row["weighted_f1"] for row in grid)
    return max((row for row in grid if best - row["weighted_f1"] <= TIE_ATOL),
               key=lambda row: row["alpha"])


def select_alpha(joined: Joined) -> tuple[dict[str, float], list[dict[str, float]]]:
    grid = [{"alpha": alpha, **metrics(joined.true, fusion_scores(joined, alpha).argmax(axis=1))}
            for alpha in ALPHA_GRID]
    return select_grid_row(grid), grid


def evaluate_fixed_alpha(joined: Joined, alpha: float) -> dict[str, Any]:
    scores = fusion_scores(joined, alpha)
    predictions = scores.argmax(axis=1)
    results = {name: metrics(joined.true, pred) for name, pred in (
        ("ta", joined.ta_predicted), ("v", joined.v_predicted), ("fusion", predictions)
    )}
    results["delta_vs_ta"] = {k: results["fusion"][k] - v for k, v in results["ta"].items()}
    per_class = [{"class_id": k, "class_name": name,
                  "support": int((joined.true == k).sum())}
                 for k, name in enumerate(LABEL_NAMES)]
    matrices = []
    for source, pred in (("ta", joined.ta_predicted), ("v", joined.v_predicted), ("fusion", predictions)):
        precision, recall, f1, _ = precision_recall_fscore_support(
            joined.true, pred, labels=LABEL_IDS, zero_division=0)
        for k, row in enumerate(per_class):
            row.update({f"{source}_precision": float(precision[k]),
                        f"{source}_recall": float(recall[k]), f"{source}_f1": float(f1[k])})
        cm = compute_confusion_matrix(joined.true, pred, labels=LABEL_IDS)
        matrices.extend({"source": source, "true_class_id": k,
                         "true_class_name": LABEL_NAMES[k],
                         **{f"pred_{name}": int(cm[k, j]) for j, name in enumerate(LABEL_NAMES)}}
                        for k in LABEL_IDS)
    for row in per_class:
        row["delta_f1_vs_ta"] = row["fusion_f1"] - row["ta_f1"]
    return {"metrics": results, "predictions": predictions, "scores": scores,
            "per_class": per_class, "confusion_matrices": matrices}


def evaluate_robustness(joins: dict[int, Joined], alpha: float) -> tuple[list[dict], dict[int, dict]]:
    """Evaluate one already selected alpha; never search other visual seeds."""
    rows, evaluations = [], {}
    for seed in SEEDS:
        joined = joins[seed]
        result = evaluate_fixed_alpha(joined, alpha)
        evaluations[seed] = result
        row = {"seed": seed, "alpha": alpha, "common_rows": len(joined.ids)}
        for source, values in result["metrics"].items():
            row.update({f"{source}_{k}": v for k, v in values.items()})
        rows.append(row)
    return rows, evaluations

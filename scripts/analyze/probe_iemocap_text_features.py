"""Run current-utterance-only diagnostics on one IEMOCAP text-feature PKL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import pickle
import re
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.preprocessing import StandardScaler


SESSIONS = tuple(f"Ses0{index}" for index in range(1, 6))
SESSION_PATTERN = re.compile(r"^(Ses0[1-5])[FM]_", re.IGNORECASE)
DEFAULT_LABELS = ("Happy", "Sad", "Neutral", "Angry", "Excited", "Frustrated")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-pkl", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-iter", type=int, default=2000)
    return parser.parse_args()


def load_feature_pkl(path: Path) -> Sequence[Any]:
    with Path(path).open("rb") as file:
        try:
            value = pickle.load(file, encoding="latin1")
        except TypeError:
            file.seek(0)
            value = pickle.load(file)
    if not isinstance(value, (list, tuple)) or len(value) != 9:
        raise ValueError("IEMOCAP feature PKL must have exactly nine items.")
    if not all(isinstance(value[index], Mapping) for index in range(7)):
        raise TypeError("The first seven IEMOCAP PKL items must be mappings.")
    return value


def dialogue_metadata(dialogue_id: Any) -> Tuple[str, str]:
    text = str(dialogue_id)
    match = SESSION_PATTERN.match(text)
    if match is None:
        raise ValueError(f"Cannot parse IEMOCAP session from dialogue ID {text!r}.")
    lowered = text.lower()
    if "_impro" in lowered:
        dialogue_type = "impro"
    elif "_script" in lowered:
        dialogue_type = "script"
    else:
        raise ValueError(f"Cannot parse impro/script type from dialogue ID {text!r}.")
    return match.group(1), dialogue_type


def flatten_utterances(value: Sequence[Any]) -> pd.DataFrame:
    """Flatten only current utterance vectors; no dialogue context is created."""

    video_ids, _, video_labels, video_text = value[:4]
    rows: List[Dict[str, Any]] = []
    feature_dim: int | None = None
    for dialogue_id in video_ids:
        session_id, dialogue_type = dialogue_metadata(dialogue_id)
        utterance_ids = list(video_ids[dialogue_id])
        labels = list(video_labels[dialogue_id])
        features = np.asarray(video_text[dialogue_id])
        if features.ndim != 2 or features.shape[0] != len(utterance_ids):
            raise ValueError(f"Invalid text feature shape for {dialogue_id!r}: {features.shape}.")
        if len(labels) != len(utterance_ids) or not np.isfinite(features).all():
            raise ValueError(f"Invalid labels or non-finite text features for {dialogue_id!r}.")
        feature_dim = feature_dim or int(features.shape[1])
        if int(features.shape[1]) != feature_dim:
            raise ValueError("Text feature dimensions differ across dialogues.")
        for utterance_index, (utterance_id, label, feature) in enumerate(
            zip(utterance_ids, labels, features)
        ):
            rows.append(
                {
                    "dialogue_id": str(dialogue_id),
                    "utterance_id": str(utterance_id),
                    "utterance_index": int(utterance_index),
                    "session_id": session_id,
                    "dialogue_type": dialogue_type,
                    "label_id": int(label),
                    "feature": np.asarray(feature, dtype=np.float32),
                }
            )
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError("IEMOCAP PKL contains no utterances.")
    missing_sessions = sorted(set(SESSIONS) - set(frame["session_id"]))
    if missing_sessions:
        raise ValueError(f"Probe requires Ses01-Ses05; missing {missing_sessions}.")
    return frame


def metric_values(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    labels = sorted(set(int(value) for value in y_true))
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "macro_f1": float(
            f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
        ),
        "uar": float(
            recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
        ),
    }


def nearest_centroid_predict(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    test_features: np.ndarray,
) -> np.ndarray:
    classes = np.unique(train_labels)
    centroids = np.stack([train_features[train_labels == label].mean(axis=0) for label in classes])
    distances = np.square(test_features[:, None, :] - centroids[None, :, :]).sum(axis=2)
    return classes[distances.argmin(axis=1)]


def cosine_one_nn_predict(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    test_features: np.ndarray,
    chunk_size: int = 512,
) -> np.ndarray:
    train_norm = train_features / np.maximum(
        np.linalg.norm(train_features, axis=1, keepdims=True), 1.0e-12
    )
    test_norm = test_features / np.maximum(
        np.linalg.norm(test_features, axis=1, keepdims=True), 1.0e-12
    )
    predictions: List[np.ndarray] = []
    for start in range(0, len(test_norm), int(chunk_size)):
        similarities = test_norm[start : start + int(chunk_size)] @ train_norm.T
        predictions.append(train_labels[similarities.argmax(axis=1)])
    return np.concatenate(predictions)


def _append_group_metrics(
    rows: List[Dict[str, Any]],
    *,
    method: str,
    held_out_session: str,
    dialogue_types: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> None:
    for group in ("all", "impro", "script"):
        mask = np.ones(len(y_true), dtype=bool) if group == "all" else dialogue_types == group
        if not np.any(mask):
            continue
        rows.append(
            {
                "method": method,
                "held_out_session": held_out_session,
                "dialogue_type": group,
                "utterance_count": int(mask.sum()),
                **metric_values(y_true[mask], y_pred[mask]),
            }
        )


def exact_duplicate_coverage(
    train_features: np.ndarray,
    test_features: np.ndarray,
) -> Tuple[int, float]:
    train_keys = {np.ascontiguousarray(row, dtype=np.float32).tobytes() for row in train_features}
    matches = sum(
        np.ascontiguousarray(row, dtype=np.float32).tobytes() in train_keys
        for row in test_features
    )
    return int(matches), float(matches / max(len(test_features), 1))


def run_probes(
    frame: pd.DataFrame,
    *,
    seed: int = 42,
    max_iter: int = 2000,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metric_rows: List[Dict[str, Any]] = []
    duplicate_rows: List[Dict[str, Any]] = []
    for held_out in SESSIONS:
        test_mask = frame["session_id"].to_numpy() == held_out
        train_frame = frame.loc[~test_mask]
        test_frame = frame.loc[test_mask]
        train_raw = np.stack(train_frame["feature"].to_numpy())
        test_raw = np.stack(test_frame["feature"].to_numpy())
        y_train = train_frame["label_id"].to_numpy(dtype=np.int64)
        y_test = test_frame["label_id"].to_numpy(dtype=np.int64)
        types = test_frame["dialogue_type"].to_numpy()

        # The scaler is deliberately fitted only on the four training sessions.
        scaler = StandardScaler().fit(train_raw)
        x_train = scaler.transform(train_raw)
        x_test = scaler.transform(test_raw)
        linear = LogisticRegression(
            random_state=int(seed),
            max_iter=int(max_iter),
            solver="lbfgs",
        ).fit(x_train, y_train)
        predictions = {
            "linear_logistic_regression": linear.predict(x_test),
            "nearest_class_centroid": nearest_centroid_predict(x_train, y_train, x_test),
            "cross_session_cosine_1nn": cosine_one_nn_predict(x_train, y_train, x_test),
        }
        for method, y_pred in predictions.items():
            _append_group_metrics(
                metric_rows,
                method=method,
                held_out_session=held_out,
                dialogue_types=types,
                y_true=y_test,
                y_pred=np.asarray(y_pred, dtype=np.int64),
            )
        matches, coverage = exact_duplicate_coverage(train_raw, test_raw)
        duplicate_rows.append(
            {
                "held_out_session": held_out,
                "held_out_utterance_count": len(test_raw),
                "exact_duplicate_matches": matches,
                "exact_duplicate_coverage": coverage,
            }
        )

    label_distribution = (
        frame.groupby(["session_id", "dialogue_type", "label_id"], dropna=False)
        .size()
        .rename("utterance_count")
        .reset_index()
    )
    return pd.DataFrame(metric_rows), label_distribution, pd.DataFrame(duplicate_rows)


def save_plots(
    output_dir: Path,
    metrics: pd.DataFrame,
    label_distribution: pd.DataFrame,
    duplicates: pd.DataFrame,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    overall = metrics[metrics["dialogue_type"] == "all"]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for method, group in overall.groupby("method"):
        ax.plot(group["held_out_session"], group["accuracy"], marker="o", label=method)
    ax.set_xlabel("Held-out session")
    ax.set_ylabel("Accuracy")
    ax.set_title("Current-utterance text probes")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "heldout_session_accuracy.png", dpi=180)
    plt.close(fig)

    pivot = label_distribution.pivot_table(
        index="session_id", columns="label_id", values="utterance_count", aggfunc="sum", fill_value=0
    )
    fig, ax = plt.subplots(figsize=(8, 4.5))
    pivot.plot(kind="bar", ax=ax)
    ax.set_xlabel("Session")
    ax.set_ylabel("Utterances")
    ax.set_title("IEMOCAP label distribution by session")
    fig.tight_layout()
    fig.savefig(output_dir / "label_distribution.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(duplicates["held_out_session"], duplicates["exact_duplicate_coverage"])
    ax.set_xlabel("Held-out session")
    ax.set_ylabel("Exact duplicate coverage")
    ax.set_title("Exact train-to-held-out feature duplicates")
    fig.tight_layout()
    fig.savefig(output_dir / "exact_duplicate_coverage.png", dpi=180)
    plt.close(fig)


def write_report(
    output_dir: Path,
    input_pkl: Path,
    metrics: pd.DataFrame,
    label_distribution: pd.DataFrame,
    duplicates: pd.DataFrame,
    seed: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(output_dir / "heldout_session_metrics.csv", index=False, encoding="utf-8-sig")
    label_distribution.to_csv(
        output_dir / "label_distribution.csv", index=False, encoding="utf-8-sig"
    )
    duplicates.to_csv(
        output_dir / "exact_duplicate_coverage.csv", index=False, encoding="utf-8-sig"
    )
    overall = metrics[metrics["dialogue_type"] == "all"]

    def markdown_table(frame: pd.DataFrame) -> str:
        columns = [str(column) for column in frame.columns]
        header = "| " + " | ".join(columns) + " |"
        separator = "|" + "|".join("---" for _ in columns) + "|"
        rows = [
            "| " + " | ".join(str(value) for value in row) + " |"
            for row in frame.itertuples(index=False, name=None)
        ]
        return "\n".join([header, separator, *rows])

    lines = [
        "# IEMOCAP current-utterance text feature probe",
        "",
        f"- Input: `{input_pkl}`",
        f"- Seed: {int(seed)}",
        "- Causality scope: `current_utterance_only_no_dialogue_context`",
        "- Standardization: fit independently on the four training sessions in each fold.",
        "- Folds: Ses01-Ses05 are each held out once; no test result selects a representation.",
        "",
        "## Overall held-out metrics",
        "",
        markdown_table(overall),
        "",
        "## Exact duplicate coverage",
        "",
        markdown_table(duplicates),
        "",
        "Improvised and scripted subsets are available in `heldout_session_metrics.csv`.",
    ]
    (output_dir / "text_feature_probe_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    save_plots(output_dir, metrics, label_distribution, duplicates)


def main() -> None:
    args = parse_args()
    input_pkl = Path(args.input_pkl)
    frame = flatten_utterances(load_feature_pkl(input_pkl))
    metrics, label_distribution, duplicates = run_probes(
        frame, seed=args.seed, max_iter=args.max_iter
    )
    write_report(
        Path(args.output_dir),
        input_pkl,
        metrics,
        label_distribution,
        duplicates,
        args.seed,
    )
    print(
        json.dumps(
            {
                "utterance_count": len(frame),
                "feature_dim": len(frame.iloc[0]["feature"]),
                "output_dir": str(args.output_dir),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()

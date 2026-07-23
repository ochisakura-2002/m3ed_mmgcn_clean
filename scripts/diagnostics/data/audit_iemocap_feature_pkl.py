"""Audit a candidate IEMOCAP feature PKL against the immutable legacy PKL."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import pickle
import sys
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.run_metadata import compute_file_sha256  # noqa: E402


ITEM_NAMES = (
    "videoIDs",
    "videoSpeakers",
    "videoLabels",
    "videoText",
    "videoAudio",
    "videoVisual",
    "videoSentence",
    "trainVid",
    "testVid",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy-pkl", required=True)
    parser.add_argument("--candidate-pkl", required=True)
    parser.add_argument("--expected-legacy-sha256", required=True)
    parser.add_argument("--expected-text-dim", type=int, default=768)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


def load_pickle(path: Path) -> Any:
    with Path(path).open("rb") as file:
        try:
            return pickle.load(file, encoding="latin1")
        except TypeError:
            file.seek(0)
            return pickle.load(file)


def is_nine_item_structure(value: Any) -> bool:
    return (
        isinstance(value, (list, tuple))
        and len(value) == 9
        and all(isinstance(item, Mapping) for item in value[:7])
        and isinstance(value[7], (list, tuple, set))
        and isinstance(value[8], (list, tuple, set))
    )


def deep_equal(left: Any, right: Any) -> bool:
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        left_array = np.asarray(left)
        right_array = np.asarray(right)
        return (
            left_array.shape == right_array.shape
            and left_array.dtype == right_array.dtype
            and np.array_equal(left_array, right_array, equal_nan=True)
        )
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return set(left) == set(right) and all(deep_equal(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return type(left) is type(right) and len(left) == len(right) and all(
            deep_equal(a, b) for a, b in zip(left, right)
        )
    if isinstance(left, set) and isinstance(right, set):
        return left == right
    try:
        result = left == right
        return bool(np.all(result)) if isinstance(result, np.ndarray) else bool(result)
    except Exception:
        return False


def _array_contract_equal(left: Any, right: Any) -> tuple[bool, bool, bool]:
    left_array = np.asarray(left)
    right_array = np.asarray(right)
    shape_equal = left_array.shape == right_array.shape
    dtype_equal = left_array.dtype == right_array.dtype
    value_equal = bool(
        shape_equal
        and dtype_equal
        and np.array_equal(left_array, right_array, equal_nan=True)
    )
    return shape_equal, dtype_equal, value_equal


def _statistics_row(source: str, modality: str, dialogue_map: Mapping[Any, Any]) -> Dict[str, Any]:
    arrays = [np.asarray(value) for value in dialogue_map.values()]
    flattened = np.concatenate([array.reshape(-1) for array in arrays]) if arrays else np.array([])
    dims = sorted({int(array.shape[-1]) for array in arrays if array.ndim == 2})
    try:
        finite = bool(flattened.size and np.isfinite(flattened).all())
        mean = float(np.mean(flattened)) if flattened.size else float("nan")
        std = float(np.std(flattened)) if flattened.size else float("nan")
        minimum = float(np.min(flattened)) if flattened.size else float("nan")
        maximum = float(np.max(flattened)) if flattened.size else float("nan")
    except (TypeError, ValueError):
        finite = False
        mean = std = minimum = maximum = float("nan")
    return {
        "source": source,
        "modality": modality,
        "dialogue_count": len(arrays),
        "element_count": int(flattened.size),
        "feature_dims": ";".join(str(value) for value in dims),
        "dtype_set": ";".join(sorted({str(array.dtype) for array in arrays})),
        "all_finite": finite,
        "mean": mean,
        "std": std,
        "min": minimum,
        "max": maximum,
    }


def audit_feature_pkls(
    legacy_pkl: Path,
    candidate_pkl: Path,
    *,
    expected_legacy_sha256: str,
    expected_text_dim: int = 768,
) -> tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Return summary, per-dialogue rows, and feature-statistics rows."""

    legacy_sha = compute_file_sha256(Path(legacy_pkl)).lower()
    expected_sha = str(expected_legacy_sha256).strip().lower()
    legacy_sha_matches = legacy_sha == expected_sha
    legacy = load_pickle(Path(legacy_pkl))
    candidate = load_pickle(Path(candidate_pkl))
    legacy_structure = is_nine_item_structure(legacy)
    candidate_structure = is_nine_item_structure(candidate)
    checks: Dict[str, bool] = {
        "legacy_sha256_matches": legacy_sha_matches,
        "legacy_nine_item_structure": legacy_structure,
        "candidate_nine_item_structure": candidate_structure,
    }
    dialogue_rows: List[Dict[str, Any]] = []
    statistics_rows: List[Dict[str, Any]] = []

    if not legacy_structure or not candidate_structure:
        summary = {
            "legacy_pkl": str(legacy_pkl),
            "candidate_pkl": str(candidate_pkl),
            "legacy_pkl_sha256": legacy_sha,
            "expected_legacy_sha256": expected_sha,
            "expected_text_dim": int(expected_text_dim),
            "checks": checks,
            "passed": False,
            "failed_checks": [name for name, passed in checks.items() if not passed],
        }
        return summary, dialogue_rows, statistics_rows

    legacy_ids = set(legacy[0])
    candidate_ids = set(candidate[0])
    common_ids = [dialogue_id for dialogue_id in legacy[0] if dialogue_id in candidate_ids]
    checks["dialogue_id_sets_identical"] = legacy_ids == candidate_ids
    checks["legacy_dialogue_mappings_aligned"] = all(
        set(legacy[index]) == legacy_ids for index in range(1, 7)
    )
    checks["candidate_dialogue_mappings_aligned"] = all(
        set(candidate[index]) == candidate_ids for index in range(1, 7)
    )
    checks["candidate_videoText_dialogue_ids_match"] = (
        set(candidate[3]) == candidate_ids
    )
    checks["utterance_counts_identical"] = all(
        len(legacy[0][dialogue_id]) == len(candidate[0][dialogue_id])
        for dialogue_id in common_ids
    ) and legacy_ids == candidate_ids
    for index, name in ((0, "videoIDs"), (1, "videoSpeakers"), (2, "videoLabels"), (6, "videoSentence")):
        checks[f"{name}_identical"] = deep_equal(legacy[index], candidate[index])
    checks["trainVid_identical"] = deep_equal(legacy[7], candidate[7])
    checks["testVid_identical"] = deep_equal(legacy[8], candidate[8])

    contract_ids = [
        dialogue_id
        for dialogue_id in common_ids
        if all(
            dialogue_id in legacy[index] and dialogue_id in candidate[index]
            for index in range(7)
        )
    ]
    all_dialogues_audited = (
        legacy_ids == candidate_ids
        and len(contract_ids) == len(legacy_ids)
        and checks["legacy_dialogue_mappings_aligned"]
        and checks["candidate_dialogue_mappings_aligned"]
    )
    audio_shapes = audio_dtypes = audio_values = True
    visual_shapes = visual_dtypes = visual_values = True
    text_float32 = text_dim_ok = text_finite = text_counts_match = no_empty = True
    for dialogue_id in contract_ids:
        legacy_count = len(legacy[0][dialogue_id])
        candidate_count = len(candidate[0][dialogue_id])
        audio_contract = _array_contract_equal(legacy[4][dialogue_id], candidate[4][dialogue_id])
        visual_contract = _array_contract_equal(legacy[5][dialogue_id], candidate[5][dialogue_id])
        audio_shapes &= audio_contract[0]
        audio_dtypes &= audio_contract[1]
        audio_values &= audio_contract[2]
        visual_shapes &= visual_contract[0]
        visual_dtypes &= visual_contract[1]
        visual_values &= visual_contract[2]
        text = np.asarray(candidate[3][dialogue_id])
        text_row_count = int(text.shape[0]) if text.ndim >= 1 else 0
        dialogue_text_float32 = text.dtype == np.float32
        dialogue_text_dim = text.ndim == 2 and text.shape[1] == int(expected_text_dim)
        try:
            dialogue_text_finite = bool(np.isfinite(text).all())
        except TypeError:
            dialogue_text_finite = False
        dialogue_text_count = text.ndim >= 1 and text_row_count == candidate_count
        dialogue_nonempty = legacy_count > 0 and candidate_count > 0 and text_row_count > 0
        text_float32 &= dialogue_text_float32
        text_dim_ok &= dialogue_text_dim
        text_finite &= dialogue_text_finite
        text_counts_match &= dialogue_text_count
        no_empty &= dialogue_nonempty
        dialogue_rows.append(
            {
                "dialogue_id": str(dialogue_id),
                "legacy_utterance_count": legacy_count,
                "candidate_utterance_count": candidate_count,
                "candidate_text_shape": "x".join(str(value) for value in text.shape),
                "candidate_text_dtype": str(text.dtype),
                "candidate_text_finite": dialogue_text_finite,
                "candidate_text_count_matches": dialogue_text_count,
                "audio_shape_identical": audio_contract[0],
                "audio_dtype_identical": audio_contract[1],
                "audio_values_identical": audio_contract[2],
                "visual_shape_identical": visual_contract[0],
                "visual_dtype_identical": visual_contract[1],
                "visual_values_identical": visual_contract[2],
            }
        )

    checks.update(
        {
            "videoAudio_shapes_identical": audio_shapes and all_dialogues_audited,
            "videoAudio_dtypes_identical": audio_dtypes and all_dialogues_audited,
            "videoAudio_values_identical": audio_values and all_dialogues_audited,
            "videoVisual_shapes_identical": visual_shapes and all_dialogues_audited,
            "videoVisual_dtypes_identical": visual_dtypes and all_dialogues_audited,
            "videoVisual_values_identical": visual_values and all_dialogues_audited,
            "candidate_videoText_float32": text_float32 and all_dialogues_audited,
            "candidate_videoText_expected_dim": text_dim_ok and all_dialogues_audited,
            "candidate_videoText_all_finite": text_finite and all_dialogues_audited,
            "candidate_videoText_utterance_counts_match": text_counts_match
            and all_dialogues_audited,
            "no_empty_dialogues": no_empty and all_dialogues_audited,
            "train_test_dialogue_overlap_zero": len(set(candidate[7]) & set(candidate[8])) == 0,
        }
    )
    non_text_check_names = [
        "dialogue_id_sets_identical",
        "legacy_dialogue_mappings_aligned",
        "candidate_dialogue_mappings_aligned",
        "utterance_counts_identical",
        "videoIDs_identical",
        "videoSpeakers_identical",
        "videoLabels_identical",
        "videoSentence_identical",
        "trainVid_identical",
        "testVid_identical",
        "videoAudio_shapes_identical",
        "videoAudio_dtypes_identical",
        "videoAudio_values_identical",
        "videoVisual_shapes_identical",
        "videoVisual_dtypes_identical",
        "videoVisual_values_identical",
    ]
    checks["only_videoText_changed"] = all(checks[name] for name in non_text_check_names)

    for source_name, value in (("legacy", legacy), ("candidate", candidate)):
        for index, modality in ((3, "text"), (4, "audio"), (5, "visual")):
            statistics_rows.append(_statistics_row(source_name, modality, value[index]))

    failed = [name for name, passed in checks.items() if not passed]
    summary = {
        "legacy_pkl": str(legacy_pkl),
        "candidate_pkl": str(candidate_pkl),
        "legacy_pkl_sha256": legacy_sha,
        "expected_legacy_sha256": expected_sha,
        "candidate_pkl_sha256": compute_file_sha256(Path(candidate_pkl)),
        "expected_text_dim": int(expected_text_dim),
        "dialogue_count": len(candidate_ids),
        "utterance_count": int(sum(len(candidate[0][key]) for key in candidate[0])),
        "checks": checks,
        "failed_checks": failed,
        "passed": not failed,
    }
    return summary, dialogue_rows, statistics_rows


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_audit_outputs(
    output_dir: Path,
    summary: Mapping[str, Any],
    dialogue_rows: Sequence[Mapping[str, Any]],
    statistics_rows: Sequence[Mapping[str, Any]],
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "feature_audit_summary.json").open("w", encoding="utf-8") as file:
        json.dump(dict(summary), file, ensure_ascii=False, indent=2)
        file.write("\n")
    _write_csv(output_dir / "dialogue_shape_audit.csv", dialogue_rows)
    _write_csv(output_dir / "feature_statistics.csv", statistics_rows)
    checks = summary.get("checks", {})
    lines = [
        "# IEMOCAP feature PKL audit",
        "",
        f"- Overall: **{'PASS' if summary.get('passed') else 'FAIL'}**",
        f"- Legacy SHA256: `{summary.get('legacy_pkl_sha256', '')}`",
        f"- Candidate SHA256: `{summary.get('candidate_pkl_sha256', '')}`",
        f"- Dialogues: {summary.get('dialogue_count', 0)}",
        f"- Utterances: {summary.get('utterance_count', 0)}",
        "",
        "## Checks",
        "",
        "| Check | Result |",
        "|---|---|",
    ]
    lines.extend(
        f"| `{name}` | {'PASS' if passed else 'FAIL'} |"
        for name, passed in checks.items()
    )
    (output_dir / "feature_audit_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    summary, dialogue_rows, statistics_rows = audit_feature_pkls(
        Path(args.legacy_pkl),
        Path(args.candidate_pkl),
        expected_legacy_sha256=args.expected_legacy_sha256,
        expected_text_dim=args.expected_text_dim,
    )
    write_audit_outputs(Path(args.output_dir), summary, dialogue_rows, statistics_rows)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.strict and not summary["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

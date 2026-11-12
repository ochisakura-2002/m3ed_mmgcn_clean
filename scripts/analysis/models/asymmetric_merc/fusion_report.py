"""Serialization and evidence-bounded Markdown for the Q2-2 runtime."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8")


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_report(output_dir: Path, state: dict) -> Path:
    synthetic = state["synthetic"]
    lines = [
        "# Q2-2 weighted log-probability fusion",
        "",
        f"Runtime status: {state['status']}",
        f"Evidence mode: {'SYNTHETIC ONLY; no scientific conclusion' if synthetic else 'frozen Dev predictions'}",
        "",
        "Protocol: alpha * log(max(P_TA, 1e-12)) + (1-alpha) * log(max(P_V, 1e-12)).",
        "Only Linear-B seed100 selects alpha on the 101-point grid by Weighted-F1.",
        "Scores within 1e-12 of the global maximum tie; the largest alpha wins.",
        "The selected alpha is then fixed for all five Linear-B seeds.",
        "Original S2D is a complementarity reference only.",
        "All metrics use the exact ID intersection for that visual source.",
        "Deltas are fractions (multiply by 100 for percentage points).",
        "",
    ]
    if "error" in state:
        lines.extend([f"STOP / REPORT MISMATCH or audit failure: {state['error']}", ""])
    if "selection" in state:
        lines.extend([f"Selected alpha: {state['selection']['alpha']:.2f}", ""])
    if "primary" in state:
        p = state["primary"]
        lines.extend([
            f"Common Dev rows: {p['support']}",
            f"Q11/Q10/Q01/Q00: {p['q11']}/{p['q10']}/{p['q01']}/{p['q00']}",
            f"Rescue rate: {p['rescue_rate']}; disagreement rate: {p['disagreement_rate']}",
            f"Oracle accuracy: {p['oracle_accuracy']:.9f}; oracle gain: {p['oracle_gain']:.9f}",
            "",
        ])
    if "robustness" in state:
        rows = state["robustness"]
        positive = sum(row["delta_vs_ta_weighted_f1"] > 0 for row in rows)
        primary = rows[0]
        lines.extend([
            "| Metric | TA (common) | Linear-B seed100 | Fusion | Fusion - TA |",
            "| --- | ---: | ---: | ---: | ---: |",
        ])
        for key in ("accuracy", "weighted_f1", "macro_f1", "uar"):
            lines.append(f"| {key} | {primary[f'ta_{key}']:.9f} | {primary[f'v_{key}']:.9f} | "
                         f"{primary[f'fusion_{key}']:.9f} | {primary[f'delta_vs_ta_{key}']:+.9f} |")
        lines.extend(["", "| V seed | Fixed alpha | Fusion W-F1 | Delta W-F1 vs matched TA |",
                      "| --- | ---: | ---: | ---: |"])
        for row in rows:
            lines.append(f"| {row['seed']} | {row['alpha']:.2f} | {row['fusion_weighted_f1']:.9f} | "
                         f"{row['delta_vs_ta_weighted_f1']:+.9f} |")
        lines.extend(["", f"Positive Weighted-F1 deltas: {positive}/{len(rows)} visual seeds.", ""])
        if not synthetic and state["status"] == "PASS":
            primary_positive = primary["delta_vs_ta_weighted_f1"] > 0
            lines.extend([
                "FACT: Dev simple weighted fusion has a positive signal."
                if primary_positive else "FACT: Dev simple weighted fusion has no positive Weighted-F1 delta.",
                "FACT: All five V seeds have a positive direction after freezing alpha."
                if positive == len(rows) else "FACT: A positive direction is not shared by all five V seeds.",
                "Q2_2_DEV_SIGNAL = POSITIVE" if primary_positive and positive == len(rows)
                else "Q2_2_DEV_SIGNAL = NOT_REPRODUCED",
                "",
            ])
    lines.extend([
        "LIMITATION: Alpha was selected on Dev; this remains development evidence.",
        "LIMITATION: Only the visual seed varies; the TA source remains fixed.",
        "LIMITATION: Plain prediction files without a split column rely on the caller's Dev declaration.",
        "Source model, feature, seed and epoch identities are declarations; CSV contents alone cannot verify them.",
        "Zero-denominator rescue rates are null/blank; absent classes have zero PR/F1/recall and remain visible.",
        "",
        "Isolation: test_access = false; test_loader_built = 0; test_evaluation_count = 0.",
        "No checkpoint loading, model training, feature fusion, calibration or learned parameters occur.",
        "",
        "Audit links: [input manifest](../provenance/input_manifest.json), "
        "[protocol](../provenance/protocol.json), [joins](../provenance/exact_join_audit.json), "
        "[regressions](../provenance/regression_audit.json), [run status](../provenance/run_status.json).",
        "All input hashes, code hashes, Git HEAD/dirty status and runtime versions are recorded in provenance.",
    ])
    path = output_dir / "reports/Q2_2_WEIGHTED_FUSION_REPORT.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path

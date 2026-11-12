"""Dev scalar-gate mechanism facts using the frozen Q2-1 quadrant definition."""

import csv
from pathlib import Path

import numpy as np

from scripts.analysis.models.asymmetric_merc.fusion_report import write_json
from scripts.analysis.models.asymmetric_merc.weighted_fusion import LABEL_NAMES

QUADRANTS = ("Q11", "Q10", "Q01", "Q00")


def assign_quadrants(true, ta_pred, v_pred):
    ta_ok = np.asarray(ta_pred) == np.asarray(true)
    v_ok = np.asarray(v_pred) == np.asarray(true)
    # Same TA-correct / V-correct bits as weighted_fusion.complementarity.
    return np.array([f"Q{int(t)}{int(v)}" for t, v in zip(ta_ok, v_ok)])


def summarize_gates(true, ta_pred, v_pred, gates):
    gates = np.asarray(gates, dtype=np.float64).reshape(-1)
    if (len(gates) != len(true) or len(ta_pred) != len(true) or len(v_pred) != len(true)
            or not np.isfinite(gates).all() or ((gates < 0) | (gates > 1)).any()):
        raise ValueError("Dev gates must be finite sigmoid scalars for every ID")
    quadrants = assign_quadrants(true, ta_pred, v_pred)

    def stats(mask):
        values = gates[mask]
        return {"count": len(values), "mean": float(values.mean()) if len(values) else None,
                "std": float(values.std(ddof=0)) if len(values) else None}

    groups = {key: stats(quadrants == key) for key in QUADRANTS}
    rescue, harm = groups["Q01"]["mean"], groups["Q10"]["mean"]
    return {"all": stats(np.ones(len(gates), dtype=bool)), "quadrants": groups,
            "delta_rescue_harm": rescue - harm if rescue is not None and harm is not None else None,
            "true_classes": {name: stats(np.asarray(true) == k) for k, name in enumerate(LABEL_NAMES)},
            "std_ddof": 0, "finite_precision_endpoint_count": int(((gates == 0) | (gates == 1)).sum()),
            "interpretation": "descriptive only; gate sign is not a PASS criterion"}


def write_gate_report(output, dev, vrg_pred, gates, *, source, synthetic):
    output = Path(output)
    ta_pred = dev["ta_prob"].argmax(1)
    v_pred = dev["v_probe_pred"]
    report = summarize_gates(dev["label"], ta_pred, v_pred, gates)
    quadrants = assign_quadrants(dev["label"], ta_pred, v_pred)
    fields = ("utterance_id", "dialogue_id", "true_label", "ta_pred", "v_pred", "vrg_pred", "gate", "quadrant")
    with (output / "dev_gates.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for i, uid in enumerate(dev["utterance_id"]):
            writer.writerow(dict(zip(fields, (uid, dev["dialogue_id"][i], int(dev["label"][i]),
                                              int(ta_pred[i]), int(v_pred[i]), int(vrg_pred[i]),
                                              float(gates[i]), quadrants[i]))))
    report.update(v_prediction_source=source, synthetic=synthetic, test_access=False)
    write_json(output / "gate_mechanism.json", report)
    lines = ["# VRG-v0 Dev gate mechanism", "",
             "Evidence: " + ("SYNTHETIC ONLY; no scientific conclusion" if synthetic else "frozen Dev common"),
             f"V prediction source: {source}", "",
             "Q11=TA correct/V correct; Q10=TA correct/V wrong; Q01=TA wrong/V correct; Q00=both wrong.",
             "Population standard deviation (ddof=0); absent groups are null. No gate-sign PASS criterion.",
             f"Finite-precision sigmoid endpoints (unclipped): {report['finite_precision_endpoint_count']}", "",
             "| Group | Count | Gate mean | Gate std |", "| --- | ---: | ---: | ---: |"]
    for key, values in [("all", report["all"]), *report["quadrants"].items(), *report["true_classes"].items()]:
        lines.append(f"| {key} | {values['count']} | {values['mean']} | {values['std']} |")
    lines.extend(["", f"delta_rescue_harm = mean(Q01) - mean(Q10) = {report['delta_rescue_harm']}", ""])
    (output / "gate_mechanism.md").write_text("\n".join(lines), encoding="utf-8")
    return report

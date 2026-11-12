"""Post-selection gate facts and the predeclared decision-fusion verdict."""

import csv
from pathlib import Path

import numpy as np

from scripts.analysis.models.asymmetric_merc.fusion_report import write_json
from scripts.analysis.models.asymmetric_merc.vrg_report import assign_quadrants, summarize_gates

LIMITATION = (
    "This experiment evaluates whether a very low-capacity dynamic decision rule can improve "
    "a frozen fixed-weight fusion baseline. Base-expert Train predictions are in-sample rather "
    "than out-of-fold, so a positive result is treated as a development signal pending "
    "reviewer-facing confirmation if required."
)
CRITERIA = {"strong_improvement": 0.005, "reliability_direction": "delta_rescue_harm > 0",
            "baseline": "frozen B0 alpha_TA=0.88, alpha_V=0.12",
            "selection": "strict Dev Weighted-F1 improvement; earliest epoch wins ties"}


def scientific_verdict(delta, direction):
    if delta <= 0:
        return "NEGATIVE"
    if delta < CRITERIA["strong_improvement"]:
        return "WEAK"
    if direction is None:
        return "UNDETERMINED_MISSING_QUADRANT"
    return "STRONG_POSITIVE" if direction > 0 else "PERFORMANCE_ONLY"


def write_decision_report(output, joined, predicted, gates, *, best_metrics, baseline, synthetic):
    output = Path(output)
    gates = np.asarray(gates).reshape(-1)
    report = summarize_gates(joined.true, joined.ta_predicted, joined.v_predicted, gates)
    report["all"].update(min=float(gates.min()), max=float(gates.max()))
    delta = best_metrics["weighted_f1"] - baseline["weighted_f1"]
    report.update(mean_abs_deviation_from_0_12=float(np.abs(gates - .12).mean()),
                  fraction_gate_gt_0_12=float((gates > .12).mean()),
                  fraction_gate_lt_0_12=float((gates < .12).mean()),
                  delta_weighted_f1=delta, criteria=CRITERIA,
                  verdict="SYNTHETIC_ONLY" if synthetic else scientific_verdict(delta, report["delta_rescue_harm"]),
                  interpretation="post-selection descriptive analysis; never used for optimization or checkpoint selection",
                  base_train_predictions="in_sample", cross_fitted=False, limitation=LIMITATION,
                  v_prediction_source="Frozen Linear-B seed100", synthetic=synthetic, test_access=False)
    quadrants = assign_quadrants(joined.true, joined.ta_predicted, joined.v_predicted)
    fields = ("utterance_id", "true_label", "ta_pred", "v_pred", "dynamic_pred", "visual_gate", "quadrant")
    with (output / "dev_gates.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for i, uid in enumerate(joined.ids):
            writer.writerow(dict(zip(fields, (uid, int(joined.true[i]), int(joined.ta_predicted[i]),
                                              int(joined.v_predicted[i]), int(predicted[i]),
                                              float(gates[i]), quadrants[i]))))
    write_json(output / "gate_mechanism.json", report)
    actions = {"STRONG_POSITIVE": "Proceed to seeds100-104 and necessary reviewer-facing confirmation in a separate task.",
               "PERFORMANCE_ONLY": "Performance signal; visual reliability claim unsupported. Seed expansion awaits GPT review.",
               "WEAK": "Insufficient magnitude; no automatic five-seed expansion.",
               "NEGATIVE": "End the current dynamic reliability route; no v2 gate, attention, router or MoE.",
               "SYNTHETIC_ONLY": "Implementation evidence only; no real-data conclusion."}
    lines = ["# TA-preserving Dynamic Decision Gate v0", "",
             f"Evidence: {'SYNTHETIC ONLY' if synthetic else 'frozen Train/Dev probabilities'}",
             f"Verdict: {report['verdict']}", actions.get(report["verdict"], "Missing quadrant; scientific decision unavailable."),
             "", LIMITATION, "", "Dev labels are used only for checkpoint selection / evaluation.",
             "Q11=TA correct/V correct; Q10=TA correct/V wrong; Q01=TA wrong/V correct; Q00=both wrong.",
             "Population std (ddof=0); absent groups are null. Mechanism computed only after selection.",
             "", "| Group | Count | Mean | Std |", "| --- | ---: | ---: | ---: |"]
    for key, value in [("all", report["all"]), *report["quadrants"].items(), *report["true_classes"].items()]:
        lines.append(f"| {key} | {value['count']} | {value['mean']} | {value['std']} |")
    lines.extend(["", f"B0 metrics (full precision): {baseline}", f"Selected Dev metrics: {best_metrics}",
                  f"Delta W-F1: {delta}", f"delta_rescue_harm: {report['delta_rescue_harm']}",
                  f"Gate min/max: {report['all']['min']} / {report['all']['max']}",
                  f"Mean |g-0.12|: {report['mean_abs_deviation_from_0_12']}",
                  f"Fractions g>0.12 / g<0.12: {report['fraction_gate_gt_0_12']} / {report['fraction_gate_lt_0_12']}",
                  "", "Fixed criteria: delta W-F1 >= 0.005 and direction > 0: STRONG_POSITIVE;",
                  "delta >= 0.005 and direction <= 0: PERFORMANCE_ONLY; 0 < delta < 0.005: WEAK; delta <= 0: NEGATIVE.",
                  "No automatic training expansion. test_access=false; no Test loader or evaluation.", ""])
    (output / "gate_mechanism.md").write_text("\n".join(lines), encoding="utf-8")
    return report

"""Sample-level evidence and fixed five-seed rules for audit Round 1."""

from collections import defaultdict
from itertools import combinations
from statistics import mean, median

import numpy as np

from datasets.iemocap.ta_v0_dataset import UID
from models.multidag_cl.ta_v0 import LABEL_VOCAB
from scripts.runtime.multidag_cl_ta_v0.training import classification_report
from .failure_audit_inference import HISTORY, MASKS, REPEATS


C = len(LABEL_VOCAB)
PAIRS = (("hap", "exc"), ("neu", "fru"), ("ang", "fru"))
GROUPS = ("SHIFT", "STABLE", "NO_HISTORY")
SINGLES = ("T_ONLY", "A_ONLY", "V_ONLY")
MIN_SEEDS = 4
EPS = 1e-6


def metadata(population, speakers=None):
    result, previous, positions = [], {}, defaultdict(int)
    if speakers is None:
        speakers = [UID.fullmatch(uid)[2] for uid, _, _ in population]
    if len(speakers) != len(population):
        raise ValueError("speaker and utterance counts differ")
    for (uid, dialogue, label), speaker in zip(population, speakers):
        key = dialogue, speaker
        last = previous.get(key)
        shift = "NO_HISTORY" if last is None else "STABLE" if last == label else "SHIFT"
        result.append({"utterance_id": uid, "dialogue_id": dialogue, "target_index": positions[dialogue],
                       "speaker": speaker, "true_label": LABEL_VOCAB[label], "shift_group": shift,
                       "previous_same_speaker_label": LABEL_VOCAB[last] if last is not None else None})
        previous[key] = label
        positions[dialogue] += 1
    return result


def details(logits, labels):
    logits = np.asarray(logits, dtype=np.float64)
    if logits.shape != (len(labels), C) or not np.isfinite(logits).all():
        raise ValueError("invalid or nonfinite frozen logits")
    exp = np.exp(logits - logits.max(axis=1, keepdims=True))
    prob = exp / exp.sum(axis=1, keepdims=True)
    idx = np.arange(len(labels))
    pred = logits.argmax(axis=1)
    true_mask = np.arange(C)[None, :] == labels[:, None]
    prob_margin = prob[idx, labels] - np.where(true_mask, -np.inf, prob).max(axis=1)
    logit_margin = logits[idx, labels] - np.where(true_mask, -np.inf, logits).max(axis=1)
    top = np.sort(prob, axis=1)[:, -2:]
    return {"prediction": pred, "correct": pred == labels, "probability": prob,
            "confidence": top[:, 1], "true_class_probability_margin": prob_margin,
            "true_class_logit_margin": logit_margin,
            "mad": np.clip(1 - np.abs(prob - 1 / C).sum(axis=1) / (2 * (1 - 1 / C)), 0, 1),
            "normalized_entropy": entropy(prob) / np.log(C), "top2_margin": top[:, 1] - top[:, 0]}


def entropy(probability):
    return -np.sum(probability * np.log(np.maximum(probability, np.finfo(float).tiny)), axis=1)


def quartiles(values):
    """Equal-frequency ranks, breaking exact ties by frozen population order."""
    result = np.empty(len(values), dtype=int)
    for index, part in enumerate(np.array_split(np.argsort(values, kind="stable"), 4), 1):
        result[part] = index
    return result


def average(values):
    return float(np.mean(values)) if len(values) else None


def scalar(values, index):
    return values[index].item()


def score(data, labels, select):
    if not np.any(select):
        return dict(support=0, accuracy=None, weighted_f1=None, macro_f1=None, uar=None, error_rate=None,
                    confidence=None, true_class_probability_margin=None, true_class_logit_margin=None)
    report = classification_report(labels[select], data["prediction"][select], loss=0)["metrics"]
    return {"support": int(select.sum()), **{key: report[key] for key in ("accuracy", "weighted_f1", "macro_f1", "uar")},
            "error_rate": 1 - report["accuracy"], **{key: average(data[key][select]) for key in
                                                     ("confidence", "true_class_probability_margin",
                                                      "true_class_logit_margin")}}


def selection_groups(meta):
    shifts = np.asarray([row["shift_group"] for row in meta])
    return {"ALL": np.ones(len(meta), dtype=bool), **{name: shifts == name for name in GROUPS}}


def shift_audit(seed, data, labels, groups):
    rows = []
    for name in GROUPS:
        select = groups[name]
        class_metrics = classification_report(labels[select], data["prediction"][select], loss=0)["per_class"]
        for cls in ("ALL", *LABEL_VOCAB):
            selection = groups[name] if cls == "ALL" else groups[name] & (labels == LABEL_VOCAB.index(cls))
            row = {"seed": seed, "group": name, "true_label": cls, **score(data, labels, selection)}
            if cls != "ALL":
                item = class_metrics[LABEL_VOCAB.index(cls)]
                row.update({f"class_{key}": item[key] for key in ("precision", "recall", "f1")})
            rows.append(row)
    shift, stable = (next(row for row in rows if row["group"] == group and row["true_label"] == "ALL")
                     for group in ("SHIFT", "STABLE"))
    return rows, {"wf1_gap": stable["weighted_f1"] - shift["weighted_f1"],
                  "error_gap": shift["error_rate"] - stable["error_rate"]}


def context_audit(seed, original, variants, labels, groups):
    rows, aggregates = [], {}
    for group in ("ALL", "SHIFT", "STABLE"):
        select = groups[group]
        baseline = score(original, labels, select)
        rows.append({"seed": seed, "group": group, "condition": "Original", "repeat": "BASE",
                     **baseline, "prediction_flip_rate": 0.0, "wf1_drop_pp": 0.0})
        for condition in ("ECCS", "EICS"):
            repeats = []
            for repeat in REPEATS:
                data = variants[condition, repeat]
                summary = score(data, labels, select)
                repeats.append({**summary, "prediction_flip_rate": average(
                    (data["prediction"] != original["prediction"])[select]),
                                "wf1_drop_pp": (baseline["weighted_f1"] - summary["weighted_f1"]) * 100})
                rows.append({"seed": seed, "group": group, "condition": condition, "repeat": repeat, **repeats[-1]})
            averaged = {key: mean(row[key] for row in repeats) for key in repeats[0]}
            rows.append({"seed": seed, "group": group, "condition": condition, "repeat": "MEAN", **averaged})
            aggregates[group, condition] = averaged["wf1_drop_pp"]
    eccs, eics = (aggregates["ALL", condition] for condition in ("ECCS", "EICS"))
    return rows, {"eccs_drop_pp": eccs, "eics_drop_pp": eics, "contrast_pp": eics - eccs}


def history_audit(seed, values, meta, groups):
    full, current = (values[name] for name in HISTORY[:2])
    delta = full["true_class_logit_margin"] - current["true_class_logit_margin"]
    helpful_flip = full["correct"] & ~current["correct"]
    harmful_flip = ~full["correct"] & current["correct"]
    helpful = helpful_flip | (delta > EPS)
    harmful = harmful_flip | (delta < -EPS)
    same = values["SAME_SPEAKER_ONLY"]["true_class_logit_margin"]
    other = values["OTHER_SPEAKER_ONLY"]["true_class_logit_margin"]
    dependency = np.select((same - other > EPS, other - same > EPS), ("SAME_FAVORED", "OTHER_FAVORED"),
                           default="TIED")
    samples, rows = [], []
    for i, item in enumerate(meta):
        base = {"seed": seed, **item, "speaker_dependency": dependency[i],
                "full_minus_current_logit_margin": scalar(delta, i), "helpful_flip": scalar(helpful_flip, i),
                "harmful_flip": scalar(harmful_flip, i), "helpful": scalar(helpful, i), "harmful": scalar(harmful, i)}
        for name, data in values.items():
            prefix = name.lower()
            base.update({f"{prefix}_prediction": LABEL_VOCAB[data["prediction"][i]],
                         f"{prefix}_correct": scalar(data["correct"], i),
                         f"{prefix}_logit_margin": scalar(data["true_class_logit_margin"], i),
                         f"{prefix}_probability_margin": scalar(data["true_class_probability_margin"], i)})
            rows.append({**base_identity(seed, item), "row_type": "sample", "condition": name,
                         "speaker_dependency": dependency[i], "predicted_label": LABEL_VOCAB[data["prediction"][i]],
                         "correct": scalar(data["correct"], i),
                         "true_class_logit_margin": scalar(data["true_class_logit_margin"], i),
                         "true_class_probability_margin": scalar(data["true_class_probability_margin"], i)})
        samples.append(base)

    strata = {}
    for name, items in (("shift_group", [row["shift_group"] for row in meta]),
                        ("speaker", [row["speaker"] for row in meta]),
                        ("true_label", [row["true_label"] for row in meta]),
                        ("dialogue_id", [row["dialogue_id"] for row in meta]), ("speaker_dependency", dependency)):
        for key in sorted(set(items)):
            strata[name, key] = np.asarray(items) == key
    for (group_type, group), select in {("all", "ALL"): groups["ALL"], **strata}.items():
        rows.append({"seed": seed, "row_type": "utility_summary", "group_type": group_type, "group": group,
                     "support": int(select.sum()), "helpful_rate": average(helpful[select]),
                     "harmful_rate": average(harmful[select]), "helpful_flip_rate": average(helpful_flip[select]),
                     "harmful_flip_rate": average(harmful_flip[select]), "mean_logit_delta": average(delta[select])})
        for name, data in values.items():
            rows.append({"seed": seed, "row_type": "condition_summary", "group_type": group_type,
                         "group": group, "condition": name, "support": int(select.sum()),
                         "accuracy": average(data["correct"][select]),
                         "true_class_logit_margin": average(data["true_class_logit_margin"][select]),
                         "true_class_probability_margin": average(data["true_class_probability_margin"][select])})
    contrasts = {}
    for label, left, right in (("shift_vs_stable", groups["SHIFT"], groups["STABLE"]),
                               ("same_vs_other", dependency == "SAME_FAVORED", dependency == "OTHER_FAVORED")):
        # Difference in helpful-minus-harmful balance between the named strata.
        # Require both strata to cover at least 5% of the population.
        contrasts[label] = ((helpful[left].mean() - harmful[left].mean())
                            - (helpful[right].mean() - harmful[right].mean())
                            if min(left.mean(), right.mean()) >= 0.05 else 0.0)
    return rows, [row for row in samples if row["helpful"]], [row for row in samples if row["harmful"]], {
        "helpful": average(helpful), "harmful": average(harmful), "helpful_flip": average(helpful_flip),
        "harmful_flip": average(harmful_flip), **contrasts}


def base_identity(seed, item):
    return {"seed": seed, **item}


def ambiguity_audit(seed, data, labels, meta, groups, disagreement, history_recovered=None):
    mad_quartile = quartiles(data["mad"])
    samples, summaries, contrasts = [], [], {}
    wrong = ~data["correct"]
    residual = None if history_recovered is None else wrong & ~history_recovered
    for i, item in enumerate(meta):
        samples.append({**base_identity(seed, item), "mad_quartile": scalar(mad_quartile, i),
                        "predicted_label": LABEL_VOCAB[data["prediction"][i]],
                        "correct": scalar(data["correct"], i), **{key: scalar(data[key], i) for key in
                            ("mad", "normalized_entropy", "top2_margin", "confidence")},
                        **{key: scalar(value, i) for key, value in disagreement.items()},
                        "history_recovered": scalar(history_recovered, i) if residual is not None else None,
                        "residual_error": scalar(residual, i) if residual is not None else None})
    for proxy, values in (("mad", data["mad"]), ("normalized_entropy", data["normalized_entropy"]),
                          ("top2_margin", -data["top2_margin"])):
        ranks = mad_quartile if proxy == "mad" else quartiles(values)
        low, high = ranks == 1, ranks == 4
        contrasts[proxy] = float(wrong[high].mean() - wrong[low].mean())
        if proxy == "mad":
            contrasts["ratio_gate"] = bool(wrong[high].mean() > 0 and wrong[high].mean() >= 2 * wrong[low].mean())
            contrasts["residual_gap"] = float(residual[high].mean() - residual[low].mean()) if residual is not None else None
            contrasts["q1_error"], contrasts["q4_error"] = average(wrong[low]), average(wrong[high])
            ordered = list(groups.items())
        else:
            ordered = [("ALL", groups["ALL"])]
        for q in range(1, 5):
            for group, selected in ordered:
                select = (ranks == q) & selected
                row = {"seed": seed, "proxy": proxy, "quartile": q, "group": group, **score(data, labels, select)}
                row.update({key: average(data[key][select]) for key in ("mad", "normalized_entropy", "top2_margin")})
                row.update({key: average(value[select]) for key, value in disagreement.items()})
                if residual is not None:
                    row["history_recovered_error_count"] = int(np.sum(select & wrong & history_recovered))
                    row["residual_error_rate"] = average(residual[select])
                if proxy == "mad":
                    for a, b in PAIRS:
                        for left, right in ((a, b), (b, a)):
                            left_id, right_id = LABEL_VOCAB.index(left), LABEL_VOCAB.index(right)
                            denominator = int((select & (labels == left_id)).sum())
                            count = int((select & (labels == left_id) & (data["prediction"] == right_id)).sum())
                            row[f"{left}_to_{right}_count"] = count
                            row[f"{left}_support"] = denominator
                            row[f"{left}_to_{right}_rate"] = count / denominator if denominator else None
                summaries.append(row)
    return samples, summaries, contrasts


def seed_disagreement(original_predictions):
    predictions = np.stack(original_predictions)
    frequency = (predictions[:, :, None] == np.arange(C)).mean(axis=0)
    return {"seed_predicted_class_count": (frequency > 0).sum(axis=1),
            "seed_majority_ratio": frequency.max(axis=1),
            "seed_prediction_entropy": entropy(frequency)}


def modality_group(values):
    t, a, v = (values[name]["prediction"] for name in SINGLES)
    return np.where((t == a) & (a == v), "ALL_AGREE",
                    np.where((t != a) & (a != v) & (t != v), "ALL_DIFFERENT", "TWO_TO_ONE"))


def modality_audit(seed, values, labels, meta):
    rows, summary = [], []
    full = values["FULL_TAV"]
    wrong = ~full["correct"]
    disagreement = modality_group(values)
    oracle = np.logical_or.reduce([values[name]["correct"] for name in SINGLES])
    for name in MASKS:
        data = values[name]
        rows.append({"seed": seed, "row_type": "overall", "condition": name,
                     **score(data, labels, np.ones(len(labels), dtype=bool))})
        for record in classification_report(labels, data["prediction"], loss=0)["per_class"]:
            rows.append({"seed": seed, "row_type": "per_class", "condition": name, **record})
        for i, item in enumerate(meta):
            rows.append({**base_identity(seed, item), "row_type": "sample", "condition": name,
                         "predicted_label": LABEL_VOCAB[data["prediction"][i]], "correct": scalar(data["correct"], i),
                         **{key: scalar(data[key], i) for key in
                            ("true_class_probability_margin", "true_class_logit_margin")}})
    for i, item in enumerate(meta):
        summary.append({**base_identity(seed, item), "row_type": "sample", "group": disagreement[i],
                        "full_wrong": scalar(wrong, i), "oracle_correct": scalar(oracle, i),
                        "fused_error_recovered": bool(wrong[i] and oracle[i]),
                        **{f"{name.lower()}_prediction": LABEL_VOCAB[values[name]["prediction"][i]]
                           for name in ("FULL_TAV", *SINGLES)}})
    selects = {"ALL": np.ones(len(labels), dtype=bool), "DISAGREE": disagreement != "ALL_AGREE",
               **{name: disagreement == name for name in ("ALL_AGREE", "TWO_TO_ONE", "ALL_DIFFERENT")}}
    for name, select in selects.items():
        errors = select & wrong
        summary.append({"seed": seed, "row_type": "summary", "group": name, "support": int(select.sum()),
                        "full_error_count": int(errors.sum()), "full_error_rate": average(wrong[select]),
                        "oracle_accuracy": average(oracle[select]), "recoverable_error_count": int((oracle & errors).sum()),
                        "recoverable_error_rate": average(oracle[errors])})
    agree, disagree = (average(wrong[selects[name]]) for name in ("ALL_AGREE", "DISAGREE"))
    return rows, summary, {"recoverable": average(oracle[wrong]) or 0.0, "oracle_accuracy": average(oracle),
                           "error_gap": disagree - agree if agree is not None and disagree is not None else 0.0}


def rate_signal(strong, moderate):
    return "STRONG" if strong >= MIN_SEEDS else "MODERATE" if moderate >= MIN_SEEDS else "WEAK"


def decision(per_seed, *, complete):
    rows = list(per_seed.values())
    shifts = [row["shift"] for row in rows]
    ambiguity = [row["ambiguity"] for row in rows]
    context = [row["context"] for row in rows if "context" in row]
    history = [row["history"] for row in rows if "history" in row]
    modality = [row["modality"] for row in rows if "modality" in row]
    shift_directions = sum(s["wf1_gap"] > 0 and s["error_gap"] > 0 for s in shifts)
    shift_effect = min(mean(s["wf1_gap"] for s in shifts), mean(s["error_gap"] for s in shifts))
    shift_signal = rate_signal(shift_directions if shift_effect >= .05 else 0, shift_directions)
    label_signal = rate_signal(sum(abs(s["eccs_drop_pp"]) <= 2 and s["eics_drop_pp"] >= 5
                                   and s["contrast_pp"] >= 3 for s in context),
                               sum(s["contrast_pp"] > 0 and s["eics_drop_pp"] > 0 for s in context)) if context else "PENDING"
    support = [s for s in history if min(s["helpful"], s["harmful"]) >= .10]
    differentiation = max((sum(direction * s[key] >= .05 for s in support)
                           for key in ("shift_vs_stable", "same_vs_other") for direction in (-1, 1)), default=0)
    selective_signal = rate_signal(differentiation, len(support)) if history else "PENDING"
    proxy_agreement = sum(sum(s[key] > 0 for s in ambiguity) >= MIN_SEEDS
                          for key in ("mad", "normalized_entropy", "top2_margin"))
    ambiguity_signal = rate_signal(sum(s["ratio_gate"] for s in ambiguity) if proxy_agreement >= 2 else 0,
                                   sum(s["mad"] > 0 for s in ambiguity))
    modality_signal = rate_signal(sum(s["recoverable"] >= .15 and s["error_gap"] >= .05 for s in modality),
                                  sum(s["recoverable"] > 0 and s["error_gap"] > 0 for s in modality)) if modality else (
                                      "NOT_AVAILABLE" if complete else "PENDING")
    history_total = "STRONG" if "STRONG" in (label_signal, selective_signal) else (
        "PENDING" if not complete else "MODERATE" if any(signal in ("STRONG", "MODERATE") for signal in
                                                      (shift_signal, label_signal, selective_signal)) else "WEAK")
    residual_count = sum(s["residual_gap"] is not None and s["residual_gap"] > 0 for s in ambiguity)
    eligible = {"H": history_total == "STRONG",
                "U": ambiguity_signal == "STRONG" and residual_count >= MIN_SEEDS,
                "M": modality_signal == "STRONG"}
    routes = {route: "ELIGIBLE" if value else "REJECTED" if complete else "PENDING"
              for route, value in eligible.items()}
    effects = {}
    if eligible["H"]:
        effects["H"] = [max(s["context"]["contrast_pp"] if label_signal == "STRONG" else 0,
                            100 * min(s["history"]["helpful_flip"], s["history"]["harmful_flip"])
                            if selective_signal == "STRONG" else 0) for s in rows]
    if eligible["U"]:
        effects["U"] = [100 * s["residual_gap"] for s in ambiguity]
    if eligible["M"]:
        effects["M"] = [100 * min(s["recoverable"], s["error_gap"]) for s in modality]
    priority, secondary, round_two, rationale = prioritize(effects, complete=complete)
    return {"H_SHIFT_SIGNAL": shift_signal, "LABEL_REPLICATION": label_signal,
            "SELECTIVE_HISTORY_SIGNAL": selective_signal, "AMBIGUITY_SIGNAL": ambiguity_signal,
            "MODALITY_RELIABILITY_SIGNAL": modality_signal, "history_total": history_total,
            "ambiguity_residual_seed_count": residual_count, "ambiguity_proxy_agreement": proxy_agreement,
            **{f"ROUTE_{key}": value for key, value in routes.items()}, "NEXT_ROUTE": priority,
            "secondary": secondary, "ROUND_2_REQUIRED": round_two, "effect_pp": effects, "rationale": rationale}


def prioritize(effects, *, complete):
    if not complete:
        return "UNRESOLVED", None, "PENDING", "The frozen perturbation forwards are pending."
    if not effects:
        return "UNRESOLVED", None, "NO", "No route met its full eligibility rule."
    ranked = sorted(effects, key=lambda key: (-median(effects[key]), "HUM".index(key)))
    if len(ranked) == 1:
        return ranked[0], None, "NO", "Only one route met its eligibility rule."

    def vote(left, right):
        delta = np.array(effects[left]) - np.array(effects[right])
        if np.sum(delta >= 5) >= MIN_SEEDS:
            return left
        if np.sum(delta <= -5) >= MIN_SEEDS:
            return right
        if np.sum(np.abs(delta) < 5) >= MIN_SEEDS:
            return min((left, right), key="HUM".index)
        return None

    winners = {pair: vote(*pair) for pair in combinations(ranked, 2)}
    for route in ranked:
        if all(winner == route for pair, winner in winners.items() if route in pair):
            return route, None, "NO", "The five-seed effects resolve the comparison under the 5 pp rule."
    return "UNRESOLVED", ranked[1], "YES", f"Compare {ranked[0]} and {ranked[1]} in one targeted Round 2 audit."

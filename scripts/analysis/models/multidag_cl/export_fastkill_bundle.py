"""Export matched seed100 Dev evidence and apply the preregistered route gates."""

from pathlib import Path
from shutil import copyfile

from models.multidag_cl.ta_v0 import LABEL_VOCAB
from scripts.runtime.multidag_cl_fastkill.spec import A1, ARMS, B1, CTRL, HOLD, SEED, UNIFORM_EPOCH, baseline
from scripts.runtime.multidag_cl_fusion.spec import F0_PARAMETER_COUNT
from scripts.runtime.multidag_cl_ta_v0.formal import json_identity, sha256_file, write_json_atomic
from scripts.runtime.multidag_cl_tav_full_audit_v1.spec import write_csv
from .export_fusion_topology_bundle import MANIFEST_NAME, pack_verified, read_json, read_run, write_run_tables

METRICS = ("accuracy", "weighted_f1", "macro_f1", "uar")
GATE_PAIRS = (("hap", "exc"), ("neu", "fru"))


def pp_delta(candidate, control):
    return {f"{metric}_delta_pp": 100 * (float(candidate[metric]) - float(control[metric]))
            for metric in METRICS}


def at_least(value, threshold):
    return value >= threshold - 1e-9


def pair_evidence(candidate, control):
    for first, second in GATE_PAIRS:
        i, j = LABEL_VOCAB.index(first), LABEL_VOCAB.index(second)
        counts = candidate["confusion_matrix"], control["confusion_matrix"]
        supports = [sum(counts[1][index]) for index in (i, j)]
        if min(supports) == 0:
            yield {"label_a": first, "label_b": second, "ctrl_a_to_b": None, "ctrl_b_to_a": None,
                   "b1_a_to_b": None, "b1_b_to_a": None, "a_to_b_delta_pp": None, "b_to_a_delta_pp": None,
                   "relative_mean_error_reduction": None, "min_recall_delta_pp": None, "pair_gate": False}
            continue
        (new_a, new_b), (old_a, old_b) = [
            (matrix[i][j] / supports[0], matrix[j][i] / supports[1]) for matrix in counts]
        relative = 1 - (new_a + new_b) / (old_a + old_b) if old_a + old_b else None
        deltas = (100 * (new_a - old_a), 100 * (new_b - old_b))
        recall_delta = 100 * min(candidate["per_class"][k]["recall"] - control["per_class"][k]["recall"]
                                 for k in (i, j))
        yield {"label_a": first, "label_b": second, "ctrl_a_to_b": old_a, "ctrl_b_to_a": old_b,
               "b1_a_to_b": new_a, "b1_b_to_a": new_b, "a_to_b_delta_pp": deltas[0],
               "b_to_a_delta_pp": deltas[1], "relative_mean_error_reduction": relative,
               "min_recall_delta_pp": recall_delta,
               "pair_gate": relative is not None and at_least(relative, 0.10)
               and at_least(1, max(deltas)) and at_least(recall_delta, -1)}


def route_decisions(selected, final, evidence, *, pairing_ok, suffix_ok, synthetic):
    b = pp_delta(selected[B1], selected[CTRL])
    a = pp_delta(final[A1], final[CTRL])
    pair_rows = list(pair_evidence(evidence[B1], evidence[CTRL]))
    b_pass = (at_least(b["weighted_f1_delta_pp"], 0.30) and at_least(b["macro_f1_delta_pp"], 0.20)
              and at_least(b["uar_delta_pp"], 0) and any(row["pair_gate"] for row in pair_rows))
    a_pass = (at_least(a["weighted_f1_delta_pp"], 0.30) and at_least(a["macro_f1_delta_pp"], 0)
              and at_least(a["uar_delta_pp"], 0))
    routes = {"B_ROUTE": ("KEEP_FOR_CONFIRMATION" if b_pass else "STOP") if pairing_ok else "INVALID",
              "A_ROUTE": ("KEEP_FOR_CONFIRMATION" if a_pass else "STOP") if pairing_ok and suffix_ok else "INVALID"}
    if synthetic:
        routes = dict.fromkeys(routes, "NOT_EVALUATED_SYNTHETIC")
    comparisons = [{"variant": arm, "stage": stage, **pp_delta(results[arm], results[CTRL])}
                   for stage, results in (("best_dev_checkpoint", selected), ("last_epoch", final))
                   for arm in (B1, A1)]
    return routes, comparisons, pair_rows


def export_bundle(directory, archive, state):
    directory, archive = Path(directory), Path(archive)
    jobs = state["jobs"]
    if set(jobs) != set(ARMS) or any(state.get(key) != value for key, value in HOLD.items()):
        raise ValueError("invalid Fast-Kill bundle arm or Test HOLD identity")
    directory.mkdir(parents=True, exist_ok=False)
    selected, final, reports, summaries, checkpoints, classes, budgets = {}, {}, {}, [], [], [], {}
    ctrl_result = jobs[CTRL]["result"]
    ctrl_population = None
    for arm in ARMS:
        job = jobs[arm]
        run = Path(job["run_dir"])
        result = job["result"]
        if (result["batch_pairing"] != ctrl_result["batch_pairing"]
                or result["shared_initialization"] != ctrl_result["shared_initialization"]):
            raise ValueError(f"{arm} initialization/RNG/batch tape mismatch")
        summary, checkpoint, history, predictions, population, evidence, _ = read_run(
            {**job, "variant": "F0"}, state, raw=baseline())
        if ctrl_population is not None and ctrl_population != population:
            raise ValueError(f"{arm} Dev IDs or labels differ from matched CTRL")
        ctrl_population = population
        steps, targets = [int(row["optimizer_steps"]) for row in history], [int(row["train_targets"]) for row in history]
        batches = [len(result["batch_pairing"]["train"][str(row["epoch"])]["batch_sha256"]) for row in history]
        train_population = result["batch_pairing"]["population"]["train"]
        if (steps != [sum(batches[:index]) for index in range(1, len(batches) + 1)]
                or set(targets) != {train_population["utterances"]}
                or {int(row["train_dialogues"]) for row in history} != {train_population["dialogues"]}
                or steps[-1] != result["optimizer_steps"]):
            raise ValueError(f"{arm} optimizer steps or valid Train exposure mismatch")
        budgets[arm] = (steps, targets)
        if budgets[arm] != budgets[CTRL] or not state["synthetic_only"] and (
                len(history), steps[-1], sum(targets)) != (30, 210, 144900):
            raise ValueError(f"{arm} differs from the frozen CT optimization budget")
        selected[arm] = evidence["metrics"]
        final[arm] = {key: history[-1][f"dev_{key}"] for key in METRICS}
        reports[arm] = evidence
        extra_parameters = 1800 if arm == B1 else 0
        source = {"variant": arm, "run_id": result["run_id"]}
        summary.update(source, final_optimizer_steps=steps[-1], total_train_targets=sum(targets),
                       inference_parameter_count=F0_PARAMETER_COUNT, training_auxiliary_parameters=extra_parameters,
                       training_parameter_count=F0_PARAMETER_COUNT + extra_parameters,
                       inference_parameter_overhead=0, inference_forward_changed=False)
        checkpoint.update(source, objective_config_sha256=json_identity(result["experiment"]),
                          git_head=state["git_head"])
        for row in (*history, *predictions):
            row["variant"] = arm
        write_run_tables(directory, summary, history, predictions, evidence)
        summaries.append(summary)
        checkpoints.append(checkpoint)
        classes.extend({"variant": arm, "seed": SEED, "split": "dev", **row} for row in evidence["per_class"])
        copyfile(run / "resolved_config.yaml", directory / f"resolved_config_{arm}.yaml")
        copyfile(run / "manifests/run_manifest.json", directory / f"run_manifest_{arm}.json")
        if arm != CTRL:
            auxiliary = run / "reports" / "fastkill"
            name = "train_loss_components.csv" if arm == B1 else "alpha_distribution.csv"
            if not (auxiliary / name).is_file():
                raise ValueError(f"missing {arm} objective evidence")
            for path in sorted(auxiliary.rglob("*")):
                if path.is_file():
                    target = directory / arm / path.relative_to(auxiliary)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    copyfile(path, target)
    a0 = Path(state["a0_directory"])
    prefix = read_json(a0 / "a0_manifest.json")
    suffix_ok = (prefix["PREFIX_SUFFIX_INVARIANCE"] == "PASS" and prefix["A0_DIAGNOSTIC"] == "PASS"
                 and prefix["synthetic_only"] is state["synthetic_only"]
                 and prefix["train_utterances"] == ctrl_result["batch_pairing"]["population"]["train"]["utterances"])
    if not suffix_ok:
        raise ValueError("A1 invalid: A0 suffix invariance or population evidence failed")
    (directory / "A0").mkdir()
    for path in sorted(a0.iterdir()):
        copyfile(path, directory / "A0" / path.name)
    routes, comparisons, pairs = route_decisions(selected, final, reports, pairing_ok=True,
                                                suffix_ok=suffix_ok, synthetic=state["synthetic_only"])
    write_csv(directory / "paired_dev_deltas.csv", comparisons)
    write_csv(directory / "b1_preregistered_pair_errors.csv", pairs)
    write_csv(directory / "seed100_summary.csv", summaries)
    write_csv(directory / "dev_per_class.csv", classes)
    write_json_atomic(directory / "checkpoint_identity.json", {
        "selector": baseline()["checkpoint"]["selector"], "checkpoints": checkpoints})
    write_json_atomic(directory / "test_hold_status.json", HOLD)
    write_json_atomic(directory / "route_decisions.json", routes)
    inventory = {path.relative_to(directory).as_posix(): sha256_file(path)
                 for path in sorted(directory.rglob("*")) if path.is_file()}
    write_json_atomic(directory / MANIFEST_NAME, {**HOLD, **routes, "schema": "ct_multidag_fastkill_bundle_v1",
                      "seed": SEED, "batch_id": state["batch_id"], "git_head": state["git_head"],
                      "git_dirty": state.get("git_dirty"), "synthetic_only": state["synthetic_only"],
                      "paired_optimizer_budget": True, "paired_train_dev_batch_tape": True,
                      "INFERENCE_PARAMETER_OVERHEAD": 0, "INFERENCE_FORWARD_CHANGED": "NO",
                      "PREFIX_SUFFIX_INVARIANCE": "PASS", "status": "PASS", "PLOT_READY": "YES",
                      "files_sha256": inventory})
    digest = pack_verified(directory, archive, inventory)
    return {**routes, "ANALYSIS_BUNDLE": "PASS", "PLOT_READY": "YES",
            "archive": archive.as_posix(), "archive_sha256": digest}

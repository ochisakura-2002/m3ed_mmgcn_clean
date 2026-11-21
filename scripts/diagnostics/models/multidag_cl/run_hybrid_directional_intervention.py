"""Evaluate eight directed message masks on the selected Hybrid checkpoint's Dev split."""

import argparse
from contextlib import ExitStack, contextmanager
import csv
import json
import math
import os
from pathlib import Path
import pickle
import sys

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import yaml
from torch.utils.data import DataLoader

from datasets.iemocap.tav_full_audit_dataset import (
    COUNTS, IDENTITY_FIELDS, MANIFEST_FIELDS, TAVAuditDataset, collate, identity_sha,
)
from models.multidag_cl.fusion.matching import EDGES, TextCenteredMatching
from models.multidag_cl.fusion.model import FusionTAV
from scripts.runtime.multidag_cl_fusion.runtime import aggregate_dev
from scripts.runtime.multidag_cl_fusion.spec import GROUP, make_screen_config
from scripts.runtime.multidag_cl_paper_reimplementation.checkpoint import load_checkpoint
from scripts.runtime.multidag_cl_ta_v0.formal import git_identity, json_identity, sha256_file, write_json_atomic
from scripts.runtime.multidag_cl_ta_v0.training import classification_report, masked_cross_entropy, seed_everything
from scripts.runtime.multidag_cl_tav_full_audit_v1.confirm import test_hold
from scripts.runtime.multidag_cl_tav_full_audit_v1.runtime import MODEL_INPUTS
from scripts.runtime.multidag_cl_tav_full_audit_v1.spec import write_csv


RUN_ID = "tav_track_hybrid_seed100"
BEST_EPOCH = 19
SCREEN_ROOT = Path("outputs/20261121") / GROUP
REPORT_PATH = Path("docs/research/fusion/HYBRID_DIRECTIONAL_INTERVENTION_20261121.md")
REPORT_MARKER = "<!-- hybrid-directional-results -->\n"
METRICS = ("accuracy", "weighted_f1", "macro_f1", "uar")
INTERVENTIONS = {
    "H0_FULL": (),
    "H1_DROP_T_FROM_A": ("text_from_audio",),
    "H2_DROP_T_FROM_V": ("text_from_visual",),
    "H3_DROP_A_FROM_T": ("audio_from_text",),
    "H4_DROP_V_FROM_T": ("visual_from_text",),
    "H5_DROP_VISUAL_PAIR": ("text_from_visual", "visual_from_text"),
    "H6_DROP_AUDIO_PAIR": ("text_from_audio", "audio_from_text"),
    "H7_ALL_FUSION_OFF": tuple(EDGES),
}


def zero_matcher_message(_module, _inputs, output):
    gate, message = output
    return gate, torch.zeros_like(message)


@contextmanager
def mask_messages(model, dropped):
    """Zero post-gate messages before receiver summation and residual scaling."""
    if model.training or torch.is_grad_enabled():
        raise ValueError("directed message masking requires inference mode")
    if (not isinstance(model.fusion, TextCenteredMatching)
            or set(model.fusion.matchers) != set(EDGES) or not set(dropped) <= set(EDGES)):
        raise ValueError("unexpected Hybrid matchers or intervention edges")
    with ExitStack() as hooks:
        for edge in dropped:
            hooks.enter_context(model.fusion.matchers[edge].register_forward_hook(zero_matcher_message))
        yield


def verify_state(expected, actual):
    if expected.keys() != actual.keys():
        raise ValueError("checkpoint state keys differ")
    for name, value in expected.items():
        candidate = actual[name]
        if isinstance(value, torch.Tensor):
            equal = (isinstance(candidate, torch.Tensor) and value.dtype == candidate.dtype
                     and torch.equal(value, candidate.cpu()))
        else:
            equal = value == candidate
        if not equal:
            raise ValueError(f"checkpoint state differs at {name}")


def read_run(run_dir):
    path = run_dir / "resolved_config.yaml"
    resolved = yaml.safe_load(path.read_text(encoding="utf-8"))
    path = run_dir / "manifests/run_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    test_hold(manifest)
    raw = resolved["audit_config"]
    if (raw != make_screen_config("Hybrid") or manifest["config_sha256"] != json_identity(raw)
            or manifest["run_id"] != RUN_ID or manifest["experiment_id"] != RUN_ID
            or manifest["seed"] != 100 or manifest["fusion"] != raw["fusion"]
            or manifest["status"] != "completed" or manifest["synthetic_only"] is not False
            or manifest["last_completed_epoch"] != raw["training"]["max_epochs"]
            or manifest["training_finished"] is not True or manifest["best_checkpoint_reloaded"] is not True
            or manifest["best_epoch"] != BEST_EPOCH
            or manifest["data_identity"] != resolved["data_identity"]
            or manifest["data_paths"] != resolved["data_paths"]):
        raise ValueError("expected completed formal Hybrid seed-100 run with best epoch 19")
    return resolved, manifest


def verified_weights(run_dir, resolved, manifest):
    structured_path = run_dir / "checkpoints/best_model.pt"
    weights_path = run_dir / "checkpoints/best_model_weights_only.pt"
    structured_sha = sha256_file(structured_path)
    if structured_sha != manifest["best_checkpoint_sha256"]:
        raise ValueError("best_model.pt SHA256 differs from the run manifest")
    # The screening writer hashes best_model.pt, then exports its state dict separately.
    # The structured artifact is trusted only after its manifest SHA has been checked.
    checkpoint = load_checkpoint(structured_path, torch.device("cpu"))
    expected = {
        "schema": "tav_full_audit_checkpoint_v1", "audit_config": resolved["audit_config"],
        "data_identity": resolved["data_identity"], "epoch": BEST_EPOCH, "epoch_complete": True,
        "seed": 100, "git_head": manifest["git_head"], "test_loader_built": 0, "test_evaluation_count": 0,
    }
    if any(checkpoint.get(key) != value for key, value in expected.items()):
        raise ValueError("manifest-verified checkpoint identity differs from the formal Hybrid run")
    weights_sha = sha256_file(weights_path)
    weights = torch.load(weights_path, map_location="cpu", weights_only=True)
    verify_state(checkpoint["model_state_dict"], weights)
    if sha256_file(structured_path) != structured_sha or sha256_file(weights_path) != weights_sha:
        raise ValueError("checkpoint changed during verification")
    return weights, {"best_checkpoint_sha256": structured_sha, "weights_only_sha256": weights_sha,
                     "weights_only_verified_against_manifest_checkpoint": True}


def dev_metadata_rows(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not set(MANIFEST_FIELDS) <= set(reader.fieldnames or ()):
            raise ValueError("missing canonical TAV manifest columns")
        for row in reader:
            fields = MANIFEST_FIELDS if row["split"] == "dev" else IDENTITY_FIELDS
            yield {key: row[key] for key in fields}


def dev_asset_view(asset):
    # The published pickle is a single container. Expose only identity fields
    # for other splits to the canonical dataset validator.
    view = {key: asset[key] for key in ("schema", "dataset_version", "label_vocab", "feature_spec", "splits")}
    dev_ids = set(view["splits"]["dev"])
    view["dialogues"] = {
        did: dialogue if did in dev_ids else {key: dialogue[key] for key in ("utterance_ids", "speakers")}
        for did, dialogue in asset["dialogues"].items()
    }
    return view


def load_frozen_dev(resolved):
    identity = resolved["data_identity"]
    paths = {key: Path(resolved["data_paths"][key]) for key in ("asset", "model_manifest", "provenance_manifest")}
    for key, path in paths.items():
        if sha256_file(path) != identity["sha256"][key]:
            raise ValueError(f"formal TAV package SHA256 mismatch for {key}")
    rows = list(dev_metadata_rows(paths["model_manifest"]))
    if identity_sha(rows) != identity["row_order_identity_sha256"]:
        raise ValueError("formal TAV row/order identity mismatch")
    with paths["asset"].open("rb") as stream:
        asset = dev_asset_view(pickle.load(stream))
    for key in ("asset", "model_manifest"):
        if sha256_file(paths[key]) != identity["sha256"][key]:
            raise ValueError(f"formal TAV package changed while loading {key}")
    return TAVAuditDataset(asset, rows, split="dev")


def frozen_dev_batches(dataset, raw, pairing):
    if dataset.split != "dev" or raw["training"]["dev_shuffle"]:
        raise ValueError("expected the unshuffled formal Dev dataset")
    loader = DataLoader(dataset, batch_size=raw["training"]["batch_size"], shuffle=False,
                        num_workers=0, collate_fn=collate,
                        generator=torch.Generator().manual_seed(raw["training"]["seed"]))
    batches, identities, batch_hashes = [], [], []
    for batch in loader:
        if batch["split"] != "dev":
            raise ValueError("Dev evaluation received a non-Dev batch")
        pairs = list(zip(batch["dialogue_ids"], batch["utterance_ids"]))
        identities.extend(pairs)
        batch_hashes.append(json_identity(pairs))
        batches.append(batch)
    utterances = [uid for _, ids in identities for uid in ids]
    counts = {"dialogues": len(identities), "utterances": len(utterances)}
    actual = {"dialogue_order_sha256": json_identity(identities), "batch_sha256": batch_hashes,
              "batch_sequence_sha256": json_identity(batch_hashes)}
    if (counts != COUNTS["dev"] or len({did for did, _ in identities}) != counts["dialogues"]
            or len(set(utterances)) != counts["utterances"]
            or pairing["population"]["dev"] != {"sha256": json_identity(identities), **counts}
            or pairing["dev"] != actual):
        raise ValueError("Dev population or batch order differs from the formal run manifest")
    return batches


@torch.inference_mode()
def evaluate_masks(model, batches, *, device):
    reports = {}
    for name, dropped in INTERVENTIONS.items():
        labels, predictions, traces, total_loss = [], [], {}, 0.0
        with mask_messages(model, dropped):
            for batch in batches:
                if batch["split"] != "dev":
                    raise ValueError("non-Dev evaluation is forbidden")
                inputs = {key: batch[key].to(device) for key in MODEL_INPUTS}
                output = model(**inputs)
                if not torch.equal(output.mask, inputs["attention_mask"]):
                    raise ValueError("Hybrid changed the valid Dev positions")
                targets = batch["labels"].to(device)
                loss, count = masked_cross_entropy(output.logits_ta, targets, output.mask)
                total_loss += float(loss) * count
                labels.extend(targets[output.mask].tolist())
                predictions.extend(output.logits_ta[output.mask].argmax(-1).tolist())
                for key, tensor in output.fusion_diagnostics.items():
                    traces.setdefault(key, []).append(tensor.cpu())
        if not labels:
            raise ValueError("empty Dev evaluation")
        report = classification_report(labels, predictions, loss=total_loss / len(labels))
        mechanism = aggregate_dev(traces, [], by_class=False)
        for edge, (_, _, label) in EDGES.items():
            stats = mechanism[label]
            if (stats["active_count"] != len(labels)
                    or edge in dropped and stats["message_receiver_ratio_mean"] != 0.0
                    or not all(math.isfinite(float(value)) for value in stats.values())):
                raise ValueError(f"unexpected message trace for {name} {label}")
        activity = mechanism["activity"]
        if (activity["active_count"] != len(labels) * 3
                or not math.isfinite(activity["residual_receiver_ratio_median"])
                or name == "H7_ALL_FUSION_OFF" and activity["residual_receiver_ratio_median"] != 0.0):
            raise ValueError(f"unexpected applied residual trace for {name}")
        reports[name] = {**report, "fusion_mechanism": mechanism}
    return reports


def result_rows(reports):
    full = reports["H0_FULL"]
    summary, per_class = [], []
    for name, report in reports.items():
        dropped = INTERVENTIONS[name]
        summary.append({"intervention": name, "dropped_edges": ";".join(EDGES[edge][2] for edge in dropped),
                        **{f"{key}_pct": 100 * report["metrics"][key] for key in METRICS},
                        **{f"delta_{key}_pp": 100 * (report["metrics"][key] - full["metrics"][key])
                           for key in METRICS}})
        for actual, baseline in zip(report["per_class"], full["per_class"]):
            per_class.append({"intervention": name, "label": actual["label"], "support": actual["support"],
                              **{f"{key}_pct": 100 * actual[key] for key in ("precision", "recall", "f1")},
                              **{f"delta_{key}_pp": 100 * (actual[key] - baseline[key]) for key in ("f1", "recall")}})
    return summary, per_class


def render_report(summary, per_class, original_mechanism, full_mechanism, weights_sha):
    lines = [f"已完成冻结 Dev 推断。weights-only SHA-256 为 `{weights_sha}`。", "",
             "| Intervention | Acc % | WF1 % | Macro % | UAR % | ΔAcc | ΔWF1 | ΔMacro | ΔUAR |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in summary:
        scores = " | ".join(f"{row[f'{key}_pct']:.4f}" for key in METRICS)
        deltas = " | ".join(f"{row[f'delta_{key}_pp']:+.4f}" for key in METRICS)
        lines.append(f"| {row['intervention']} | {scores} | {deltas} |")
    lines += ["", "| Epoch 19 logger 读数 | 正式日志 | H0 重算 |", "| --- | ---: | ---: |"]
    for _, _, edge in EDGES.values():
        key = "message_receiver_ratio_mean"
        lines.append(f"| `{edge}` message/receiver | {original_mechanism[edge][key]:.6f} | "
                     f"{full_mechanism[edge][key]:.6f} |")
    key = "residual_receiver_ratio_median"
    lines.append(f"| applied residual/receiver | {original_mechanism['activity'][key]:.6f} | "
                 f"{full_mechanism['activity'][key]:.6f} |")

    lines += ["", "每条边均以移除后 ΔWF1 的方向判断当前 checkpoint 的影响，类别指标可显示其中的取舍。", ""]
    for row in summary[1:5]:
        delta = row["delta_weighted_f1_pp"]
        if delta < 0:
            result = "移除后 WF1 下降，提示该边有正贡献"
        elif delta > 0:
            result = "移除后 WF1 上升，提示该边可能存在冗余或冲突"
        else:
            result = "移除后 WF1 持平"
        lines.append(f"- `{row['dropped_edges']}` {result}（ΔWF1 {delta:+.4f} pp）。")
    rows = {row["intervention"]: row for row in summary}
    h2_max = max(abs(rows["H2_DROP_T_FROM_V"][f"delta_{key}_pp"]) for key in METRICS)
    lines.append(f"- H2 的四项汇总变化中，绝对值最大为 {h2_max:.4f} pp。")
    classes = {(row["intervention"], row["label"]): row for row in per_class}
    recovery = [classes["H4_DROP_V_FROM_T", label] for label in ("fru", "sad")]
    changes = "，".join(f"{row['label']} ΔF1 {row['delta_f1_pp']:+.4f} pp、"
                       f"Δrecall {row['delta_recall_pp']:+.4f} pp" for row in recovery)
    recovered = rows["H4_DROP_V_FROM_T"]["delta_weighted_f1_pp"] > 0 and all(
        row["delta_f1_pp"] > 0 and row["delta_recall_pp"] > 0 for row in recovery)
    interpretation = "符合 T → V 冗余或类别冲突的预期" if recovered else "WF1、fru 和 sad 的改善条件未同时满足"
    lines.append(f"- H4 的类别变化为 {changes}。{interpretation}。")
    visual = rows["H5_DROP_VISUAL_PAIR"]["delta_weighted_f1_pp"]
    interpretation = "支持下一版优先验证精简视觉交互" if visual > 0 else "本轮 WF1 未支持同时移除视觉双向边"
    lines.append(f"- H5 的 ΔWF1 为 {visual:+.4f} pp。{interpretation}。")
    lines += ["", "| Intervention | 类别 | Precision % | Recall % | F1 % | Support | ΔF1 | ΔRecall |",
              "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in per_class:
        lines.append(f"| {row['intervention']} | {row['label']} | {row['precision_pct']:.4f} | "
                     f"{row['recall_pct']:.4f} | {row['f1_pct']:.4f} | {row['support']} | "
                     f"{row['delta_f1_pp']:+.4f} | {row['delta_recall_pp']:+.4f} |")
    return "\n".join(lines) + "\n"


def run(run_dir, output_dir, *, update_report):
    if output_dir.exists():
        raise FileExistsError(output_dir)
    report_path = ROOT / REPORT_PATH
    if update_report:
        report_prefix, marker, _ = report_path.read_text(encoding="utf-8").partition(REPORT_MARKER)
        if not marker:
            raise ValueError("formal report is missing its results marker")
    resolved, manifest = read_run(run_dir)
    weights, checkpoint_identity = verified_weights(run_dir, resolved, manifest)
    raw = resolved["audit_config"]
    batches = frozen_dev_batches(load_frozen_dev(resolved), raw, manifest["batch_pairing"])
    mechanism_path = run_dir / "logs" / f"fusion_mechanism_epoch_{BEST_EPOCH:02d}.json"
    original_mechanism = json.loads(mechanism_path.read_text(encoding="utf-8"))
    device = torch.device("cuda")
    seed_everything(device, raw["training"]["seed"])
    model = FusionTAV(raw)
    model.load_state_dict(weights, strict=True)
    model.to(device).eval()
    reports = evaluate_masks(model, batches, device=device)
    verify_state(weights, model.state_dict())
    full_report = reports["H0_FULL"]
    original_metrics = manifest["best_dev_metrics"]
    if (full_report["metrics"]["valid_target_count"] != COUNTS["dev"]["utterances"]
            or original_metrics["valid_target_count"] != COUNTS["dev"]["utterances"]
            or any(not math.isclose(full_report["metrics"][key], original_metrics[key], rel_tol=0, abs_tol=1e-12)
                   for key in METRICS)):
        raise ValueError("H0_FULL does not reproduce the four formal Hybrid Dev metrics")

    summary, per_class = result_rows(reports)
    rendered = render_report(summary, per_class, original_mechanism, full_report["fusion_mechanism"],
                             checkpoint_identity["weights_only_sha256"])
    result = {
        "schema": "ct_multidag_hybrid_directional_intervention_v1", "status": "completed",
        "source_run": run_dir.as_posix(), "run_id": RUN_ID, "source_git_head": manifest["git_head"],
        **git_identity(), "config_sha256": manifest["config_sha256"], **checkpoint_identity,
        "best_epoch": BEST_EPOCH, "seed": 100, "split": "dev", "population": COUNTS["dev"],
        "data_identity": resolved["data_identity"], "dev_batch_pairing": manifest["batch_pairing"]["dev"],
        "scope": "within_checkpoint_mechanism_intervention", "h0_matches_formal_metrics": True,
        "device": torch.cuda.get_device_name(device), "torch_version": str(torch.__version__),
        "formal_train_runs": 0, "optimizer_steps": 0, "test_policy": "HOLD", "test_loader_built": 0,
        "test_metrics_computed": 0, "test_labels_inspected": 0, "test_features_inspected": 0,
        "metrics_csv_units": "percent and percentage-point deltas against H0_FULL",
        "formal_best_dev_metrics": original_metrics, "formal_epoch_19_mechanism": original_mechanism,
        "interventions": {name: {"dropped_edges": [EDGES[edge][2] for edge in dropped],
                                 **reports[name]} for name, dropped in INTERVENTIONS.items()},
        "report": REPORT_PATH.as_posix() if update_report else None,
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    write_csv(output_dir / "hybrid_directional_summary.csv", summary)
    write_csv(output_dir / "hybrid_directional_per_class.csv", per_class)
    if update_report:
        temporary = report_path.with_suffix(".md.tmp")
        with temporary.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(report_prefix + REPORT_MARKER + "\n" + rendered)
        temporary.replace(report_path)
    write_json_atomic(output_dir / "hybrid_directional_manifest.json", result)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=SCREEN_ROOT / "runs" / RUN_ID)
    parser.add_argument("--output-dir", type=Path, default=SCREEN_ROOT / "analysis/hybrid_directional_intervention")
    parser.add_argument("--update-report", action="store_true", help="fill the tracked Markdown report with Dev results")
    args = parser.parse_args(argv)
    if os.name == "nt" or not torch.cuda.is_available():
        parser.error("formal Dev evaluation requires remote Ubuntu CUDA")
    run(args.run_dir, args.output_dir, update_report=args.update_report)
    print(f"HYBRID_DIRECTIONAL_INTERVENTION = COMPLETE\nFORMAL_TRAIN_RUNS = 0\nTEST_TOUCHED = NO\n"
          f"OUTPUT = {args.output_dir.as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

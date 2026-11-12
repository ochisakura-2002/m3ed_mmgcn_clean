"""Dev prediction CSV and contextual preclassifier NPZ, joined by exact utterance ID."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import yaml

from models.multidag_cl.ta_v0 import LABEL_VOCAB
from models.multidag_cl.ta_v0.causal_history_gate import collect_gate_diagnostics
from scripts.models.multidag_cl.ta_v0.train import DEFAULT_CONFIG, synthetic_runner, validate_config

PREDICTION_COLUMNS = (
    "utterance_id", "true_label", "predicted_label",
    *(f"logit_{i}" for i in range(6)), *(f"prob_{i}" for i in range(6)),
)


def export_dev(runner, batches, output_dir, *, expected_ids, checkpoint_identity,
               config_identity, expected_rows=980, split="dev", formal_layout=False):
    """Validate the entire export before writing; exclude padding, sort by exact ID.

    Real Dev callers must supply the authoritative 980-ID set. Synthetic callers
    explicitly override expected_rows. No inference from observed counts.
    """
    if split != "dev":
        raise ValueError("Dev exporter only; Test predictions forbidden")
    if (not isinstance(checkpoint_identity, str) or not checkpoint_identity.strip()
            or not isinstance(config_identity, str) or not config_identity.strip()):
        raise ValueError("checkpoint and config identity are required")
    expected_ids = tuple(expected_ids)
    if not expected_ids or len(set(expected_ids)) != len(expected_ids):
        raise ValueError("expected Dev IDs must be nonempty and unique")
    if expected_rows is not None and len(expected_ids) != expected_rows:
        raise ValueError("expected Dev row count mismatch")
    records = {}
    was_training = runner.model.training
    try:
        with collect_gate_diagnostics(runner.model) as diagnostics, torch.no_grad():
            for batch in batches:
                output, _ = runner.forward_batch(batch, split="dev")
                ids_by_dialogue = batch["utterance_ids"]
                if len(ids_by_dialogue) != output.mask.shape[0]:
                    raise ValueError("batch ID shape mismatch")
                for b, ids in enumerate(ids_by_dialogue):
                    n = int(output.mask[b].sum())
                    if len(ids) != n:
                        raise ValueError("ID length/padding mismatch")
                    for t, uid in enumerate(ids):
                        if not isinstance(uid, str) or not uid or uid in records:
                            raise ValueError("duplicate/invalid Dev utterance ID")
                        label = int(batch["labels"][b, t])
                        records[uid] = (
                            label, output.logits_ta[b, t].cpu().numpy().copy(),
                            output.prob_ta[b, t].cpu().numpy().copy(),
                            output.h_ta[b, t].cpu().numpy().copy(),
                        )
    finally:
        runner.model.train(was_training)
    if set(records) != set(expected_ids):
        raise ValueError("prediction/representation/authoritative Dev ID sets differ")
    ids = sorted(records)
    labels = np.array([records[uid][0] for uid in ids], dtype=np.int64)
    logits = np.stack([records[uid][1] for uid in ids])
    probabilities = np.stack([records[uid][2] for uid in ids])
    h_ta = np.stack([records[uid][3] for uid in ids])
    if (h_ta.shape != (len(ids), 1500) or logits.shape != (len(ids), 6)
            or probabilities.shape != logits.shape
            or not all(np.isfinite(x).all() for x in (logits, probabilities, h_ta))
            or not np.allclose(probabilities.sum(-1), 1, atol=1e-6)
            or np.any(probabilities < 0)):
        raise ValueError("nonfinite/invalid TA output contract")
    output_dir = Path(output_dir)
    if formal_layout:
        prediction_dir, representation_dir = output_dir / "predictions", output_dir / "representations"
        # Validate both destinations before creating either; a formal run owns
        # its parent, but each export subtree must still be fresh.
        if prediction_dir.exists() or representation_dir.exists():
            raise FileExistsError("Dev export destination already exists")
        prediction_dir.mkdir(parents=True, exist_ok=False)
        representation_dir.mkdir(exist_ok=False)
        prediction_path = prediction_dir / "validation_best_predictions.tsv"
        representation_path = representation_dir / "validation_best_h_ta.npz"
        metadata_path = representation_dir / "validation_best_h_ta.metadata.json"
    else:
        output_dir.mkdir(parents=True, exist_ok=False)
        prediction_path = output_dir / "predictions.csv"
        representation_path = output_dir / "h_ta.npz"
        metadata_path = output_dir / "metadata.json"
    with prediction_path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t" if formal_layout else ",")
        writer.writerow(PREDICTION_COLUMNS)
        for i, uid in enumerate(ids):
            writer.writerow((uid, int(labels[i]), int(logits[i].argmax()),
                             *logits[i].tolist(), *probabilities[i].tolist()))
    np.savez_compressed(
        representation_path, utterance_ids=np.array(ids), h_ta=h_ta,
        labels=labels, split=np.array(["dev"] * len(ids)),
    )
    metadata = {
        "representation_name": "ta_v0_contextual_preclassifier", "dimension": 1500,
        "label_vocab": list(LABEL_VOCAB), "checkpoint_identity": checkpoint_identity,
        "config_identity": config_identity, "split": "dev", "rows": len(ids),
        "unique_utterance_ids": len(ids), "padding_rows": 0, "duplicate_ids": 0,
        "prediction_representation_id_set_identical": True,
        "prediction_representation_order_identical": True,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    if runner.model.config.use_causal_history_gate:
        (output_dir / "chg_gate_diagnostics.json").write_text(json.dumps(
            {"split": "dev", "checkpoint_identity": checkpoint_identity,
             "config_identity": config_identity, "groups": diagnostics.summary()},
            indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return metadata


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--synthetic-dry-run", action="store_true")
    parser.add_argument("--output-dir", required=True, help="fresh synthetic export directory")
    args = parser.parse_args(argv)
    if not args.synthetic_dry_run:
        parser.error("this stage requires --synthetic-dry-run; real checkpoint export is not implemented")
    path = Path(args.config)
    config = validate_config(yaml.safe_load(path.read_text(encoding="utf-8")))
    torch.set_num_threads(1)
    runner = synthetic_runner(config)
    batches = list(runner.loader("dev"))
    ids = [uid for batch in batches for dialogue in batch["utterance_ids"] for uid in dialogue]
    result = export_dev(
        runner, reversed(batches), args.output_dir, expected_ids=ids, expected_rows=len(ids),
        checkpoint_identity="synthetic-untrained-seed100",
        config_identity=hashlib.sha256(path.read_bytes()).hexdigest(),
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

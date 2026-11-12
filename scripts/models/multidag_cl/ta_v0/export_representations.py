"""Read-only frozen TA-v0 seed100 best-epoch18 Train/Dev representations.

Unlike export_dev.py's legacy sorted-ID files, these files preserve dataset
dialogue/utterance order. Fusion consumers must still join by exact utterance_id.
Only trusted structured checkpoints and the existing portable TA package are used.
"""

import argparse
import csv
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader

from datasets.iemocap.ta_v0_dataset import ta_v0_collate
from models.multidag_cl.ta_v0 import TAV0, TAV0Config
from scripts.models.multidag_cl.ta_v0.train import MODEL_INPUTS
from scripts.runtime.multidag_cl_ta_v0.formal import (
    SPLIT_COUNTS, git_identity, load_train_dev, sha256_file,
)
from scripts.runtime.multidag_cl_ta_v0.training import (
    checkpoint_identity, reload_model_checkpoint,
)

PREDICTION_COLUMNS = (
    "utterance_id", "dialogue_id", "true_label", "pred_label",
    *(f"prob_{i}" for i in range(6)),
)
BEST_EPOCH = 18


def normalize_split(split):
    if split not in ("train", "dev", "validation"):
        raise ValueError("only train/dev/validation allowed; Test access forbidden")
    return "dev" if split == "validation" else split


def load_frozen_model(checkpoint_path, *, data_identity, device="cpu"):
    """Reuse the formal strict reload path without constructing an optimizer."""
    model = TAV0(TAV0Config())  # OFF CHG preserves the frozen pre-CHG identity.

    def validate(checkpoint):
        for key, expected in checkpoint_identity(model.config, data_identity).items():
            if checkpoint.get(key) != expected:
                raise ValueError(f"frozen checkpoint {key} identity mismatch")
        selector = checkpoint.get("selector", {})
        best = selector.get("best") or {}
        if (checkpoint.get("epoch") != BEST_EPOCH
                or best.get("epoch") != BEST_EPOCH
                or checkpoint.get("epoch_complete") is not True
                or best.get("val_weighted_f1") != checkpoint.get("best_dev_weighted_f1")
                or best.get("val_loss") != checkpoint.get("best_dev_loss")):
            raise ValueError("frozen seed100 best checkpoint must be completed epoch 18")
        if (checkpoint.get("test_loader_built") != 0
                or checkpoint.get("test_evaluation_count") != 0
                or selector.get("test_split_used_for_selection") is not False):
            raise ValueError("frozen checkpoint Test isolation mismatch")

    checkpoint = reload_model_checkpoint(model, checkpoint_path, validate=validate)
    model.to(torch.device(device)).eval().requires_grad_(False)
    return model, checkpoint


def export_representations(model, dataset, output_dir, *, split="dev",
                           expected_rows=None, provenance):
    """Validate before writing; synthetic callers explicitly override row count."""
    split = normalize_split(split)  # Reject Test before any dataset access.
    if dataset.split != split:
        raise ValueError("dataset split identity mismatch")
    if model.config != TAV0Config():
        raise ValueError("representation export requires frozen TA-v0 with CHG OFF")
    expected_rows = SPLIT_COUNTS[split]["utterances"] if expected_rows is None else expected_rows
    device = next(model.parameters()).device
    # Training loaders shuffle and forward_batch(train) enables dropout. Reuse
    # the dataset/collator with a sequential loader and direct eval-only forward.
    batches = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0,
                         collate_fn=ta_v0_collate)
    ids, dialogue_ids, labels, features, logits, probabilities = [], [], [], [], [], []
    seen = set()
    was_training = model.training
    try:
        model.eval()
        with torch.inference_mode():
            for batch in batches:
                if batch["split"] != split:
                    raise ValueError("batch split identity mismatch")
                output = model(**{key: batch[key].to(device) for key in MODEL_INPUTS})
                for b, utterances in enumerate(batch["utterance_ids"]):
                    n = int(output.mask[b].sum())
                    if len(utterances) != n:
                        raise ValueError("ID length/padding mismatch")
                    did = batch["dialogue_ids"][b]
                    if not isinstance(did, str) or not did.strip():
                        raise ValueError("empty/invalid dialogue_id")
                    for t, uid in enumerate(utterances):
                        if not isinstance(uid, str) or not uid.strip() or uid in seen:
                            raise ValueError("duplicate/empty/invalid utterance_id")
                        seen.add(uid)
                        ids.append(uid)
                        dialogue_ids.append(did)
                        labels.append(int(batch["labels"][b, t]))
                    features.append(output.h_ta[b, :n].cpu().numpy().copy())
                    logits.append(output.logits_ta[b, :n].cpu().numpy().copy())
                    probabilities.append(output.prob_ta[b, :n].cpu().numpy().copy())
    finally:
        model.train(was_training)
    if not ids or len(ids) != expected_rows:
        raise ValueError(f"{split} row count mismatch: observed={len(ids)}, expected={expected_rows}")
    arrays = {
        "utterance_id": np.asarray(ids), "dialogue_id": np.asarray(dialogue_ids),
        "label": np.asarray(labels, dtype=np.int64), "h_ta": np.concatenate(features),
        "logits_ta": np.concatenate(logits), "prob_ta": np.concatenate(probabilities),
    }
    h_ta, logits, probabilities = (arrays[k] for k in ("h_ta", "logits_ta", "prob_ta"))
    if (h_ta.shape != (len(ids), 1500) or logits.shape != (len(ids), 6)
            or probabilities.shape != logits.shape
            or np.any((arrays["label"] < 0) | (arrays["label"] > 5))
            or not all(np.isfinite(x).all() for x in (h_ta, logits, probabilities))
            or np.any((probabilities < 0) | (probabilities > 1))
            or not np.allclose(probabilities.sum(-1), 1, rtol=0, atol=1e-6)):
        raise ValueError("invalid labels or nonfinite/invalid TA output contract")
    predictions = logits.argmax(-1)
    if not np.array_equal(probabilities.argmax(-1), predictions):
        raise ValueError("argmax(prob_ta) differs from pred_label")
    prefix = "train" if split == "train" else "validation"
    manifest = dict(
        provenance, split=split, row_count=len(ids), h_ta_dim=1500, num_classes=6,
        order="dataset.dialogue_ids then canonical utterance order",
        test_access=False,
    )
    # Serialize provenance before creating any destination.
    manifest_json = json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output_dir / f"{prefix}_best_h_ta.npz", **arrays)
    with (output_dir / f"{prefix}_best_predictions.tsv").open(
            "x", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t")
        writer.writerow(PREDICTION_COLUMNS)
        for i, uid in enumerate(ids):
            writer.writerow((uid, dialogue_ids[i], labels[i], predictions[i],
                             *probabilities[i].tolist()))
    (output_dir / "representation_export_manifest.json").write_text(
        manifest_json, encoding="utf-8")
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("train", "dev", "validation"), default="dev")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--run-dir", help="frozen run; reads checkpoints/best_model.pt")
    source.add_argument("--checkpoint", help="explicit trusted structured best checkpoint")
    parser.add_argument("--asset-path", required=True)
    parser.add_argument("--model-manifest-path", required=True)
    parser.add_argument("--provenance-manifest-path", required=True)
    parser.add_argument("--output-dir", required=True, help="fresh directory for this split")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args(argv)
    split = normalize_split(args.split)
    checkpoint_path = (Path(args.checkpoint) if args.checkpoint else
                       Path(args.run_dir) / "checkpoints" / "best_model.pt")
    datasets, data_identity, _ = load_train_dev(
        args.asset_path, args.model_manifest_path, args.provenance_manifest_path)
    checkpoint_sha256 = sha256_file(checkpoint_path)
    model, checkpoint = load_frozen_model(
        checkpoint_path, data_identity=data_identity, device=args.device)
    if sha256_file(checkpoint_path) != checkpoint_sha256:
        raise ValueError("checkpoint changed while loading")
    git = git_identity()
    manifest = export_representations(
        model, datasets[split], args.output_dir, split=split,
        provenance={
            "checkpoint_path": checkpoint_path.as_posix(),
            "checkpoint_sha256": checkpoint_sha256,
            "checkpoint_best_epoch": checkpoint["epoch"], "seed": checkpoint["seed"],
            "git_commit": git["git_head"], "git_dirty": git["git_dirty"],
            "checkpoint_git_commit": checkpoint.get("git_head"),
            "data_identity": data_identity,
        },
    )
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

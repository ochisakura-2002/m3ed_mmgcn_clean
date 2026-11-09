"""Train/dev development interface. This stage permits synthetic CPU dry-runs only."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import yaml

from datasets.iemocap.ta_v0_dataset import (
    SCHEMA, VERSION, TAV0Dataset, ta_v0_collate,
)
from models.multidag_cl.ta_v0 import LABEL_VOCAB, TAV0, TAV0Config

DEFAULT_CONFIG = "configs/multidag_cl/ta_v0/iemocap/causal_context/asymmetric_merc_v1/baseline.yaml"
MODEL_INPUTS = ("text_features", "audio_features", "speaker_ids_int", "lengths", "attention_mask")
BATCH_FIELDS = set(MODEL_INPUTS) | {"labels", "utterance_ids", "dialogue_ids", "split"}


def validate_config(raw):
    if set(raw) != {"model", "dataset", "runtime", "checkpoint", "output"}:
        raise ValueError("unexpected/missing TA-v0 config sections")
    config = TAV0Config(**raw["model"])
    if raw["dataset"] != {
        "package": VERSION, "label_vocab": list(LABEL_VOCAB),
        "asset_path": None, "bridge_path": None, "canonical_path": None,
    }:
        raise ValueError("this stage freezes dataset identity and unresolved real-data paths")
    if raw["runtime"] != {
        "mode": "development", "synthetic_only": True, "device": "cpu",
        "splits": ["train", "dev"], "curriculum": False, "seed": 100,
    }:
        raise ValueError("runtime must be seed100 CPU synthetic train/dev with curriculum OFF")
    if raw["checkpoint"] != {
        "selection_metric": "dev_weighted_f1", "test_evaluation_count": 0,
        "test_split_used_for_selection": False,
    }:
        raise ValueError("checkpoint selection must use Dev Weighted-F1 with zero Test evaluations")
    # Exact types avoid bool/int aliases in the frozen protocol.
    if (type(raw["runtime"]["seed"]) is not int
            or raw["runtime"]["curriculum"] is not False
            or raw["runtime"]["synthetic_only"] is not True
            or type(raw["checkpoint"]["test_evaluation_count"]) is not int
            or raw["checkpoint"]["test_split_used_for_selection"] is not False):
        raise ValueError("protocol flags have invalid types")
    if raw["output"] != {"root": "outputs", "experiment_date": None, "experiment_group": "ta_v0_development"}:
        raise ValueError("unexpected output layout")
    return config


def require_development_split(split):
    if split not in ("train", "dev"):
        raise ValueError("development allows only train/dev; Test loader forbidden")


class DevelopmentRunner:
    """Lazy split boundary; no Test loader is stored or constructed."""

    def __init__(self, model, dataset_factory, *, batch_size=2):
        if any(p.device.type != "cpu" for p in model.parameters()):
            raise ValueError("TA-v0 development is CPU only")
        self.model = model
        self.dataset_factory = dataset_factory
        self.batch_size = batch_size

    def loader(self, split):
        require_development_split(split)
        dataset = self.dataset_factory(split)
        if dataset.split != split:
            raise ValueError("dataset split identity mismatch")
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=split == "train",
                          collate_fn=ta_v0_collate, num_workers=0)

    def forward_batch(self, batch, *, split):
        require_development_split(split)
        if set(batch) != BATCH_FIELDS or batch["split"] != split:
            raise ValueError("TA-only batch fields/split identity mismatch")
        if any(batch[k].device.type != "cpu" for k in MODEL_INPUTS + ("labels",)):
            raise ValueError("TA-v0 development is CPU only")
        self.model.train(split == "train")
        output = self.model(**{k: batch[k] for k in MODEL_INPUTS})
        labels = batch["labels"]
        if (labels.dtype != torch.int64 or labels.shape != output.mask.shape
                or not torch.all(labels[~output.mask] == -100)
                or not torch.all((labels[output.mask] >= 0) & (labels[output.mask] < 6))):
            raise ValueError("invalid labels/padding")
        return output, F.cross_entropy(output.logits_ta[output.mask], labels[output.mask])


@dataclass
class DevCheckpointSelector:
    """In-memory selection contract; checkpoint persistence is a future runtime step."""

    best_score: float = -math.inf
    identity: str | None = None

    def consider(self, *, split, dev_weighted_f1, checkpoint_identity):
        if split != "dev":
            raise ValueError("checkpoint selection requires dev")
        if not math.isfinite(dev_weighted_f1) or not 0 <= dev_weighted_f1 <= 1:
            raise ValueError("Dev Weighted-F1 must be finite in [0,1]")
        if not isinstance(checkpoint_identity, str) or not checkpoint_identity.strip():
            raise ValueError("checkpoint identity required")
        if dev_weighted_f1 > self.best_score:
            self.best_score, self.identity = dev_weighted_f1, checkpoint_identity
            return True
        return False


def weighted_f1(labels, predictions):
    confusion = torch.bincount(labels * 6 + predictions, minlength=36).reshape(6, 6).double()
    support = confusion.sum(1)
    denom = support + confusion.sum(0)
    f1 = torch.where(denom > 0, 2 * confusion.diag() / denom.clamp_min(1), 0)
    return float((f1 * support).sum() / support.sum())


def synthetic_package():
    """Deliberately scrambled storage; Test has metadata only and is never loaded."""
    rng = np.random.default_rng(100)
    dialogues, bridge, canonical = {}, [], []
    split_ids = {"train": [], "dev": [], "test": []}
    for number, (split, n) in enumerate((("train", 3), ("train", 2), ("dev", 3), ("dev", 1), ("test", 1)), 1):
        did = f"Ses01F_impro{number:02d}"
        split_ids[split].append(did)
        canonical_ids = [f"{did}_{'F' if i % 2 == 0 else 'M'}{i:03d}" for i in range(n)]
        permutation = list(reversed(range(n)))
        ids = [canonical_ids[i] for i in permutation]
        labels = [i % 6 for i in permutation]
        dialogues[did] = {
            "utterance_ids": ids, "speakers": [uid.rsplit("_", 1)[1][0] for uid in ids],
            "benchmark_labels": [LABEL_VOCAB[i] for i in labels], "benchmark_label_ids": labels,
            "sentences": [f"synthetic {uid}" for uid in ids],
            "text_features": rng.normal(size=(n, 768)).astype(np.float32),
            "audio_features": rng.normal(size=(n, 1582)).astype(np.float32),
        }
        for row_index, canonical_index in enumerate(permutation):
            uid = ids[row_index]
            speaker = dialogues[did]["speakers"][row_index]
            bridge.append({
                "dataset_version": VERSION, "utterance_id": uid, "dialogue_id": did,
                "ta_dialogue_id": did, "speaker": speaker, "split": split,
                "benchmark_label": LABEL_VOCAB[labels[row_index]], "benchmark_label_id": labels[row_index],
                "ta_row_index": row_index, "canonical_dialogue_order": canonical_index + 1,
            })
            canonical.append({"utterance_id": uid, "dialogue_id": did, "speaker": speaker,
                              "global_order": canonical_index + 1})
    asset = {"schema": SCHEMA, "dataset_version": VERSION,
             "feature_spec": {"text": {"dimension": 768}, "audio_ta": {"dimension": 1582}},
             "label_vocab": {"itos": list(LABEL_VOCAB)}, "dialogues": dialogues, "splits": split_ids}
    return asset, bridge, canonical


def synthetic_runner(config=None):
    torch.manual_seed(100)
    package = synthetic_package()
    return DevelopmentRunner(
        TAV0(config), lambda split: TAV0Dataset(*package, split=split),
    )


def synthetic_dry_run(config):
    runner = synthetic_runner(config)
    train = next(iter(runner.loader("train")))
    output, loss = runner.forward_batch(train, split="train")
    loss.backward()  # Synthetic gradient contract only; no optimizer or checkpoint write.
    runner.model.zero_grad(set_to_none=True)
    labels, predictions = [], []
    with torch.no_grad():
        for batch in runner.loader("dev"):
            dev, _ = runner.forward_batch(batch, split="dev")
            labels.append(batch["labels"][dev.mask])
            predictions.append(dev.logits_ta[dev.mask].argmax(-1))
    score = weighted_f1(torch.cat(labels), torch.cat(predictions))
    selector = DevCheckpointSelector()
    selector.consider(split="dev", dev_weighted_f1=score, checkpoint_identity="synthetic-untrained-seed100")
    return {"status": "PASS", "device": "cpu", "seed": 100, "synthetic_only": True,
            "h_ta_shape": list(output.h_ta.shape), "train_loss": float(loss.detach()),
            "dev_weighted_f1": score, "test_evaluation_count": 0, "optimizer_steps": 0,
            "training_started": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--synthetic-dry-run", action="store_true")
    args = parser.parse_args(argv)
    if not args.synthetic_dry_run:
        parser.error("this stage requires --synthetic-dry-run; real training is not implemented")
    raw = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    config = validate_config(raw)
    torch.set_num_threads(1)
    print(json.dumps(synthetic_dry_run(config), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

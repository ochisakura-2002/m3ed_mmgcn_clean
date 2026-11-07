"""Export frozen S2D Uniform16 Train/Dev features; never train or export Test."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile


PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch.utils.data import DataLoader

from models.s2d.iemocap.config import load_config, resolve_path
from models.s2d.iemocap.features import extract_uniform16_feature_bundle
from models.s2d.iemocap.model import build_s2d
from scripts.models.s2d.iemocap.checkpoints import reload_model_weights


ALLOWED_SPLITS = frozenset({"train", "dev"})
SCHEMA_VERSION = "s2d_uniform16_feature_bank_v1"
FEATURE_CONTRACT_VERSION = "s2d_q1_a0_c0_d0_b_v1"
DEFAULT_SHARD_SIZE = 256
FROZEN_CHECKPOINT_SHA256 = "e54ccb228002578fa2c9a0faa8bdacbd61774e8952fcf06a306afa7b8518639b"
FEATURE_SHAPES = {
    "a0_rgb_mean": (16, 768),
    "c0_mcp_static_mean": (16, 768),
    "d0_static_residual_mean": (16, 768),
    "s2d_contextual_slots": (16, 768),
    "h_v": (768,),
    "logits_v": (6,),
}


def validate_splits(splits):
    """No implicit split expansion, aliases, empty requests or Test fallback."""
    values = tuple(splits.split(",")) if isinstance(splits, str) else tuple(splits)
    if not values or any(value not in ALLOWED_SPLITS for value in values) or len(set(values)) != len(values):
        raise ValueError("Only explicitly requested, unique train/dev splits are allowed; Test is forbidden")
    return values


def file_identity(path):
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path), "sha256": digest.hexdigest(), "size_bytes": path.stat().st_size}


def read_json(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def load_frozen_run(run_dir, checkpoint=None, *, expected_sha256=FROZEN_CHECKPOINT_SHA256):
    """Read existing selection evidence; do not replay selection or read final Test results."""
    run_dir = resolve_path(str(run_dir))
    config_path = run_dir / "resolved_config.yaml"
    config, vocab = load_config(config_path)
    metadata = read_json(run_dir / "run_metadata.json")
    logged_vocab = read_json(run_dir / "logs/label_vocab.json")
    if logged_vocab != vocab.as_dict() or metadata.get("label_vocab") != logged_vocab or vocab.num_classes != 6:
        raise ValueError("Frozen run label vocab disagrees with resolved config")
    if (not config["model"]["enabled"] or metadata.get("test_split_used_for_selection") is not False
            or metadata.get("checkpoint_selection_metric") != "val_weighted_f1"):
        raise ValueError("Expected enabled S2D and Dev-only frozen checkpoint selection")
    run_id = metadata.get("run_id")
    if run_id != run_dir.name or run_id != config["output"]["run_id"]:
        raise ValueError("Frozen run identity mismatch")
    for field, expected in (("seed", config["train"]["seed"]),
                            ("manifest_path", config["dataset"]["manifest_path"]),
                            ("data_root", config["dataset"]["data_root"]),
                            ("face_cache_path", config["face"]["cache_path"])):
        if metadata.get(field) != expected:
            raise ValueError(f"Frozen runtime provenance mismatch: {field}")
    # Only read the selected epoch and its Dev evidence, never compute a new argmax.
    records = [json.loads(line) for line in (run_dir / "logs/epochs.jsonl").read_text(encoding="utf-8").splitlines()]
    if not records:
        raise ValueError("Missing frozen epoch evidence")
    last = records[-1]
    if [row.get("epoch") for row in records] != list(range(1, len(records) + 1)):
        raise ValueError("Invalid frozen epoch sequence")
    early = config["early_stopping"]
    stopped = (early["enabled"] and len(records) >= early["min_epochs"]
               and last.get("epochs_without_improvement", -1) >= early["patience"])
    if len(records) > config["train"]["epochs"] or not (stopped or len(records) == config["train"]["epochs"]):
        raise ValueError("Frozen Train/Dev run has not reached a stopping boundary")
    epoch, score = last.get("best_epoch"), last.get("best_val_weighted_f1")
    if (type(epoch) is not int or not 1 <= epoch <= len(records)
            or type(score) not in (int, float) or not math.isfinite(score) or not 0 <= score <= 1
            or records[epoch - 1].get("epoch") != epoch
            or records[epoch - 1].get("dev", {}).get("weighted_f1") != score):
        raise ValueError("Inconsistent frozen selected epoch / Dev Weighted-F1")
    for field, value in (("best_epoch", epoch), ("best_val_weighted_f1", score)):
        if field in metadata and metadata[field] != value:
            raise ValueError(f"Frozen metadata disagrees with logs: {field}")
    # The known weights-only file is the sole default. Never try a sibling on failure.
    checkpoint = (resolve_path(str(checkpoint)) if checkpoint is not None
                  else run_dir / "checkpoints/best_model_weights_only.pt")
    identity = file_identity(checkpoint)
    if (not isinstance(expected_sha256, str) or len(expected_sha256) != 64
            or any(c not in "0123456789abcdef" for c in expected_sha256)
            or identity["sha256"] != expected_sha256):
        raise ValueError("Frozen checkpoint SHA256 mismatch")
    evidence = {
        "source_run_id": run_id, "source_run_path": str(run_dir),
        "source_git_head": metadata.get("git_commit"),
        "selected_epoch": epoch, "selected_dev_weighted_f1": score,
        "checkpoint": {**identity, "filename": checkpoint.name, "expected_sha256": expected_sha256,
                       "weights_only": True, "strict": True},
        "resolved_config": file_identity(config_path), "label_vocab": logged_vocab,
    }
    return config, vocab, checkpoint, evidence


def load_frozen_model(config, vocab, checkpoint):
    # A pure wrapper state_dict contains BOTH core and landmark. Initializing
    # from upstream pretraining here is redundant and could use legacy pickle.
    model_config = {key: value for key, value in config["model"].items()
                    if key not in {"pretrained_checkpoint", "landmark_checkpoint"}}
    model = build_s2d(vocab, model_config)
    reload_model_weights(model, checkpoint)  # weights_only=True, strict=True; exceptions propagate.
    return model.to(torch.device(config["system"]["device"])).eval()


def split_coverage(manifest, cache, split):
    validate_splits((split,))
    universe = manifest.for_split(split, available_only=False)
    available = tuple(row for row in universe if row.v_available)
    cache.require_ids(row.utterance_id for row in available)
    unusable = [row for row in available if not cache.entries[row.utterance_id].face_usable]
    return {
        "manifest_total": len(universe), "v_available": len(available),
        "face_cache_usable": len(available) - len(unusable),
        "exported_count": 0, "excluded_count": None,
        "exclusions": {
            "v_unavailable": [row.utterance_id for row in universe if not row.v_available],
            "cached_face_unusable": [row.utterance_id for row in unusable],
        },
        "face_unusable_reasons": {row.utterance_id: sorted(set(cache.entries[row.utterance_id].reasons))
                                 for row in unusable},
        "unknown_exclusion_count": 0,
        "failure_policy": "Missing cache, decode/crop mismatch or extraction error aborts; never silently excluded",
    }


def prepare_datasets(config, vocab, splits):
    splits = validate_splits(splits)  # Before opening any source, including manifest/cache.
    from datasets.iemocap.s2d_dataset import IEMOCAPS2DDataset
    from datasets.iemocap.s2d_faces import FaceBBoxCache, PREPROCESSING_VERSION, build_face_processor
    from datasets.iemocap.s2d_manifest import S2DManifest

    data = config["dataset"]
    root = resolve_path(data["data_root"])
    manifest_path = resolve_path(data["manifest_path"], root)
    cache_path = resolve_path(config["face"]["cache_path"])
    if root is None or not root.is_dir() or manifest_path is None or cache_path is None:
        raise ValueError("Existing frozen data root, manifest and face cache are required")
    manifest = S2DManifest(manifest_path, root, vocab, data.get("columns"))
    cache = FaceBBoxCache.load(cache_path)
    face = build_face_processor(config["face"])
    datasets, coverage = {}, {}
    for split in splits:  # Never use manifest.counts(), SPLITS, or a finalizer helper.
        coverage[split] = split_coverage(manifest, cache, split)
        datasets[split] = IEMOCAPS2DDataset(
            manifest, split, face, face_cache=cache, require_face_cache_for_training=True)
    sources = {"manifest": file_identity(manifest_path),
               "face_cache": {**file_identity(cache_path), "preprocessing_version": PREPROCESSING_VERSION},
               "data_root": str(root), "manifest_columns": data.get("columns", {})}
    return datasets, coverage, sources


def atomic_write(path, writer):
    """Write a same-directory .tmp, fsync/close, then publish with os.replace."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite {path}")
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w+b", dir=path.parent, prefix=f".{path.name}.",
                                         suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            writer(stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def atomic_json(path, value):
    content = (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")
    atomic_write(path, lambda stream: stream.write(content))


def build_provenance(evidence, sources, splits, shard_size, *, git_head):
    splits = validate_splits(splits)
    return {
        **evidence, **sources, "schema_version": SCHEMA_VERSION,
        "created_by": "scripts/models/s2d/iemocap/export_features.py", "git_head": git_head,
        "feature_contract_version": FEATURE_CONTRACT_VERSION,
        "sampling_policy": "Uniform16", "num_frames": 16, "dtype": "float32",
        "feature_fields": {key: list(shape) for key, shape in FEATURE_SHAPES.items()},
        "shard_size": shard_size, "splits": list(splits), "shuffle": False,
        "representation_roles": {"main": ["d0_static_residual_mean", "s2d_contextual_slots"],
                                 "diagnostics": ["a0_rgb_mean", "c0_mcp_static_mean"]},
        "q3_preselection_candidate": "d0_static_residual_mean",
        "contextual_slots_preselection_efficiency_claims_allowed": False,
    }


def write_split(model, loader, destination, split, *, shard_size, device):
    """Stream at most shard_size feature rows plus one inference batch in RAM."""
    validate_splits((split,))
    from torch.utils.data import SequentialSampler

    if (loader.dataset.split != split or not isinstance(loader.sampler, SequentialSampler)
            or loader.drop_last):
        raise ValueError("Export requires the requested split, shuffle=False and drop_last=False")
    records = loader.dataset.records
    destination = Path(destination)
    destination.mkdir()
    pending, index, parts, offset = [], [], [], 0

    def flush():
        if not pending:
            return
        filename = f"part-{len(parts):05d}.pt"
        payload = {"schema_version": SCHEMA_VERSION, "split": split,
                   "utterance_ids": [row["utterance_id"] for row in pending],
                   **{key: torch.stack([row[key] for row in pending])
                      for key in ("labels", "frame_indices", *FEATURE_SHAPES)}}
        atomic_write(destination / filename, lambda stream: torch.save(payload, stream))
        start = len(index)
        index.extend({"global_row": start + i, "utterance_id": row["utterance_id"],
                      "shard": filename, "row_in_shard": i} for i, row in enumerate(pending))
        parts.append({"filename": filename, "row_count": len(pending)})
        pending.clear()

    for batch in loader:
        ids = list(batch["utterance_id"])
        count = len(ids)
        expected = records[offset:offset + count]
        if (not count or ids != [row.utterance_id for row in expected]
                or list(batch["split"]) != [split] * count
                or not bool(batch["v_valid"].all()) or not bool(batch["face_usable"].all())):
            raise ValueError("Batch identity/order/split or visual validity disagrees with the frozen dataset")
        labels, frames = batch["label_index"], batch["frame_indices"]
        if (labels.dtype != torch.long or labels.shape != (count,)
                or labels.tolist() != [row.label_index for row in expected]
                or bool(((labels < 0) | (labels >= 6)).any())
                or frames.dtype != torch.long or frames.shape != (count, 16)
                or bool((frames < 0).any()) or bool((frames[:, 1:] < frames[:, :-1]).any())):
            raise ValueError("Invalid canonical labels / frame_indices contract")
        bundle = extract_uniform16_feature_bundle(model, batch["video"].to(device))
        features = {}
        for key, shape in FEATURE_SHAPES.items():
            value = bundle[key]
            if tuple(value.shape) != (count, *shape) or not value.is_floating_point() or not bool(torch.isfinite(value).all()):
                raise ValueError(f"Invalid feature contract: {key}")
            features[key] = value.detach().to(device="cpu", dtype=torch.float32)
        for i, uid in enumerate(ids):
            # Clone rows: slices must not retain storage from all earlier batches.
            pending.append({"utterance_id": uid, "labels": labels[i].clone(),
                            "frame_indices": frames[i].clone(),
                            **{key: value[i].clone() for key, value in features.items()}})
            if len(pending) == shard_size:
                flush()
        offset += count
    if offset != len(records):
        raise ValueError("Export did not visit every usable dataset record")
    flush()
    atomic_json(destination / "index.json", {"schema_version": SCHEMA_VERSION, "split": split,
                                           "row_count": offset, "shards": parts, "rows": index})
    return offset


def export_feature_bank(model, datasets, coverage, output_dir, *, splits, provenance,
                        shard_size=DEFAULT_SHARD_SIZE, batch_size=4, num_workers=0, device="cpu"):
    """Data-writing engine; synthetic datasets can exercise it without real IO."""
    splits = validate_splits(splits)
    if set(datasets) != set(splits) or set(coverage) != set(splits):
        raise ValueError("Datasets and coverage must contain exactly the requested train/dev splits")
    for name, value, lower in (("shard_size", shard_size, 1), ("batch_size", batch_size, 1),
                               ("num_workers", num_workers, 0)):
        if type(value) is not int or value < lower:
            raise ValueError(f"Invalid {name}")
    if any(module.training for module in model.modules()):
        raise ValueError("Export requires model.eval(); the writer never changes training flags")
    if (provenance.get("splits") != list(splits) or provenance.get("shard_size") != shard_size
            or provenance.get("dtype") != "float32"):
        raise ValueError("Provenance disagrees with export request")
    output_dir = Path(output_dir)
    # Exclusive reservation also protects concurrent writers and interrupted banks.
    # Even an existing empty directory is refused; choose a fresh destination.
    output_dir.mkdir(parents=True, exist_ok=False)
    atomic_json(output_dir / "provenance.json", {**provenance, "completion_marker": "complete.json"})
    completed = {}
    for split in splits:
        loader = DataLoader(datasets[split], batch_size=batch_size, num_workers=num_workers,
                            shuffle=False, drop_last=False, generator=torch.Generator().manual_seed(100))
        count = write_split(model, loader, output_dir / split, split, shard_size=shard_size, device=device)
        row = dict(coverage[split])
        if count != row["face_cache_usable"]:
            raise ValueError("Exported count disagrees with usable face/cache coverage")
        row.update(exported_count=count, excluded_count=row["manifest_total"] - count)
        if row["excluded_count"] != sum(len(ids) for ids in row["exclusions"].values()):
            raise ValueError("Coverage exclusion accounting mismatch")
        completed[split] = row
    atomic_json(output_dir / "coverage.json", {"schema_version": SCHEMA_VERSION, "splits": completed})
    # Separate completion evidence avoids ever replacing existing provenance.
    atomic_json(output_dir / "complete.json", {"schema_version": SCHEMA_VERSION, "status": "complete",
                                              "splits": list(splits)})
    return completed


def export_run(run_dir, output_dir, *, splits, checkpoint=None, expected_sha256=FROZEN_CHECKPOINT_SHA256,
               shard_size=DEFAULT_SHARD_SIZE):
    splits = validate_splits(splits)  # Fail closed before config, checkpoint, dataset or output IO.
    output_dir = resolve_path(str(output_dir))
    if output_dir.exists():
        raise FileExistsError(f"Refusing existing output directory: {output_dir}")
    config, vocab, checkpoint, evidence = load_frozen_run(run_dir, checkpoint, expected_sha256=expected_sha256)
    model = load_frozen_model(config, vocab, checkpoint)
    datasets, coverage, sources = prepare_datasets(config, vocab, splits)
    git_head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT,
                              check=True, capture_output=True, text=True).stdout.strip()
    provenance = build_provenance(evidence, sources, splits, shard_size, git_head=git_head)
    return export_feature_bank(model, datasets, coverage, output_dir, splits=splits, provenance=provenance,
                               shard_size=shard_size, batch_size=config["train"]["batch_size"],
                               num_workers=config["train"]["num_workers"], device=config["system"]["device"])


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, help="Existing frozen run with resolved config and logged vocab")
    parser.add_argument("--output-dir", required=True, help="New feature-bank directory; existing directories refused")
    split_args = parser.add_mutually_exclusive_group(required=True)
    split_args.add_argument("--split", choices=sorted(ALLOWED_SPLITS))
    split_args.add_argument("--splits", help="Explicit comma-separated train/dev request")
    parser.add_argument("--checkpoint", help="Explicit pure wrapper state_dict; default: run/checkpoints/best_model_weights_only.pt")
    parser.add_argument("--expected-checkpoint-sha256", default=FROZEN_CHECKPOINT_SHA256)
    parser.add_argument("--shard-size", type=int, default=DEFAULT_SHARD_SIZE)
    args = parser.parse_args(argv)
    try:
        args.splits = validate_splits(args.splits if args.splits is not None else (args.split,))
        if args.shard_size < 1:
            raise ValueError("shard_size must be positive")
    except ValueError as error:
        parser.error(str(error))
    return args


def main(argv=None):
    args = parse_args(argv)
    export_run(args.run_dir, args.output_dir, splits=args.splits, checkpoint=args.checkpoint,
               expected_sha256=args.expected_checkpoint_sha256, shard_size=args.shard_size)


if __name__ == "__main__":
    main()

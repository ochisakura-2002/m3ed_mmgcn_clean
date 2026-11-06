"""Finalize a completed S2D Train/Dev run without entering the training loop."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile


PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.s2d.iemocap.config import RUNTIME_PATH_FIELDS, load_config, resolve_path, validate_runtime_paths
from scripts.models.s2d.iemocap.checkpoints import reload_model_weights
from scripts.models.s2d.iemocap import train as training


ATTEMPT_FILE = training.FINAL_TEST_ATTEMPT_FILE
FINAL_OUTPUTS = training.FINAL_TEST_OUTPUTS


def read_json(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def require_equal(actual, expected, name):
    if type(actual) is not type(expected) or actual != expected:
        raise ValueError(f"Inconsistent {name}: {actual!r} != {expected!r}")


def recover_training_state(config, metadata, records):
    """Replay the unchanged selection policy; logs must end at its first stop."""
    early = config["early_stopping"]
    require_equal(metadata.get("early_stopping"), early, "early_stopping")
    require_equal(metadata.get("checkpoint_selection_metric"), "val_weighted_f1", "selection metric")
    selection = training.EarlyStopping(early["enabled"], early["min_epochs"], early["patience"])
    limit = config["train"]["epochs"]
    if not records or len(records) > limit:
        raise ValueError("Missing epochs or epoch count exceeds configured max_epochs")
    for epoch, row in enumerate(records, 1):
        require_equal(row.get("epoch"), epoch, "epoch sequence")
        score = row["dev"]["weighted_f1"]
        if type(score) not in (int, float) or not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError("Invalid logged Dev Weighted-F1")
        selection.update(epoch, score)
        for field in ("best_epoch", "best_val_weighted_f1", "epochs_without_improvement"):
            require_equal(row.get(field), getattr(selection, field), f"epoch {epoch} {field}")
        if selection.should_stop(epoch) and epoch != len(records):
            raise ValueError("Epochs continue beyond the first early-stopping boundary")
    stopped = len(records)
    if not selection.should_stop(stopped) and stopped != limit:
        raise ValueError("Training phase has not reached early stopping or max_epochs")
    result = {"stop_reason": "early_stopping" if selection.should_stop(stopped) else "max_epochs",
              "stopped_epoch": stopped, "best_epoch": selection.best_epoch,
              "best_val_weighted_f1": selection.best_val_weighted_f1,
              "min_epochs": selection.min_epochs, "patience": selection.patience,
              "epochs_without_improvement_at_stop": selection.epochs_without_improvement}
    for field, expected in result.items():
        if field in metadata:
            require_equal(metadata[field], expected, f"metadata {field}")
    return result


def validate_run(run_dir, checkpoint=None, *, overrides=None):
    """Read-only preflight. No model, dataset, Test loader, or output creation."""
    run_dir = resolve_path(str(run_dir))
    if not run_dir.is_dir():
        raise FileNotFoundError(f"run_dir must exist: {run_dir}")
    for name in ("run_metadata.json", "resolved_config.yaml", "logs/epochs.jsonl",
                 "logs/label_vocab.json", "metrics/coverage.json"):
        if not (run_dir / name).is_file():
            raise FileNotFoundError(f"Required run provenance missing: {name}")
    metadata = read_json(run_dir / "run_metadata.json")
    # Missing/ambiguous evidence also fails closed, including a prior interrupted attempt.
    require_equal(metadata.get("test_evaluation_count"), 0, "test_evaluation_count")
    require_equal(metadata.get("test_split_used_for_selection"), False, "test_split_used_for_selection")
    require_equal(metadata.get("training_completed"), False, "training_completed")
    for name in (*FINAL_OUTPUTS, ATTEMPT_FILE):
        if (run_dir / name).exists():
            raise FileExistsError(f"Refusing to repeat Test or overwrite finalization evidence: {name}")
    checkpoint = resolve_path(str(checkpoint)) if checkpoint is not None else run_dir / "checkpoints/best_model.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Best checkpoint missing: {checkpoint}")
    original, vocab = load_config(run_dir / "resolved_config.yaml")
    if not original["model"]["enabled"]:
        raise ValueError("S2D branch disabled")
    if overrides and set(overrides) - (set(RUNTIME_PATH_FIELDS) - {"output_dir"}):
        raise ValueError("Finalization permits only existing input runtime-path overrides")
    require_equal(metadata.get("label_vocab"), vocab.as_dict(), "metadata label_vocab")
    require_equal(read_json(run_dir / "logs/label_vocab.json"), vocab.as_dict(), "logged label_vocab")
    require_equal(metadata.get("seed"), original["train"]["seed"], "seed")
    require_equal(metadata.get("run_id"), original["output"]["run_id"], "run_id")
    require_equal(run_dir.name, metadata["run_id"], "run directory identity")
    for field, expected in (("data_root", original["dataset"]["data_root"]),
                            ("manifest_path", original["dataset"]["manifest_path"]),
                            ("face_cache_path", original["face"]["cache_path"])):
        require_equal(metadata.get(field), expected, field)
    for name, field in (("s2d", "pretrained_checkpoint"), ("landmark", "landmark_checkpoint")):
        provenance = metadata["checkpoints"][name]
        require_equal(provenance.get("path"), original["model"][field], f"{name} checkpoint path")
        require_equal(provenance.get("expected_sha256"), original["model"].get(f"{field}_sha256"),
                      f"{name} checkpoint expected_sha256")
    records = [json.loads(line) for line in (run_dir / "logs/epochs.jsonl").read_text(encoding="utf-8").splitlines()]
    state = recover_training_state(original, metadata, records)
    from datasets.iemocap.s2d_faces import PREPROCESSING_VERSION

    coverage = read_json(run_dir / "metrics/coverage.json")
    require_equal(metadata.get("face_preprocessing_version"), PREPROCESSING_VERSION, "face preprocessing")
    require_equal(coverage.get("face_preprocessing_version"), PREPROCESSING_VERSION, "coverage preprocessing")
    config, _ = load_config(run_dir / "resolved_config.yaml", overrides=overrides)
    validate_runtime_paths(config)  # Reuse the training path resolver, including input existence checks.
    return run_dir, checkpoint, config, vocab, metadata, state, coverage


def prepare_evaluation_inputs(config, vocab, coverage):
    """Validate manifest/cache coverage without constructing a Test dataset or reading video."""
    from datasets.iemocap.s2d_faces import FaceBBoxCache, build_face_processor
    from datasets.iemocap.s2d_manifest import S2DManifest

    data = config["dataset"]
    root = resolve_path(data["data_root"])
    manifest = S2DManifest(resolve_path(data["manifest_path"], root), root, vocab, data.get("columns"))
    cache = FaceBBoxCache.load(resolve_path(config["face"]["cache_path"]))
    cache.require_ids(row.utterance_id for row in manifest.records if row.v_available)
    require_equal(manifest.counts(), coverage.get("counts"), "manifest coverage counts")
    require_equal([r.utterance_id for r in manifest.records if not r.v_available],
                  coverage.get("v_unavailable_utterance_ids"), "V-unavailable identities")
    for split in ("train", "dev", "test"):
        rows = manifest.for_split(split)
        expected = {
            "face_usable": sum(cache.entries[r.utterance_id].face_usable for r in rows),
            "face_unusable_utterance_ids": [r.utterance_id for r in rows if not cache.entries[r.utterance_id].face_usable],
            "valid_detection_count": sum(cache.entries[r.utterance_id].valid_detection_count for r in rows),
            "fallback_frame_count": sum(sum(cache.entries[r.utterance_id].fallback_used) for r in rows)}
        require_equal(coverage["face_coverage"].get(split), expected, f"{split} face coverage")
    return manifest, build_face_processor(config["face"]), cache


def make_test_loader(config, inputs):
    import torch
    from torch.utils.data import DataLoader
    from datasets.iemocap.s2d_dataset import IEMOCAPS2DDataset

    manifest, face, cache = inputs
    dataset = IEMOCAPS2DDataset(
        manifest, "test", face, face_cache=cache,
        require_face_cache_for_training=config["face"].get("require_cache_for_training", True))
    if not len(dataset):
        raise ValueError("No available V rows for test")
    return DataLoader(dataset, batch_size=config["train"]["batch_size"],
                      num_workers=config["train"]["num_workers"], shuffle=False,
                      worker_init_fn=training.worker_seed,
                      generator=torch.Generator().manual_seed(config["train"]["seed"]), drop_last=False)


def atomic_write_metadata(path, value):
    """Keep the previous metadata intact on serialization/write failure."""
    content = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=".finalization_metadata_", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def finalize_run(run_dir, checkpoint=None, *, overrides=None):
    run_dir, checkpoint, config, vocab, metadata, state, coverage = validate_run(
        run_dir, checkpoint, overrides=overrides)
    import torch
    from models.s2d.iemocap.model import build_s2d
    from utils.seed import set_seed

    inputs = prepare_evaluation_inputs(config, vocab, coverage)
    set_seed(config["train"]["seed"])
    device = torch.device(config["system"]["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; finalize real S2D runs remotely")
    model = build_s2d(vocab, dict(config["model"])).to(device)
    with checkpoint.open("rb") as stream:
        digest = hashlib.sha256()
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    reload_model_weights(model, checkpoint)  # weights_only=True + strict=True; no legacy fallback.
    evidence = {"finalization_mode": "existing_run", "finalization_checkpoint_path": str(checkpoint),
                "finalization_checkpoint_sha256": digest.hexdigest(),
                "finalization_runtime_paths": {name: config[section][field]
                                               for name, (section, field) in RUNTIME_PATH_FIELDS.items()
                                               if name != "output_dir"}}
    training.reserve_test_evaluation(run_dir, {**evidence, **state})
    report, batches = training.run_epoch(model, make_test_loader(config, inputs), device, vocab, collect=True)
    report.update({"selected_epoch": state["best_epoch"],
                   "selected_val_weighted_f1": state["best_val_weighted_f1"]})
    for name in ("metrics", "predictions"):
        (run_dir / name).mkdir(exist_ok=True)
    with (run_dir / FINAL_OUTPUTS[0]).open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    training.export_predictions(batches, run_dir / FINAL_OUTPUTS[1], vocab)
    required = list(FINAL_OUTPUTS[:2])
    if config["export"]["representations"]:
        (run_dir / "representations").mkdir(exist_ok=True)
        with (run_dir / FINAL_OUTPUTS[2]).open("xb") as stream:
            torch.save({"label_vocab": vocab.as_dict(), "split": "test", "batches": batches}, stream)
        required.append(FINAL_OUTPUTS[2])
    if any(not (run_dir / name).is_file() or (run_dir / name).stat().st_size == 0 for name in required):
        raise RuntimeError("Required final exports missing or empty")
    atomic_write_metadata(run_dir / "run_metadata.json", {
        **metadata, **state, **evidence, "training_completed": True,
        "test_evaluation_count": 1, "test_split_used_for_selection": False})
    print(f"Existing run finalized; Test evaluated once: {run_dir}")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, help="Existing run; read its resolved_config.yaml")
    parser.add_argument("--checkpoint", help="Pure state_dict; relative to project root or absolute. Default: run checkpoints/best_model.pt")
    for name in RUNTIME_PATH_FIELDS:
        if name != "output_dir":
            parser.add_argument(f"--{name.replace('_', '-')}", help="Existing training runtime-path override")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    overrides = {name: getattr(args, name) for name in RUNTIME_PATH_FIELDS
                 if name != "output_dir" and getattr(args, name) is not None}
    finalize_run(args.run_dir, args.checkpoint, overrides=overrides)


if __name__ == "__main__":
    main()

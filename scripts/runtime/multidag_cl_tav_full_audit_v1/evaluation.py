"""Seven deterministic Dev masks on the same selected full-TAV checkpoint."""

from pathlib import Path

from scripts.runtime.multidag_cl_ta_v0.formal import sha256_file, write_json_atomic
from scripts.runtime.multidag_cl_ta_v0.training import reload_model_checkpoint
from .spec import MODALITIES


def evaluate_masks(runtime, run_dir, rows, destination, *, expected_checkpoint_sha256):
    if runtime.model.audit_config["active_modalities"] != "TAV":
        raise ValueError("missing-modality audit requires a full-TAV trained checkpoint")
    if [row["active_modalities"] for row in rows] != list(MODALITIES) or any(
            row["split"] != "dev" or row["audit_type"] != "TEST_TIME_MODALITY_MASK"
            or row["source_experiment_id"] != runtime.model.audit_config["experiment_id"] for row in rows):
        raise ValueError("exactly seven ordered Dev masks for one source required")
    destination = Path(destination)
    targets = [destination / (row["evaluation_id"] + ".json") for row in rows]
    if any(path.exists() for path in targets):
        raise FileExistsError("existing evaluation cannot be overwritten")
    destination.mkdir(parents=True, exist_ok=True)
    checkpoint = Path(run_dir) / "checkpoints" / "best_model.pt"
    if sha256_file(checkpoint) != expected_checkpoint_sha256:
        raise ValueError("selected checkpoint SHA256 mismatch")
    loaded = reload_model_checkpoint(runtime.model, checkpoint, validate=runtime.validate_checkpoint)
    initial_steps = runtime.optimizer_steps
    results = []
    for row, path in zip(rows, targets):
        report = runtime.evaluate(active_modalities=row["active_modalities"])
        result = {**row, "status": "completed", "checkpoint_sha256": expected_checkpoint_sha256,
                  "source_run_dir": Path(run_dir).as_posix(), "metrics": report["metrics"],
                  "best_epoch": loaded["epoch"], "parameter_count": sum(p.numel() for p in runtime.model.parameters()),
                  "data_identity": runtime.data_identity, "optimizer_steps_performed": 0,
                  "test_loader_built": 0, "test_evaluation_count": 0}
        # Reserve each result atomically before writing. Partial files from a
        # failed write remain evidence and cannot be silently replaced.
        with path.open("x", encoding="utf-8"):
            pass
        write_json_atomic(path, result)
        results.append(result)
    if runtime.optimizer_steps != initial_steps or sha256_file(checkpoint) != expected_checkpoint_sha256:
        raise ValueError("missing-modality evaluation mutated training/checkpoint state")
    return results

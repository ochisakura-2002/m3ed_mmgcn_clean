"""Conservative qualification of historical CE/LS checkpoints and exact prefixes."""

import csv
import json
import subprocess
import sys
from pathlib import Path

import torch

from scripts.models.multidag_cl.group_isolated import ROOT
from scripts.runtime.multidag_cl_group_isolated.evaluation import checkpoint_path
from scripts.runtime.multidag_cl_group_isolated.fixed_tail_average import authenticated_run, read_checkpoint, validate_state
from scripts.runtime.multidag_cl_group_isolated.protocol import ENDPOINT, fold_config
from scripts.runtime.multidag_cl_ta_v0.formal import json_identity, sha256_file


def historical_imports():
    """Inventory the original train entrypoint in a process without G3 imports."""
    code = (
        "import json; from scripts.models.multidag_cl.group_isolated import ROOT; "
        "from scripts.runtime.multidag_cl_group_isolated.provenance import source_hashes; "
        "print(json.dumps(source_hashes(ROOT, ())))"
    )
    return json.loads(subprocess.check_output([sys.executable, "-B", "-c", code], cwd=ROOT))


def match_provenance(manifest, prepared, source, reference):
    if (manifest["initialization"] != prepared.initialization
            or manifest["training_initial_rng"] != prepared.initial_rng):
        raise ValueError("historical initialization or initial RNG differs")
    provenance = manifest["runtime_provenance"]
    if json_identity(provenance["environment"]) != json_identity(source["environment"]):
        raise ValueError("historical training environment differs")
    hashes = {**provenance["files_sha256"], **provenance["late_imports_sha256"]}
    if not reference or any(hashes.get(name) != digest for name, digest in reference.items()):
        raise ValueError("historical runtime source inventory does not cover the original entrypoint")
    for name, digest in hashes.items():
        path = (ROOT / name).resolve()
        if not path.is_relative_to(ROOT) or not path.is_file() or sha256_file(path) != digest:
            raise ValueError(f"historical source or configuration differs: {name}")


def match_tape(run, manifest, tape):
    steps = manifest["steps_per_epoch"]
    expected = [(epoch, epoch * steps, json_identity([
        row["batch_sha256"] for row in tape[(epoch - 1) * steps:epoch * steps]
    ])) for epoch in range(1, ENDPOINT + 1)]
    path = run / "logs" / "fit_updates.csv"
    with path.open(encoding="utf-8", newline="") as stream:
        actual = [(int(row["epoch"]), int(row["optimizer_steps"]), row["batch_tape_sha256"])
                  for row in csv.DictReader(stream)]
    if len(tape) != ENDPOINT * steps or actual != expected:
        raise ValueError("historical batch tape differs from the complete frozen trajectory")
    return sha256_file(path)


def checkpoint_record(run, manifest, epoch):
    return {"path": checkpoint_path(run, epoch).relative_to(run).as_posix(), "epoch": epoch,
            "sha256": manifest["checkpoint_sha256"][str(epoch)],
            "optimizer_steps": epoch * manifest["steps_per_epoch"]}


def checkpoint_at(run, manifest, epoch, model):
    source = checkpoint_record(run, manifest, epoch)
    state = read_checkpoint(run, source, manifest["run_identity"])
    validate_state(state, model.state_dict())
    return state, source


def inspect_history(root, outer, is_ls, frozen, raw, prepared, source, reference, selected):
    config = fold_config(outer, g1=is_ls)
    run = root / "runs" / config["run_id"]
    manifest, digest = authenticated_run(run, config, frozen, raw)
    match_provenance(manifest, prepared, source, reference)
    tape_sha = match_tape(run, manifest, prepared.tape)
    steps = manifest["steps_per_epoch"]
    epoch, remainder = divmod(selected, steps)
    environment = source["environment"]
    if remainder and not (
        environment["torch_deterministic_algorithms"]
        and environment["cublas_workspace_config"] in (":4096:8", ":16:8")
    ):
        raise ValueError("historical prefix requires matching deterministic algorithms and cuBLAS workspace")
    model = prepared.runtime.model
    state30, endpoint = checkpoint_at(run, manifest, ENDPOINT, model)
    if remainder:
        # Read every completed epoch before the target for an exact replay comparison.
        anchors = {e: checkpoint_record(run, manifest, e) for e in range(1, epoch + 1)}
        if any(not (run / record["path"]).is_file() for record in anchors.values()):
            raise ValueError("historical prefix anchor is missing")
        selected_state, selected_source = None, None
    else:
        anchors = {}
        selected_state, selected_source = ((state30, endpoint) if epoch == ENDPOINT
                                           else checkpoint_at(run, manifest, epoch, model))
    return {"e30": state30, "selected": selected_state, "anchors": anchors, "run": run,
            "identity": manifest["run_identity"],
            "evidence": {"run_manifest_sha256": digest, "fit_updates_sha256": tape_sha,
                         "run_id": manifest["run_id"], "endpoint": endpoint,
                         "selected": selected_source, "prefix_anchors": list(anchors.values()),
                         "historical_fit_wall_seconds": manifest.get("fit_wall_seconds")}}


def verify_anchor(history, epoch, model):
    reference = read_checkpoint(history["run"], history["anchors"][epoch], history["identity"])
    state = model.state_dict()
    validate_state(reference, state)
    if any((not torch.equal(value.detach().cpu(), reference[key])) if isinstance(value, torch.Tensor)
           else value != reference[key] for key, value in state.items()):
        raise ValueError(f"deterministic prefix differs from historical weights at epoch {epoch}")

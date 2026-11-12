"""CPU synthetic CHG forward/backward and Dev diagnostics; no optimizer or files."""

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import yaml

from models.multidag_cl.ta_v0.causal_history_gate import collect_gate_diagnostics
from scripts.models.multidag_cl.ta_v0.train import synthetic_runner
from scripts.runtime.multidag_cl_ta_v0.formal import validate_formal_config

CHG_CONFIG = ("configs/multidag_cl/ta_v0/iemocap/causal_context/"
              "asymmetric_merc_v1/formal_train_dev_chg_v0.yaml")


def smoke(config):
    if not config.use_causal_history_gate:
        raise ValueError("CHG smoke requires the gate ON")
    runner = synthetic_runner(config)
    output, loss = runner.forward_batch(next(iter(runner.loader("train"))), split="train")
    loss.backward()
    gates = [layer.attention.history_gate for layer in runner.model.graph_layers]
    for gate in gates:
        parameters = list(gate.parameters())
        if any(p.grad is None or not torch.isfinite(p.grad).all() for p in parameters):
            raise AssertionError("missing/nonfinite CHG gradients")
        if any(float(p.grad.abs().sum()) <= 0 for p in parameters):
            raise AssertionError("CHG parameter tensor received no nonzero gradient")
    with collect_gate_diagnostics(runner.model) as diagnostics, torch.no_grad():
        for batch in runner.loader("dev"):
            runner.forward_batch(batch, split="dev")
    groups = diagnostics.summary()
    if groups["all"]["edge_count"] == 0:
        raise AssertionError("no Dev history edges collected")
    # A single-utterance Dev dialogue contributes no edges and no padded edges.
    assert groups["all"]["edge_count"] == 12
    json.dumps(groups, allow_nan=False)
    baseline = synthetic_runner(replace(config, use_causal_history_gate=False))
    baseline_output, baseline_loss = baseline.forward_batch(
        next(iter(baseline.loader("train"))), split="train")
    baseline_loss.backward()
    assert baseline_output.h_ta.shape == output.h_ta.shape
    assert all(layer.attention.history_gate is None for layer in baseline.model.graph_layers)
    added = sum(p.numel() for p in runner.model.parameters()) - sum(
        p.numel() for p in baseline.model.parameters())
    return {"status": "PASS", "synthetic_only": True, "device": "cpu",
            "model_construct": "PASS", "forward": "PASS", "backward": "PASS",
            "chg_parameter_gradients": "PASS", "gate_diagnostics": groups,
            "disabled_forward_backward": "PASS", "loss": float(loss.detach()),
            "h_ta_shape": list(output.h_ta.shape), "added_parameters": added,
            "optimizer_steps": 0, "training_started": False, "test_evaluation_count": 0}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=CHG_CONFIG)
    args = parser.parse_args(argv)
    config = validate_formal_config(yaml.safe_load(Path(args.config).read_text(encoding="utf-8")))
    torch.set_num_threads(1)
    print(json.dumps(smoke(config), sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

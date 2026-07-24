from __future__ import annotations

from pathlib import Path

import torch

from models.registry.causal import build_new_causal_baseline
from scripts.analysis.causal.audit_model_causality import build_synthetic_batch
from scripts.workflows.causal_graph.evaluate import evaluate_checkpoint
from scripts.runtime.causal_graph import (
    build_optimizer,
    forward_batch,
    load_checkpoint,
    load_yaml_config,
    normalized_training_config,
    rebuild_model_from_checkpoint,
)
from scripts.workflows.causal_graph.train import checkpoint_payload


def test_checkpoint_rebuild_preserves_eval_logits_and_writes_outputs(
    tmp_path: Path,
) -> None:
    config = normalized_training_config(
        load_yaml_config(
            Path(
                "configs/dialoguegcn/unified/synthetic/causal_context/"
                "synthetic/smoke_end_to_end.yaml"
            )
        )
    )
    torch.manual_seed(77)
    model = build_new_causal_baseline(config).eval()
    optimizer = build_optimizer(model, config)
    batch = build_synthetic_batch(config, batch_size=2, seq_len=5, seed=81)
    with torch.no_grad():
        before = forward_batch(model, config, batch, return_aux=False)["logits"]

    checkpoint_path = tmp_path / "run" / "checkpoints" / "best_model.pt"
    checkpoint_path.parent.mkdir(parents=True)
    torch.save(checkpoint_payload(model, optimizer, config, 1, 0.25), checkpoint_path)
    checkpoint = load_checkpoint(checkpoint_path, torch.device("cpu"))
    reloaded = rebuild_model_from_checkpoint(checkpoint, torch.device("cpu"))
    with torch.no_grad():
        after = forward_batch(reloaded, config, batch, return_aux=False)["logits"]
    torch.testing.assert_close(before, after, atol=0.0, rtol=0.0)
    assert checkpoint["test_split_used_for_selection"] is False
    assert checkpoint["checkpoint_selection"] == "validation_val_weighted_f1"

    output_dir = tmp_path / "evaluation"
    evaluated = evaluate_checkpoint(
        checkpoint_path, "val", output_dir=output_dir, device_override="cpu"
    )
    assert evaluated["checkpoint_epoch"] == 1
    assert {path.name for path in output_dir.iterdir()} == {
        "metrics.csv",
        "predictions.csv",
        "confusion_matrix.csv",
        "per_class_recall.csv",
    }

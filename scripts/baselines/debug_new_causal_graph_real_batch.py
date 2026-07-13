"""Run one real IEMOCAP batch through a new causal graph baseline."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any, Dict

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.baselines.causal_baseline_registry import (  # noqa: E402
    build_new_causal_baseline,
    get_new_causal_model_family,
)
from scripts.baselines.new_causal_graph_runtime import (  # noqa: E402
    build_dataloader,
    compute_batch_loss,
    forward_batch,
    get_device,
    load_yaml_config,
    move_batch,
    normalized_training_config,
    resolve_path,
    validate_runtime_config,
    verify_feature_sha256,
)
from utils.run_metadata import build_run_metadata  # noqa: E402
from utils.seed import set_seed  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), default="train")
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def padding_excluded_from_loss(
    config: Dict[str, Any], output: Dict[str, Any], batch: Dict[str, Any]
) -> bool:
    original = compute_batch_loss(config, output, batch)["total_loss"]
    changed_output = dict(output)
    changed_logits = output["logits"].clone()
    padding = ~batch["attention_mask"].bool()
    if bool(padding.any()):
        changed_logits[padding] = 1000.0
    changed_output["logits"] = changed_logits
    changed_batch = dict(batch)
    changed_labels = batch["labels"].clone()
    changed_labels[padding] = 0
    changed_batch["labels"] = changed_labels
    changed = compute_batch_loss(config, changed_output, changed_batch)["total_loss"]
    return bool(torch.allclose(original, changed, atol=0.0, rtol=0.0))


def run_real_batch_debug(
    config_path: Path, split: str = "train", device_override: str = None
) -> Dict[str, Any]:
    config = normalized_training_config(load_yaml_config(resolve_path(str(config_path))))
    validate_runtime_config(config)
    if str(config["dataset"].get("name", "")).upper() != "IEMOCAP":
        raise ValueError("Real-batch debug requires dataset.name=IEMOCAP.")
    verified_sha = verify_feature_sha256(config)
    set_seed(int(config.get("system", {}).get("seed", 42)))
    device = get_device(config, device_override)
    batch_size = int(config["training"]["batch_size"])
    loader = build_dataloader(config, split, shuffle=False, batch_size=batch_size)
    raw_batch = next(iter(loader))
    batch = move_batch(raw_batch, device)
    model = build_new_causal_baseline(config).to(device)
    model.train()
    output = forward_batch(model, config, batch, return_aux=True)
    losses = compute_batch_loss(config, output, batch)
    padding_ok = padding_excluded_from_loss(config, output, batch)
    losses["total_loss"].backward()
    finite_gradients = all(
        parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
        for parameter in model.parameters()
    )
    expected_logits = (
        batch["text_features"].shape[0],
        batch["text_features"].shape[1],
        int(config["model"]["num_classes"]),
    )
    logits_shape_ok = tuple(output["logits"].shape) == expected_logits
    metadata = build_run_metadata(config, PROJECT_ROOT)

    print(f"dialogue_ids={raw_batch['dialogue_ids']}")
    print(f"lengths={batch['lengths'].tolist()}")
    for key in ("text_features", "audio_features", "visual_features"):
        print(f"{key}_shape={tuple(batch[key].shape)}")
    print(f"logits_shape={tuple(output['logits'].shape)}")
    print(f"finite_gradients={finite_gradients}")
    print(f"padding_excluded_from_loss={padding_ok}")
    print(f"run_metadata_generatable={bool(metadata)}")

    family = get_new_causal_model_family(config)
    extras: Dict[str, Any] = {}
    if family == "gsmcc":
        extras = {
            "low_frequency_shape": tuple(output["low_frequency_repr"].shape),
            "high_frequency_shape": tuple(output["high_frequency_repr"].shape),
            "fused_shape": tuple(output["fused_repr"].shape),
            "node_counts": output["causal_diagnostics"]["valid_node_count_per_dialogue"].tolist(),
            "edge_counts": output["causal_diagnostics"]["edge_count_per_dialogue"].tolist(),
        }
        print(f"low_frequency_shape={extras['low_frequency_shape']}")
        print(f"high_frequency_shape={extras['high_frequency_shape']}")
        print(f"fused_shape={extras['fused_shape']}")
        print(f"node_counts={extras['node_counts']}")
        print(f"edge_counts={extras['edge_counts']}")
        print(
            "losses="
            f"classification:{float(losses['classification_loss']):.6f},"
            f"consistency:{float(losses['consistency_loss']):.6f},"
            f"complementarity:{float(losses['complementarity_loss']):.6f}"
        )
    else:
        relation_counts = {}
        for relation_id, relation_name in output["relation_mapping"].items():
            count = int(
                ((output["relation_ids"] == int(relation_id)) & output["adjacency"]).sum().item()
            )
            relation_counts[str(relation_name)] = count
        valid_rows = batch["attention_mask"].bool()
        row_sums = output["edge_attention"].sum(dim=-1)[valid_rows]
        extras = {
            "context_shape": tuple(output["context_repr"].shape),
            "graph_shape": tuple(output["graph_repr"].shape),
            "relation_mapping": output["relation_mapping"],
            "relation_edge_counts": relation_counts,
            "edge_attention_row_sum_min": float(row_sums.min().item()),
            "edge_attention_row_sum_max": float(row_sums.max().item()),
        }
        print(f"context_shape={extras['context_shape']}")
        print(f"graph_shape={extras['graph_shape']}")
        print(f"relation_mapping={extras['relation_mapping']}")
        print(f"relation_edge_counts={relation_counts}")
        print(
            "edge_attention_row_sum_range="
            f"[{extras['edge_attention_row_sum_min']:.6f}, "
            f"{extras['edge_attention_row_sum_max']:.6f}]"
        )

    if not finite_gradients or not logits_shape_ok or not padding_ok:
        raise RuntimeError("Real-batch debug checks failed.")
    return {
        "verified_feature_sha256": verified_sha,
        "finite_gradients": finite_gradients,
        "logits_shape_ok": logits_shape_ok,
        "padding_excluded_from_loss": padding_ok,
        "metadata": metadata,
        "extras": extras,
    }


def main() -> None:
    args = parse_args()
    run_real_batch_debug(Path(args.config), args.split, args.device)


if __name__ == "__main__":
    main()

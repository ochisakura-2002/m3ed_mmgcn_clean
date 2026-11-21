"""YAML-selectable production fusion implementations."""

from .base import DirectConcatFusion
from .experts import UtteranceModalityExpertRouter
from .matching import TextCenteredMatching


DIRECT = "direct_concat"
LOW_RANK = "text_centered_low_rank_matching"
FULL_RANK = "text_centered_full_rank_matching"
ROUTER = "utterance_modality_expert_router"

REGISTRY = {
    DIRECT: DirectConcatFusion,
    LOW_RANK: TextCenteredMatching,
    FULL_RANK: TextCenteredMatching,
    ROUTER: UtteranceModalityExpertRouter,
}


def build_fusion(config):
    if not isinstance(config, dict) or config.get("type") not in REGISTRY:
        raise ValueError("unknown CT-MultiDAG fusion type")
    kind = config["type"]
    fields = {
        DIRECT: set(),
        LOW_RANK: {"rank", "residual_scale"},
        FULL_RANK: {"residual_scale"},
        ROUTER: {"router_hidden_dim", "expert_hidden_dim", "residual_scale"},
    }
    options = {key: value for key, value in config.items() if key != "type"}
    optional = {"active_edges"} if kind == LOW_RANK else set()
    if not fields[kind] <= options.keys() <= fields[kind] | optional:
        raise ValueError(f"unexpected/missing settings for {kind}")
    if "active_edges" in options and options["active_edges"] is None:
        raise ValueError("explicit active_edges must contain matching edge labels")
    if "residual_scale" in options and options["residual_scale"] != 0.1:
        raise ValueError("first-screen fusion residual_scale must be 0.1")
    if kind == LOW_RANK and (type(options["rank"]) is not int or not 1 <= options["rank"] <= 16):
        raise ValueError("low-rank matching rank must be an integer from 1 to 16")
    if kind == ROUTER and (options["router_hidden_dim"], options["expert_hidden_dim"]) != (64, 256):
        raise ValueError("first-screen router/expert widths must be 64/256")
    return REGISTRY[kind](**options)

"""Small explicit GraphCore controls, separate from frozen TA-v0 configuration."""

from dataclasses import dataclass
import math

from models.multidag_cl.ta_v0 import TAV0Config

PARENT_BASELINE = "ta_v0_t1024_ab1_diagnostic"
VARIANTS = {
    "track": {
        "model_name": "multidag_track", "model_lineage": "multidag_track_t1024_v0",
        "source_paper": "Pei et al., Multi-Track Message Passing, ICML 2024",
    },
    "coop": {
        "model_name": "multidag_coop", "model_lineage": "multidag_coop_t1024_v0",
        "source_paper": "Finkelshtein et al., Cooperative Graph Neural Networks, ICML 2024",
    },
    "dcn": {
        "model_name": "multidag_dcn", "model_lineage": "multidag_dcn_t1024_v0",
        "source_paper": "Rey et al., Directed Acyclic Graph Convolutional Networks, 2025 v1",
    },
}


def baseline_config():
    return TAV0Config(text_dim=1024, diagnostic_t1024_ab1=True)


@dataclass(frozen=True)
class GraphCoreConfig:
    variant: str
    enabled: bool = True
    track_initial_residual: float = 0.1

    def __post_init__(self):
        if self.variant not in VARIANTS:
            raise ValueError("unknown independent graph_core variant")
        if type(self.enabled) is not bool:
            raise TypeError("graph_core.enabled must be bool")
        alpha = self.track_initial_residual
        if (isinstance(alpha, bool) or not isinstance(alpha, (int, float))
                or not math.isfinite(alpha) or not 0 <= alpha <= 1):
            raise ValueError("track_initial_residual must lie in [0,1]")
        if self.variant != "track" and alpha != 0.1:
            raise ValueError("track-only parameter supplied to another variant")

    def resolved(self):
        result = {"variant": self.variant, "enabled": self.enabled}
        if self.variant == "track":
            result["track_initial_residual"] = float(self.track_initial_residual)
        return result

    @classmethod
    def from_mapping(cls, raw):
        if not isinstance(raw, dict) or "variant" not in raw:
            raise ValueError("explicit graph_core.variant is required")
        allowed = {"variant", "enabled"}
        if raw["variant"] == "track":
            allowed.add("track_initial_residual")
        if set(raw) - allowed:
            raise ValueError("unexpected graph_core parameters")
        return cls(**raw)

"""Graph-only TA wrapper; frozen modules are initialized by the original TAV0."""

import torch

from models.multidag_cl.ta_v0 import TAV0
from models.multidag_cl.ta_v0.model import TAOutput
from .common import valid_prefix
from .config import GraphCoreConfig, PARENT_BASELINE, VARIANTS, baseline_config
from .coop import CoopCore
from .dcn import DCNCore
from .track import TrackCore


class GraphCoreTA(TAV0):
    def __init__(self, graph_config):
        if not isinstance(graph_config, GraphCoreConfig):
            raise TypeError("graph_config must be GraphCoreConfig")
        # Preserve the parent's encoder/fusion/classifier seed initialization order.
        super().__init__(baseline_config())
        self.graph_config = graph_config
        if graph_config.enabled:
            del self.graph_layers
            if graph_config.variant == "track":
                self.graph_core = TrackCore(initial_residual=graph_config.track_initial_residual)
            elif graph_config.variant == "coop":
                self.graph_core = CoopCore()
            else:
                self.graph_core = DCNCore()

    def get_extra_state(self):
        state = super().get_extra_state()
        if self.graph_config.enabled:
            state.update(VARIANTS[self.graph_config.variant], parent_baseline=PARENT_BASELINE,
                         graph_core=self.graph_config.resolved())
        return state

    def forward(self, *, text_features, audio_features, speaker_ids_int, lengths, attention_mask):
        inputs = dict(text_features=text_features, audio_features=audio_features,
                      speaker_ids_int=speaker_ids_int, lengths=lengths, attention_mask=attention_mask)
        if not self.graph_config.enabled:
            return super().forward(**inputs)
        # Reproduce the baseline's explicit boundary checks; all mathematics below
        # calls the original modules. No dataset or training framework is copied.
        mask = valid_prefix(attention_mask)
        b, t = mask.shape
        if lengths.dtype != torch.int64 or lengths.shape != (b,):
            raise TypeError("lengths must be int64 [B]")
        if speaker_ids_int.dtype != torch.int64 or speaker_ids_int.shape != (b, t):
            raise TypeError("speaker_ids_int must be int64 [B,T]")
        if len({x.device for x in inputs.values()}) != 1:
            raise ValueError("all inputs must share one device")
        if not torch.equal(lengths, mask.sum(-1)):
            raise ValueError("lengths must match the contiguous mask")
        for name, value, dim in (("text", text_features, 1024), ("audio", audio_features, 1582)):
            if value.dtype != torch.float32 or value.shape != (b, t, dim):
                raise TypeError(f"{name} must be float32 [B,T,{dim}]")
            if not torch.isfinite(value).all() or torch.count_nonzero(value[~mask]):
                raise ValueError(f"{name} must be finite with exact zero padding")
        if torch.any(speaker_ids_int[mask] < 0):
            raise ValueError("valid speaker IDs must be nonnegative")
        text, audio = self.encoders(text_features, audio_features, lengths, mask)
        h0 = self.ta_fusion_projection(torch.cat((text, audio), dim=-1)) * mask[..., None]
        adjacency = self.predecessor_builder.build(speaker_ids_int, attention_mask)
        relations = self.relation_builder.build(speaker_ids_int, attention_mask)
        states = self.graph_core(h0, adjacency, relations, attention_mask)
        h_ta = torch.cat((h0, *states), dim=-1)
        logits = self.classifier(h_ta) * mask[..., None]
        return TAOutput(h_ta, logits, torch.softmax(logits, -1), mask)

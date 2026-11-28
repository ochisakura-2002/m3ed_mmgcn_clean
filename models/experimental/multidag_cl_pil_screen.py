"""Isolated CT-MultiDAG C01 forward and within-dialogue predecessor selector."""

import torch
from torch import nn
from torch.nn import functional as F

from models.common.causal_graph.graph_builders import build_causal_utterance_adjacency
from models.multidag_cl.fusion.base import MODALITIES as BLOCK_ORDER
from models.multidag_cl.fusion.model import FusionTAV, FusionTAVOutput
from scripts.runtime.multidag_cl_tav_full_audit_v1.spec import MODALITIES


def predecessors(adjacency, speakers, mask):
    """Return -1 or the latest valid same-speaker position within each row."""
    positions = torch.arange(mask.shape[1], device=mask.device)
    allowed = (adjacency.bool() & (positions[:, None] > positions)
               & mask[:, :, None] & mask[:, None, :]
               & (speakers[:, :, None] == speakers[:, None, :]))
    return torch.where(allowed, positions, -1).amax(dim=-1)


def gather_previous(values, previous):
    rows = torch.arange(previous.shape[0], device=previous.device)[:, None]
    return values[rows, previous.clamp_min(0)]


class C01FusionTAV(FusionTAV):
    def __init__(self, raw, arm):
        if arm not in ("C01-CURRENT", "C01-PREV"):
            raise ValueError("invalid C01 arm")
        super().__init__(raw)
        self.arm = arm
        self.side_weight = nn.Parameter(torch.zeros(300, 200))

    def get_extra_state(self):
        return {**super().get_extra_state(), "pil_arm": self.arm}

    def forward(self, *, text_features, audio_features, visual_features, lengths,
                speaker_ids_int, attention_mask, active_modalities=None, availability=None,
                history_attenuation=None):
        active = self.audit_config["active_modalities"] if active_modalities is None else active_modalities
        if active not in MODALITIES:
            raise ValueError("invalid encoded modality mask")
        if self.training and (active != "TAV" or availability is not None):
            raise ValueError("C01 training requires complete TAV")
        mask = self._validate_common_inputs(text_features, audio_features, visual_features,
                                            attention_mask, lengths, speaker_ids_int, None)
        encoded = self.paper_encoder(text_features=text_features, audio_features=audio_features,
                                     visual_features=visual_features, lengths=lengths, attention_mask=attention_mask)
        available = torch.tensor([m in active for m in BLOCK_ORDER], device=mask.device)
        if availability is not None:
            if availability.shape != (*mask.shape, len(BLOCK_ORDER)) or availability.dtype != torch.bool:
                raise TypeError("availability must be bool [B,T,3] in A,V,T order")
            if availability.device != mask.device:
                raise ValueError("availability and attention_mask must share a device")
            available = available & availability
        available = available.expand(*mask.shape, -1)
        diagnostics = {}
        h0 = self.fusion(text=encoded.text, audio=encoded.audio, visual=encoded.visual,
                         attention_mask=attention_mask, availability=available,
                         diagnostics=None if self.training else diagnostics)
        encoded_parts = {key: value.masked_fill(~(mask & available[..., index])[..., None], 0.0)
                         for index, (key, value) in enumerate(zip(BLOCK_ORDER,
                                                                 (encoded.audio, encoded.visual, encoded.text)))}
        adjacency = build_causal_utterance_adjacency(attention_mask, lengths, window_past=None, self_loop=False)
        relations = self.relation_builder.build(speaker_ids_int, attention_mask)
        graph_options = {} if history_attenuation is None else {"history_attenuation": history_attenuation}
        states = (h0, *self.graph_core(h0, adjacency, relations, attention_mask, **graph_options))
        representation = torch.cat(states, -1)

        previous = predecessors(adjacency, speaker_ids_int, mask)
        av = torch.cat((encoded_parts["A"], encoded_parts["V"]), dim=-1)
        visible = available[..., :2].all(dim=-1)
        eligible = (previous >= 0) & visible & gather_previous(visible, previous)
        previous_av = gather_previous(av, previous)
        source = previous_av if self.arm == "C01-PREV" else av
        side = torch.where(eligible[..., None], source, 0.0)
        logits = self.classifier[0](representation) + F.linear(side, self.side_weight)
        for index in range(1, len(self.classifier)):
            logits = self.classifier[index](logits)
        logits = logits * mask[..., None]
        return FusionTAVOutput(representation, logits, torch.softmax(logits, -1), mask,
                               states, encoded_parts, adjacency, diagnostics)

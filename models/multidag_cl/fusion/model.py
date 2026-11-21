"""One CT-MultiDAG backbone shared by all four fusion arms."""

from copy import deepcopy
from dataclasses import dataclass

import torch

from models.common.causal_graph.graph_builders import build_causal_utterance_adjacency
from models.multidag_cl.graph_core_variants.track import TrackCore
from models.multidag_cl.tav_full_audit_v1.model import TAVFullAudit, TAVOutput
from scripts.runtime.multidag_cl_fusion.spec import validate_screen_config
from scripts.runtime.multidag_cl_tav_full_audit_v1.spec import MODALITIES, make_config
from .base import MODALITIES as BLOCK_ORDER, OUTPUT_WIDTH, WIDTH
from .registry import build_fusion


@dataclass(frozen=True)
class FusionTAVOutput(TAVOutput):
    fusion_diagnostics: dict


class FusionTAV(TAVFullAudit):
    def __init__(self, raw):
        raw = validate_screen_config(raw)
        super().__init__(make_config("track"))
        self.audit_config = raw
        self.fusion = build_fusion(raw["fusion"])
        projections = (self.paper_encoder.text_projection, self.paper_encoder.audio_projection,
                       self.paper_encoder.visual_projection)
        if (tuple((layer.in_features, layer.out_features) for layer in projections)
                != ((1024, WIDTH), (1582, WIDTH), (342, WIDTH))
                or self.paper_encoder.text_encoder.hidden_size != WIDTH
                or self.paper_encoder.text_encoder.bidirectional or WIDTH != 100 or OUTPUT_WIDTH != 300
                or not isinstance(self.graph_core, TrackCore) or self.graph_core.hidden_dim != 300
                or len(self.graph_core.readouts) != 4 or self.classifier[0].in_features != 1500):
            raise ValueError("FUSION_FORMAL_INTEGRATION = BLOCKED_ARCHITECTURE_DRIFT")

    def get_extra_state(self):
        return {"schema": "ct_multidag_fusion_model_v1", "audit_config": deepcopy(self.audit_config)}

    def forward(self, *, text_features, audio_features, visual_features, lengths,
                speaker_ids_int, attention_mask, active_modalities=None, availability=None):
        active = self.audit_config["active_modalities"] if active_modalities is None else active_modalities
        if active not in MODALITIES:
            raise ValueError("invalid encoded modality mask")
        if self.training and (active != self.audit_config["active_modalities"] or availability is not None):
            raise ValueError("inference mask override is Dev-only")
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
        diagnostics = {}
        h0 = self.fusion(text=encoded.text, audio=encoded.audio, visual=encoded.visual,
                         attention_mask=attention_mask, availability=available.expand(*mask.shape, -1),
                         diagnostics=None if self.training else diagnostics)
        encoded_parts = {key: value.masked_fill(~(mask & available[..., index])[..., None], 0.0)
                         for index, (key, value) in enumerate(zip(BLOCK_ORDER,
                                                                 (encoded.audio, encoded.visual, encoded.text)))}
        adjacency = build_causal_utterance_adjacency(attention_mask, lengths, window_past=None, self_loop=False)
        relations = self.relation_builder.build(speaker_ids_int, attention_mask)
        states = (h0, *self.graph_core(h0, adjacency, relations, attention_mask))
        representation = torch.cat(states, -1)
        logits = self.classifier(representation) * mask[..., None]
        return FusionTAVOutput(representation, logits, torch.softmax(logits, -1), mask,
                               states, encoded_parts, adjacency, diagnostics)

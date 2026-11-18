"""Historical causal TAV frontend and classifier, with shared graph cores."""

from copy import deepcopy
from dataclasses import dataclass

import torch
import yaml

from models.common.causal_graph.graph_builders import build_causal_utterance_adjacency
from models.multidag_cl.graph_core_variants.coop import CoopCore
from models.multidag_cl.graph_core_variants.track import TrackCore
from models.multidag_cl.graph_core_variants.dcn import DCNCore
from models.multidag_cl.paper_reimplementation.config import MultiDAGCLConfig
from models.multidag_cl.paper_reimplementation.model import MultiDAGCLPaperReimplementation
from scripts.runtime.multidag_cl_tav_full_audit_v1.spec import CONTEXTS, MODALITIES, ROOT, SOURCE, validate_config


@dataclass(frozen=True)
class TAVOutput:
    h_ta: torch.Tensor
    logits_ta: torch.Tensor
    prob_ta: torch.Tensor
    mask: torch.Tensor
    layer_states: tuple
    encoded_modalities: dict
    adjacency: torch.Tensor


class TAVFullAudit(MultiDAGCLPaperReimplementation):
    def __init__(self, raw):
        raw = validate_config(raw)
        core = deepcopy(yaml.safe_load((ROOT / SOURCE).read_text(encoding="utf-8"))["model_core"])
        # Encoder mask is applied below so a Dev-only override never changes
        # the immutable training/checkpoint identity.
        core["modality_ablation"] = {"enabled": False, "active_modalities": ["text", "audio", "visual"]}
        # The source config describes reusable modules only. Actual CL OFF/ON
        # is owned by audit_config and CurriculumRuntime, never model forward.
        super().__init__(MultiDAGCLConfig.from_mapping(core))
        self.audit_config = raw
        if raw["graph_core"]["enabled"]:
            del self.graph_layers
            self.graph_core = {"coop": CoopCore, "track": TrackCore, "dcn": DCNCore}[raw["graph_core"]["variant"]]()

    def get_extra_state(self):
        return {"schema": "tav_full_audit_model_v1", "audit_config": deepcopy(self.audit_config)}

    def set_extra_state(self, state):
        if state != self.get_extra_state():
            raise ValueError("TAV checkpoint architecture/training mask identity mismatch")

    def load_state_dict(self, state_dict, strict=True, assign=False):
        if state_dict.get("_extra_state") != self.get_extra_state():
            raise ValueError("TAV checkpoint architecture/training mask identity mismatch")
        # Parent checks its historical signature, which is deliberately a
        # different lineage. The nn.Module strict copier is shared unchanged.
        return torch.nn.Module.load_state_dict(self, state_dict, strict=strict, assign=assign)

    def forward(self, *, text_features, audio_features, visual_features, lengths,
                speaker_ids_int, attention_mask, active_modalities=None):
        active = self.audit_config["active_modalities"] if active_modalities is None else active_modalities
        if active not in MODALITIES:
            raise ValueError("invalid encoded modality mask")
        if self.training and active != self.audit_config["active_modalities"]:
            raise ValueError("inference mask override is Dev-only")
        mask = self._validate_common_inputs(text_features, audio_features, visual_features,
                                            attention_mask, lengths, speaker_ids_int, None)
        encoded = self.paper_encoder(text_features=text_features, audio_features=audio_features,
                                    visual_features=visual_features, lengths=lengths, attention_mask=attention_mask)
        parts = {"A": encoded.audio * int("A" in active), "V": encoded.visual * int("V" in active),
                 "T": encoded.text * int("T" in active)}
        h0 = torch.cat((parts["A"], parts["V"], parts["T"]), -1)
        adjacency = build_causal_utterance_adjacency(
            attention_mask, lengths, window_past=CONTEXTS[self.audit_config["context_mode"]], self_loop=False)
        relations = self.relation_builder.build(speaker_ids_int, attention_mask)
        states = [h0]
        if self.audit_config["graph_core"]["enabled"]:
            states.extend(self.graph_core(h0, adjacency, relations, attention_mask))
        else:
            for layer in self.graph_layers:
                states.append(layer(states[-1], adjacency, relations, attention_mask).state)
        representation = torch.cat(states, -1)
        logits = self.classifier(representation) * mask[..., None]
        return TAVOutput(representation, logits, torch.softmax(logits, -1), mask, tuple(states), parts, adjacency)

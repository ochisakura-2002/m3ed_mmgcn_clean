"""Canonical utterance-order causal TA baseline; no wall-clock streaming claim."""

from dataclasses import asdict, dataclass

import torch
from torch import nn

from models.multidag_cl.paper_reimplementation.dag_layer import MultiDAGLayer
from models.multidag_cl.paper_reimplementation.graph import (
    CausalPredecessorBuilder, SpeakerRelationBuilder,
)
from .config import LABEL_VOCAB, TAV0Config
from .encoders import TAEncoders


@dataclass(frozen=True)
class TAOutput:
    h_ta: torch.Tensor
    logits_ta: torch.Tensor
    prob_ta: torch.Tensor
    mask: torch.Tensor


class TAV0(nn.Module):
    """Trainable T/A encoders followed by the unchanged historical DAG components."""

    def __init__(self, config=None):
        super().__init__()
        self.config = config if config is not None else TAV0Config()
        if not isinstance(self.config, TAV0Config):
            raise TypeError("config must be TAV0Config")
        c = self.config
        self.encoders = TAEncoders(c)
        self.ta_fusion_projection = nn.Linear(2 * c.encoder_dim, c.dag_hidden_dim)
        self.predecessor_builder = CausalPredecessorBuilder(c.window_past_same_speaker)
        self.relation_builder = SpeakerRelationBuilder()
        self.graph_layers = nn.ModuleList(MultiDAGLayer(c.dag_hidden_dim) for _ in range(c.dag_layers))
        self.classifier = nn.Sequential(
            nn.Linear(c.representation_dim, 300), nn.ReLU(),
            nn.Linear(300, 300), nn.ReLU(), nn.Dropout(c.classifier_dropout),
            nn.Linear(300, c.num_classes),
        )

    def get_extra_state(self):
        return {"lineage": "ta_v0", "config": asdict(self.config), "label_vocab": LABEL_VOCAB}

    def set_extra_state(self, state):
        if state != self.get_extra_state():
            raise ValueError("TA-v0 checkpoint identity/config/label vocabulary mismatch")

    def load_state_dict(self, state_dict, strict=True, assign=False):
        # Check before copying any parameters, including when strict=False.
        if state_dict.get("_extra_state") != self.get_extra_state():
            raise ValueError("TA-v0 checkpoint identity/config/label vocabulary mismatch")
        return super().load_state_dict(state_dict, strict=strict, assign=assign)

    def forward(self, *, text_features, audio_features, speaker_ids_int, lengths, attention_mask):
        if attention_mask.ndim != 2 or attention_mask.dtype not in (torch.bool, torch.int64):
            raise TypeError("attention_mask must be bool/int64 [B,L]")
        b, length = attention_mask.shape
        if b < 1 or length < 1:
            raise ValueError("empty batch/dialogue")
        if not torch.all((attention_mask == 0) | (attention_mask == 1)):
            raise ValueError("attention_mask must contain only 0/1")
        mask = attention_mask.bool()
        if lengths.dtype != torch.int64 or lengths.shape != (b,):
            raise TypeError("lengths must be int64 [B]")
        if speaker_ids_int.dtype != torch.int64 or speaker_ids_int.shape != (b, length):
            raise TypeError("speaker_ids_int must be int64 [B,L]")
        values = (text_features, audio_features, speaker_ids_int, lengths, attention_mask)
        if len({x.device for x in values}) != 1:
            raise ValueError("all inputs must share one device")
        if torch.any(lengths < 1) or torch.any(lengths > length):
            raise ValueError("lengths must be in [1,L]")
        expected = torch.arange(length, device=mask.device)[None, :] < lengths[:, None]
        if not torch.equal(mask, expected):
            raise ValueError("mask must be a contiguous prefix matching lengths")
        for name, value, dim in (("text", text_features, self.config.text_dim),
                                 ("audio", audio_features, self.config.audio_dim)):
            if value.dtype != torch.float32 or value.shape != (b, length, dim):
                raise TypeError(f"{name} must be float32 [B,L,{dim}]")
            if not torch.isfinite(value).all() or torch.count_nonzero(value[~mask]):
                raise ValueError(f"{name} must be finite with exact zero padding")
        if torch.any(speaker_ids_int[mask] < 0):
            raise ValueError("valid speaker IDs must be nonnegative")
        text, audio = self.encoders(text_features, audio_features, lengths, mask)
        h0 = self.ta_fusion_projection(torch.cat((text, audio), dim=-1)) * mask.unsqueeze(-1)
        adjacency = self.predecessor_builder.build(speaker_ids_int, attention_mask)
        relations = self.relation_builder.build(speaker_ids_int, attention_mask)
        states = [h0]
        for layer in self.graph_layers:
            states.append(layer(states[-1], adjacency, relations, attention_mask).state)
        h_ta = torch.cat(states, dim=-1)
        logits = self.classifier(h_ta) * mask.unsqueeze(-1)
        # Padding has a finite uniform distribution; mask excludes it from all consumers.
        return TAOutput(h_ta, logits, torch.softmax(logits, dim=-1), mask)

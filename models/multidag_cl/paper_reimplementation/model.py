"""Profile-fixed composition of the independent MultiDAG-CL core."""

from __future__ import annotations

from typing import Any, Mapping, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ConformanceProfile, MultiDAGCLConfig
from .contracts import (
    ContextVisibilityIdentity,
    MISSING_LABEL_INDEX,
    ModelDiagnostics,
    ModelOutput,
)
from .dag_layer import MultiDAGLayer
from .encoders import (
    OfficialSourceInputAdapter,
    OfficialSourceSingleProjectionEncoder,
    PaperFormulaModalityEncoder,
)
from .graph import CausalPredecessorBuilder, SpeakerRelationBuilder


class MultiDAGCLPaperReimplementation(nn.Module):
    """Independent paper reimplementation with a construction-fixed profile."""

    model_key = "multidag_cl_paper_reimplementation"
    implementation_identity = "paper_reimplementation"

    def __init__(
        self,
        config: MultiDAGCLConfig,
        *,
        collect_attention_diagnostics: bool = False,
    ) -> None:
        super().__init__()
        if not isinstance(config, MultiDAGCLConfig):
            raise TypeError("config must be MultiDAGCLConfig")
        if not isinstance(collect_attention_diagnostics, bool):
            raise TypeError("collect_attention_diagnostics must be bool")
        self.config = config
        self.collect_attention_diagnostics = collect_attention_diagnostics

        if config.conformance_profile is ConformanceProfile.PAPER_FORMULA_BEHAVIOR:
            self.paper_encoder: Optional[PaperFormulaModalityEncoder] = (
                PaperFormulaModalityEncoder(config)
            )
            self.source_input_adapter: Optional[OfficialSourceInputAdapter] = None
            self.source_encoder: Optional[OfficialSourceSingleProjectionEncoder] = None
            classifier_input_dim = (config.graph_layers + 1) * config.hidden_dim
        else:
            self.paper_encoder = None
            self.source_input_adapter = OfficialSourceInputAdapter(config)
            self.source_encoder = OfficialSourceSingleProjectionEncoder(config)
            raw_dim = (
                config.text_feature_dim
                + config.audio_feature_dim
                + config.visual_feature_dim
            )
            classifier_input_dim = (config.graph_layers + 1) * config.hidden_dim + raw_dim

        self.predecessor_builder = CausalPredecessorBuilder(
            config.window_past_same_speaker,
            config.predecessor_profile,
        )
        self.relation_builder = SpeakerRelationBuilder()
        self.graph_layers = nn.ModuleList(
            MultiDAGLayer(
                config.hidden_dim,
                collect_diagnostics=collect_attention_diagnostics,
            )
            for _ in range(config.graph_layers)
        )
        self.classifier_input_dim = classifier_input_dim
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, config.classifier_hidden_dim),
            nn.ReLU(),
            nn.Linear(config.classifier_hidden_dim, config.classifier_hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.classifier_dropout),
            nn.Linear(config.classifier_hidden_dim, config.num_classes),
        )

    def _model_signature(self) -> dict[str, Any]:
        return {
            "canonical_name": self.config.canonical_name,
            "implementation_identity": self.config.implementation_identity,
            "conformance_profile": self.config.conformance_profile.value,
            "encoder_profile": self.config.encoder_profile.value,
            "feature_dims": [
                self.config.text_feature_dim,
                self.config.audio_feature_dim,
                self.config.visual_feature_dim,
            ],
            "num_classes": self.config.num_classes,
            "hidden_dim": self.config.hidden_dim,
            "graph_layers": self.config.graph_layers,
            "causal_text_ablation": self.config.causal_text_ablation,
            "representation": self.config.representation,
            "classifier_input_dim": self.classifier_input_dim,
        }

    def get_extra_state(self) -> dict[str, Any]:
        return {"model_signature": self._model_signature()}

    def set_extra_state(self, state: Any) -> None:
        if not isinstance(state, Mapping) or state.get("model_signature") != self._model_signature():
            raise ValueError("state_dict model/profile identity does not match this model")

    def load_state_dict(
        self,
        state_dict: Mapping[str, Any],
        strict: bool = True,
        assign: bool = False,
    ):
        extra_state = state_dict.get("_extra_state")
        if not isinstance(extra_state, Mapping):
            raise ValueError("state_dict is missing the required model/profile identity")
        if extra_state.get("model_signature") != self._model_signature():
            raise ValueError("state_dict model/profile identity mismatch")
        return super().load_state_dict(state_dict, strict=strict, assign=assign)

    def _validate_common_inputs(
        self,
        text_features: torch.Tensor,
        audio_features: torch.Tensor,
        visual_features: torch.Tensor,
        attention_mask: torch.Tensor,
        lengths: torch.Tensor,
        speaker_ids_int: torch.Tensor,
        labels: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if attention_mask.dtype not in (torch.int64, torch.bool) or attention_mask.dim() != 2:
            raise TypeError("attention_mask must be int64 or bool [B,T]")
        batch_time = attention_mask.shape
        if lengths.dtype is not torch.int64 or lengths.shape != (batch_time[0],):
            raise TypeError("lengths must be int64 [B]")
        if speaker_ids_int.dtype is not torch.int64 or speaker_ids_int.shape != batch_time:
            raise TypeError("speaker_ids_int must be int64 [B,T]")
        tensor_inputs = (
            text_features,
            audio_features,
            visual_features,
            attention_mask,
            lengths,
            speaker_ids_int,
        )
        if labels is not None:
            tensor_inputs = tensor_inputs + (labels,)
        if len({value.device for value in tensor_inputs}) != 1:
            raise ValueError("all model inputs must share one device")
        if labels is not None and (labels.dtype is not torch.int64 or labels.shape != batch_time):
            raise TypeError("labels must be int64 [B,T]")
        mask = attention_mask.bool()
        if not torch.equal(mask.long().sum(dim=1), lengths):
            raise ValueError("lengths must equal attention_mask sums")
        max_time = batch_time[1]
        positions = torch.arange(max_time, device=lengths.device).unsqueeze(0)
        if not torch.equal(mask, positions < lengths.unsqueeze(1)):
            raise ValueError("attention_mask must be a contiguous valid prefix")
        if labels is not None:
            padded = ~mask
            if not torch.all(labels[padded] == self.config.loss_ignore_index):
                raise ValueError("padded labels must be -100")
            valid_labels = labels[mask]
            legal = (valid_labels == self.config.loss_ignore_index) | (
                valid_labels == MISSING_LABEL_INDEX
            ) | (
                (valid_labels >= 0) & (valid_labels < self.config.num_classes)
            )
            if not torch.all(legal):
                raise ValueError("labels must be -1, -100, or in [0,num_classes)")
        return mask

    def forward(
        self,
        *,
        text_features: torch.Tensor,
        audio_features: torch.Tensor,
        visual_features: torch.Tensor,
        attention_mask: torch.Tensor,
        lengths: torch.Tensor,
        speaker_ids_int: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> ModelOutput:
        mask = self._validate_common_inputs(
            text_features,
            audio_features,
            visual_features,
            attention_mask,
            lengths,
            speaker_ids_int,
            labels,
        )

        if self.config.conformance_profile is ConformanceProfile.PAPER_FORMULA_BEHAVIOR:
            if self.paper_encoder is None:
                raise RuntimeError("paper encoder was not constructed")
            encoded_modalities = self.paper_encoder(
                text_features=text_features,
                audio_features=audio_features,
                visual_features=visual_features,
                lengths=lengths,
                attention_mask=attention_mask,
            )
            initial_state = encoded_modalities.fused
            raw_concat = None
            context_identity = encoded_modalities.context_visibility_identity
        else:
            if self.source_input_adapter is None or self.source_encoder is None:
                raise RuntimeError("source behavior encoder was not constructed")
            raw_concat = self.source_input_adapter.pack(
                text_features=text_features,
                audio_features=audio_features,
                visual_features=visual_features,
                attention_mask=attention_mask,
            )
            initial_state = self.source_encoder(raw_concat, attention_mask)
            encoded_modalities = None
            context_identity = ContextVisibilityIdentity(
                dag_topology_causal=True,
                end_to_end_causal=False,
                end_to_end_causal_assuming_local_text_features=False,
                upstream_text_feature_locality="not_evaluated_by_model_core",
                context_leakage_risk=("unverified_source_upstream_feature_context",),
            )

        adjacency = self.predecessor_builder.build(speaker_ids_int, attention_mask)
        relations = self.relation_builder.build(speaker_ids_int, attention_mask)
        states = [initial_state]
        layer_results = []
        for layer in self.graph_layers:
            result = layer(states[-1], adjacency, relations, attention_mask)
            states.append(result.state)
            if self.collect_attention_diagnostics:
                layer_results.append(result)

        representation_parts = list(states)
        if raw_concat is not None:
            representation_parts.append(raw_concat)
        representation = torch.cat(representation_parts, dim=-1)
        representation = representation * mask.unsqueeze(-1).to(representation.dtype)
        logits = self.classifier(representation)
        logits = logits * mask.unsqueeze(-1).to(logits.dtype)

        loss = None
        if labels is not None:
            loss_positions = mask & (labels != self.config.loss_ignore_index)
            labeled = loss_positions & (labels != MISSING_LABEL_INDEX)
            if not bool(labeled.any().item()):
                raise ValueError("a labeled batch must contain at least one valid label")
            loss = F.cross_entropy(
                logits[loss_positions],
                labels[loss_positions],
                weight=None,
                ignore_index=MISSING_LABEL_INDEX,
                reduction="mean",
                label_smoothing=self.config.label_smoothing,
            )

        diagnostics = None
        if self.collect_attention_diagnostics:
            diagnostics = ModelDiagnostics(
                adjacency=adjacency,
                relations=relations,
                layer_diagnostics=tuple(layer_results),
            )
        return ModelOutput(
            logits=logits,
            loss=loss,
            encoded_state=initial_state,
            encoded_modalities=encoded_modalities,
            raw_concat=raw_concat,
            layer_states=tuple(states),
            representation=representation,
            profile_identity=self.config.conformance_profile.value,
            context_visibility_identity=context_identity,
            diagnostics=diagnostics,
        )


__all__ = ["MultiDAGCLPaperReimplementation"]

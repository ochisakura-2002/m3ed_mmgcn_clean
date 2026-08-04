"""Immutable configuration for the independent MultiDAG-CL reimplementation.

The values in this module encode the Stage-B1 frozen paper-formula and
released-source-behavior profiles.  This module intentionally performs no
YAML parsing, filesystem access, registry lookup, or runtime profile switch.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Mapping, Type, TypeVar


class ConformanceProfile(str, Enum):
    PAPER_FORMULA_BEHAVIOR = "paper_formula_behavior"
    OFFICIAL_SOURCE_BEHAVIOR = "official_source_behavior"


class EncoderProfile(str, Enum):
    PAPER_MODALITY_SPECIFIC = "paper_modality_specific"
    OFFICIAL_SOURCE_SINGLE_PROJECTION = "official_source_single_projection"


class DataTrack(str, Enum):
    PROJECT_FAIR = "project_fair"
    PAPER_DATA = "paper_data"


class PredecessorProfile(str, Enum):
    OFFICIAL_SAME_SPEAKER_COUNT_WINDOW = "official_same_speaker_count_window"


class CurriculumPartitionProfile(str, Enum):
    BALANCED_STABLE_CONTIGUOUS = "balanced_stable_contiguous"
    SOURCE_CEILING_CHUNKS = "source_ceiling_chunks"


class CurriculumScheduleProfile(str, Enum):
    OFFICIAL_ONE_BUCKET_PER_EPOCH = "official_one_bucket_per_epoch"


EnumType = TypeVar("EnumType", bound=Enum)


def _coerce_enum(value: Any, enum_type: Type[EnumType], name: str) -> EnumType:
    if isinstance(value, enum_type):
        return value
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string or {enum_type.__name__}")
    try:
        return enum_type(value)
    except ValueError as error:
        raise ValueError(f"unknown {name}: {value!r}") from error


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return value


def _section(config: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    if name not in config:
        raise ValueError(f"missing required config section: {name}")
    return _require_mapping(config[name], name)


def _only_keys(mapping: Mapping[str, Any], allowed: set[str], name: str) -> None:
    unknown = sorted(set(mapping) - allowed)
    if unknown:
        raise ValueError(f"unknown {name} fields: {unknown}")


def _require_bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be bool")
    return value


def _require_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be int")
    return value


def _require_float(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    return float(value)


@dataclass(frozen=True)
class MultiDAGCLConfig:
    """Fully resolved, construction-validated Stage-B2 model configuration."""

    canonical_name: str
    display_name: str
    implementation_identity: str
    conformance_profile: ConformanceProfile
    data_track: DataTrack

    text_feature_dim: int
    audio_feature_dim: int
    visual_feature_dim: int
    num_classes: int

    encoder_profile: EncoderProfile
    modality_order: tuple[str, str, str]
    audio_output_dim: int
    visual_output_dim: int
    text_output_dim: int
    text_sequence_axis: str
    text_bidirectional: bool
    causal_text_ablation: bool
    single_projection: bool

    predecessor_profile: PredecessorProfile
    window_past_same_speaker: int
    window_future: int
    allow_self_edge: bool
    allow_future_edge: bool
    global_nodal_attention: bool

    attention_score: str
    relation_on_values: bool
    attention_dropout: float

    hidden_dim: int
    graph_layers: int
    dual_gru: str
    layer_parameter_sharing: bool
    representation: str
    raw_feature_skip: bool

    classifier_hidden_dim: int
    classifier_hidden_layers: int
    classifier_activation: str
    classifier_dropout: float

    curriculum_enabled: bool
    bucket_count: int
    curriculum_partition: CurriculumPartitionProfile
    curriculum_schedule: CurriculumScheduleProfile

    training_epochs: int
    training_batch_size: int
    training_seed: int
    optimizer_name: str
    learning_rate: float
    optimizer_betas: tuple[float, float]
    optimizer_eps: float
    optimizer_weight_decay: float
    optimizer_bias_correction: bool
    optimizer_parameter_grouping: str
    gradient_clip_norm: float
    lr_scheduler: str
    amp: bool
    early_stopping: bool

    loss_name: str
    class_weight: None
    label_smoothing: float
    loss_ignore_index: int
    test_split_used_for_selection: bool

    def __post_init__(self) -> None:
        enum_fields = (
            ("conformance_profile", ConformanceProfile),
            ("data_track", DataTrack),
            ("encoder_profile", EncoderProfile),
            ("predecessor_profile", PredecessorProfile),
            ("curriculum_partition", CurriculumPartitionProfile),
            ("curriculum_schedule", CurriculumScheduleProfile),
        )
        for name, enum_type in enum_fields:
            object.__setattr__(
                self,
                name,
                _coerce_enum(getattr(self, name), enum_type, name),
            )
        object.__setattr__(self, "modality_order", tuple(self.modality_order))
        object.__setattr__(self, "optimizer_betas", tuple(self.optimizer_betas))
        self._validate_types()
        self._validate_semantics()

    def _validate_types(self) -> None:
        for name in (
            "canonical_name",
            "display_name",
            "implementation_identity",
            "text_sequence_axis",
            "attention_score",
            "dual_gru",
            "representation",
            "classifier_activation",
            "optimizer_name",
            "optimizer_parameter_grouping",
            "lr_scheduler",
            "loss_name",
        ):
            if not isinstance(getattr(self, name), str):
                raise TypeError(f"{name} must be str")
        for name in (
            "text_feature_dim",
            "audio_feature_dim",
            "visual_feature_dim",
            "num_classes",
            "audio_output_dim",
            "visual_output_dim",
            "text_output_dim",
            "window_past_same_speaker",
            "window_future",
            "hidden_dim",
            "graph_layers",
            "classifier_hidden_dim",
            "classifier_hidden_layers",
            "bucket_count",
            "training_epochs",
            "training_batch_size",
            "training_seed",
            "loss_ignore_index",
        ):
            _require_int(getattr(self, name), name)
        for name in (
            "text_bidirectional",
            "causal_text_ablation",
            "single_projection",
            "allow_self_edge",
            "allow_future_edge",
            "global_nodal_attention",
            "relation_on_values",
            "layer_parameter_sharing",
            "raw_feature_skip",
            "curriculum_enabled",
            "optimizer_bias_correction",
            "amp",
            "early_stopping",
            "test_split_used_for_selection",
        ):
            _require_bool(getattr(self, name), name)
        for name in (
            "attention_dropout",
            "classifier_dropout",
            "learning_rate",
            "optimizer_eps",
            "optimizer_weight_decay",
            "gradient_clip_norm",
            "label_smoothing",
        ):
            _require_float(getattr(self, name), name)
        if len(self.modality_order) != 3 or not all(
            isinstance(item, str) for item in self.modality_order
        ):
            raise TypeError("modality_order must contain exactly three names")
        if len(self.optimizer_betas) != 2:
            raise TypeError("optimizer_betas must contain exactly two values")
        for value in self.optimizer_betas:
            _require_float(value, "optimizer_betas item")

    def _validate_semantics(self) -> None:
        if self.canonical_name != "multidag_cl_paper_reimplementation":
            raise ValueError("canonical_name must identify the independent reimplementation")
        if self.display_name != "MultiDAG-CL Paper Reimplementation":
            raise ValueError("display_name must use the frozen reimplementation identity")
        if self.implementation_identity != "paper_reimplementation":
            raise ValueError("author_official and other implementation identities are forbidden")
        if "author_official" in (
            self.canonical_name,
            self.display_name,
            self.implementation_identity,
        ):
            raise ValueError("author_official identity claims are forbidden")

        for name in (
            "text_feature_dim",
            "audio_feature_dim",
            "visual_feature_dim",
            "num_classes",
            "hidden_dim",
            "classifier_hidden_dim",
            "classifier_hidden_layers",
            "bucket_count",
            "training_epochs",
            "training_batch_size",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.training_seed < 0:
            raise ValueError("training_seed must be non-negative")
        if self.window_past_same_speaker < 1:
            raise ValueError("window_past_same_speaker must be at least 1")
        if self.window_future != 0 or self.allow_future_edge:
            raise ValueError("future edges and nonzero future windows are forbidden")
        if self.allow_self_edge:
            raise ValueError("self edges are forbidden")
        if self.global_nodal_attention:
            raise ValueError("global nodal attention is forbidden")
        if self.attention_score != "concat_linear":
            raise ValueError("attention_score must be concat_linear")
        if not self.relation_on_values:
            raise ValueError("speaker relation must act on attention values")
        if self.attention_dropout != 0.0:
            raise ValueError("attention_dropout must be exactly zero")
        if self.hidden_dim != 300:
            raise ValueError("hidden_dim must be 300 for both frozen profiles")
        if self.graph_layers not in (1, 4):
            raise ValueError("graph_layers must be 4, with 1 allowed for component tests")
        if self.dual_gru != "swapped_input_hidden_sum":
            raise ValueError("dual_gru must be swapped_input_hidden_sum")
        if self.layer_parameter_sharing:
            raise ValueError("DAG layer parameter sharing is forbidden")
        if self.classifier_hidden_dim != 300 or self.classifier_hidden_layers != 2:
            raise ValueError("classifier must have two 300-dimensional hidden layers")
        if self.classifier_activation != "relu" or self.classifier_dropout != 0.4:
            raise ValueError("classifier must use ReLU and dropout 0.4")
        if self.loss_name != "cross_entropy" or self.class_weight is not None:
            raise ValueError("loss must be class-unweighted cross entropy")
        if self.label_smoothing != 0.0 or self.loss_ignore_index != -100:
            raise ValueError("loss smoothing must be 0 and ignore_index must be -100")
        if self.test_split_used_for_selection:
            raise ValueError("test selection fields are forbidden")
        if self.curriculum_schedule is not CurriculumScheduleProfile.OFFICIAL_ONE_BUCKET_PER_EPOCH:
            raise ValueError("only official_one_bucket_per_epoch is supported")
        self._validate_training_metadata()

        if self.conformance_profile is ConformanceProfile.PAPER_FORMULA_BEHAVIOR:
            self._validate_paper_profile()
        else:
            self._validate_source_profile()

    def _validate_training_metadata(self) -> None:
        expected = {
            "training_epochs": 30,
            "training_batch_size": 16,
            "training_seed": 100,
            "optimizer_name": "adamw_transformers_3_5_1_compatible",
            "optimizer_betas": (0.9, 0.999),
            "optimizer_eps": 1.0e-6,
            "optimizer_weight_decay": 0.0,
            "optimizer_bias_correction": True,
            "optimizer_parameter_grouping": "all_parameters_single_group",
            "gradient_clip_norm": 5.0,
            "lr_scheduler": "none",
            "amp": False,
            "early_stopping": False,
        }
        for name, value in expected.items():
            if getattr(self, name) != value:
                raise ValueError(f"{name} must equal frozen metadata value {value!r}")
        if self.learning_rate != 5.0e-4:
            raise ValueError("learning_rate must be 5e-4")

    def _validate_paper_profile(self) -> None:
        if self.encoder_profile is not EncoderProfile.PAPER_MODALITY_SPECIFIC:
            raise ValueError("paper_formula_behavior requires the paper modality encoder")
        if self.modality_order != ("audio", "visual", "text"):
            raise ValueError("paper_formula_behavior requires explicit audio,visual,text order")
        if (
            self.audio_output_dim,
            self.visual_output_dim,
            self.text_output_dim,
        ) != (100, 100, 100):
            raise ValueError("paper modality outputs must be 100/100/100")
        if sum((self.audio_output_dim, self.visual_output_dim, self.text_output_dim)) != self.hidden_dim:
            raise ValueError("paper encoder output dimensions must sum to hidden_dim")
        if self.text_sequence_axis != "dialogue_utterance":
            raise ValueError("paper text sequence axis must be dialogue_utterance")
        if self.causal_text_ablation:
            if self.text_bidirectional:
                raise ValueError("causal text ablation must use a unidirectional LSTM")
        elif not self.text_bidirectional:
            raise ValueError("primary paper encoder must use the frozen bidirectional LSTM")
        if self.single_projection:
            raise ValueError("paper profile cannot use the source single projection")
        if self.raw_feature_skip:
            raise ValueError("paper profile raw feature skip is forbidden")
        if self.representation != "encoder_plus_all_dag_layers":
            raise ValueError("paper profile representation must exclude raw features")
        if self.bucket_count != 5:
            raise ValueError("paper profile bucket_count must be 5")
        if self.curriculum_partition is not CurriculumPartitionProfile.BALANCED_STABLE_CONTIGUOUS:
            raise ValueError("paper profile requires balanced stable contiguous buckets")

    def _validate_source_profile(self) -> None:
        if self.encoder_profile is not EncoderProfile.OFFICIAL_SOURCE_SINGLE_PROJECTION:
            raise ValueError("source behavior requires its isolated single-projection encoder")
        if self.modality_order != ("text", "audio", "visual"):
            raise ValueError("source behavior requires explicit text,audio,visual order")
        if (self.audio_output_dim, self.visual_output_dim, self.text_output_dim) != (0, 0, 0):
            raise ValueError("source behavior has no separate modality encoder outputs")
        if self.text_sequence_axis != "not_applicable" or self.text_bidirectional:
            raise ValueError("source behavior has no active recurrent text encoder")
        if self.causal_text_ablation:
            raise ValueError("causal text ablation belongs only to the paper encoder profile")
        if not self.single_projection:
            raise ValueError("source behavior requires a single 300-dimensional projection")
        if not self.raw_feature_skip:
            raise ValueError("source behavior requires its explicit raw feature skip")
        if self.representation != "encoder_plus_all_dag_layers_plus_raw_features":
            raise ValueError("source representation must include raw features")
        if self.bucket_count != 12:
            raise ValueError("source behavior default bucket_count must be 12")
        if self.curriculum_partition is not CurriculumPartitionProfile.SOURCE_CEILING_CHUNKS:
            raise ValueError("source behavior requires ceiling-chunk partitioning")

    @classmethod
    def from_mapping(cls, config: Mapping[str, Any]) -> "MultiDAGCLConfig":
        """Build a validated config from primitive nested mappings only."""

        root = _require_mapping(config, "config")
        _only_keys(
            root,
            {
                "identity",
                "data",
                "encoder",
                "graph",
                "attention",
                "dag",
                "classifier",
                "curriculum",
                "training",
                "loss",
                "checkpoint",
            },
            "top-level config",
        )
        identity = _section(root, "identity")
        data = _section(root, "data")
        encoder = _section(root, "encoder")
        graph = _section(root, "graph")
        attention = _section(root, "attention")
        dag = _section(root, "dag")
        classifier = _section(root, "classifier")
        curriculum = _section(root, "curriculum")
        training = _section(root, "training")
        loss = _section(root, "loss")
        checkpoint = _section(root, "checkpoint")
        optimizer = _section(training, "optimizer")
        output_dims = _require_mapping(encoder.get("modality_output_dims"), "encoder.modality_output_dims")

        _only_keys(identity, {"canonical_name", "display_name", "implementation_identity", "conformance_profile"}, "identity")
        _only_keys(data, {"track", "text_feature_dim", "audio_feature_dim", "visual_feature_dim", "num_classes"}, "data")
        _only_keys(encoder, {"profile", "modality_order", "modality_output_dims", "text_sequence_axis", "text_bidirectional", "causal_text_ablation", "single_projection"}, "encoder")
        _only_keys(output_dims, {"audio", "visual", "text"}, "encoder.modality_output_dims")
        _only_keys(graph, {"predecessor_profile", "window_past_same_speaker", "window_future", "allow_self_edge", "allow_future_edge", "global_nodal_attention"}, "graph")
        _only_keys(attention, {"score", "relation_on_values", "dropout"}, "attention")
        _only_keys(dag, {"hidden_dim", "layers", "dual_gru", "layer_parameter_sharing", "representation", "raw_feature_skip"}, "dag")
        _only_keys(classifier, {"hidden_dim", "hidden_layers", "activation", "dropout"}, "classifier")
        _only_keys(curriculum, {"enabled", "bucket_count", "partition", "schedule"}, "curriculum")
        _only_keys(training, {"epochs", "batch_size", "seed", "optimizer", "gradient_clip_norm", "scheduler", "amp", "early_stopping"}, "training")
        _only_keys(optimizer, {"name", "learning_rate", "betas", "eps", "weight_decay", "bias_correction", "parameter_grouping"}, "training.optimizer")
        _only_keys(loss, {"name", "class_weight", "label_smoothing", "ignore_index"}, "loss")
        _only_keys(checkpoint, {"test_split_used_for_selection"}, "checkpoint")

        modality_order = encoder.get("modality_order")
        if not isinstance(modality_order, (list, tuple)):
            raise TypeError("encoder.modality_order must be a primitive sequence")
        betas = optimizer.get("betas")
        if not isinstance(betas, (list, tuple)):
            raise TypeError("training.optimizer.betas must be a primitive sequence")

        return cls(
            canonical_name=identity.get("canonical_name"),
            display_name=identity.get("display_name"),
            implementation_identity=identity.get("implementation_identity"),
            conformance_profile=identity.get("conformance_profile"),
            data_track=data.get("track"),
            text_feature_dim=_require_int(data.get("text_feature_dim"), "data.text_feature_dim"),
            audio_feature_dim=_require_int(data.get("audio_feature_dim"), "data.audio_feature_dim"),
            visual_feature_dim=_require_int(data.get("visual_feature_dim"), "data.visual_feature_dim"),
            num_classes=_require_int(data.get("num_classes"), "data.num_classes"),
            encoder_profile=encoder.get("profile"),
            modality_order=tuple(modality_order),
            audio_output_dim=_require_int(output_dims.get("audio"), "encoder.modality_output_dims.audio"),
            visual_output_dim=_require_int(output_dims.get("visual"), "encoder.modality_output_dims.visual"),
            text_output_dim=_require_int(output_dims.get("text"), "encoder.modality_output_dims.text"),
            text_sequence_axis=encoder.get("text_sequence_axis"),
            text_bidirectional=_require_bool(encoder.get("text_bidirectional"), "encoder.text_bidirectional"),
            causal_text_ablation=_require_bool(encoder.get("causal_text_ablation"), "encoder.causal_text_ablation"),
            single_projection=_require_bool(encoder.get("single_projection"), "encoder.single_projection"),
            predecessor_profile=graph.get("predecessor_profile"),
            window_past_same_speaker=_require_int(graph.get("window_past_same_speaker"), "graph.window_past_same_speaker"),
            window_future=_require_int(graph.get("window_future"), "graph.window_future"),
            allow_self_edge=_require_bool(graph.get("allow_self_edge"), "graph.allow_self_edge"),
            allow_future_edge=_require_bool(graph.get("allow_future_edge"), "graph.allow_future_edge"),
            global_nodal_attention=_require_bool(graph.get("global_nodal_attention"), "graph.global_nodal_attention"),
            attention_score=attention.get("score"),
            relation_on_values=_require_bool(attention.get("relation_on_values"), "attention.relation_on_values"),
            attention_dropout=_require_float(attention.get("dropout"), "attention.dropout"),
            hidden_dim=_require_int(dag.get("hidden_dim"), "dag.hidden_dim"),
            graph_layers=_require_int(dag.get("layers"), "dag.layers"),
            dual_gru=dag.get("dual_gru"),
            layer_parameter_sharing=_require_bool(dag.get("layer_parameter_sharing"), "dag.layer_parameter_sharing"),
            representation=dag.get("representation"),
            raw_feature_skip=_require_bool(dag.get("raw_feature_skip"), "dag.raw_feature_skip"),
            classifier_hidden_dim=_require_int(classifier.get("hidden_dim"), "classifier.hidden_dim"),
            classifier_hidden_layers=_require_int(classifier.get("hidden_layers"), "classifier.hidden_layers"),
            classifier_activation=classifier.get("activation"),
            classifier_dropout=_require_float(classifier.get("dropout"), "classifier.dropout"),
            curriculum_enabled=_require_bool(curriculum.get("enabled"), "curriculum.enabled"),
            bucket_count=_require_int(curriculum.get("bucket_count"), "curriculum.bucket_count"),
            curriculum_partition=curriculum.get("partition"),
            curriculum_schedule=curriculum.get("schedule"),
            training_epochs=_require_int(training.get("epochs"), "training.epochs"),
            training_batch_size=_require_int(training.get("batch_size"), "training.batch_size"),
            training_seed=_require_int(training.get("seed"), "training.seed"),
            optimizer_name=optimizer.get("name"),
            learning_rate=_require_float(optimizer.get("learning_rate"), "training.optimizer.learning_rate"),
            optimizer_betas=tuple(_require_float(value, "training.optimizer.betas item") for value in betas),
            optimizer_eps=_require_float(optimizer.get("eps"), "training.optimizer.eps"),
            optimizer_weight_decay=_require_float(optimizer.get("weight_decay"), "training.optimizer.weight_decay"),
            optimizer_bias_correction=_require_bool(optimizer.get("bias_correction"), "training.optimizer.bias_correction"),
            optimizer_parameter_grouping=optimizer.get("parameter_grouping"),
            gradient_clip_norm=_require_float(training.get("gradient_clip_norm"), "training.gradient_clip_norm"),
            lr_scheduler=training.get("scheduler"),
            amp=_require_bool(training.get("amp"), "training.amp"),
            early_stopping=_require_bool(training.get("early_stopping"), "training.early_stopping"),
            loss_name=loss.get("name"),
            class_weight=loss.get("class_weight"),
            label_smoothing=_require_float(loss.get("label_smoothing"), "loss.label_smoothing"),
            loss_ignore_index=_require_int(loss.get("ignore_index"), "loss.ignore_index"),
            test_split_used_for_selection=_require_bool(checkpoint.get("test_split_used_for_selection"), "checkpoint.test_split_used_for_selection"),
        )

    def to_mapping(self) -> dict[str, Any]:
        """Serialize to JSON/YAML-compatible primitive values without I/O."""

        raw = asdict(self)
        for name in (
            "conformance_profile",
            "data_track",
            "encoder_profile",
            "predecessor_profile",
            "curriculum_partition",
            "curriculum_schedule",
        ):
            raw[name] = getattr(self, name).value
        return {
            "identity": {
                "canonical_name": raw["canonical_name"],
                "display_name": raw["display_name"],
                "implementation_identity": raw["implementation_identity"],
                "conformance_profile": raw["conformance_profile"],
            },
            "data": {
                "track": raw["data_track"],
                "text_feature_dim": raw["text_feature_dim"],
                "audio_feature_dim": raw["audio_feature_dim"],
                "visual_feature_dim": raw["visual_feature_dim"],
                "num_classes": raw["num_classes"],
            },
            "encoder": {
                "profile": raw["encoder_profile"],
                "modality_order": list(raw["modality_order"]),
                "modality_output_dims": {
                    "audio": raw["audio_output_dim"],
                    "visual": raw["visual_output_dim"],
                    "text": raw["text_output_dim"],
                },
                "text_sequence_axis": raw["text_sequence_axis"],
                "text_bidirectional": raw["text_bidirectional"],
                "causal_text_ablation": raw["causal_text_ablation"],
                "single_projection": raw["single_projection"],
            },
            "graph": {
                "predecessor_profile": raw["predecessor_profile"],
                "window_past_same_speaker": raw["window_past_same_speaker"],
                "window_future": raw["window_future"],
                "allow_self_edge": raw["allow_self_edge"],
                "allow_future_edge": raw["allow_future_edge"],
                "global_nodal_attention": raw["global_nodal_attention"],
            },
            "attention": {
                "score": raw["attention_score"],
                "relation_on_values": raw["relation_on_values"],
                "dropout": raw["attention_dropout"],
            },
            "dag": {
                "hidden_dim": raw["hidden_dim"],
                "layers": raw["graph_layers"],
                "dual_gru": raw["dual_gru"],
                "layer_parameter_sharing": raw["layer_parameter_sharing"],
                "representation": raw["representation"],
                "raw_feature_skip": raw["raw_feature_skip"],
            },
            "classifier": {
                "hidden_dim": raw["classifier_hidden_dim"],
                "hidden_layers": raw["classifier_hidden_layers"],
                "activation": raw["classifier_activation"],
                "dropout": raw["classifier_dropout"],
            },
            "curriculum": {
                "enabled": raw["curriculum_enabled"],
                "bucket_count": raw["bucket_count"],
                "partition": raw["curriculum_partition"],
                "schedule": raw["curriculum_schedule"],
            },
            "training": {
                "epochs": raw["training_epochs"],
                "batch_size": raw["training_batch_size"],
                "seed": raw["training_seed"],
                "optimizer": {
                    "name": raw["optimizer_name"],
                    "learning_rate": raw["learning_rate"],
                    "betas": list(raw["optimizer_betas"]),
                    "eps": raw["optimizer_eps"],
                    "weight_decay": raw["optimizer_weight_decay"],
                    "bias_correction": raw["optimizer_bias_correction"],
                    "parameter_grouping": raw["optimizer_parameter_grouping"],
                },
                "gradient_clip_norm": raw["gradient_clip_norm"],
                "scheduler": raw["lr_scheduler"],
                "amp": raw["amp"],
                "early_stopping": raw["early_stopping"],
            },
            "loss": {
                "name": raw["loss_name"],
                "class_weight": raw["class_weight"],
                "label_smoothing": raw["label_smoothing"],
                "ignore_index": raw["loss_ignore_index"],
            },
            "checkpoint": {
                "test_split_used_for_selection": raw["test_split_used_for_selection"],
            },
        }


__all__ = [
    "ConformanceProfile",
    "CurriculumPartitionProfile",
    "CurriculumScheduleProfile",
    "DataTrack",
    "EncoderProfile",
    "MultiDAGCLConfig",
    "PredecessorProfile",
]

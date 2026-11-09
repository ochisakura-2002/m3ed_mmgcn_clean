"""Frozen current-protocol TA-v0 architecture (separate checkpoint lineage)."""

from dataclasses import dataclass

LABEL_VOCAB = ("hap", "sad", "neu", "ang", "exc", "fru")


@dataclass(frozen=True)
class TAV0Config:
    text_dim: int = 768
    audio_dim: int = 1582
    encoder_dim: int = 100
    dag_hidden_dim: int = 300
    dag_layers: int = 4
    num_classes: int = 6
    window_past_same_speaker: int = 1
    classifier_dropout: float = 0.4

    def __post_init__(self):
        for name, expected in (
            ("text_dim", 768), ("audio_dim", 1582), ("encoder_dim", 100),
            ("dag_hidden_dim", 300), ("dag_layers", 4), ("num_classes", 6),
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) != expected:
                raise ValueError(f"TA-v0 freezes {name}={expected}")
        if type(self.window_past_same_speaker) is not int or self.window_past_same_speaker < 1:
            raise ValueError("window_past_same_speaker must be a positive integer")
        if (isinstance(self.classifier_dropout, bool)
                or not isinstance(self.classifier_dropout, (int, float))
                or not 0 <= self.classifier_dropout < 1):
            raise ValueError("classifier_dropout must lie in [0,1)")

    @property
    def representation_dim(self):
        return (self.dag_layers + 1) * self.dag_hidden_dim

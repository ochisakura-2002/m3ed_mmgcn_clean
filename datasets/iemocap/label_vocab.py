"""Explicit shared benchmark vocabulary; never infer indices by sorting labels."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping


@dataclass(frozen=True)
class LabelVocab:
    index_to_label: tuple[str, ...]
    source: str
    ta_mapping_verified: bool

    def __post_init__(self):
        if (not self.index_to_label or len(set(self.index_to_label)) != len(self.index_to_label)
                or any(not isinstance(x, str) or not x.strip() for x in self.index_to_label)):
            raise ValueError("Vocabulary must contain unique, nonempty label strings")
        if not self.source:
            raise ValueError("Vocabulary source must be explicit")
        if type(self.ta_mapping_verified) is not bool:
            raise ValueError("ta_mapping_verified must be a boolean")

    @property
    def label_to_index(self) -> Mapping[str, int]:
        return MappingProxyType({label: i for i, label in enumerate(self.index_to_label)})

    @property
    def num_classes(self) -> int:
        return len(self.index_to_label)

    def encode(self, benchmark_label: str) -> int:
        try:
            return self.label_to_index[benchmark_label]
        except KeyError as error:
            raise ValueError(f"Unknown benchmark_label: {benchmark_label!r}") from error

    def as_dict(self) -> dict:
        return {"target": "benchmark_label", "label_to_index": dict(self.label_to_index),
                "index_to_label": list(self.index_to_label), "source": self.source,
                "ta_mapping_verified": self.ta_mapping_verified}


def load_label_vocab(path: str | Path) -> LabelVocab:
    import yaml

    with Path(path).open(encoding="utf-8-sig") as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, dict) or value.get("target") != "benchmark_label":
        raise ValueError("S2D-V0 target must be benchmark_label")
    vocab = LabelVocab(tuple(value["index_to_label"]), value["source"],
                       value.get("ta_mapping_verified", False))
    mapping = value.get("label_to_index")
    if (mapping != dict(vocab.label_to_index)
            or any(type(i) is not int for i in mapping.values())):
        raise ValueError("label_to_index and index_to_label must be exact inverses")
    return vocab

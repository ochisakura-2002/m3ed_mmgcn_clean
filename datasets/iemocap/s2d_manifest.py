"""CSV-only adapter. The full MERC universe is retained separately from V rows."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from typing import Mapping

from .label_vocab import LabelVocab


SPLITS = ("train", "dev", "test")


@dataclass(frozen=True)
class VisualRecord:
    utterance_id: str
    video_path: Path | None
    benchmark_label: str
    label_index: int
    split: str
    v_available: bool


class S2DManifest:
    """Read provided split membership verbatim; no split generation or video IO.

    Required CSV columns: utterance_id, video_path, benchmark_label, split,
    v_available. label_index is derived from the explicit vocab and checked if
    present. ``columns`` explicitly maps logical names to actual CSV headers.
    """

    def __init__(self, manifest_path: str | Path, data_root: str | Path,
                 vocab: LabelVocab, columns: Mapping[str, str] | None = None):
        self.manifest_path = Path(manifest_path)
        self.data_root = Path(data_root).expanduser().resolve()
        names = {n: n for n in ("utterance_id", "video_path", "benchmark_label",
                               "label_index", "split", "v_available")}
        if columns:
            if set(columns) - set(names):
                raise ValueError(f"Unknown manifest column keys: {set(columns) - set(names)}")
            names.update(columns)
        records = []
        seen = set()
        with self.manifest_path.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            required = {names[n] for n in names if n != "label_index"}
            if not required.issubset(reader.fieldnames or ()):
                raise ValueError(f"Missing CSV columns: {required - set(reader.fieldnames or ())}")
            for line, row in enumerate(reader, 2):
                def get(key):
                    return (row.get(names[key]) or "").strip()
                uid, split, label = get("utterance_id"), get("split"), get("benchmark_label")
                if not uid or uid in seen:
                    raise ValueError(f"Missing/duplicate utterance_id at line {line}: {uid!r}")
                seen.add(uid)
                if split not in SPLITS:
                    raise ValueError(f"{uid}: split must be train/dev/test, got {split!r}")
                flag = get("v_available").lower()
                if flag not in ("0", "1", "false", "true"):
                    raise ValueError(f"{uid}: invalid v_available {flag!r}")
                available = flag in ("1", "true")
                index = vocab.encode(label)
                if names["label_index"] in row and get("label_index") != str(index):
                    raise ValueError(f"{uid}: label_index disagrees with benchmark_label/vocab")
                raw_path = get("video_path")
                if available and not raw_path:
                    raise ValueError(f"{uid}: available V requires video_path")
                path = None
                if raw_path:
                    # Accept portable relative CSV paths and native absolute paths.
                    # Reject a Windows absolute path on Linux rather than rebasing it silently.
                    path = Path(raw_path.replace("\\", "/"))
                    if PureWindowsPath(raw_path).is_absolute() and not path.is_absolute():
                        raise ValueError(f"{uid}: rebase non-native absolute path in the manifest")
                    path = (path if path.is_absolute() else self.data_root / path).resolve()
                records.append(VisualRecord(uid, path, label, index, split, available))
        if not records:
            raise ValueError("Empty manifest")
        self.records = tuple(records)
        self.by_id = {row.utterance_id: row for row in records}

    def for_split(self, split: str, *, available_only: bool = True) -> tuple[VisualRecord, ...]:
        if split not in SPLITS:
            raise ValueError(f"Unknown split: {split}")
        return tuple(r for r in self.records if r.split == split and (r.v_available or not available_only))

    def counts(self) -> dict:
        return {s: {"universe": len(self.for_split(s, available_only=False)),
                    "v_available": len(self.for_split(s)),
                    "v_unavailable": sum(not r.v_available for r in self.for_split(s, available_only=False))}
                for s in SPLITS}

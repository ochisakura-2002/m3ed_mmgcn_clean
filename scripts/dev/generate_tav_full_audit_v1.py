"""Materialize the frozen audit design into a new, independent config tree."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import yaml

from scripts.runtime.multidag_cl_tav_full_audit_v1.spec import (
    CONFIG_ROOT, TRAIN_MATRIX, EVAL_MATRIX, design, evaluation_design, make_config, write_csv,
)


def main():
    directory = ROOT / CONFIG_ROOT
    if directory.exists():
        raise FileExistsError("refusing to overwrite existing audit configuration")
    rows = design()
    for row in rows:
        path = ROOT / row["config_path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8") as stream:
            yaml.safe_dump(make_config(row["graph_core"], row["modalities"], row["curriculum"], row["context_mode"]),
                           stream, sort_keys=False)
    write_csv(directory / TRAIN_MATRIX, rows)
    write_csv(directory / EVAL_MATRIX, evaluation_design())
    print("FORMAL_TRAININGS=24 MISSING_MODALITY_DEV_EVALUATIONS=56")


if __name__ == "__main__":
    main()

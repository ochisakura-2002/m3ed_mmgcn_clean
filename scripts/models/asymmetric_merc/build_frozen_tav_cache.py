"""Build a new frozen TA/S2D-B exact-ID Train/Dev cache; never open Test."""

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.models.asymmetric_merc.frozen_tav_cache import build_cache


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("ta-train-npz", "ta-dev-npz", "s2d-bank-dir", "output-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--v-quadrant-predictions", type=Path,
                        help="Q2-1 frozen Linear-B seed100 Dev CSV/TSV; required for subsequent VRG training")
    parser.add_argument("--synthetic", action="store_true", help="Synthetic fixtures only; relax real row counts")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        result = build_cache(**vars(args))
    except (ValueError, OSError, KeyError, TypeError, RuntimeError) as error:
        print(f"CACHE FAILED: {error}", file=sys.stderr)
        return 1
    print({"status": result["status"], "synthetic": result["synthetic"], "joins": result["joins"]})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

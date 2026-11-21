"""Remote launcher for the seed-100 CT-MultiDAG topology retraining screen."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.workflows.benchmarks.run_ct_multidag_fusion_seed100 import main


if __name__ == "__main__":
    raise SystemExit(main(topology=True))

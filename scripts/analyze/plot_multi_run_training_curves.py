"""Compatibility wrapper for canonical multi-run training curves."""

from pathlib import Path
import sys


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scripts import _load_compat_module  # noqa: E402


_CANONICAL = _load_compat_module(
    __name__, "scripts.analysis.common.plot_multi_run_training_curves"
)

if __name__ == "__main__":
    raise SystemExit(_CANONICAL.main())

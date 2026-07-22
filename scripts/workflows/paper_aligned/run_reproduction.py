"""Compatibility entrypoint for the original-MERC reproduction pipeline."""

import sys
from pathlib import Path


PROJECT_ROOT = next(
    candidate
    for candidate in Path(__file__).resolve().parents
    if (candidate / "AGENTS.md").is_file() and (candidate / "scripts").is_dir()
)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.workflows.paper_aligned.run_pipeline import main


if __name__ == "__main__":
    main()

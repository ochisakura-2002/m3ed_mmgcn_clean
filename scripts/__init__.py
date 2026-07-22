"""Canonical execution-script packages and legacy-entrypoint compatibility."""

from importlib import import_module
import sys
from types import ModuleType


def _load_compat_module(legacy_name: str, canonical_name: str) -> ModuleType:
    """Load one canonical module and alias an imported legacy module to it."""
    module = import_module(canonical_name)
    if legacy_name != "__main__":
        sys.modules[legacy_name] = module
    return module

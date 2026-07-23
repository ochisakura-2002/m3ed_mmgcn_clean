"""
Environment check script for the M3ED + MMGCN project.

This script does not train models or load M3ED data.
It only checks whether the current project skeleton and Python environment are ready.
"""

from pathlib import Path
import sys
import os
import site
import importlib


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def print_section(title: str) -> None:
    """Print a section title."""
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def check_python_env() -> None:
    """Check Python executable, version, and conda environment."""
    print_section("Python environment")

    print("Project root:", PROJECT_ROOT)
    print("Python executable:", sys.executable)
    print("Python version:", sys.version.replace("\n", " "))
    print("CONDA_DEFAULT_ENV:", os.environ.get("CONDA_DEFAULT_ENV", "None"))
    print("PYTHONNOUSERSITE:", os.environ.get("PYTHONNOUSERSITE", "Not set"))


def check_user_site() -> None:
    """
    Check whether this conda environment is polluted by ~/.local packages.
    Old user-level packages may silently affect imports.
    """
    print_section("User site-packages check")

    print("site.ENABLE_USER_SITE:", site.ENABLE_USER_SITE)
    print("User site path:", site.getusersitepackages())

    local_paths = [p for p in sys.path if ".local" in p]

    if len(local_paths) == 0:
        print("[OK] No .local path found in sys.path.")
    else:
        print("[WARN] .local paths found in sys.path:")
        for p in local_paths:
            print("  ", p)


def check_import(display_name: str, import_name: str = None) -> None:
    """Try to import a package and print its version."""
    if import_name is None:
        import_name = display_name

    try:
        module = importlib.import_module(import_name)
        version = getattr(module, "__version__", "version_unknown")
        print(f"[OK] {display_name}: {version}")
    except ImportError:
        print(f"[MISS] {display_name}: not installed")


def check_basic_packages() -> None:
    """Check lightweight packages needed at the skeleton stage."""
    print_section("Basic package imports")

    check_import("numpy")
    check_import("pandas")
    check_import("pyyaml", "yaml")
    check_import("tqdm")
    check_import("scikit-learn", "sklearn")


def check_torch_cuda() -> None:
    """
    Check torch and CUDA status.
    Torch is optional at the current skeleton stage.
    """
    print_section("Torch / CUDA check")

    try:
        import torch

        print("[OK] torch:", torch.__version__)
        print("CUDA available:", torch.cuda.is_available())
        print("GPU count:", torch.cuda.device_count())

        if torch.cuda.is_available():
            print("GPU name:", torch.cuda.get_device_name(0))

    except ImportError:
        print("[MISS] torch: not installed yet")
        print("This is acceptable at the skeleton stage.")


def check_project_dirs() -> None:
    """Check whether required project directories exist."""
    print_section("Project directory check")

    expected_dirs = [
        "configs",
        "data/raw/M3ED",
        "data/processed",
        "data/metadata",
        "datasets",
        "models",
        "models/baselines",
        "models/baselines/mmgcn",
        "models/encoders",
        "models/fusion",
        "models/modules",
        "models/heads",
        "losses",
        "trainers",
        "scripts",
        "utils",
        "outputs",
        "third_party",
        "third_party/pretrained",
        "third_party/MMGCN",
        "docs",
        ".vscode",
    ]

    for rel_path in expected_dirs:
        path = PROJECT_ROOT / rel_path
        status = "OK" if path.exists() else "MISS"
        print(f"[{status}] {rel_path}")


def check_config() -> None:
    """Check whether the YAML config exists and can be parsed."""
    print_section("Config file check")

    config_path = PROJECT_ROOT / "configs" / "train_mmgcn_m3ed.yaml"
    print("Config path:", config_path)

    if not config_path.exists():
        print("[MISS] configs/mmgcn/unified/m3ed/full_context/m3ed_features/skeleton.yaml")
        return

    print("[OK] configs/mmgcn/unified/m3ed/full_context/m3ed_features/skeleton.yaml exists")

    try:
        import yaml

        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        print("[OK] YAML can be parsed")
        print("Project name:", config.get("project", {}).get("name"))
        print("Experiment name:", config.get("project", {}).get("experiment_name"))
        print("Dataset:", config.get("dataset", {}).get("name"))
        print("Model:", config.get("model", {}).get("name"))

    except ImportError:
        print("[MISS] pyyaml: YAML parsing skipped")
    except Exception as e:
        print("[ERROR] Failed to parse YAML:", repr(e))


def main() -> None:
    """Run all checks."""
    print_section("M3ED + MMGCN environment check")

    check_python_env()
    check_user_site()
    check_basic_packages()
    check_torch_cuda()
    check_project_dirs()
    check_config()

    print("\nEnvironment check finished.")


if __name__ == "__main__":
    main()

"""
Random seed utilities.

This file is responsible for setting random seeds for reproducibility.
Torch is optional at the current project stage.
If torch is not installed, the torch-related seed setting will be skipped.
固定随机种子
"""

import os
import random
import numpy as np


def set_seed(seed: int = 42) -> None:
    """
    Set random seeds for Python, NumPy, and Torch if available.

    Args:
        seed: random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import torch

        torch.manual_seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        # Make CUDA behavior more deterministic.
        # This may slightly reduce speed, but helps reproducibility.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        print(f"[Seed] Python / NumPy / Torch seed set to {seed}")

    except ImportError:
        print(f"[Seed] Python / NumPy seed set to {seed}")
        print("[Seed] Torch is not installed, torch seed skipped.")

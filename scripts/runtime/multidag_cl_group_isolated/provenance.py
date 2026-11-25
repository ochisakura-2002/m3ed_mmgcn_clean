"""Capture the actual G1 runtime's local source and environment identities."""

import hashlib
from importlib import metadata
import os
from pathlib import Path
import platform
import subprocess
import sys

import torch

from scripts.runtime.multidag_cl_ta_v0.formal import sha256_file


def source_hashes(root, config_files):
    files = set(config_files)
    for module in tuple(sys.modules.values()):
        filename = getattr(module, "__file__", None)
        if filename and filename.endswith(".py") and Path(filename).is_absolute():
            path = Path(filename).resolve()
            if path.is_relative_to(root):
                files.add(path)
    return {path.relative_to(root).as_posix(): sha256_file(path) for path in sorted(files)}


def runtime_provenance(root, device, config_files):
    def git(*args):
        return subprocess.check_output(["git", *args], cwd=root)

    status = git("status", "--porcelain=v1", "-z", "--untracked-files=all")
    diff = git("diff", "--binary", "--no-ext-diff", "--no-textconv", "HEAD", "--")
    cuda = device == "cuda"
    return {
        "files_sha256": source_hashes(root, config_files),
        "git_head": git("rev-parse", "HEAD").decode("ascii").strip(),
        "git_status_records": status.decode("utf-8", errors="backslashreplace").split("\0")[:-1],
        "tracked_diff_against_head_sha256": hashlib.sha256(diff).hexdigest(),
        "environment": {
            "python": platform.python_version(), "platform": platform.platform(),
            "packages": {name: metadata.version(name) for name in (
                "torch", "numpy", "PyYAML", "scikit-learn", "scipy",
            )},
            "torch_cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version() if cuda else None,
            "gpu": torch.cuda.get_device_name() if cuda else None,
            "gpu_compute_capability": torch.cuda.get_device_capability() if cuda else None,
            "visible_gpu_count": torch.cuda.device_count() if cuda else 0,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "torch_deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "torch_threads": torch.get_num_threads(),
        },
    }

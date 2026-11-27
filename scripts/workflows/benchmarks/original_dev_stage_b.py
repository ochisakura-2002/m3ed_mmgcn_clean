"""Strict historical identity and on-disk recovery for Original Dev Alignment."""

from contextlib import contextmanager
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import sys
from uuid import uuid4


SOURCE_POLICY = Path("scripts/workflows/benchmarks/original_dev_source_compatibility.json")
SOURCE_ROLES = {
    "scripts/workflows/benchmarks/run_original_dev_alignment.py": "environment authentication and orchestration",
    "scripts/workflows/benchmarks/original_dev_stage_b.py": "environment authentication and failure recovery",
    "scripts/analysis/models/multidag_cl/plot_original_dev_alignment.py": "isolated figure rendering and font checks",
    "scripts/analysis/models/multidag_cl/original_dev_analysis.py": "post-fit table completeness and empty curve exports",
}
PHASES = ("pre_fit", "fit_started", "trained", "publishing", "published")


def canonical(value):
    """Preserve JSON types, normalizing sequence containers and NumPy scalars."""
    def normalize(item):
        if isinstance(item, dict):
            if any(not isinstance(key, str) for key in item):
                raise TypeError("identity requires string keys")
            return {key: normalize(value) for key, value in item.items()}
        if isinstance(item, (tuple, list)):
            return [normalize(value) for value in item]
        numpy = sys.modules.get("numpy")
        if numpy is not None and isinstance(item, numpy.generic):
            scalar = item.item()
            if type(scalar) not in (str, bool, int, float):
                raise TypeError(f"NumPy scalar lacks a lossless JSON representation: {item.dtype}")
            return scalar
        return item

    # Comparing the serialized form also distinguishes True, 1, 1.0 and -0.0.
    return json.dumps(normalize(value), sort_keys=True, ensure_ascii=True, allow_nan=False)


def differences(expected, actual, prefix=""):
    missing = "<missing field>"
    result = {}
    for key in sorted(expected.keys() | actual.keys()):
        name = f"{prefix}.{key}" if prefix else key
        if key not in expected or key not in actual:
            result[name] = {"reviewed": expected.get(key, missing), "actual": actual.get(key, missing)}
        elif canonical(expected[key]) != canonical(actual[key]):
            if isinstance(expected[key], dict) and isinstance(actual[key], dict):
                result.update(differences(expected[key], actual[key], name))
            else:
                result[name] = {"reviewed": expected[key], "actual": actual[key]}
    return result


def check_environment(reviewed, actual, *, diagnostics=None):
    changed = differences(reviewed, actual)
    if changed:
        detail = {"differences": changed, "actual_process_diagnostics": diagnostics}
        raise ValueError(f"Stage B environment differs from reviewed Stage A: {canonical(detail)}")


def process_diagnostics():
    maps = Path("/proc/self/maps")
    libraries = set()
    if maps.is_file():
        for line in maps.read_text(encoding="utf-8", errors="replace").splitlines():
            columns = line.split(maxsplit=5)
            if len(columns) == 6 and columns[5].startswith("/"):
                path = columns[5]
                if re.match(r"lib(mkl|blas|openblas|blis|iomp|omp|gomp|scipy_openblas)", Path(path).name):
                    libraries.add(path)
    return {
        "mappings_available": maps.is_file(), "math_libraries": sorted(libraries),
        "import_order": [name for name in sys.modules
                         if name in ("torch", "numpy", "scipy", "sklearn", "matplotlib", "pandas")],
        "math_environment": {name: os.environ.get(name) for name in
                             ("MKL_THREADING_LAYER", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                              "LD_PRELOAD")},
    }


def certify_sources(root, approved_sha, old, new):
    delta = {key: (old.get(key), new.get(key)) for key in old.keys() | new.keys() if old.get(key) != new.get(key)}
    policy_bytes = (root / SOURCE_POLICY).read_bytes()
    policy = json.loads(policy_bytes)
    if policy["schema"] != 1 or set(policy["files"]) != set(SOURCE_ROLES):
        raise ValueError("unexpected Stage A source compatibility policy")
    for path, record in policy["files"].items():
        if (record["scope"] != SOURCE_ROLES[path]
                or any(not re.fullmatch(r"[0-9a-f]{64}", digest) for digest in
                       (record["new_sha256"], record["old_sha256"] or "0" * 64))):
            raise ValueError(f"invalid Stage A source policy entry: {path}")
    for path, (before, after) in sorted(delta.items()):
        record = policy["files"].get(path)
        if record is None or after is None or (before, after) != (record["old_sha256"], record["new_sha256"]):
            raise ValueError(f"unapproved Stage A source difference: {path} ({before} -> {after})")
    if any(new.get(path) != record["new_sha256"] for path, record in policy["files"].items()):
        raise ValueError("Stage B source bytes differ from the approved compatibility policy")
    return {
        "schema": 1, "stage_a_sha256": approved_sha, "policy_sha256": sha256(policy_bytes).hexdigest(),
        "reviewed_inventory_sha256": sha256(canonical(old).encode()).hexdigest(),
        "current_inventory_sha256": sha256(canonical(new).encode()).hexdigest(),
        "changes": {key: policy["files"][key] for key in sorted(delta)},
    }


def present(path):
    return os.path.lexists(path)


def read_attempt(path):
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"expected a regular Stage B attempt receipt: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


@contextmanager
def batch_lock(path, *, readonly=False):
    # Never unlink an advisory lock file, so contenders always use the same inode.
    if path.is_symlink():
        raise ValueError(f"unexpected Stage B lock symlink: {path}")
    if readonly and not present(path):
        yield
        return
    with path.open("r+b" if readonly else "a+b") as stream:
        try:
            if os.name == "nt":
                from msvcrt import LK_NBLCK, locking

                locking(stream.fileno(), LK_NBLCK, 1)
            else:
                from fcntl import LOCK_EX, LOCK_NB, flock

                flock(stream, LOCK_EX | LOCK_NB)
        except BlockingIOError as error:
            raise ValueError(f"another Stage B process holds {path}") from error
        yield


def inspect_attempt(receipt, destination, archive, d1, approved_sha, *, resume_only=False):
    pending = receipt.with_suffix(".json.tmp")
    for path in (receipt, destination, archive, d1, pending):
        if path.is_symlink():
            raise ValueError(f"unexpected symlink at Stage B path: {path}")
    state = read_attempt(receipt) if present(receipt) else None
    if state is None:
        if resume_only or any(present(p) for p in (destination, archive, d1, pending)):
            raise ValueError("unowned or missing Stage B assets/receipt; inspect paths before retry")
        return "fresh", None
    if state.get("stage_a_sha256") != approved_sha:
        raise ValueError("existing Stage B receipt belongs to a different Stage A approval")

    phase = state.get("phase")
    status = state.get("status")
    if phase is None:
        # The legacy launcher created just these two paths before preflight.
        if (status != "FAIL" or present(d1) or present(archive) or present(pending)
                or (present(destination) and (not destination.is_dir() or any(destination.iterdir())))):
            raise ValueError("legacy Stage B evidence is ambiguous or training assets exist; inspect without overwriting")
        if resume_only:
            raise ValueError("no completed D1 to resume")
        return "legacy_pre_fit", state

    if phase not in PHASES or status not in ("RUNNING", "FAIL", "PASS"):
        raise ValueError("unknown Stage B phase or status")
    if phase in ("publishing", "published"):
        return "publish", state
    if status == "PASS" or present(destination) or present(archive):
        raise ValueError("unexpected published Stage B assets")
    if present(d1):
        if phase not in ("fit_started", "trained") or present(d1 / ".training.lock"):
            raise ValueError("D1 owner/phase is inconsistent or training lock is present; inspect D1")
        return "verify_d1", state
    if phase != "pre_fit" or resume_only:
        raise ValueError("D1 training may have started; preserve its receipt and inspect the run")
    return "pre_fit", state


def attempt_workspace(destination, attempt):
    if not re.fullmatch(r"[0-9a-f]{32}", attempt):
        raise ValueError("invalid Stage B attempt ID")
    path = destination.parent / f".{destination.name}_stage_b_{attempt}"
    if path.is_symlink() or (present(path) and not path.is_dir()):
        raise ValueError(f"unexpected Stage B workspace: {path}")
    return path


def new_attempt_history(receipt):
    parent = receipt.parent / "stage_b_attempts"
    if parent.is_symlink():
        raise ValueError(f"unexpected Stage B history symlink: {parent}")
    archived = parent / uuid4().hex
    archived.mkdir(parents=True, exist_ok=False)
    return archived


def archive_attempt(receipt, destination, state, mode):
    archived = new_attempt_history(receipt)
    # Move the ownership receipt last. An interruption can be retried without
    # deriving permission to move any path from an unowned directory.
    pending = receipt.with_suffix(".json.tmp")
    if present(pending):
        pending.rename(archived / pending.name)
    if mode == "legacy_pre_fit" and present(destination):
        if any(destination.iterdir()):
            raise ValueError(f"legacy Stage B directory is no longer empty: {destination}")
        destination.rename(archived / "empty_analysis")
    if "attempt" in state:
        workspace = attempt_workspace(destination, state["attempt"])
        if present(workspace):
            workspace.rename(archived / "workspace")
    # Keep the original receipt until the new owner can replace it atomically.
    (archived / receipt.name).write_bytes(receipt.read_bytes())
    return archived

import uuid
import shutil
from pathlib import Path

import pytest
import torch


@pytest.fixture(autouse=True, scope="module")
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def artifact_dir():
    # pathlib mkdir inherits workspace ACLs; the installed Python 3.9's
    # tempfile directory permissions are incompatible with this Windows sandbox.
    parent = (Path(__file__).resolve().parents[4] / "tmp").resolve()
    path = parent / ("tav_test_" + uuid.uuid4().hex)
    path.mkdir(parents=True)
    yield path
    # Only this fixture's exact freshly allocated directory is disposable.
    if path.resolve() != path or path.parent != parent or not path.name.startswith("tav_test_"):
        raise AssertionError("synthetic cleanup path escaped its workspace")
    shutil.rmtree(path, ignore_errors=True)

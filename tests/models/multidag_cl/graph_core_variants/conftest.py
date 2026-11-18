import pytest
import torch


@pytest.fixture(autouse=True, scope="session")
def small_cpu_checks():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)

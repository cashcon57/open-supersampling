import pytest
import torch


@pytest.fixture
def cuda_device():
    if not torch.cuda.is_available():
        pytest.skip("CUDA device required")
    return torch.device("cuda:0")


@pytest.fixture
def kernels_built():
    try:
        from oss.cuda import oss_cuda  # noqa: F401

        if not getattr(oss_cuda.rasterizer, "_COMPILED", False):
            pytest.skip("oss_cuda extension not compiled")
    except ImportError as e:
        pytest.skip(f"oss_cuda not importable: {e}")


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0xC0DA)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0xC0DA)

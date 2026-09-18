import pytest
import torch


def _gpu_available() -> tuple[bool, str]:
    if not torch.cuda.is_available():
        return False, "no CUDA GPU (run on Colab)"
    from torch.utils.cpp_extension import CUDA_HOME

    if CUDA_HOME is None:
        return False, "CUDA toolkit (nvcc) not found"
    return True, ""


def pytest_collection_modifyitems(config, items):
    ok, reason = _gpu_available()
    if ok:
        return
    skip = pytest.mark.skip(reason=reason)
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip)

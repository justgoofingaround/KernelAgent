"""CPU-only checks on the op specs. These run locally without a GPU."""

import numpy as np
import pytest
import torch

from kernelagent.harness.build import summarize_log
from kernelagent.ops import OPS, get_op

ALL_OPS = list(OPS.values())


@pytest.mark.parametrize("spec", ALL_OPS, ids=lambda s: s.name)
def test_spec_is_complete(spec):
    assert "forward(" in spec.cpp_signature
    assert spec.baseline_path.exists(), f"missing baseline kernel {spec.baseline_path}"
    for dtype in spec.dtypes:
        assert dtype in spec.tolerances


@pytest.mark.parametrize("spec", ALL_OPS, ids=lambda s: s.name)
def test_inputs_have_requested_shape_and_dtype(spec):
    for shape in spec.shapes:
        args = spec.make_inputs(shape, torch.float16, "cpu", 0)
        assert args[0].shape == shape
        assert args[0].dtype == torch.float16
        out = spec.reference_output(args, torch.float16)
        assert out.shape == shape and out.dtype == torch.float16


@pytest.mark.parametrize("spec", ALL_OPS, ids=lambda s: s.name)
def test_inputs_are_deterministic(spec):
    a = spec.make_inputs(spec.shapes[3], torch.float32, "cpu", 42)
    b = spec.make_inputs(spec.shapes[3], torch.float32, "cpu", 42)
    assert torch.equal(a[0], b[0])


def test_softmax_reference_matches_numpy():
    spec = get_op("softmax")
    (x,) = spec.make_inputs((8, 1000), torch.float32, "cpu", 0)
    xn = x.double().numpy()
    e = np.exp(xn - xn.max(axis=-1, keepdims=True))
    expected = e / e.sum(axis=-1, keepdims=True)
    np.testing.assert_allclose(spec.reference(x).numpy(), expected, rtol=1e-5, atol=1e-7)


def test_rmsnorm_reference_matches_numpy():
    spec = get_op("rmsnorm")
    x, w, eps = spec.make_inputs((8, 1000), torch.float32, "cpu", 0)
    xn, wn = x.double().numpy(), w.double().numpy()
    expected = xn / np.sqrt((xn**2).mean(axis=-1, keepdims=True) + eps) * wn
    np.testing.assert_allclose(spec.reference(x, w, eps).numpy(), expected, rtol=1e-5, atol=1e-6)


def test_summarize_log_keeps_error_lines():
    log = "\n".join(["ninja: building"] * 50 + ["kernel.cu(12): error: identifier \"foo\" is undefined", "1 error detected"])
    summary = summarize_log(log)
    assert "identifier \"foo\" is undefined" in summary
    assert summary.count("ninja: building") <= 1

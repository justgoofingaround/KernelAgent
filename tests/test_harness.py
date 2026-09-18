"""GPU tests for the compile -> verify -> benchmark harness. Run on Colab."""

import pytest
import torch

from kernelagent.harness import benchmark, compile_kernel, verify
from kernelagent.ops import OPS, get_op

pytestmark = pytest.mark.gpu

# Naive one-thread-per-row softmax, with a bug: it skips the last column.
OFF_BY_ONE_SOFTMAX = r"""
#include <torch/extension.h>

__global__ void k(const float* x, float* y, int64_t rows, int64_t cols) {
  int64_t r = blockIdx.x * blockDim.x + threadIdx.x;
  if (r >= rows) return;
  float m = -INFINITY, s = 0.f;
  for (int64_t c = 0; c < cols - 1; ++c) m = fmaxf(m, x[r * cols + c]);
  for (int64_t c = 0; c < cols - 1; ++c) s += expf(x[r * cols + c] - m);
  for (int64_t c = 0; c < cols; ++c) y[r * cols + c] = expf(x[r * cols + c] - m) / s;
}

torch::Tensor forward(torch::Tensor x) {
  auto xf = x.contiguous().to(torch::kFloat);
  auto y = torch::empty_like(xf);
  int64_t cols = xf.size(-1), rows = xf.numel() / cols;
  k<<<(rows + 127) / 128, 128>>>(xf.data_ptr<float>(), y.data_ptr<float>(), rows, cols);
  return y.to(x.scalar_type());
}
"""

SYNTAX_ERROR_SOURCE = r"""
#include <torch/extension.h>
torch::Tensor forward(torch::Tensor x) { return x + undefined_symbol; }
"""


@pytest.mark.parametrize("spec", list(OPS.values()), ids=lambda s: s.name)
def test_baseline_compiles_and_passes(spec):
    build = compile_kernel(spec.baseline_source(), spec)
    assert build.ok, build.error_summary
    report = verify(build.module.forward, spec)
    assert report.ok, report.summary()


def test_compile_error_is_returned_not_raised():
    build = compile_kernel(SYNTAX_ERROR_SOURCE, get_op("softmax"))
    assert not build.ok
    assert build.module is None
    assert "undefined_symbol" in build.error_summary


def test_off_by_one_kernel_fails_verification():
    build = compile_kernel(OFF_BY_ONE_SOFTMAX, get_op("softmax"))
    assert build.ok, build.error_summary
    report = verify(build.module.forward, get_op("softmax"))
    assert not report.ok
    assert "mismatch" in report.summary()


def test_benchmark_reports_time_and_bandwidth():
    spec = get_op("softmax")
    args = spec.make_inputs((1024, 1024), torch.float16, "cuda", 0)
    result = benchmark(spec.reference, args, warmup=2, iters=10, bytes_moved=4 * 1024 * 1024)
    assert result.median_ms > 0
    assert result.p10_ms <= result.median_ms <= result.p90_ms
    assert result.gbps is not None and result.gbps > 0

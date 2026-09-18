"""GPU tests: run the real ncu and compute-sanitizer on planted-bug kernels. Run on Colab."""

import pytest
import torch

from kernelagent.harness import profile_kernel, run_sanitizer
from kernelagent.harness.tools import find_compute_sanitizer, find_ncu
from kernelagent.ops import get_op

pytestmark = pytest.mark.gpu
needs_sanitizer = pytest.mark.skipif(find_compute_sanitizer() is None, reason="compute-sanitizer not found")
needs_ncu = pytest.mark.skipif(find_ncu() is None, reason="ncu not found")

SOFTMAX = get_op("softmax")
SMALL = {"shapes": [(4, 256)], "dtypes": [torch.float32]}

# Copies x to y, but the last thread of the last block reads one element past the end of x.
OUT_OF_BOUNDS = r"""
#include <torch/extension.h>

__global__ void oob_kernel(const float* x, float* y, int64_t n) {
  int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  int64_t j = (i == n - 1) ? n : i;
  y[i] = x[j];
}

torch::Tensor forward(torch::Tensor x) {
  auto y = torch::empty_like(x);
  int64_t n = x.numel();
  oob_kernel<<<(n + 255) / 256, 256>>>(x.data_ptr<float>(), y.data_ptr<float>(), n);
  return y;
}
"""

# Block-wide sum through shared memory with the __syncthreads() missing:
# threads read slots that other warps may not have written yet.
RACY = r"""
#include <torch/extension.h>

__global__ void racy_kernel(const float* x, float* y, int64_t cols) {
  __shared__ float buf[256];
  const float* row = x + blockIdx.x * cols;
  buf[threadIdx.x] = threadIdx.x < cols ? row[threadIdx.x] : 0.f;
  float total = 0.f;
  for (int k = 0; k < blockDim.x; ++k) total += buf[k];
  if (threadIdx.x < cols) y[blockIdx.x * cols + threadIdx.x] = row[threadIdx.x] / total;
}

torch::Tensor forward(torch::Tensor x) {
  auto y = torch::empty_like(x);
  int64_t cols = x.size(-1), rows = x.numel() / cols;
  racy_kernel<<<rows, 256>>>(x.data_ptr<float>(), y.data_ptr<float>(), cols);
  return y;
}
"""


def _line_of(source: str, text: str) -> int:
    return next(i for i, line in enumerate(source.splitlines(), 1) if text in line)


@needs_sanitizer
def test_memcheck_catches_out_of_bounds_read_at_the_right_line():
    report = run_sanitizer(OUT_OF_BOUNDS, SOFTMAX, "memcheck", **SMALL)
    assert not report.ok, report.output[-3000:]
    lines = {loc.source_line for f in report.device_findings for loc in f.locations}
    assert _line_of(OUT_OF_BOUNDS, "y[i] = x[j];") in lines, report.summary()


@needs_sanitizer
def test_racecheck_catches_missing_syncthreads():
    report = run_sanitizer(RACY, SOFTMAX, "racecheck", **SMALL)
    assert not report.ok, report.output[-3000:]
    lines = {loc.source_line for f in report.device_findings for loc in f.locations}
    assert lines & {_line_of(RACY, "buf[threadIdx.x] ="), _line_of(RACY, "total += buf[k]")}, report.summary()


@needs_sanitizer
@pytest.mark.parametrize("tool", ["memcheck", "racecheck"])
def test_baseline_softmax_is_clean(tool):
    report = run_sanitizer(SOFTMAX.baseline_source(), SOFTMAX, tool, shapes=[(3, 33), (8, 1000)])
    assert report.ok, report.summary()


@needs_ncu
def test_profile_baseline_softmax():
    profile = profile_kernel(SOFTMAX.baseline_source(), SOFTMAX, (4096, 1024), torch.float16)
    k = profile.primary
    assert "softmax_kernel" in k.name
    assert k.metrics["duration_ns"] and k.metrics["duration_ns"] > 0
    assert profile.roofline.bound in {"memory", "latency", "compute"}
    text = profile.summary()
    assert "Bound:" in text and "Occupancy:" in text

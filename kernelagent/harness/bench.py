"""Time kernels with CUDA events and compare against PyTorch eager / torch.compile."""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Any, Callable

import torch

from ..ops.base import OpSpec, Shape

# Peak DRAM bandwidth (GB/s), matched by substring of the device name, most specific first.
PEAK_BANDWIDTH_GBPS = [
    ("H100 SXM", 3350), ("H100", 2000),
    ("A100-SXM4-80GB", 2039), ("A100 80GB", 1935), ("A100", 1555),
    ("L4", 300), ("T4", 320), ("V100", 900), ("A10G", 600),
    ("RTX 4090", 1008), ("RTX 4060 Laptop", 256),
]


@dataclass
class BenchResult:
    median_ms: float
    p10_ms: float
    p90_ms: float
    gbps: float | None = None


def peak_bandwidth_gbps(device: int | None = None) -> float | None:
    name = torch.cuda.get_device_name(device)
    for key, gbps in PEAK_BANDWIDTH_GBPS:
        if key in name:
            return gbps
    return None


def _l2_flush_buffer() -> torch.Tensor:
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    l2_bytes = getattr(props, "L2_cache_size", 0) or 64 * 1024 * 1024
    return torch.empty(2 * l2_bytes, dtype=torch.uint8, device="cuda")


def benchmark(fn: Callable[..., Any], args: tuple[Any, ...], warmup: int = 10, iters: int = 100,
              flush_l2: bool = True, bytes_moved: int | None = None) -> BenchResult:
    """Median time of fn(*args). The L2 cache is flushed before each timed call so
    results reflect DRAM traffic, like a kernel running inside a real model."""
    flush = _l2_flush_buffer() if flush_l2 else None
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for start, end in zip(starts, ends):
        if flush is not None:
            flush.zero_()
        start.record()
        fn(*args)
        end.record()
    torch.cuda.synchronize()

    times = sorted(s.elapsed_time(e) for s, e in zip(starts, ends))
    median = statistics.median(times)
    gbps = bytes_moved / (median * 1e-3) / 1e9 if bytes_moved else None
    return BenchResult(median, times[len(times) // 10], times[(9 * len(times)) // 10], gbps)


@dataclass
class Comparison:
    op: str
    shape: Shape
    dtype: str
    ours: BenchResult
    eager: BenchResult
    compiled: BenchResult | None  # None if torch.compile is unavailable

    @property
    def speedup_vs_eager(self) -> float:
        return self.eager.median_ms / self.ours.median_ms


def compare(fn: Callable[..., Any], spec: OpSpec, shape: Shape, dtype: torch.dtype,
            use_torch_compile: bool = True, **bench_kwargs: Any) -> Comparison:
    args = spec.make_inputs(shape, dtype, "cuda", 0)
    nbytes = spec.bytes_moved(args, spec.reference(*args))

    ours = benchmark(fn, args, bytes_moved=nbytes, **bench_kwargs)
    eager = benchmark(spec.reference, args, bytes_moved=nbytes, **bench_kwargs)
    compiled = None
    if use_torch_compile:
        try:
            compiled = benchmark(torch.compile(spec.reference), args, bytes_moved=nbytes, **bench_kwargs)
        except Exception:  # e.g. Triton missing; eager comparison still stands
            compiled = None
    return Comparison(spec.name, shape, str(dtype).removeprefix("torch."), ours, eager, compiled)

"""OpSpec: the contract a kernel must satisfy.

Every kernel, hand-written or agent-generated, is a single CUDA source file that
exports the C++ function declared in `cpp_signature`. The harness compiles it,
checks it against `reference` on every (shape, dtype) pair, and benchmarks it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
BASELINES_DIR = REPO_ROOT / "kernels" / "baselines"

Shape = tuple[int, ...]


@dataclass(frozen=True)
class Tolerance:
    atol: float
    rtol: float


@dataclass
class OpSpec:
    name: str
    description: str
    # C++ declaration the kernel source must define, e.g.
    # "torch::Tensor forward(torch::Tensor x);"
    cpp_signature: str
    # PyTorch reference. Called with the same args as the kernel, but with
    # floating tensors upcast to fp32, so it acts as the "exact" answer.
    reference: Callable[..., torch.Tensor]
    # (shape, dtype, device, seed) -> args tuple passed to forward()
    make_inputs: Callable[[Shape, torch.dtype, str, int], tuple[Any, ...]]
    # Correctness shapes: include awkward sizes (1, primes, non-multiples of 8).
    shapes: list[Shape]
    # Realistic LLM sizes used for benchmarking.
    bench_shapes: list[Shape]
    tolerances: dict[torch.dtype, Tolerance]
    # (args, output) -> minimum bytes read + written, used for achieved GB/s.
    bytes_moved: Callable[[tuple[Any, ...], torch.Tensor], int]
    dtypes: list[torch.dtype] = field(
        default_factory=lambda: [torch.float32, torch.float16, torch.bfloat16]
    )

    @property
    def baseline_path(self) -> Path:
        return BASELINES_DIR / f"{self.name}.cu"

    def baseline_source(self) -> str:
        return self.baseline_path.read_text(encoding="utf-8")

    def reference_output(self, args: tuple[Any, ...], dtype: torch.dtype) -> torch.Tensor:
        """Reference computed in fp32, cast to the dtype under test."""
        upcast = tuple(
            a.float() if isinstance(a, torch.Tensor) and a.is_floating_point() else a
            for a in args
        )
        return self.reference(*upcast).to(dtype)


def tensor_bytes(t: torch.Tensor) -> int:
    return t.numel() * t.element_size()

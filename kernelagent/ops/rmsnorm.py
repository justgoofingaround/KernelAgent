"""RMSNorm (Llama-style): y = x / sqrt(mean(x^2) + eps) * w, over the last dimension."""

import torch

from .base import OpSpec, Tolerance, tensor_bytes

EPS = 1e-6


def _reference(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps) * w


def _make_inputs(shape, dtype, device, seed):
    g = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn(shape, generator=g)
    w = 1.0 + 0.1 * torch.randn(shape[-1], generator=g)
    return (x.to(device=device, dtype=dtype), w.to(device=device, dtype=dtype), EPS)


RMSNORM = OpSpec(
    name="rmsnorm",
    description=(
        "y = x * rsqrt(mean(x^2, dim=-1) + eps) * w for x of shape (rows, hidden) "
        "and weight w of shape (hidden,). Accumulate in fp32."
    ),
    cpp_signature="torch::Tensor forward(torch::Tensor x, torch::Tensor w, double eps);",
    reference=_reference,
    make_inputs=_make_inputs,
    shapes=[(1, 1), (1, 7), (3, 64), (8, 1000), (32, 2048), (16, 4096), (5, 4099), (2, 8192)],
    bench_shapes=[(4096, 2048), (4096, 4096), (16384, 8192)],
    tolerances={
        torch.float32: Tolerance(atol=1e-5, rtol=1e-5),
        torch.float16: Tolerance(atol=2e-3, rtol=2e-3),
        torch.bfloat16: Tolerance(atol=1e-2, rtol=1e-2),
    },
    bytes_moved=lambda args, out: tensor_bytes(args[0]) + tensor_bytes(args[1]) + tensor_bytes(out),
)

"""Row-wise softmax over the last dimension (attention scores, LM-head sampling)."""

import torch

from .base import OpSpec, Tolerance, tensor_bytes


def _reference(x: torch.Tensor) -> torch.Tensor:
    return torch.softmax(x, dim=-1)


def _make_inputs(shape, dtype, device, seed):
    g = torch.Generator(device="cpu").manual_seed(seed)
    # Scale 3 gives a realistic spread of logits without overflowing fp16.
    x = torch.randn(shape, generator=g) * 3.0
    return (x.to(device=device, dtype=dtype),)


SOFTMAX = OpSpec(
    name="softmax",
    description="y = softmax(x, dim=-1) for a 2-D tensor x of shape (rows, cols).",
    cpp_signature="torch::Tensor forward(torch::Tensor x);",
    reference=_reference,
    make_inputs=_make_inputs,
    shapes=[(1, 1), (1, 7), (3, 33), (8, 1000), (128, 1024), (64, 4096), (17, 8191), (4, 32768)],
    bench_shapes=[(4096, 1024), (4096, 4096), (1024, 32768)],
    tolerances={
        torch.float32: Tolerance(atol=1e-6, rtol=1e-5),
        torch.float16: Tolerance(atol=1e-3, rtol=2e-3),
        torch.bfloat16: Tolerance(atol=4e-3, rtol=1e-2),
    },
    bytes_moved=lambda args, out: tensor_bytes(args[0]) + tensor_bytes(out),
)

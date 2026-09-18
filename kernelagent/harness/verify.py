"""Check a kernel against the op's PyTorch reference on every (shape, dtype) case."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import torch

from ..ops.base import OpSpec, Shape


@dataclass
class CaseResult:
    shape: Shape
    dtype: str
    ok: bool
    max_abs_err: float = float("nan")
    max_rel_err: float = float("nan")
    message: str = ""


@dataclass
class VerifyReport:
    op: str
    cases: list[CaseResult] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return bool(self.cases) and all(c.ok for c in self.cases)

    @property
    def failures(self) -> list[CaseResult]:
        return [c for c in self.cases if not c.ok]

    def summary(self, max_failures: int = 5) -> str:
        passed = sum(c.ok for c in self.cases)
        lines = [f"{self.op}: {passed}/{len(self.cases)} cases passed"]
        for c in self.failures[:max_failures]:
            lines.append(f"  FAIL shape={c.shape} dtype={c.dtype}: {c.message}")
        if len(self.failures) > max_failures:
            lines.append(f"  ... and {len(self.failures) - max_failures} more failures")
        return "\n".join(lines)


def _dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).removeprefix("torch.")


def _check_case(fn: Callable[..., Any], spec: OpSpec, shape: Shape, dtype: torch.dtype,
                device: str, seed: int) -> CaseResult:
    name = _dtype_name(dtype)
    args = spec.make_inputs(shape, dtype, device, seed)
    originals = [a.clone() if isinstance(a, torch.Tensor) else None for a in args]
    expected = spec.reference_output(args, dtype)

    try:
        out = fn(*args)
        torch.cuda.synchronize()
    except Exception as exc:
        return CaseResult(shape, name, False, message=f"kernel raised {type(exc).__name__}: {exc}")

    for i, (a, orig) in enumerate(zip(args, originals)):
        if orig is not None and not torch.equal(a, orig):
            return CaseResult(shape, name, False, message=f"kernel modified input #{i} in place")

    if not isinstance(out, torch.Tensor):
        return CaseResult(shape, name, False, message=f"returned {type(out).__name__}, expected Tensor")
    if out.shape != expected.shape:
        return CaseResult(shape, name, False,
                          message=f"output shape {tuple(out.shape)} != expected {tuple(expected.shape)}")
    if out.dtype != dtype:
        return CaseResult(shape, name, False, message=f"output dtype {out.dtype} != expected {dtype}")

    got, want = out.float(), expected.float()
    diff = (got - want).abs()
    max_abs = diff.max().item() if diff.numel() else 0.0
    max_rel = (diff / want.abs().clamp_min(1e-6)).max().item() if diff.numel() else 0.0

    bad_values = ~torch.isfinite(got) & torch.isfinite(want)
    if bad_values.any():
        idx = tuple(int(i) for i in bad_values.nonzero()[0])
        return CaseResult(shape, name, False, max_abs, max_rel,
                          message=f"non-finite output at index {idx}: got {got[idx].item()}, expected {want[idx].item()}")

    tol = spec.tolerances[dtype]
    close = torch.isclose(got, want, atol=tol.atol, rtol=tol.rtol, equal_nan=True)
    if not close.all():
        idx = tuple(int(i) for i in (~close).nonzero()[0])
        n_bad = int((~close).sum())
        return CaseResult(
            shape, name, False, max_abs, max_rel,
            message=(f"{n_bad}/{close.numel()} elements mismatch (atol={tol.atol}, rtol={tol.rtol}); "
                     f"first at index {idx}: got {got[idx].item():.6g}, expected {want[idx].item():.6g}; "
                     f"max_abs_err={max_abs:.3g}"),
        )
    return CaseResult(shape, name, True, max_abs, max_rel)


def verify(fn: Callable[..., Any], spec: OpSpec, device: str = "cuda", seed: int = 0,
           stop_on_first_failure: bool = False) -> VerifyReport:
    """Run `fn` on every spec shape x dtype and compare to the fp32 reference.

    A kernel that hits an illegal memory access corrupts the CUDA context, so
    later cases in the same process will also fail; M2 adds compute-sanitizer
    for pinpointing those.
    """
    report = VerifyReport(spec.name)
    for dtype in spec.dtypes:
        for i, shape in enumerate(spec.shapes):
            result = _check_case(fn, spec, shape, dtype, device, seed + i)
            report.cases.append(result)
            if stop_on_first_failure and not result.ok:
                return report
    return report

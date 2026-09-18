"""Run a kernel in a fresh process, so ncu / compute-sanitizer can wrap it.

    python -m kernelagent.harness.launch --op softmax --source k.cu --mode profile \
        --shapes 4096x1024 --dtypes float16

--mode profile: one warm-up call, then exactly one call inside a
                cudaProfilerStart/Stop range (ncu runs with --profile-from-start off).
--mode run:     call the kernel on every shape x dtype (for compute-sanitizer).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from ..ops import OPS, get_op
from .build import compile_kernel

DTYPES = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}


def parse_shapes(text: str) -> list[tuple[int, ...]]:
    return [tuple(int(d) for d in s.split("x")) for s in text.split(",") if s]


def parse_dtypes(text: str) -> list[torch.dtype]:
    return [DTYPES[d] for d in text.split(",") if d]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--op", required=True, choices=list(OPS))
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--mode", choices=["profile", "run"], default="run")
    parser.add_argument("--shapes", required=True, type=parse_shapes)
    parser.add_argument("--dtypes", default="float32", type=parse_dtypes)
    args = parser.parse_args(argv)

    spec = get_op(args.op)
    build = compile_kernel(args.source.read_text(encoding="utf-8"), spec)
    if not build.ok:
        print(build.error_summary, file=sys.stderr)
        return 2
    fn = build.module.forward

    if args.mode == "profile":
        inputs = spec.make_inputs(args.shapes[0], args.dtypes[0], "cuda", 0)
        fn(*inputs)  # warm-up: lazy CUDA init and module load stay outside the profiled range
        torch.cuda.synchronize()
        torch.cuda.profiler.start()
        fn(*inputs)
        torch.cuda.synchronize()
        torch.cuda.profiler.stop()
        return 0

    for dtype in args.dtypes:
        for i, shape in enumerate(args.shapes):
            fn(*spec.make_inputs(shape, dtype, "cuda", i))
            torch.cuda.synchronize()
    return 0


if __name__ == "__main__":
    sys.exit(main())

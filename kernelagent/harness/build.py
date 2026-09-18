"""Compile a CUDA kernel source into an importable Python module.

Compiler failures are returned as text instead of raised: the agent loop (M3)
feeds `error_summary` straight back into the prompt.
"""

from __future__ import annotations

import hashlib
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..ops.base import REPO_ROOT, OpSpec

CACHE_DIR = REPO_ROOT / ".cache" / "kernels"
DEFAULT_CUDA_FLAGS = ["-O3", "-lineinfo", "--expt-relaxed-constexpr"]
MAX_SUMMARY_CHARS = 4000


@dataclass
class BuildResult:
    ok: bool
    module: Any | None
    name: str
    seconds: float
    log: str = ""
    build_dir: Path | None = None

    @property
    def error_summary(self) -> str:
        return summarize_log(self.log)


def source_hash(source: str, spec: OpSpec, flags: list[str]) -> str:
    key = "\0".join([spec.name, spec.cpp_signature, source, *flags])
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def compile_kernel(
    source: str,
    spec: OpSpec,
    extra_cuda_cflags: list[str] | None = None,
    verbose: bool = False,
) -> BuildResult:
    """Build `source` so that `result.module.forward` matches `spec.cpp_signature`.

    Identical (source, flags) pairs reuse the cached build in .cache/kernels/.
    """
    # Imported lazily so CPU-only code paths don't need a CUDA toolchain.
    from torch.utils.cpp_extension import load_inline

    flags = DEFAULT_CUDA_FLAGS + list(extra_cuda_cflags or [])
    name = f"ka_{spec.name}_{source_hash(source, spec, flags)}"
    build_dir = CACHE_DIR / name
    build_dir.mkdir(parents=True, exist_ok=True)

    start = time.perf_counter()
    try:
        module = load_inline(
            name=name,
            cpp_sources=spec.cpp_signature,
            cuda_sources=source,
            functions=["forward"],
            extra_cuda_cflags=flags,
            build_directory=str(build_dir),
            verbose=verbose,
        )
    except Exception as exc:  # torch raises RuntimeError with the full compiler output
        return BuildResult(False, None, name, time.perf_counter() - start, log=str(exc), build_dir=build_dir)
    return BuildResult(True, module, name, time.perf_counter() - start, build_dir=build_dir)


def compile_file(path: str | Path, spec: OpSpec, **kwargs: Any) -> BuildResult:
    return compile_kernel(Path(path).read_text(encoding="utf-8"), spec, **kwargs)


_ERROR_LINE = re.compile(r"error|undefined|fatal", re.IGNORECASE)


def summarize_log(log: str, context: int = 2) -> str:
    """Keep compiler error lines (plus a little context), capped in length."""
    if not log:
        return ""
    lines = log.splitlines()
    keep: set[int] = set()
    for i, line in enumerate(lines):
        if _ERROR_LINE.search(line):
            keep.update(range(max(0, i - 1), min(len(lines), i + context + 1)))
    picked = [lines[i] for i in sorted(keep)] if keep else lines[-60:]
    summary = "\n".join(picked)
    if len(summary) > MAX_SUMMARY_CHARS:
        summary = summary[:MAX_SUMMARY_CHARS] + "\n... [truncated]"
    return summary

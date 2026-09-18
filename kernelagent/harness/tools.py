"""Locate NVIDIA command-line tools (ncu, compute-sanitizer) and run the kernel launcher under them."""

from __future__ import annotations

import glob
import os
import shutil
import subprocess
import sys
from pathlib import Path

import torch

from ..ops.base import REPO_ROOT, OpSpec


def _cuda_home() -> str | None:
    from torch.utils.cpp_extension import CUDA_HOME

    return os.environ.get("CUDA_HOME") or CUDA_HOME


def _find(name: str, patterns: list[str]) -> str | None:
    if path := shutil.which(name):
        return path
    for pattern in patterns:
        matches = sorted(glob.glob(pattern), reverse=True)  # newest version first
        if matches:
            return matches[0]
    return None


def find_ncu() -> str | None:
    home = _cuda_home() or "/usr/local/cuda"
    return _find("ncu", [
        f"{home}/bin/ncu",
        f"{home}/nsight-compute*/ncu",
        "/usr/local/cuda*/bin/ncu",
        "/opt/nvidia/nsight-compute/*/ncu",
    ])


def find_compute_sanitizer() -> str | None:
    home = _cuda_home() or "/usr/local/cuda"
    return _find("compute-sanitizer", [
        f"{home}/bin/compute-sanitizer",
        f"{home}/compute-sanitizer/compute-sanitizer",
        "/usr/local/cuda*/bin/compute-sanitizer",
    ])


def dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).removeprefix("torch.")


def shape_arg(shape: tuple[int, ...]) -> str:
    return "x".join(map(str, shape))


def launcher_command(source_path: Path, spec: OpSpec, mode: str, shapes: list[tuple[int, ...]],
                     dtypes: list[torch.dtype]) -> list[str]:
    """Command that runs a kernel in a fresh Python process (see launch.py)."""
    return [
        sys.executable, "-m", "kernelagent.harness.launch",
        "--op", spec.name,
        "--source", str(source_path),
        "--mode", mode,
        "--shapes", ",".join(shape_arg(s) for s in shapes),
        "--dtypes", ",".join(dtype_name(d) for d in dtypes),
    ]


def run_tool(cmd: list[str], timeout: float, env_overrides: dict[str, str] | None = None) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(REPO_ROOT), env.get("PYTHONPATH")]))
    env.update(env_overrides or {})
    return subprocess.run(cmd, cwd=REPO_ROOT, env=env, capture_output=True, text=True, timeout=timeout)

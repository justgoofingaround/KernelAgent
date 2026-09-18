"""Profile one kernel launch with Nsight Compute and turn the report into structured data.

Flow:
  1. Compile the kernel (cached) so the profiled process only loads it.
  2. Run launch.py under `ncu --profile-from-start off`; exactly one call to the
     kernel happens inside the cudaProfilerStart/Stop range.
  3. Read the saved report back as CSV: the raw metrics page gives the numbers,
     the source page gives per-line warp-stall samples ("hot lines").

The .ncu-rep file is kept in .cache/profiles/ so it can be opened in the
Nsight Compute GUI.
"""

from __future__ import annotations

import csv
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch

from ..ops.base import REPO_ROOT, OpSpec, Shape
from .build import compile_kernel
from .roofline import Roofline, classify
from .sourcemap import SourceMap
from .tools import dtype_name, find_ncu, launcher_command, run_tool, shape_arg

PROFILE_DIR = REPO_ROOT / ".cache" / "profiles"

# Friendly name -> Nsight Compute metric.
METRICS: dict[str, str] = {
    "duration_ns": "gpu__time_duration.sum",
    "dram_pct": "dram__throughput.avg.pct_of_peak_sustained_elapsed",
    "mem_pct": "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed",
    "sm_pct": "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "dram_bytes_read": "dram__bytes_read.sum",
    "dram_bytes_write": "dram__bytes_write.sum",
    "l2_hit_pct": "lts__t_sector_hit_rate.pct",
    "achieved_occupancy_pct": "sm__warps_active.avg.pct_of_peak_sustained_active",
    "theoretical_occupancy_pct": "sm__maximum_warps_per_active_cycle_pct",
    "registers_per_thread": "launch__registers_per_thread",
    "smem_static_bytes": "launch__shared_mem_per_block_static",
    "smem_dynamic_bytes": "launch__shared_mem_per_block_dynamic",
    "block_size": "launch__block_size",
    "grid_size": "launch__grid_size",
    "waves_per_sm": "launch__waves_per_multiprocessor",
    "occ_limit_registers": "launch__occupancy_limit_registers",
    "occ_limit_smem": "launch__occupancy_limit_shared_mem",
    "occ_limit_warps": "launch__occupancy_limit_warps",
    "occ_limit_blocks": "launch__occupancy_limit_blocks",
    "gld_sectors": "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum",
    "gld_requests": "l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum",
    "gst_sectors": "l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum",
    "gst_requests": "l1tex__t_requests_pipe_lsu_mem_global_op_st.sum",
    "smem_ld_bank_conflicts": "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum",
    "smem_st_bank_conflicts": "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum",
    "tensor_pipe_pct": "sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active",
    "ffma": "sm__sass_thread_inst_executed_op_ffma_pred_on.sum",
    "fadd": "sm__sass_thread_inst_executed_op_fadd_pred_on.sum",
    "fmul": "sm__sass_thread_inst_executed_op_fmul_pred_on.sum",
    "hfma": "sm__sass_thread_inst_executed_op_hfma_pred_on.sum",
    "hadd": "sm__sass_thread_inst_executed_op_hadd_pred_on.sum",
    "hmul": "sm__sass_thread_inst_executed_op_hmul_pred_on.sum",
    # Peaks, for the roofline (same formulas Nsight Compute's roofline chart uses).
    "peak_ffma_per_cycle": "sm__sass_thread_inst_executed_op_ffma_pred_on.sum.peak_sustained",
    "sm_cycles_per_sec": "sm__cycles_elapsed.avg.per_second",
    "peak_dram_bytes_per_cycle": "dram__bytes.sum.peak_sustained",
    "dram_cycles_per_sec": "dram__cycles_elapsed.avg.per_second",
}

# Warp stall reasons: average cycles a warp spends stalled per issued instruction.
STALL_REASONS: dict[str, str] = {
    "long_scoreboard": "waiting on global/local memory (L1TEX) results",
    "short_scoreboard": "waiting on shared memory or special math (MUFU) results",
    "barrier": "waiting at __syncthreads()",
    "membar": "waiting on a memory fence",
    "mio_throttle": "shared-memory / special-instruction queue is full",
    "lg_throttle": "global/local memory instruction queue is full (too many small accesses)",
    "math_pipe_throttle": "math pipeline is busy (compute-bound)",
    "wait": "fixed-latency dependency between instructions",
    "not_selected": "eligible but another warp was picked (healthy)",
    "selected": "issuing an instruction (healthy)",
    "no_instruction": "instruction cache miss or branch target not ready",
    "dispatch_stall": "dispatch stalled by the pipeline",
    "drain": "waiting for memory writes to drain at exit",
    "branch_resolving": "waiting for a branch target to be computed",
    "tex_throttle": "texture unit queue is full",
    "imc_miss": "constant cache miss",
    "sleeping": "warp is sleeping",
}
STALL_METRIC = "smsp__average_warps_issue_stalled_{}_per_issue_active.ratio"

_MISSING_METRIC = re.compile(r"Failed to find metrics?\s*(?:regex:)?\s*([\w.]+)")
_PERMISSION_HINT = "ERR_NVGPUCTRPERM"


class ProfileError(RuntimeError):
    pass


@dataclass
class KernelProfile:
    name: str
    block: str
    grid: str
    metrics: dict[str, float | None]
    stalls: dict[str, float]

    @property
    def duration_ns(self) -> float:
        return self.metrics.get("duration_ns") or 0.0


@dataclass
class Hotspot:
    line: int
    text: str
    samples: float
    pct: float


@dataclass
class ProfileResult:
    op: str
    shape: Shape
    dtype: str
    gpu: str
    report_path: str
    kernels: list[KernelProfile]
    hotspots: list[Hotspot] = field(default_factory=list)
    skipped_metrics: list[str] = field(default_factory=list)

    @property
    def primary(self) -> KernelProfile:
        """The longest-running kernel in the profiled call (usually the only one)."""
        return max(self.kernels, key=lambda k: k.duration_ns)

    @property
    def roofline(self) -> Roofline:
        return classify(self.primary.metrics)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["roofline"] = self.roofline.to_dict()
        return d

    def to_json(self, path: str | Path | None = None) -> str:
        text = json.dumps(self.to_dict(), indent=2, default=str)
        if path:
            Path(path).write_text(text, encoding="utf-8")
        return text

    def summary(self) -> str:
        from .feedback import format_profile

        return format_profile(self)


# ---------------------------------------------------------------- CSV parsing

def _number(text: str) -> float | None:
    text = text.strip().replace(",", "")
    if not text or text.lower() in {"n/a", "nan", "-"}:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _csv_rows(text: str) -> list[list[str]]:
    """ncu mixes its own "==PROF==" / "==WARNING==" lines into CSV output; drop them."""
    lines = [ln for ln in text.splitlines() if ln.strip() and not ln.startswith("==")]
    return list(csv.reader(lines))


def parse_raw_csv(text: str) -> list[KernelProfile]:
    """Parse `ncu --csv --page raw --print-units base` output: one row per kernel."""
    rows = _csv_rows(text)
    header_idx = next((i for i, r in enumerate(rows) if r and r[0] == "ID"), None)
    if header_idx is None:
        return []
    header = rows[header_idx]
    col = {name: i for i, name in enumerate(header)}

    kernels = []
    for row in rows[header_idx + 1:]:
        if len(row) != len(header) or not row[0].strip():
            continue  # units row (empty ID) or a malformed line

        def value(metric: str) -> float | None:
            i = col.get(metric)
            return _number(row[i]) if i is not None else None

        metrics = {key: value(metric) for key, metric in METRICS.items()}
        stalls = {r: v for r in STALL_REASONS if (v := value(STALL_METRIC.format(r))) is not None}
        kernels.append(KernelProfile(
            name=row[col["Kernel Name"]] if "Kernel Name" in col else "?",
            block=row[col["Block Size"]] if "Block Size" in col else "?",
            grid=row[col["Grid Size"]] if "Grid Size" in col else "?",
            metrics=metrics,
            stalls=stalls,
        ))
    return kernels


def parse_source_csv(text: str, source_map: SourceMap, top: int = 5) -> list[Hotspot]:
    """Parse `ncu --csv --page source --print-source cuda` into the hottest lines of
    the kernel author's source, ranked by warp-stall samples.

    The source page layout differs across ncu versions, so rows are matched to
    the original file by their source text rather than by reported line number.
    """
    samples_by_line: dict[int, float] = {}
    header: list[str] | None = None
    text_col = samples_col = None
    for row in _csv_rows(text):
        if "Source" in row:
            header = row
            text_col = row.index("Source")
            samples_col = next((i for i, h in enumerate(row) if "Warp Stall Sampling (All" in h),
                               next((i for i, h in enumerate(row) if "Sampling" in h), None))
            continue
        if header is None or samples_col is None or len(row) != len(header):
            continue
        samples = _number(row[samples_col])
        line = source_map.find_text(row[text_col])
        if samples and line is not None:
            samples_by_line[line] = samples_by_line.get(line, 0.0) + samples

    total = sum(samples_by_line.values())
    ranked = sorted(samples_by_line.items(), key=lambda kv: kv[1], reverse=True)[:top]
    return [Hotspot(line, source_map.lines[line - 1].strip(), s, 100 * s / total) for line, s in ranked]


# ---------------------------------------------------------------- running ncu

def _all_metric_names() -> list[str]:
    return list(METRICS.values()) + [STALL_METRIC.format(r) for r in STALL_REASONS]


def profile_kernel(source: str, spec: OpSpec, shape: Shape, dtype: torch.dtype = torch.float16,
                   hotspots: bool = True, timeout: float = 1200) -> ProfileResult:
    """Profile one launch of `source` on the given shape/dtype. Raises ProfileError
    with the ncu log if profiling fails."""
    ncu = find_ncu()
    if ncu is None:
        raise ProfileError("Nsight Compute (ncu) not found; install the CUDA toolkit or run on Colab")

    build = compile_kernel(source, spec)
    if not build.ok:
        raise ProfileError(f"kernel does not compile:\n{build.error_summary}")
    source_path = build.build_dir / "source.cu"
    source_path.write_text(source, encoding="utf-8")

    PROFILE_DIR.mkdir(parents=True, exist_ok=True)
    report = PROFILE_DIR / f"{build.name}_{shape_arg(shape)}_{dtype_name(dtype)}"
    app = launcher_command(source_path, spec, "profile", [shape], [dtype])

    metrics = _all_metric_names()
    skipped: list[str] = []
    for _ in range(10):  # retry, dropping metrics this GPU / ncu version doesn't have
        cmd = [ncu, "--profile-from-start", "off", "--target-processes", "application-only",
               "--metrics", ",".join(metrics), "--export", str(report), "--force-overwrite"]
        if hotspots:
            cmd += ["--section", "SourceCounters", "--import-source", "yes"]
        proc = run_tool(cmd + app, timeout)
        log = proc.stdout + proc.stderr
        missing = [m for m in _MISSING_METRIC.findall(log) if m in metrics]
        if proc.returncode == 0 or not missing:
            break
        skipped += missing
        metrics = [m for m in metrics if m not in missing]

    if _PERMISSION_HINT in log:
        raise ProfileError("ncu is not allowed to read GPU performance counters (ERR_NVGPUCTRPERM). "
                           "Run as root or enable counter access for non-admin users.")
    if proc.returncode != 0:
        raise ProfileError(f"ncu failed (exit {proc.returncode}):\n{log[-4000:]}")

    rep_file = report.with_suffix(".ncu-rep")
    raw = run_tool([ncu, "--import", str(rep_file), "--csv", "--page", "raw", "--print-units", "base"], timeout)
    kernels = parse_raw_csv(raw.stdout)
    if not kernels:
        raise ProfileError(f"no kernels were profiled. ncu output:\n{log[-4000:]}\n{raw.stdout[-2000:]}{raw.stderr[-2000:]}")

    hot: list[Hotspot] = []
    if hotspots:
        src = run_tool([ncu, "--import", str(rep_file), "--csv", "--page", "source", "--print-source", "cuda"], timeout)
        if src.returncode == 0:
            hot = parse_source_csv(src.stdout, SourceMap.from_build_dir(build.build_dir, source))

    return ProfileResult(spec.name, tuple(shape), dtype_name(dtype), torch.cuda.get_device_name(),
                         str(rep_file), kernels, hot, skipped)

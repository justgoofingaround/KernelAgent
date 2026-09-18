"""Classify a profiled kernel as memory-, compute- or latency-bound.

Uses the same signals as Nsight Compute's "GPU Speed Of Light" section:
- Memory throughput %: how busy the busiest memory unit (DRAM, L2, L1) is.
- SM throughput %:     how busy the busiest compute pipeline is.
If neither is near its peak, the kernel is waiting on latency (too little
parallelism, stalls, or it's so small that launch overhead dominates).

It also reports classic roofline numbers: arithmetic intensity (FLOPs per DRAM
byte) against the GPU's ridge point (peak FLOP/s ÷ peak bytes/s).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

BUSY_THRESHOLD_PCT = 60.0   # below this on both axes -> latency-bound
TINY_KERNEL_NS = 5_000      # launch overhead is a few microseconds


@dataclass
class Roofline:
    bound: str                  # "memory", "compute", "latency" or "unknown"
    memory_pct: float | None
    sm_pct: float | None
    duration_ns: float | None
    flops: float | None
    dram_bytes: float | None
    arithmetic_intensity: float | None  # FLOPs / DRAM byte
    peak_flops: float | None            # fp32 FLOP/s
    peak_bandwidth: float | None        # DRAM bytes/s
    ridge_point: float | None           # FLOPs / byte where the roofline bends
    achieved_bandwidth: float | None    # DRAM bytes/s
    explanation: str

    def to_dict(self) -> dict:
        return asdict(self)


def _get(m: dict[str, float | None], key: str) -> float | None:
    v = m.get(key)
    return None if v is None else float(v)


def _mul(*values: float | None) -> float | None:
    out = 1.0
    for v in values:
        if v is None:
            return None
        out *= v
    return out


def fp_flops(m: dict[str, float | None]) -> float | None:
    """Non-tensor-core FLOPs (fp32 + fp16), counting an FMA as 2 FLOPs."""
    parts = {k: _get(m, k) for k in ("ffma", "fadd", "fmul", "hfma", "hadd", "hmul")}
    if all(v is None for v in parts.values()):
        return None
    p = {k: v or 0.0 for k, v in parts.items()}
    return 2 * p["ffma"] + p["fadd"] + p["fmul"] + 2 * p["hfma"] + p["hadd"] + p["hmul"]


def classify(m: dict[str, float | None], threshold: float = BUSY_THRESHOLD_PCT) -> Roofline:
    memory_pct = max((v for v in (_get(m, "mem_pct"), _get(m, "dram_pct")) if v is not None), default=None)
    sm_pct = _get(m, "sm_pct")
    duration = _get(m, "duration_ns")

    reads, writes = _get(m, "dram_bytes_read"), _get(m, "dram_bytes_write")
    dram_bytes = None if reads is None and writes is None else (reads or 0.0) + (writes or 0.0)
    flops = fp_flops(m)
    intensity = flops / dram_bytes if flops is not None and dram_bytes else None

    peak_flops = _mul(_get(m, "peak_ffma_per_cycle"), 2.0, _get(m, "sm_cycles_per_sec"))
    peak_bw = _mul(_get(m, "peak_dram_bytes_per_cycle"), _get(m, "dram_cycles_per_sec"))
    ridge = peak_flops / peak_bw if peak_flops and peak_bw else None
    achieved_bw = dram_bytes / (duration * 1e-9) if dram_bytes is not None and duration else None

    if memory_pct is None or sm_pct is None:
        bound, why = "unknown", "memory or SM throughput metric was not collected"
    elif max(memory_pct, sm_pct) < threshold:
        bound = "latency"
        why = (f"neither memory ({memory_pct:.0f}%) nor compute ({sm_pct:.0f}%) is near peak; "
               "the GPU is waiting on latency (low parallelism, stalls or synchronization)")
        if duration is not None and duration < TINY_KERNEL_NS:
            why += f"; the kernel only runs {duration / 1000:.1f} us, so launch overhead dominates"
    elif memory_pct >= sm_pct:
        bound = "memory"
        why = (f"memory throughput is {memory_pct:.0f}% of peak vs {sm_pct:.0f}% for compute; "
               "speedups must come from moving fewer bytes (fusion, vectorization, better caching)")
    else:
        bound = "compute"
        why = (f"compute throughput is {sm_pct:.0f}% of peak vs {memory_pct:.0f}% for memory; "
               "speedups must come from fewer or cheaper instructions (or Tensor Cores)")

    return Roofline(bound, memory_pct, sm_pct, duration, flops, dram_bytes, intensity,
                    peak_flops, peak_bw, ridge, achieved_bw, why)

"""Turn profiler and sanitizer results into short plain-text feedback.

This text is what the agent sees in M3/M4, so it states facts with numbers,
explains what each number means, and flags likely problems. It deliberately
doesn't prescribe fixes; that's the critic agent's job.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .ncu import ProfileResult
    from .sanitizer import SanitizerReport

HEALTHY_STALLS = {"selected", "not_selected"}


def _pct(v: float | None) -> str:
    return "n/a" if v is None else f"{v:.0f}%"


def _num(v: float | None, fmt: str = "{:.0f}") -> str:
    return "n/a" if v is None else fmt.format(v)


def _ratio(a: float | None, b: float | None) -> float | None:
    return a / b if a is not None and b else None


def _occupancy_limiter(m: dict[str, float | None]) -> str | None:
    """Which resource caps blocks per SM (lowest limit wins)."""
    limits = {name: m.get(key) for name, key in (
        ("registers", "occ_limit_registers"), ("shared memory", "occ_limit_smem"),
        ("warps per SM", "occ_limit_warps"), ("blocks per SM", "occ_limit_blocks"))}
    known = {k: v for k, v in limits.items() if v is not None}
    return min(known, key=known.get) if known else None


def observations(profile: "ProfileResult") -> list[str]:
    """Rule-based flags for things that commonly limit kernel performance."""
    from .ncu import STALL_REASONS

    k = profile.primary
    m = k.metrics
    notes = []

    waves = m.get("waves_per_sm")
    if waves is not None and waves < 1:
        notes.append(f"Grid fills only {waves:.2f} waves of the GPU: some SMs sit idle. "
                     "Consider more, smaller blocks or more work per launch.")

    achieved, theoretical = m.get("achieved_occupancy_pct"), m.get("theoretical_occupancy_pct")
    if achieved is not None and theoretical and achieved < 0.6 * theoretical:
        notes.append(f"Achieved occupancy ({achieved:.0f}%) is far below theoretical ({theoretical:.0f}%): "
                     "warps finish unevenly or the grid is too small to keep SMs full.")
    if theoretical is not None and theoretical < 50:
        notes.append(f"Theoretical occupancy is only {theoretical:.0f}%, limited by "
                     f"{_occupancy_limiter(m) or 'an unknown resource'}.")

    conflicts = (m.get("smem_ld_bank_conflicts") or 0) + (m.get("smem_st_bank_conflicts") or 0)
    if conflicts > 0:
        notes.append(f"{conflicts:.0f} shared-memory bank conflicts: threads in a warp hit the same bank; "
                     "consider padding or a different shared-memory layout.")

    load_spr = _ratio(m.get("gld_sectors"), m.get("gld_requests"))
    if load_spr is not None and load_spr > 16.5:
        notes.append(f"Global loads use {load_spr:.1f} sectors per request (more than a fully coalesced "
                     "128-bit access needs): accesses within a warp are likely scattered.")

    stalls = {r: v for r, v in k.stalls.items() if r not in HEALTHY_STALLS}
    if stalls:
        top, cycles = max(stalls.items(), key=lambda kv: kv[1])
        total = sum(stalls.values())
        if total and cycles / total > 0.4:
            notes.append(f"Most stall time is '{top}' ({STALL_REASONS[top]}), "
                         f"{cycles:.1f} cycles per instruction.")

    if profile.roofline.bound == "memory" and profile.roofline.memory_pct and profile.roofline.memory_pct > 80:
        notes.append("Memory throughput is already above 80% of peak: further gains need fewer bytes "
                     "moved (e.g. fusing with neighbouring ops), not faster code.")
    return notes


def format_profile(profile: "ProfileResult") -> str:
    from .ncu import STALL_REASONS

    k = profile.primary
    m = k.metrics
    r = profile.roofline
    duration_us = (m.get("duration_ns") or 0) / 1000

    lines = [
        f"PROFILE {profile.op} shape={profile.shape} dtype={profile.dtype} on {profile.gpu}",
        f"Kernel: {k.name}",
        f"  grid {k.grid} x block {k.block}, {duration_us:.1f} us (ncu locks clocks to base, so "
        "this is slower than benchmark timings)",
        f"Bound: {r.bound.upper()}: {r.explanation}",
        f"  memory throughput {_pct(r.memory_pct)} | DRAM {_pct(m.get('dram_pct'))} | SM {_pct(r.sm_pct)}"
        + (f" | tensor pipe {_pct(m.get('tensor_pipe_pct'))}" if m.get("tensor_pipe_pct") else ""),
    ]
    if r.achieved_bandwidth is not None:
        peak = f" of {r.peak_bandwidth / 1e9:.0f} GB/s peak" if r.peak_bandwidth else ""
        lines.append(f"  DRAM traffic {r.dram_bytes / 1e6:.2f} MB -> {r.achieved_bandwidth / 1e9:.0f} GB/s{peak}")
    if r.arithmetic_intensity is not None:
        ridge = f" (ridge point {r.ridge_point:.1f})" if r.ridge_point else ""
        lines.append(f"  arithmetic intensity {r.arithmetic_intensity:.2f} FLOP/byte{ridge}")

    limiter = _occupancy_limiter(m)
    lines += [
        f"Occupancy: achieved {_pct(m.get('achieved_occupancy_pct'))} / theoretical "
        f"{_pct(m.get('theoretical_occupancy_pct'))}"
        + (f" (limited by {limiter})" if limiter and (m.get("theoretical_occupancy_pct") or 100) < 100 else ""),
        f"  {_num(m.get('registers_per_thread'))} registers/thread, shared memory "
        f"{_num(m.get('smem_static_bytes'))} B static + {_num(m.get('smem_dynamic_bytes'))} B dynamic per block, "
        f"{_num(m.get('waves_per_sm'), '{:.2f}')} waves",
        "Memory access:",
        f"  global load {_num(_ratio(m.get('gld_sectors'), m.get('gld_requests')), '{:.1f}')} sectors/request, "
        f"store {_num(_ratio(m.get('gst_sectors'), m.get('gst_requests')), '{:.1f}')} sectors/request "
        "(4 = coalesced 32-bit, 16 = coalesced 128-bit per thread)",
        f"  L2 hit rate {_pct(m.get('l2_hit_pct'))}; shared-memory bank conflicts: "
        f"{_num(m.get('smem_ld_bank_conflicts'))} load, {_num(m.get('smem_st_bank_conflicts'))} store",
    ]

    if k.stalls:
        top = sorted(((r_, v) for r_, v in k.stalls.items() if r_ not in HEALTHY_STALLS),
                     key=lambda kv: kv[1], reverse=True)[:4]
        lines.append("Top warp stalls (cycles per issued instruction):")
        lines += [f"  {reason:<18} {v:6.2f}  {STALL_REASONS[reason]}" for reason, v in top]

    if profile.hotspots:
        lines.append("Hottest source lines (share of warp-stall samples):")
        lines += [f"  line {h.line:<4} {h.pct:5.1f}%  {h.text}" for h in profile.hotspots]

    if len(profile.kernels) > 1:
        others = ", ".join(f"{o.name} ({o.duration_ns / 1000:.1f} us)" for o in profile.kernels if o is not k)
        lines.append(f"Other kernels launched by forward(): {others}")

    notes = observations(profile)
    if notes:
        lines.append("Observations:")
        lines += [f"  - {n}" for n in notes]
    if profile.skipped_metrics:
        lines.append(f"(metrics unavailable on this GPU/ncu: {', '.join(profile.skipped_metrics)})")
    return "\n".join(lines)


def format_feedback(profile: "ProfileResult | None" = None,
                    sanitizer: "dict[str, SanitizerReport] | None" = None) -> str:
    """Combined report: safety first (a racy kernel's speed doesn't matter), then performance."""
    parts = []
    if sanitizer:
        parts.append("SANITIZER\n" + "\n".join(rep.summary() for rep in sanitizer.values()))
    if profile:
        parts.append(format_profile(profile))
    return "\n\n".join(parts)

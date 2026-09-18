"""CPU tests for the ncu / compute-sanitizer parsers, roofline and feedback text.

The sample outputs below follow the formats Nsight Compute and compute-sanitizer
print; the GPU tests in test_tools.py check the real tools end to end.
"""

import csv
import io

import pytest

from kernelagent.harness.feedback import format_feedback
from kernelagent.harness.ncu import METRICS, STALL_METRIC, ProfileResult, parse_raw_csv, parse_source_csv
from kernelagent.harness.roofline import classify
from kernelagent.harness.sanitizer import SanitizerReport, parse_output
from kernelagent.harness.sourcemap import SourceMap

KERNEL_SOURCE = """#include <torch/extension.h>
__global__ void k(const float* x, float* y, int64_t n) {
  int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  float v = x[i];
  y[i] = v * 2.f;
}
"""
BUILD = "/content/KernelAgent/.cache/kernels/ka_softmax_0123/cuda.cu"
OFFSET = 3  # load_inline prepends three #include lines


def _raw_csv(values: dict[str, str], stalls: dict[str, str]) -> str:
    """Build `ncu --csv --page raw --print-units base` output with a units row and ncu log lines."""
    fixed = ["ID", "Process ID", "Process Name", "Host Name", "Kernel Name", "Context", "Stream",
             "Block Size", "Grid Size", "Device", "CC"]
    metric_cols = list(METRICS.values()) + [STALL_METRIC.format(r) for r in stalls]
    all_values = {METRICS[k]: v for k, v in values.items()}
    all_values.update({STALL_METRIC.format(r): v for r, v in stalls.items()})

    buf = io.StringIO()
    w = csv.writer(buf, quoting=csv.QUOTE_ALL, lineterminator="\n")
    w.writerow(fixed + metric_cols)
    w.writerow([""] * len(fixed) + ["" for _ in metric_cols])
    w.writerow(["0", "4242", "python3", "127.0.0.1", "void softmax_kernel<c10::Half>(const T1 *, T1 *, long)",
                "1", "7", "(256, 1, 1)", "(4096, 1, 1)", "0", "8.9"]
               + [all_values.get(c, "n/a") for c in metric_cols])
    return "==PROF== Connected to process 4242 (/usr/bin/python3)\n" + buf.getvalue() + "==PROF== Disconnected from process 4242\n"


MEMORY_BOUND = {
    "duration_ns": "33,568", "dram_pct": "87.5", "mem_pct": "88.1", "sm_pct": "23.4",
    "dram_bytes_read": "8,392,704", "dram_bytes_write": "8,388,608",
    "achieved_occupancy_pct": "40.1", "theoretical_occupancy_pct": "100",
    "registers_per_thread": "32", "smem_static_bytes": "264", "smem_dynamic_bytes": "0",
    "waves_per_sm": "5.3", "gld_sectors": "262,144", "gld_requests": "65,536",
    "smem_ld_bank_conflicts": "0", "smem_st_bank_conflicts": "12",
    "ffma": "4,194,304", "fadd": "0", "fmul": "4,194,304",
    "peak_ffma_per_cycle": "7,680", "sm_cycles_per_sec": "1,410,000,000",
    "peak_dram_bytes_per_cycle": "1,280", "dram_cycles_per_sec": "1,215,000,000",
    "occ_limit_registers": "8", "occ_limit_smem": "32", "occ_limit_warps": "8", "occ_limit_blocks": "16",
}
STALLS = {"long_scoreboard": "14.2", "barrier": "2.1", "selected": "1.0"}


def test_parse_raw_csv_reads_metrics_and_skips_units_row():
    kernels = parse_raw_csv(_raw_csv(MEMORY_BOUND, STALLS))
    assert len(kernels) == 1
    k = kernels[0]
    assert "softmax_kernel" in k.name
    assert k.grid == "(4096, 1, 1)" and k.block == "(256, 1, 1)"
    assert k.metrics["duration_ns"] == 33568  # thousands separators stripped
    assert k.metrics["dram_pct"] == 87.5
    assert k.metrics["tensor_pipe_pct"] is None  # "n/a"
    assert k.stalls == {"long_scoreboard": 14.2, "barrier": 2.1, "selected": 1.0}


def test_parse_raw_csv_without_header_returns_nothing():
    assert parse_raw_csv("==ERROR== something went wrong\n") == []


@pytest.mark.parametrize("mem, sm, duration, expected", [
    (88, 23, 33_568, "memory"),
    (30, 85, 50_000, "compute"),
    (20, 15, 50_000, "latency"),
    (5, 3, 2_000, "latency"),
])
def test_roofline_classification(mem, sm, duration, expected):
    r = classify({"mem_pct": mem, "dram_pct": mem, "sm_pct": sm, "duration_ns": duration})
    assert r.bound == expected
    if duration < 5_000:
        assert "launch overhead" in r.explanation


def test_roofline_numbers():
    k = parse_raw_csv(_raw_csv(MEMORY_BOUND, STALLS))[0]
    r = classify(k.metrics)
    assert r.dram_bytes == 8_392_704 + 8_388_608
    assert r.flops == 2 * 4_194_304 + 4_194_304
    assert r.arithmetic_intensity == pytest.approx(r.flops / r.dram_bytes)
    assert r.peak_bandwidth == pytest.approx(1280 * 1.215e9)
    assert r.ridge_point == pytest.approx(7680 * 2 * 1.41e9 / (1280 * 1.215e9))
    assert r.achieved_bandwidth == pytest.approx(r.dram_bytes / 33_568e-9)


def test_missing_metrics_are_unknown_not_a_crash():
    assert classify({}).bound == "unknown"


def test_sourcemap_undoes_load_inline_prefix(tmp_path):
    header = "#include <torch/types.h>\n#include <cuda.h>\n#include <cuda_runtime.h>\n"
    (tmp_path / "cuda.cu").write_text(header + KERNEL_SOURCE)
    sm = SourceMap.from_build_dir(tmp_path, KERNEL_SOURCE)
    assert sm.offset == OFFSET
    assert sm.lookup(str(tmp_path / "cuda.cu"), 4 + OFFSET) == (4, "float v = x[i];")
    assert sm.lookup(str(tmp_path / "cuda.cu"), 1) is None             # inside the prepended header
    assert sm.lookup("/usr/include/c10/Half.h", 4 + OFFSET) is None    # not the kernel file


MEMCHECK_OUTPUT = f"""========= COMPUTE-SANITIZER
========= Invalid __global__ read of size 4 bytes
=========     at k(const float *, float *, long)+0x2c0 in {BUILD}:{4 + OFFSET}
=========     by thread (31,0,0) in block (3,0,0)
=========     Address 0x7f3c2e000400 is out of bounds
=========     and is 1 bytes after the nearest allocation at 0x7f3c2e000000 of size 1024 bytes
=========     Saved host backtrace up to driver entry point at kernel launch time
=========     Host Frame: [0x2f0e3f] in libcuda.so.1
=========     Host Frame:k.cu:99 [0x1234] in python3
=========
========= Program hit cudaErrorLaunchFailure (error 719) due to "unspecified launch failure" on CUDA API call to cudaStreamSynchronize.
=========     Saved host backtrace up to driver entry point at error
=========
Traceback (most recent call last):
RuntimeError: CUDA error: unspecified launch failure
========= ERROR SUMMARY: 2 errors
"""

RACECHECK_OUTPUT = f"""========= COMPUTE-SANITIZER
========= Error: Race reported between Write access at k(const float *, float *, long)+0x90 in {BUILD}:{3 + OFFSET}
=========     and Read access at k(const float *, float *, long)+0x100 in {BUILD}:{5 + OFFSET} [7168 hazards]
=========
========= RACECHECK SUMMARY: 1 hazard displayed (1 error, 0 warnings)
"""


def _source_map(tmp_path):
    return SourceMap(KERNEL_SOURCE, OFFSET)


def test_parse_memcheck_output(tmp_path):
    findings, count = parse_output("memcheck", MEMCHECK_OUTPUT, _source_map(tmp_path))
    assert count == 2
    device = [f for f in findings if not f.is_api_error]
    assert len(device) == 1
    f = device[0]
    assert f.message == "Invalid __global__ read of size 4 bytes"
    assert f.kernel == "k(const float *, float *, long)"
    assert [(l.source_line, l.source_text) for l in f.locations] == [(4, "float v = x[i];")]
    assert not any("Host Frame" in d for d in f.details)  # backtrace dropped
    assert "line 4: `float v = x[i];`" in f.describe()
    assert "1 bytes after the nearest allocation" in f.describe()


def test_parse_racecheck_output(tmp_path):
    findings, count = parse_output("racecheck", RACECHECK_OUTPUT, _source_map(tmp_path))
    assert count == 1
    assert len(findings) == 1
    lines = [l.source_line for l in findings[0].locations]
    assert lines == [3, 5]  # both sides of the race
    assert findings[0].message.startswith("Race reported between Write access")
    described = findings[0].describe()
    assert described.startswith("Race reported between Write access and Read access [7168 hazards] in k(")
    assert "/content/" not in described  # build paths replaced by source lines


def test_clean_sanitizer_report():
    findings, count = parse_output("memcheck", "========= COMPUTE-SANITIZER\n========= ERROR SUMMARY: 0 errors\n")
    assert findings == [] and count == 0
    assert SanitizerReport("memcheck", 0, findings, count, "").summary() == "memcheck: clean"


def test_parse_source_csv_ranks_hot_lines():
    sm = SourceMap(KERNEL_SOURCE, OFFSET)
    csv_text = (
        '"#","Address","Source","Warp Stall Sampling (All Samples)","Warp Stall Sampling (Not-issued Samples)"\n'
        '"1","","  int64_t i = blockIdx.x * blockDim.x + threadIdx.x;","10","2"\n'
        '"2","","  float v = x[i];","70","60"\n'
        '"3","","  y[i] = v * 2.f;","20","5"\n'
        '"4","","}","99","99"\n'  # braces are ignored
    )
    hot = parse_source_csv(csv_text, sm)
    assert [(h.line, round(h.pct)) for h in hot] == [(4, 70), (5, 20), (3, 10)]


def test_feedback_text_mentions_key_facts():
    kernels = parse_raw_csv(_raw_csv(MEMORY_BOUND, STALLS))
    profile = ProfileResult("softmax", (4096, 1024), "float16", "NVIDIA L4", "/tmp/r.ncu-rep", kernels)
    findings, count = parse_output("racecheck", RACECHECK_OUTPUT, SourceMap(KERNEL_SOURCE, OFFSET))
    text = format_feedback(profile, {"racecheck": SanitizerReport("racecheck", 99, findings, count, "")})

    assert text.index("SANITIZER") < text.index("PROFILE")  # safety first
    assert "racecheck: 1 error(s)" in text
    assert "Bound: MEMORY" in text
    assert "long_scoreboard" in text
    assert "12 shared-memory bank conflicts" in text
    assert "limited by" not in text  # theoretical occupancy is 100%, nothing limits it
    assert "above 80% of peak" in text
    assert "4.0 sectors/request" in text

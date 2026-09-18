"""Run a kernel under compute-sanitizer and parse the findings.

Tools:
  memcheck   out-of-bounds / misaligned global and shared memory accesses
  racecheck  shared-memory data races (e.g. a missing __syncthreads())
  initcheck  reads of global memory that was never written

PyTorch's caching allocator hands out slices of large cudaMalloc blocks, which
hides small out-of-bounds accesses from memcheck. The sanitized process runs
with PYTORCH_NO_CUDA_MEMORY_CACHING=1 so every tensor is its own allocation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import torch

from ..ops.base import OpSpec, Shape
from .build import compile_kernel
from .sourcemap import SourceMap
from .tools import find_compute_sanitizer, launcher_command, run_tool

DEFAULT_TOOLS = ("memcheck", "racecheck", "initcheck")

_HEADER = re.compile(r"^========= (\S.*)$")
_DETAIL = re.compile(r"^=========\s{2,}(\S.*)$")
_LOCATION = re.compile(r"([^\s:()]+\.(?:cu|cuh|h|hpp|cpp)):(\d+)")
_KERNEL = re.compile(r"\bat (?:0x[0-9a-f]+ in )?(.+?)(?:\+0x[0-9a-f]+)? in \S+:\d+")
_ERROR_SUMMARY = re.compile(r"ERROR SUMMARY: (\d+) error")
_RACE_SUMMARY = re.compile(r"RACECHECK SUMMARY: \d+ hazards? displayed \((\d+) errors?")
_AT_LOCATION = re.compile(r"\s+at \S.*? in \S+:\d+")
_IGNORED_HEADERS = ("COMPUTE-SANITIZER", "ERROR SUMMARY", "RACECHECK SUMMARY", "LEAK SUMMARY",
                    "Target application returned")


@dataclass
class SourceLocation:
    file: str
    line: int
    source_line: int | None = None  # line in the kernel author's file
    source_text: str | None = None


@dataclass
class SanitizerFinding:
    tool: str
    message: str
    kernel: str | None = None
    locations: list[SourceLocation] = field(default_factory=list)
    details: list[str] = field(default_factory=list)

    @property
    def is_api_error(self) -> bool:
        """Host-side follow-on errors ("Program hit cudaError...") caused by a device fault."""
        return self.message.startswith("Program hit")

    def describe(self) -> str:
        where = [f"line {l.source_line}: `{l.source_text}`" for l in self.locations if l.source_line]
        if not where:
            where = [f"{l.file}:{l.line}" for l in self.locations]
        text = _AT_LOCATION.sub("", self.message)
        # racecheck puts the second access on a detail line: "and Read access at ... [N hazards]"
        text += "".join(" " + _AT_LOCATION.sub("", d) for d in self.details if d.startswith("and ") and "access" in d)
        if self.kernel:
            text += f" in {self.kernel}"
        if where:
            text += " at " + " / ".join(where)
        extra = [d for d in self.details if d.startswith(("by thread", "Address", "and is"))]
        if extra:
            text += " (" + "; ".join(extra) + ")"
        return text


@dataclass
class SanitizerReport:
    tool: str
    exit_code: int
    findings: list[SanitizerFinding]
    error_count: int | None
    output: str

    @property
    def ok(self) -> bool:
        return self.exit_code == 0 and not self.findings

    @property
    def device_findings(self) -> list[SanitizerFinding]:
        return [f for f in self.findings if not f.is_api_error]

    def summary(self, max_findings: int = 5) -> str:
        if self.ok:
            return f"{self.tool}: clean"
        findings = self.device_findings or self.findings
        count = self.error_count if self.error_count is not None else len(findings)
        lines = [f"{self.tool}: {count} error(s)"]
        lines += [f"  - {f.describe()}" for f in findings[:max_findings]]
        if not findings:
            tail = "\n    ".join(self.output.strip().splitlines()[-15:])
            lines.append(f"  process exited with code {self.exit_code}; last output:\n    {tail}")
        return "\n".join(lines)


def parse_output(tool: str, text: str, source_map: SourceMap | None = None) -> tuple[list[SanitizerFinding], int | None]:
    findings: list[SanitizerFinding] = []
    current: SanitizerFinding | None = None
    in_backtrace = False
    error_count = None

    for raw in text.splitlines():
        line = raw.rstrip()
        if m := (_ERROR_SUMMARY.search(line) or _RACE_SUMMARY.search(line)):
            error_count = int(m.group(1))
        if header := _HEADER.match(line):
            message = header.group(1)
            if message.startswith(_IGNORED_HEADERS):
                current = None
                continue
            current = SanitizerFinding(tool, message.removeprefix("Error: "))
            findings.append(current)
            in_backtrace = False
            _add_locations(current, message, source_map)
            continue
        if current is None:
            continue
        if detail := _DETAIL.match(line):
            text_ = detail.group(1)
            if text_.startswith("Saved host backtrace"):
                in_backtrace = True
            if in_backtrace:
                continue
            current.details.append(text_)
            _add_locations(current, text_, source_map)
        elif line.strip() == "=========":
            current = None  # blank separator ends the finding

    return findings, error_count


def _add_locations(finding: SanitizerFinding, text: str, source_map: SourceMap | None) -> None:
    if finding.kernel is None and (k := _KERNEL.search(text)):
        finding.kernel = k.group(1).strip()
    for file, line in _LOCATION.findall(text):
        loc = SourceLocation(file, int(line))
        if source_map and (hit := source_map.lookup(file, int(line))):
            loc.source_line, loc.source_text = hit
        finding.locations.append(loc)


def run_sanitizer(source: str, spec: OpSpec, tool: str = "memcheck", shapes: list[Shape] | None = None,
                  dtypes: list[torch.dtype] | None = None, timeout: float = 1800) -> SanitizerReport:
    sanitizer = find_compute_sanitizer()
    if sanitizer is None:
        raise RuntimeError("compute-sanitizer not found; install the CUDA toolkit or run on Colab")

    build = compile_kernel(source, spec)
    if not build.ok:
        raise RuntimeError(f"kernel does not compile:\n{build.error_summary}")
    source_path = build.build_dir / "source.cu"
    source_path.write_text(source, encoding="utf-8")

    app = launcher_command(source_path, spec, "run", shapes or spec.shapes, dtypes or spec.dtypes)
    cmd = [sanitizer, "--tool", tool, "--print-limit", "20", "--error-exitcode", "99"] + app
    proc = run_tool(cmd, timeout, env_overrides={"PYTORCH_NO_CUDA_MEMORY_CACHING": "1"})
    output = proc.stdout + proc.stderr
    findings, count = parse_output(tool, output, SourceMap.from_build_dir(build.build_dir, source))
    return SanitizerReport(tool, proc.returncode, findings, count, output)


def sanitize(source: str, spec: OpSpec, tools: tuple[str, ...] = DEFAULT_TOOLS,
             **kwargs) -> dict[str, SanitizerReport]:
    return {tool: run_sanitizer(source, spec, tool, **kwargs) for tool in tools}

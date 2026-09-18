"""Map line numbers in the compiled file back to the kernel author's source.

`load_inline` writes the kernel to <build_dir>/cuda.cu with a few #include lines
prepended, so ncu and compute-sanitizer report lines that are shifted from the
original .cu file. SourceMap undoes that shift.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

GENERATED_CUDA_FILE = "cuda.cu"


@dataclass
class SourceMap:
    source: str
    offset: int  # number of lines prepended before the original source

    @classmethod
    def from_build_dir(cls, build_dir: Path | None, source: str) -> "SourceMap":
        offset = 0
        if build_dir is not None:
            generated = Path(build_dir) / GENERATED_CUDA_FILE
            if generated.exists():
                text = generated.read_text(encoding="utf-8", errors="replace")
                idx = text.find(source)
                if idx >= 0:
                    offset = text[:idx].count("\n")
        return cls(source, offset)

    @property
    def lines(self) -> list[str]:
        return self.source.splitlines()

    def lookup(self, file: str, line: int) -> tuple[int, str] | None:
        """(original line number, source text) for a reported location, or None
        if the location is outside the kernel source (a header, a torch file)."""
        if Path(file).name != GENERATED_CUDA_FILE:
            return None
        original = line - self.offset
        if 1 <= original <= len(self.lines):
            return original, self.lines[original - 1].strip()
        return None

    def find_text(self, text: str) -> int | None:
        """Line number of the first source line whose stripped text equals `text`."""
        wanted = text.strip()
        if len(wanted) < 3:  # skip braces and blank lines, which match everywhere
            return None
        for i, line in enumerate(self.lines, start=1):
            if line.strip() == wanted:
                return i
        return None

"""Print how much of KernelAgent is built, based on the checkboxes in MILESTONES.md."""

import re
import sys
from pathlib import Path

MILESTONES = Path(__file__).resolve().parent.parent / "MILESTONES.md"
HEADING = re.compile(r"^##\s+(M\d+)\s+(.*)$")
TASK = re.compile(r"^\s*-\s+\[( |x|X)\]\s+")
BAR_WIDTH = 20


def parse(text: str) -> list[tuple[str, int, int]]:
    """Return [(milestone title, done, total)] in file order."""
    milestones: list[tuple[str, int, int]] = []
    for line in text.splitlines():
        if m := HEADING.match(line):
            milestones.append((f"{m.group(1)} {m.group(2).strip()}", 0, 0))
        elif (m := TASK.match(line)) and milestones:
            title, done, total = milestones[-1]
            milestones[-1] = (title, done + (m.group(1) != " "), total + 1)
    return milestones


def bar(done: int, total: int) -> str:
    filled = round(BAR_WIDTH * done / total) if total else 0
    return "[" + "#" * filled + "." * (BAR_WIDTH - filled) + "]"


def main() -> None:
    milestones = parse(MILESTONES.read_text(encoding="utf-8"))
    if not milestones:
        sys.exit(f"No milestones found in {MILESTONES}")

    width = max(len(title) for title, _, _ in milestones)
    for title, done, total in milestones:
        pct = 100 * done / total if total else 0
        status = "done" if total and done == total else ""
        print(f"{title:<{width}}  {bar(done, total)} {pct:5.1f}%  ({done}/{total}) {status}")

    done = sum(d for _, d, _ in milestones)
    total = sum(t for _, _, t in milestones)
    pct = 100 * done / total
    print("-" * (width + 40))
    print(f"{'TOTAL':<{width}}  {bar(done, total)} {pct:5.1f}% built | {100 - pct:.1f}% remaining  ({done}/{total} tasks)")


if __name__ == "__main__":
    main()

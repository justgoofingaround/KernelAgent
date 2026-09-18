"""Profile a kernel with Nsight Compute and/or check it with compute-sanitizer.

Usage (on a GPU machine with the CUDA toolkit, e.g. Colab):
    python scripts/profile_kernel.py --op softmax                       # baseline kernel, profile only
    python scripts/profile_kernel.py --op rmsnorm --sanitize            # + memcheck/racecheck/initcheck
    python scripts/profile_kernel.py --op softmax --source my.cu --shape 4096x4096 --dtype bfloat16
    python scripts/profile_kernel.py --op softmax --json out.json       # machine-readable profile
"""

import argparse
import sys
from pathlib import Path

from kernelagent.harness import ProfileError, format_feedback, profile_kernel, sanitize
from kernelagent.harness.launch import DTYPES, parse_shapes
from kernelagent.ops import OPS, get_op


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--op", required=True, choices=list(OPS))
    parser.add_argument("--source", type=Path, help="kernel .cu file (default: the op's baseline)")
    parser.add_argument("--shape", help="e.g. 4096x1024 (default: the op's first benchmark shape)")
    parser.add_argument("--dtype", default="float16", choices=list(DTYPES))
    parser.add_argument("--sanitize", action="store_true", help="also run compute-sanitizer")
    parser.add_argument("--no-profile", action="store_true", help="skip Nsight Compute")
    parser.add_argument("--no-hotspots", action="store_true", help="skip per-line stall sampling (faster)")
    parser.add_argument("--json", type=Path, help="write the profile as JSON")
    args = parser.parse_args()

    spec = get_op(args.op)
    source = (args.source or spec.baseline_path).read_text(encoding="utf-8")
    shape = parse_shapes(args.shape)[0] if args.shape else spec.bench_shapes[0]

    reports, profile, ok = None, None, True
    if args.sanitize:
        print("Running compute-sanitizer (memcheck, racecheck, initcheck); this takes a few minutes ...", flush=True)
        reports = sanitize(source, spec)
        ok = all(r.ok for r in reports.values())
    if not args.no_profile:
        print(f"Profiling {spec.name} {shape} {args.dtype} with Nsight Compute ...", flush=True)
        try:
            profile = profile_kernel(source, spec, shape, DTYPES[args.dtype], hotspots=not args.no_hotspots)
        except ProfileError as exc:
            print(f"Profiling failed: {exc}")
            ok = False
        else:
            print(f"Report saved to {profile.report_path} (open it in the Nsight Compute GUI)")
            if args.json:
                profile.to_json(args.json)

    print()
    print(format_feedback(profile, reports))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

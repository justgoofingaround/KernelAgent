"""Build, verify and benchmark the hand-written baseline kernels.

Usage (on a GPU machine, e.g. Colab):
    python scripts/run_baselines.py                 # all ops
    python scripts/run_baselines.py --ops softmax   # one op
    python scripts/run_baselines.py --no-compile    # skip torch.compile comparison
"""

import argparse
import sys

import torch

from kernelagent.harness import compare, compile_kernel, peak_bandwidth_gbps, verify
from kernelagent.ops import OPS, get_op


def fmt_ms(result) -> str:
    return f"{result.median_ms:.4f}" if result else "n/a"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ops", nargs="+", default=list(OPS), choices=list(OPS))
    parser.add_argument("--no-compile", action="store_true", help="skip the torch.compile baseline")
    parser.add_argument("--verbose", action="store_true", help="show the full compiler output")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("No CUDA GPU found. Run this on Colab (see notebooks/colab_runner.ipynb).")
        return 1

    gpu = torch.cuda.get_device_name()
    peak = peak_bandwidth_gbps()
    print(f"GPU: {gpu} | peak DRAM bandwidth: {peak or 'unknown'} GB/s | torch {torch.__version__}\n")

    rows, all_ok = [], True
    for name in args.ops:
        spec = get_op(name)
        print(f"[{name}] compiling {spec.baseline_path.name} ...", flush=True)
        build = compile_kernel(spec.baseline_source(), spec, verbose=args.verbose)
        if not build.ok:
            print(build.error_summary)
            all_ok = False
            continue
        print(f"[{name}] built in {build.seconds:.1f}s, verifying ...", flush=True)

        report = verify(build.module.forward, spec)
        print(report.summary())
        if not report.ok:
            all_ok = False
            continue

        for dtype in spec.dtypes:
            for shape in spec.bench_shapes:
                c = compare(build.module.forward, spec, shape, dtype, use_torch_compile=not args.no_compile)
                pct = f"{100 * c.ours.gbps / peak:.0f}%" if peak and c.ours.gbps else "n/a"
                rows.append([name, c.dtype, "x".join(map(str, shape)), fmt_ms(c.ours), fmt_ms(c.eager),
                             fmt_ms(c.compiled), f"{c.speedup_vs_eager:.2f}x", f"{c.ours.gbps:.0f}", pct])
        print()

    header = ["op", "dtype", "shape", "ours ms", "torch ms", "torch.compile ms", "vs torch", "GB/s", "% peak BW"]
    print(f"### Results on {gpu}\n")
    print("| " + " | ".join(header) + " |")
    print("|" + "---|" * len(header))
    for row in rows:
        print("| " + " | ".join(row) + " |")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())

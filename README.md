# KernelAgent

An agent-driven compiler that writes, verifies, profiles and tunes **CUDA kernels for LLM inference**, then swaps them into real models through a `torch.compile` backend.

```
PyTorch model ──► FX graph capture ──► find hot fusable subgraphs
                                              │  op spec + reference
                                              ▼
        ┌──────────── agent loop (per op) ─────────────┐
        │ plan ─► write CUDA ─► compile ─► verify      │
        │   ▲                                 │        │
        │   └── critic ◄── Nsight Compute ◄── bench    │
        └──────────────────────┬───────────────────────┘
                               ▼
        best kernel ─► custom op ─► rewrite pass ─► faster model
```

**Status:** see [MILESTONES.md](MILESTONES.md), or run `python scripts/progress.py`.

## How it works (so far: Milestone 1)

Every kernel is a single `.cu` file that exports a `forward` function matching its op's **OpSpec** (`kernelagent/ops/`). The harness then:

| Step | Module | What it does |
|---|---|---|
| Build | `kernelagent/harness/build.py` | Compiles the source with nvcc via `load_inline`. Compiler errors come back as text, not exceptions |
| Verify | `kernelagent/harness/verify.py` | Runs every shape × dtype (fp32, fp16, bf16) against an fp32 PyTorch reference. Catches wrong values, NaNs, bad shapes and in-place writes to inputs |
| Bench | `kernelagent/harness/bench.py` | Times with CUDA events, flushing L2 before each call. Reports GB/s and % of peak bandwidth vs PyTorch eager and `torch.compile` |

The hand-written baselines are in `kernels/baselines/`:
- **softmax.cu:** online softmax (single-pass max+sum), warp-shuffle and shared-memory block reductions, fp32 math.
- **rmsnorm.cu:** 128-bit vectorized loads and stores with a scalar fallback, fp32 accumulation.

## Running

**Locally (no GPU needed):** CPU tests only. GPU tests are skipped.
```bash
pip install -e ".[dev]"
python -m pytest
python scripts/progress.py
```

**On Google Colab (GPU):**
1. Push this repo to GitHub.
2. Open `notebooks/colab_runner.ipynb` in Colab and pick an **L4** or **A100** runtime.
3. Set `REPO` in the notebook, then run all cells. This runs the GPU tests and the baseline benchmark table.

Or run the commands directly on any Linux machine with a GPU and the CUDA toolkit:
```bash
python -m pytest -v
python scripts/run_baselines.py
```

## Results

_Baseline benchmark table goes here after the first Colab run._

## Layout

```
kernelagent/ops/       OpSpecs: signature, reference, inputs, tolerances
kernelagent/harness/   build / verify / bench
kernels/baselines/     hand-written CUDA kernels
scripts/               progress.py, run_baselines.py
notebooks/             Colab runner
tests/                 CPU tests + GPU tests (marked `gpu`)
```

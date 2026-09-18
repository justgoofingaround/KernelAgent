# KernelAgent Milestones

Progress tracker. Tick a task (`- [x]`) when it's done, then run:

```
python scripts/progress.py
```

---

## M1 Harness + hand-written baselines
Goal: softmax and RMSNorm CUDA kernels compile, pass fuzzed tests and are benchmarked vs PyTorch on Colab.

- [x] Project skeleton (pyproject, package layout, .gitignore, README)
- [x] Progress tracker script (`scripts/progress.py`)
- [x] OpSpec contract (`kernelagent/ops/base.py`) + softmax and RMSNorm specs
- [ ] Build harness: compile CUDA source → module, errors returned as text (`harness/build.py`)
- [x] Verify harness: fuzzed shapes × dtypes vs PyTorch reference (`harness/verify.py`)
- [x] Bench harness: CUDA events, L2 flush, GB/s, vs eager and torch.compile (`harness/bench.py`)
- [ ] Hand-written CUDA softmax (warp-shuffle reductions)
- [ ] Hand-written CUDA RMSNorm (vectorized 128-bit loads)
- [ ] Tests: CPU spec tests + GPU harness tests (broken-kernel and off-by-one cases)
- [ ] Colab runner notebook + `scripts/run_baselines.py` results table
- [ ] Run on Colab GPU: all tests green, benchmark table recorded in README

## M2 Profiling + safety tools
Goal: the agent can "see" why a kernel is slow or unsafe.

- [ ] `ncu` wrapper: profile a single kernel launch from a script
- [ ] Parse ncu metrics to JSON (DRAM BW %, SM occupancy, achieved FLOPs, warp stall reasons, bank conflicts)
- [ ] Roofline classification: memory-bound vs compute-bound vs latency-bound
- [ ] compute-sanitizer wrapper (memcheck, racecheck, initcheck) with parsed error output
- [ ] Map sanitizer and ncu findings back to source lines (`-lineinfo`)
- [ ] Human-readable "profile summary" text (future agent prompt input)
- [ ] Tests: planted race and out-of-bounds kernels are caught

## M3 Agent loop v1 (OpenRouter)
Goal: an LLM writes a working kernel that beats PyTorch, with no human edits.

- [ ] OpenRouter client (model selection, retries, token/cost accounting)
- [ ] Prompt templates: op spec + signature contract + constraints
- [ ] Code extraction and sanity checks on LLM output
- [ ] Loop: generate → compile → verify → bench → feedback, with an iteration budget
- [ ] Attempt log (JSONL: code, errors, metrics, cost per attempt)
- [ ] CLI: `kernelagent run --op softmax --model <id> --budget N`
- [ ] Result: agent-written softmax beats PyTorch eager on Colab

## M4 Planner + Critic + search
Goal: profiler-guided optimization, not just retry-on-error.

- [ ] Planner agent: choose strategy (tiling, smem, vectorization, warp layout)
- [ ] Critic agent: reads ncu + sanitizer summaries → diagnosis + next change
- [ ] Top-K beam search over candidate kernels
- [ ] Launch-parameter autotuning (block size, rows per block)
- [ ] SQLite kernel cache keyed by (op, shape bucket, dtype, GPU arch)
- [ ] Parallel compile/verify workers
- [ ] Ablation hook: toggle planner / critic / profiler feedback

## M5 LLM inference ops
Goal: fused kernels that matter for real LLM serving.

- [ ] OpSpec + reference: fused RMSNorm + FP8 quant (needs L4, sm_89)
- [ ] OpSpec + reference: SwiGLU (SiLU(gate) * up)
- [ ] OpSpec + reference: RoPE + KV-cache write
- [ ] OpSpec + reference: paged-attention decode
- [ ] Agent-generated kernels for all four
- [ ] Benchmarks vs FlashInfer / PyTorch

## M6 Tensor Core kernels
Goal: compute-bound kernels on Tensor Cores.

- [ ] Hand-written WMMA GEMM baseline
- [ ] CUTLASS GEMM integration in the build harness
- [ ] Agent-generated GEMM (fp16/bf16) vs cuBLAS
- [ ] MoE grouped GEMM spec + agent-generated kernel
- [ ] Benchmarks: TFLOPs and % of peak

## M7 Compiler integration
Goal: generated kernels run inside a real LLM.

- [ ] Register kernels as `torch.library` custom ops
- [ ] `torch.compile` custom backend (FX graph capture)
- [ ] Pattern matcher: RMSNorm, SwiGLU, RoPE, attention subgraphs
- [ ] Rewrite pass: swap subgraphs for generated kernels, with fallback
- [ ] End-to-end Llama-3.2-1B: outputs match baseline
- [ ] End-to-end tokens/s vs eager and torch.compile

## M8 Evaluation + polish
Goal: portfolio-ready results.

- [ ] KernelBench subset evaluation (correct %, speedup)
- [ ] Ablations: no profiler feedback / no planner / single-shot
- [ ] Speedup-vs-iterations and cost charts
- [ ] README with flowchart, results tables and reproduction steps
- [ ] Blog post / write-up
- [ ] Upstream PR (FlashInfer / CUTLASS / PyTorch)

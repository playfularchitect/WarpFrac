# WarpFrac — Series 1

_Original public benchmark — November 7, 2025_

Series 1 is a CUDA/cuBLASLt benchmark and reference implementation performing **bit-for-bit exact `INT8 × INT8 → INT32` matrix multiplication** on GPU, with reproducible proof of correctness and transparent ops accounting.

This is not a theoretical claim: the benchmark ships with machine-generated artifacts (CSV/MD/JSON reports), a reviewer trust pack (Nsight reports and clock logs), and GMP-based exactness witnesses.

> **Note:** the macro throughput figures originally published for this series were inflated 4× by a panel-accounting error. See [Errata](#errata--panel-accounting) below. Series 2 supersedes these numbers with standard accounting.

## Reproduce it

**[► Run the benchmark on Google Colab](https://colab.research.google.com/drive/1D-KihKFEz6qmU7R-mvba7VeievKudvQ8?usp=sharing)** (A100 runtime)

Or locally with CUDA 12.x on an A100 (SM80) — a pinned environment is provided in [`src/Dockerfile`](src/Dockerfile):

```bash
cd src
nvcc -O3 -std=c++17 -arch=sm_80 fx_int8_kpanel_tiled_swarm_v1.cu -lcublasLt -lcublas -lgmp -o fx
./fx --m 5120 --n 5120 --k 5120 --streams 32 --graphNodes 64 --batchPerNode 4 \
     --tileK 1280 --epochs 1 --warmup 3 --tryAlgos 16 --workspaceMB 1024 --validate 1
```

## Contents

```
A100_Benchmark_2025-11-06.ipynb        Colab notebook (full run)
src/
├── fx_int8_kpanel_tiled_swarm_v1.cu   Macro benchmark: K-panel tiled swarm (5120³)
├── pb4_gmp_vs_gpu_micro.cu            Micro head-to-head vs. GMP (128³/192³/256³)
├── pb6_micro_reps_graph.cu            Micro amortized throughput (CUDA Graph replays)
├── pb_probe_env.cu                    Environment probe
└── Dockerfile                         CUDA 12.2 build environment
scripts/
├── run_public_bench.py                One-click driver: builds, runs, validates,
│                                      and generates the report artifacts
└── make_reviewer_trust_pack.py        Generates Nsight/clocks/provenance trust pack
results/
├── public_bench_report.md / .csv      Generated benchmark report
├── public_bench_meta.json             Environment and run metadata
├── ops_accounting.md                  Independent recount of every reported rate
├── throughput_conventions.md          MAC/ops definitions, overflow guardrails
├── verification_checklist.txt         Nsight + clock-sampling commands for reviewers
├── benchmark_output.log               Raw run output
└── reviewer_trust_pack_output.log     Raw trust-pack output
```

## Where GMP fits, and why it matters

The GPU computes `INT8 × INT8 → INT32` accumulate (exact integers) via cuBLASLt. **GMP is not the performance target — it is the ground-truth validator.** The benchmark recomputes the same GEMM on CPU with GMP integers and checks bit-for-bit equality of the INT32 output, and additionally prints dyadic rationals (real-valued view = `int / 2^(fracA+fracB)`, here `2⁻⁸`). This keeps the exactness claim auditable rather than asserted.

## Results (A100-SXM4-40GB)

### Macro throughput, M = N = K = 5120 — as measured by the panelized harness

| Path | Avg per GEMM | Rate (panel-counted) |
|---|---:|---:|
| INT8 → INT32 (exact) | 0.894 ms | 300.3 T-ops/s |
| FP16 → FP32 | 1.136 ms | 236.3 TFLOPS |
| TF32 → FP32 | 2.301 ms | 116.7 TFLOPS |

All three paths ran in the same harness with the same ops formula, so the **relative** comparison stands: the integer-exact path was ~27% faster than FP16 and ~2.6× TF32 in this setup — with provable integer correctness. For the **absolute** physical rates, see the [errata](#errata--panel-accounting): the INT8 macro figure normalizes to ~75 T-ops/s (~37.5 T-MAC/s).

### Micro accuracy vs. exact (realistic, non-dyadic scales)

Quantization scales for the FP comparison: `sA = 1/17`, `sB = 1/29` — deliberately *not* powers of two, like real dequantization.

Micro (256³), full matrix:

| Path | max err | RMS | rel-L2 |
|---|---:|---:|---:|
| FP16 → FP32 | 0.220184 | 0.049544 | 2.821×10⁻⁴ |
| TF32 → FP32 | 0.220123 | 0.049540 | 2.821×10⁻⁴ |

Macro (5120³), sampled N = 4096: FP max err 0.725952, RMS 0.225998, rel-L2 2.958×10⁻⁴.

With dyadic scales (e.g. 1/16 × 1/16) FP can *appear* exact because values land on power-of-two fractions. With realistic non-dyadic scales, FP shows small but measurable rounding error, while the INT32 path remains the ground truth.

### Micro throughput (amortized, CUDA Graph replays)

- 256³: 2.026 T-ops/s, avg per-GEMM 0.01656 ms

The CPU-vs-GPU "speedup" lines in the notebook compare against the CPU GMP reference to demonstrate correctness at scale; the performance headline is the GPU rate above.

## Ops accounting

Throughput is computed from measured time using the canonical GEMM count `ops_per_gemm = 2 × M × N × K`. INT8→INT32 is reported as T-ops/s and FP baselines as TFLOPS using the same formula, so the comparison is apples-to-apples. [`results/ops_accounting.md`](results/ops_accounting.md) independently recomputes every reported rate from the recorded times (0.0000% deviation), and [`results/throughput_conventions.md`](results/throughput_conventions.md) documents the INT32 overflow guardrails for every benchmark shape.

## Reviewer trust pack

Generated by [`scripts/make_reviewer_trust_pack.py`](scripts/make_reviewer_trust_pack.py):

- Nsight Compute capture files (`.ncu-rep`) for macro and micro runs
- Clock/power log (`nvidia-smi` sampled during a macro run)
- Provenance JSON — driver/runtime versions, compile flags, binary and source checksums
- A compact summary appended to the main report

[`results/verification_checklist.txt`](results/verification_checklist.txt) lists the exact `ncu` and `nvidia-smi` commands for independent verification.

## Parameters you can vary

- **Scales** — non-dyadic (e.g. 1/17, 1/29) to expose FP rounding; dyadic for the zero-error edge case
- **Shapes** — micro (256³) vs. macro (5120³), or add your own (overflow bounds documented per shape)
- **Timing** — warmup/reps; CUDA Graphs are used for stable amortized timing
- **Validation** — GMP witness sampling and validator blocks can be toggled

## Errata — panel accounting

The originally reported macro throughput used `ops = 2·M·N·K·number_of_gemms` with K = 5120, but the macro harness splits each GEMM into 4 K-panels of size `tileK = 1280`, and `number_of_gemms` counts panels. Each K-panel was therefore credited with the ops of a full 5120³ GEMM, **inflating the macro throughput by 4×**.

- Published headline: 5120³ at 300.26 T-ops/s (150.13 T-MAC/s) — a *logical, panel-counted* rate
- Physical rate, normalized to full 5120³ GEMMs: **~75 T-ops/s (~37.5 T-MAC/s)**

All INT8-vs-FP baselines shared the same panelized harness and ops formula, so relative comparisons remain valid; what changes is the interpretation against the A100's theoretical INT8 peak (312 T-MAC/s, 624 T-ops/s). Series 2 reports physical rates with standard accounting and labels any logical counting explicitly.

## License & contact

Apache-2.0. Released for expert validation and to explore applications — **ewesley541@gmail.com**.

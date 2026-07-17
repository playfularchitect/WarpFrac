# WarpFrac — Series 2 (Fastlane INT8 Path)

_Current-generation benchmark, standard accounting — November 16, 2025_

Series 2 is the "clean accounting" iteration of WarpFrac. It does three things:

1. Measures **`INT8 × INT8 → INT32`** GEMM throughput on an NVIDIA A100 via cuBLASLt.
2. Verifies the math is **bit-for-bit exact** against a CPU reference.
3. Records the **GPU / CUDA / cuBLAS environment** and binary provenance so the results are reproducible.

Unlike Series 1, all headline numbers here are **physical** rates using the standard `MACs = M·N·K` formula — no panel counting, no logical multipliers (the optional "effective k-lanes" view is computed separately and explicitly labeled).

## Headline result (A100-SXM4-40GB, M = N = K = 5120)

| Metric | Value |
|---|---:|
| RAW throughput | **209.82 T-MAC/s** (419.65 T-ops/s) |
| Per-GEMM latency | 0.640 ms |
| Stability (3 runs) | ±111.9 G-MAC/s (±0.05%) |
| Exactness | max \|CPU − GPU\| = 0 |

Full evidence in [`results/`](results/): per-run stats, Nsight Compute capture, clock/power log sampled during the run, and provenance JSON with the binary's SHA-256.

## Reproduce it

**[► Run on Google Colab](https://colab.research.google.com/drive/1L9GShHz_Hi0XmMPA7W52jtkAy9HEi9Yb?usp=sharing)** (A100 runtime)

Or locally with CUDA 12.x on an A100 (SM80) — pinned environment in [`src/Dockerfile`](src/Dockerfile):

```bash
cd src

# 1. Throughput benchmark
nvcc -O3 -std=c++17 -arch=sm_80 fastlane_best_rr_allinone.cu -lcublasLt -lcublas -o fastlane_bench
./fastlane_bench

# 2. Exactness witness (CPU vs GPU)
nvcc -O3 -std=c++17 -arch=sm_80 fastlane_exact_witness.cu -lcublas -o witness
./witness

# 3. Environment report
nvcc -O3 -std=c++17 -arch=sm_80 fastlane_env_report.cu -lcublas -o env_report
./env_report
```

`src/run_short_fastlane.sh` wraps the quick benchmark repro (~1–2 minutes).

## Contents

```
WarpFrac_Series_2_2025-11-16.ipynb     Colab notebook (full run)
src/
├── fastlane_best_rr_allinone.cu       INT8 throughput benchmark (cuBLASLt)
├── fastlane_exact_witness.cu          Exactness tests: CPU vs GPU
├── fastlane_env_report.cu             Environment dump (GPU, CUDA, cuBLAS versions)
├── run_short_fastlane.sh              Quick build-and-run script
└── Dockerfile                         CUDA 12.2 build environment
scripts/
└── run_fastlane_bench.py              One-click driver (builds, runs, generates trust pack)
results/
├── fastlane_trust_pack.md             Trust pack summary
├── fastlane_macro_stats.json          Per-run throughput statistics (3× runs)
├── fastlane_macro.ncu-rep             Nsight Compute capture (open in Nsight UI)
├── fastlane_clocks_power_log.csv      nvidia-smi clock/power samples during the run
├── fastlane_provenance.json           Compiler versions, flags, binary SHA-256
├── fastlane_provenance_min.json       Compact provenance view
└── fastlane_output.log                Raw run output
```

## What the benchmark measures

[`fastlane_best_rr_allinone.cu`](src/fastlane_best_rr_allinone.cu) runs a 5120×5120×5120 `INT8 × INT8 → INT32` GEMM (row-major A, B, C) through cuBLASLt. It requests several candidate algorithms from the cuBLASLt heuristics, times them, picks the fastest, then runs it repeatedly (200 replays by default) and reports:

- **`RAW G-MAC/s`** — physical multiply-accumulates per second (`MACs = M·N·K`)
- **`RAW GFLOP/s`** — `2 × G-MAC/s`, the standard GEMM convention counting multiply and add separately
- **`EFFECTIVE (k=9)`** — an optional *logical-ops* view for WarpFrac's encoding scheme, treating each MAC as 9 logical lanes. This is explicitly labeled and never mixed into the RAW numbers.

## What the exactness witness checks

[`fastlane_exact_witness.cu`](src/fastlane_exact_witness.cu) runs `INT8 × INT8 → INT32` GEMMs via `cublasGemmEx` and compares against CPU references:

1. **Micro test (256³), random data** — A and B are random INT8. The CPU reference accumulates in `int64` and stores `int32` (safe at this size, per the documented overflow bounds); the GPU uses the Tensor Core path. Pass criterion: `max |CPU − GPU| = 0`. A few entries are printed as explicit `C(i,j) = Σ A(i,k)·B(k,j)` equations with CPU and GPU values side by side.
2. **Macro test (5120³), known answer** — A and B are all-ones, so every element of C must equal exactly 5120. The GPU result is checked element-wise.
3. **Edge-case equations** — tiny 1×1×K examples with extreme INT8 values, printed as CPU-only demonstrations (not part of pass/fail).

## Environment report

[`fastlane_env_report.cu`](src/fastlane_env_report.cu) prints the CUDA driver and runtime versions, GPU name and compute capability, memory size, SM count, clocks, and cuBLAS version — a copy-pasteable block documenting exactly what stack produced the numbers.

## Relationship to Series 1

Series 1 ([`../series-1/`](../series-1/)) is the original benchmark, including its panel-accounting errata. Series 2 was built in response: same exactness guarantees, standard physical accounting throughout.

## License & contact

Apache-2.0. Questions and independent validation welcome — **ewesley541@gmail.com**.

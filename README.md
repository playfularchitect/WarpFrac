# WarpFrac

**Bit-for-bit exact integer matrix multiplication at Tensor Core speeds.**

![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)
![Platform](https://img.shields.io/badge/GPU-NVIDIA_A100_(SM80)-76B900.svg)
![Precision](https://img.shields.io/badge/Arithmetic-INT8%C3%97INT8%E2%86%92INT32_exact-informational.svg)

WarpFrac is a research project exploring **exact arithmetic on GPU hardware**. 

Floating-point GEMM is fast but approximate; arbitrary-precision CPU libraries (GMP) are exact but orders of magnitude too slow for large-scale compute. 

WarpFrac targets the gap between the two: NVIDIA Tensor Cores accelerate `INT8 × INT8 → INT32` multiply-accumulate, an operation that is **exact by construction** — no rounding, no error accumulation. 

Exact integer kernels managing the numerators and denominators of rational numbers are the foundation for an exact-rational compute pipeline.

Every performance claim in this repository ships with the artifacts needed to check it: source code, raw benchmark logs, Nsight Compute captures, clock/power logs, provenance JSON (compiler versions, binary checksums), and CPU/GMP exactness witnesses.

---

## Measured Results

### Series 2 — current (A100-SXM4-40GB, 5120×5120×5120 GEMM, cuBLASLt)

| Metric | Value | Evidence |
|---|---|---|
| Throughput (physical) | **209.8 T-MAC/s** (419.6 T-ops/s) | [`fastlane_macro_stats.json`](series-2/results/fastlane_macro_stats.json) |
| Per-GEMM latency | 0.640 ms | 3 runs, σ ≈ 0.0005 ms |
| Run-to-run variation | ±0.05% | min 209.69, max 209.97 T-MAC/s |
| Fraction of A100 dense INT8 peak (312 T-MAC/s) | ~67% | — |
| Exactness vs. CPU reference | **max \|diff\| = 0** | [`fastlane_exact_witness.cu`](series-2/src/fastlane_exact_witness.cu) |

Throughput uses the standard accounting `MACs = M·N·K`, `ops = 2·M·N·K`, recomputed from measured device time. No logical multipliers are included in the headline numbers.

### Series 1 — original benchmark (A100-SXM4-40GB)

The original K-panel swarm harness demonstrated the exact INT8 path outperforming FP16 and TF32 baselines under identical accounting, with GMP witnesses confirming bit-for-bit correctness. A panel-counting error inflated its absolute throughput figures by 4×; this is documented in full in the [Series 1 errata](series-1/README.md#errata--panel-accounting) rather than quietly revised. Relative INT8-vs-FP comparisons are unaffected (all paths shared the same harness).

---

## Why exactness matters

For iterative, chaotic, or long-running numerical systems, floating-point error accumulates and it becomes impossible to distinguish a real result from a numerical artifact. With an exact integer pipeline, any pattern in the output is a property of the model, not of the arithmetic. The benchmarks here include FP16/TF32 baselines run with realistic non-dyadic quantization scales, showing measurable floating-point rounding error (rel-L2 ≈ 3×10⁻⁴) where the INT32 path is exact.

---

## Repository layout

```
series-1/                      Original benchmark (Nov 2025)
├── README.md                  Methodology, results, and errata
├── A100_Benchmark_2025-11-06.ipynb
├── src/                       CUDA sources + Dockerfile
├── scripts/                   One-click benchmark & trust-pack drivers (Colab)
└── results/                   Reports (MD/CSV/JSON), raw logs, verification docs

series-2/                      Current benchmark, standard accounting (Nov 2025)
├── README.md                  Methodology and build instructions
├── WarpFrac_Series_2_2025-11-16.ipynb
├── src/                       CUDA sources (bench, exactness witness, env report)
├── scripts/                   One-click driver (Colab)
└── results/                   Trust pack: stats, Nsight capture, clocks/power log,
                               provenance JSON, raw output
```

## Reproducing the results

**Google Colab (one click, A100 runtime):**
- [Series 1 benchmark](https://colab.research.google.com/drive/1D-KihKFEz6qmU7R-mvba7VeievKudvQ8?usp=sharing)
- [Series 2 benchmark](https://colab.research.google.com/drive/1L9GShHz_Hi0XmMPA7W52jtkAy9HEi9Yb?usp=sharing)

**Locally (CUDA 12.x, A100 / SM80):**

```bash
cd series-2/src
nvcc -O3 -std=c++17 -arch=sm_80 fastlane_best_rr_allinone.cu -lcublasLt -lcublas -o fastlane_bench
./fastlane_bench                       # throughput
nvcc -O3 -std=c++17 -arch=sm_80 fastlane_exact_witness.cu -lcublas -o witness
./witness                              # bit-for-bit exactness vs CPU
```

Dockerfiles for a pinned environment are provided in each series' `src/` directory. See the per-series READMEs for the full verification workflow (Nsight metrics, clock sampling, GMP witnesses).

## Verification methodology

- **Exactness witnesses** — the same GEMM is computed on CPU (int64 accumulate, and GMP integers in Series 1) and compared element-wise against the GPU result; the pass criterion is `max |CPU − GPU| = 0`.
- **Transparent ops accounting** — throughput is always recomputed from measured device time via `ops = 2·M·N·K`; conventions are documented in [`series-1/results/throughput_conventions.md`](series-1/results/throughput_conventions.md) and [`series-1/results/ops_accounting.md`](series-1/results/ops_accounting.md).
- **Trust packs** — Nsight Compute captures, `nvidia-smi` clock/power logs sampled during runs, and provenance JSON with compiler versions and binary SHA-256 checksums.
- **Device-side timing** — CUDA events (and CUDA Graph replays for amortized micro benchmarks), not host wall-clock.

## Roadmap

- **Series 3 (AVX2)** — a CPU backend using AVX2 integer SIMD, in development.
- Rational (numerator/denominator) pipeline built on the exact integer kernels.

## License & contact

Licensed under [Apache 2.0](LICENSE).

Questions, review, or collaboration: **ewesley541@gmail.com**. Independent validation is welcome — everything needed to replicate or falsify the claims above is in this repository.

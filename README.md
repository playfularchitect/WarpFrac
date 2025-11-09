# WarpFrac
Fast Provably Exact Math

November 7th, 2025

![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)

WarpFrac is a CUDA/cuBLASLt benchmark and reference that performs bit-for-bit exact INT8 × INT8 → INT32 matrix multiplication on GPU, with reproducible proof of correctness and transparent ops accounting. On A100 it sustains ~300 T-ops/s on a 5120³ macro test while remaining integer-exact.

This is not a theoretical claim — it’s a replicable benchmark with artifacts (CSV/MD/JSON), a reviewer Trust Pack (Nsight reports + clocks log), and GMP-based witnesses.

##  Replicate it Yourself (1-Click)


**[► Click Here to Run the Benchmark on Google Colab](https://colab.research.google.com/drive/1D-KihKFEz6qmU7R-mvba7VeievKudvQ8?usp=sharing)**


---

Where GMP is used & why it matters

I do INT8×INT8 → INT32 accumulate (exact integers) on the GPU (via cuBLASLt).

GMP is not our performance target, it’s the ground truth validator. 
In the public bench we compute the same GEMM on CPU with GMP integers and check bit-for-bit equality of the INT32 output, plus print dyadic rationals (real view = int/2^(fracA+fracB), here 2^-8). 
This keeps the “exactness” claim auditable.
---
Headline Results (A100-SXM4-40GB)
Macro Throughput (M=N=K=5120)
Path	Avg per GEMM	Rate
INT8→INT32 (exact)	0.894 ms	300.263 T-ops/s
FP16→FP32	1.135923 ms	236.315 TFLOPS
TF32→FP32	2.301030 ms	116.659 TFLOPS
---

Takeaway: The integer-exact path is ~27% faster than FP16 here and ~2.6× TF32 — with provable integer correctness!

---
Micro Accuracy vs Exact (Realistic, Non-Dyadic Scales)

Scales used in the FP comparison: sA=1/17, sB=1/29 (i.e., not powers of two — like real dequant).

Micro (256³), full matrix:

FP16→FP32: max=0.220184, RMS=0.0495439, rel-L2 = 2.82102×10⁻⁴

TF32→FP32: max=0.220123, RMS=0.0495398, rel-L2 = 2.82079×10⁻⁴
---
Macro (5120³), sampled N=4096:

FP (same scales): max=0.725952, RMS=0.225998, rel-L2 = 2.95844×10⁻⁴

With dyadic scales (e.g., 1/16 × 1/16), FP can appear exact in this setup because values land on power-of-two fractions. With realistic non-dyadic scales, FP shows small but measurable rounding error; the INT32 path remains the ground truth.
---
Micro Throughput (Amortized, CUDA Graph Replays)

256³: 2.026233816 T-ops/s, avg per-GEMM 0.016560 ms

CPU vs GPU (context only)

The “speedup” lines in the notebook compare GPU to a CPU GMP reference to demonstrate correctness and scale.

The real performance headline is the GPU number above.
---
Ops Accounting (consistent across FP & INT8)

Throughput is computed from measured time using the canonical GEMM count:
ops_per_gemm = 2 × M × N × K.

INT8→INT32 is reported as T-ops/s.

FP baselines are reported as TFLOPS.
Same formula → apples-to-apples comparison.

We also publish an “Ops Accounting” appendix and re-compute rates from time to avoid rounding artifacts.
---
 What’s in the Trust Pack

Nsight Compute capture files (.ncu-rep) for macro and micro runs

Clocks/Power log (nvidia-smi sampled during a macro run)

Provenance JSON (driver/runtime, compile flags, binary/source checksums)

A compact summary appended to the main report
---
 What You Can Tweak

Scales: choose non-dyadic (e.g., 1/17, 1/29) to expose FP rounding; choose dyadic for a zero-error edge case.

Shapes: switch between micro (256³) and macro (5120³), or add your own.

Timing: adjust warmup/reps; CUDA Graphs are used for stable amortized timing.

Validation: enable/disable validator blocks and GMP witness sampling.
---
TL;DR 

WarpFrac — integer-exact INT8 matmul at GPU speed.
On A100 (5120³), INT8→INT32 exact achieves ~300 T-ops/s, outpacing FP16 at ~236 TFLOPS and TF32 at ~117 TFLOPS.
With realistic, non-dyadic dequant scales, FP paths show small but measurable error (rel-L2 ≈ 3e-4), while INT8→INT32 remains a provable ground truth.
Ops are computed from time via the standard 2×M×N×K formula and reported consistently for FP and INT8.
---



Contact

I’m releasing this for expert validation and to explore applications.
Email: ewesley541@gmail.com

License: Apache-2.0

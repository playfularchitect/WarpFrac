# WarpFrac
Fast Provably Exact Math

November 7th, 2025

![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)

This repository contains the source and benchmarks for **WarpFrac**, a CUDA kernel series that performs bit-for-bit exact matrix multiplication at speeds over **21,000x faster than GMP** (the arbitrary-precision gold standard).


This is not a theoretical claim. This is a replicable benchmark.

I am releasing this for expert validation and to find applications for this new capability and my problem solving skills. 

##  Replicate it Yourself (1-Click)

You don't have to trust me. Run the benchmark yourself on a Google Colab instance. With one click the notebook will compile the kernel, run the benchmarks, and validate the bit-for-bit correctness of every calculation against a CPU-based validator.

**[► Click Here to Run the Benchmark on Google Colab](https://colab.research.google.com/drive/1D-KihKFEz6qmU7R-mvba7VeievKudvQ8?usp=sharing)**


---

Where GMP is used & why it matters

I do INT8×INT8 → INT32 accumulate (exact integers) on the GPU (via cuBLASLt).

GMP is not our performance target, it’s the ground truth validator. 
In the public bench we compute the same GEMM on CPU with GMP integers and check bit-for-bit equality of the INT32 output, plus print dyadic rationals (real view = int/2^(fracA+fracB), here 2^-8). 
This keeps the “exactness” claim auditable.

What the speedups are relative to

The “speedup” lines compare the GPU to a CPU GMP reference (just to show correctness + scale). The real throughput headline is the GPU number:

Macro (5120³, A100-SXM4-40GB): ~300.26 T-ops/s (ops = 2×M×N×K), per-gemm 0.894 ms.

Micro amortized (256³, CUDA Graph replay): ~2.026 T-ops/s, avg per-gemm 0.01656 ms.

Micro H2H (128³/192³/256³) all PASS exactness vs GMP.

---



I can make it even faster. Much faster. 


Contact: ewesley541@gmail.com

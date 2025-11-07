# WarpFrac
Fast Provably Exact Math

November 7th, 2025

![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)

This repository contains the source and benchmarks for **WarpFrac**, a CUDA kernel that performs bit-for-bit exact matrix multiplication at speeds over **21,000x faster than GMP** (the arbitrary-precision gold standard).

This is not a theoretical claim. This is a replicable benchmark.

I am releasing this for expert validation and to find applications for this new capability.

##  Replicate it Yourself (1-Click)

You don't have to trust me. Run the benchmark yourself on a Google Colab instance. With one click the notebook will compile the kernel, run the benchmarks, and validate the bit-for-bit correctness of every calculation against a CPU-based validator.

**[► Click Here to Run the Benchmark on Google Colab](https://colab.research.google.com/drive/1D-KihKFEz6qmU7R-mvba7VeievKudvQ8?usp=sharing)**


---

## The Proof: Benchmark Logs

The kernel (the GPU code) is designed to perform perfectly exact fraction math.
It  uses tiny, fast whole numbers (int8s) as inputs, but it treats them as fractions. The "$2^{-4}$" scaling means the number 5 is treated as the fraction 5/16.It then multiplies and adds all these fractions, accumulating the result into a much larger integer container (int32) that is guaranteed to be big enough to hold the final answer with no rounding errors.

Contact: ewesley541@gmail.com

The following benchmarks were run on a single **NVIDIA A100-SXM4-40GB GPU**.

I can make it even faster. Much faster. 
I'm just getting started.


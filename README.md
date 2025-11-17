# WarpFrac: An Exact Arithmetic Compute Architecture

_Last Updated: November 16, 2025_

WarpFrac is an R&D architecture designed for **bit-for-bit exact rational arithmetic** at **high-performance (Tensor Core) speeds**.

This project is a new approach to scientific computing that rejects floating-point approximation. It was built to power a true "exact physics engine" for fundamental R&D, like my "Rosetta Stone of Physics" project.

---

## The Core Problem: Speed vs. Exactness

In modern science, we are forced to choose between two bad options:

> **1. Speed (Floating-Point):** We use `FP64` or `FP32` for high-speed GPU computing. This is fast, but it's an **approximation**. Every operation introduces numerical error, and for complex, iterative, or chaotic systems, this error accumulates. We can never be sure if a strange result is a real discovery or a numerical artifact.

> **2. Exactness (Arbitrary Precision):** We use CPU-based libraries like GMP. This is **bit-for-bit exact**, but it is **orders of magnitude too slow** for high-performance matrix compute, making it unusable for large-scale simulations.

WarpFrac is designed to **eliminate this compromise**. It delivers both exactness and speed.

---

## The WarpFrac Architecture

WarpFrac is not a simple math library. It is a new kernel architecture designed to leverage hardware that is *already* exact.

The core of the architecture is built on `INT8xINT8->INT32` matrix multiplication, an operation accelerated by NVIDIA Tensor Cores. Unlike floating-point operations, this integer-based multiply-accumulate is **fundamentally exact** and **does not lose information**.

By building a pipeline that uses these exact integer kernels to manage the numerators and denominators of large rational numbers (fractions), WarpFrac can perform massive computations with **zero numerical error**.

The `Series 2` implementation, for example, has been formally benchmarked and verified:

* **Exact:** It is **bit-for-bit exact** against a CPU reference (`max |diff| = 0`).
* **Fast:** It achieves a stable **~419 T-ops/s** (or ~210 T-MAC/s) on an NVIDIA A100.

This proves that exact compute is not a "slow" novelty; it can be a high-performance, world-class reality.

---

## Project Structure

This repository is organized into distinct "Series" of development. Each series represents a major architectural iteration. Please see the `README.md` inside each folder for specific source code, benchmarks, and other info.

###  Series 1: Logical TOPs (The OG)

This is the original WarpFrac architecture. Its focus was on demonstrating that **fast and exact are not a tradeoff** anybody needs to accept. While its raw TMAC/s are lower, it established the foundation for the entire project.

###  Series 2: High-Performance (The "Trust Pack")

This is the current-generation, high-performance kernel. Its design was focused on further maximizing raw, physical **TMACs/GMACs** by optimizing for `INT8` GEMM performance.

---

## The Goal

The goal of WarpFrac is to **eliminate numerical approximation as a variable in scientific discovery.**

When a simulation or model runs on this architecture, any resulting pattern or anomaly is known, with 100% certainty, to be a feature of the *physics model itself*, not a ghost created by floating-point error.

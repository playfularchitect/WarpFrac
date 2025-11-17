# WarpFrac — Series 2 (Fastlane INT8 Path)

This folder is the **Series 2** WarpFrac setup.

It does three things:

1. Measure **INT8×INT8→INT32** GEMM speed on an A100.
2. Check that the math is **bit-for-bit exact** for a couple of real matrix sizes.
3. Record the **GPU / CUDA / cuBLAS** environment so people can reproduce it.

Series 1 (the original WarpFrac benchmark + errata) lives in `series1/`.  
Series 2 is the “clean, standard accounting” version.


## Files in this folder

- **`fastlane_best_rr_allinone.cu`**  
  Fast INT8 throughput benchmark using cuBLASLt.

- **`fastlane_exact_witness.cu`**  
  Exactness tests: CPU vs GPU for INT8×INT8→INT32 GEMM.

- **`fastlane_env_report.cu`**  
  Environment dump: GPU model, CUDA version, cuBLAS version, etc.


## What the benchmark measures

`fastlane_best_rr_allinone.cu`:

- Shape: **M = N = K = 5120**
- Data type: **INT8×INT8→INT32**
- Library: **cuBLASLt**
- Layout: A, B, C are row-major
- It:
  - Asks cuBLASLt for several candidate algorithms.
  - Times them.
  - Picks the fastest one.
  - Runs it many times and prints:

    - `RAW G-MAC/s` — real multiply-accumulates per second  
    - `RAW GFLOP/s` — just `2 × G-MAC/s`, standard GEMM convention  
    - `EFFECTIVE (k=9)` — optional “logical ops” view, treating each MAC as 9 logical lanes

Important:

- **RAW numbers** are the actual hardware work.
- **EFFECTIVE (k=9)** is just a way of counting logical work in my encoding scheme, not extra physical flops.


## What the exactness witness checks

`fastlane_exact_witness.cu`:

All tests are **INT8×INT8→INT32** GEMMs via `cublasGemmEx`.

It does:

1. **Micro test (256³)**  
   - `M = N = K = 256`  
   - A and B are random INT8.  
   - CPU:
     - Uses an `int64` accumulator.
     - Stores the result as `int32` (safe for this size).  
   - GPU:
     - Uses `cublasGemmEx` INT8×INT8→INT32 (Tensor Core path).  
   - It checks:
     - `max |CPU − GPU| = 0`  
     - Prints a few example entries as explicit equations:
       `C(i,j) = sum A(i,k)*B(k,j)` with CPU and GPU values side by side.

2. **Macro test (5120³)**  
   - `M = N = K = 5120`  
   - A and B are filled with **1**.  
   - Expected result:
     - `C(i,j) = 5120` for every element.  
   - It:
     - Runs `cublasGemmEx` on the GPU.
     - Copies C back.
     - Checks that every entry is exactly 5120, prints the max difference (should be 0).

3. **Edge-case equations (no GPU)**  
   - A few tiny 1×1×K examples with extreme INT8 values.  
   - These are printed as simple CPU formulas (just to show the math).  
   - They do **not** affect pass/fail and don’t call the GPU.


## What the env report shows

`fastlane_env_report.cu`:

- CUDA driver version
- CUDA runtime version
- GPU name and compute capability
- Memory size, SM count, clocks
- cuBLAS version

You can copy this block into your repo docs so people know exactly what stack you ran on.


## How to build and run (A100 / SM80)

These examples assume:

- A100 (SM 8.0)
- CUDA 12.x or similar
- cuBLAS and cuBLASLt installed

### 1. Run the Fastlane benchmark

```bash
nvcc -O3 -std=c++17 \
     -gencode=arch=compute_80,code=sm_80 \
     fastlane_best_rr_allinone.cu \
     -lcublasLt -lcublas \
     -o fastlane_bench

./fastlane_bench

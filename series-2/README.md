# WarpFrac Series 2

Series 2 measures dense signed INT8 matrix multiplication with INT32 accumulation on a single NVIDIA A100-SXM4-40GB. The throughput benchmark times cuBLASLt on device-resident all-ones inputs for `M = N = K = 5120`.

## Recorded results

| Metric | Result |
| --- | ---: |
| Throughput, mean of three runs | **209.82 T-MAC/s** |
| Arithmetic throughput | **419.65 T-ops/s** |
| Mean device time per GEMM | **0.640 ms** |
| Run-to-run throughput range | 209.69–209.97 T-MAC/s |

The benchmark times eligible cuBLASLt algorithms, selects the fastest candidate, and measures repeated GEMMs with CUDA events. The throughput calculation uses `M × N × K` multiply-accumulates per complete matrix product. T-ops/s counts multiplication and addition separately. [CUDA benchmark source](src/fastlane_best_rr_allinone.cu) · [Per-run statistics](results/fastlane_macro_stats.json)

## Correctness checks

The companion executable uses cuBLAS `cublasGemmEx` for these checks.

- **256³ random signed INT8 inputs.** Compares every output element with a CPU reference that accumulates in INT64. The saved run reports a maximum absolute difference of zero.
- **5,120³ all-ones inputs.** Checks every output element against the exact value 5,120. The saved run reports a maximum absolute difference of zero.

The timed benchmark uses cuBLASLt with its own algorithm selection. The two checks show exact results for the inputs exercised by the companion executable. [Correctness source](src/fastlane_exact_witness.cu) · [Saved correctness output](results/fastlane_output.log)

## Reproduce

Select an A100 runtime in [Google Colab](https://colab.research.google.com/drive/1L9GShHz_Hi0XmMPA7W52jtkAy9HEi9Yb?usp=sharing), or open the [saved notebook](WarpFrac_Series_2_2025-11-16.ipynb).

### Run locally

Use an NVIDIA A100 with CUDA 12.x and cuBLAS. From the repository root, compile the benchmark and correctness check.

```bash
cd series-2/src
nvcc -O3 -std=c++17 -arch=sm_80 fastlane_best_rr_allinone.cu -lcublasLt -lcublas -o fastlane_bench
./fastlane_bench
nvcc -O3 -std=c++17 -arch=sm_80 fastlane_exact_witness.cu -lcublas -o witness
./witness
```

Read the benchmark's `RAW G-MAC/s` line for the device throughput. For the correctness check, look for a printed maximum difference of zero in both cases. The [CUDA 12.2 Dockerfile](src/Dockerfile) describes the build environment. [Colab driver](scripts/run_fastlane_bench.py) · [Environment query source](src/fastlane_env_report.cu)

## Evidence

- [Benchmark statistics](results/fastlane_macro_stats.json) and [raw console output](results/fastlane_output.log)
- [Nsight Compute capture](results/fastlane_macro.ncu-rep) and [GPU clock and power log](results/fastlane_clocks_power_log.csv)
- [Toolchain details, compile flags, and binary hash](results/fastlane_provenance.json)

# WarpFrac Series 1

Series 1 is the original November 2025 WarpFrac benchmark. It uses cuBLASLt to run a tiled INT8 matrix multiplication on an NVIDIA A100-SXM4-40GB and includes CPU integer and GMP correctness checks.

## Reported results

The original project notes recorded these average timings in the 5,120³ K-panel harness.

| Path | Reported time per panel |
| --- | ---: |
| INT8 inputs, INT32 accumulation | 0.894 ms |
| FP16 inputs, FP32 accumulation | 1.136 ms |
| TF32 inputs, FP32 accumulation | 2.301 ms |

The original notes also reported about `2.82 × 10⁻⁴` relative L2 error on the 256³ floating-point cases with non-dyadic scales of `1/17` and `1/29`. The micro integer checks passed at 128³, 192³, and 256³, with sampled GMP values recorded in the public report. [Original project notes](https://github.com/playfularchitect/WarpFrac/blob/2cc798afd1def221e9cc04ab5932105c3a7d499a/Series%201/Benchmark%20Code%20And%20Output/readme.md) · [INT8 run log](results/benchmark_output.log) · [Microbenchmarks](results/public_bench_report.md)

## Throughput correction

The original macro report counted four K-panels per complete matrix product and credited each panel with the work of a full GEMM. Dividing the original figure of about 300 T-ops/s by four gives **about 75 T-ops/s** for the complete matrix products. In the saved run, 65,536 panels correspond to 16,384 full matrix products over 58.616 seconds. The report and logs preserve the original panel-counted numbers. The current throughput figure is in [Series 2](../series-2/). [Archived timing and panel counts](results/benchmark_output.log) · [Throughput conventions](results/throughput_conventions.md)

## Reproduce

Select an A100 runtime in [Google Colab](https://colab.research.google.com/drive/1D-KihKFEz6qmU7R-mvba7VeievKudvQ8?usp=sharing), or open the [saved notebook](A100_Benchmark_2025-11-06.ipynb).

### Run locally

Use an NVIDIA A100 with CUDA 12.x, cuBLAS, and the GMP development libraries. From the repository root, build and run the archived harness.

```bash
cd series-1/src
nvcc -O3 -std=c++17 -arch=sm_80 fx_int8_kpanel_tiled_swarm_v1.cu -lcublasLt -lcublas -lgmp -o fx
./fx --m 5120 --n 5120 --k 5120 --streams 32 --graphNodes 64 --batchPerNode 4 \
     --tileK 1280 --epochs 1 --warmup 3 --tryAlgos 16 --workspaceMB 1024 --validate 1
```

The [CUDA 12.2 Dockerfile](src/Dockerfile) lists the build environment. The [Colab driver](scripts/run_public_bench.py) includes the GMP microbenchmark and CUDA Graph timing. Use the [reviewer checklist](results/verification_checklist.txt) for profiling commands and environment checks.

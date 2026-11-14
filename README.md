# WarpFrac

WarpFrac is a CUDA research project for high-throughput, exact integer matrix multiplication. It benchmarks signed INT8 inputs and INT32 accumulation on NVIDIA Tensor Cores. When the dot products fit in INT32, the matrix product preserves the exact integer values. That makes the operation a building block for computing with rational numerators and denominators.

## Benchmark results

On a single **NVIDIA A100-SXM4-40GB**, Series 2 averaged **209.82 trillion multiply-accumulates per second** for a 5,120 × 5,120 × 5,120 GEMM using cuBLASLt. The mean device time was **0.640 ms per GEMM** over three runs. Each multiply-accumulate counts as two arithmetic operations, for **419.65 T-ops/s**. [Recorded timing data](series-2/results/fastlane_macro_stats.json)

The companion correctness program reported zero difference against a CPU integer reference on a random 256³ case and against the known result on a 5,120³ all-ones case. It uses cuBLAS `cublasGemmEx`, while the throughput measurement uses cuBLASLt. [Correctness source](series-2/src/fastlane_exact_witness.cu) · [Saved output](series-2/results/fastlane_output.log)

## Run the benchmarks

| Benchmark | What it covers | Reproduce |
| --- | --- | --- |
| [Series 2](series-2/) | Dense INT8 GEMM throughput, integer correctness checks, and profiler captures | [Google Colab on an A100](https://colab.research.google.com/drive/1L9GShHz_Hi0XmMPA7W52jtkAy9HEi9Yb?usp=sharing) · [Local CUDA build](series-2/README.md#run-locally) |
| [Series 1](series-1/) | Original K-panel benchmark, GMP checks, and floating-point comparison | [Google Colab on an A100](https://colab.research.google.com/drive/1D-KihKFEz6qmU7R-mvba7VeievKudvQ8?usp=sharing) · [Local CUDA build](series-1/README.md#run-locally) · [Throughput correction](series-1/README.md#throughput-correction) |

Each series includes CUDA source, a Colab notebook, and recorded results. The Series 2 results include the Nsight Compute capture, clock and power samples, and toolchain and binary provenance. [Series 2 evidence](series-2/README.md#evidence)

Licensed under [Apache 2.0](LICENSE). Project contact [Evan Wesley](mailto:ewesley541@gmail.com).

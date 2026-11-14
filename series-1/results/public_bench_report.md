# INT8→INT32 Exact GEMM — Public Benchmark

_Generated: 2025-11-07T10:02:29+00:00_

## Headline (Macro Arena)
- **Throughput:** **300.13 T-ops/s**  
- **Config:** M=N=K=5120, streams=32, nodes=64, tileK=1280 (panels=4), epochs=2  
- **cuBLASLt algo:** index=0  workspace=1024 MB  
- **Exactness:** micro validator **PASS** vs CPU int32; GMP sampled witnesses printed.

## Micro Head-to-Head vs GMP (Exact Integers)
| Shape | Shift | GPU ms | GPU T-ops/s | GMP ms | GMP G-ops/s | Speedup | Exact |
|---:|---:|---:|---:|---:|---:|---:|:--:|
| 128³ | 8 | 4.944 | 0.000848362 | 42.306 | 0.099142000 | 8.56× | PASS |
| 192³ | 8 | 5.015 | 0.002822687 | 140.976 | 0.100412000 | 28.11× | PASS |
| 256³ | 8 | 5.063 | 0.006627381 | 335.310 | 0.100070000 | 66.23× | PASS |

_Note:_ GPU/GMP throughputs are computed from time using `2×M×N×K` and shown with **9 decimal places**.

_Note:_ We recomputed `GPU T-ops/s` from the recorded GPU time using `2×M×N×K` and now show **9 decimal places** to avoid rounding artifacts.

### Dyadic `mpq` samples

**Shape 128×128×128**

```
C[0,0] int32=-8188   C_real=-2047/64
C[21,42] int32=31289   C_real=31289/256
C[42,84] int32=-64593   C_real=-64593/256
C[63,126] int32=41221   C_real=41221/256
C[85,40] int32=-62714   C_real=-31357/128
C[106,82] int32=45047   C_real=45047/256
```

**Shape 192×192×192**

```
C[0,0] int32=-44682   C_real=-22341/128
C[32,0] int32=45766   C_real=22883/128
C[64,0] int32=-20511   C_real=-20511/256
C[96,0] int32=-29605   C_real=-29605/256
C[128,0] int32=91352   C_real=11419/32
C[160,0] int32=15501   C_real=15501/256
```

**Shape 256×256×256**

```
C[0,0] int32=-73289   C_real=-73289/256
C[42,170] int32=43845   C_real=43845/256
C[85,84] int32=127827   C_real=127827/256
C[127,254] int32=159434   C_real=79717/128
C[170,168] int32=-70952   C_real=-8869/32
C[213,82] int32=-5279   C_real=-5279/256
```

## Micro Throughput (Amortized — CUDA Graph + Reps)
`256³`, reps=500 (warmup=5):  
- **GPU:** elapsed=8.280 ms, avg/ GEMM=0.016560 ms, **2.026 T-ops/s**  
- **GMP (one GEMM):** 312.940 ms, 0.107223 G-ops/s  
- **Speedup:** 18897.23× per-GEMM vs GMP  

---
## Provenance & Environment
- Timestamp (UTC): `2025-11-07T10:02:31+00:00`
- GPU: `NVIDIA A100-SXM4-40GB`  |  SMs: `108`  |  Memory: `40506 MB`
- Recommended NVCC arch: ``-arch=sm_80``
- CUDA Runtime: `12.5.0 (12050)`  |  Driver: `12.4.0 (12040)`
- cuBLAS version: `120503`  |  cuBLASLt version: `120503`
- Ops definition: `logical ops/s = 2 * M * N * K` per GEMM.
- Arithmetic: INT8×INT8 → INT32 accumulate, exact; real-valued interpretation via dyadic `2^-(fracA+fracB)`.

---
## Ops Accounting — exact counting and what we report
_Generated: 2025-11-07T10:02:31+00:00_

**Canonical definition (GEMM):** For `C = A × B` with shapes `M×K` and `K×N`,
we count **`ops_per_gemm = 2 × M × N × K`** (one multiply + one add per inner term).

**PB-1 (Macro Swarm):** `panels = K/tileK`; each C slice performs `panels` GEMMs (first with β=0, rest β=1).
Per epoch: `gemms_per_epoch = streams × graph_nodes × batch_per_node × panels`.
Across epochs: `total_ops = epochs × gemms_per_epoch × (2×M×N×K)`; we report `total_ops/time`.

**PB-4 (Micro H2H):** Single GEMM at 128³/192³/256³; GPU T-ops/s = `2×M×N×K / ms_gpu × 1e-6 / 1000`.

**PB-6 (Graph+reps):** Capture one GEMM and replay `reps`; average per-GEMM time = `elapsed_ms/reps`;
GPU T-ops/s = `2×M×N×K / avg_ms × 1e-6 / 1000`.

### Independent recount (from recorded times)
| Section | Shape | Logged T-ops/s | Recomputed T-ops/s | Abs rel error |
|:--|:--|--:|--:|--:|
| macro | 5120×5120×5120 | 300.263373602 | 300.263373602 | 0.0000% |
| micro | 128×128×128 | 0.000848362 | 0.000848362 | 0.0001% |
| micro | 192×192×192 | 0.002822687 | 0.002822687 | 0.0000% |
| micro | 256×256×256 | 0.006627381 | 0.006627381 | 0.0000% |
| micro_amortized | 256×256×256 | 2.026233816 | 2.026233816 | 0.0000% |

_Small differences, when present, were from earlier rounded prints; all values here are recomputed from time._

---
## Throughput Conventions & Raw/Effective Ops
_Generated: 2025-11-07T10:02:31+00:00_

- **MAC definition:** A **MAC** is one multiply–accumulate. We report **RAW T-MAC/s** (hardware rate).
- **Ops definition:** Some readers count multiply and add separately; then **T-ops/s = 2 × T-MAC/s**.
- **Algorithmic multiplier k:** k = 1 (no algorithmic scaling). To show “effective ops,” set `ALG_K` and re-run this module.

**Arithmetic & scale**
- Precision: **INT8 × INT8 → INT32 accumulate** (exact as integers). Real-valued interpretation via dyadic shift `2^-(fracA+fracB)`; here `fracA=4`, `fracB=4` → shift = **2^-8**.
- Overflow guardrail (clamp = ±120): worst-case `|∑ a_k b_k| ≤ K·(120^2)`; must be `< 2^31` for int32. Bench shapes:
  - 5120×5120×5120: K=5120, bound ≤ 73,728,000 → int32 safe: YES
  - 128×128×128: K=128, bound ≤ 1,843,200 → int32 safe: YES
  - 192×192×192: K=192, bound ≤ 2,764,800 → int32 safe: YES
  - 256×256×256: K=256, bound ≤ 3,686,400 → int32 safe: YES

**Timing source**
- PB-4 uses **CUDA events**: YES  
- PB-6 uses **CUDA events**: YES  
(Device-side timing; avoids host wall-clock skew.)

### RAW and Effective Rates (derived from time)
| Section | Shape | GPU ms | RAW T-MAC/s | RAW T-ops/s | Exact |
|:--|:--|--:|--:|--:|:--:|
| macro | 5120×5120×5120 | 0.894 | 150.131686801 | 300.263373602 |  |
| micro | 128×128×128 | 4.944 | 0.000424181 | 0.000848362 | PASS |
| micro | 192×192×192 | 5.015 | 0.001411344 | 0.002822687 | PASS |
| micro | 256×256×256 | 5.063 | 0.003313691 | 0.006627381 | PASS |
| micro_amortized | 256×256×256 | 0.017 | 1.013116908 | 2.026233816 | PASS |

### Quick Verification Checklist
1) **Timing source** — confirm CUDA events (device time):
   - PB-4 cudaEventRecord: FOUND  
   - PB-6 cudaEventRecord: FOUND  
2) **Utilization (Nsight Compute)** — suggested commands:
```bash
ncu --set full --metrics \
  sm__pipe_tensor_cycles_active.avg.pct,\
  sm__inst_executed.avg.pct,\
  dram__throughput.avg.pct,\
  sm__warps_active.avg.pct_of_peak_sustained_active,\
  sm__maximum_warps_per_active_cycle_pct,\
  sm__average_active_threads_per_warp,\
  sm__throughput.avg.pct_of_peak_sustained_active,\
  achieved_occupancy \
  --target-processes all --launch-count 1 \
  /content/fx_int8_kpanel_tiled_swarm_v1 --m 5120 --n 5120 --k 5120 --streams 32 --graphNodes 64 --batchPerNode 4 --tileK 1280 --epochs 1 --warmup 3 --tryAlgos 16 --workspaceMB 1024 --validate 0

ncu --set full --metrics sm__pipe_tensor_cycles_active.avg.pct,sm__inst_executed.avg.pct,dram__throughput.avg.pct,achieved_occupancy \
  --target-processes all /content/pb4_gmp_vs_gpu_micro --m 256 --n 256 --k 256 --gmpPrint 0
```
3) **Clocks during the run** — sample with nvidia-smi:
```bash
nvidia-smi --query-gpu=name,clocks.gr,clocks.mem,pstate --format=csv -lms 200 > clocks_log.csv &
SAMPLER_PID=$!
/content/fx_int8_kpanel_tiled_swarm_v1 --m 5120 --n 5120 --k 5120 --streams 32 --graphNodes 64 --batchPerNode 4 --tileK 1280 --epochs 2 --warmup 6 --tryAlgos 64 --workspaceMB 1024 --validate 0
kill ${SAMPLER_PID}
```
**Current clocks sample (best-effort):**

```
NVIDIA A100-SXM4-40GB, 1095 MHz, 1215 MHz, P0
```

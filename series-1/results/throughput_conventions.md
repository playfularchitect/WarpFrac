
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


---
## FASTLANE Trust Pack Summary
_Generated: 2025-11-17T00:19:44+00:00_

**Macro stability (3× FASTLANE runs, 5120×5120×5120 INT8→INT32):**

- RAW G-MAC/s : mean=209824.36  ±111.93  (min=209691.70, max=209965.47)
- per-matmul  : mean=0.640 ms  ±0.000  (min=0.639, max=0.640)
- Derived RAW T-MAC/s ≈ 209.82  (RAW T-ops/s ≈ 419.65)

**Clock/Power log:** sampled with `nvidia-smi` during one FASTLANE run.  
_Interpretation:_ no samples / not available  
_Tail of log (`fastlane_clocks_power_log.csv`):_

```text
2025/11/17 00:19:42.957, NVIDIA A100-SXM4-40GB, 0, P0, 1410 MHz, 1215 MHz, 0 %, 0 %, 81.68 W
2025/11/17 00:19:43.183, NVIDIA A100-SXM4-40GB, 0, P0, 1410 MHz, 1215 MHz, 2 %, 0 %, 268.49 W
2025/11/17 00:19:43.387, NVIDIA A100-SXM4-40GB, 0, P0, 1410 MHz, 1215 MHz, 97 %, 21 %, 132.74 W
2025/11/17 00:19:43.591, NVIDIA A100-SXM4-40GB, 0, P0, 1410 MHz, 1215 MHz, 56 %, 17 %, 82.36 W
2025/11/17 00:19:43.794, NVIDIA A100-SXM4-40GB, 0, P0, 1410 MHz, 1215 MHz, 0 %, 0 %, 81.68 W
2025/11/17 00:19:43.997, NVIDIA A100-SXM4-40GB, 0, P0, 1410 MHz, 1215 MHz, 0 %, 0 %, 81.14 W
2025/11/17 00:19:44.201, NVIDIA A100-SXM4-40GB, 0, P0, 1410 MHz, 1215 MHz, 0 %, 0 %, 81.22 W
2025/11/17 00:19:44.405, NVIDIA A100-SXM4-40GB, 0, P0, 1410 MHz, 1215 MHz, 0 %, 0 %, 81.14 W
```

**Nsight Compute capture (macro):**
- Macro: `fastlane_macro.ncu-rep` (open in Nsight Compute UI to inspect kernel-level metrics)

**Provenance (short view):**

```json
{
  "timestamp_utc": "2025-11-17T00:19:44+00:00",
  "binary": {
    "path": "/content/fastlane_best_rr_allinone",
    "sha256": "9af22d21207567882cc65f9b12c1588dbbfa9c2f9e620f94eaddf9850c0802fc"
  },
  "nvcc_version_line": [
    "nvcc: NVIDIA (R) Cuda compiler driver"
  ],
  "fastlane_cmd": "/content/fastlane_best_rr_allinone"
}
```
Full provenance JSON: `fastlane_provenance.json`  

**What these numbers are about:**
- INT8×INT8→INT32 accumulate on a single A100-class GPU using cuBLASLt heuristics.
- RAW G-MAC/s is the physical MAC rate for a 5120³ GEMM, using the standard `MACs = M·N·K` formula.
- RAW GFLOP/s (or T-ops/s) uses `ops = 2·M·N·K` to count multiply+add as two operations.
- EFFECTIVE (k-lanes) is an optional logical counting scheme used by the FASTLANE code (e.g., k=9) and is explicitly labeled.

Additional correctness witnesses (e.g., CPU or GMP cross-checks) can be layered on in companion modules; this pack focuses on performance and environment reproducibility.

**Repro (~1–2 minutes):** use the generated `Dockerfile_fastlane` and `run_short_fastlane.sh` in the `fastlane_trust_pack` directory.

**Units:** MACs per GEMM = M·N·K, ops per GEMM = 2·M·N·K (ops in 10⁰; rates reported in G=10⁹ or T=10¹²).

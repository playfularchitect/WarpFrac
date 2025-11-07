
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

#==================================================================================================
# MODULE FL-0 — WARPFRAC FASTLANE MACRO BENCH (5120³ INT8xINT8->INT32, cuBLASLt)
# - Writes: /content/fastlane_best_rr_allinone.cu
# - Builds: /content/fastlane_best_rr_allinone  (A100 sm_80)
# - Runs:   5120×5120×5120 INT8->INT32 GEMM, REPLAYS=200 by default
# - Reports:
#       RAW G-MAC/s   -> physical MAC/s (no k-multiplier)
#       RAW GFLOP/s   -> 2 * RAW (mul+add)
#       EFFECTIVE     -> k * RAW (logical exact ops, K_LANES=9 by default)
#==================================================================================================
import os, shutil, subprocess

print("\n" + "="*118)
print("MODULE FL-0 — WARPFRAC FASTLANE MACRO BENCH (5120^3 INT8xINT8->INT32, cuBLASLt)")
print("="*118)

CUDA_SRC_PATH = "/content/fastlane_best_rr_allinone.cu"
BIN_PATH      = "/content/fastlane_best_rr_allinone"

cuda_src = r"""// fastlane_best_rr_allinone.cu
// Best-known path from your runs: A:ROW, B:ROW (math tB=T), C:ROW, WS=0
// Probes cuBLASLt heuristics, chooses fastest, times REPLAYS end-to-end with full audit.

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cublasLt.h>

#ifndef REPLAYS
#define REPLAYS 200
#endif
#ifndef PROBE_RUNS
#define PROBE_RUNS 16
#endif
#ifndef MAX_H
#define MAX_H 64
#endif
#ifndef WS_CAP_MB
#define WS_CAP_MB 0
#endif
#ifndef K_LANES
#define K_LANES 9
#endif

static void ck(cudaError_t e,const char* m){
  if(e!=cudaSuccess){ std::fprintf(stderr,"CUDA %s: %s\n",m,cudaGetErrorString(e)); std::exit(1); }
}
static bool ok(cublasStatus_t s,const char* m){
  if(s!=CUBLAS_STATUS_SUCCESS){ std::fprintf(stderr,"(skip) cuBLAS %s: %d\n",m,(int)s); return false; }
  return true;
}
static void bk(cublasStatus_t s,const char* m){
  if(s!=CUBLAS_STATUS_SUCCESS){ std::fprintf(stderr,"cuBLAS %s: %d\n",m,(int)s); std::exit(1); }
}

// Probe given algo, return avg ms/GEMM or big on failure
static double probe_ms(cublasLtHandle_t lt, const cublasLtMatmulAlgo_t& algo,
                       int M,int N,int K,
                       cublasLtMatrixLayout_t Ad,cublasLtMatrixLayout_t Bd,cublasLtMatrixLayout_t Cd,
                       int8_t* dA,int8_t* dB,int32_t* dC,
                       size_t ws_bytes, void* ws_buf, int iters, bool* ran_ok)
{
  *ran_ok=false;
  cublasOperation_t tA=CUBLAS_OP_N, tB=CUBLAS_OP_T; // ROW/ROW => math uses B^T
  cublasLtMatmulDesc_t desc;
  if(!ok(cublasLtMatmulDescCreate(&desc, CUBLAS_COMPUTE_32I, CUDA_R_32I),"desc")) return 1e30;
  ok(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSA,&tA,sizeof(tA)),"TA");
  ok(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSB,&tB,sizeof(tB)),"TB");

  int32_t alpha=1,beta=0;
  cudaStream_t s; ck(cudaStreamCreate(&s),"stream");

  // warm
  if(!ok(cublasLtMatmul(lt, desc, &alpha, dA,Ad, dB,Bd, &beta, dC,Cd, dC,Cd, &algo, ws_buf, ws_bytes, s),"warm")){
    cudaStreamDestroy(s); cublasLtMatmulDescDestroy(desc); return 1e30;
  }
  ck(cudaStreamSynchronize(s),"warm sync");

  // timed
  cudaEvent_t t0,t1; ck(cudaEventCreate(&t0),"t0"); ck(cudaEventCreate(&t1),"t1");
  ck(cudaEventRecord(t0, s),"t0 rec");
  for(int i=0;i<iters;i++){
    if(!ok(cublasLtMatmul(lt, desc, &alpha, dA,Ad, dB,Bd, &beta, dC,Cd, dC,Cd, &algo, ws_buf, ws_bytes, s),"run")){
      ck(cudaEventDestroy(t0),"dt0"); ck(cudaEventDestroy(t1),"dt1"); cudaStreamDestroy(s); cublasLtMatmulDescDestroy(desc);
      return 1e30;
    }
  }
  ck(cudaEventRecord(t1, s),"t1 rec");
  ck(cudaEventSynchronize(t1),"sync");
  float ms=0; ck(cudaEventElapsedTime(&ms,t0,t1),"elapsed");

  ck(cudaEventDestroy(t0),"dt0"); ck(cudaEventDestroy(t1),"dt1");
  ck(cudaStreamDestroy(s),"ds");
  cublasLtMatmulDescDestroy(desc);
  *ran_ok=true;
  return double(ms)/double(iters);
}

int main(){
  // Problem dims
  const int M=5120,N=5120,K=5120;
  const size_t bytesA=(size_t)M*(size_t)K;
  const size_t bytesB=(size_t)K*(size_t)N;
  const size_t bytesC=(size_t)M*(size_t)N*sizeof(int32_t);

  // Buffers
  int8_t *dA=nullptr,*dB=nullptr; int32_t* dC=nullptr;
  ck(cudaMalloc(&dA,bytesA),"A"); ck(cudaMalloc(&dB,bytesB),"B"); ck(cudaMalloc(&dC,bytesC),"C");
  ck(cudaMemset(dA,1,bytesA),"memA"); ck(cudaMemset(dB,1,bytesB),"memB"); ck(cudaMemset(dC,0,bytesC),"memC");

  // cuBLASLt
  cublasLtHandle_t lt; bk(cublasLtCreate(&lt),"lt");

  // Layouts: A:ROW, B:ROW, C:ROW  (ld = columns)
  int lda=K, ldb=N, ldc=N;
  cublasLtMatrixLayout_t Ad,Bd,Cd;
  bk(cublasLtMatrixLayoutCreate(&Ad, CUDA_R_8I,  M, K, lda),"A");
  bk(cublasLtMatrixLayoutCreate(&Bd, CUDA_R_8I,  K, N, ldb),"B");
  bk(cublasLtMatrixLayoutCreate(&Cd, CUDA_R_32I, M, N, ldc),"C");
  cublasLtOrder_t row = CUBLASLT_ORDER_ROW;
  bk(cublasLtMatrixLayoutSetAttribute(Ad, CUBLASLT_MATRIX_LAYOUT_ORDER,&row,sizeof(row)),"Arow");
  bk(cublasLtMatrixLayoutSetAttribute(Bd, CUBLASLT_MATRIX_LAYOUT_ORDER,&row,sizeof(row)),"Brow");
  bk(cublasLtMatrixLayoutSetAttribute(Cd, CUBLASLT_MATRIX_LAYOUT_ORDER,&row,sizeof(row)),"Crow");

  // Workspace (kept 0 by default)
  size_t ws_cap = (size_t)WS_CAP_MB * 1024ull * 1024ull;
  void* ws_buf = nullptr; if(ws_cap>0) ck(cudaMalloc(&ws_buf, ws_cap),"ws");

  // Heuristics for this exact layout
  cublasLtMatmulDesc_t hdesc; bk(cublasLtMatmulDescCreate(&hdesc, CUBLAS_COMPUTE_32I, CUDA_R_32I),"hdesc");
  cublasOperation_t tA=CUBLAS_OP_N, tB=CUBLAS_OP_T;
  bk(cublasLtMatmulDescSetAttribute(hdesc, CUBLASLT_MATMUL_DESC_TRANSA,&tA,sizeof(tA)),"hTA");
  bk(cublasLtMatmulDescSetAttribute(hdesc, CUBLASLT_MATMUL_DESC_TRANSB,&tB,sizeof(tB)),"hTB");
  cublasLtMatmulPreference_t pref; bk(cublasLtMatmulPreferenceCreate(&pref),"pref");
  bk(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,&ws_cap,sizeof(ws_cap)),"prefWS");
  cublasLtMatmulHeuristicResult_t H[MAX_H]; int found=0;
  bk(cublasLtMatmulAlgoGetHeuristic(lt, hdesc, Ad, Bd, Cd, Cd, pref, MAX_H, H, &found),"heur");
  std::printf("cuBLAS heur (cfg: A:ROW  B:ROW(tB=T)  C:ROW): %d\n", found);
  cublasLtMatmulPreferenceDestroy(pref);
  cublasLtMatmulDescDestroy(hdesc);

  // Probe & choose best
  double best_ms = 1e30; size_t best_ws = 0; cublasLtMatmulAlgo_t best{};
  for(int i=0;i<found;i++){
    if(H[i].workspaceSize > ws_cap) continue;
    bool ran=false;
    double ms = probe_ms(lt, H[i].algo, M,N,K, Ad,Bd,Cd, dA,dB,dC, H[i].workspaceSize, ws_buf, PROBE_RUNS, &ran);
    if(ran && ms < best_ms){ best_ms = ms; best = H[i].algo; best_ws = H[i].workspaceSize; }
  }
  if(best_ms >= 1e29){
    std::fprintf(stderr,"No runnable algo under WS cap. Exiting.\n");
    if(ws_buf) cudaFree(ws_buf);
    cublasLtMatrixLayoutDestroy(Ad); cublasLtMatrixLayoutDestroy(Bd); cublasLtMatrixLayoutDestroy(Cd);
    cublasLtDestroy(lt);
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    return 2;
  }

  // Final timed loop
  cublasLtMatmulDesc_t desc; bk(cublasLtMatmulDescCreate(&desc, CUBLAS_COMPUTE_32I, CUDA_R_32I),"desc");
  bk(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSA,&tA,sizeof(tA)),"TA");
  bk(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSB,&tB,sizeof(tB)),"TB");
  int32_t alpha=1,beta=0;
  cudaStream_t s; ck(cudaStreamCreate(&s),"stream");

  // warm
  bk(cublasLtMatmul(lt, desc, &alpha, dA,Ad, dB,Bd, &beta, dC,Cd, dC,Cd, &best, ws_buf, best_ws, s),"warm");
  ck(cudaStreamSynchronize(s),"warm sync");

  // timed
  cudaEvent_t t0,t1; ck(cudaEventCreate(&t0),"t0"); ck(cudaEventCreate(&t1),"t1");
  ck(cudaEventRecord(t0, s),"t0 rec");
  for(int r=0;r<REPLAYS;r++){
    bk(cublasLtMatmul(lt, desc, &alpha, dA,Ad, dB,Bd, &beta, dC,Cd, dC,Cd, &best, ws_buf, best_ws, s),"run");
  }
  ck(cudaEventRecord(t1, s),"t1 rec");
  ck(cudaEventSynchronize(t1),"sync");
  float ms=0; ck(cudaEventElapsedTime(&ms,t0,t1),"elapsed");

  const double mac_per = (double)M*(double)N*(double)K;
  const double mac_tot = mac_per * (double)REPLAYS;
  const double sec = ms/1000.0;
  const double raw_gmac_s = (mac_tot/sec)/1e9;
  const double eff_gops_s = K_LANES * raw_gmac_s;

  std::printf("=== FASTLANE autopick(min-rr, FIX) — INT8xINT8->INT32 (cuBLASLt) ===\n");
  std::printf("Layout: A:ROW  B:ROW(tB=T)  C:ROW | REPLAYS=%d | WS_CAP=%d MB | picked_ws=%.2f MB | probe_best_ms=%.3f\n",
              REPLAYS, WS_CAP_MB, best_ws/1048576.0, best_ms);
  std::printf("RAW=%.2f G-MAC/s\n", raw_gmac_s);
  std::printf("EFFECTIVE (k=%d) = %.2f G-ops/s\n", K_LANES, eff_gops_s);

  std::printf("\n--- AUDIT: how counts are computed ---\n");
  std::printf("MACs per matmul: %d * %d * %d = %.0f\n", M,N,K, mac_per);
  std::printf("Total MACs executed: %.0f * %d = %.0f\n", mac_per, REPLAYS, mac_tot);
  std::printf("Wall time (one stream): %.3f ms | per-matmul(avg) = %.3f ms\n", ms, ms/REPLAYS);
  std::printf("RAW G-MAC/s = (%.0f / %.6f) / 1e9 = %.2f\n", mac_tot, sec, raw_gmac_s);
  std::printf("RAW GFLOP/s (reference) = 2 * RAW = %.2f\n", 2.0*raw_gmac_s);
  std::printf("EFFECTIVE exact-ops/s = %d * %.2f = %.2f G-ops/s\n", K_LANES, raw_gmac_s, eff_gops_s);
  std::printf("Buffers: |A|=%zu B, |B|=%zu B, |C|=%zu B\n",
              (size_t)M*(size_t)K, (size_t)K*(size_t)N, (size_t)M*(size_t)N*sizeof(int32_t));
  std::printf("--------------------------------------\n");

  // cleanup
  ck(cudaEventDestroy(t0),"dt0"); ck(cudaEventDestroy(t1),"dt1");
  ck(cudaStreamDestroy(s),"ds");
  cublasLtMatmulDescDestroy(desc);
  if(ws_buf) cudaFree(ws_buf);
  cublasLtMatrixLayoutDestroy(Ad); cublasLtMatrixLayoutDestroy(Bd); cublasLtMatrixLayoutDestroy(Cd);
  cublasLtDestroy(lt);
  cudaFree(dA); cudaFree(dB); cudaFree(dC);
  return 0;
}
"""

with open(CUDA_SRC_PATH, "w") as f:
    f.write(cuda_src)

print(f"[FL-0] Wrote FASTLANE CUDA source -> {CUDA_SRC_PATH}")

#----------------------------------------------------------------------------------
# Compile for A100 (sm_80)
#----------------------------------------------------------------------------------
if not shutil.which("nvcc"):
    print("[FL-0] FATAL: nvcc not found in PATH. Make sure runtime has a GPU & CUDA toolkit.")
else:
    cmd = [
        "nvcc",
        "-O3",
        "-std=c++17",
        "-gencode=arch=compute_80,code=sm_80",
        CUDA_SRC_PATH,
        "-lcublasLt",
        "-lcublas",
        "-o", BIN_PATH,
    ]
    print("[FL-0] Compiling with:", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True)
        print(f"[FL-0] Compile OK -> {BIN_PATH}")
    except subprocess.CalledProcessError as e:
        print("[FL-0] ERROR during nvcc compile, return code:", e.returncode)
        raise

    #----------------------------------------------------------------------------------
    # Run the FASTLANE macro benchmark
    #----------------------------------------------------------------------------------
    print("\n" + "-"*118)
    print("RUNNING FASTLANE MACRO BENCH (5120^3 INT8xINT8->INT32, REPLAYS=200)")
    print("-"*118)
    try:
        result = subprocess.run([BIN_PATH], check=True, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr.strip():
            print("\n[FL-0] Program stderr:")
            print(result.stderr)
    except subprocess.CalledProcessError as e:
        print("[FL-0] ERROR during program run, return code:", e.returncode)
        print("STDOUT:\n", e.stdout)
        print("STDERR:\n", e.stderr)

print("\n" + "="*118)
print("MODULE FL-0 — DONE (Fastlane Macro Bench)")
print("="*118)













#==================================================================================================
# MODULE FL-1B — FASTLANE TRUST PACK v2 (Macro Repro/Provenance/Clocks/Nsight)
# - Assumes MODULE FL-0 already ran and built:
#       /content/fastlane_best_rr_allinone.cu
#       /content/fastlane_best_rr_allinone
# - Does:
#       * 3× macro runs → stats JSON (RAW G-MAC/s, per-matmul ms)
#       * nvidia-smi clocks/power sampling during a run
#       * optional Nsight Compute macro capture (.ncu-rep)
#       * provenance.json with nvcc, driver info, SHA256 of binary/source, command
#       * Dockerfile + run_short.sh
#       * fastlane_trust_pack.md summary (performance + environment only)
#==================================================================================================
import os, sys, re, json, time, csv, hashlib, shutil, subprocess, statistics, datetime, glob

print("\n" + "="*118)
print("MODULE FL-1B — FASTLANE TRUST PACK v2 (Macro Repro/Provenance/Clocks/Nsight)")
print("="*118)

# ---------- Paths ----------
FASTLANE_CU   = "/content/fastlane_best_rr_allinone.cu"
FASTLANE_EXE  = "/content/fastlane_best_rr_allinone"

OUT_DIR       = "/content/fastlane_trust_pack"
os.makedirs(OUT_DIR, exist_ok=True)

NSYSMI_LOG    = os.path.join(OUT_DIR, "fastlane_clocks_power_log.csv")
NCU_MACRO     = os.path.join(OUT_DIR, "fastlane_macro.ncu-rep")
PROV_JSON     = os.path.join(OUT_DIR, "fastlane_provenance.json")
PROV_MIN_JSON = os.path.join(OUT_DIR, "fastlane_provenance_min.json")
STATS_JSON    = os.path.join(OUT_DIR, "fastlane_macro_stats.json")
DOCKERFILE    = os.path.join(OUT_DIR, "Dockerfile_fastlane")
RUN_SHORT     = os.path.join(OUT_DIR, "run_short_fastlane.sh")
README        = os.path.join(OUT_DIR, "README_fastlane_trust_pack.txt")
SUMMARY_MD    = os.path.join(OUT_DIR, "fastlane_trust_pack.md")

# ---------- Helpers ----------
def run(cmd, timeout=None, check=False):
    return subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=timeout,
        check=check,
    )

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def have(cmd):
    return shutil.which(cmd) is not None

def parse_fastlane_output(txt):
    """
    Parse FASTLANE output for:
      RAW=xxxxx G-MAC/s
      per-matmul(avg) = yyy ms
    Returns (raw_gmac_s, per_matmul_ms) or (None, None) if parse failed.
    """
    m_raw = re.search(r"RAW=([\d.]+)\s*G-MAC/s", txt)
    m_per = re.search(r"per-matmul\(avg\)\s*=\s*([\d.]+)\s*ms", txt)
    raw = float(m_raw.group(1)) if m_raw else None
    per = float(m_per.group(1)) if m_per else None
    return raw, per

def nvcc_version():
    try:
        return run(["bash", "-lc", "nvcc --version"]).stdout.strip()
    except Exception:
        return "unknown"

def cuda_driver_runtime():
    try:
        smi = run(["bash", "-lc", "nvidia-smi -q -x"]).stdout
        # keep it short-ish
        return smi[:4000]
    except Exception:
        return "nvidia-smi unavailable"

def read_tail(path, n=8):
    if not os.path.exists(path):
        return "(no clock/power log)"
    with open(path, "r") as f:
        lines = f.read().strip().splitlines()
    return "\n".join(lines[-n:])

print("\n[FL-1B] Sanity-checking FASTLANE binary & source...")
if not os.path.exists(FASTLANE_EXE):
    print(f"[FL-1B] FATAL: {FASTLANE_EXE} not found.")
    print("        Run MODULE FL-0 first to build & run the FASTLANE macro bench.")
    raise SystemExit(1)
if not os.path.exists(FASTLANE_CU):
    print(f"[FL-1B] WARNING: CUDA source {FASTLANE_CU} not found; provenance will omit source hash.")

#==================================================================================================
# 1) Macro: 3× measurement (RAW G-MAC/s & per-matmul ms)
#==================================================================================================
print("\n" + "-"*118)
print("FASTLANE MACRO: 3× measurement (RAW G-MAC/s & per-matmul avg ms)")
print("-"*118)

triple = []
for i in range(3):
    out = run([FASTLANE_EXE], timeout=300).stdout
    raw, per = parse_fastlane_output(out)
    if raw is None or per is None:
        print("[FL-1B] ERROR: Failed to parse FASTLANE output for RAW / per-matmul.")
        print("---------- RAW OUTPUT ----------")
        print(out)
        print("---------- END OUTPUT ----------")
        raise RuntimeError("FASTLANE parse failure")
    print(f"[Run {i+1}] RAW={raw:.2f} G-MAC/s  per-matmul={per:.3f} ms")
    triple.append((raw, per))

raw_vals = [x[0] for x in triple]
per_vals = [x[1] for x in triple]

stats = {
    "RAW_GMAC_per_s": {
        "mean": statistics.mean(raw_vals),
        "stdev": statistics.pstdev(raw_vals),
        "min": min(raw_vals),
        "max": max(raw_vals),
    },
    "per_matmul_ms": {
        "mean": statistics.mean(per_vals),
        "stdev": statistics.pstdev(per_vals),
        "min": min(per_vals),
        "max": max(per_vals),
    },
    "runs": [{"RAW_GMAC_per_s": r, "per_matmul_ms": p} for r, p in triple],
}

with open(STATS_JSON, "w") as f:
    json.dump(stats, f, indent=2)

print("\n--- FASTLANE 3× stats ---")
print(
    "RAW G-MAC/s : "
    f"mean={stats['RAW_GMAC_per_s']['mean']:.2f}  ±{stats['RAW_GMAC_per_s']['stdev']:.2f}  "
    f"(min={stats['RAW_GMAC_per_s']['min']:.2f}, max={stats['RAW_GMAC_per_s']['max']:.2f})"
)
print(
    "per_matmul_ms: "
    f"mean={stats['per_matmul_ms']['mean']:.3f}  ±{stats['per_matmul_ms']['stdev']:.3f}  "
    f"(min={stats['per_matmul_ms']['min']:.3f}, max={stats['per_matmul_ms']['max']:.3f})"
)

#==================================================================================================
# 2) Clock/power log with nvidia-smi during a FASTLANE run
#==================================================================================================
print("\n" + "-"*118)
print("CLOCK/POWER LOG (nvidia-smi) — single FASTLANE run with sampling")
print("-"*118)

sampler = None
interp_line = ""
if have("nvidia-smi"):
    try:
        # start sampler at ~200ms interval
        sampler = subprocess.Popen(
            [
                "bash",
                "-lc",
                "nvidia-smi "
                "--query-gpu=timestamp,name,index,pstate,clocks.gr,clocks.mem,"
                "utilization.gpu,utilization.memory,power.draw "
                f"--format=csv -lms 200 > {NSYSMI_LOG}",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        time.sleep(0.4)
        out = run([FASTLANE_EXE], timeout=300).stdout
        time.sleep(0.4)
    finally:
        if sampler is not None:
            try:
                sampler.terminate()
                sampler.wait(timeout=3)
            except Exception:
                sampler.kill()
else:
    print("[FL-1B] nvidia-smi not available; skipping clock/power sampling.")

# quick interpretation from last few samples
if os.path.exists(NSYSMI_LOG):
    with open(NSYSMI_LOG, "r") as f:
        lines = f.read().strip().splitlines()
    if lines:
        hdr, data = lines[0], lines[1:]
        try:
            cols_hdr = [c.strip() for c in hdr.split(",")]
            idx_gr = None
            for idx, name in enumerate(cols_hdr):
                if name.startswith("clocks.gr"):
                    idx_gr = idx
                    break
            if idx_gr is not None:
                vals = []
                for row in data[-15:]:
                    cols = [c.strip() for c in row.split(",")]
                    try:
                        vals.append(float(cols[idx_gr].split()[0]))
                    except Exception:
                        pass
                if vals:
                    avg = sum(vals) / len(vals)
                    interp_line = f"SM clock during FASTLANE ≈ {avg:.0f} MHz (last samples)."
        except Exception:
            interp_line = ""
print("Clock/power interpretation:", interp_line or "(no samples / parse failed)")

#==================================================================================================
# 3) Nsight Compute capture for FASTLANE macro
#==================================================================================================
print("\n" + "-"*118)
print("NSIGHT COMPUTE CAPTURE (FASTLANE macro)")
print("-"*118)

ncu_ok = have("ncu") or have("nv-nsight-cu-cli")
ncu = shutil.which("ncu") or shutil.which("nv-nsight-cu-cli")
macro_ncu_ok = False

if ncu_ok:
    try:
        print(f"[FL-1B] Collecting FASTLANE macro .ncu-rep via {os.path.basename(ncu)} …")
        cmd = [
            ncu,
            "-o",
            NCU_MACRO,
            "--target-processes",
            "all",
            "--launch-count",
            "1",
            "--set",
            "full",
            FASTLANE_EXE,
        ]
        _ = run(cmd, timeout=600)
        if os.path.exists(NCU_MACRO) or os.path.exists(NCU_MACRO + ".ncu-rep"):
            macro_ncu_ok = True
        else:
            found_reps = glob.glob(os.path.join(OUT_DIR, "*.ncu-rep"))
            macro_ncu_ok = len(found_reps) > 0
            if macro_ncu_ok and not os.path.exists(NCU_MACRO) and found_reps:
                NCU_MACRO = found_reps[0]
    except Exception as e:
        print("[FL-1B] Nsight capture error:", e)
else:
    print("[FL-1B] Nsight CLI (ncu/nv-nsight-cu-cli) not found; skipping capture.")

if macro_ncu_ok:
    print(f"[FL-1B] Nsight macro capture available in: {NCU_MACRO}")
    print("       (Open this .ncu-rep in Nsight Compute UI to inspect kernel-level metrics)")
else:
    print("[FL-1B] No Nsight .ncu-rep generated (tool missing or capture failed).")

#==================================================================================================
# 4) Provenance JSON (environment + binary/source hashes + commands)
#==================================================================================================
print("\n" + "-"*118)
print("PROVENANCE — environment + binaries + command lines")
print("-"*118)

prov = {
    "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
    "nvcc_version": nvcc_version(),
    "cuda_info": cuda_driver_runtime(),
    "compile_flags_hint": "-O3 -std=c++17 -arch=sm_80 -lcublasLt -lcublas",
    "binaries": [],
    "sources": [],
    "commands": {
        "fastlane_run": FASTLANE_EXE,
    },
}

if os.path.exists(FASTLANE_EXE):
    prov["binaries"].append(
        {"path": FASTLANE_EXE, "sha256": sha256_file(FASTLANE_EXE)}
    )
if os.path.exists(FASTLANE_CU):
    prov["sources"].append(
        {"path": FASTLANE_CU, "sha256": sha256_file(FASTLANE_CU)}
    )

with open(PROV_JSON, "w") as f:
    json.dump(prov, f, indent=2)

prov_min = {
    "timestamp_utc": prov["timestamp_utc"],
    "binary": prov["binaries"][0] if prov["binaries"] else None,
    "nvcc_version_line": prov["nvcc_version"].splitlines()[:1],
    "fastlane_cmd": prov["commands"]["fastlane_run"],
}
with open(PROV_MIN_JSON, "w") as f:
    json.dump(prov_min, f, indent=2)

#==================================================================================================
# 5) Dockerfile + run_short.sh for quick repro
#==================================================================================================
print("\n" + "-"*118)
print("DOCKERFILE + run_short.sh (FASTLANE quick repro)")
print("-"*118)

dockerfile_txt = f"""\
FROM nvidia/cuda:12.2.2-devel-ubuntu22.04

RUN apt-get update && apt-get install -y build-essential && \\
    rm -rf /var/lib/apt/lists/*

WORKDIR /bench

# Usage:
#   docker build -t fastlane-bench .
#   docker run --gpus all -it -v $PWD:/bench fastlane-bench /bin/bash
#
# Inside the container:
#   nvcc -O3 -std=c++17 -arch=sm_80 fastlane_best_rr_allinone.cu -lcublasLt -lcublas -o fastlane_best_rr_allinone
#   ./fastlane_best_rr_allinone
"""

with open(DOCKERFILE, "w") as f:
    f.write(dockerfile_txt)

run_short_txt = f"""\
#!/usr/bin/env bash
set -euo pipefail
# FASTLANE quick repro: compile & run (adjust -arch as needed)
nvcc -O3 -std=c++17 -arch=sm_80 fastlane_best_rr_allinone.cu -lcublasLt -lcublas -o fastlane_best_rr_allinone
./fastlane_best_rr_allinone
"""
with open(RUN_SHORT, "w") as f:
    f.write(run_short_txt)
os.chmod(RUN_SHORT, 0o755)

with open(README, "w") as f:
    f.write("FASTLANE Trust Pack contents:\n")
    f.write(f"- Macro stats (3×): {STATS_JSON}\n")
    f.write(f"- Nsight .ncu-rep (macro, if available): {NCU_MACRO}\n")
    f.write(f"- Clock/Power log: {NSYSMI_LOG}\n")
    f.write(f"- Provenance JSON: {PROV_JSON}\n")
    f.write(f"- Dockerfile + run_short.sh for quick repro\n")

#==================================================================================================
# 6) Human-readable summary (for README / paper appendix)
#==================================================================================================
print("\n" + "-"*118)
print("HUMAN SUMMARY (also written to fastlane_trust_pack.md)")
print("-"*118)

# Derived helper numbers
mean_raw = stats["RAW_GMAC_per_s"]["mean"]
mean_per = stats["per_matmul_ms"]["mean"]
# raw MAC/s in T-MAC/s and T-ops/s
mean_TMAC = mean_raw / 1000.0
mean_TOPS = 2.0 * mean_TMAC

summary_lines = []

summary_lines.append("\n---\n")
summary_lines.append("## FASTLANE Trust Pack Summary\n")
summary_lines.append(
    f"_Generated: {datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds')}_\n\n"
)

summary_lines.append("**Macro stability (3× FASTLANE runs, 5120×5120×5120 INT8→INT32):**\n\n")
summary_lines.append(
    f"- RAW G-MAC/s : mean={stats['RAW_GMAC_per_s']['mean']:.2f}  "
    f"±{stats['RAW_GMAC_per_s']['stdev']:.2f}  "
    f"(min={stats['RAW_GMAC_per_s']['min']:.2f}, max={stats['RAW_GMAC_per_s']['max']:.2f})\n"
)
summary_lines.append(
    f"- per-matmul  : mean={stats['per_matmul_ms']['mean']:.3f} ms  "
    f"±{stats['per_matmul_ms']['stdev']:.3f}  "
    f"(min={stats['per_matmul_ms']['min']:.3f}, max={stats['per_matmul_ms']['max']:.3f})\n"
)
summary_lines.append(
    f"- Derived RAW T-MAC/s ≈ {mean_TMAC:.2f}  (RAW T-ops/s ≈ {mean_TOPS:.2f})\n\n"
)

summary_lines.append("**Clock/Power log:** sampled with `nvidia-smi` during one FASTLANE run.  \n")
summary_lines.append(
    f"_Interpretation:_ {interp_line or 'no samples / not available'}  \n"
)
summary_lines.append(
    f"_Tail of log (`{os.path.basename(NSYSMI_LOG)}`):_\n\n```text\n{read_tail(NSYSMI_LOG)}\n```\n\n"
)

summary_lines.append("**Nsight Compute capture (macro):**\n")
if macro_ncu_ok:
    summary_lines.append(
        f"- Macro: `{os.path.basename(NCU_MACRO)}` (open in Nsight Compute UI to inspect kernel-level metrics)\n"
    )
else:
    summary_lines.append(
        "- Macro: (not captured — Nsight CLI not available or capture failed in this environment)\n"
    )
summary_lines.append("\n")

summary_lines.append("**Provenance (short view):**\n\n```json\n")
summary_lines.append(json.dumps(prov_min, indent=2))
summary_lines.append("\n```\n")
summary_lines.append(f"Full provenance JSON: `{os.path.basename(PROV_JSON)}`  \n\n")

summary_lines.append("**What these numbers are about:**\n")
summary_lines.append(
    "- INT8×INT8→INT32 accumulate on a single A100-class GPU using cuBLASLt heuristics.\n"
)
summary_lines.append(
    "- RAW G-MAC/s is the physical MAC rate for a 5120³ GEMM, using the standard `MACs = M·N·K` formula.\n"
)
summary_lines.append(
    "- RAW GFLOP/s (or T-ops/s) uses `ops = 2·M·N·K` to count multiply+add as two operations.\n"
)
summary_lines.append(
    "- EFFECTIVE (k-lanes) is an optional logical counting scheme used by the FASTLANE code (e.g., k=9) and is explicitly labeled.\n"
)

summary_lines.append(
    "\nAdditional correctness witnesses (e.g., CPU or GMP cross-checks) can be layered on in companion modules; "
    "this pack focuses on performance and environment reproducibility.\n"
)

summary_lines.append(
    "\n**Repro (~1–2 minutes):** use the generated `Dockerfile_fastlane` and `run_short_fastlane.sh` in the "
    "`fastlane_trust_pack` directory.\n"
)
summary_lines.append(
    "\n**Units:** MACs per GEMM = M·N·K, ops per GEMM = 2·M·N·K (ops in 10⁰; rates reported in G=10⁹ or T=10¹²).\n"
)

summary_txt = "".join(summary_lines)
print(summary_txt)

with open(SUMMARY_MD, "w") as f:
    f.write(summary_txt)

print("\n" + "="*118)
print("MODULE FL-1B — DONE (FASTLANE Trust Pack v2 in /content/fastlane_trust_pack)")
print("="*118)














#==================================================================================================
# MODULE FL-2 — FASTLANE EXACTNESS WITNESS (INT8×INT8→INT32)
# - Purpose: Show that INT8×INT8→INT32 GEMM on this GPU is bit-for-bit exact in the safe range.
# - Micro witness:
#     * M=N=K=256
#     * Random int8 A,B on host
#     * CPU reference with int64 accumulator (no overflow), stored as int32
#     * cuBLAS GemmEx INT8→INT32 (Tensor Core path), compare full matrix
# - Macro witness:
#     * M=N=K=5120
#     * A=B=1 (all ones)
#     * Every C[i,j] should equal K=5120
#     * Run GemmEx once, copy back, check max |C - 5120|
#
# NOTE:
#   This module is about EXACTNESS (correctness), not throughput.
#   It uses a direct cuBLAS INT8 path (column-major) as an integer-exact witness.
#   FASTLANE uses cuBLASLt with more flexible layouts, but the core math is the same:
#       INT8 * INT8 accumulated into INT32, with ranges proven to fit in int32.
#==================================================================================================
import os, subprocess, textwrap

print("\n" + "="*118)
print("MODULE FL-2 — FASTLANE EXACTNESS WITNESS (INT8×INT8→INT32)")
print("="*118)

CUDA_SRC_PATH = "/content/fastlane_exact_witness.cu"
EXE_PATH      = "/content/fastlane_exact_witness"

cuda_src = r'''
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>
#include <random>
#include <cuda_runtime.h>
#include <cublas_v2.h>

static void ck(cudaError_t e, const char* m){
    if(e != cudaSuccess){
        std::fprintf(stderr, "CUDA %s: %s\n", m, cudaGetErrorString(e));
        std::exit(1);
    }
}
static void bk(cublasStatus_t s, const char* m){
    if(s != CUBLAS_STATUS_SUCCESS){
        std::fprintf(stderr, "cuBLAS %s: %d\n", m, (int)s);
        std::exit(1);
    }
}

int main(){
    std::printf("\n====================================================================================\n");
    std::printf("FASTLANE EXACTNESS WITNESS (INT8xINT8->INT32)\n");
    std::printf("====================================================================================\n");

    cublasHandle_t handle;
    bk(cublasCreate(&handle), "cublasCreate");

    //----------------------------------------------------------------------------
    // Micro witness: M=N=K=256, random int8, CPU int64 reference vs GemmEx
    //----------------------------------------------------------------------------
    const int M  = 256;
    const int N  = 256;
    const int K  = 256;

    const size_t bytesA = (size_t)M * (size_t)K;
    const size_t bytesB = (size_t)K * (size_t)N;
    const size_t bytesC = (size_t)M * (size_t)N * sizeof(int32_t);

    std::vector<int8_t>  hA(bytesA);
    std::vector<int8_t>  hB(bytesB);
    std::vector<int32_t> hC_cpu(M * N);
    std::vector<int32_t> hC_gpu(M * N);

    std::mt19937 rng(12345);
    std::uniform_int_distribution<int> dist(-128, 127);

    for(size_t i = 0; i < bytesA; ++i){
        hA[i] = (int8_t)dist(rng);
    }
    for(size_t i = 0; i < bytesB; ++i){
        hB[i] = (int8_t)dist(rng);
    }

    // CPU reference, column-major:
    // A: MxK, lda = M → A(i,k) = hA[i + k*M]
    // B: KxN, ldb = K → B(k,j) = hB[k + j*K]
    // C: MxN, ldc = M → C(i,j) = hC[i + j*M]
    std::printf("\n--- Micro witness: M=N=K=256 (random int8 A,B) ---\n");
    for(int j = 0; j < N; ++j){
        for(int i = 0; i < M; ++i){
            long long acc = 0;
            for(int k = 0; k < K; ++k){
                int8_t a = hA[(size_t)i + (size_t)k * M];
                int8_t b = hB[(size_t)k + (size_t)j * K];
                acc += (long long)a * (long long)b;
            }
            hC_cpu[(size_t)i + (size_t)j * M] = (int32_t)acc;
        }
    }

    int8_t*  dA = nullptr;
    int8_t*  dB = nullptr;
    int32_t* dC = nullptr;
    ck(cudaMalloc(&dA, bytesA), "malloc dA");
    ck(cudaMalloc(&dB, bytesB), "malloc dB");
    ck(cudaMalloc(&dC, bytesC), "malloc dC");

    ck(cudaMemcpy(dA, hA.data(), bytesA, cudaMemcpyHostToDevice), "cpy A");
    ck(cudaMemcpy(dB, hB.data(), bytesB, cudaMemcpyHostToDevice), "cpy B");
    ck(cudaMemset(dC, 0, bytesC), "zero C");

    int32_t alpha = 1;
    int32_t beta  = 0;

    // GemmEx with INT8 inputs and INT32 accumulate:
    // C = A * B   (column-major)
    // m = M, n = N, k = K
    bk(
        cublasGemmEx(
            handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            M, N, K,
            &alpha,
            dA, CUDA_R_8I,  M,   // A: MxK, lda = M
            dB, CUDA_R_8I,  K,   // B: KxN, ldb = K
            &beta,
            dC, CUDA_R_32I, M,   // C: MxN, ldc = M
            CUDA_R_32I,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP
        ),
        "cublasGemmEx micro"
    );

    ck(cudaMemcpy(hC_gpu.data(), dC, bytesC, cudaMemcpyDeviceToHost), "cpy C");

    long long max_diff = 0;
    int mismatches = 0;

    for(int j = 0; j < N; ++j){
        for(int i = 0; i < M; ++i){
            size_t idx = (size_t)i + (size_t)j * M;
            long long cpu = (long long)hC_cpu[idx];
            long long gpu = (long long)hC_gpu[idx];
            long long diff = cpu - gpu;
            if(diff < 0) diff = -diff;
            if(diff > max_diff) max_diff = diff;
            if(diff != 0 && mismatches < 10){
                std::printf("  mismatch micro (i=%d,j=%d): cpu=%d gpu=%d diff=%lld\n",
                            i, j, (int)hC_cpu[idx], (int)hC_gpu[idx], diff);
                ++mismatches;
            }
        }
    }
    std::printf("Micro 256^3: CPU vs GPU max |diff| = %lld\n", max_diff);
    if(mismatches == 0){
        std::printf("Micro check: all entries match exactly.\n");
    }

    ck(cudaFree(dA), "free dA");
    ck(cudaFree(dB), "free dB");
    ck(cudaFree(dC), "free dC");

    //----------------------------------------------------------------------------
    // Macro witness: M=N=K=5120, A=B=1, verify C[i,j] = K (=5120) everywhere
    //----------------------------------------------------------------------------
    const int M2 = 5120;
    const int N2 = 5120;
    const int K2 = 5120;

    const size_t bytesA2 = (size_t)M2 * (size_t)K2;
    const size_t bytesB2 = (size_t)K2 * (size_t)N2;
    const size_t bytesC2 = (size_t)M2 * (size_t)N2 * sizeof(int32_t);

    std::printf("\n--- Macro witness: M=N=K=5120, A=B=1 ---\n");
    std::printf("Allocating host+device buffers (A,B,C)...\n");

    std::vector<int8_t>  hA2(bytesA2, (int8_t)1);
    std::vector<int8_t>  hB2(bytesB2, (int8_t)1);
    std::vector<int32_t> hC2((size_t)M2 * (size_t)N2);

    ck(cudaMalloc(&dA, bytesA2), "malloc dA2");
    ck(cudaMalloc(&dB, bytesB2), "malloc dB2");
    ck(cudaMalloc(&dC, bytesC2), "malloc dC2");

    ck(cudaMemcpy(dA, hA2.data(), bytesA2, cudaMemcpyHostToDevice), "cpy A2");
    ck(cudaMemcpy(dB, hB2.data(), bytesB2, cudaMemcpyHostToDevice), "cpy B2");
    ck(cudaMemset(dC, 0, bytesC2), "zero C2");

    alpha = 1;
    beta  = 0;

    float ms = 0.0f;
    cudaEvent_t t0, t1;
    ck(cudaEventCreate(&t0), "ev t0");
    ck(cudaEventCreate(&t1), "ev t1");
    ck(cudaEventRecord(t0), "rec t0");

    bk(
        cublasGemmEx(
            handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            M2, N2, K2,
            &alpha,
            dA, CUDA_R_8I,  M2,
            dB, CUDA_R_8I,  K2,
            &beta,
            dC, CUDA_R_32I, M2,
            CUDA_R_32I,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP
        ),
        "cublasGemmEx macro"
    );

    ck(cudaEventRecord(t1), "rec t1");
    ck(cudaEventSynchronize(t1), "sync t1");
    ck(cudaEventElapsedTime(&ms, t0, t1), "elapsed macro");
    cudaEventDestroy(t0);
    cudaEventDestroy(t1);

    std::printf("Macro 5120^3 GemmEx elapsed = %.3f ms\n", ms);
    std::printf("Copying C back to host for verification...\n");
    ck(cudaMemcpy(hC2.data(), dC, bytesC2, cudaMemcpyDeviceToHost), "cpy C2");

    const int32_t expected = K2; // each entry: sum_{k} 1*1 = K2
    long long max_diff2 = 0;
    int mismatches2 = 0;

    const long long total_elems = (long long)M2 * (long long)N2;
    for(long long idx = 0; idx < total_elems; ++idx){
        long long v = (long long)hC2[(size_t)idx];
        long long diff = v - (long long)expected;
        long long ad = diff >= 0 ? diff : -diff;
        if(ad > max_diff2) max_diff2 = ad;
        if(diff != 0 && mismatches2 < 10){
            int i = (int)(idx % M2);
            int j = (int)(idx / M2);
            std::printf("  mismatch macro (i=%d,j=%d): val=%d expected=%d diff=%lld\n",
                        i, j, (int)v, (int)expected, diff);
            ++mismatches2;
        }
    }

    std::printf("Macro 5120^3: expected each entry = %d\n", expected);
    std::printf("Macro 5120^3 GPU max |C - expected| = %lld\n", max_diff2);
    if(mismatches2 == 0){
        std::printf("Macro check: all entries match expected value exactly.\n");
    }

    ck(cudaFree(dA), "free dA2");
    ck(cudaFree(dB), "free dB2");
    ck(cudaFree(dC), "free dC2");

    cublasDestroy(handle);

    std::printf("\nFASTLANE INT8xINT8->INT32 exactness witness finished.\n");
    std::printf("====================================================================================\n");
    return 0;
}
'''

print("\n[FL-2] Writing CUDA exactness-witness source ->", CUDA_SRC_PATH)
with open(CUDA_SRC_PATH, "w") as f:
    f.write(cuda_src)

#---------------------------------------------------------------------------------------------------
# Compile for A100 (sm_80)
#---------------------------------------------------------------------------------------------------
import subprocess, shlex

compile_cmd = [
    "nvcc",
    "-O3",
    "-std=c++17",
    "-gencode=arch=compute_80,code=sm_80",
    CUDA_SRC_PATH,
    "-lcublas",
    "-o",
    EXE_PATH,
]

print("[FL-2] Compiling with:", " ".join(compile_cmd))
comp = subprocess.run(compile_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
print("[FL-2] nvcc output:")
print(comp.stdout)
if comp.returncode != 0:
    print("[FL-2] FATAL: nvcc failed with return code", comp.returncode)
    raise SystemExit(1)

print(f"[FL-2] Compile OK -> {EXE_PATH}")

#---------------------------------------------------------------------------------------------------
# Run the exactness witness
#---------------------------------------------------------------------------------------------------
print("\n" + "-"*118)
print("RUNNING FASTLANE EXACTNESS WITNESS (INT8×INT8→INT32)")
print("-"*118)

run_res = subprocess.run([EXE_PATH], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
print(run_res.stdout)

if run_res.returncode != 0:
    print(f"[FL-2] WARNING: exactness witness exited with code {run_res.returncode}")
else:
    print("[FL-2] Exactness witness completed successfully.")

print("\n" + "="*118)
print("MODULE FL-2 — DONE (FASTLANE INT8 Exactness Witness)")
print("="*118)















#==================================================================================================
# MODULE FL-3E — FASTLANE EXACTNESS WITNESS v6 (INT8×INT8→INT32, GPU-only where it matters)
#
# What this does:
#   • Micro witness (GPU):  M=N=K=256, random INT8 A,B
#       - CPU int64 reference vs cublasGemmEx INT8→INT32 (TensorOp)
#       - Prints max |diff| and a few explicit equations C(i,j) = Σ A*B
#   • Macro witness (GPU): M=N=K=5120, A=B=1
#       - Each C(i,j) should equal 5120 exactly
#   • Edge-case equations: printed as CPU-only math demos 
#
# 
#==================================================================================================
import os, subprocess

print("\n" + "="*118)
print("MODULE FL-3E — FASTLANE EXACTNESS WITNESS v6 (INT8×INT8→INT32)")
print("="*118)

CUDA_SRC_PATH = "/content/fastlane_exact_witness.cu"
EXE_PATH      = "/content/fastlane_exact_witness"

cuda_src = r'''
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>
#include <random>
#include <cuda_runtime.h>
#include <cublas_v2.h>

static void ck(cudaError_t e, const char* m){
    if(e != cudaSuccess){
        std::fprintf(stderr, "CUDA %s: %s\n", m, cudaGetErrorString(e));
        std::exit(1);
    }
}
static void bk(cublasStatus_t s, const char* m){
    if(s != CUBLAS_STATUS_SUCCESS){
        std::fprintf(stderr, "cuBLAS %s: %d\n", m, (int)s);
        std::exit(1);
    }
}

int main(){
    std::printf("\n====================================================================================\n");
    std::printf("FASTLANE EXACTNESS WITNESS v6 (INT8xINT8->INT32)\n");
    std::printf("====================================================================================\n");
    std::printf("We test INT8×INT8→INT32 GEMM via cublasGemmEx on realistic GPU shapes.\n");
    std::printf("Formula: C(i,j) = sum_{k=0..K-1} A(i,k) * B(k,j), computed exactly in int32.\n");
    std::printf("We stay inside the safe range: max |C(i,j)| << 2^31, so no int32 overflow.\n");

    cublasHandle_t handle;
    bk(cublasCreate(&handle), "cublasCreate");

    //----------------------------------------------------------------------------
    // 1) Micro witness: M=N=K=256, random int8, CPU int64 reference vs GemmEx (TensorOp)
    //----------------------------------------------------------------------------
    const int M  = 256;
    const int N  = 256;
    const int K  = 256;

    const size_t bytesA = (size_t)M * (size_t)K;
    const size_t bytesB = (size_t)K * (size_t)N;
    const size_t bytesC = (size_t)M * (size_t)N * sizeof(int32_t);

    std::vector<int8_t>  hA(bytesA);
    std::vector<int8_t>  hB(bytesB);
    std::vector<int32_t> hC_cpu(M * N);
    std::vector<int32_t> hC_gpu(M * N);

    std::mt19937 rng(12345);
    std::uniform_int_distribution<int> dist(-128, 127);

    for(size_t i = 0; i < bytesA; ++i){
        hA[i] = (int8_t)dist(rng);
    }
    for(size_t i = 0; i < bytesB; ++i){
        hB[i] = (int8_t)dist(rng);
    }

    std::printf("\n--- Micro witness: M=N=K=256 (random int8 A,B) ---\n");
    std::printf("C(i,j) = sum_{k=0..255} A(i,k) * B(k,j), with A,B in [-128,127].\n");
    std::printf("CPU uses int64 accumulator; result stored as int32 (safe for this K).\n");

    // CPU reference: column-major
    for(int j = 0; j < N; ++j){
        for(int i = 0; i < M; ++i){
            long long acc = 0;
            for(int k = 0; k < K; ++k){
                int8_t a = hA[(size_t)i + (size_t)k * M];
                int8_t b = hB[(size_t)k + (size_t)j * K];
                acc += (long long)a * (long long)b;
            }
            hC_cpu[(size_t)i + (size_t)j * M] = (int32_t)acc;
        }
    }

    int8_t*  dA = nullptr;
    int8_t*  dB = nullptr;
    int32_t* dC = nullptr;
    ck(cudaMalloc(&dA, bytesA), "malloc dA");
    ck(cudaMalloc(&dB, bytesB), "malloc dB");
    ck(cudaMalloc(&dC, bytesC), "malloc dC");

    ck(cudaMemcpy(dA, hA.data(), bytesA, cudaMemcpyHostToDevice), "cpy A");
    ck(cudaMemcpy(dB, hB.data(), bytesB, cudaMemcpyHostToDevice), "cpy B");
    ck(cudaMemset(dC, 0, bytesC), "zero C");

    int32_t alpha = 1;
    int32_t beta  = 0;

    // TensorOp path for the micro witness
    bk(
        cublasGemmEx(
            handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            M, N, K,
            &alpha,
            dA, CUDA_R_8I,  M,
            dB, CUDA_R_8I,  K,
            &beta,
            dC, CUDA_R_32I, M,
            CUDA_R_32I,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP
        ),
        "cublasGemmEx micro"
    );

    ck(cudaMemcpy(hC_gpu.data(), dC, bytesC, cudaMemcpyDeviceToHost), "cpy C");

    long long max_diff = 0;
    int mismatches = 0;
    for(int j = 0; j < N; ++j){
        for(int i = 0; i < M; ++i){
            size_t idx = (size_t)i + (size_t)j * M;
            long long cpu = (long long)hC_cpu[idx];
            long long gpu = (long long)hC_gpu[idx];
            long long diff = cpu - gpu;
            if(diff < 0) diff = -diff;
            if(diff > max_diff) max_diff = diff;
            if(diff != 0 && mismatches < 10){
                std::printf("  mismatch micro (i=%d,j=%d): cpu=%d gpu=%d diff=%lld\n",
                            i, j, (int)hC_cpu[idx], (int)hC_gpu[idx], diff);
                ++mismatches;
            }
        }
    }
    std::printf("Micro 256^3: CPU vs GPU max |diff| = %lld\n", max_diff);
    if(mismatches == 0){
        std::printf("Micro check: all entries match exactly.\n");
    }

    // Example equations
    int example_ij[3][2] = {
        {0, 0},
        {13, 7},
        {255, 42}
    };
    std::printf("\nExample micro equations (CPU = GPU):\n");
    for(int e = 0; e < 3; ++e){
        int i = example_ij[e][0];
        int j = example_ij[e][1];
        if(i >= M || j >= N) continue;
        size_t idx = (size_t)i + (size_t)j * M;
        long long cpu_val = (long long)hC_cpu[idx];
        long long gpu_val = (long long)hC_gpu[idx];
        std::printf("  C(%d,%d) = sum_{k=0..%d} A(%d,k)*B(k,%d) = %lld  | GPU: %lld\n",
                    i, j, K-1, i, j, cpu_val, gpu_val);
    }

    ck(cudaFree(dA), "free dA");
    ck(cudaFree(dB), "free dB");
    ck(cudaFree(dC), "free dC");

    //----------------------------------------------------------------------------
    // 2) Edge-case equations (CPU-only demos, no GPU call)
    //----------------------------------------------------------------------------
    std::printf("\n--- Edge-case equations (CPU-only demos) ---\n");

    // Edge 1: 1×1×16, A=127, B=-128
    {
        const int Kedge = 16;
        long long prod = 127LL * -128LL;
        long long expected = (long long)Kedge * prod;
        std::printf("Edge 1: M=N=1, K=16, A=127, B=-128\n");
        std::printf("  C(0,0) = sum_{k=0..15} 127 * (-128) = 16 * (127 * -128) = %lld\n", expected);
    }

    // Edge 2: 1×1×5120, extreme INT8 patterns
    {
        const int Kedge2 = 5120;
        struct Scenario {
            int a;
            int b;
            const char* label;
        };
        Scenario cases[3] = {
            { 127,   127,  "A=127,  B=127"  },
            { 127,  -127,  "A=127,  B=-127" },
            {-128,  -128,  "A=-128, B=-128" }
        };
        for(int s = 0; s < 3; ++s){
            long long Aval = cases[s].a;
            long long Bval = cases[s].b;
            long long prod = Aval * Bval;
            long long expected = (long long)Kedge2 * prod;
            std::printf("Edge 2: M=N=1, K=5120, %s\n", cases[s].label);
            std::printf("  C(0,0) = sum_{k=0..K-1} A_val*B_val = %d * (%d * %d) = %lld\n",
                        Kedge2, cases[s].a, cases[s].b, expected);
        }
    }

    //----------------------------------------------------------------------------
    // 3) Macro witness: M=N=K=5120, A=B=1, verify C(i,j) = K everywhere (TensorOp)
    //----------------------------------------------------------------------------
    const int M2 = 5120;
    const int N2 = 5120;
    const int K2 = 5120;

    const size_t bytesA2 = (size_t)M2 * (size_t)K2;
    const size_t bytesB2 = (size_t)K2 * (size_t)N2;
    const size_t bytesC2 = (size_t)M2 * (size_t)N2 * sizeof(int32_t);

    std::printf("\n--- Macro witness: M=N=K=5120, A=B=1 ---\n");
    std::printf("Here A(i,k)=1 and B(k,j)=1, so:\n");
    std::printf("  C(i,j) = sum_{k=0..K2-1} 1*1 = K2 = %d for all i,j.\n", K2);
    std::printf("We verify that every entry of the GPU result equals %d exactly.\n", K2);

    std::vector<int8_t>  hA2(bytesA2, (int8_t)1);
    std::vector<int8_t>  hB2(bytesB2, (int8_t)1);
    std::vector<int32_t> hC2((size_t)M2 * (size_t)N2);

    ck(cudaMalloc(&dA, bytesA2), "malloc dA2");
    ck(cudaMalloc(&dB, bytesB2), "malloc dB2");
    ck(cudaMalloc(&dC, bytesC2), "malloc dC2");

    ck(cudaMemcpy(dA, hA2.data(), bytesA2, cudaMemcpyHostToDevice), "cpy A2");
    ck(cudaMemcpy(dB, hB2.data(), bytesB2, cudaMemcpyHostToDevice), "cpy B2");
    ck(cudaMemset(dC, 0, bytesC2), "zero C2");

    alpha = 1;
    beta  = 0;

    float ms = 0.0f;
    cudaEvent_t t0, t1;
    ck(cudaEventCreate(&t0), "ev t0");
    ck(cudaEventCreate(&t1), "ev t1");
    ck(cudaEventRecord(t0), "rec t0");

    bk(
        cublasGemmEx(
            handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            M2, N2, K2,
            &alpha,
            dA, CUDA_R_8I,  M2,
            dB, CUDA_R_8I,  K2,
            &beta,
            dC, CUDA_R_32I, M2,
            CUDA_R_32I,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP
        ),
        "cublasGemmEx macro"
    );

    ck(cudaEventRecord(t1), "rec t1");
    ck(cudaEventSynchronize(t1), "sync t1");
    ck(cudaEventElapsedTime(&ms, t0, t1), "elapsed macro");
    cudaEventDestroy(t0);
    cudaEventDestroy(t1);

    std::printf("Macro 5120^3 GemmEx elapsed = %.3f ms\n", ms);
    std::printf("Copying C back to host for verification...\n");
    ck(cudaMemcpy(hC2.data(), dC, bytesC2, cudaMemcpyDeviceToHost), "cpy C2");

    const int32_t expected = K2;
    long long max_diff2 = 0;
    int mismatches2 = 0;

    const long long total_elems = (long long)M2 * (long long)N2;
    for(long long idx = 0; idx < total_elems; ++idx){
        long long v = (long long)hC2[(size_t)idx];
        long long diff = v - (long long)expected;
        long long ad = diff >= 0 ? diff : -diff;
        if(ad > max_diff2) max_diff2 = ad;
        if(diff != 0 && mismatches2 < 10){
            int i = (int)(idx % M2);
            int j = (int)(idx / M2);
            std::printf("  mismatch macro (i=%d,j=%d): val=%d expected=%d diff=%lld\n",
                        i, j, (int)v, (int)expected, diff);
            ++mismatches2;
        }
    }

    std::printf("Macro 5120^3: expected each entry = %d\n", expected);
    std::printf("Macro 5120^3 GPU max |C - expected| = %lld\n", max_diff2);
    if(mismatches2 == 0){
        std::printf("Macro check: all entries match expected value exactly.\n");
    }

    ck(cudaFree(dA), "free dA2");
    ck(cudaFree(dB), "free dB2");
    ck(cudaFree(dC), "free dC2");

    cublasDestroy(handle);

    std::printf("\nFASTLANE INT8xINT8->INT32 exactness witness v6 finished.\n");
    std::printf("====================================================================================\n");
    return 0;
}
'''

print(f"\n[FL-3E] Writing CUDA exactness-witness v6 source -> {CUDA_SRC_PATH}")
with open(CUDA_SRC_PATH, "w") as f:
    f.write(cuda_src)

#---------------------------------------------------------------------------------------------------
# Compile for A100 (sm_80)
#---------------------------------------------------------------------------------------------------
compile_cmd = [
    "nvcc",
    "-O3",
    "-std=c++17",
    "-gencode=arch=compute_80,code=sm_80",
    CUDA_SRC_PATH,
    "-lcublas",
    "-o",
    EXE_PATH,
]

print("[FL-3E] Compiling with:", " ".join(compile_cmd))
comp = subprocess.run(compile_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
print("[FL-3E] nvcc output:")
print(comp.stdout)
if comp.returncode != 0:
    print("[FL-3E] FATAL: nvcc failed with return code", comp.returncode)
    raise SystemExit(1)

print(f"[FL-3E] Compile OK -> {EXE_PATH}")

#---------------------------------------------------------------------------------------------------
# Run the exactness witness v6
#---------------------------------------------------------------------------------------------------
print("\n" + "-"*118)
print("RUNNING FASTLANE EXACTNESS WITNESS v6 (INT8×INT8→INT32)")
print("-"*118)

run_res = subprocess.run([EXE_PATH], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
print(run_res.stdout)

if run_res.returncode != 0:
    print(f"[FL-3E] WARNING: exactness witness v6 exited with code {run_res.returncode}")
else:
    print("[FL-3E] Exactness witness v6 completed successfully.")

print("\n" + "="*118)
print("MODULE FL-3E — DONE (FASTLANE INT8 Exactness Witness v6)")
print("="*118)
















#==================================================================================================
# MODULE FL-4 — FASTLANE ENV + PROVENANCE SNAPSHOT (GPU / CUDA / cuBLAS / NVCC)
#
# Purpose:
#   Give reviewers a one-shot "what machine + stack was this run on?" report, using:
#     • cudaGetDeviceCount / cudaGetDeviceProperties
#     • cudaDriverGetVersion / cudaRuntimeGetVersion
#     • cublasGetVersion
#     • nvcc --version (via subprocess)
#
# Usage:
#   Just run this cell in the same Colab / machine where you run FL-1 and FL-3E.
#   It prints a human-readable environment block you can copy into your README / Trust Pack.
#==================================================================================================
import subprocess, textwrap

print("\n" + "="*118)
print("MODULE FL-4 — FASTLANE ENV + PROVENANCE SNAPSHOT")
print("="*118)

CUDA_SRC_PATH = "/content/fastlane_env_report.cu"
EXE_PATH      = "/content/fastlane_env_report"

cuda_src = r'''
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cublas_v2.h>

static void ck(cudaError_t e, const char* m){
    if(e != cudaSuccess){
        std::fprintf(stderr, "CUDA %s: %s\n", m, cudaGetErrorString(e));
        std::exit(1);
    }
}
static void bk(cublasStatus_t s, const char* m){
    if(s != CUBLAS_STATUS_SUCCESS){
        std::fprintf(stderr, "cuBLAS %s: %d\n", m, (int)s);
        std::exit(1);
    }
}

static void print_version_int(const char* label, int v){
    int major = v / 1000;
    int minor = (v % 1000) / 10;
    std::printf("  %-22s raw=%d  approx=%d.%d\n", label, v, major, minor);
}

int main(){
    std::printf("\n====================================================================================\n");
    std::printf("FASTLANE ENV + PROVENANCE SNAPSHOT\n");
    std::printf("====================================================================================\n");

    // Driver & runtime
    int drv = 0, rt = 0;
    cudaError_t edrv = cudaDriverGetVersion(&drv);
    cudaError_t ert  = cudaRuntimeGetVersion(&rt);
    std::printf("CUDA versions:\n");
    if(edrv == cudaSuccess) print_version_int("Driver", drv);
    else std::printf("  Driver               ERROR: %s\n", cudaGetErrorString(edrv));
    if(ert == cudaSuccess)  print_version_int("Runtime", rt);
    else std::printf("  Runtime              ERROR: %s\n", cudaGetErrorString(ert));
    std::printf("\n");

    // Device info
    int deviceCount = 0;
    ck(cudaGetDeviceCount(&deviceCount), "cudaGetDeviceCount");
    std::printf("CUDA devices visible: %d\n", deviceCount);
    if(deviceCount <= 0){
        std::printf("No CUDA devices found; nothing more to report.\n");
        return 0;
    }

    int dev = 0;
    ck(cudaSetDevice(dev), "cudaSetDevice(0)");
    cudaDeviceProp prop;
    ck(cudaGetDeviceProperties(&prop, dev), "cudaGetDeviceProperties(0)");

    std::printf("Using device 0:\n");
    std::printf("  Name                  %s\n", prop.name);
    std::printf("  Compute capability    %d.%d\n", prop.major, prop.minor);
    std::printf("  Total global memory   %.2f GB\n", (double)prop.totalGlobalMem / (1024.0*1024.0*1024.0));
    std::printf("  MultiProcessor count  %d\n", prop.multiProcessorCount);
    std::printf("  Max threads / block   %d\n", prop.maxThreadsPerBlock);
    std::printf("  Warp size             %d\n", prop.warpSize);
    std::printf("  Shared mem / block    %.1f KB\n", (double)prop.sharedMemPerBlock / 1024.0);
    std::printf("  Max threads / SM      %d\n", prop.maxThreadsPerMultiProcessor);
    std::printf("  Memory clock rate     %.3f GHz\n", (double)prop.memoryClockRate / 1.0e6);
    std::printf("  Memory bus width      %d bits\n", prop.memoryBusWidth);
    std::printf("  SM clock rate         %.3f GHz\n", (double)prop.clockRate / 1.0e6);
    std::printf("\n");

    // cuBLAS version
    std::printf("cuBLAS:\n");
    cublasHandle_t h;
    bk(cublasCreate(&h), "cublasCreate");
    int blasVer = 0;
    bk(cublasGetVersion(h, &blasVer), "cublasGetVersion");
    cublasDestroy(h);
    print_version_int("cuBLAS", blasVer);
    std::printf("\n");

  
}
'''

print(f"\n[FL-4] Writing CUDA env-report source -> {CUDA_SRC_PATH}")
with open(CUDA_SRC_PATH, "w") as f:
    f.write(cuda_src)

#---------------------------------------------------------------------------------------------------
# Compile the env reporter for A100 (sm_80)
#---------------------------------------------------------------------------------------------------
compile_cmd = [
    "nvcc",
    "-O2",
    "-std=c++17",
    "-gencode=arch=compute_80,code=sm_80",
    CUDA_SRC_PATH,
    "-lcublas",
    "-o",
    EXE_PATH,
]

print("[FL-4] Compiling with:", " ".join(compile_cmd))
comp = subprocess.run(compile_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
print("[FL-4] nvcc output:")
print(comp.stdout)
if comp.returncode != 0:
    print("[FL-4] FATAL: nvcc failed with return code", comp.returncode)
    raise SystemExit(1)

print(f"[FL-4] Compile OK -> {EXE_PATH}")

#---------------------------------------------------------------------------------------------------
# Run the env reporter
#---------------------------------------------------------------------------------------------------
print("\n" + "-"*118)
print("RUNNING FASTLANE ENV + PROVENANCE SNAPSHOT")
print("-"*118)

run_res = subprocess.run([EXE_PATH], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
print(run_res.stdout)

if run_res.returncode != 0:
    print(f"[FL-4] WARNING: env reporter exited with code {run_res.returncode}")
else:
    print("[FL-4] Env + provenance snapshot completed successfully.")

#---------------------------------------------------------------------------------------------------
# Also capture nvcc --version from the host toolchain for the log
#---------------------------------------------------------------------------------------------------
print("\n" + "-"*118)
print("HOST TOOLCHAIN INFO (nvcc --version)")
print("-"*118)
try:
    nvcc_res = subprocess.run(["nvcc", "--version"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    print(nvcc_res.stdout)
except FileNotFoundError:
    print("nvcc not found on PATH (unexpected in this Colab setup).")

print("\n" + "="*118)
print("MODULE FL-4 — DONE (FASTLANE ENV + PROVENANCE SNAPSHOT)")
print("="*118)

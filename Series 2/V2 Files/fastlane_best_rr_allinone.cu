// fastlane_best_rr_allinone.cu
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

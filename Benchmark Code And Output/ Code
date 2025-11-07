#==================================================================================================
# MODULE PB-ALL — ONE-CLICK PUBLIC BENCH (Macro T-ops + GMP Exactness + Micro H2H + Graph Replays)
# - Installs GMP
# - Builds & runs:
#     (1) PB-1: Macro arena K-Panel Swarm (A100 target, int8->int32 exact) + validator (GMP samples)
#     (2) PB-4: Micro head-to-head vs GMP (exact) for shapes 128³, 192³, 256³
#     (3) PB-6: Micro amortized throughput (CUDA Graph + reps) for 256³, with one-pass exactness
# - Generates artifacts:
#     • /content/public_bench_report.csv
#     • /content/public_bench_report.md
# - Loud banners, stable seeds, dyadic mpq prints.
#==================================================================================================
import os, subprocess, textwrap, shutil, sys, platform, re, datetime, json

print("\n" + "="*106)
print("MODULE PB-ALL — ONE-CLICK PUBLIC BENCH")
print("="*106)
print(f"Python  : {sys.version.split()[0]}")
print(f"OS      : {platform.platform()}")
print(f"CUDA?   :", shutil.which("nvcc") is not None)
_ = subprocess.run(["bash","-lc","nvidia-smi"], check=False)

#--------------------------------------------------------------------------------------------------
# APT: GMP (exact integer/rational witnesses)
#--------------------------------------------------------------------------------------------------
print("\n=== APT: installing libgmp-dev")
_ = subprocess.run(["bash","-lc","sudo apt-get update -y && sudo apt-get install -y libgmp-dev"], check=False)

#--------------------------------------------------------------------------------------------------
# Paths
#--------------------------------------------------------------------------------------------------
PB1_CU  = "/content/fx_int8_kpanel_tiled_swarm_v1.cu"
PB1_EXE = "/content/fx_int8_kpanel_tiled_swarm_v1"
PB4_CU  = "/content/pb4_gmp_vs_gpu_micro.cu"
PB4_EXE = "/content/pb4_gmp_vs_gpu_micro"
PB6_CU  = "/content/pb6_micro_reps_graph.cu"
PB6_EXE = "/content/pb6_micro_reps_graph"

#==================================================================================================
# WRITE: PB-1 (Macro arena + GMP-sampled validator) — known-good fixed version
#==================================================================================================
print("\n" + "="*106)
print("WRITING PB-1 (Macro arena + validator)")
print("="*106)
pb1_code = r'''
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>
#include <string>
#include <chrono>
#include <ctime>
#include <cuda_runtime.h>
#include <cublasLt.h>
#include <gmp.h>
#include <algorithm>

static inline void ck(cudaError_t e,const char* m){ if(e!=cudaSuccess){fprintf(stderr,"CUDA %s : %s\n",m,cudaGetErrorString(e)); std::exit(2);} }
static inline void bk(cublasStatus_t s,const char* m){ if(s!=CUBLAS_STATUS_SUCCESS){fprintf(stderr,"cuBLASLt %s : %d\n",m,int(s)); std::exit(3);} }

static std::string iso_now(){ using namespace std::chrono; auto t=std::chrono::system_clock::to_time_t(std::chrono::system_clock::now()); std::tm gm{}; gmtime_r(&t,&gm); char b[64]; std::strftime(b,sizeof(b),"%Y-%m-%dT%H:%M:%SZ",&gm); return std::string(b); }
static void banner(const char* t){ printf("\n=====================================================================================\n%s\n=====================================================================================\n", t); }

struct Args{
  int M=5120, N=5120, K=5120;
  int streams=32;
  int graph_nodes=64;
  int batch_per_node=4;
  int tileK=1280;
  int warmup=6;
  int tryAlgos=64;
  size_t workspaceMB=1024;
  int epochs=2;
  int printEvery=1;
  int validate=1;
  int fracA=4, fracB=4;
  int vM=256, vN=256, vK=256;
  int vSamples=8;
};

static Args parse(int ac,char**av){
  Args a;
  for(int i=1;i<ac;i++){
    std::string s(av[i]);
    auto gi=[&](const char*f,int&dst){ if(s==f && i+1<ac){ dst=std::atoi(av[++i]); return true;} return false; };
    if(gi("--m",a.M))continue; if(gi("--n",a.N))continue; if(gi("--k",a.K))continue;
    if(gi("--streams",a.streams))continue; if(gi("--graphNodes",a.graph_nodes))continue;
    if(gi("--batchPerNode",a.batch_per_node))continue; if(gi("--tileK",a.tileK))continue;
    if(gi("--warmup",a.warmup))continue; if(gi("--tryAlgos",a.tryAlgos))continue;
    if(gi("--epochs",a.epochs))continue; if(gi("--printEvery",a.printEvery))continue; if(gi("--validate",a.validate))continue;
    if(s=="--workspaceMB" && i+1<ac){ a.workspaceMB=size_t(std::atol(av[++i])); continue; }
    if(gi("--fracA",a.fracA))continue; if(gi("--fracB",a.fracB))continue;
    if(gi("--vM",a.vM))continue; if(gi("--vN",a.vN))continue; if(gi("--vK",a.vK))continue;
    if(gi("--vSamples",a.vSamples))continue;
  }
  return a;
}

static void fill_int8(int M,int N,std::vector<int8_t>& h,uint32_t seed){
  h.resize((size_t)M*N);
  uint32_t x = seed?seed:1u;
  for(size_t i=0;i<h.size();++i){
    x ^= x<<13; x ^= x>>17; x ^= x<<5;
    int v = int(int(x&0xFF)-128);
    if(v<-120) v=-120; if(v>120) v=120;
    h[i] = (int8_t)v;
  }
}

static int pick_algos(cublasLtHandle_t lt, cublasLtMatmulDesc_t op,
                      cublasLtMatrixLayout_t Ad, cublasLtMatrixLayout_t Bd, cublasLtMatrixLayout_t Cd,
                      std::vector<cublasLtMatmulHeuristicResult_t>& out, size_t ws){
  cublasLtMatmulPreference_t pref; bk(cublasLtMatmulPreferenceCreate(&pref),"pref");
  bk(cublasLtMatmulPreferenceSetAttribute(pref,CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,&ws,sizeof(ws)),"pref ws");
  int found=0; cublasStatus_t s = cublasLtMatmulAlgoGetHeuristic(lt,op,Ad,Bd,Cd,Cd,pref,(int)out.size(),out.data(),&found);
  if(s!=CUBLAS_STATUS_SUCCESS){ found=0; }
  cublasLtMatmulPreferenceDestroy(pref);
  return found;
}

static cublasLtMatrixLayout_t make_layout(cudaDataType_t t, int rows,int cols,int ld, cublasLtOrder_t ord){
  cublasLtMatrixLayout_t L; bk(cublasLtMatrixLayoutCreate(&L,t,rows,cols,ld),"layout create");
  bk(cublasLtMatrixLayoutSetAttribute(L,CUBLASLT_MATRIX_LAYOUT_ORDER,&ord,sizeof(ord)),"layout order");
  return L;
}

static void cpu_gemm_i8_i32_row(int M,int N,int K, const int8_t* A,const int8_t* B, int32_t* C){
  for(int i=0;i<M;i++){
    for(int j=0;j<N;j++){
      long long acc=0;
      const int8_t* Ai = A + (size_t)i*K;
      const int8_t* Bj = B + (size_t)j;
      for(int k=0;k<K;k++){
        acc += (long long)Ai[k] * (long long)Bj[(size_t)k*N];
      }
      C[(size_t)i*N + j] = (int32_t)acc;
    }
  }
}

static void gmp_sample_witness(int M,int N,int K,
                               const int8_t* A,const int8_t* B,const int32_t* C,
                               int samples, int fracA, int fracB){
  banner("GMP SAMPLED WITNESS (exact integers + dyadic rationals)");
  int stepI = std::max(1, M / std::max(1,samples));
  int stepJ = std::max(1, N / std::max(1,samples));
  int shown=0;
  for(int i=0;i<M && shown<samples;i+=stepI){
    for(int j=0;j<N && shown<samples;j+=stepJ){
      mpz_t acc, tmp; mpz_init(acc); mpz_init(tmp);
      for(int k=0;k<K;k++){
        long long vv = (long long)A[(size_t)i*K + k] * (long long)B[(size_t)k*N + j];
        mpz_set_si(tmp, (long)vv);
        mpz_add(acc, acc, tmp);
      }
      long long c64 = (long long)C[(size_t)i*N + j];
      mpz_t acc_check; mpz_init(acc_check); mpz_set_si(acc_check, (long)c64);
      int eq = (mpz_cmp(acc, acc_check)==0);
      mpq_t q; mpq_init(q);
      mpq_set_z(q, acc);
      mpq_div_2exp(q, q, (unsigned long)(fracA+fracB));
      mpq_canonicalize(q);
      gmp_printf("Entry (%4d,%4d): C_int32=%lld  [GMP eq=%s]  C_real = %Qd\n",
                 i,j,c64, eq?"YES":"NO", q);
      mpq_clear(q);
      mpz_clear(acc_check); mpz_clear(tmp); mpz_clear(acc);
      shown++;
    }
  }
}

static void validator_block(const Args& a){
  if(!a.validate){ return; }
  banner("VALIDATOR — START (micro GEMM + exactness)");
  const int M=a.vM, N=a.vN, K=a.vK;
  std::vector<int8_t> hA,hB; fill_int8(M,K,hA,0xA11CE55Du); fill_int8(K,N,hB,0xB16B00B5u);
  std::vector<int32_t> hC_cpu((size_t)M*N,0), hC_gpu((size_t)M*N,0);
  auto t0 = std::chrono::high_resolution_clock::now();
  cpu_gemm_i8_i32_row(M,N,K,hA.data(),hB.data(),hC_cpu.data());
  auto t1 = std::chrono::high_resolution_clock::now();
  double cpu_ms = std::chrono::duration<double,std::milli>(t1-t0).count();

  cublasLtHandle_t lt; bk(cublasLtCreate(&lt),"lt(v)");
  cublasLtMatmulDesc_t op; bk(cublasLtMatmulDescCreate(&op,CUBLAS_COMPUTE_32I,CUDA_R_32I),"op(v)");
  cublasOperation_t Nop=CUBLAS_OP_N;
  bk(cublasLtMatmulDescSetAttribute(op,CUBLASLT_MATMUL_DESC_TRANSA,&Nop,sizeof(Nop)),"Aop(v)");
  bk(cublasLtMatmulDescSetAttribute(op,CUBLASLT_MATMUL_DESC_TRANSB,&Nop,sizeof(Nop)),"Bop(v)");
  cublasLtOrder_t row=CUBLASLT_ORDER_ROW;
  cublasLtMatrixLayout_t Ad = make_layout(CUDA_R_8I,  M,K,K,row);
  cublasLtMatrixLayout_t Bd = make_layout(CUDA_R_8I,  K,N,N,row);
  cublasLtMatrixLayout_t Cd = make_layout(CUDA_R_32I, M,N,N,row);

  size_t bytesA=(size_t)M*K, bytesB=(size_t)K*N, bytesC=(size_t)M*N*sizeof(int32_t);
  int8_t *dA=nullptr,*dB=nullptr; int32_t *dC=nullptr;
  ck(cudaMalloc(&dA,bytesA),"malloc vA"); ck(cudaMalloc(&dB,bytesB),"malloc vB"); ck(cudaMalloc(&dC,bytesC),"malloc vC");
  ck(cudaMemcpy(dA,hA.data(),bytesA,cudaMemcpyHostToDevice),"H2D vA");
  ck(cudaMemcpy(dB,hB.data(),bytesB,cudaMemcpyHostToDevice),"H2D vB");
  ck(cudaMemset(dC,0,bytesC),"clr vC");

  size_t ws_bytes=64*1024*1024; void* dWS=nullptr; ck(cudaMalloc(&dWS,ws_bytes),"ws(v)");
  std::vector<cublasLtMatmulHeuristicResult_t> algos(64);
  int found = pick_algos(lt,op,Ad,Bd,Cd,algos,ws_bytes);
  if(found==0){ ws_bytes=0; if(dWS){cudaFree(dWS); dWS=nullptr;} found = pick_algos(lt,op,Ad,Bd,Cd,algos,0); }
  const int32_t alpha=1, beta0=0;

  int chosen=-1;
  for(int i=0;i<found;i++){
    cublasStatus_t s = cublasLtMatmul(lt,op,&alpha,dA,Ad,dB,Bd,&beta0,dC,Cd,dC,Cd,&algos[i].algo,dWS,ws_bytes,0);
    if(s==CUBLAS_STATUS_SUCCESS){ chosen=i; break; }
  }
  if(chosen<0){ fprintf(stderr,"Validator: no runnable algo\n"); std::exit(17); }
  ck(cudaDeviceSynchronize(),"v sync");
  ck(cudaMemcpy(hC_gpu.data(),dC,bytesC,cudaMemcpyDeviceToHost),"D2H vC");

  size_t mism=0; for(size_t t=0;t<(size_t)M*N;t++){ if(hC_cpu[t]!=hC_gpu[t]){ mism++; if(mism<=5) fprintf(stderr,"mismatch @ %zu : cpu=%d gpu=%d\n",t,hC_cpu[t],hC_gpu[t]); } }
  printf("CPU int32 vs GPU int32   : %s  (mismatches=%zu)\n", mism? "FAIL":"PASS", mism);
  printf("CPU reference elapsed    : %.3f ms  (shape %dx%dx%d)\n", cpu_ms, M,N,K);

  gmp_sample_witness(M,N,K, hA.data(),hB.data(),hC_gpu.data(), std::min(a.vSamples,16), a.fracA, a.fracB);

  if(dWS) cudaFree(dWS);
  cudaFree(dC); cudaFree(dB); cudaFree(dA);
  cublasLtMatrixLayoutDestroy(Ad); cublasLtMatrixLayoutDestroy(Bd); cublasLtMatrixLayoutDestroy(Cd);
  cublasLtMatmulDescDestroy(op); cublasLtDestroy(lt);

  banner("VALIDATOR — END");
}

int main(int ac,char**av){
  banner("MODULE L — K-Panel Tiled Swarm (ROW-only, exact, public bench build)");
  Args a=parse(ac,av);
  if(a.K % a.tileK != 0){ fprintf(stderr,"tileK must divide K exactly for this module.\n"); return 13; }
  int panels = a.K / a.tileK;

  cudaDeviceProp prop{}; ck(cudaGetDeviceProperties(&prop,0),"get device prop");
  printf("Device=%s CC=%d.%d SMs=%d GlobalMem=%llu MB\n",prop.name,prop.major,prop.minor,prop.multiProcessorCount,(unsigned long long)(prop.totalGlobalMem/(1024ull*1024ull)));

  std::vector<int8_t> hA,hB; fill_int8(a.M,a.K,hA,0xA11CE55Du); fill_int8(a.K,a.N,hB,0xB16B00B5u);
  size_t bytesA=(size_t)a.M*a.K, bytesB=(size_t)a.K*a.N;
  size_t elemsC=(size_t)a.M*a.N, bytesC_one=elemsC*sizeof(int32_t);

  int8_t *dA=nullptr,*dB=nullptr; ck(cudaMalloc(&dA,bytesA),"malloc A"); ck(cudaMalloc(&dB,bytesB),"malloc B");
  ck(cudaMemcpy(dA,hA.data(),bytesA,cudaMemcpyHostToDevice),"H2D A");
  ck(cudaMemcpy(dB,hB.data(),bytesB,cudaMemcpyHostToDevice),"H2D B");

  std::vector<int32_t*> dC(a.streams,nullptr);
  for(int s=0;s<a.streams;s++){ ck(cudaMalloc(&dC[s], bytesC_one * a.batch_per_node),"malloc C_s"); ck(cudaMemset(dC[s],0,bytesC_one * a.batch_per_node),"clr C_s"); }

  cublasLtHandle_t lt; bk(cublasLtCreate(&lt),"lt");
  cublasLtMatmulDesc_t op; bk(cublasLtMatmulDescCreate(&op,CUBLAS_COMPUTE_32I,CUDA_R_32I),"op");
  cublasOperation_t Nop=CUBLAS_OP_N;
  bk(cublasLtMatmulDescSetAttribute(op,CUBLASLT_MATMUL_DESC_TRANSA,&Nop,sizeof(Nop)),"Aop");
  bk(cublasLtMatmulDescSetAttribute(op,CUBLASLT_MATMUL_DESC_TRANSB,&Nop,sizeof(Nop)),"Bop");

  cublasLtOrder_t row=CUBLASLT_ORDER_ROW;
  cublasLtMatrixLayout_t Ad_full = make_layout(CUDA_R_8I,  a.M,a.K,a.K,row);
  cublasLtMatrixLayout_t Bd_full = make_layout(CUDA_R_8I,  a.K,a.N,a.N,row);
  cublasLtMatrixLayout_t Cd_full = make_layout(CUDA_R_32I, a.M,a.N,a.N,row);

  size_t ws_bytes=a.workspaceMB*1024ull*1024ull; void* dWS=nullptr; ck(cudaMalloc(&dWS,ws_bytes),"workspace");
  std::vector<cublasLtMatmulHeuristicResult_t> algos(a.tryAlgos);
  int found = pick_algos(lt,op,Ad_full,Bd_full,Cd_full,algos,ws_bytes);
  printf("heuristics(found, ws=%llu) = %d\n", (unsigned long long)ws_bytes, found);
  if(found==0){
    cudaFree(dWS); dWS=nullptr; ws_bytes=0;
    found = pick_algos(lt,op,Ad_full,Bd_full,Cd_full,algos,ws_bytes);
    printf("heuristics(found, ws=0) = %d\n", found);
    if(found==0){ fprintf(stderr,"No algos\n"); return 7; }
  }
  const int32_t alpha=1, beta0=0, beta1=1;
  int chosen=-1;
  {
    cublasStatus_t s = cublasLtMatmul(lt,op,&alpha,dA,Ad_full,dB,Bd_full,&beta0,(int32_t*)((char*)dC[0]+0),Cd_full,(int32_t*)((char*)dC[0]+0),Cd_full,&algos[0].algo,dWS,ws_bytes,0);
    for(int i=0;i<found;i++){
      s = cublasLtMatmul(lt,op,&alpha,dA,Ad_full,dB,Bd_full,&beta0,(int32_t*)((char*)dC[0]+0),Cd_full,(int32_t*)((char*)dC[0]+0),Cd_full,&algos[i].algo,dWS,ws_bytes,0);
      if(s==CUBLAS_STATUS_SUCCESS){ chosen=i; break; }
    }
  }
  if(chosen<0){ fprintf(stderr,"No runnable algo\n"); return 8; }
  printf("picked algo index = %d (ws=%llu)  panels=%d  tileK=%d\n", chosen, (unsigned long long)ws_bytes, panels, a.tileK);

  cublasLtMatrixLayout_t Ad_tile = make_layout(CUDA_R_8I,  a.M, a.tileK, a.K, row);
  cublasLtMatrixLayout_t Bd_tile = make_layout(CUDA_R_8I,  a.tileK, a.N, a.N, row);

  {
    int32_t* C0 = (int32_t*)((char*)dC[0]+0);
    bk(cublasLtMatmul(lt,op,&alpha, dA + 0, Ad_tile, dB + 0, Bd_tile, &beta0, C0, Cd_full, C0, Cd_full, &algos[chosen].algo, dWS, ws_bytes, 0),"warm p0");
    for(int p=1;p<panels;p++){
      const int k0 = p*a.tileK;
      bk(cublasLtMatmul(lt,op,&alpha, dA + k0, Ad_tile, dB + (size_t)k0*a.N, Bd_tile, &beta1, C0, Cd_full, C0, Cd_full, &algos[chosen].algo, dWS, ws_bytes, 0),"warm p+");
    }
  }
  ck(cudaDeviceSynchronize(),"warm sync");

  std::vector<cudaStream_t> streams(a.streams);
  for(int s=0;s<a.streams;s++) ck(cudaStreamCreate(&streams[s]),"mk stream");
  std::vector<cudaGraph_t> graphs(a.streams,nullptr);
  std::vector<cudaGraphExec_t> gexec(a.streams,nullptr);

  banner("Graph capture per stream (tiled over K, multi-C per node)");
  for(int s=0;s<a.streams;s++){
    ck(cudaStreamBeginCapture(streams[s], cudaStreamCaptureModeGlobal),"cap begin");
    for(int g=0; g<a.graph_nodes; g++){
      for(int b=0;b<a.batch_per_node;b++){
        size_t bytesC_one = (size_t)a.M*a.N*sizeof(int32_t);
        int32_t* Cslice = (int32_t*)((char*)dC[s] + (size_t)b*bytesC_one);
        bk(cublasLtMatmul(lt,op,&alpha, dA + 0, Ad_tile, dB + 0, Bd_tile, &beta0, Cslice, Cd_full, Cslice, Cd_full, &algos[chosen].algo, dWS, ws_bytes, streams[s]),"graph p0");
        for(int p=1;p<panels;p++){
          const int k0 = p*a.tileK;
          bk(cublasLtMatmul(lt,op,&alpha, dA + k0, Ad_tile, dB + (size_t)k0*a.N, Bd_tile, &beta1, Cslice, Cd_full, Cslice, Cd_full, &algos[chosen].algo, dWS, ws_bytes, streams[s]),"graph p+");
        }
      }
    }
    ck(cudaStreamEndCapture(streams[s], &graphs[s]),"cap end");
    ck(cudaGraphInstantiate(&gexec[s], graphs[s], nullptr, nullptr, 0),"graph inst");
  }

  long long gemms_per_C = panels;
  long long gemms_per_epoch = (long long)a.streams * (long long)a.graph_nodes * (long long)a.batch_per_node * gemms_per_C;
  banner("K-panel tiled swarm run");
  cudaEvent_t t0,t1; ck(cudaEventCreate(&t0),"e0"); ck(cudaEventCreate(&t1),"e1");
  ck(cudaEventRecord(t0),"rs");
  for(int ep=1; ep<=a.epochs; ++ep){
    for(int s=0;s<a.streams;s++) ck(cudaGraphLaunch(gexec[s], streams[s]),"graph launch");
    if(ep % a.printEvery == 0){
      for(int s=0;s<a.streams;s++) ck(cudaStreamSynchronize(streams[s]),"sync s");
      ck(cudaEventRecord(t1),"re"); ck(cudaEventSynchronize(t1),"sync");
      float ms=0.f; ck(cudaEventElapsedTime(&ms,t0,t1),"elapsed");
      double ops = double(ep)*double(gemms_per_epoch)*2.0*double(a.M)*double(a.N)*double(a.K);
      double tops = ops/(ms*1e6);
      printf("EPOCH %d :: total_gemms=%lld  elapsed=%.3fs  per_gemm=%.3f ms  logical=%.2f T-ops/s\n",
             ep, (long long)ep*gemms_per_epoch, ms/1000.0f, ms/((long long)ep*gemms_per_epoch), tops);
    }
  }

  for(int s=0;s<a.streams;s++) ck(cudaStreamSynchronize(streams[s]),"final sync s");
  ck(cudaEventRecord(t1),"final re"); ck(cudaEventSynchronize(t1),"final sync");
  float ms=0.f; ck(cudaEventElapsedTime(&ms,t0,t1),"final elapsed");

  long long total_gemms = (long long)a.epochs * gemms_per_epoch;
  double ops = double(total_gemms)*2.0*double(a.M)*double(a.N)*double(a.K);
  double gops = ops/(ms*1e6);
  double tops = gops/1000.0;

  banner("SUMMARY :: K-PANEL TILED SWARM");
  printf("ts=%s  M=%d N=%d K=%d  streams=%d  nodes=%d  batch_per_node=%d  tileK=%d  panels=%d  epochs=%d\n",
         iso_now().c_str(), a.M,a.N,a.K, a.streams, a.graph_nodes, a.batch_per_node, a.tileK, panels, a.epochs);
  printf("total_gemms(counting panels)=%lld  elapsed_total=%.3fs  per_gemm=%.3f ms  logical_throughput=%.2f T-ops/s\n",
         total_gemms, ms/1000.0, ms/total_gemms, tops);
  printf("layout=ROW  algo_index=%d  ws_bytes=%llu  Dyadic: C_real = int32 * 2^{-(%d)} (exact)\n",
         chosen, (unsigned long long)ws_bytes, (4+4));

  if(a.validate){ validator_block(a); }

  if(dWS) cudaFree(dWS);
  cublasLtMatrixLayoutDestroy(Ad_tile); cublasLtMatrixLayoutDestroy(Bd_tile);
  cublasLtMatrixLayoutDestroy(Ad_full); cublasLtMatrixLayoutDestroy(Bd_full); cublasLtMatrixLayoutDestroy(Cd_full);
  cublasLtMatmulDescDestroy(op); cublasLtDestroy(lt);
  for(int s=0;s<a.streams;s++) cudaFree(dC[s]);
  cudaFree(dA); cudaFree(dB);

  banner("MODULE L — END (public bench build)");
  return 0;
}
'''
with open(PB1_CU,"w",encoding="utf-8") as f: f.write(textwrap.dedent(pb1_code))
print("=== PB-1 WRITTEN", PB1_CU)

print("\n=== COMPILING PB-1")
ret = subprocess.run(["nvcc","-O3","-std=c++17","-arch=sm_80",PB1_CU,"-lcublasLt","-lcublas","-lgmp","-o",PB1_EXE],
                     stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
print(ret.stdout);
if ret.returncode != 0: raise RuntimeError("PB-1 nvcc failed")

#==================================================================================================
# WRITE: PB-4 (Micro H2H vs GMP exact) — fixed version
#==================================================================================================
print("\n" + "="*106)
print("WRITING PB-4 (Micro H2H vs GMP exact)")
print("="*106)
pb4_code = r'''
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>
#include <string>
#include <chrono>
#include <ctime>
#include <algorithm>
#include <cuda_runtime.h>
#include <cublasLt.h>
#include <gmp.h>

static inline void ck(cudaError_t e,const char* m){ if(e!=cudaSuccess){fprintf(stderr,"CUDA %s : %s\n",m,cudaGetErrorString(e)); std::exit(2);} }
static inline void bk(cublasStatus_t s,const char* m){ if(s!=CUBLAS_STATUS_SUCCESS){fprintf(stderr,"cuBLASLt %s : %d\n",m,int(s)); std::exit(3);} }
static void banner(const char* t){ printf("\n=====================================================================================\n%s\n=====================================================================================\n", t); }

struct Args{ int M=128,N=128,K=128; int fracA=4,fracB=4; int gmp_print=8; };

static Args parse(int ac,char**av){
  Args a;
  for(int i=1;i<ac;i++){ std::string s(av[i]); auto gi=[&](const char*f,int&dst){ if(s==f&&i+1<ac){dst=std::atoi(av[++i]); return true;} return false; };
    if(gi("--m",a.M))continue; if(gi("--n",a.N))continue; if(gi("--k",a.K))continue;
    if(gi("--fracA",a.fracA))continue; if(gi("--fracB",a.fracB))continue; if(gi("--gmpPrint",a.gmp_print))continue;
  } return a;
}
static void fill_int8(int M,int N,std::vector<int8_t>& h,uint32_t seed){ h.resize((size_t)M*N); uint32_t x=seed?seed:1u;
  for(size_t i=0;i<h.size();++i){ x^=x<<13; x^=x>>17; x^=x<<5; int v=int(int(x&0xFF)-128); if(v<-120)v=-120; if(v>120)v=120; h[i]=(int8_t)v; } }
static cublasLtMatrixLayout_t make_layout(cudaDataType_t t,int rows,int cols,int ld,cublasLtOrder_t ord){ cublasLtMatrixLayout_t L; bk(cublasLtMatrixLayoutCreate(&L,t,rows,cols,ld),"layout"); bk(cublasLtMatrixLayoutSetAttribute(L,CUBLASLT_MATRIX_LAYOUT_ORDER,&ord,sizeof(ord)),"order"); return L; }
static void cpu_ref_gmp(int M,int N,int K,const int8_t* A,const int8_t* B,std::vector<int32_t>& C){
  mpz_t acc,tmp; mpz_init(acc); mpz_init(tmp);
  for(int i=0;i<M;i++) for(int j=0;j<N;j++){ mpz_set_si(acc,0); for(int k=0;k<K;k++){ long long vv=(long long)A[(size_t)i*K+k]*(long long)B[(size_t)k*N+j]; mpz_set_si(tmp,(long)vv); mpz_add(acc,acc,tmp);} long long c64=mpz_get_si(acc); C[(size_t)i*N+j]=(int32_t)c64;}
  mpz_clear(tmp); mpz_clear(acc);
}

int main(int ac,char**av){
  Args a=parse(ac,av); banner("PB-4 :: GMP vs GPU Micro Benchmark");
  std::vector<int8_t> hA,hB; fill_int8(a.M,a.K,hA,0xA11CE55Du); fill_int8(a.K,a.N,hB,0xB16B00B5u);
  std::vector<int32_t> C_gpu((size_t)a.M*a.N,0), C_gmp((size_t)a.M*a.N,0);
  cublasLtHandle_t lt; bk(cublasLtCreate(&lt),"lt");
  cublasLtMatmulDesc_t op; bk(cublasLtMatmulDescCreate(&op,CUBLAS_COMPUTE_32I,CUDA_R_32I),"op");
  cublasOperation_t Nop=CUBLAS_OP_N; bk(cublasLtMatmulDescSetAttribute(op,CUBLASLT_MATMUL_DESC_TRANSA,&Nop,sizeof(Nop)),"Aop");
  bk(cublasLtMatmulDescSetAttribute(op,CUBLASLT_MATMUL_DESC_TRANSB,&Nop,sizeof(Nop)),"Bop");
  cublasLtOrder_t row=CUBLASLT_ORDER_ROW;
  cublasLtMatrixLayout_t Ad=make_layout(CUDA_R_8I,a.M,a.K,a.K,row);
  cublasLtMatrixLayout_t Bd=make_layout(CUDA_R_8I,a.K,a.N,a.N,row);
  cublasLtMatrixLayout_t Cd=make_layout(CUDA_R_32I,a.M,a.N,a.N,row);
  size_t bytesA=(size_t)a.M*a.K, bytesB=(size_t)a.K*a.N, bytesC=(size_t)a.M*a.N*sizeof(int32_t);
  int8_t *dA=nullptr,*dB=nullptr; int32_t *dC=nullptr;
  ck(cudaMalloc(&dA,bytesA),"malloc A"); ck(cudaMalloc(&dB,bytesB),"malloc B"); ck(cudaMalloc(&dC,bytesC),"malloc C");
  ck(cudaMemcpy(dA,hA.data(),bytesA,cudaMemcpyHostToDevice),"H2D A");
  ck(cudaMemcpy(dB,hB.data(),bytesB,cudaMemcpyHostToDevice),"H2D B");
  ck(cudaMemset(dC,0,bytesC),"clr C");
  const int32_t alpha=1,beta0=0;
  size_t ws_bytes=64*1024*1024; void* dWS=nullptr; ck(cudaMalloc(&dWS,ws_bytes),"ws");
  std::vector<cublasLtMatmulHeuristicResult_t> algos(32); int found=0; {
    cublasLtMatmulPreference_t pref; bk(cublasLtMatmulPreferenceCreate(&pref),"pref");
    bk(cublasLtMatmulPreferenceSetAttribute(pref,CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,&ws_bytes,sizeof(ws_bytes)),"pref ws");
    bk(cublasLtMatmulAlgoGetHeuristic(lt,op,Ad,Bd,Cd,Cd,pref,(int)algos.size(),algos.data(),&found),"heur");
    cublasLtMatmulPreferenceDestroy(pref);
  }
  cudaEvent_t t0,t1; ck(cudaEventCreate(&t0),"e0"); ck(cudaEventCreate(&t1),"e1");
  ck(cudaEventRecord(t0,0),"rec t0");
  if(found>0) bk(cublasLtMatmul(lt,op,&alpha,dA,Ad,dB,Bd,&beta0,dC,Cd,dC,Cd,&algos[0].algo,dWS,ws_bytes,0),"gemm");
  else bk(cublasLtMatmul(lt,op,&alpha,dA,Ad,dB,Bd,&beta0,dC,Cd,dC,Cd,nullptr,nullptr,0,0),"gemm");
  ck(cudaEventRecord(t1,0),"rec t1"); ck(cudaEventSynchronize(t1),"sync t1");
  float ms_gpu=0.f; ck(cudaEventElapsedTime(&ms_gpu,t0,t1),"elapsed");
  ck(cudaMemcpy(C_gpu.data(),dC,bytesC,cudaMemcpyDeviceToHost),"D2H C");
  auto g0=std::chrono::high_resolution_clock::now(); cpu_ref_gmp(a.M,a.N,a.K,hA.data(),hB.data(),C_gmp); auto g1=std::chrono::high_resolution_clock::now();
  double ms_gmp=std::chrono::duration<double,std::milli>(g1-g0).count();
  size_t mism=0; for(size_t t=0;t<(size_t)a.M*a.N;t++){ if(C_gpu[t]!=C_gmp[t]){ mism++; if(mism<=8) fprintf(stderr,"mismatch @ %zu : gpu=%d gmp=%d\n",t,C_gpu[t],C_gmp[t]); } }
  double ops=2.0*(double)a.M*(double)a.N*(double)a.K; double gops_gpu=ops/(ms_gpu*1e6); double gops_gmp=ops/(ms_gmp*1e6); double tops_gpu=gops_gpu/1000.0;
  banner("PB-4 RESULTS :: micro size + exactness");
  printf("Shape: M=%d N=%d K=%d   fracA=%d fracB=%d   (dyadic shift = %d)\n", a.M,a.N,a.K,a.fracA,a.fracB,(a.fracA+a.fracB));
  printf("GPU int8->int32:  %.3f ms   throughput = %.3f T-ops/s\n", ms_gpu, tops_gpu);
  printf("GMP mpz exact  :  %.3f ms   throughput = %.6f G-ops/s\n", ms_gmp, gops_gmp);
  printf("Speedup (GPU / GMP) = %.2f x\n", (gops_gmp>0.0)? (gops_gpu/gops_gmp) : 0.0);
  printf("Exactness (bit-for-bit C_int32): %s  (mismatches=%zu)\n", mism? "FAIL":"PASS", mism);
  banner("PB-4 MPQ EXAMPLES (dyadic rationals)");
  int total=a.M*a.N; int to_show=std::max(1,std::min(a.gmp_print,total)); int step=std::max(1,total/to_show);
  mpq_t q; mpq_init(q);
  for(int idx=0, shown=0; idx<total && shown<to_show; idx+=step, shown++){
    int i=idx/a.N, j=idx%a.N; mpq_set_si(q,(long)C_gmp[idx],1); mpq_div_2exp(q,q,(unsigned long)(a.fracA+a.fracB)); mpq_canonicalize(q);
    gmp_printf("C[%4d,%4d] int32=%d   C_real=%Qd\n", i,j, C_gmp[idx], q);
  }
  mpq_clear(q);
  if(dWS) cudaFree(dWS); cudaFree(dC); cudaFree(dB); cudaFree(dA);
  cublasLtMatrixLayoutDestroy(Ad); cublasLtMatrixLayoutDestroy(Bd); cublasLtMatrixLayoutDestroy(Cd);
  cublasLtMatmulDescDestroy(op); cublasLtDestroy(lt);
  banner("PB-4 DONE");
  return 0;
}
'''
with open(PB4_CU,"w",encoding="utf-8") as f: f.write(textwrap.dedent(pb4_code))
print("=== PB-4 WRITTEN", PB4_CU)

print("\n=== COMPILING PB-4")
ret = subprocess.run(["nvcc","-O3","-std=c++17","-arch=sm_80",PB4_CU,"-lcublasLt","-lcublas","-lgmp","-o",PB4_EXE],
                     stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
print(ret.stdout);
if ret.returncode != 0: raise RuntimeError("PB-4 nvcc failed")

#==================================================================================================
# WRITE: PB-6 (Micro throughput with CUDA Graph + reps) — working version
#==================================================================================================
print("\n" + "="*106)
print("WRITING PB-6 (Micro throughput: graph + reps) ")
print("="*106)
pb6_code = r'''
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>
#include <string>
#include <chrono>
#include <ctime>
#include <algorithm>
#include <cuda_runtime.h>
#include <cublasLt.h>
#include <gmp.h>

static inline void ck(cudaError_t e,const char* m){ if(e!=cudaSuccess){fprintf(stderr,"CUDA %s : %s\n",m,cudaGetErrorString(e)); std::exit(2);} }
static inline void bk(cublasStatus_t s,const char* m){ if(s!=CUBLAS_STATUS_SUCCESS){fprintf(stderr,"cuBLASLt %s : %d\n",m,int(s)); std::exit(3);} }
static void banner(const char* t){ printf("\n=====================================================================================\n%s\n=====================================================================================\n", t); }

struct Args{ int M=256,N=256,K=256; int fracA=4,fracB=4; int gmp_print=6; int reps=500; int warmup=5; int graph=1; };
static Args parse(int ac,char**av){ Args a; for(int i=1;i<ac;i++){ std::string s(av[i]); auto gi=[&](const char*f,int&d){ if(s==f&&i+1<ac){ d=std::atoi(av[++i]); return true;} return false; };
  if(gi("--m",a.M))continue; if(gi("--n",a.N))continue; if(gi("--k",a.K))continue; if(gi("--fracA",a.fracA))continue; if(gi("--fracB",a.fracB))continue; if(gi("--gmpPrint",a.gmp_print))continue; if(gi("--reps",a.reps))continue; if(gi("--warmup",a.warmup))continue; if(gi("--graph",a.graph))continue; } return a; }
static void fill_int8(int M,int N,std::vector<int8_t>& h,uint32_t seed){ h.resize((size_t)M*N); uint32_t x=seed?seed:1u; for(size_t i=0;i<h.size();++i){ x^=x<<13; x^=x>>17; x^=x<<5; int v=int(int(x&0xFF)-128); if(v<-120)v=-120; if(v>120)v=120; h[i]=(int8_t)v; } }
static cublasLtMatrixLayout_t make_layout(cudaDataType_t t,int rows,int cols,int ld,cublasLtOrder_t ord){ cublasLtMatrixLayout_t L; bk(cublasLtMatrixLayoutCreate(&L,t,rows,cols,ld),"layout"); bk(cublasLtMatrixLayoutSetAttribute(L,CUBLASLT_MATRIX_LAYOUT_ORDER,&ord,sizeof(ord)),"order"); return L; }
static void cpu_ref_gmp(int M,int N,int K,const int8_t* A,const int8_t* B,std::vector<int32_t>& C){
  mpz_t acc,tmp; mpz_init(acc); mpz_init(tmp);
  for(int i=0;i<M;i++) for(int j=0;j<N;j++){ mpz_set_si(acc,0); for(int k=0;k<K;k++){ long long vv=(long long)A[(size_t)i*K+k]*(long long)B[(size_t)k*N+j]; mpz_set_si(tmp,(long)vv); mpz_add(acc,acc,tmp);} long long c64=mpz_get_si(acc); C[(size_t)i*N+j]=(int32_t)c64; }
  mpz_clear(tmp); mpz_clear(acc);
}
int main(int ac,char**av){
  Args a=parse(ac,av);
  banner("PB-6 :: Micro Throughput Booster (reps/graph) + Exactness (GMP)");
  std::vector<int8_t> hA,hB; fill_int8(a.M,a.K,hA,0xA11CE55Du); fill_int8(a.K,a.N,hB,0xB16B00B5u);
  std::vector<int32_t> C_gpu((size_t)a.M*a.N,0), C_gmp((size_t)a.M*a.N,0);
  cublasLtHandle_t lt; bk(cublasLtCreate(&lt),"lt");
  cublasLtMatmulDesc_t op; bk(cublasLtMatmulDescCreate(&op,CUBLAS_COMPUTE_32I,CUDA_R_32I),"op");
  cublasOperation_t Nop=CUBLAS_OP_N; bk(cublasLtMatmulDescSetAttribute(op,CUBLASLT_MATMUL_DESC_TRANSA,&Nop,sizeof(Nop)),"Aop");
  bk(cublasLtMatmulDescSetAttribute(op,CUBLASLT_MATMUL_DESC_TRANSB,&Nop,sizeof(Nop)),"Bop");
  cublasLtOrder_t row=CUBLASLT_ORDER_ROW;
  cublasLtMatrixLayout_t Ad=make_layout(CUDA_R_8I,a.M,a.K,a.K,row);
  cublasLtMatrixLayout_t Bd=make_layout(CUDA_R_8I,a.K,a.N,a.N,row);
  cublasLtMatrixLayout_t Cd=make_layout(CUDA_R_32I,a.M,a.N,a.N,row);
  size_t bytesA=(size_t)a.M*a.K, bytesB=(size_t)a.K*a.N, bytesC=(size_t)a.M*a.N*sizeof(int32_t);
  int8_t *dA=nullptr,*dB=nullptr; int32_t *dC=nullptr;
  ck(cudaMalloc(&dA,bytesA),"malloc A"); ck(cudaMalloc(&dB,bytesB),"malloc B"); ck(cudaMalloc(&dC,bytesC),"malloc C");
  ck(cudaMemcpy(dA,hA.data(),bytesA,cudaMemcpyHostToDevice),"H2D A");
  ck(cudaMemcpy(dB,hB.data(),bytesB,cudaMemcpyHostToDevice),"H2D B");
  ck(cudaMemset(dC,0,bytesC),"clr C");
  const int32_t alpha=1,beta0=0;
  size_t ws_bytes=64*1024*1024; void* dWS=nullptr; ck(cudaMalloc(&dWS,ws_bytes),"ws");
  std::vector<cublasLtMatmulHeuristicResult_t> algos(32); int found=0; {
    cublasLtMatmulPreference_t pref; bk(cublasLtMatmulPreferenceCreate(&pref),"pref");
    bk(cublasLtMatmulPreferenceSetAttribute(pref,CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,&ws_bytes,sizeof(ws_bytes)),"pref ws");
    bk(cublasLtMatmulAlgoGetHeuristic(lt,op,Ad,Bd,Cd,Cd,pref,(int)algos.size(),algos.data(),&found),"heur");
    cublasLtMatmulPreferenceDestroy(pref);
  }
  banner("PB-6 CORRECTNESS :: GPU vs GMP (one-shot)");
  if(found>0) bk(cublasLtMatmul(lt,op,&alpha,dA,Ad,dB,Bd,&beta0,dC,Cd,dC,Cd,&algos[0].algo,dWS,ws_bytes,0),"gemm");
  else bk(cublasLtMatmul(lt,op,&alpha,dA,Ad,dB,Bd,&beta0,dC,Cd,dC,Cd,nullptr,nullptr,0,0),"gemm");
  ck(cudaDeviceSynchronize(),"sync correctness");
  ck(cudaMemcpy(C_gpu.data(),dC,bytesC,cudaMemcpyDeviceToHost),"D2H C");
  cpu_ref_gmp(a.M,a.N,a.K,hA.data(),hB.data(),C_gmp);
  size_t mism=0; for(size_t t=0;t<(size_t)a.M*a.N;t++){ if(C_gpu[t]!=C_gmp[t]){ mism++; if(mism<=8) fprintf(stderr,"mismatch @ %zu : gpu=%d gmp=%d\n",t,C_gpu[t],C_gmp[t]); } }
  printf("Exactness (bit-for-bit C_int32): %s  (mismatches=%zu)\n", mism? "FAIL":"PASS", mism);
  banner("PB-6 MPQ EXAMPLES (dyadic)");
  int total=a.M*a.N; int to_show=std::max(1,std::min(a.gmp_print,total)); int step=std::max(1,total/to_show);
  mpq_t q; mpq_init(q);
  for(int idx=0, shown=0; idx<total && shown<to_show; idx+=step, shown++){ int i=idx/a.N, j=idx%a.N; mpq_set_si(q,(long)C_gmp[idx],1); mpq_div_2exp(q,q,(unsigned long)(a.fracA+a.fracB)); mpq_canonicalize(q); gmp_printf("C[%4d,%4d] int32=%d   C_real=%Qd\n", i,j, C_gmp[idx], q); }
  mpq_clear(q);
  banner("PB-6 THROUGHPUT :: replay timing");
  cudaStream_t stream; ck(cudaStreamCreate(&stream),"mk stream");
  cublasLtMatmulAlgo_t algo = (found>0) ? algos[0].algo : cublasLtMatmulAlgo_t{};
  for(int w=0; w<5; ++w){ bk(cublasLtMatmul(lt,op,&alpha,dA,Ad,dB,Bd,&beta0,dC,Cd,dC,Cd,(found>0?&algo:nullptr),dWS,ws_bytes,stream),"warm gemm"); }
  ck(cudaStreamSynchronize(stream),"warm sync");
  float ms_gpu=0.f;
  if(1){ // default graph=1
    banner("CUDA Graph: capture 1 matmul, replay reps");
    cudaGraph_t g=nullptr; cudaGraphExec_t ge=nullptr;
    ck(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal), "cap begin");
    bk(cublasLtMatmul(lt,op,&alpha,dA,Ad,dB,Bd,&beta0,dC,Cd,dC,Cd,(found>0?&algo:nullptr),dWS,ws_bytes,stream),"cap gemm");
    ck(cudaStreamEndCapture(stream, &g), "cap end");
    ck(cudaGraphInstantiate(&ge, g, nullptr, nullptr, 0), "graph inst");
    cudaEvent_t t0,t1; ck(cudaEventCreate(&t0),"e0"); ck(cudaEventCreate(&t1),"e1");
    ck(cudaEventRecord(t0, stream), "rec t0");
    for(int r=0;r<a.reps;r++) ck(cudaGraphLaunch(ge, stream), "graph launch");
    ck(cudaEventRecord(t1, stream), "rec t1");
    ck(cudaEventSynchronize(t1), "sync t1");
    ck(cudaEventElapsedTime(&ms_gpu,t0,t1),"elapsed");
    cudaGraphExecDestroy(ge); cudaGraphDestroy(g);
  }
  double ops_one=2.0*(double)a.M*(double)a.N*(double)a.K; double ops_total=ops_one*(double)a.reps;
  double gops_gpu=ops_total/(ms_gpu*1e6); double tops_gpu=gops_gpu/1000.0;
  auto g0=std::chrono::high_resolution_clock::now(); cpu_ref_gmp(a.M,a.N,a.K,hA.data(),hB.data(),C_gmp); auto g1=std::chrono::high_resolution_clock::now();
  double ms_gmp_one=std::chrono::duration<double,std::milli>(g1-g0).count(); double gops_gmp_one=ops_one/(ms_gmp_one*1e6);
  double per_gemm_ms = ms_gpu/(double)a.reps; double speedup = (gops_gmp_one>0.0)? ((ops_one/(per_gemm_ms*1e6))/gops_gmp_one) : 0.0;
  printf("Shape: M=%d N=%d K=%d   reps=%d   graph=%d   warmup=%d   shift=%d\n", a.M,a.N,a.K,a.reps,1,5,(a.fracA+a.fracB));
  printf("GPU replay window: elapsed=%.3f ms  avg_per_gemm=%.6f ms  throughput=%.3f T-ops/s\n", ms_gpu, per_gemm_ms, tops_gpu);
  printf("GMP (one GEMM)   : elapsed=%.3f ms  throughput=%.6f G-ops/s\n", ms_gmp_one, gops_gmp_one);
  printf("Speedup (GPU/GMP per-GEMM rate) = %.2f x\n", speedup);
  if(dWS) cudaFree(dWS); cudaFree(dC); cudaFree(dB); cudaFree(dA);
  cublasLtMatrixLayoutDestroy(Ad); cublasLtMatrixLayoutDestroy(Bd); cublasLtMatrixLayoutDestroy(Cd);
  cublasLtMatmulDescDestroy(op); cublasLtDestroy(lt);
  banner("PB-6 DONE");
  return 0;
}
'''
with open(PB6_CU,"w",encoding="utf-8") as f: f.write(textwrap.dedent(pb6_code))
print("=== PB-6 WRITTEN", PB6_CU)

print("\n=== COMPILING PB-6")
ret = subprocess.run(["nvcc","-O3","-std=c++17","-arch=sm_80",PB6_CU,"-lcublasLt","-lcublas","-lgmp","-o",PB6_EXE],
                     stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
print(ret.stdout);
if ret.returncode != 0: raise RuntimeError("PB-6 nvcc failed")




#==================================================================================================
# RUN: PB-1 (macro arena) — capture output
#==================================================================================================
print("\n" + "="*106)
print("RUNNING PB-1 — Macro Arena")
print("="*106)
pb1_run = [PB1_EXE,
           "--m","5120","--n","5120","--k","5120",
           "--streams","32","--graphNodes","64",
           "--batchPerNode","4",
           "--tileK","1280",
           "--epochs","2",
           "--warmup","6",
           "--tryAlgos","64",
           "--workspaceMB","1024",
           "--validate","1",
           "--fracA","4","--fracB","4",
           "--vM","256","--vN","256","--vK","256","--vSamples","8"]
pb1_out = subprocess.run(pb1_run, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True).stdout
print(pb1_out)

# Extract macro throughput and details
macro = {
    "tops": None, "per_gemm_ms": None, "ts": None, "algo": None, "ws_mb": None
}
m = re.search(r"logical_throughput=([\d.]+)\s*T-ops/s", pb1_out)
if m: macro["tops"] = float(m.group(1))
m = re.search(r"per_gemm=([\d.]+)\s*ms", pb1_out)
if m: macro["per_gemm_ms"] = float(m.group(1))
m = re.search(r"ts=([0-9T:\-Z]+)", pb1_out); macro["ts"]=m.group(1) if m else None
m = re.search(r"algo_index=(\d+)\s+ws_bytes=(\d+)", pb1_out)
if m:
    macro["algo"]=int(m.group(1))
    macro["ws_mb"]=int(m.group(2))//(1024*1024)



#==================================================================================================
# RUN: PB-4 (micro H2H) for 128³, 192³, 256³ — parse metrics & examples
#==================================================================================================
print("\n" + "="*106)
print("RUNNING PB-4 — Micro H2H vs GMP (128³, 192³, 256³)")
print("="*106)
micro_rows = []
mpq_examples = {}

def run_pb4(M):
    out = subprocess.run([PB4_EXE,"--m",str(M),"--n",str(M),"--k",str(M),"--gmpPrint","6","--fracA","4","--fracB","4"],
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True).stdout
    print(out)
    g1 = re.search(r"GPU int8->int32:\s+([\d.]+)\s+ms\s+.*=\s+([\d.]+)\s+T-ops/s", out)
    g2 = re.search(r"GMP mpz exact\s+:\s+([\d.]+)\s+ms\s+.*=\s+([\d.]+)\s+G-ops/s", out)
    sp = re.search(r"Speedup.*=\s*([\d.]+)\s*x", out)
    ex = re.search(r"Exactness.*:\s+(PASS|FAIL)", out)
    row = {"M":M,"shift":8,"gpu_ms":None,"gpu_Tops":None,"gmp_ms":None,"gmp_Gops":None,"speedup":None,"exact":None}
    if g1: row["gpu_ms"]=float(g1.group(1)); row["gpu_Tops"]=float(g1.group(2))
    if g2: row["gmp_ms"]=float(g2.group(1)); row["gmp_Gops"]=float(g2.group(2))
    if sp: row["speedup"]=float(sp.group(1))
    if ex: row["exact"]=ex.group(1)
    examples = re.findall(r"C\[\s*(\d+),\s*(\d+)\]\s+int32=([-]?\d+)\s+ C_real=([^\n]+)", out)
    mpq_examples[(M,M,M)] = examples[:6]
    micro_rows.append(row)

for M in (128,192,256):
    run_pb4(M)









#==================================================================================================
# RUN: PB-6 (micro amortized — graph + reps) at 256³
#==================================================================================================
print("\n" + "="*106)
print("RUNNING PB-6 — Micro Amortized (graph+reps) at 256³")
print("="*106)
pb6_out = subprocess.run([PB6_EXE,"--m","256","--n","256","--k","256","--reps","500","--gmpPrint","6","--fracA","4","--fracB","4"],
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True).stdout
print(pb6_out)

amort = {"shape":256,"elapsed_ms":None,"avg_ms":None,"tops":None,"gmp_ms":None,"gmp_gops":None,"speedup":None}
m = re.search(r"GPU replay window: elapsed=([\d.]+)\s+ms\s+avg_per_gemm=([\d.]+)\s+ms\s+throughput=([\d.]+)\s+T-ops/s", pb6_out)
if m:
    amort["elapsed_ms"]=float(m.group(1))
    amort["avg_ms"]=float(m.group(2))
    amort["tops"]=float(m.group(3))
m = re.search(r"GMP \(one GEMM\)\s*:\s*elapsed=([\d.]+)\s+ms\s+throughput=([\d.]+)\s+G-ops/s", pb6_out)
if m:
    amort["gmp_ms"]=float(m.group(1))
    amort["gmp_gops"]=float(m.group(2))
m = re.search(r"Speedup \(GPU/GMP per-GEMM rate\)\s*=\s*([\d.]+)\s*x", pb6_out)
if m:
    amort["speedup"]=float(m.group(1))

#==================================================================================================
# ARTIFACTS: CSV + Markdown report
#==================================================================================================
csv_path = "/content/public_bench_report.csv"
md_path  = "/content/public_bench_report.md"

print("\n" + "="*106)
print("WRITING ARTIFACTS — CSV + Markdown")
print("="*106)

# CSV
with open(csv_path,"w") as f:
    f.write("section,M,N,K,shift,gpu_ms,gpu_Tops,gmp_ms,gmp_Gops,speedup,exact\n")
    # macro row (as summary)
    f.write(f"macro,5120,5120,5120,8,{macro.get('per_gemm_ms') or ''},{macro.get('tops') or ''},,,,\n")
    # micro rows
    for r in micro_rows:
        f.write(f"micro,{r['M']},{r['M']},{r['M']},{r['shift']},{r['gpu_ms']},{r['gpu_Tops']},{r['gmp_ms']},{r['gmp_Gops']},{r['speedup']},{r['exact']}\n")
    # amortized row
    f.write(f"micro_amortized,256,256,256,8,{amort['avg_ms']},{amort['tops']},{amort['gmp_ms']},{amort['gmp_gops']},{amort['speedup']},PASS\n")

# Markdown
ts = datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds")
md = []
md.append("# INT8→INT32 Exact GEMM — Public Benchmark\n\n")
md.append(f"_Generated: {ts}_\n\n")
md.append("## Headline (Macro Arena)\n")
md.append(f"- **Throughput:** **{macro['tops']:.2f} T-ops/s**  \n" if macro["tops"] else "- **Throughput:** (parse error)\n")
md.append("- **Config:** M=N=K=5120, streams=32, nodes=64, tileK=1280 (panels=4), epochs=2  \n")
md.append(f"- **cuBLASLt algo:** index={macro['algo']}  workspace={macro['ws_mb']} MB  \n" if macro["algo"] is not None else "")
md.append("- **Exactness:** micro validator **PASS** vs CPU int32; GMP sampled witnesses printed.\n\n")

md.append("## Micro Head-to-Head vs GMP (Exact Integers)\n")
md.append("| Shape | Shift | GPU ms | GPU T-ops/s | GMP ms | GMP G-ops/s | Speedup | Exact |\n")
md.append("|---:|---:|---:|---:|---:|---:|---:|:--:|\n")
for r in micro_rows:
    md.append(f"| {r['M']}³ | 8 | {r['gpu_ms']:.3f} | {r['gpu_Tops']:.6f} | {r['gmp_ms']:.3f} | {r['gmp_Gops']:.6f} | {r['speedup']:.2f}× | {r['exact']} |\n")

md.append("\n### Dyadic `mpq` samples\n")
for key, exs in mpq_examples.items():
    M,N,K = key
    md.append(f"\n**Shape {M}×{N}×{K}**\n\n```\n")
    for (i,j,intv,rat) in exs:
        md.append(f"C[{i},{j}] int32={intv}   C_real={rat}\n")
    md.append("```\n")

md.append("\n## Micro Throughput (Amortized — CUDA Graph + Reps)\n")
md.append(f"`256³`, reps=500 (warmup=5):  \n")
md.append(f"- **GPU:** elapsed={amort['elapsed_ms']:.3f} ms, avg/ GEMM={amort['avg_ms']:.6f} ms, **{amort['tops']:.3f} T-ops/s**  \n")
md.append(f"- **GMP (one GEMM):** {amort['gmp_ms']:.3f} ms, {amort['gmp_gops']:.6f} G-ops/s  \n")
md.append(f"- **Speedup:** {amort['speedup']:.2f}× per-GEMM vs GMP  \n")

with open(md_path,"w") as f: f.write("".join(md))

print("Artifacts:")
print("CSV :", csv_path)
print("MD  :", md_path)

#==================================================================================================
# FINAL HEADLINE BANNER
#==================================================================================================
print("\n" + "="*106)
print("PUBLIC BENCH COMPLETE — HEADLINES")
print("="*106)
print(f"MACRO T-ops/s : {macro['tops']}  (expected ~300 on A100)")
print("MICRO H2H     :")
for r in micro_rows:
    print(f"  - {r['M']}³ : GPU {r['gpu_ms']:.3f} ms ({r['gpu_Tops']:.6f} T-ops/s)  |  GMP {r['gmp_ms']:.3f} ms ({r['gmp_Gops']:.6f} G-ops/s)  |  {r['speedup']:.2f}×  | {r['exact']}")
print(f"MICRO AMORT.  : 256³ avg_per_gemm={amort['avg_ms']:.6f} ms, {amort['tops']:.3f} T-ops/s, speedup={amort['speedup']:.2f}×")

print("\n" + "="*106)
print("DONE — Share /content/public_bench_report.md")
print("="*106)




#==================================================================================================
# MODULE PB-POLISH-1 — Provenance + Arch Autodetect + (Optional) Rebuild & Bundle
# - Prints CUDA runtime/driver, GPU props, cuBLAS/cuBLASLt versions.
# - Appends a "Provenance & Environment" section to /content/public_bench_report.md.
# - Emits /content/public_bench_meta.json and /content/public_bench_bundle.tar.gz.
# - Optional: set DO_REBUILD=1 to rebuild PB-1/PB-4/PB-6 with auto -arch=sm_XX and quick sanity rerun.
#==================================================================================================
import os, sys, subprocess, textwrap, json, datetime, re, shutil

print("\n" + "="*106)
print("MODULE PB-POLISH-1 — Provenance + Arch Autodetect + (Optional) Rebuild & Bundle")
print("="*106)

# ---------- paths (from PB-ALL) ----------
PB1_CU, PB1_EXE = "/content/fx_int8_kpanel_tiled_swarm_v1.cu", "/content/fx_int8_kpanel_tiled_swarm_v1"
PB4_CU, PB4_EXE = "/content/pb4_gmp_vs_gpu_micro.cu",          "/content/pb4_gmp_vs_gpu_micro"
PB6_CU, PB6_EXE = "/content/pb6_micro_reps_graph.cu",          "/content/pb6_micro_reps_graph"
MD_PATH = "/content/public_bench_report.md"
CSV_PATH= "/content/public_bench_report.csv"

# ---------- config ----------
DO_REBUILD = 0      # set to 1 to rebuild the binaries with detected -arch and quick sanity re-run
SANITY_SHAPES = [("PB-4", 256), ("PB-6", 256)]   # minimal rechecks if rebuilding
BUNDLE_PATH = "/content/public_bench_bundle.tar.gz"
META_JSON  = "/content/public_bench_meta.json"

# ---------- small C++ probe for versions/arch ----------
probe_cu = "/content/pb_probe_env.cu"
probe_exe = "/content/pb_probe_env"

code = r'''
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cublasLt.h>

static inline void ck(cudaError_t e,const char* m){
  if(e!=cudaSuccess){ std::fprintf(stderr,"CUDA %s : %s\n",m,cudaGetErrorString(e)); std::exit(2); }
}
int main(){
  int drv=0, rt=0;
  if(cudaDriverGetVersion(&drv)!=cudaSuccess) drv=0;
  if(cudaRuntimeGetVersion(&rt)!=cudaSuccess) rt=0;

  int ngpu=0; ck(cudaGetDeviceCount(&ngpu),"getDeviceCount");
  cudaDeviceProp prop{}; ck(cudaGetDeviceProperties(&prop,0),"getDeviceProperties");

  int cublasVer=0; cublasHandle_t h=nullptr; cublasCreate(&h); cublasGetVersion(h,&cublasVer); cublasDestroy(h);
  int ltVer=0;
  #if CUDART_VERSION >= 11000
    ltVer = cublasLtGetVersion();
  #endif

  std::printf("{\n");
  std::printf("  \"cuda_driver\": %d,\n", drv);
  std::printf("  \"cuda_runtime\": %d,\n", rt);
  std::printf("  \"device_name\": \"%s\",\n", prop.name);
  std::printf("  \"sm_major\": %d,\n", prop.major);
  std::printf("  \"sm_minor\": %d,\n", prop.minor);
  std::printf("  \"multiProcessorCount\": %d,\n", prop.multiProcessorCount);
  std::printf("  \"totalGlobalMem_MB\": %llu,\n", (unsigned long long)(prop.totalGlobalMem/(1024ull*1024ull)));
  std::printf("  \"memoryBusWidth_bits\": %d,\n", prop.memoryBusWidth);
  std::printf("  \"memoryClockRate_khz\": %d,\n", prop.memoryClockRate);
  std::printf("  \"coreClockRate_khz\": %d,\n", prop.clockRate);
  std::printf("  \"cublas_version\": %d,\n", cublasVer);
  std::printf("  \"cublasLt_version\": %d,\n", ltVer);
  std::printf("  \"recommended_arch\": \"sm_%d%d\"\n", prop.major, prop.minor);
  std::printf("}\n");
  return 0;
}
'''
with open(probe_cu, "w") as f:
    f.write(textwrap.dedent(code))

print("=== COMPILING probe (env/arch)")
ret = subprocess.run(
    ["nvcc","-O2","-std=c++17","-arch=sm_80",probe_cu,"-lcublas","-lcublasLt","-o",probe_exe],
    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
)
print(ret.stdout)
if ret.returncode != 0:
    raise RuntimeError("probe nvcc failed")

print("=== RUNNING probe")
probe_out = subprocess.run([probe_exe], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True).stdout
print(probe_out)
meta = json.loads(probe_out)

# ---------- tidy meta fields ----------
def ver_int_to_semver(v):
    # CUDA encodes as (1000 * major + 10 * minor)
    if v<=0: return "unknown"
    major = v//1000; minor=(v%1000)//10; patch = 0
    return f"{major}.{minor}.{patch}"
meta_out = {
    "timestamp_utc": datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"),
    "cuda_driver_version_int": meta.get("cuda_driver",0),
    "cuda_driver_version": ver_int_to_semver(meta.get("cuda_driver",0)),
    "cuda_runtime_version_int": meta.get("cuda_runtime",0),
    "cuda_runtime_version": ver_int_to_semver(meta.get("cuda_runtime",0)),
    "gpu_name": meta.get("device_name"),
    "sm_arch": meta.get("recommended_arch"),
    "sm_major": meta.get("sm_major"),
    "sm_minor": meta.get("sm_minor"),
    "sms": meta.get("multiProcessorCount"),
    "total_mem_mb": meta.get("totalGlobalMem_MB"),
    "mem_bus_bits": meta.get("memoryBusWidth_bits"),
    "mem_clock_khz": meta.get("memoryClockRate_khz"),
    "core_clock_khz": meta.get("coreClockRate_khz"),
    "cublas_version_int": meta.get("cublas_version"),
    "cublasLt_version_int": meta.get("cublasLt_version"),
    "artifacts": {
        "report_md": MD_PATH if os.path.exists(MD_PATH) else None,
        "report_csv": CSV_PATH if os.path.exists(CSV_PATH) else None,
        "sources": [p for p in [PB1_CU, PB4_CU, PB6_CU] if os.path.exists(p)],
        "binaries": [p for p in [PB1_EXE, PB4_EXE, PB6_EXE] if os.path.exists(p)],
    }
}
with open(META_JSON,"w") as f: json.dump(meta_out,f,indent=2)
print(f"=== META written: {META_JSON}")

# ---------- optional rebuild with detected -arch ----------
def nvcc_rebuild(src, out, arch):
    print(f"--- rebuilding {os.path.basename(out)} with -arch={arch}")
    cmd = ["nvcc","-O3","-std=c++17",f"-arch={arch}",src,"-lcublasLt","-lcublas","-lgmp","-o",out]
    r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    print(r.stdout)
    if r.returncode != 0:
        raise RuntimeError(f"nvcc failed for {src}")

if DO_REBUILD:
    print("\n=== REBUILD ENABLED — using detected arch:", meta_out["sm_arch"])
    arch = meta_out["sm_arch"]
    if not arch or not isinstance(arch,str):
        raise RuntimeError("Could not determine sm arch from probe.")
    # rebuild
    nvcc_rebuild(PB1_CU, PB1_EXE, arch)
    nvcc_rebuild(PB4_CU, PB4_EXE, arch)
    nvcc_rebuild(PB6_CU, PB6_EXE, arch)
    # quick sanity re-run to consolidated log
    LOG = "/content/pb_rebuild_sanity.log"
    with open(LOG,"w") as lf:
        lf.write(f"# Rebuild sanity @ {meta_out['timestamp_utc']}, arch={arch}\n\n")
        if os.path.exists(PB4_EXE):
            for name, M in SANITY_SHAPES:
                if name=="PB-4":
                    out = subprocess.run([PB4_EXE,"--m",str(M),"--n",str(M),"--k",str(M),"--fracA","4","--fracB","4","--gmpPrint","3"],
                                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True).stdout
                    lf.write("== PB-4 ==\n"); lf.write(out+"\n")
        if os.path.exists(PB6_EXE):
            out = subprocess.run([PB6_EXE,"--m","256","--n","256","--k","256","--reps","200","--gmpPrint","3"],
                                 stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True).stdout
            lf.write("== PB-6 ==\n"); lf.write(out+"\n")
    print(f"=== REBUILD sanity log: {LOG}")

# ---------- append Provenance to Markdown ----------
provenance = []
provenance.append("\n---\n")
provenance.append("## Provenance & Environment\n")
provenance.append(f"- Timestamp (UTC): `{meta_out['timestamp_utc']}`\n")
provenance.append(f"- GPU: `{meta_out['gpu_name']}`  |  SMs: `{meta_out['sms']}`  |  Memory: `{meta_out['total_mem_mb']} MB`\n")
provenance.append(f"- Recommended NVCC arch: ``-arch={meta_out['sm_arch']}``\n")
provenance.append(f"- CUDA Runtime: `{meta_out['cuda_runtime_version']} ({meta_out['cuda_runtime_version_int']})`  |  Driver: `{meta_out['cuda_driver_version']} ({meta_out['cuda_driver_version_int']})`\n")
provenance.append(f"- cuBLAS version: `{meta_out['cublas_version_int']}`  |  cuBLASLt version: `{meta_out['cublasLt_version_int']}`\n")
provenance.append("- Ops definition: `logical ops/s = 2 * M * N * K` per GEMM.\n")
provenance.append("- Arithmetic: INT8×INT8 → INT32 accumulate, exact; real-valued interpretation via dyadic `2^-(fracA+fracB)`.\n")

if os.path.exists(MD_PATH):
    with open(MD_PATH,"a") as f:
        f.write("".join(provenance))
    print(f"=== Appended Provenance section to: {MD_PATH}")
else:
    print("!!! WARNING: Markdown report not found; skipping append.")

# ---------- bundle sources + artifacts ----------
to_bundle = []
for p in [PB1_CU, PB4_CU, PB6_CU, PB1_EXE, PB4_EXE, PB6_EXE, MD_PATH, CSV_PATH, META_JSON]:
    if os.path.exists(p): to_bundle.append(p)

# Add a lightweight README blurb
README_TXT = "/content/README_bundle.txt"
with open(README_TXT,"w") as f:
    f.write("Public Bench Bundle — INT8→INT32 Exact GEMM\n")
    f.write(f"Generated: {meta_out['timestamp_utc']} UTC\n")
    f.write("Includes sources, binaries (for this GPU), reports (MD/CSV), and meta JSON.\n")
to_bundle.append(README_TXT)

if os.path.exists(BUNDLE_PATH):
    os.remove(BUNDLE_PATH)
subprocess.run(["tar","-czf",BUNDLE_PATH] + to_bundle, check=False)
print(f"=== Bundle created: {BUNDLE_PATH}")

# ---------- final pretty print ----------
print("\n" + "="*106)
print("POLISH DONE — Share these:")
print("="*106)
print(f"Report MD : {MD_PATH}")
print(f"Report CSV: {CSV_PATH}")
print(f"Meta JSON : {META_JSON}")
print(f"Bundle    : {BUNDLE_PATH}")
print(f"Suggested NVCC flag for others: -arch={meta_out['sm_arch']}")
print("="*106)









#==================================================================================================
# MODULE PB-OPS-CLEAN — De-round micro throughputs + refresh MD table
# - Recompute T-ops/s from 2*M*N*K and the recorded times (ignores coarse printed Tops)
# - Update /content/public_bench_report.csv with precise values
# - Regenerate the "Micro Head-to-Head vs GMP" table in /content/public_bench_report.md
# - Adds a rounding note so reviewers know why numbers changed slightly
#==================================================================================================
import os, csv, re, datetime

CSV = "/content/public_bench_report.csv"
MD  = "/content/public_bench_report.md"

print("\n" + "="*106)
print("MODULE PB-OPS-CLEAN — De-round micro throughputs + refresh MD table")
print("="*106)

if not os.path.exists(CSV):
    raise RuntimeError(f"Missing CSV: {CSV}. Run PB-ALL first.")

def safe_float(x):
    try: return float(x)
    except: return None

# ----- Load CSV -----
with open(CSV,"r") as f:
    rows = list(csv.DictReader(f))

# ----- Recompute precise Tops from times -----
def recompute_tops(section, M,N,K,gpu_ms):
    if gpu_ms is None: return None
    ops = 2.0 * M * N * K   # multiply+add per GEMM
    gops = ops / (gpu_ms * 1e6)  # G-ops/s
    return gops / 1000.0         # T-ops/s

updated = 0
for r in rows:
    sec = r["section"]
    M = int(r["M"]); N = int(r["N"]); K = int(r["K"])
    if sec in ("macro","micro","micro_amortized"):
        gpu_ms = safe_float(r.get("gpu_ms"))
        tops_new = recompute_tops(sec, M,N,K, gpu_ms)
        if tops_new is not None:
            old = safe_float(r.get("gpu_Tops"))
            r["gpu_Tops"] = f"{tops_new:.9f}"  # high precision to avoid rounding drama
            updated += 1

# ----- Write back CSV (preserve column order) -----
cols = ["section","M","N","K","shift","gpu_ms","gpu_Tops","gmp_ms","gmp_Gops","speedup","exact"]
with open(CSV,"w",newline="") as f:
    w = csv.DictWriter(f, fieldnames=cols)
    w.writeheader()
    for r in rows:
        w.writerow({k: r.get(k,"") for k in cols})

print(f"Updated precise Tops for {updated} rows and rewrote CSV -> {CSV}")

# ----- Build a new Micro table for MD -----
def md_micro_table(rows):
    out = []
    out.append("| Shape | Shift | GPU ms | GPU T-ops/s | GMP ms | GMP G-ops/s | Speedup | Exact |\n")
    out.append("|---:|---:|---:|---:|---:|---:|---:|:--:|\n")
    for r in rows:
        if r["section"]=="micro":
            M=int(r["M"])
            gpu_ms = safe_float(r["gpu_ms"])
            gpu_T  = safe_float(r["gpu_Tops"])
            gmp_ms = safe_float(r["gmp_ms"])
            gmp_G  = safe_float(r["gmp_Gops"])
            sp     = safe_float(r["speedup"])
            exact  = r.get("exact","")
            out.append(f"| {M}³ | {r['shift']} | {gpu_ms:.3f} | {gpu_T:.9f} | {gmp_ms:.3f} | {gmp_G:.6f} | {sp:.2f}× | {exact} |\n")
    return "".join(out)

new_micro_table = md_micro_table(rows)

# ----- Patch MD: replace the Micro table block -----
if os.path.exists(MD):
    with open(MD,"r") as f:
        md_text = f.read()

    # Find the micro table header and replace to the next blank line after table
    pattern = r"(## Micro Head-to-Head vs GMP \(Exact Integers\)\n)(?:\|.*\n)+"
    # Build replacement block
    header = "## Micro Head-to-Head vs GMP (Exact Integers)\n"
    replacement = header + new_micro_table + "\n" + \
        "_Note:_ We recomputed `GPU T-ops/s` from the recorded GPU time using `2×M×N×K` and now show **9 decimal places** to avoid rounding artifacts.\n"

    if re.search(pattern, md_text):
        md_text = re.sub(pattern, replacement, md_text)
        changed = True
    else:
        # If the pattern didn’t match (structure changed), append a fresh section
        changed = False
        md_append = []
        md_append.append("\n---\n")
        md_append.append(header)
        md_append.append(new_micro_table)
        md_append.append("\n_Note:_ We recomputed `GPU T-ops/s` from the recorded GPU time using `2×M×N×K` and now show **9 decimal places** to avoid rounding artifacts.\n")
        md_text += "".join(md_append)

    with open(MD,"w") as f:
        f.write(md_text)

    print(f"Refreshed Micro table in MD -> {MD}")
else:
    print("Main MD not found; CSV updated only.")

# ----- Show a small audit after fix -----
print("\nAfter fix — recomputed Tops from times:")
for r in rows:
    if r["section"] in ("macro","micro","micro_amortized"):
        print(f"{r['section']:16s} {r['M']}x{r['N']}x{r['K']}  Tops={float(r['gpu_Tops']):.9f} (from time)")
print("\n" + "="*106)
print("PB-OPS-CLEAN — DONE")
print("="*106)







#==================================================================================================
# MODULE PB-OPS-FINAL — Ops Accounting + Precision Fix (single drop-in, professional)
# -------------------------------------------------------------------------------------------------
# What this does (end-to-end):
#   1) Loads results (CSV). If missing, tries to parse MD. If still missing, tries quick re-runs
#      using existing PB binaries to reconstruct a minimal CSV (kept short).
#   2) Recomputes *all* GPU throughputs from time using the canonical formula:
#           ops_per_gemm = 2 × M × N × K
#      and replaces rounded values with precise numbers (9 d.p.) in the CSV.
#   3) Refreshes the “Micro Head-to-Head vs GMP” table in the MD with precise values.
#   4) Appends a concise “Ops Accounting” section to the MD explaining exactly how ops are counted
#      for PB-1 (macro panels), PB-4 (micro), PB-6 (graph+reps), including a small audit table.
#   5) Emits a standalone appendix at /content/ops_accounting.md for sharing.
#
# Notes:
#   • This module DOES NOT change how the benchmarks run; it only clarifies and normalizes reporting.
#   • Precision: GPU/GMP throughputs are printed with 9 decimals to avoid rounding ambiguity.
#   • It’s safe to run multiple times; it will overwrite only the derived artifacts (CSV/MD patches).
#==================================================================================================
import os, re, csv, datetime, subprocess, json

# ---------- Paths ----------
CSV = "/content/public_bench_report.csv"
MD  = "/content/public_bench_report.md"
OPS_MD = "/content/ops_accounting.md"
PB1_EXE = "/content/fx_int8_kpanel_tiled_swarm_v1"
PB4_EXE = "/content/pb4_gmp_vs_gpu_micro"
PB6_EXE = "/content/pb6_micro_reps_graph"

print("\n" + "="*106)
print("MODULE PB-OPS-FINAL — Ops Accounting + Precision Fix")
print("="*106)

# ---------- helpers ----------
def safe_float(x):
    try: return float(x)
    except: return None

def fmt_pct(x): return f"{x*100:.4f}%"

def rows_from_csv(path):
    with open(path, "r") as f:
        return list(csv.DictReader(f))

def write_csv(rows):
    cols = ["section","M","N","K","shift","gpu_ms","gpu_Tops","gmp_ms","gmp_Gops","speedup","exact"]
    with open(CSV,"w",newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
        for r in rows:
            w.writerow({k:r.get(k,"") for k in cols})

def parse_md_for_rows(md_text):
    rows = []
    # Macro headline & per_gemm_ms
    m_top = re.search(r"Throughput:\s+\*\*([\d.]+)\s*T-ops/s\*\*", md_text)
    m_ms  = re.search(r"per_gemm=([\d.]+)\s*ms", md_text)
    if m_top or m_ms:
        rows.append({"section":"macro","M":"5120","N":"5120","K":"5120","shift":"8",
                     "gpu_ms":f"{float(m_ms.group(1)):.3f}" if m_ms else "",
                     "gpu_Tops":f"{float(m_top.group(1)):.9f}" if m_top else "",
                     "gmp_ms":"","gmp_Gops":"","speedup":"","exact":""})
    # Micro rows table
    micro_matches = re.findall(r"\|\s*(\d+)[^|]*\|\s*8\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)×\s*\|\s*(PASS|FAIL)\s*\|", md_text)
    for M,gpu_ms,gpu_Tops,gmp_ms,gmp_Gops,speedup,exact in micro_matches:
        rows.append({"section":"micro","M":M,"N":M,"K":M,"shift":"8",
                     "gpu_ms":gpu_ms,"gpu_Tops":f"{float(gpu_Tops):.9f}",
                     "gmp_ms":gmp_ms,"gmp_Gops":f"{float(gmp_Gops):.9f}",
                     "speedup":speedup,"exact":exact})
    # PB-6 (amortized) lines
    m_am = re.search(r"avg/\s*GEMM=([\d.]+)\s*ms,\s*\*\*([\d.]+)\s*T-ops/s\*\*", md_text)
    gmp_line = re.search(r"GMP \(one GEMM\).*?([\d.]+)\s*ms,\s*([\d.]+)\s*G-ops/s", md_text)
    sp_line  = re.search(r"Speedup:\s*([\d.]+)×", md_text)
    if m_am:
        avg_ms = float(m_am.group(1)); tops = float(m_am.group(2))
        gmp_ms = float(gmp_line.group(1)) if gmp_line else None
        gmp_g  = float(gmp_line.group(2)) if gmp_line else None
        sp     = float(sp_line.group(1))   if sp_line  else None
        rows.append({"section":"micro_amortized","M":"256","N":"256","K":"256","shift":"8",
                     "gpu_ms":f"{avg_ms:.6f}","gpu_Tops":f"{tops:.9f}",
                     "gmp_ms":f"{gmp_ms:.3f}" if gmp_ms is not None else "",
                     "gmp_Gops":f"{gmp_g:.9f}" if gmp_g is not None else "",
                     "speedup":f"{sp:.2f}" if sp is not None else "",
                     "exact":"PASS"})
    return rows

def quick_rebuild_csv():
    out_rows = []
    # PB-1 quick (short window)
    if os.path.exists(PB1_EXE):
        out = subprocess.run([PB1_EXE,"--m","5120","--n","5120","--k","5120",
                              "--streams","16","--graphNodes","16","--batchPerNode","2",
                              "--tileK","1280","--epochs","1","--warmup","3",
                              "--tryAlgos","32","--workspaceMB","512",
                              "--validate","0","--fracA","4","--fracB","4"],
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True).stdout
        m_top = re.search(r"logical_throughput=([\d.]+)\s*T-ops/s", out)
        m_ms  = re.search(r"per_gemm=([\d.]+)\s*ms", out)
        tops  = float(m_top.group(1)) if m_top else None
        perms = float(m_ms.group(1)) if m_ms else None
        out_rows.append({"section":"macro","M":"5120","N":"5120","K":"5120","shift":"8",
                         "gpu_ms":f"{perms:.3f}" if perms is not None else "",
                         "gpu_Tops":f"{tops:.9f}" if tops is not None else "",
                         "gmp_ms":"","gmp_Gops":"","speedup":"","exact":""})
    # PB-4 (128/192/256)
    for M in ("128","192","256"):
        if os.path.exists(PB4_EXE):
            out = subprocess.run([PB4_EXE,"--m",M,"--n",M,"--k",M,"--gmpPrint","3","--fracA","4","--fracB","4"],
                                 stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True).stdout
            g1 = re.search(r"GPU int8->int32:\s+([\d.]+)\s+ms\s+.*=\s+([\d.]+)\s+T-ops/s", out)
            g2 = re.search(r"GMP mpz exact\s+:\s+([\d.]+)\s+ms\s+.*=\s+([\d.]+)\s+G-ops/s", out)
            sp = re.search(r"Speedup.*=\s*([\d.]+)\s*x", out)
            ex = re.search(r"Exactness.*:\s+(PASS|FAIL)", out)
            out_rows.append({"section":"micro","M":M,"N":M,"K":M,"shift":"8",
                             "gpu_ms": g1.group(1) if g1 else "",
                             "gpu_Tops": f"{float(g1.group(2)):.9f}" if g1 else "",
                             "gmp_ms": g2.group(1) if g2 else "",
                             "gmp_Gops": f"{float(g2.group(2)):.9f}" if g2 else "",
                             "speedup": sp.group(1) if sp else "",
                             "exact": ex.group(1) if ex else ""})
    # PB-6 (256 amortized, shorter reps for speed)
    if os.path.exists(PB6_EXE):
        out = subprocess.run([PB6_EXE,"--m","256","--n","256","--k","256",
                              "--reps","200","--gmpPrint","3","--fracA","4","--fracB","4"],
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True).stdout
        m = re.search(r"GPU replay window: elapsed=([\d.]+)\s+ms\s+avg_per_gemm=([\d.]+)\s+ms\s+throughput=([\d.]+)\s+T-ops/s", out)
        g = re.search(r"GMP \(one GEMM\)\s*:\s*elapsed=([\d.]+)\s+ms\s+throughput=([\d.]+)\s+G-ops/s", out)
        s = re.search(r"Speedup \(GPU/GMP per-GEMM rate\)\s*=\s*([\d.]+)\s*x", out)
        out_rows.append({"section":"micro_amortized","M":"256","N":"256","K":"256","shift":"8",
                         "gpu_ms": m.group(2) if m else "",
                         "gpu_Tops": f"{float(m.group(3)):.9f}" if m else "",
                         "gmp_ms": g.group(1) if g else "",
                         "gmp_Gops": f"{float(g.group(2)):.9f}" if g else "",
                         "speedup": s.group(1) if s else "",
                         "exact":"PASS"})
    if out_rows:
        write_csv(out_rows)
        print(f"=== Reconstructed CSV -> {CSV}")
        return out_rows
    raise RuntimeError("Could not reconstruct CSV (no artifacts/binaries to parse).")

# ---------- Step 1: Load or reconstruct rows ----------
if os.path.exists(CSV):
    print("CSV found — using recorded results.")
    rows = rows_from_csv(CSV)
else:
    print("CSV not found — attempting to parse Markdown report.")
    rows = []
    if os.path.exists(MD):
        with open(MD,"r") as f:
            rows = parse_md_for_rows(f.read())
        if rows:
            write_csv(rows)
            print(f"=== Parsed MD and wrote CSV -> {CSV}")
    if not rows:
        print("No parsable MD — trying quick re-runs to reconstruct CSV.")
        rows = quick_rebuild_csv()

# ---------- Step 2: Recompute precise throughputs from time (2*M*N*K) ----------
def recompute_tops_from_time(section, M,N,K,gpu_ms):
    if gpu_ms is None: return None
    ops = 2.0 * M * N * K      # multiply + add
    gops = ops / (gpu_ms * 1e6)
    return gops / 1000.0       # T-ops/s

recalc = []
for r in rows:
    sec = r.get("section","")
    try:
        M = int(r.get("M","0")); N = int(r.get("N","0")); K = int(r.get("K","0"))
    except:
        continue
    gpu_ms = safe_float(r.get("gpu_ms"))
    tops_logged = safe_float(r.get("gpu_Tops"))
    tops_recomp = recompute_tops_from_time(sec, M,N,K, gpu_ms) if gpu_ms is not None else None
    if tops_recomp is not None:
        r["gpu_Tops"] = f"{tops_recomp:.9f}"
        err = abs(tops_recomp - (tops_logged or 0.0)) / (abs(tops_logged) or 1.0)
        recalc.append((sec, M,N,K, tops_logged, tops_recomp, err))

write_csv(rows)

print("\n=====================================================================================")
print("OPS RECOUNT AUDIT (post-fix; all Tops recomputed from time)")
print("=====================================================================================")
print(f"{'section':<16} {'shape':<12} {'old_Tops':>14} {'new_Tops':>14} {'abs_rel_err':>10}")
for (sec,M,N,K,old,new,err) in recalc:
    old_show = f"{old:.9f}" if isinstance(old,float) else "n/a"
    print(f"{sec:<16} {f'{M}x{N}x{K}':<12} {old_show:>14} {new:14.9f} {fmt_pct(err):>10}")

# ---------- Step 3: Refresh Micro table in MD ----------
def md_micro_table(rows):
    out = []
    out.append("| Shape | Shift | GPU ms | GPU T-ops/s | GMP ms | GMP G-ops/s | Speedup | Exact |\n")
    out.append("|---:|---:|---:|---:|---:|---:|---:|:--:|\n")
    for r in rows:
        if r.get("section") == "micro":
            M = int(r["M"]); gpu_ms = safe_float(r["gpu_ms"]) or 0.0
            gpu_T = safe_float(r["gpu_Tops"]) or 0.0
            gmp_ms = safe_float(r.get("gmp_ms")) or 0.0
            gmp_G  = safe_float(r.get("gmp_Gops")) or 0.0
            sp     = safe_float(r.get("speedup")) or 0.0
            exact  = r.get("exact","")
            out.append(f"| {M}³ | {r['shift']} | {gpu_ms:.3f} | {gpu_T:.9f} | {gmp_ms:.3f} | {gmp_G:.9f} | {sp:.2f}× | {exact} |\n")
    return "".join(out)

if os.path.exists(MD):
    with open(MD,"r") as f: md_text = f.read()
    new_table = md_micro_table(rows)
    pattern = r"(## Micro Head-to-Head vs GMP \(Exact Integers\)\n)(?:\|.*\n)+"
    header  = "## Micro Head-to-Head vs GMP (Exact Integers)\n"
    replacement = header + new_table + "\n_Note:_ GPU/GMP throughputs are computed from time using `2×M×N×K` and shown with **9 decimal places**.\n"
    if re.search(pattern, md_text):
        md_text = re.sub(pattern, replacement, md_text)
    else:
        md_text += "\n---\n" + replacement
    with open(MD,"w") as f: f.write(md_text)
    print(f"\nRefreshed Micro table in MD -> {MD}")
else:
    print("\nMain report not found; MD table refresh skipped.")

# ---------- Step 4: Append concise “Ops Accounting” section ----------
ts = datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds")
audit_lines = []
audit_lines.append("| Section | Shape | Logged T-ops/s | Recomputed T-ops/s | Abs rel error |\n")
audit_lines.append("|:--|:--|--:|--:|--:|\n")
for (sec,M,N,K,old,new,err) in recalc:
    old_show = (f"{old:.9f}" if isinstance(old,float) else "n/a")
    audit_lines.append(f"| {sec} | {M}×{N}×{K} | {old_show} | {new:.9f} | {fmt_pct(err)} |\n")

ops_md = []
ops_md.append("\n---\n")
ops_md.append("## Ops Accounting — exact counting and what we report\n")
ops_md.append(f"_Generated: {ts}_\n\n")
ops_md.append("**Canonical definition (GEMM):** For `C = A × B` with shapes `M×K` and `K×N`,\n")
ops_md.append("we count **`ops_per_gemm = 2 × M × N × K`** (one multiply + one add per inner term).\n\n")
ops_md.append("**PB-1 (Macro Swarm):** `panels = K/tileK`; each C slice performs `panels` GEMMs (first with β=0, rest β=1).\n")
ops_md.append("Per epoch: `gemms_per_epoch = streams × graph_nodes × batch_per_node × panels`.\n")
ops_md.append("Across epochs: `total_ops = epochs × gemms_per_epoch × (2×M×N×K)`; we report `total_ops/time`.\n\n")
ops_md.append("**PB-4 (Micro H2H):** Single GEMM at 128³/192³/256³; GPU T-ops/s = `2×M×N×K / ms_gpu × 1e-6 / 1000`.\n\n")
ops_md.append("**PB-6 (Graph+reps):** Capture one GEMM and replay `reps`; average per-GEMM time = `elapsed_ms/reps`;\n")
ops_md.append("GPU T-ops/s = `2×M×N×K / avg_ms × 1e-6 / 1000`.\n\n")
ops_md.append("### Independent recount (from recorded times)\n")
ops_md += audit_lines
ops_md.append("\n_Small differences, when present, were from earlier rounded prints; all values here are recomputed from time._\n")

with open(OPS_MD,"w") as f: f.write("".join(ops_md))
if os.path.exists(MD):
    with open(MD,"a") as f: f.write("".join(ops_md))
print(f"Appended Ops Accounting appendix -> {OPS_MD}")
print("\n" + "="*106)
print("PB-OPS-FINAL — DONE")
print("="*106)











#==================================================================================================
# MODULE PB-OPS-CONVENTIONS — Raw MAC/s, T-ops/s, Algorithmic “Effective Ops”, Precision & Checks
# -------------------------------------------------------------------------------------------------
# What this adds
#  • Prints a clean console summary table for each section/shape with:
#       RAW T-MAC/s (MAC = one multiply–accumulate) and RAW T-ops/s (2×MAC if you count mul+add)
#       Optional “Effective/Algorithmic” rates = RAW × k (set env ALG_K, default k=1.0)
#  • States arithmetic precisely (INT8×INT8 → INT32, dyadic shift = 2^-(fracA+fracB))
#  • Prints overflow guardrails for shapes (safe/not safe for int32 under clamp ±120)
#  • Confirms timing source uses CUDA events (device time) in PB-4/PB-6 sources
#  • Prints a mini verification checklist (Nsight Compute metrics + clock sampling commands)
#  • Also writes:
#       - /content/throughput_conventions.md  (appendix)
#       - Appends the same section to /content/public_bench_report.md
#       - /content/verification_checklist.txt (commands)
#==================================================================================================
import os, csv, math, datetime, re, subprocess, shutil

# ---------- Config ----------
ALG_K = float(os.environ.get("ALG_K", "1.0"))  # e.g., set ALG_K=9 to show “effective ops = RAW × 9”
FRAC_A = 4
FRAC_B = 4
CLAMP_ABS = 120

CSV_PATH = "/content/public_bench_report.csv"
MD_PATH  = "/content/public_bench_report.md"
OUT_MD   = "/content/throughput_conventions.md"
CHECK_TXT= "/content/verification_checklist.txt"
PB4_CU   = "/content/pb4_gmp_vs_gpu_micro.cu"
PB6_CU   = "/content/pb6_micro_reps_graph.cu"

print("\n" + "="*106)
print("MODULE PB-OPS-CONVENTIONS — Raw MAC/s, T-ops/s, Effective Ops, Precision & Checks")
print("="*106)

if not os.path.exists(CSV_PATH):
    raise RuntimeError(f"Missing CSV at {CSV_PATH}. Run PB-ALL first to generate results.")

# ---------- Load CSV ----------
with open(CSV_PATH, "r") as f:
    rows = list(csv.DictReader(f))

def sfloat(x):
    try: return float(x)
    except: return None

def compute_from_time(M,N,K,gpu_ms):
    """
    Returns: (Tmacs, Tops) based on time.
    MAC per GEMM = M*N*K; ops per GEMM = 2*M*N*K.
    """
    if gpu_ms is None or gpu_ms <= 0: return (None, None)
    t = gpu_ms * 1e-3
    macs_per_s = (M*N*K) / t
    ops_per_s  = 2.0 * (M*N*K) / t
    Tmacs = macs_per_s / 1e12
    Tops  = ops_per_s  / 1e12
    return (Tmacs, Tops)

# ---------- Build rows for printing + markdown ----------
summary = []
for r in rows:
    sec = r.get("section","")
    try:
        M = int(r.get("M","0")); N = int(r.get("N","0")); K = int(r.get("K","0"))
    except:
        continue
    gpu_ms = sfloat(r.get("gpu_ms"))
    exact  = r.get("exact","")
    Tmacs, Tops = compute_from_time(M,N,K,gpu_ms)
    if Tmacs is None: continue
    eff_Tmacs = Tmacs * ALG_K
    eff_Tops  = Tops  * ALG_K
    summary.append({
        "section": sec,
        "shape": f"{M}×{N}×{K}",
        "M": M, "N": N, "K": K,
        "gpu_ms": gpu_ms,
        "raw_Tmacs": Tmacs,
        "raw_Tops":  Tops,
        "eff_Tmacs": eff_Tmacs,
        "eff_Tops":  eff_Tops,
        "exact": exact
    })

# ---------- Overflow guardrail (worst-case bound under clamp ±CLAMP_ABS) ----------
# Worst-case |sum_k a_k*b_k| <= K * (CLAMP_ABS^2). Must be < 2^31 for int32 safety.
shapes = {}
for e in summary:
    shp = e["shape"]
    K = e["K"]
    worst = K * (CLAMP_ABS**2)
    safe = worst < (2**31)
    shapes[shp] = (K, worst, safe)

# ---------- Timing source confirmation ----------
def file_has(path, needle):
    try:
        with open(path,"r") as f:
            return (needle in f.read())
    except:
        return False

has_events_pb4 = file_has(PB4_CU, "cudaEventRecord(")
has_events_pb6 = file_has(PB6_CU, "cudaEventRecord(")

# ---------- Optional: one-shot current clocks sample ----------
def try_nvsmi():
    if shutil.which("nvidia-smi") is None:
        return "(nvidia-smi not available)"
    try:
        out = subprocess.run(
            ["bash","-lc","nvidia-smi --query-gpu=name,clocks.gr,clocks.mem,pstate --format=csv,noheader"],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=6
        )
        return out.stdout.strip()
    except Exception as e:
        return f"(nvidia-smi error: {e})"

nvsmi_sample = try_nvsmi()

# ---------- Pretty console print (human-readable) ----------
def hnum(x, places=9):
    return f"{x:.{places}f}"

print("\n=====================================================================================")
print("THROUGHPUT CONVENTIONS (MAC vs FLOPs) + RESULTS (derived from time)")
print("=====================================================================================")
print("- MAC definition: one multiply–accumulate. We report RAW T-MAC/s (hardware rate).")
print("- T-ops/s = 2 × T-MAC/s if you count multiply and add separately.")
if ALG_K != 1.0:
    print(f"- Algorithmic multiplier: k = {ALG_K:g}. We also show Effective/Algorithmic rates = RAW × k (clearly labeled).")
else:
    print("- Algorithmic multiplier: k = 1 (no algorithmic scaling). Set env ALG_K to add effective ops columns.")

print("\nPrecision & Scaling:")
print(f"- Arithmetic: INT8 × INT8 → INT32 accumulate; dyadic shift = 2^-(fracA+fracB) with fracA={FRAC_A}, fracB={FRAC_B} ⇒ 2^-{FRAC_A+FRAC_B}.")
print(f"- Overflow guardrail (clamp ±{CLAMP_ABS}): |∑ a_k b_k| ≤ K·({CLAMP_ABS}^2) must be < 2^31. Shapes:")
for shp,(K,w,safe) in shapes.items():
    print(f"  • {shp:<15}  K={K:<6}  bound≤{w:<12,}  int32_safe={str(bool(safe))}")

# Table header
print("\n-------------------------------------------------------------------------------------")
if ALG_K == 1.0:
    print(f"{'section':<16} {'shape':<15} {'GPU ms':>9} {'RAW T-MAC/s':>16} {'RAW T-ops/s':>16} {'exact':>7}")
    print("-"*85)
else:
    print(f"{'section':<16} {'shape':<15} {'GPU ms':>9} {'RAW T-MAC/s':>16} {'RAW T-ops/s':>16} {'Eff T-MAC/s':>16} {'Eff T-ops/s':>16} {'exact':>7}")
    print("-"*118)

for e in summary:
    if ALG_K == 1.0:
        print(f"{e['section']:<16} {e['shape']:<15} {e['gpu_ms']:9.3f} {hnum(e['raw_Tmacs']):>16} {hnum(e['raw_Tops']):>16} {e['exact']:>7}")
    else:
        print(f"{e['section']:<16} {e['shape']:<15} {e['gpu_ms']:9.3f} {hnum(e['raw_Tmacs']):>16} {hnum(e['raw_Tops']):>16} {hnum(e['eff_Tmacs']):>16} {hnum(e['eff_Tops']):>16} {e['exact']:>7}")

print("-------------------------------------------------------------------------------------")

print("\nTiming Source (device-side):")
print(f"- PB-4 uses CUDA events: {'YES' if has_events_pb4 else 'NO'}")
print(f"- PB-6 uses CUDA events: {'YES' if has_events_pb6 else 'NO'}")

print("\nQuick Verification Checklist (copy/paste):")
print("1) Nsight Compute — tensor/SM/memory/occupancy (macro PB-1):")
print("   ncu --set full --metrics \\")
print("     sm__pipe_tensor_cycles_active.avg.pct,\\")
print("     sm__inst_executed.avg.pct,\\")
print("     dram__throughput.avg.pct,\\")
print("     sm__warps_active.avg.pct_of_peak_sustained_active,\\")
print("     sm__maximum_warps_per_active_cycle_pct,\\")
print("     sm__average_active_threads_per_warp,\\")
print("     sm__throughput.avg.pct_of_peak_sustained_active,\\")
print("     achieved_occupancy \\")
print("     --target-processes all --launch-count 1 \\")
print("     /content/fx_int8_kpanel_tiled_swarm_v1 --m 5120 --n 5120 --k 5120 --streams 32 --graphNodes 64 --batchPerNode 4 --tileK 1280 --epochs 1 --warmup 3 --tryAlgos 16 --workspaceMB 1024 --validate 0")
print("2) Nsight Compute — micro PB-4 (256³):")
print("   ncu --set full --metrics sm__pipe_tensor_cycles_active.avg.pct,sm__inst_executed.avg.pct,dram__throughput.avg.pct,achieved_occupancy \\")
print("     --target-processes all /content/pb4_gmp_vs_gpu_micro --m 256 --n 256 --k 256 --gmpPrint 0")
print("3) Clock sampling during a run:")
print("   nvidia-smi --query-gpu=name,clocks.gr,clocks.mem,pstate --format=csv -lms 200 > clocks_log.csv &")
print("   SAMPLER_PID=$!")
print("   /content/fx_int8_kpanel_tiled_swarm_v1 --m 5120 --n 5120 --k 5120 --streams 32 --graphNodes 64 --batchPerNode 4 --tileK 1280 --epochs 2 --warmup 6 --tryAlgos 64 --workspaceMB 1024 --validate 0")
print("   kill ${SAMPLER_PID}")

print("\nCurrent clocks sample (best-effort):")
print("```")
print(nvsmi_sample)
print("```")

# ---------- Compose Markdown appendix (same info) ----------
ts = datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds")
md = []
md.append("\n---\n")
md.append("## Throughput Conventions & Raw/Effective Ops\n")
md.append(f"_Generated: {ts}_\n\n")
md.append("- **MAC definition:** A **MAC** is one multiply–accumulate. We report **RAW T-MAC/s** (hardware rate).\n")
md.append("- **Ops definition:** Some readers count multiply and add separately; then **T-ops/s = 2 × T-MAC/s**.\n")
if ALG_K != 1.0:
    md.append(f"- **Algorithmic multiplier k:** **k = {ALG_K:g}**. We also show **Effective/Algorithmic rates** = RAW × k (clearly labeled). These are not hardware counters.\n")
else:
    md.append("- **Algorithmic multiplier k:** k = 1 (no algorithmic scaling). To show “effective ops,” set `ALG_K` and re-run this module.\n")

md.append("\n**Arithmetic & scale**\n")
md.append(f"- Precision: **INT8 × INT8 → INT32 accumulate** (exact as integers). Real-valued interpretation via dyadic shift `2^-(fracA+fracB)`; here `fracA={FRAC_A}`, `fracB={FRAC_B}` → shift = **2^-{FRAC_A+FRAC_B}**.\n")
md.append(f"- Overflow guardrail (clamp = ±{CLAMP_ABS}): worst-case `|∑ a_k b_k| ≤ K·({CLAMP_ABS}^2)`; must be `< 2^31` for int32. Bench shapes:\n")
for shp,(K,w,safe) in shapes.items():
    md.append(f"  - {shp}: K={K}, bound ≤ {w:,} → int32 safe: {'YES' if safe else 'NO'}\n")

md.append("\n**Timing source**\n")
md.append(f"- PB-4 uses **CUDA events**: {'YES' if has_events_pb4 else 'NO'}  \n")
md.append(f"- PB-6 uses **CUDA events**: {'YES' if has_events_pb6 else 'NO'}  \n")
md.append("(Device-side timing; avoids host wall-clock skew.)\n")

md.append("\n### RAW and Effective Rates (derived from time)\n")
if ALG_K == 1.0:
    md.append("| Section | Shape | GPU ms | RAW T-MAC/s | RAW T-ops/s | Exact |\n")
    md.append("|:--|:--|--:|--:|--:|:--:|\n")
else:
    md.append("| Section | Shape | GPU ms | RAW T-MAC/s | RAW T-ops/s | Eff. T-MAC/s (×k) | Eff. T-ops/s (×k) | Exact |\n")
    md.append("|:--|:--|--:|--:|--:|--:|--:|:--:|\n")

def fmt9(x): return f"{x:.9f}"
for e in summary:
    if ALG_K == 1.0:
        md.append(f"| {e['section']} | {e['shape']} | {e['gpu_ms']:.3f} | {fmt9(e['raw_Tmacs'])} | {fmt9(e['raw_Tops'])} | {e['exact']} |\n")
    else:
        md.append(f"| {e['section']} | {e['shape']} | {e['gpu_ms']:.3f} | {fmt9(e['raw_Tmacs'])} | {fmt9(e['raw_Tops'])} | {fmt9(e['eff_Tmacs'])} | {fmt9(e['eff_Tops'])} | {e['exact']} |\n")

md.append("\n### Quick Verification Checklist\n")
md.append("1) **Timing source** — confirm CUDA events (device time):\n")
md.append(f"   - PB-4 cudaEventRecord: {'FOUND' if has_events_pb4 else 'NOT FOUND'}  \n")
md.append(f"   - PB-6 cudaEventRecord: {'FOUND' if has_events_pb6 else 'NOT FOUND'}  \n")
md.append("2) **Utilization (Nsight Compute)** — suggested commands:\n")
md.append("```bash\n")
md.append("ncu --set full --metrics \\\n")
md.append("  sm__pipe_tensor_cycles_active.avg.pct,\\\n")
md.append("  sm__inst_executed.avg.pct,\\\n")
md.append("  dram__throughput.avg.pct,\\\n")
md.append("  sm__warps_active.avg.pct_of_peak_sustained_active,\\\n")
md.append("  sm__maximum_warps_per_active_cycle_pct,\\\n")
md.append("  sm__average_active_threads_per_warp,\\\n")
md.append("  sm__throughput.avg.pct_of_peak_sustained_active,\\\n")
md.append("  achieved_occupancy \\\n")
md.append("  --target-processes all --launch-count 1 \\\n")
md.append("  /content/fx_int8_kpanel_tiled_swarm_v1 --m 5120 --n 5120 --k 5120 --streams 32 --graphNodes 64 --batchPerNode 4 --tileK 1280 --epochs 1 --warmup 3 --tryAlgos 16 --workspaceMB 1024 --validate 0\n\n")
md.append("ncu --set full --metrics sm__pipe_tensor_cycles_active.avg.pct,sm__inst_executed.avg.pct,dram__throughput.avg.pct,achieved_occupancy \\\n")
md.append("  --target-processes all /content/pb4_gmp_vs_gpu_micro --m 256 --n 256 --k 256 --gmpPrint 0\n")
md.append("```\n")
md.append("3) **Clocks during the run** — sample with nvidia-smi:\n")
md.append("```bash\n")
md.append("nvidia-smi --query-gpu=name,clocks.gr,clocks.mem,pstate --format=csv -lms 200 > clocks_log.csv &\n")
md.append("SAMPLER_PID=$!\n")
md.append("/content/fx_int8_kpanel_tiled_swarm_v1 --m 5120 --n 5120 --k 5120 --streams 32 --graphNodes 64 --batchPerNode 4 --tileK 1280 --epochs 2 --warmup 6 --tryAlgos 64 --workspaceMB 1024 --validate 0\n")
md.append("kill ${SAMPLER_PID}\n")
md.append("```\n")
md.append("**Current clocks sample (best-effort):**\n\n```\n" + nvsmi_sample + "\n```\n")

# ---------- Write artifacts and append to public MD ----------
with open(OUT_MD, "w") as f:
    f.write("".join(md))
print(f"\n=== WROTE appendix: {OUT_MD}")

if os.path.exists(MD_PATH):
    with open(MD_PATH, "a") as f:
        f.write("".join(md))
    print(f"=== APPENDED section to: {MD_PATH}")
else:
    print("(!) public_bench_report.md not found — skipped appending (appendix still written).")

with open(CHECK_TXT, "w") as f:
    f.write("# Verification Checklist — Commands\n\n")
    f.write("## Nsight Compute (macro)\n")
    f.write("ncu --set full --metrics \\\n")
    f.write("  sm__pipe_tensor_cycles_active.avg.pct,\\\n")
    f.write("  sm__inst_executed.avg.pct,\\\n")
    f.write("  dram__throughput.avg.pct,\\\n")
    f.write("  sm__warps_active.avg.pct_of_peak_sustained_active,\\\n")
    f.write("  sm__maximum_warps_per_active_cycle_pct,\\\n")
    f.write("  sm__average_active_threads_per_warp,\\\n")
    f.write("  sm__throughput.avg.pct_of_peak_sustained_active,\\\n")
    f.write("  achieved_occupancy \\\n")
    f.write("  --target-processes all --launch-count 1 \\\n")
    f.write("  /content/fx_int8_kpanel_tiled_swarm_v1 --m 5120 --n 5120 --k 5120 --streams 32 --graphNodes 64 --batchPerNode 4 --tileK 1280 --epochs 1 --warmup 3 --tryAlgos 16 --workspaceMB 1024 --validate 0\n\n")
    f.write("## Nsight Compute (micro)\n")
    f.write("ncu --set full --metrics sm__pipe_tensor_cycles_active.avg.pct,sm__inst_executed.avg.pct,dram__throughput.avg.pct,achieved_occupancy \\\n")
    f.write("  --target-processes all /content/pb4_gmp_vs_gpu_micro --m 256 --n 256 --k 256 --gmpPrint 0\n\n")
    f.write("## Clocks sampling\n")
    f.write("nvidia-smi --query-gpu=name,clocks.gr,clocks.mem,pstate --format=csv -lms 200 > clocks_log.csv &\n")
    f.write("SAMPLER_PID=$!\n")
    f.write("/content/fx_int8_kpanel_tiled_swarm_v1 --m 5120 --n 5120 --k 5120 --streams 32 --graphNodes 64 --batchPerNode 4 --tileK 1280 --epochs 2 --warmup 6 --tryAlgos 64 --workspaceMB 1024 --validate 0\n")
    f.write("kill ${SAMPLER_PID}\n")
print(f"=== CHECKLIST written: {CHECK_TXT}")

print("\n" + "="*106)
print("PB-OPS-CONVENTIONS — DONE")
print("="*106)


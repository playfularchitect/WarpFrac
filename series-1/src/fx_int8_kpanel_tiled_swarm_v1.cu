
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

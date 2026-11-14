
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

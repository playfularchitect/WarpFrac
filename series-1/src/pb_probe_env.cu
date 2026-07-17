
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

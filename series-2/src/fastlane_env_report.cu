// fastlane_env_report.cu — WarpFrac Series 2 environment/provenance snapshot
// Standalone source extracted from the Colab driver (scripts/run_fastlane_bench.py).
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

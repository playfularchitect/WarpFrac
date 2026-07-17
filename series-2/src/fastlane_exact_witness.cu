// fastlane_exact_witness.cu — WarpFrac Series 2 exactness witness (v6)
// Standalone source extracted from the Colab driver (scripts/run_fastlane_bench.py).
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

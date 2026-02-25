#pragma GCC optimize("Ofast, unroll-loops")
#pragma GCC target("avx2,fma")

#include "naive_gemm_omp.h"
#include <omp.h>

std::vector<float> NaiveGemmOMP(const std::vector<float>& a,
                                const std::vector<float>& b,
                                int n) {

    std::vector<float> c(n * n, 0.0f);

    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        for (int k = 0; k < n; k++) {
            float tmp = a[i * n + k];
            #pragma omp simd
            for (int j = 0; j < n; j++) {
                c[i * n + j] += tmp * b[k * n + j];
            }
        }
    }
    
    return c;
}
#pragma GCC optimize("Ofast, unroll-loops")
#pragma GCC target("avx2,fma")

#include "block_gemm_omp.h"

#include <algorithm>
#include <omp.h>

std::vector<float> BlockGemmOMP(const std::vector<float>& a,
                                const std::vector<float>& b,
                                int n) {
    std::vector<float> c(n * n, 0.0f);

    const int BLOCK = 64;
    
    #pragma omp parallel for schedule(static)
    for (int bi = 0; bi < n; bi += BLOCK) {
        for (int bj = 0; bj < n; bj += BLOCK) {
            for (int bk = 0; bk < n; bk += BLOCK) {

                for (int i = bi; i < std::min(bi + BLOCK, n); i++) {
                    for (int k = bk; k < std::min(bk + BLOCK, n); k++) {
                        float tmp = a[i * n + k];
                        #pragma omp simd
                        for (int j = bj; j < std::min(bj + BLOCK, n); j++) {
                            c[i * n + j] += tmp * b[k * n + j];
                        }
                    }
                }
            }
        }
    }

    return c;
}
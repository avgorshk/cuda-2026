#include "block_gemm_omp.h"
#include <vector>
#include <cmath>
#include <algorithm>
#include <omp.h>
#include <immintrin.h>
#include <random>
#include<chrono>
#include<iostream>

std::vector<float> BlockGemmOMP(const std::vector<float>& a,
    const std::vector<float>& b,
    int n) {

    std::vector<float> c(n * n, 0.0f);
    int B = 256;

#pragma omp parallel for schedule(static)
    for (int ii = 0; ii < n; ii += B) {
        for (int jj = 0; jj < n; jj += B) {
            for (int kk = 0; kk < n; kk += B) {

                int i_max = std::min(ii + B, n);
                int j_max = std::min(jj + B, n);
                int k_max = std::min(kk + B, n);

                for (int i = ii; i < i_max; ++i) {
                    for (int k = kk; k < k_max; ++k) {
                        float aik = a[i * n + k];
                        const float* __restrict pb = &b[k * n];
                        float* __restrict pc = &c[i * n];
#pragma omp simd
                        for (int j = jj; j < j_max; ++j)
                            pc[j] += aik * pb[j];
                    }
                }
            }
        }
    }
    return c;
}
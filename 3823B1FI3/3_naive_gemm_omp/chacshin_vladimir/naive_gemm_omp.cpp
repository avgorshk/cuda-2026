#include "naive_gemm_omp.h"
#include <vector>

std::vector<float> NaiveGemmOMP(const std::vector<float>& a,
    const std::vector<float>& b,
    int n)
{
    std::vector<float> c(n * n, 0.0f);
#pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        for (int k = 0; k < n; ++k) {
            float aik = a[i * n + k];
#pragma omp simd
            for (int j = 0; j < n; ++j) {
                c[i * n + j] += aik * b[k * n + j];
            }
        }
    }
    return c;
}
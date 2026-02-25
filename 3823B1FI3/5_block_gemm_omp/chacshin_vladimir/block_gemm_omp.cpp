#include "block_gemm_omp.h"
#include <vector>

std::vector<float> BlockGemmOMP(const std::vector<float>& a,
    const std::vector<float>& b,
    int n) {

    std::vector<float> c(n * n, 0.0f);
    int B = 256;
#pragma omp parallel for collapse(2)
    for (int ii = 0; ii < n; ii += B) {
        for (int jj = 0; jj < n; jj += B) {

            for (int kk = 0; kk < n; kk += B) {

                for (int i = ii; i < ii + B; ++i) {
                    for (int k = kk; k < kk + B; ++k) {
                        float aik = a[i * n + k];
#pragma omp simd
                        for (int j = jj; j < jj + B; ++j) {
                            c[i * n + j] += aik * b[k * n + j];
                        }
                    }
                }
            }
        }
    }
    return c;
}
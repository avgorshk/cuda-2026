//используемые ключи MSVC: /arch:AVX2, /fp:fast, /O2, /openmp, -openmp:experimental (для прагма симд)
#pragma GCC optimize("Ofast, unroll-loops")
#pragma GCC target("avx2,fma")

#include <vector>
#include <omp.h>

#include "naive_gemm_omp.h"


std::vector<float> NaiveGemmOMP(const std::vector<float>& a, const std::vector<float>& b, int n)
{
    std::vector<float> result(n * n);

#pragma omp parallel for
    for (int i = 0; i < n; i++)
    {
        for (int k = 0; k < n; k++)
        {
            float temp = a[i * n + k];
#pragma omp simd
            for (int j = 0; j < n; j++)
            {
                result[i * n + j] += temp * b[k * n + j];
            }
        }
    }
    return result;
}
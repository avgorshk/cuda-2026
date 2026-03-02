#pragma GCC optimize("Ofast")
#include "naive_gemm_omp.h"

std::vector<float> NaiveGemmOMP(const std::vector<float> &a,
                                const std::vector<float> &b,
                                int n)
{
    std::vector<float> res(n * n);

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; i++)
    {
        for (int k = 0; k < n; k++)
        {
            for (int j = 0; j < n; j++)
            {
                res[i * n + j] += a[i * n + k] * b[k * n + j];
            }
        }
    }

    return res;
}
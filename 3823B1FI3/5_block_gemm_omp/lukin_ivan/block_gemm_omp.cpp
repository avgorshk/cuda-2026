#include <vector>
#include <algorithm>
#include <omp.h>

#include "block_gemm_omp.h"

#pragma GCC optimize("Ofast, unroll-loops")
#pragma GCC target("avx2,fma") 

std::vector<float> BlockGemmOMP(const std::vector<float>& a, const std::vector<float>& b, int n)
{    
    std::vector<float> result(n * n, 0.0f); 
    int blockSize = 64; 

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; i += blockSize) 
    {
        for (int j = 0; j < n; j += blockSize)
        {
            for (int k = 0; k < n; k += blockSize) 
            {
                int max_i = std::min(i + blockSize, n);
                int max_k = std::min(k + blockSize, n);   
                int max_j = std::min(j + blockSize, n); 
                for (int ii = i; ii < max_i; ++ii) 
                {
                    for (int kk = k; kk < max_k; ++kk) 
                    {
                        float temp = a[ii * n + kk];
                        #pragma omp simd
                        for (int jj = j; jj < max_j; ++jj) 
                        {
                            result[ii * n + jj] += temp * b[kk * n + jj];
                        }
                    }
                }
            }
        }
    }
    return result;
}
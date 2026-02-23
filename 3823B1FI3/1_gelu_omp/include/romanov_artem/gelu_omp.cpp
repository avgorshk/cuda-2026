#pragma GCC optimize("O3")
#pragma GCC optimize("fast-math")
#pragma GCC optimize("unroll-loops")

#include "gelu_omp.h"

#include <omp.h>
#include <vector>
#include <cmath>

#define E 2.7182818284f
#define PI 3.1415926535f

// sqrtf(2/PI) 
#define C1 0.797884561f
#define _2C1 1.595769122f

#define C2 0.044715f

#pragma GCC target("avx2")
std::vector<float> GeluOMP(const std::vector<float>& input) {
    int n = input.size();

    std::vector<float> result(n);

    const float* input_ptr = input.data();
    float* result_ptr = result.data();

    constexpr int BLOCK_SIZE = 32;
    int BLOCKS_CNT = n / BLOCK_SIZE;

#pragma omp parallel for
    for (int i = 0; i < BLOCKS_CNT; ++i) {
        int l = i * BLOCK_SIZE;
        int r = l + BLOCK_SIZE;
        #pragma omp simd
        for (int j = l; j < r; ++j) {
            float x = input_ptr[j];
            float _2y = x * _2C1 * (1.0f + C2 * x * x);
            float e2y = expf(_2y);
            float res = x * (1.0f - 1.0f / (e2y + 1.0f));
            result_ptr[j] = res;
        }
    }

    for (int i = n - n % BLOCK_SIZE; i < n; ++i) {
        float x = input_ptr[i];
        result_ptr[i] = x * (1.0f - 1.0f / (expf(x * _2C1 * (1.0f + C2 * x * x)) + 1.0f));
    }

    return result;  
}
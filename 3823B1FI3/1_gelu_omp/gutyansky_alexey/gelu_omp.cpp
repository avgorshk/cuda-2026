// Compiler flags used: -O3 -mavx2 -ffast-math -fopenmp -flto

#pragma GCC optimize("Ofast")

#include "gelu_omp.h"

#include <vector>
#include <cmath>

#pragma GCC target("avx2,fma")
std::vector<float> GeluOMP(const std::vector<float>& input) {
    const int N = input.size();
    std::vector<float> output(N);

    const float _2SQRT2PI = 2.0f * sqrtf(2.0f / M_PI);
    const float C1 = 0.044715f;

    #pragma omp parallel for
    for (int i = 0; i < N; i++) {
        float x = input[i];

        float x3 = x * x * x;
        float arg = _2SQRT2PI * (x + C1 * x3);
        float ex = expf(-arg);

        output[i] = x / (1.0f + ex);
    }

    return output;
}
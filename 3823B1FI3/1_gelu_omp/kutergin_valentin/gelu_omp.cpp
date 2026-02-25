#pragma GCC optimize("Ofast")
#pragma GCC optimize("unroll-loops")
#pragma GCC target("avx")
#pragma GCC target("avx2")
#pragma GCC target("fma")

#include "gelu_omp.h"
#include <cmath>
#include <omp.h>

std::vector<float> GeluOMP(const std::vector<float>& input) {
    size_t n = input.size();
    std::vector<float> output(n);

    const float double_sqrt_2_over_pi = 1.59576912f; // 2 * sqrt(2 / pi)
    const float coeff = 0.044715f;

    // parallel for - выполнение цикла for в нескольких потоках
    // simd - использование векторых инструкций 
    // schedule(static) - фиксированное распределение итераций цикла между потоками
    #pragma omp parallel for simd schedule(static)
    for (int i = 0; i < (int)n; ++i) {
        float x = input[i];
        float arg = double_sqrt_2_over_pi * (x + coeff * x * x * x);
        output[i] = x / (1.0f + std::exp(-arg)); // формула, аппроксимирующая GELU
    }

    return output;
}
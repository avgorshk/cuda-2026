#include "gelu_omp.h"

#include <omp.h>
#include <vector>
#include <cmath>

#define C1 1.595769122f
#define C2 0.044715f

#pragma GCC optimize("Ofast,unroll-loops")
#pragma GCC target("avx2")
std::vector<float> GeluOMP(const std::vector<float>& input) {
    int _size = input.size();
    float x;
    float _exp;
    float tmp;

    std::vector<float> result(_size);

    const float* input_ptr = input.data();
    float* result_ptr = result.data();

#pragma omp parallel for simd schedule(static)
    for (int i = 0; i < _size; ++i) {
        x = input_ptr[i];
        _exp = std::exp2f(-(x * C1 * (1.0f + C2 * x * x)));
        tmp = x / (_exp + 1.0f);
        result_ptr[i] = tmp;
    }

    return result;  
}

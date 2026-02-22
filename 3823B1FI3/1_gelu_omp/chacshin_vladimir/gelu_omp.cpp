#include <cmath>
#include <omp.h>
#include <vector>
#include <iostream>
#include <chrono>
#include <random>

#pragma GCC optimize("O3")
#pragma GCC optimize("fast-math")

#pragma GCC target("avx2")
std::vector<float> GeluOMP(const std::vector<float>& input) {
    size_t n = input.size();
    std::vector<float> output(n);
    //разделяем на блоки, которые заполнят L2 кеш
    const size_t block_size = 16800;
#pragma omp parallel for schedule(guided)
    for (size_t b = 0; b < n; b += block_size) {
        size_t end = std::min(b + block_size, n);
        size_t i;
#pragma omp simd
        for (i = b; i < end; i += 1) {
            float x0 = input[i + 0];
            float t0 = 1.702f * x0;
            float s0 = ((-0.004f * t0 + 0.197f) * t0 + 0.5f);
            output[i + 0] = x0 * s0;
        }
    }
    return output;

}
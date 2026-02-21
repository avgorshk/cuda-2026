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
    const size_t block_size = 290000;
#pragma omp parallel for schedule(static)
    for (size_t b = 0; b < n; b += block_size) {
        size_t end = std::min(b + block_size, n);
        size_t i;
#pragma omp simd
        //разворачиваем цикл для AVX2
        for (i = b; i < end - 7; i += 8) {
            float x0 = input[i + 0];
            float x1 = input[i + 1];
            float x2 = input[i + 2];
            float x3 = input[i + 3];
            float x4 = input[i + 4];
            float x5 = input[i + 5];
            float x6 = input[i + 6];
            float x7 = input[i + 7];

            float t0 = 1.702f * x0;
            float t1 = 1.702f * x1;
            float t2 = 1.702f * x2;
            float t3 = 1.702f * x3;
            float t4 = 1.702f * x4;
            float t5 = 1.702f * x5;
            float t6 = 1.702f * x6;
            float t7 = 1.702f * x7;

            //используем полиномиальную аппроксимацию sigmoid(t)
            float s0 = ((-0.004f * t0 + 0.197f) * t0 + 0.5f);
            float s1 = ((-0.004f * t1 + 0.197f) * t1 + 0.5f);
            float s2 = ((-0.004f * t2 + 0.197f) * t2 + 0.5f);
            float s3 = ((-0.004f * t3 + 0.197f) * t3 + 0.5f);
            float s4 = ((-0.004f * t4 + 0.197f) * t4 + 0.5f);
            float s5 = ((-0.004f * t5 + 0.197f) * t5 + 0.5f);
            float s6 = ((-0.004f * t6 + 0.197f) * t6 + 0.5f);
            float s7 = ((-0.004f * t7 + 0.197f) * t7 + 0.5f);

            output[i + 0] = x0 * s0;
            output[i + 1] = x1 * s1;
            output[i + 2] = x2 * s2;
            output[i + 3] = x3 * s3;
            output[i + 4] = x4 * s4;
            output[i + 5] = x5 * s5;
            output[i + 6] = x6 * s6;
            output[i + 7] = x7 * s7;
        }
        //обрабатываем хвост
        for (; i < end; ++i) {
            float x = input[i];
            float t = 1.702f * x;
            float s = ((-0.004f * t + 0.197f) * t + 0.5f);
            output[i] = x * s;
        }
    }
    return output;

}
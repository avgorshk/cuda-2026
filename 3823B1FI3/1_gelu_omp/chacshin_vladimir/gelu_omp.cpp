#include <cmath>
#include <algorithm>
#include <omp.h>
#include <immintrin.h>
#include <random>
#include<chrono>
#include<iostream>
#include "gelu_omp.h"


/*inline __m256 exp256_ps_high_prec(__m256 x) {
    const __m256 ln2 = _mm256_set1_ps(0.6931471805599453f);
    const __m256 inv_ln2 = _mm256_set1_ps(1.4426950408889634f);
    const __m256 c1 = _mm256_set1_ps(1.0f / 2.0f);
    const __m256 c2 = _mm256_set1_ps(1.0f / 6.0f);
    const __m256 c3 = _mm256_set1_ps(1.0f / 24.0f);
    const __m256 c4 = _mm256_set1_ps(1.0f / 120.0f);
    const __m256 c5 = _mm256_set1_ps(1.0f / 720.0f);
    const __m256 c6 = _mm256_set1_ps(1.0f / 5040.0f);
    const __m256 c7 = _mm256_set1_ps(1.0f / 40320.0f);
    const __m256 one = _mm256_set1_ps(1.0f);


    __m256 t = _mm256_mul_ps(x, inv_ln2);
    __m256i n = _mm256_cvttps_epi32(t);


    __m256 f = _mm256_sub_ps(x, _mm256_mul_ps(_mm256_cvtepi32_ps(n), ln2));


    __m256 f2 = _mm256_mul_ps(f, f);
    __m256 f3 = _mm256_mul_ps(f2, f);
    __m256 f4 = _mm256_mul_ps(f3, f);
    __m256 f5 = _mm256_mul_ps(f4, f);
    __m256 f6 = _mm256_mul_ps(f5, f);
    __m256 f7 = _mm256_mul_ps(f6, f);
    __m256 f8 = _mm256_mul_ps(f7, f);

    __m256 poly = _mm256_add_ps(one,
        _mm256_add_ps(f,
            _mm256_add_ps(_mm256_mul_ps(f2, c1),
                _mm256_add_ps(_mm256_mul_ps(f3, c2),
                    _mm256_add_ps(_mm256_mul_ps(f4, c3),
                        _mm256_add_ps(_mm256_mul_ps(f5, c4),
                            _mm256_add_ps(_mm256_mul_ps(f6, c5),
                                _mm256_add_ps(_mm256_mul_ps(f7, c6),
                                    _mm256_mul_ps(f8, c7)
                                ))))))));


    __m256i e = _mm256_add_epi32(n, _mm256_set1_epi32(127));
    e = _mm256_slli_epi32(e, 23);
    __m256 pow2n = _mm256_castsi256_ps(e);

    return _mm256_mul_ps(poly, pow2n);
}

std::vector<float> GeluOMP(const std::vector<float>& input) {
    size_t n = input.size();
    std::vector<float> output(n);
    int block_size = 16800;


    const float alpha = 0.79788456f;
    const float beta = 0.044715f;

#pragma omp parallel for schedule(guided)
    for (size_t b = 0; b < n; b += block_size) {
        size_t end = std::min(b + block_size, n);
        size_t i;

        for (i = b; i < end - 7; i += 8) {
            __m256 x = _mm256_loadu_ps(input.data() + i);
            __m256 x2 = _mm256_mul_ps(x, x);
            __m256 x3 = _mm256_mul_ps(x2, x);

            __m256 u = _mm256_fmadd_ps(_mm256_set1_ps(beta), x3, x);
            u = _mm256_mul_ps(u, _mm256_set1_ps(alpha));

            __m256 two_u = _mm256_add_ps(u, u);
            __m256 exp2u = exp256_ps_high_prec(two_u);

            __m256 tanh_u = _mm256_sub_ps(_mm256_set1_ps(1.0f),
                _mm256_div_ps(_mm256_set1_ps(2.0f),
                    _mm256_add_ps(exp2u, _mm256_set1_ps(1.0f))));

            __m256 res = _mm256_mul_ps(_mm256_set1_ps(0.5f),
                _mm256_mul_ps(x, _mm256_add_ps(_mm256_set1_ps(1.0f), tanh_u)));

            _mm256_storeu_ps(output.data() + i, res);
        }
#pragma omp simd
        for (int j = i; j < end; ++j) {
            float x = input[j];
            float x3 = x * x * x;
            float u = alpha * (x + beta * x3);
            float tanh_u = 1.0f - 2.0f / (expf(2.0f * u) + 1.0f);
            output[j] = 0.5f * x * (1.0f + tanh_u);
        }
    }

    return output;
}*/

#include "gelu_omp.h"
#include <vector>
#include <cmath>
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
#include <cmath>
#include <algorithm>
#include <omp.h>
#include <immintrin.h>
#include <random>
#include<chrono>
#include<iostream>
#include "gelu_omp.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <iomanip>


inline __m256 exp256_ps_small(__m256 x) {
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

inline __m256 _mm256_qexp_estrin_ps(__m256 _x)
{
    const __m256 _one = _mm256_set1_ps(1.0f);
    const __m256 _ln2r = _mm256_set1_ps(1.44269504088896341f);

    const __m256 _ln2a = _mm256_set1_ps(-0.693145751953125f);
    const __m256 _ln2b = _mm256_set1_ps(-1.428606820309417e-6f);

    const __m256 _c7 = _mm256_set1_ps(1.9875691500E-4f);
    const __m256 _c6 = _mm256_set1_ps(1.3981999507E-3f);
    const __m256 _c5 = _mm256_set1_ps(8.3334519073E-3f);
    const __m256 _c4 = _mm256_set1_ps(4.1665795894E-2f);
    const __m256 _c3 = _mm256_set1_ps(1.6666665459E-1f);
    const __m256 _c2 = _mm256_set1_ps(5.0000001201E-1f);

    __m256 _n = _mm256_round_ps(
        _mm256_mul_ps(_x, _ln2r),
        _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);

    __m256 _g = _mm256_fmadd_ps(_n, _ln2b,
        _mm256_fmadd_ps(_n, _ln2a, _x));

    __m256 _g2 = _mm256_mul_ps(_g, _g);
    __m256 _g4 = _mm256_mul_ps(_g2, _g2);

    __m256 _p1 = _mm256_fmadd_ps(_c7, _g, _c6);
    __m256 _p2 = _mm256_fmadd_ps(_c5, _g, _c4);
    __m256 _p3 = _mm256_fmadd_ps(_p1, _g2, _p2);

    __m256 _p4 = _mm256_fmadd_ps(_c3, _g, _c2);
    __m256 _p5 = _mm256_fmadd_ps(_p4, _g2, _mm256_add_ps(_g, _one));

    __m256 _y = _mm256_fmadd_ps(_p3, _g4, _p5);

    __m256i _n_int = _mm256_cvtps_epi32(_n);
    __m256i _exp_offset = _mm256_add_epi32(_n_int, _mm256_set1_epi32(127));
    __m256i _exp_shifted = _mm256_slli_epi32(_exp_offset, 23);
    __m256 _two_pow_n = _mm256_castsi256_ps(_exp_shifted);

    return _mm256_mul_ps(_y, _two_pow_n);
}

std::vector<float> GeluOMP(const std::vector<float>& input) {
    size_t n = input.size();
    std::vector<float> output(n);
    int block_size = 12400;


    const float alpha = 0.79788456f;
    const float beta = 0.044715f;
    const __m256 thresh_high = _mm256_set1_ps(10.0f);
    const __m256 thresh_low = _mm256_set1_ps(-9.0f);
    const __m256 zero_vec = _mm256_setzero_ps();

#pragma omp parallel for schedule(guided)
    for (size_t b = 0; b < n; b += block_size) {
        size_t end = std::min(b + block_size, n);
        size_t i;

        for (i = b; i < end - 7; i += 8) {
            _mm_prefetch((const char*)(input.data() + i + 128), _MM_HINT_T0);
            __m256 x = _mm256_loadu_ps(input.data() + i);

            __m256 mask_high = _mm256_cmp_ps(x, thresh_high, _CMP_GT_OS);   // x > 10
            __m256 mask_low = _mm256_cmp_ps(x, thresh_low, _CMP_LT_OS);    // x < -9

            __m256 x2 = _mm256_mul_ps(x, x);
            __m256 x3 = _mm256_mul_ps(x2, x);

            __m256 u = _mm256_fmadd_ps(_mm256_set1_ps(beta), x3, x);
            u = _mm256_mul_ps(u, _mm256_set1_ps(alpha));

            __m256 two_u = _mm256_add_ps(u, u);

            __m256 exp2u = _mm256_qexp_estrin_ps(two_u);


            __m256 tanh_u = _mm256_sub_ps(_mm256_set1_ps(1.0f),
                _mm256_div_ps(_mm256_set1_ps(2.0f),
                    _mm256_add_ps(exp2u, _mm256_set1_ps(1.0f))));

            __m256 res = _mm256_mul_ps(_mm256_set1_ps(0.5f),
                _mm256_mul_ps(x, _mm256_add_ps(_mm256_set1_ps(1.0f), tanh_u)));

            res = _mm256_blendv_ps(res, x, mask_high); // если x>10 ставим x
            res = _mm256_blendv_ps(res, zero_vec, mask_low); // если x<-9 ставим 0

            _mm256_stream_ps(output.data() + i, res);
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
    _mm_sfence();
    return output;
}

double test_performance(const std::vector<float>& input, size_t repeats = 4, int block_size = 16800) {
    // Warming-up
    GeluOMP(input);
    // Performance Measuring
    std::vector<double> min_times;
    for (int j = 0; j < repeats; j++) {
        std::vector<double> time_list;
        for (size_t i = 0; i < 4; ++i) {
            auto start = std::chrono::high_resolution_clock::now();
            GeluOMP(input);
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> duration = end - start;
            time_list.push_back(duration.count());
        }
        double min_time = *std::min_element(time_list.begin(), time_list.end());
        min_times.push_back(min_time);
    }
    double avg = std::accumulate(min_times.begin(), min_times.end(), 0.0) / min_times.size();
    std::cout << "Min execution time for block size " << block_size << " over " << repeats << " runs: "
        << *std::min_element(min_times.begin(), min_times.end()) << " seconds\n"<<"aver :"<<avg<<" seconds\n";
    return avg;
}

/*int main() {
    omp_set_num_threads(4);
    using clock = std::chrono::high_resolution_clock;

    std::vector<float> input2(134217728);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-100,100);

    for (auto& x : input2) x = dist(rng);

    test_performance(input2, 100);

    return 0;
}*/
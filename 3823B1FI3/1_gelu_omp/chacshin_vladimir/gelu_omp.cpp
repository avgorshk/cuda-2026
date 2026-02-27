#include <cmath>
#include <algorithm>
#include <omp.h>
#include <immintrin.h>
#include <random>
#include<chrono>
#include<iostream>
#include "gelu_omp.h"


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

std::vector<float> GeluOMP(const std::vector<float>& input) {
    size_t n = input.size();
    std::vector<float> output(n);
    int block_size = 16800;


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
            __m256 x = _mm256_loadu_ps(input.data() + i);

            __m256 mask_high = _mm256_cmp_ps(x, thresh_high, _CMP_GT_OS);   // x > 10
            __m256 mask_low = _mm256_cmp_ps(x, thresh_low, _CMP_LT_OS);    // x < -9

            __m256 x2 = _mm256_mul_ps(x, x);
            __m256 x3 = _mm256_mul_ps(x2, x);

            __m256 u = _mm256_fmadd_ps(_mm256_set1_ps(beta), x3, x);
            u = _mm256_mul_ps(u, _mm256_set1_ps(alpha));

            __m256 two_u = _mm256_add_ps(u, u);

            __m256 exp2u = exp256_ps_small(two_u);


            __m256 tanh_u = _mm256_sub_ps(_mm256_set1_ps(1.0f),
                _mm256_div_ps(_mm256_set1_ps(2.0f),
                    _mm256_add_ps(exp2u, _mm256_set1_ps(1.0f))));

            __m256 res = _mm256_mul_ps(_mm256_set1_ps(0.5f),
                _mm256_mul_ps(x, _mm256_add_ps(_mm256_set1_ps(1.0f), tanh_u)));

            res = _mm256_blendv_ps(res, x, mask_high); // если x>10 ставим x
            res = _mm256_blendv_ps(res, zero_vec, mask_low); // если x<-9 ставим 0

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
}

/*double test_performance(const std::vector<float>& input, size_t repeats = 4, int block_size = 16800) {
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

int main() {
    FILE* fgold = fopen("golden_100_100.bin", "rb");
    if (!fgold) {
        std::cerr << "Cannot open golden file\n";
        return 1;
    }

    std::vector<float> input;
    std::vector<float> golden;

    // читаем из stdin и golden одновременно
    float x, y;
    while (fread(&x, sizeof(float), 1, stdin) == 1 &&
        fread(&y, sizeof(float), 1, fgold) == 1) {
        input.push_back(x);
        golden.push_back(y);
    }

    fclose(fgold);

    std::vector<float> outp;
    outp = GeluOMP(input);
    for (int i=0;i< input.size();i++){
        if (std::abs(outp[i] - golden[i]) > 1e-7) {
            std::cout<<i
                << "inp:" << input[i]
                << " outp:" << outp[i]
                << " exp:" << golden[i]
                << " diff:" << (outp[i] - golden[i])
                << "False"
                << "\n";
        }
        else {
            std::cout << i
                << "inp:" << input[i]
                << " outp:" << outp[i]
                << " exp:" << golden[i]
                << " diff:" << (outp[i] - golden[i])
                << "\n";
        }
    }
    omp_set_num_threads(4);
    using clock = std::chrono::high_resolution_clock;

    std::vector<float> input2(134217728); // пример: 16M элементов
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-100,100);

    for (auto& x : input2) x = dist(rng);

    test_performance(input2, 20);

    return 0;
}*/
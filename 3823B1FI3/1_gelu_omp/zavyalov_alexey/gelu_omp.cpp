#include "gelu_omp.h"

std::vector<float> GeluOMP(const std::vector<float>& input) {
    std::vector<float> res(input.size());
    int n = input.size();
#pragma omp parallel for
    for (int i = 0; i < n; i++) {
        float x = input[i];
        float sqrt_2_div_pi = 0.797884560802865355879892f;
        float tanh_arg = sqrt_2_div_pi * (x + 0.044715 * x * x * x);
        float exp_x = exp(tanh_arg);
        float exp_minus_x = exp(-tanh_arg);
        float tanh_cherez_exp = (exp_x - exp_minus_x) / (exp_x + exp_minus_x);
        res[i] = 0.5 * x * (1 + tanh_cherez_exp);
    }
    return res;
}

std::vector<float> GeluOMP_slow(const std::vector<float>& input) {
    std::vector<float> res(input.size());
    int n = input.size();
    // #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        float x = input[i];

        res[i] = 0.5 * x * (1 + tanh(0.797884560802865355879892f * (x + 0.044715 * x * x * x)));
    }
    return res;
}

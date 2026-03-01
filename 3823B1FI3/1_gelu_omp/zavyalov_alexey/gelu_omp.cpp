#include "gelu_omp.h"

std::vector<float> GeluOMP(const std::vector<float>& input) {
    std::vector<float> res(input.size());
    int n = input.size();
    int i = 0;
    float sqrt_2_div_pi = 0.7978845608028653558798921198687637369517172623298693153318516593f;

#pragma omp parallel for
    for (i = 0; i < n - 8; i += 8) {
        float x = input[i];
        float tanh_arg = sqrt_2_div_pi * (x + 0.044715f * x * x * x);
        float exp_precalc = exp(2.0f * tanh_arg);
        float cnst = (1.0f - 1.0f / (exp_precalc + 1.0f));
        res[i] = x * cnst;
        //res[i] = 0.5f * x * (1 + tanh(tanh_arg));

        x = input[i + 1];
        tanh_arg = sqrt_2_div_pi * (x + 0.044715f * x * x * x);
        exp_precalc = exp(2.0f * tanh_arg);
        cnst = (1.0f - 1.0f / (exp_precalc + 1.0f));
        res[i + 1] = x * cnst;

        x = input[i + 2];
        tanh_arg = sqrt_2_div_pi * (x + 0.044715f * x * x * x);
        exp_precalc = exp(2.0f * tanh_arg);
        cnst = (1.0f - 1.0f / (exp_precalc + 1.0f));
        res[i + 2] = x * cnst;

        x = input[i + 3];
        tanh_arg = sqrt_2_div_pi * (x + 0.044715f * x * x * x);
        exp_precalc = exp(2.0f * tanh_arg);
        cnst = (1.0f - 1.0f / (exp_precalc + 1.0f));
        res[i + 3] = x * cnst;

        x = input[i + 4];
        tanh_arg = sqrt_2_div_pi * (x + 0.044715f * x * x * x);
        exp_precalc = exp(2.0f * tanh_arg);
        cnst = (1.0f - 1.0f / (exp_precalc + 1.0f));
        res[i + 4] = x * cnst;

        x = input[i + 5];
        tanh_arg = sqrt_2_div_pi * (x + 0.044715f * x * x * x);
        exp_precalc = exp(2.0f * tanh_arg);
        cnst = (1.0f - 1.0f / (exp_precalc + 1.0f));
        res[i + 5] = x * cnst;

        x = input[i + 6];
        tanh_arg = sqrt_2_div_pi * (x + 0.044715f * x * x * x);
        exp_precalc = exp(2.0f * tanh_arg);
        cnst = (1.0f - 1.0f / (exp_precalc + 1.0f));
        res[i + 6] = x * cnst;

        x = input[i + 7];
        tanh_arg = sqrt_2_div_pi * (x + 0.044715f * x * x * x);
        exp_precalc = exp(2.0f * tanh_arg);
        cnst = (1.0f - 1.0f / (exp_precalc + 1.0f));
        res[i + 7] = x * cnst;
    }

    for (int i = std::max(0, n - 16); i < n; i++) {
        float x = input[i];
        float tanh_arg = sqrt_2_div_pi * (x + 0.044715f * x * x * x);
        float exp_precalc = exp(2.0f * tanh_arg);
        float cnst = (1.0f - 1.0f / (exp_precalc + 1.0f));
        res[i] = x * cnst;
    }
    return res;
}

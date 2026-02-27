#include <vector>
#include <cmath>
#include <omp.h>
#include "gelu_omp.h"

const float COEF = 1.595769f; // 2 * sqrt(2 / pi)
const float COEFF = 0.044715f;

std::vector<float> GeluOMP(const std::vector<float>& input)
{
    size_t size = input.size();
    std::vector<float> output(size);
#pragma omp parallel for simd schedule(static)
    for (size_t i = 0; i < size; ++i) {
        float x = input[i];
        float argument = COEF * (x + COEFF * x * x * x);
        output[i] = x - x / (expf(argument) + 1.0f);
    }
    return output;
}

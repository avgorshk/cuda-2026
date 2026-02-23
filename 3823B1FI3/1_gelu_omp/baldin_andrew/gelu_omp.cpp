#pragma GCC optimize("Ofast,unroll-loops")
#pragma GCC target("avx2,fma")

#include "gelu_omp.h"
#include <cmath>
#include <omp.h>

/*
GELU(x) = 0.5 * x * (1 + tanh(z))
z = sqrt(2/pi) * (x + 0.044715 * x^3)

tanh(z) = (e^z - e^-z) / (e^z + e^-z):
1 + tanh(z) = 1 + (e^z - e^-z) / (e^z + e^-z)
            = (e^z + e^-z + e^z - e^-z) / (e^z + e^-z)
            = 2 * e^z / (e^z + e^-z)

1 + tanh(z) = 2 / (1 + e^(-2z))
GELU(x) = 0.5 * x * (2 / (1 + e^(-2z))) = x / (1 + e^(-2z))
*/

const float SQRT_2_OVER_PI = 0.7978845608f;
const float COEFF = 0.044715f;

std::vector<float> GeluOMP(const std::vector<float>& input) {
    size_t n = input.size();
    
    std::vector<float> output(n);

    #pragma omp parallel for simd schedule(static)
    for (size_t i = 0; i < n; i++) {
        float x = input[i];
        float x3 = x * x * x;
        float arg = SQRT_2_OVER_PI * (x + COEFF * x3);
        output[i] = x / (1.0f + std::exp(-2.0f * arg));
    }

    return output;
}

#include <vector>
#include <cmath>
#include <omp.h>

#include "gelu_omp.h"

const float PI_coeff = 0.797884F;
const float coeff = 0.044715F;
const int elem_count = 134217728;

//Используемые ключи MSVC: /arch:AVX2, /fp:fast, /O2, /openmp
//Ручное разворачивание цикла, как и разворачивание с помощью #pragma unroll только замедлило код

//Основные моменты
//1. Использование прагмы параллель
//2. Сокращение преобразований типов в цикле
//3. Сведение к минимуму вызовов функций в теле цикла
//4. Тангенс вычисляется через exp (причем прямо в цикле)

std::vector<float> GeluOMP(const std::vector<float>& input)
{
    int size = static_cast<int>(input.size());

    std::vector<float> result(size);
    
#pragma omp parallel for schedule(static) num_threads(omp_get_max_threads())
    for (int i = 0; i < size; i++)
    {
        float x = input[i];
        float exp_value = expf(2.0F * PI_coeff * (x + coeff * x * x * x));
        float tanh_value = 1.0F - 2.0F / (exp_value + 1.0F);
        result[i] = 0.5F * x * (1.0F + tanh_value);
    }
    return result;
}
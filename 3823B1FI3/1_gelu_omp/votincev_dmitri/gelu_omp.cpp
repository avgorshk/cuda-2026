#include "gelu_omp.h"
#include <cmath>
#include <omp.h>
#include <iostream>

/*
Ключи в Visual Studio:
/openmp (/openmp:llvm) - многопоточность
/fp:fast вместо /fp:precise - быстрые floating point (expf становится быстрее)
/arch:AVX2 - векторные инструкции (в регистре одновременно 8 float)
AVX2 регстры: 256 бит; float: 32 бит; 256/32 = 8 
/GL - глобальная оптимизация
/O2  - агрессивная оптимизация
*/


// Оптимизация гиперболического тангенса тангенса (Hint 1):
// базовая версия: tanh(N) = (exp(N) - exp(-N)) / (exp(N) + exp(-N))
// можно заметить: 4 вызова exp
// улучшение: для гиперболического тангенса есть формула: 1.0f - 2.0f / (expf(2.0f * x) + 1.0f).
// чем лучше: exp вызывается всего 1 раз
inline float fast_tanh(float x) {
    return 1.0f - 2.0f / (expf(2.0f * x) + 1.0f);
}





// GELU(x) = 1/2 * x * (1+ tanh(sqrt(2/pi)*(x+0.044715*x^3)))
// 
// GELU(x) = a1 * a2
// a1 = 0.5f * x
// a2 = 1.0f + tanh(c1*f1)
// c1 = sqrtf(2/pi)
// f1 = (x+k*pow(x,3) )
// k = 0.044715




std::vector<float> GeluOMP(const std::vector<float>& input) {
    size_t sz = input.size();

    // answer(sz,0.0f) - инициализация нулями - это долго и нет необходимости, поэтому:
    std::vector<float> answer(sz);

    // c1 = sqrtf(2.0f / M_PI) 
    // можно вычислять в цикле (будет считаться на каждой итерации  - долго)
    // можно вынести за цикл и считать 1 раз:
    const float c1 = 0.79788456f; 
    const float k = 0.044715f;

    
    // for (int i = 0; i < sz; i++).
    // можно распараллелить
    // #pragma omp parallel for schedule(static).
    // schedule - распределение итераций цикла между потоками
    // schedule(static) - статическое (до начала выполнения цикла)
#pragma omp parallel for schedule(static)
    for (int i = 0; i <= (int)sz - 4; i += 4) {

        // улучшение: развертка цикла на 4 (Hint 2: loop unrolling):

        for (int j = 0; j < 4; ++j) {
            float x = input[i + j];

            // pow(x,3) - предназначена для любых степеней (в том числе вещественных)
            // x*x*x - это всего 2 умножения (что быстрее, чем pow(x,3))
            float x3 = x * x * x;

            float tanh_arg = c1 * (x + k * x3);

            answer[i + j] = 0.5f * x * (1.0f + fast_tanh(tanh_arg));
        }
    }

    // мы развернули цикл на 4
    // могли остаться недоработанные элементы
    // дорабатываем хвост
    for (size_t i = (sz / 4) * 4; i < sz; ++i) {
        float x = input[i];
        float x3 = x * x * x;
        float tanh_arg = c1 * (x + k * x3);
        answer[i] = 0.5f * x * (1.0f + fast_tanh(tanh_arg));
    }

    return answer;
}


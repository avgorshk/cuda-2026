#include "naive_gemm_omp.h"
#include "omp.h"
#include <vector>

// GEMM - general matrix multiplication

/*
Ключи в Visual Studio:
/openmp (/openmp:llvm) - многопоточность
/fp:fast вместо /fp:precise - быстрые floating point (expf становится быстрее)
/arch:AVX2 - векторные инструкции (в регистре одновременно 8 float)
AVX2 регстры: 256 бит; float: 32 бит; 256/32 = 8
/GL - глобальная оптимизация
/O2  - агрессивная оптимизация
/openmp:experimental - для simd (векторизации)
*/

std::vector<float> NaiveGemmOMP(const std::vector<float>& a,
    const std::vector<float>& b,
    int n) {

    std::vector<float> answer(n * n, 0.0f);

    // блочное множение 
    const int bl_size = 64;
    const int bl_count = n / bl_size;

    // распараллеливаем внешние блоки
#pragma omp parallel for
    for (int bl_j = 0; bl_j < bl_count; bl_j++) {
        for (int bl_k = 0; bl_k < bl_count; bl_k++) {
            for (int bl_i = 0; bl_i < bl_count; bl_i++) {

                // j - определяет строку в a
                for (int j = bl_j * bl_size; j < (bl_j + 1) * bl_size; j++) {
                    // k - определяет элемент, который умножаем
                    for (int k = bl_k * bl_size; k < (bl_k + 1) * bl_size; k++) {

                        // a[j * n + k] - не зависит от i
                        // можно посчитать 1 раз (вынести из цикла по i)
                        float a_jk = a[j * n + k];

                        // i - определяет столбец в b
                        

                        // для разворачивания цикла по i
                        int row_idx = j * n;
                        int b_idx = k * n;

                        // веторизация цикла
                        #pragma omp simd
                        for (int i = bl_i * bl_size; i < (bl_i + 1) * bl_size; i += 4) {
                            // разворачивание цикла на 4
                            answer[i + row_idx] += a_jk * b[i + b_idx];
                            answer[i + 1 + row_idx] += a_jk * b[i + 1 + b_idx];
                            answer[i + 2 + row_idx] += a_jk * b[i + 2 + b_idx];
                            answer[i + 3 + row_idx] += a_jk * b[i + 3 + b_idx];
                        }
                    }
                }
            }
        }
    }
    
    return answer;
}

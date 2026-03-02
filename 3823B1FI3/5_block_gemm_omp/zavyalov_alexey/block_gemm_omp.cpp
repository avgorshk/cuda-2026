#include "block_gemm_omp.h"
#define BLOCK_WIDTH 1024
#define BLOCK_HEIGHT 2

std::vector<float> BlockGemmOMP(const std::vector<float>& a,
    const std::vector<float>& b,
    int n) {

    std::vector<float> c(n * n);

#pragma omp parallel for
    for (int block_row = 0; block_row < n / BLOCK_HEIGHT; block_row++) {
        for (int block_column = 0; block_column < n / BLOCK_WIDTH; ++block_column) {
            int next_block_row = ((block_row + 1) * BLOCK_HEIGHT);
            
            for (int a_ind = block_row * BLOCK_HEIGHT; a_ind < next_block_row; a_ind++) {
                for (int k = 0; k < n; k++) {
                    int a_ind_mult_n = a_ind * n;
                    int k_mult_n = k * n;
                    int next_block_column = ((block_column + 1) * BLOCK_WIDTH);
                    float a_elem = a[a_ind_mult_n + k];

                    #pragma omp simd
                    for (int b_ind = block_column * BLOCK_WIDTH; b_ind < next_block_column; b_ind ++) {
                        c[a_ind_mult_n + b_ind] += a_elem * b[k_mult_n + b_ind];
                    }
                }
            }
        }

    }

    return c;
}
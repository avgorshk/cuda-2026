#pragma GCC optimize("O3")
#pragma GCC optimize("fast-math")

#include "naive_gemm_omp.h"

#include <omp.h>
#include <vector>

inline void swap(float& a, float& b) {
	float c = b;
	b = a;
	a = c;
}

void GetTransposedMatrix(int n, const float* src, float* res) {
#pragma omp parallel for
	for (int i = 0; i < n * n; ++i) {
		res[i] = src[i];
	}
	
#pragma omp parallel for
	for (int i = 0; i < n; ++i) {
		for (int j = i + 1; j < n; ++j) {
			swap(res[i * n + j], res[j * n + i]);
		}
	}
}

#pragma GCC target("avx2")
void MatrixProduct(const float* A, const float* BT, float* res, int n) {
#pragma omp parallel for
	for (int i = 0; i < n; ++i) {
		for (int j = 0; j < n; ++j) {
			float temp = 0.0f;
			#pragma omp simd
			for (int k = 0; k < n; ++k) {
				temp += A[i * n + k] * BT[j * n + k];
			}
			res[i * n + j] = temp;
		}
	}
}

std::vector<float> NaiveGemmOMP(const std::vector<float>& a,
                                const std::vector<float>& b,
                                int n) {

    const float* A = a.data();
	const float* B = b.data();

	std::vector<float> bt(n * n, 0.0f);
	std::vector<float> result(n * n, 0.0f);

	GetTransposedMatrix(n, B, bt.data());
	
	const float* BT = bt.data();
	float* res = result.data();

    MatrixProduct(A, BT, res, n);

	return result;
}
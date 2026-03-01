#pragma GCC optimize("O2")
#pragma GCC optimize("fast-math")
#pragma GCC optimize("unroll-loops")

#include "gelu_omp.h"

#define PI_coeff 0.7978845608028654f

#pragma GCC target("avx2")

std::vector<float> GeluOMP(const std::vector<float>& input) {
	int n = input.size();
	std::vector<float> result(n);
#pragma omp parallel
{
		int thread_count = omp_get_num_threads();
		int tid = omp_get_thread_num();
		int chunk = (n + thread_count - 1) / thread_count;
		int start = tid * chunk;
		int end = std::min(start + chunk, n);
#pragma omp simd 
	for (int i = start; i < end; i++) {
		float x = input[i];
		float x2 = x * x;
		float x3 = x2 * x;
		float p = PI_coeff * (x + 0.044715f * x3);
		result[i] = x * (1.0f - 1.0f / (1.0f + expf(2.0f * p)));
   }
}
	return result;
}
// Compiler flags MSVC: /O2 /openmp:experimental /arch:AVX2 /fp:fast
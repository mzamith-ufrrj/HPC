#include <cmath>
#include <immintrin.h>
void add_explicitly_optimization_d(double * __restrict__ C, double * __restrict__ A, double * __restrict__ B, unsigned long elements){
    unsigned long i = 0;
    __m256d vec_A, vec_B, vec_C;
  for ( i = 0; i < elements; i+= 4){
    vec_A = _mm256_load_pd(A + i);
    vec_B = _mm256_load_pd(B + i);
    vec_C = _mm256_add_pd(vec_A, vec_B);
    vec_C = _mm256_mul_pd(vec_C, vec_A);
    vec_C = _mm256_sqrt_pd(vec_C);
    _mm256_store_pd(C + i, vec_C);
  }


}

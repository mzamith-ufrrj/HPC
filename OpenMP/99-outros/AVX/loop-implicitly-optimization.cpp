#include <cmath>
void add_implicitly_optimization_d(double * __restrict__ C, double * __restrict__ A, double * __restrict__ B, unsigned long elements){
    unsigned long i = 0;
    for ( i  = 0; i < elements; i+=4){
        C[i + 0] = A[i + 0] + B[i + 0];
        C[i + 1] = A[i + 1] + B[i + 1];
        C[i + 2] = A[i + 2] + B[i + 2];
        C[i + 3] = A[i + 3] + B[i + 3];

        C[i + 0] = C[i + 0] * A[i + 0];
        C[i + 1] = C[i + 1] * A[i + 1];
        C[i + 2] = C[i + 2] * A[i + 2];
        C[i + 3] = C[i + 3] * A[i + 3];

        C[i + 0] = sqrt(C[i + 0]);
        C[i + 1] = sqrt(C[i + 1]);
        C[i + 2] = sqrt(C[i + 2]);
        C[i + 3] = sqrt(C[i + 3]);
    }


}

#include <iostream>
#include <cassert>
#include <cstdlib>
#include <chrono>
#include <iomanip>
/* Funções declaradas em arquivos externos */
void add_no_optimization_d(double * __restrict__ C, double * __restrict__ A, double * __restrict__ B, unsigned long elements);
void add_implicitly_optimization_d(double * __restrict__ C, double * __restrict__ A, double * __restrict__ B, unsigned long elements);
void add_explicitly_optimization_d(double * __restrict__ C, double * __restrict__ A, double * __restrict__ B, unsigned long elements);

using namespace std;
int main(int ac, char **av){
    int           align  = 64;
    unsigned long memory = 24 * 1024 * 1024,  //4 MB
                  elements = memory / sizeof(double),
                  elapsed_no_optimization = 0,
                  elapsed_implicitly_optimization = 0,
                  elapsed_explicitly_optimization = 0;
    double   *A = NULL,
             *B = NULL,
             *C = NULL;
    auto start_all = std::chrono::high_resolution_clock::now();
    cout << endl << "\tExercicio de vetorização" << endl;
    assert(posix_memalign(reinterpret_cast <void**>(&A), align, memory) == 0);
    assert(posix_memalign(reinterpret_cast <void**>(&B), align, memory) == 0);
    assert(posix_memalign(reinterpret_cast <void**>(&C), align, memory) == 0);

    for (unsigned long  i = 0; i < elements; i++){
        B[i] = static_cast<double>((i+256) % 127) * 1E-2;
        A[i] = B[i] * 1E2;
        C[i] = 0.0f;
    }

    auto start = std::chrono::high_resolution_clock::now();
        add_no_optimization_d(C, A, B, elements);
    auto stop = std::chrono::high_resolution_clock::now();
    auto elapsed = chrono::duration_cast<chrono::microseconds>(stop - start);
    elapsed_no_optimization = elapsed.count();

    start = std::chrono::high_resolution_clock::now();
        add_implicitly_optimization_d(C, A, B, elements/4);
    stop = std::chrono::high_resolution_clock::now();
    elapsed = chrono::duration_cast<chrono::microseconds>(stop - start);
    elapsed_implicitly_optimization = elapsed.count();

    start = std::chrono::high_resolution_clock::now();
        add_explicitly_optimization_d(C, A, B, elements/4);
    stop = std::chrono::high_resolution_clock::now();
    elapsed = chrono::duration_cast<chrono::microseconds>(stop - start);
    elapsed_explicitly_optimization = elapsed.count();

    free(A);
    free(B);
    free(C);
    auto stop_all = std::chrono::high_resolution_clock::now();

    //chrono::hours, chrono::minutes, chrono::seconds, chrono::milliseconds, or chrono::microseconds.
    auto elapsed_all = chrono::duration_cast<chrono::microseconds>(stop_all - start_all);
    cout << "\t                  Tempo de execução do programa: " << setw(10) << elapsed_all.count() << endl;
    cout << "\t        Tempo de execução add_no_optimization_d: " << setw(10) << elapsed_no_optimization << endl;
    cout << "\tTempo de execução add_implicitly_optimization_d: " << setw(10) << elapsed_implicitly_optimization << " speedup: " << (elapsed_no_optimization / elapsed_implicitly_optimization) << endl;
    cout << "\tTempo de execução add_explicitly_optimization_d: " << setw(10) << elapsed_explicitly_optimization << " speedup: " << (elapsed_no_optimization / elapsed_explicitly_optimization) << endl;
    return EXIT_SUCCESS;
}


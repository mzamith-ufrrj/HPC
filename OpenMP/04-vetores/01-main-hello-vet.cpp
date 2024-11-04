#include <iostream>
#include <cassert>
#include <string>
#include <cstring>
#include <random>
#include <omp.h>
#include <unistd.h>
#define ALING 64
using namespace std;

void wVet(unsigned int *vetA, unsigned int size){

    #pragma omp parallel for
    for (unsigned int i = 0; i < size; i++){
        vetA[i] = omp_get_thread_num();
    }
}

void wVet2(unsigned int *vetA, unsigned int size){
    unsigned int a = 0,
                 b = 0;

    #pragma omp parallel private (a, b) shared (size)
    {
        unsigned int subsize = size / omp_get_num_threads(),
                    offset = 0;

        a = omp_get_thread_num() * subsize;
        b = (omp_get_thread_num() + 1) * subsize;


        offset = size %  omp_get_num_threads();
        if (omp_get_thread_num() == 0){
            b += offset;
        }else{
            a += offset;
            b += offset;
        }



        for (unsigned int i = a; i < b; i++){
            vetA[i] = omp_get_thread_num();
        }

    }

}


int main (int ac, char **av){
    unsigned int *vetA  = NULL,
                 size = stoi(av[2]),
                 nthread = stoi(av[1]);

    assert(nthread <= size);
    cout << endl << "Exemplo de manipulação de vetor" << endl;
    posix_memalign(reinterpret_cast <void**>(&vetA), ALING, size * sizeof(unsigned int));

    omp_set_num_threads(nthread);
    wVet(vetA, size);
    cout << "Saída 1" << endl;
    for (unsigned int i = 0; i < size; i++){
        cout << "\t" << i << " -> " << vetA[i] << endl;
    }

    wVet2(vetA, size);
    cout << "Saída 2" << endl;
    for (unsigned int i = 0; i < size; i++){
        cout << "\t" << i << " -> " << vetA[i] << endl;
    }
    free(vetA);
    return EXIT_SUCCESS;
}

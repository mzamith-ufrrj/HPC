#include <stdio.h>
#include <stdlib.h>
#ifdef _OPENMP
#include <omp.h>
#endif
//gcc -fopenmp 01-hello-omp.c -o hello.exec
int main (int av, char **ac){
    printf("\nExemplo usando OpenMP em C\n");

#ifdef _OPENMP
    /*
     *
     * A função:omp_set_num_threads(2); permite definir a quantidade de threads
     * que serão criadas. Se a função não for chamada, o valor padrão de quantidade
     * de threads igual ao número de cores.
     *
     */
    omp_set_num_threads(2);
    #pragma omp parallel
    {
        int tid =  omp_get_thread_num();
        int tsi =  omp_get_num_threads();
        printf(" - Eu sou a thread %d / %d \n", tid, tsi);
    }
#else
    printf("Apenas um hello word com uma linha de execução");
#endif
    exit(EXIT_FAILURE);
    printf("\n\n");
    return EXIT_SUCCESS;
}

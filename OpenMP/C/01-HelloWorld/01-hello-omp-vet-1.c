#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
//gcc -fopenmp 01-hello-omp-vet-1.c -o hello-vet-1.exec
int main (int av, char **ac){
    printf("\nExemplo usando OpenMP em C\n");
    int vet[10];
    #pragma omp parallel for
    for (int i = 0; i < 10; i++){
        vet[i] = omp_get_thread_num();
    }

    printf("Imprimindo resultado: \n");
    for (int i = 0; i < 10; i++){
        printf("Vetor na posição %d => %d \n", i, vet[i]);
    }


    return EXIT_SUCCESS;
}

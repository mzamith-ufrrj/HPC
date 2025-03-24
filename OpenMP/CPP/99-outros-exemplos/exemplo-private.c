#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <unistd.h>
#define ALING 64

int main (int ac, char **av){
    //unsigned int nthread = stoi(av[1]);
    printf("\nHello world!\n");
    printf("\t%d\n", omp_get_num_threads());
    int valor = 42;
    #pragma omp parallel shared(valor)
    {
	
       if (omp_get_thread_num() == 0){ 
          while(1);
	  valor = omp_get_thread_num() + 1 * 10;
       }
	  //#pragma omp barrier
       printf("\t%d/%d = %d \n", omp_get_thread_num(), omp_get_num_threads(), valor);
    }
    return EXIT_SUCCESS; 
}

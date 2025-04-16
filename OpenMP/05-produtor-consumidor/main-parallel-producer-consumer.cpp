#include <cassert>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <string>
#include <cstring>
#include <chrono>
#include <climits>
#include <omp.h>
#include <unistd.h>
#include <random>
using namespace std;
#define ALING 64
#define MAX_ROUND 10000
#define VECTOR_SIZE 42
void producer(unsigned int * __restrict__ vet,
              const unsigned long s,
              const unsigned int round){

        cout << "\tProdutor rodada " <<  round << endl;
        cout.flush();
        for (unsigned long i = 0; i < s; i++){
            assert(vet[i] == 0);
            vet[i] = round;
        }
}

void consumer(unsigned int * __restrict__ vet,
              const unsigned long s,
              const unsigned int round){
        bool flag = false;
        cout << "\tConsumidor rodada " <<  round << endl;
        cout.flush();
        do{
            if (flag) flag = false;
            for (unsigned long i = 0; i < s; i++){
                bool ifcond = (vet[i] > 0);
                flag = (flag | ifcond);
                if (ifcond) vet[i]--;
            }

        }while(flag);

}


int main (int ac, char **av){
    unsigned int *vet = NULL,
                 round = 1;
    unsigned long size = VECTOR_SIZE;
    omp_lock_t    lock_c, lock_p;

    cout << "Exemplo 1 produtor e 1 consumidor" << endl;
    posix_memalign(reinterpret_cast <void**>(&vet), ALING, size * sizeof(unsigned long));

    assert(posix_memalign(reinterpret_cast <void**>(&vet), ALING, size * sizeof(unsigned long)) == 0);
    bzero(vet, size * sizeof(unsigned long));
    omp_set_num_threads(2); //Fixo o número de threads

    omp_init_lock(&lock_c);
    omp_init_lock(&lock_p);


    producer(vet, size, round);
    round++;
    omp_set_lock(&lock_p);

    #pragma omp parallel shared(vet, size, round)
    {
        while(round < MAX_ROUND){
            if (omp_get_thread_num() == 0){
                if (omp_test_lock(&lock_p)){
                        producer(vet, size, round);
                        round++;
                    //omp_unset_lock(&lock_c);
                    omp_unset_lock(&lock_c);
                }//if (omp_test_lock(&consumer)){
            }//if (omp_get_thread_num() == 0){

            if (omp_get_thread_num() == 1){
                if (omp_test_lock(&lock_c)){
                    consumer(vet, size, round);
                    omp_unset_lock(&lock_p);
                }//if (omp_test_lock(&consumer)){
            }//if (omp_get_thread_num() == 1){
        }//while(round < 5){

    }//#pragma omp parallel


    return EXIT_SUCCESS;
}


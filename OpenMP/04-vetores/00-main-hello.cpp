#include <iostream>
#include <iomanip>
#include <omp.h>
#include <unistd.h>
#define ALING 64
using namespace std;
//g++ 00-main-hello.cpp -fopenmp -o hello-mp.exec

// /./hello-mp.exec 10 <- número de threads

int main (int ac, char **av){
    unsigned int nthread = stoi(av[1]);
    cout << endl << "Hello world in OpenMP" << endl;
    cout << "\t Threads: " << nthread << endl;
    omp_set_num_threads(nthread);
    //cout << "\t" <<  omp_get_num_threads() << endl;
#pragma omp parallel
{

    //usleep(1000 * omp_get_thread_num());
    cout << "\t Hello, Eu sou a thread: " << omp_get_thread_num() << " de um total de " <<  omp_get_num_threads() <<  endl;
}//#pragma omp parallel
    return EXIT_SUCCESS;
}

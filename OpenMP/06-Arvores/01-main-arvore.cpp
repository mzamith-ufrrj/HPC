#include <iostream>
#include <iomanip>
#include <cassert>
#include <sys/types.h>
#include <sys/stat.h>
#include <fstream>
#include <string>
#include <cstring>
#include <random>
#include <cstdio>
#include <omp.h>
#include <chrono>
using namespace std;
void foo(const unsigned int level){
    omp_set_num_threads(level+1);
    #pragma omp parallel shared(level)
    {
        #pragma omp single
        {
            cout << omp_get_thread_num() << "/"<< omp_get_num_threads() << endl;
            cout << "Level: " << level << endl;
            cout.flush();
        }

    }

    if (level == 12) return;
    foo(level + 1);
}
int main (int ac, char **av){

    cout << endl << "Exemplo de arvore" << endl;

    cout << "Threads: " << omp_get_num_threads() << endl;
    omp_set_num_threads(1);
    #pragma omp parallel
    {
        foo(0);

    }

    return EXIT_SUCCESS;
}

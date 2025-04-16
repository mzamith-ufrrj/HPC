#include <iostream>
#include <iomanip>
#include <cassert>
#include <sys/types.h>
#include <sys/stat.h>
#include <fstream>
#include <string>
#include <cstring>
#include <chrono>
#include <random>
#include <omp.h>
#define ALING 64
using namespace std;
/*
void saveLog(string logFile,
             int w,
             int h,
             int s,
             unsigned long total,
             unsigned long comput){

    fstream logout;
    FILE *ptr = NULL;
    struct stat buffer;
    int ret = stat(logFile.c_str(), &buffer);
    if (ret == -1){
        logout.open(logFile, fstream::trunc|fstream::out);
        //ptr = fopen(logFile.c_str(), "w+");
        assert(logout.is_open());
        logout << "width;height;steps;overhead_time_microseconds;comput_time_microseconds;total_time_microseconds" << endl;


    }else{
        logout.open(logFile, fstream::app | fstream::out);

    }
    unsigned long overhead = total - comput;

    logout << w << ";" << h << ";" << s << ";" << overhead << ";" << comput << ";" << total <<  endl;

    logout.close();
    cout << "Log file[" << logFile << "] saved" << endl;

}
*/
void addVet(double *vetC, double *vetA, double *vetB, unsigned int size){

    #pragma omp parallel for
    for (unsigned int i = 0; i < size; i++){
        //vetC[i] = static_cast<double>(omp_get_thread_num());
        vetC[i] = vetA[i] + vetB[i];
    }
}


int main (int ac, char **av){
    double *vetA  = NULL,
           *vetB  = NULL,
           *vetC  = NULL;
    unsigned int size = 10,
                 nthread = stoi(av[1]);
    std::mt19937_64  generator (time(nullptr)); //64 bits
    uniform_real_distribution<double> unif(0, 1); //uniform distribuiti


    //unsigned long  processorElapsedTime = 0;
    //auto start = std::chrono::high_resolution_clock::now();


    cout << endl << "Exemplo de manipulação de vetor" << endl;
    posix_memalign(reinterpret_cast <void**>(&vetA), ALING, size * sizeof(double));
    posix_memalign(reinterpret_cast <void**>(&vetB), ALING, size * sizeof(double));
    posix_memalign(reinterpret_cast <void**>(&vetC), ALING, size * sizeof(double));
    for (unsigned int i = 0; i < size; i++){
        vetA[i] = unif(generator) * 100.0f;
        vetB[i] = unif(generator) / 100.0f;
        vetC[i] = 0.0f;
    }
    omp_set_num_threads(nthread);
    addVet(vetC, vetB, vetA, size);

    for (unsigned int i = 0; i < size; i++){

        cout << i << " " << setw(10) << fixed << setprecision(6) << vetC[i];
        cout << " " << setw(10) << fixed << setprecision(6) << vetA[i];
        cout << " " << setw(10) << fixed << setprecision(6) << vetB[i] << endl;

    }


    free(vetA);
    free(vetB);
    free(vetC);

    //auto stop = std::chrono::high_resolution_clock::now();

    //chrono::hours, chrono::minutes, chrono::seconds, chrono::milliseconds, or chrono::microseconds.
    //auto elapsed = chrono::duration_cast<chrono::microseconds>(stop - start);
    //cout << elapsed.count() << endl;
    //saveLog(filename, mLattice.x, mLattice.y, mLattice.t, elapsed.count(), processorElapsedTime);
    return EXIT_SUCCESS;
}

#include <iostream>
#include <iomanip>
#include <cassert>
#include <sys/types.h>
#include <sys/stat.h>
#include <fstream>
#include <string>
#include <cstring>
#include <random>
#include <gol-cpu.hpp>
#include <cstdio>
#include <chrono>
using namespace std;

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

int main (int ac, char **av){
    double *elapsedtime = NULL;
    stLattice mLattice;
    //bool  flagSave = stoi(av[5]);
    //Inicializa variável
    mLattice.x   = stoi(av[1]);
    mLattice.y   = stoi(av[2]);
    mLattice.t   = stoi(av[3]);
    unsigned int nThreads = stoi(av[4]);
    //string filename = av[4];
    //string type     = av[5];

    auto start_time = std::chrono::high_resolution_clock::now();


    cout << endl << "GAME OF LIFE" << endl;
    cout << "\tDomain( x = " << mLattice.x << ",y = " <<  mLattice.y << ",t = " <<  mLattice.t << ")" << endl;
    cout << "\tThreads " << nThreads << endl;

    unsigned long mem =  mLattice.x *  mLattice.y * sizeof(unsigned char);
    //Allocing align
    posix_memalign(reinterpret_cast <void**>(&mLattice.buff0), ALING, mem);
    posix_memalign(reinterpret_cast <void**>(&mLattice.buff1), ALING, mem);

    InitRandness(&mLattice, 0.25);

    CPU_GOL(&mLattice, &elapsedtime, nThreads);
    //CPU_GOL(&mLattice, type);
    for (unsigned int i = 0; i < nThreads; i++){
        cout << "\tTempo gasto pela thread " << i << " foi de " << elapsedtime[i] << " em segundos" << endl;
    }

    //print2Console(&mLattice);



    free(mLattice.buff0);
    free(mLattice.buff1);
    free(elapsedtime);
    auto stop_time = std::chrono::high_resolution_clock::now();

    //chrono::hours, chrono::minutes, chrono::seconds, chrono::milliseconds, or chrono::microseconds.
    std::chrono::duration<double> total_time = stop_time - start_time;

    cout << "Tempo de execução: " << total_time.count() << " em segundos " << endl;
    //cout << elapsed.count() << endl;
    //saveLog(filename, mLattice.x, mLattice.y, mLattice.t, elapsed.count(), processorElapsedTime);
    return EXIT_SUCCESS;
}

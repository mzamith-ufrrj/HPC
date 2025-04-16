//sh -c "echo 2 > /proc/sys/kernel/perf_event_paranoid"
#include <papi.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <cmath>
#include <immintrin.h>
#include <string>
#include <ctime>
#include <fstream>
#include <cassert>
using namespace std;



void init_vector(unsigned long * __restrict__ V,
                 const unsigned long s);

void compute_vector(unsigned long * __restrict__ V,
                    const unsigned long s,
                    const unsigned long alpha);

void saveLog(string logFile, long long *values, unsigned long mem, long long clocks);
//./memory.exec 1000 1234 64 1
int main (int ac, char **av){

    unsigned long size  = stoi(av[1]),
                  alpha = stoi(av[2]),
                  *vet  = NULL;

    long long *values_events = NULL;
    bool type_allocation = stoi(av[4]);
    string logFile  = av[5];

    int align = stoi(av[3]),
        event_set = PAPI_NULL,
        *events = NULL;

    size *= 1024 / sizeof(unsigned long);


    cout << endl;
    assert(PAPI_library_init(PAPI_VER_CURRENT) == PAPI_VER_CURRENT);
    const PAPI_hw_info_t *hwinfo = PAPI_get_hardware_info();
    cout << "=================================================================" << endl;
    cout << "System Info:" << endl;
    cout << "\t" << hwinfo->vendor_string << endl;
    cout << "\t" << hwinfo->model_string << endl;
    cout << endl;
    cout << "=================================================================" << endl;

    cout << "Teste de memória - alocação em megabytes" << endl;
    cout << "\t   Memoria alocada:[" <<  size * sizeof(unsigned long) << "] bytes" << endl;

    if (type_allocation)
        assert(posix_memalign(reinterpret_cast <void**>(&vet), align, size * sizeof(unsigned long)) == 0);
    else
        vet = reinterpret_cast <unsigned long *> (malloc(size * sizeof(unsigned long)));
    cout << "\t          Endereço:" << hex << vet << endl;
    cout << "\t Alinhamento cache:" << hex << align << endl;
    cout << "\t    Arquivo de log:" << logFile << endl;

    if (type_allocation)
        cout << "\t          Alocação: posix_memalign" << endl;
    else
        cout << "\t          Alocação: malloc" << endl;

    int nEvents = 5;
    assert(posix_memalign(reinterpret_cast <void**>(&values_events), align, nEvents * sizeof(long long)) == 0);
    assert(posix_memalign(reinterpret_cast <void**>(&events), align, nEvents * sizeof(int)) == 0);

    events[0] = PAPI_L1_TCM;
    events[1] = PAPI_L2_TCM;
    events[2] = PAPI_L3_TCM;
    events[3] = PAPI_L2_TCA;
    events[4] = PAPI_L3_TCA;

    assert(PAPI_create_eventset(&event_set) == PAPI_OK);
    assert(PAPI_add_events(event_set, events, nEvents) == PAPI_OK);
    assert(PAPI_start(event_set) == PAPI_OK);

    long long start_cyc = PAPI_get_real_cyc();
            init_vector(vet, size);
            compute_vector(vet, size, alpha);
    long long end_cyc = PAPI_get_real_cyc();
    assert(PAPI_stop(event_set, values_events) == PAPI_OK);
    assert(PAPI_cleanup_eventset(event_set) == PAPI_OK);
    assert(PAPI_destroy_eventset(&event_set) == PAPI_OK);


    saveLog(logFile, values_events, size * sizeof(unsigned long), end_cyc - start_cyc);

    free(vet);
    free(values_events);
    free(events);
    cout << endl << endl;
    return EXIT_SUCCESS;
}



void init_vector(unsigned long * __restrict__ V,
                 const unsigned long s){

    for (unsigned long i = 0; i < s; i++){
      V[i] = ((1024 * 1024) + i) / (sizeof(unsigned long) + i + 1);

    }//   for (unsigned long i = 0; i < s; i++){

}//void init_vector(unsigned long * __restrict__ V, const unsigned long s){

void compute_vector(unsigned long * __restrict__ V,
                    const unsigned long s,
                    const unsigned long alpha){
    unsigned long offset = 8;
    while (offset < 256){
        for(unsigned long i = 0; i < s-offset; i+=offset){
            for (int t = 0; t < alpha; t++)
                for(unsigned long j = i; j < i+offset; j++){
                        V[j] = V[i] + j ^ offset | alpha;
                }
        }//for(unsigned long i = 0; i < s-8; i+=8){
        offset <<= 1;
    }//while (offset < 256){
}

void saveLog(string logFile, long long *values, unsigned long mem, long long clocks){

    fstream  logout;
    FILE     *ptr = NULL;
    struct   stat buffer;
    int ret = stat(logFile.c_str(), &buffer);
    if (ret == -1){
        logout.open(logFile, fstream::trunc|fstream::out);
        //ptr = fopen(logFile.c_str(), "w+");
        assert(logout.is_open());
        logout << "CLOCK;MEMORY_USED;PAPI_L1_TCM;PAPI_L2_TCM;PAPI_L3_TCM;PAPI_L2_TCA;PAPI_L3_TCA" << endl;
    }else{
        logout.open(logFile, fstream::app | fstream::out);
    }


    logout << clocks << ";" << mem << ";" << values[0] << ";" << values[1] << ";" << values[2] << ";" << values[3] << ";" << values[4] <<  endl;

    logout.close();
    cout << "Log file[" << logFile << "] saved" << endl;

}

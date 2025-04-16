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


void init_vector(int * __restrict__ V,
                 const int s){

    for (int i = 0; i < s; i++){
      V[i] = 1024 / sizeof(int);
    }

}

void compute_vector(int * __restrict__ V,
                    const int s,
                    const int alpha){

   for(int i = 0; i < alpha; i++){
     int j = 0;
     while (j < s){
       j = j + V[j];
     }
   }

}


int main (int ac, char **av){

    int s     = stoi(av[1]);
    int alpha = stoi(av[2]);
    long long output[6];

    int  *V = NULL;

    cout << endl;
    cout << "Benchmarking" << endl;
    cout << "\t Memoria alocada: [" <<  (s * sizeof(int)) << "] Bytes" << endl;
    output[0] =  (s * sizeof(int));
//-------------------------------------------------------------------------------------------------
    posix_memalign(reinterpret_cast <void**>(&V), 64, s * sizeof(int));



  init_vector(V, s);
  compute_vector(V, s, alpha);
  free(V);
  cout << endl << endl;

  return EXIT_SUCCESS;
}

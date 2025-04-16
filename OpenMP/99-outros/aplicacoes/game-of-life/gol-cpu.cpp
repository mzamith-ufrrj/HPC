#include <iostream>
#include <cassert>
#include <cstring>
#include <random>
#include <omp.h>
#include <gol-cpu.hpp>

using namespace std;
/*
 * Show the last CA state to console.
 */
void print2Console(stLattice *mLattice)
{
    cout << endl;
    cout << "Last state: " << endl;
    for (int j = 0; j < mLattice->y; j++){
        for (int i = 0; i < mLattice->x; i++){
            int k = j * mLattice->x +  i;
            int v = static_cast<int> (mLattice->buff0[k]);
            cout << v;

        }//end-for (int j = 0; j < mLattice->height; j++){
        cout << endl;
    }//end-void InitRandness(tpLattice *mLattice, float p){


}

/*
 * Game of Life initial condition - random approach
 */
void InitRandness(stLattice *mLattice, double p){
    std::mt19937_64  generator (42); //64 bits
    uniform_real_distribution<double> unif(0, 1); //uniform distribuition
    memset(mLattice->buff0, 0x00,  mLattice->x * mLattice->y * sizeof(unsigned char));
    memset(mLattice->buff1, 0x00,  mLattice->x * mLattice->y * sizeof(unsigned char));

  for (int j = 0; j < mLattice->y; j++){
      for (int i = 0; i < mLattice->x; i++){
          int k = j * mLattice->x +  i;
          if (unif(generator) <= p)
                mLattice->buff0[k] = 1;
      }//end-  for (int i = 0; i < mLattice->width; i++){
  }//end-for (int j = 0; j < mLattice->height; j++){
}//end-void InitRandness(tpLattice *mLattice, float p){


//double acc  = 0.0f;
void CPU_GOL(stLattice *mLattice, double **ptr, unsigned int threads){
    double *elapsedtime = NULL;
    assert(posix_memalign(reinterpret_cast <void**>(&elapsedtime), ALING, threads * sizeof(double)) == 0);

    omp_set_num_threads(threads);
    #pragma omp parallel shared(mLattice)
    {
        double et = omp_get_wtime();
        unsigned char nw = 0, n = 0,  ne = 0,
                    w  = 0, c = 0,  e  = 0,
                    sw = 0, s = 0, se  = 0,
                    sum;
            for (int t = 0; t < mLattice->t; t++){
                #pragma omp for nowait schedule(dynamic)
                for (int j = 0; j < mLattice->y; j++){
                    int alphaJ = static_cast<int> (j-1 < 0);
                    int minJ   = (alphaJ * (mLattice->y-1)) + ( (1 - alphaJ) * (j-1) );
                    int maxJ   = (j+1) % mLattice->y;

                        for (int i = 0; i < mLattice->x; i++){
                        int alphaI = static_cast<int> (i-1 < 0);
                        int minI   = (alphaI * (mLattice->x-1)) + ( (1 - alphaI) * (i-1) );
                        int maxI   = (i+1) % mLattice->x;

                                c  = mLattice->buff0[j * mLattice->x  +  i];
                                nw = mLattice->buff0[minJ * mLattice->x  +  minI];
                                n  = mLattice->buff0[minJ * mLattice->x  +  i];
                                ne = mLattice->buff0[minJ * mLattice->x  +  maxI];
                                w  = mLattice->buff0[j    * mLattice->x  +  minI];
                                e  = mLattice->buff0[j    * mLattice->x  +  maxI];
                                sw = mLattice->buff0[maxJ * mLattice->x  +  minI];
                                s  = mLattice->buff0[maxJ * mLattice->x  +  i];
                                se = mLattice->buff0[maxJ * mLattice->x  +  maxI];

                        sum = nw + n + ne + w + e + sw + s + se;
                        //The GOL rule
                        if ((sum == 3) && (c == 0))
                            mLattice->buff1[j  * mLattice->x  +  i] = 1;
                        else if ((sum >= 2) && (sum <= 3) && (c == 1))
                            mLattice->buff1[j  * mLattice->x  +  i] = 1;
                        else
                            mLattice->buff1[j  * mLattice->x  +  i] = 0;

                    }//for (int i = 0; i < mLattice->x; i+){
                }//for (int j = 0; j < mLattice->y; j++){
                #pragma omp barrier
                #pragma omp single
                {
                    unsigned char *swp = mLattice->buff0;
                    mLattice->buff0 = mLattice->buff1;
                    mLattice->buff1 = swp;
                }
                #pragma omp barrier
            }


            elapsedtime[omp_get_thread_num()] = omp_get_wtime() - et;
    }//#pragma omp parallel shared(mLattice)
    if (ptr == NULL)
        free(elapsedtime);
    else
        *ptr = elapsedtime;
}


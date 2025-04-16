#ifndef GOL_CPU_HPP
#define GOL_CPU_HPP

#include <gol-cpu.hpp>

struct stLattice{
    unsigned char *buff0;
    unsigned char *buff1;
    int x;
    int y;
    int t;
};

void print2Console(stLattice *mLattice);
void InitRandness(stLattice *mLattice, double p);
void CPU_GOL(stLattice *, double **, unsigned int);
#endif

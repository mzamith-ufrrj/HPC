#include <cassert>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <string>
#include <cstring>
#include <omp.h>
#include <chrono>
#include <climits>
using namespace std;
double normal_fdp(double var, double ave, double x){
    double a = (x - ave) / var;
    double b = pow(a, 2.0f);
    double e = exp(-0.5 * b);
    return e / (var * sqrt(2.0 * M_PI));
}
int main (int ac, char**av){
//184467440737095
//1000000000000
    auto start_time = std::chrono::high_resolution_clock::now();
    unsigned long long steps = 1000000000; //(ULLONG_MAX / 100000);
    double h = 10.0f / static_cast<double>(steps),
           pdf = 0.0f,
           x = 0.0f,
           X = 0.0f;

    unsigned int  nThreads = stoi(av[1]);
    cout << "Integração numérica em OpenMP " << endl;
    cout << " Threads:"  << nThreads << endl;

    omp_set_num_threads(nThreads);
    #pragma omp parallel private(x) firstprivate(X) shared(steps, h)
    {
        X = -5.0f;

        #pragma omp parallel for  reduction(+:pdf)
        for (unsigned long long s = 0; s < steps; s++){
            double C = 1.0f;
            if ((s > 0) && (s < steps - 1)){
                C = 2.0f;
            }
            x = X + static_cast<double> (s) * h;
            pdf += (C * normal_fdp(1.0f, 0.0f, x));


        }

        #pragma omp first
        {
            pdf *= h/2.0f;
        }

    }


    cout << "Resultado da integral: " << pdf << endl;
    auto stop_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> total_time = stop_time - start_time;
    cout << "\tTempo total de execução: " << total_time.count() << endl;
    return EXIT_SUCCESS;
}


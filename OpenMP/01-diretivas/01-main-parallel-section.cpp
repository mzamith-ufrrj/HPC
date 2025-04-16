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
#include <unistd.h>
using namespace std;
#define ALING 64
double getPI(double epsilon){
    double pi0  = 0.0f,
           pi1  = 0.0f,
           a    = 0.0f;
    bool  flag  = true;

    do{
        pi1 += (4.0f / (2.0 * a + 1.0f)) * pow(-1.0f, a);
        flag = fabs(pi1 - pi0) > epsilon;
        pi0 = pi1;
        a++;
    }while (flag);
    return pi1;

}

double getGoldenRation(double epsilon){
    unsigned long a  = 1.0f,
                  b  = 1.0f,
                  c  = 0.0f;

    double phi0 = 0.0f,
           phi1 = 0.0f;
    bool  flag  = true;

    do{
        c = b + a;
        a = b;
        b = c;
        phi1 = static_cast<double>(b) / static_cast<double>(a);
        flag = fabs(phi1 - phi0) > epsilon;
        phi0 = phi1;
    }while (flag);
    return phi1;

}

int main (int ac, char **av){
    double epsilon = 1E-8,
           *elapsedtime = NULL,
           pi, phi,
           et;
    unsigned int nThreads = 2;
    cout << "\n\tExemplo: " << __FILE__ << " de uso de diretiva do OpenMP" << endl;
    cout << "\t Erro absoluto menor que: " << scientific << epsilon << endl;

    omp_set_num_threads(nThreads);

    posix_memalign(reinterpret_cast <void**>(&elapsedtime), ALING, nThreads * sizeof(double));
#pragma omp parallel shared(elapsedtime, pi, phi, epsilon) private(et)
{
    et  = omp_get_wtime();
    #pragma omp sections nowait
    {
        #pragma omp section
        {
            pi = getPI(epsilon);
            cout << "Pi encontrado " << endl;
            cout.flush();
        }

        #pragma omp section
        {
            phi = getGoldenRation(epsilon);
            cout << "Número de ouro encontrado " << endl;
            cout.flush();
        }

    }

    elapsedtime[omp_get_thread_num()] = omp_get_wtime() - et;
}
    cout << setprecision(20) << fixed;
    for (unsigned int i = 0; i < nThreads; i++){
        cout << "\tTempo gasto pela thread " << i << " foi de " << elapsedtime[i] << " em segundos" << endl;
    }
    cout << endl << endl;
    cout << "    Valor de pi: " << setprecision(20) << fixed <<  pi << endl;
    cout << "Relação de ouro: "  <<  phi << endl;
    return EXIT_SUCCESS;
}

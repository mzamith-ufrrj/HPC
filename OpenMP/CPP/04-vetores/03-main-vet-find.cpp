#include <iostream>
#include <cassert>
#include <fstream>
#include <string>
#include <cstring>
#include <chrono>
#include <climits>
#include <omp.h>
#include <unistd.h>
using namespace std;
#define ALING 64
/**
 *
 * Função para realizar pesquisa sequencial em um vetor. Valores únicos e ordenados - Versão sequencial.
 *                                                                                    =================
 * @param size é o tamanho do vetor
 * @param value é o valor a ser pesquisado
 * @param *vet ponteiro para o vetor
 * @param *out ponteiro para uma variável para armazenar a posição do vetor com o valor encontrado.
 * @param *elapsedtime vetor de double para armazenar o tempo de processamento de cada thread
 * @return verdadeiro ou falso caso encontre ou não o valor
 *
 * */
bool find_linear(unsigned long size, unsigned long value, unsigned long *vet, unsigned long *out, double *elapsedtime);

/**
 *
 * Função para realizar pesquisa sequencial em um vetor. Valores únicos e ordenados - Versão paralela.
 *                                                                                    =================
 * @param size é o tamanho do vetor
 * @param value é o valor a ser pesquisado
 * @param *vet ponteiro para o vetor
 * @param *out ponteiro para uma variável para armazenar a posição do vetor com o valor encontrado.
 * @param *elapsedtime vetor de double para armazenar o tempo de processamento de cada thread
 * @return verdadeiro ou falso caso encontre ou não o valor
 *
 * */
bool find_linear_omp(unsigned long size, unsigned long value, unsigned long *vet, unsigned long *out, double *elapsedtime);

//find_binary(size, value, vet, &out, elapsedtime);


bool find_binary(unsigned long high,
                 unsigned long value,
                 unsigned long *vet,
                 unsigned long *out,
                 double *elapsedtime){
    unsigned long l = 0, h = high;
    bool found = false;
    while ((l <= h) && (!found)){
        unsigned long pm = (h + l) / 2;
        if (vet[pm] == value){
            *out = pm;
            found = true;
        }else if(value < vet[pm]){
            if (pm > 0)
                h = pm - 1;
            else{
                h = 0;
                l = 1;
            }

        }else
            l = pm + 1;
        //cout << l << "," << h << "," << pm << endl;
    }//while (l < h) && (!found){


    return found;
}
// ./find.exec 31000 7864320099

int main (int ac, char **av){
    unsigned long *vet = NULL,
                  size = strtoul(av[1], NULL, 0),
                  v    = ULLONG_MAX,
                  out   = 0,
                  value = strtoul(av[2], NULL, 0);

    unsigned int nThreads = stoi(av[3]);

    cout << endl << "\tPesquisa no vetor" << endl;
    size *= (1024 * 1024) / sizeof(unsigned long);
    cout << "               Tamanho:" << size * sizeof(unsigned long) << " em bytes " << endl;

    posix_memalign(reinterpret_cast <void**>(&vet), ALING, size * sizeof(unsigned long));

    //Iniciando o vetor apenas com valores ímpares
    //Deslocado de 100 para permitir pesquisas abaixo de um valor, considerando
    //   que o vetor é de unsigned long.
    #pragma omp parallel for
    for (unsigned long i = 0; i < size; i++){
        vet[i] = 100 +  (2 * i + 1);
    }

    cout << "       Valores ímpares: " << vet[0] << ", " << vet[size-1] << endl;
    cout << "Valor a ser pesquisado: " << value << endl;



    double *elapsedtime = NULL;
    posix_memalign(reinterpret_cast <void**>(&elapsedtime), ALING, nThreads * sizeof(double));

    omp_set_num_threads(nThreads);
    //bool ret = find_linear(size, value, vet, &out, elapsedtime);
    bool ret = find_binary(size, value, vet, &out, elapsedtime);

    if (ret)
        cout << "Valor encontrado na posição: " << out << " valor -> " << vet[out] << endl;
    else
        cout << "Valor não encontrado!" << endl;

    cout << "Tempo de processamento por threads: " << endl;
    for (unsigned int i = 0; i < nThreads; i++){
        cout << "\t Thread índice: " << i << " gastou: " << elapsedtime[i] << " (s) " << endl;
    }


    free(vet);
    free(elapsedtime);
    return EXIT_SUCCESS;
}

bool find_linear(unsigned long size, unsigned long value, unsigned long *vet, unsigned long *out, double *elapsedtime){
    double et = 0.0f;
    unsigned long index = 0;
    bool found = false;
    et = omp_get_wtime();
    while ((index < size) && (!found)){
        if (value > vet[index])
            index++;
        else if (value < vet[index]){
            index = size;
        }else{
            *out = index;
            found = true;
        }
    }//while ((index < size) && (!found)){
    et = omp_get_wtime() - et;
    *elapsedtime = et;
    return found;
}


bool find_linear_omp(unsigned long size, unsigned long value, unsigned long *vet, unsigned long *out, double *elapsedtime){

    unsigned long l_bound = 0,
                  u_bound = size;

    bool          found   = false,
                  t_found = false;
    double et;

    omp_lock_t writelock;

    omp_init_lock(&writelock);


#pragma omp parallel private (l_bound, u_bound, t_found, et) shared (vet, out, value, found)
{
    et  = omp_get_wtime();
    unsigned int subsize = size / omp_get_num_threads(),
                 residual = 0;

    if (omp_get_thread_num() == omp_get_num_threads() - 1){
        residual = size % omp_get_num_threads();
    }

    l_bound = subsize * omp_get_thread_num();
    u_bound= subsize * (omp_get_thread_num() + 1) + residual;

    while ((l_bound < u_bound) && (!t_found)){
        if (value > vet[l_bound])
            l_bound++;
        else if (value < vet[l_bound]){
            l_bound = u_bound;
        }else{
            t_found = true;
        }
    }//while ((index < size) && (!found)){

    if (t_found){
        omp_set_lock(&writelock);
        *out = l_bound;
        found = t_found;
        omp_unset_lock(&writelock);
    }//if (found){
    et = omp_get_wtime() - et;
    elapsedtime[omp_get_thread_num()] = et;
}
    omp_destroy_lock(&writelock);
    return found;
}



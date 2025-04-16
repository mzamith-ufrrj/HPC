//-nvcc -arch=sm_11 -m64 -O3 main.cu -o atomic.bin


#include<iostream>
#include<cstdlib>
#include <cuda_runtime.h>
#include <cassert>
#include <vector>



#define CHECK_ERROR(call) do {                                                    \
   if( cudaSuccess != call) {                                                             \
      std::cerr << std::endl << "CUDA ERRO: " <<                             \
         cudaGetErrorString(call) <<  " in file: " << __FILE__                \
         << " in line: " << __LINE__ << std::endl;                               \
         exit(0);                                                                                 \
   } } while (0)
   
__global__ 
void kernel (int *vet, int *flag){
   
   unsigned int index = blockDim.x * blockIdx.x + threadIdx.x;
   vet[index] = index + 1;
   
   if (threadIdx.x == 0)
      atomicAdd(&flag[0], 1);
       

}



   
using namespace std;
int main(int argc, char *argv[]){

                      
 
   
     
   int  dominio   = 32,
         threads  = 4;
   
   vector <int> h_vet;
   int *d_Vet  = NULL,
        *d_Flag = NULL;
   


   
   
   cout << "\nOperacao atomica\n";
   
     //Reset no device
   CHECK_ERROR(cudaDeviceReset());


   //Alocando memória
   h_vet.resize(dominio);
   cudaMalloc(reinterpret_cast<void**> (&d_Vet), dominio * sizeof(int));
   cudaMalloc(reinterpret_cast<void**> (&d_Flag), 1 * sizeof(int));
   
   //Inicializando variáveis
   bzero(&(h_vet[0]), dominio * sizeof(float));
   CHECK_ERROR(cudaMemset(d_Vet, 0, dominio * sizeof(int)));
   CHECK_ERROR(cudaMemset(d_Flag, 0, 1 * sizeof(int)));
   
   int blocos = dominio / threads;
   
   cout << "Blocos: " << blocos << endl;
   cout << "Threads: " << threads << endl;
    


   kernel<<<blocos, threads>>> (d_Vet, d_Flag);
   
   CHECK_ERROR(cudaDeviceSynchronize());
   
   cudaMemcpy(&(h_vet[0]), d_Vet, dominio * sizeof(int),  cudaMemcpyDeviceToHost); 
   
   for (int k = 0; k < dominio; k++)
      cout <<  h_vet[k] << endl;
   cout << endl;
    
   cudaMemcpy(&(h_vet[0]), d_Flag, 1 * sizeof(int),  cudaMemcpyDeviceToHost); 
   
   cout << "cada thread[0] soma 1: " << h_vet[0] << endl;

   
   
   cudaFree(d_Vet);
   cudaFree(d_Flag);
   
   return EXIT_SUCCESS;
}


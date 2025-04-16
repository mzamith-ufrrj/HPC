#include <iostream>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>

#define CHECK_ERROR(call) do {         \
   if( cudaSuccess != call) {          \
      std::cerr << std::endl << "CUDA ERRO: " <<             \
         cudaGetErrorString(call) <<  " in file: " << __FILE__  \
         << " in line: " << __LINE__ << std::endl;   \
         exit(0); \
   } } while (0)


using namespace std;
__device__ float myMax(float a, float b){
    if (a > b) return a;
    else return b;
}

__global__  void AddVet(float *c, float *a, float *b){

  const int i = blockDim.x * blockIdx.x + threadIdx.x;
  c[i] = myMax(a[i], b[i]) + a[i] + b[i];

}

int main (int argc, char **argv){
   int h_Size = 21;
   
      float  *h_VetA = NULL,
           *h_VetB = NULL,
           *h_VetC = NULL,
           *d_VetA = NULL,
           *d_VetB = NULL,
           *d_VetC = NULL;

    size_t free = 0,
          total = 0;
   cout << endl << "CUDA runtime versao: " << CUDART_VERSION << endl;
   
   //Reset no device
   CHECK_ERROR(cudaDeviceReset());

   //Verificando espaço livre em memória
   CHECK_ERROR(cudaMemGetInfo(&free, &total));
   cout << "Memoria livre: " << (free / 1024 / 1024)   << " MB\n";
   cout << "Memoria total: " << (total / 1024 / 1024)  << " MB\n";
   
 
   
   //Aloca memória GPU
   CHECK_ERROR(cudaMalloc((void**) &d_VetA, h_Size * sizeof(float)));
   
    CHECK_ERROR(cudaMalloc((void**) &d_VetB, h_Size * sizeof(float)));
   
    
    CHECK_ERROR(cudaMalloc((void**) &d_VetC, h_Size * sizeof(float)));
   
   //Alocando memória na CPU
   h_VetA = new float [h_Size];
   h_VetB = new float [h_Size];
   h_VetC = new float [h_Size];
   
   //Inicializa vetor C (resultado) com zero - illustração
   CHECK_ERROR(cudaMemset(d_VetC, 0, h_Size * sizeof(float)));    

   //Inicializando memoria da CPU
   memset(h_VetC, 0,  h_Size * sizeof(float));
   
   //Preenchendo vetores
   for (int i = 0; i < h_Size; i++){
      h_VetA[i] = static_cast <float> (i+1);
      h_VetB[i] =  h_VetA[i] * 0.01f;
   }
   
   //Copiando CPU --> GPU

   
   CHECK_ERROR(cudaMemcpy(d_VetA, h_VetA, h_Size * sizeof(float),  cudaMemcpyHostToDevice)); 
   
    
    CHECK_ERROR(cudaMemcpy(d_VetB, h_VetB, h_Size * sizeof(float),  cudaMemcpyHostToDevice));
   
   int numBlocks = 3;
   int threadsPerBlock = h_Size / numBlocks;
   
   
   AddVet<<<numBlocks, threadsPerBlock>>> (d_VetC, d_VetA, d_VetB);
   
   CHECK_ERROR(cudaDeviceSynchronize());
   
   
   CHECK_ERROR(cudaMemcpy(h_VetC, d_VetC, h_Size * sizeof(float),  cudaMemcpyDeviceToHost));
   
   
   cout <<  endl << "Resultado: "<<  endl;
   
   for (int i = 0; i < h_Size; i++)
      cout << h_VetC[i] << endl;
  
   
   CHECK_ERROR(cudaFree(d_VetA));  //Liberando memorias GPU e CPU
   CHECK_ERROR(cudaFree(d_VetB));  //Liberando memorias GPU e CPU
   CHECK_ERROR(cudaFree(d_VetC));  //Liberando memorias GPU e CPU
  
   
   
   delete[] h_VetA;
   delete[] h_VetB;
   delete[] h_VetC;
   

   cout << "FIM" << endl;
   
   return EXIT_SUCCESS;
}



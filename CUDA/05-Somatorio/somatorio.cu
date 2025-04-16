#include <iostream>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>

#define CHECK_ERROR(call) do {                                                    \
   if( cudaSuccess != call) {                                                             \
      std::cerr << std::endl << "CUDA ERRO: " <<                             \
         cudaGetErrorString(call) <<  " in file: " << __FILE__                \
         << " in line: " << __LINE__ << std::endl;                               \
         exit(0);                                                                                 \
   } } while (0)


using namespace std;


__global__  void Soma2(float *b, float *a){

   extern __shared__ float partialSum[];

    int t = threadIdx.x;

   partialSum[t] = a[t];
   __syncthreads();

  
   for (unsigned int stride = blockDim.x >> 1; stride >= 0; stride >>= 1){
      __syncthreads();
      if (t % (2* (stride)) == 0)
        partialSum[t] += partialSum[t+stride];
   }

   __syncthreads();
   if (threadIdx.x  == 0)
      b[t] = partialSum[t];


}

__global__  void Soma1(float *b, float *a){

   extern __shared__ float partialSum[];

    int t = threadIdx.x;

   partialSum[t] = a[t];
   __syncthreads();

   for (unsigned int stride = 1; stride < blockDim.x; stride *=2){
      __syncthreads();
      if (t % (2* (stride)) == 0)
        partialSum[t] += partialSum[t+stride];
   }

   __syncthreads();
   if (threadIdx.x  == 0)
      b[t] = partialSum[t];


}

int main (int argc, char **argv){
   int h_Size = 4;
   
   float elapsedTimeGPU = 0.0f,
         elapsedTimeMEM = 0.0f;
   
   float   *h_VetA = NULL,
           *d_VetA = NULL,
           *d_VetB = NULL;
   
   cudaEvent_t e_Start,
               e_Stop;

   size_t free = 0,
          total = 0;
   cout << endl << "CUDA runtime versao: " << CUDART_VERSION << endl;
   
   //Reset no device
   CHECK_ERROR(cudaDeviceReset());

   //Verificando espaço livre em memória
   CHECK_ERROR(cudaMemGetInfo(&free, &total));
   cout << "Memoria livre: " << (free / 1024 / 1024)   << " MB\n";
   cout << "Memoria total: " << (total / 1024 / 1024)  << " MB\n";
   
 
      //Criando eventos
   CHECK_ERROR(cudaEventCreate(&e_Start));
   CHECK_ERROR(cudaEventCreate(&e_Stop));
   
   //Aloca memória GPU
   CHECK_ERROR(cudaMalloc((void**) &d_VetA, h_Size * sizeof(float)));
   CHECK_ERROR(cudaMalloc((void**) &d_VetB, h_Size * sizeof(float)));
   
   
   //Alocando memória na CPU
   h_VetA = new float [h_Size];
   
   
   //Preenchendo vetores
   for (int i = 0; i < h_Size; i++){
      h_VetA[i] = static_cast <float> (i+1);
   }
   
   //Copiando CPU --> GPU
   CHECK_ERROR(cudaEventRecord(e_Start, cudaEventDefault));
   CHECK_ERROR(cudaMemcpy(d_VetA, h_VetA, h_Size * sizeof(float),  cudaMemcpyHostToDevice)); 
   
   CHECK_ERROR(cudaEventRecord(e_Stop, cudaEventDefault));
   CHECK_ERROR(cudaEventSynchronize(e_Stop));
   CHECK_ERROR(cudaEventElapsedTime(&elapsedTimeMEM, e_Start, e_Stop));

   CHECK_ERROR(cudaEventRecord(e_Start, cudaEventDefault));
   
   int numBlocks = 1;
   int threadsPerBlock = h_Size / numBlocks;
   
   
   Soma2<<<numBlocks, threadsPerBlock,  h_Size * sizeof(float) >>> (d_VetB, d_VetA);
   
   CHECK_ERROR(cudaDeviceSynchronize());
   
   CHECK_ERROR(cudaEventRecord(e_Stop, cudaEventDefault));
   CHECK_ERROR(cudaEventSynchronize(e_Stop));
   CHECK_ERROR(cudaEventElapsedTime(&elapsedTimeGPU, e_Start, e_Stop));

   
   //Copiando GPU --> CPU
   float elapsedTime = 0.0f;
   CHECK_ERROR(cudaEventRecord(e_Start, cudaEventDefault));
   
   CHECK_ERROR(cudaMemcpy(h_VetA, d_VetB, h_Size * sizeof(float),  cudaMemcpyDeviceToHost));
   
   CHECK_ERROR(cudaEventRecord(e_Stop, cudaEventDefault));
   CHECK_ERROR(cudaEventSynchronize(e_Stop));
   CHECK_ERROR(cudaEventElapsedTime(&elapsedTime, e_Start, e_Stop));
   elapsedTimeMEM += elapsedTime;
  
   cout << endl << "Tempo gasto [MEM]: " << elapsedTimeMEM << " (ms)" << endl;
   cout << endl << "Tempo gasto [GPU]: " << elapsedTimeGPU << " (ms)" << endl;
   
   cout <<  endl << "Resultado: "<<  endl;
   
/*
   for (int i = 0; i < h_Size; i++)
      cout << h_VetA[i] << endl;
*/  
      cout << h_VetA[0] << endl;
  
   
   CHECK_ERROR(cudaFree(d_VetA));  //Liberando memorias GPU e CPU
   
   delete[] h_VetA;
   

   cout << "FIM" << endl;
   
   return EXIT_SUCCESS;
}

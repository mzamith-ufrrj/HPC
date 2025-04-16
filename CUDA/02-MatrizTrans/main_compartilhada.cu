
#include<iostream>
#include<cstdlib>
#include <cuda_runtime.h>
#include <cassert>

#define BLOCOS        2
#define THREADS      4
#define REPETICOES  4

#define CHECK_ERROR(call) do {                                                    \
   if( cudaSuccess != call) {                                                             \
      std::cerr << std::endl << "CUDA ERRO: " <<                             \
         cudaGetErrorString(call) <<  " in file: " << __FILE__                \
         << " in line: " << __LINE__ << std::endl;                               \
         exit(0);                                                                                 \
   } } while (0)


/*
 *************************************************************************
   unsigned int width = gridDim.x * blockDim.x;
   unsigned int height = gridDim.y * blockDim.y;
   unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;
   unsigned int y = blockDim.y * blockIdx.y + threadIdx.y;
   unsigned int kn = y * width + x;
 *************************************************************************
 N blocks x M threads  <---- IPC
*/



__global__ 
void interveMatriz_compartilhada(float *out_Matriz, float *in_Matriz, const int largura, const int altura){
   
   __shared__ float buff[THREADS * REPETICOES][THREADS * REPETICOES];
   unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;
   unsigned int y = blockDim.y * blockIdx.y + threadIdx.y;
   
   
  
   
   for (int j = 0; j < REPETICOES; j++)
      for (int i = 0; i < REPETICOES; i++){
         buff[ threadIdx.y + (j * blockDim.y)][ threadIdx.x + (i * blockDim.x) ] = in_Matriz[(y+(j * blockDim.y * gridDim.y)) * largura + (x+(i * blockDim.x * gridDim.x))];     
      }

   __syncthreads();

   for (int j = 0; j < REPETICOES; j++)
      for (int i = 0; i < REPETICOES; i++){
          out_Matriz[ (x+(i * blockDim.x * gridDim.x)) * largura + (y+(j * blockDim.y * gridDim.y))] =  buff[ threadIdx.y + (j * blockDim.y)][ threadIdx.x + (i * blockDim.x) ];
      }



}

using namespace std;
int main (int argc, char **argv){


        
   int largura  = BLOCOS * THREADS * REPETICOES, 
        altura   = BLOCOS * THREADS * REPETICOES;
        
   dim3 numBlocos (BLOCOS, BLOCOS, 1),
           numThreads(THREADS , THREADS, 1);


   float  *h_InMatriz    = NULL,
           *h_OutMatriz  = NULL,
           *d_InMatriz    = NULL,
           *d_OutMatriz  = NULL;

   
   float elapsedTimeGPU = 0.0f,
          elapsedTimeMEM = 0.0f;
   
   unsigned int qtdeDados = largura * altura * sizeof(float);
   
   cudaEvent_t e_Start,
                      e_Stop;
                      
   size_t free = 0,
            total = 0;
   
   cudaDeviceProp deviceProp;                   //Levantar a capacidade do device
   cudaGetDeviceProperties(&deviceProp, 0);
   
   
   cout << "\nMatriz transposta\n";
   
     //Reset no device
   CHECK_ERROR(cudaDeviceReset());

   

   //Verificando espaço livre em memória
   CHECK_ERROR(cudaMemGetInfo(&free, &total));
   cout << "Memoria livre: " << (free / 1024 / 1024)   << " MB\n";
   cout << "Memoria total: " << (total / 1024 / 1024)  << " MB\n";
   cout << "Tamanho da matriz: " << "(" << largura << ", " << altura << endl;
 
      //Criando eventos
   CHECK_ERROR(cudaEventCreate(&e_Start));
   CHECK_ERROR(cudaEventCreate(&e_Stop));
   
   //Aloca memória GPU
   CHECK_ERROR( cudaMalloc((void**) &d_InMatriz, qtdeDados) );
   CHECK_ERROR( cudaMalloc((void**) &d_OutMatriz, qtdeDados) );

   //Alocando memória na CPU
   h_InMatriz = new float [largura * altura];
   h_OutMatriz = new float[largura * altura];
      
   //Inicializações
   CHECK_ERROR( cudaMemset(d_OutMatriz, 0,  qtdeDados) );
   memset(h_OutMatriz, 0,  qtdeDados);
   
   for (int j = 0; j < altura; j++){
      for (int i = 0; i < largura; i++){
         int ptr = j * largura + i;
         h_InMatriz[ptr] = static_cast <float> (ptr+10);
         cout << h_InMatriz[ptr] << " ";
      }
     cout << endl;
   }
   
/*
   dim3 numBlocos (blocoX, blocoY, 1),
           numThreads(largura / numBlocos.x / numBlocos.x, altura / numBlocos.y / numBlocos.y, 1);
*/
   
   cout << "Blocos: " << numBlocos.x << ", " << numBlocos.y << endl;
   cout << "Threads: " << numThreads.x << ", " << numThreads.y << endl;
   
   CHECK_ERROR(cudaEventRecord(e_Start, cudaEventDefault));
   
   CHECK_ERROR(cudaMemcpy(d_InMatriz, h_InMatriz, qtdeDados,  cudaMemcpyHostToDevice)); 

   
   CHECK_ERROR(cudaEventRecord(e_Stop, cudaEventDefault));
   CHECK_ERROR(cudaEventSynchronize(e_Stop));
   CHECK_ERROR(cudaEventElapsedTime(&elapsedTimeMEM, e_Start, e_Stop));

   assert( (numThreads.x * numThreads.y) <= deviceProp.maxThreadsDim[0]);

   CHECK_ERROR(cudaEventRecord(e_Start, cudaEventDefault));

//   cout << deviceProp.maxThreadsDim[0];
                     
    interveMatriz_compartilhada<<<numBlocos, numThreads>>> (d_OutMatriz, d_InMatriz, largura, altura);
    //interveMatriz<<<numBlocos, numThreads>>> (d_OutMatriz, d_InMatriz, largura, altura);
   
   CHECK_ERROR(cudaDeviceSynchronize());
   
   CHECK_ERROR(cudaEventRecord(e_Stop, cudaEventDefault));
   CHECK_ERROR(cudaEventSynchronize(e_Stop));
   CHECK_ERROR(cudaEventElapsedTime(&elapsedTimeGPU, e_Start, e_Stop));

   
   //Copiando GPU --> CPU
   float elapsedTime = 0.0f;
   CHECK_ERROR(cudaEventRecord(e_Start, cudaEventDefault));

   CHECK_ERROR(cudaMemcpy(h_OutMatriz, d_OutMatriz, qtdeDados,  cudaMemcpyDeviceToHost)); 

   CHECK_ERROR(cudaEventRecord(e_Stop, cudaEventDefault));
   CHECK_ERROR(cudaEventSynchronize(e_Stop));
   CHECK_ERROR(cudaEventElapsedTime(&elapsedTime, e_Start, e_Stop));
   elapsedTimeMEM += elapsedTime;
   elapsedTimeMEM /= 1000.0f;
   elapsedTimeGPU /= 1000.0f;
   cout << endl << "Tempo gasto [MEM]: " << elapsedTimeMEM << " (s)" << endl;
   cout << endl << "Tempo gasto [GPU]: " << elapsedTimeGPU << " (s)" << endl;
   float gsample = largura * altura  / elapsedTimeGPU;
   cout << endl<< "Sample: " << gsample << endl;
   gsample *= 1.0e-9;
   cout << endl<< "Gigasample: " << gsample << endl;

   /*
   cout <<  endl << "Resultado: "<<  endl;
   for (int j = 0; j < altura; j++){
      for (int i = 0; i < largura; i++){
         int ptr = j * largura + i;
         cout << h_OutMatriz[ptr] << " ";
      }
      cout << endl;
   }
   */
   
   CHECK_ERROR( cudaFree(d_InMatriz) );  //Liberando memorias GPU e CPU
   CHECK_ERROR( cudaFree(d_OutMatriz) );  //Liberando memorias GPU e CPU
   
   delete[] h_InMatriz;
   delete[] h_OutMatriz;
   cout << "\nFIM\n";
   return EXIT_SUCCESS;
}

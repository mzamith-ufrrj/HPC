#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#define BLOCK_SIZE 4



#define CHECK_ERROR(call) do {                                                    \
   if( cudaSuccess != call) {                                                     \
      fprintf(stderr,"CUDA ERROR:%s in file: %s in line: ", cudaGetErrorString(call),  __FILE__, __LINE__); \
         exit(0);                                                                                 \
   } } while (0)

__device__ float * getSubMatrix(float *A, int row, int col, int stride)
{
  float *Asub;
//  Asub.width = BLOCK_SIZE;
//  Asub.height = BLOCK_SIZE;
//  Asub.stride = A.stride;
  Asub = &A[stride * BLOCK_SIZE * row + BLOCK_SIZE * col];
  return Asub;
}

// Get a matrix element

__device__ float getElement(const float *A, int row, int col,  int stride){
  return A[row *  stride + col];
}

// Set a matrix element
__device__ void setElement(float *A, int row, int col, int stride, float value)
{
  A[row * stride + col] = value ;
}


/*
 * Multiplicação usando memória compartilhada
 */
__global__ void multMatrixS(float *C, float *A, float *B, const int width)
{

  int blockRow = blockIdx.y;
  int blockCol = blockIdx.x;

  float *Csub = getSubMatrix(C, blockRow, blockCol, width);

  float Cvalue = 0;

  int row = threadIdx.y;
  int col = threadIdx.x;

  for (int m = 0; m < (width / BLOCK_SIZE); ++m) {

      float *Asub = getSubMatrix(A, blockRow, m, width);

      float *Bsub = getSubMatrix(B, m, blockCol, width);

      __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
      __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

      As[row][col] = getElement(Asub, row, col, width);
      Bs[row][col] = getElement(Bsub, row, col, width);

      __syncthreads();
      for (int e = 0; e < BLOCK_SIZE; ++e)
         Cvalue += As[row][e] * Bs[e][col];

      __syncthreads();
  }

  setElement(Csub, row, col, width, Cvalue);
}


/*
 * Multiplicação usando memória global
 */

__global__ void multMatrixG(float *C, float *A, float *B, const int width)
{
    float Cvalue = 0;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    for (int e = 0; e < width; ++e){
      Cvalue += getElement(A, row, e, width) * getElement(B, e, col, width);
      //Cvalue += A[row * width + e] * B[e * width + col];
    }
    setElement(C, row, col, width, Cvalue);
//    C[row * width + col] = Cvalue;
}



void printMatrix(float *m, float w, float h){
   int i, j;

   printf("\n");

   for (j = 0; j < h; j++){
      for (i = 0; i < w; i++){
         int k = j * w + i;
         printf("%.2f ", m[k]);
      }
      printf("\n");
   }

}

int main (int argc, char **argv){

   assert(argc == 4);
   float *h_A, *h_B, *h_C;
   int iC, jC;

   int width      = atoi(argv[1]);
   int height     = atoi(argv[2]);
   
   char memType[256];

   float elapsedTimeGPU = 0.0f,
         elapsedTimeMEM = 0.0f;
   
   float   *d_C = NULL,
           *d_A = NULL,
           *d_B = NULL;
 
   cudaEvent_t e_Start,
               e_Stop;

   strcpy(memType, argv[3]);

   printf("\nMultiplicando matriz - GPU\n");
   printf("Tamanho da matriz: %d x %d - memoria: [%s]\n", width, height, memType);

    h_A = (float*) malloc (width * height * sizeof(float));
    h_B = (float*) malloc (width * height * sizeof(float));
    h_C = (float*) malloc (width * height * sizeof(float));


   for (jC = 0; jC < height; jC++){
      for (iC = 0; iC < width; iC++){
         int kC = jC * width + iC;
         h_A[kC] = (float) kC + 1;
         if (jC == iC)
           h_B[kC] = 1.0f;
         else
            h_B[kC] = 0.0f;
         
      }
   }

   //Reset no device
   CHECK_ERROR(cudaDeviceReset());

 
      //Criando eventos
   CHECK_ERROR(cudaEventCreate(&e_Start));
   CHECK_ERROR(cudaEventCreate(&e_Stop));
   
   //Aloca memória GPU
   CHECK_ERROR(cudaMalloc((void**) &d_A, width * height * sizeof(float)));
   CHECK_ERROR(cudaMalloc((void**) &d_B, width * height * sizeof(float)));
   CHECK_ERROR(cudaMalloc((void**) &d_C, width * height * sizeof(float)));
   
   
   
   //Copiando CPU --> GPU
   CHECK_ERROR(cudaEventRecord(e_Start, cudaEventDefault));
   CHECK_ERROR(cudaMemcpy(d_A, h_A, width * height * sizeof(float),  cudaMemcpyHostToDevice)); 
   CHECK_ERROR(cudaMemcpy(d_B, h_B, width * height * sizeof(float),  cudaMemcpyHostToDevice)); 
   
   CHECK_ERROR(cudaEventRecord(e_Stop, cudaEventDefault));
   CHECK_ERROR(cudaEventSynchronize(e_Stop));
   CHECK_ERROR(cudaEventElapsedTime(&elapsedTimeMEM, e_Start, e_Stop));

   CHECK_ERROR(cudaEventRecord(e_Start, cudaEventDefault));
   
   //int numBlocks = 1;
   //int threadsPerBlock = WIDTH*HEIGHT / numBlocks;

  dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE, 1);
  dim3 numBlocks(width / threadsPerBlock.x, height / threadsPerBlock.y, 1);
   

//   printMatrix(h_A, WIDTH, HEIGHT);
   if (strcmp(memType, "shared") == 0){
      printf(" s ");
      multMatrixS <<<numBlocks, threadsPerBlock >>> (d_C, d_A, d_B, width);
   }else{
     printf(" g ");
     multMatrixG <<<numBlocks, threadsPerBlock >>> (d_C, d_A, d_B, width);
  }
   CHECK_ERROR(cudaDeviceSynchronize());
   
   CHECK_ERROR(cudaEventRecord(e_Stop, cudaEventDefault));
   CHECK_ERROR(cudaEventSynchronize(e_Stop));
   CHECK_ERROR(cudaEventElapsedTime(&elapsedTimeGPU, e_Start, e_Stop));

   
   //Copiando GPU --> CPU
   float elapsedTime = 0.0f;
   CHECK_ERROR(cudaEventRecord(e_Start, cudaEventDefault));
   
   CHECK_ERROR(cudaMemcpy(h_C, d_C,  width*height * sizeof(float),  cudaMemcpyDeviceToHost));
   
   CHECK_ERROR(cudaEventRecord(e_Stop, cudaEventDefault));
   CHECK_ERROR(cudaEventSynchronize(e_Stop));
   CHECK_ERROR(cudaEventElapsedTime(&elapsedTime, e_Start, e_Stop));
   elapsedTimeMEM += elapsedTime;
  
   printf("Tempo gasto [MEM]: %lf (ms) \n", elapsedTimeMEM);
   printf("Tempo gasto [GPU]: %lf (ms) \n", elapsedTimeGPU);
   printf("Resultado:\n");
   printMatrix(h_C, width, height);
 
   
   CHECK_ERROR(cudaFree(d_A));  //Liberando memorias GPU e CPU
   CHECK_ERROR(cudaFree(d_B));  //Liberando memorias GPU e CPU
   CHECK_ERROR(cudaFree(d_C));  //Liberando memorias GPU e CPU
   
   free(h_A);
   free(h_B);
   free(h_C);
   
   printf("FIM\n");

   return EXIT_SUCCESS;
}

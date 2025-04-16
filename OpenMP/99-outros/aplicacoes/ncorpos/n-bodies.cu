#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <App.h>
#include <cuda_profiler_api.h>
#define BLOCK_SIZE 1024
//#define BLOCK_SIZE 1024
#define EPSILON    1E-9
#define G          1.0
#define DT         1.5
/*Declarando as structs de particula e forca*/
struct stCoord{
    float x,
          y,
          z;
};
typedef struct stCoord tpCoord;
struct stParticle
{
    tpCoord p,
            v,
            f;
    float m;
};
typedef struct stParticle tpParticle;


void printLog(tpParticle *particles, int nParticles){
    char fileName[128];
    sprintf(fileName, "%s.txt", __FILE__);
    fprintf(stdout, "Saving file [%s] ", fileName); fflush(stdout);
    FILE *ptr = fopen(fileName, "w+");
      for(int i = 0; i < nParticles; i++){
            fprintf(ptr, "%d \t %.10f %.10f %.10f \t %.10f %.10f %.10f \t %.10f %.10f %.10f \n", i,  particles[i].p.x, particles[i].p.y, particles[i].p.z,  particles[i].v.x, particles[i].v.y, particles[i].v.z, particles[i].f.x, particles[i].f.y, particles[i].f.z);

    }
    fclose(ptr);
    fprintf(stdout, "[OK]\n"); fflush(stdout);
}


void initialCondition(tpParticle *particles, int nParticles){

    srand(42);

    memset(particles, 0x00, nParticles * sizeof(tpParticle));
    
    for (int i = 0; i < nParticles ; i++){
        particles[i].p.x =  2.0 * (rand() / (float)RAND_MAX) - 1.0;
        particles[i].p.y =  2.0 * (rand() / (float)RAND_MAX) - 1.0;
        particles[i].p.z =  2.0 * (rand() / (float)RAND_MAX) - 1.0;
        particles[i].m =  1;
     
     
      }




}

__device__ float distance(  float* dx,
                            float* dy,
                            float* dz,
                            const tpParticle A,
                            const tpParticle B){
    float x = A.p.x - B.p.x;
    float y = A.p.y - B.p.y;
    float z = A.p.z - B.p.z;
    *dx = x;
    *dy = y;
    *dz = z;
    float d = (x * x) + (y * y) + (z * z) + EPSILON;
    return d;

}

__global__ void particleParticle_k(tpParticle *particles){
    extern __shared__ tpParticle subParticles[];
    int i  = blockDim.x * blockIdx.x + threadIdx.x;
//  __shared__ tpParticle subParticles[BLOCK_SIZE];
    float sum_fx = 0.0,
          sum_fy = 0.0,
          sum_fz = 0.0,
          mi = particles[i].m;

    for (int blk = 0; blk < gridDim.x; blk++){
        subParticles[threadIdx.x] = particles[ blockDim.x * blk + threadIdx.x];
        __syncthreads();

        for (int j = 0; j < blockDim.x; j++){
            float dx = 0.0f,  dy = 0.0f, dz = 0.0f;
            float d  = distance(&dx, &dy, &dz, particles[i], subParticles[j]);
            float mj = subParticles[j].m;
            sum_fx += (dx * mj * mi) / (d * d * d) ;
            sum_fy += (dy * mj * mi) / (d * d * d) ;
            sum_fz += (dz * mj * mi) / (d * d * d) ;
        }//end-for (int j = 0; j < blockDim.x; j++){

        __syncthreads();
        particles[i].f.x = -G * sum_fx;
        particles[i].f.y = -G * sum_fy;
        particles[i].f.z =- G * sum_fz;
    }//end-for (int blk = 0; blk < gridDim.x; blk++){



    particles[i].v.x += DT * particles[i].f.x;
    particles[i].v.y += DT * particles[i].f.y;
    particles[i].v.z += DT * particles[i].f.z;

    particles[i].p.x += DT * particles[i].v.x;
    particles[i].p.y += DT * particles[i].v.y;
    particles[i].p.z += DT * particles[i].v.z;
}

void saveLog(char *logFile, 
             unsigned long mem, 
             double processor, double total){

   FILE *ptr = NULL;
   struct stat buffer;   
   int ret = stat(logFile, &buffer); 
   if (ret == -1){
      ptr = fopen(logFile, "w+");
      assert(ptr != NULL);
      fprintf(ptr, "memoryUsed;totalTime;processorTime;overheadTime\n");

   }else{
      ptr = fopen(logFile, "a+");
   }
   double overhead = total - processor;
   
   fprintf(ptr, "%lu;%lf;%lf;%lf\n", mem, total, processor, overhead);
   fclose(ptr);
   printf("Arquivo: %s salvo\n", logFile);
}
int main (int ac, char **av){
    int timesteps  = atoi(av[1]),
        nParticles = atoi(av[2]),
        flagSave = atoi(av[4]),
        threads = 0,
        blocks  = 0;

    char logFile[1024];
    Stopwatch   totalTime, processorTime;
    tpParticle *h_particles = NULL,
               *d_particles = NULL;

    START_STOPWATCH(totalTime);
    unsigned long mem = nParticles * sizeof(tpParticle);

    strcpy(logFile, av[3]);

    fprintf(stdout, "\nP2P particle system - CUDA version\n");
    fprintf(stdout, "Memory used %lu bytes - CUDA memory used: %lu \n", mem, mem);
    fprintf(stdout, "Log %s \n", logFile);


    h_particles = (tpParticle *) aligned_alloc(ALING, mem);
    assert(h_particles != NULL);
    initialCondition(h_particles, nParticles);
    
    threads = BLOCK_SIZE,
    blocks  = nParticles / BLOCK_SIZE;
    if (nParticles < 1000){
      blocks = 1;
      threads = nParticles;
    }
    assert(cudaDeviceReset()== cudaSuccess);
    assert(cudaMalloc((void**) &d_particles, mem) == cudaSuccess);
    assert(cudaMemcpy(d_particles, h_particles, mem,  cudaMemcpyHostToDevice) == cudaSuccess);
    //assert((nParticles % threads) == 0);
    fprintf(stdout, "\n B(%d) T(%d) \n", blocks, threads);

    START_STOPWATCH(processorTime);
    cudaProfilerStart();
    for (int t = 0; t < timesteps; t++){
      particleParticle_k<<<blocks, threads, threads * sizeof(tpParticle)>>>(d_particles);
      assert( cudaDeviceSynchronize() == cudaSuccess);
    }//end-for (int t = 0; t < timesteps; t++){
    cudaProfilerStop();
    STOP_STOPWATCH(processorTime);
    assert(cudaMemcpy(h_particles, d_particles, nParticles * sizeof(tpParticle),  cudaMemcpyDeviceToHost) == cudaSuccess);
    cudaFree(d_particles);

    if(flagSave)
        printLog(h_particles, nParticles);
    free(h_particles);

    STOP_STOPWATCH(totalTime);
    saveLog(logFile, mem, processorTime.mElapsedTime, totalTime.mElapsedTime);
    return EXIT_SUCCESS;
}

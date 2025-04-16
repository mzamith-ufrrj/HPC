#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <omp.h>
#include <sys/types.h>
#include <sys/stat.h>
#include "App.h"
#define EPSILON 1E-9
#define G 1.0
#define DT 1.5
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

float distance(float *dx, float *dy, float *dz, tpParticle A, tpParticle B){
    float x = A.p.x - B.p.x;
    float y = A.p.y - B.p.y;
    float z = A.p.z - B.p.z;
    *dx = x;
    *dy = y;
    *dz = z;
    float d = (x * x) + (y * y) + (z * z) + EPSILON;
    return d;

}



double particleParticle (tpParticle *particles, int nParticles){
Stopwatch   processorTime;
#pragma omp parallel shared(particles, nParticles)
{
//---------------------------------------------------
   
    if (omp_get_thread_num() == 0){
        START_STOPWATCH(processorTime);
    }
    #pragma omp for
    for(int i = 0; i < nParticles; i++){
        float sum_fx = 0.0,
                sum_fy = 0.0,
                sum_fz = 0.0,
                mi = particles[i].m;
        
        for(int j = 0; j < nParticles; j++){
            if (i != j){
                float dx = 0.0f, dy = 0.0f, dz = 0.0f;
                float d  = distance(&dx, &dy, &dz, particles[i], particles[j]);
                float mj = particles[j].m;
                sum_fx += (dx * mj * mi) / (d * d * d) ;
                sum_fy += (dy * mj * mi) / (d * d * d) ;
                sum_fz += (dz * mj * mi) / (d * d * d) ;

            }
        }//for(int j = 0; j < nParticles; j++){

        particles[i].f.x = -G * sum_fx;
        particles[i].f.y = -G * sum_fy;
        particles[i].f.z =- G * sum_fz;

    

    }//end-for(int i = 0; i < nParticles; i++){

    #pragma omp barrier
    #pragma omp for
    for(int i = 0; i < nParticles; i++){
        particles[i].v.x += DT * particles[i].f.x;
        particles[i].v.y += DT * particles[i].f.y;
        particles[i].v.z += DT * particles[i].f.z;

        particles[i].p.x += DT * particles[i].v.x;
        particles[i].p.y += DT * particles[i].v.y;
        particles[i].p.z += DT * particles[i].v.z;

    }//end-for(int i = 0; i < nParticles; i++){

    #pragma omp barrier
    if (omp_get_thread_num() == 0){
        STOP_STOPWATCH(processorTime);
    }

    
}//#pragma omp parallel shared(particles, nParticles, dt)
    return processorTime.mElapsedTime;
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
        flagSave = atoi(av[4]);

    char logFile[1024];
    
    tpParticle *particles = NULL;
    
    Stopwatch   totalTime;
    double      processorTime = 0.0f;
    START_STOPWATCH(totalTime);
    
    unsigned long mem = nParticles * sizeof(tpParticle);
    strcpy(logFile, av[3]);

    fprintf(stdout, "\nP2P particle system - OpenMP version\n");
    fprintf(stdout, "Memory used %lu bytes \n", mem);
    fprintf(stdout, "Log %s \n", logFile);


    particles = (tpParticle *) aligned_alloc(ALING, nParticles * sizeof(tpParticle));
    assert(particles != NULL);

     
    initialCondition(particles, nParticles);
    //printParticle(particles, nParticles);
    //printLog(particles, nParticles, 0);
    processorTime = 0.0f;
    for (int t = 1; t < timesteps; t++){
        processorTime += particleParticle (particles, nParticles);
    }
   
    if(flagSave)
        printLog(particles, nParticles);
    free(particles);
    STOP_STOPWATCH(totalTime);

    saveLog(logFile, mem, processorTime, totalTime.mElapsedTime);
    return EXIT_SUCCESS;
}

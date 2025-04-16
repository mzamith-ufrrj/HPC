/* 
 * File:   main.c
 * Author: marcelozamith
 *
 * Created on June 9, 2010, 3:23 PM
 */

#include <stdio.h>
#include <stdlib.h>
#include <vector_types.h>
#include <cuda.h>
#include <cuda_runtime_api.h>


void GPUinfo(void)
{
    cudaDeviceProp deviceProp;

    int gpu = -1,
        device = 0;

    cudaGetDevice(&gpu);
    fprintf(stdout, "\n   GPU default: %d", gpu);
    cudaGetDeviceCount(&gpu);
    fprintf(stdout, "\nNumber of GPUs: %d", gpu);

    for (device = 0; device < gpu; ++device)
    {
        cudaGetDeviceProperties(&deviceProp, device);
        fprintf(stdout, "\n                                                                 Device: %s", deviceProp.name);
        fprintf(stdout, "\n                                              Number of multiprocessors: %d", deviceProp.multiProcessorCount);
        fprintf(stdout, "\n                                                 Clock rate (kilohertz): %d", deviceProp.clockRate);
        fprintf(stdout, "\n                                            Total memory global (bytes): %d", deviceProp.totalGlobalMem);
        fprintf(stdout, "\n                                        Shared memory per block (bytes): %d", deviceProp.sharedMemPerBlock);
        fprintf(stdout, "\n                                  32-bit registers available per block : %d", deviceProp.regsPerBlock);
        fprintf(stdout, "\n                            Constant memory available on device (bytes): %d", deviceProp.totalConstMem);
        fprintf(stdout, "\n                                     Alignment requirement for textures: %d", deviceProp.textureAlignment);
        fprintf(stdout, "\n                                                     Compute capability: (%d,%d)", deviceProp.major, deviceProp.minor);
        fprintf(stdout, "\n                                    Maximum number of threads per block: %d", deviceProp.maxThreadsPerBlock);
        fprintf(stdout, "\n                              Maximum size of each dimension of a block: (x = %d, y = %d, z = %d)", deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1], deviceProp.maxThreadsDim[2]);
        fprintf(stdout, "\n                               Maximum size of each dimension of a grid: (x = %d, y = %d, z = %d)", deviceProp.maxGridSize[0], deviceProp.maxGridSize[1], deviceProp.maxGridSize[2]);
        fprintf(stdout, "\n                        Maximum pitch in bytes allowed by memory copies: %d", deviceProp.memPitch);
        fprintf(stdout, "\n                                                   Warp size in threads: %d", deviceProp.warpSize);
        fprintf(stdout, "\n                                   There is a run time limit on kernels: %d", deviceProp.kernelExecTimeoutEnabled);
        fprintf(stdout, "\n               Device can concurrently copy memory and execute a kernel: %d", deviceProp.deviceOverlap);
        fprintf(stdout, "\n Device can map host memory with cudaHostAlloc/cudaHostGetDevicePointer: %d", deviceProp.canMapHostMemory);
        fprintf(stdout, "\n\n");


       

//int 	computeMode//
// 	Compute mode (See cudaComputeMode).
//int 	integrated
 //	Device is integrated as opposed to discrete.


    }
}
/*
 * 
 */
int main(int argc, char** argv) {

    GPUinfo();
    return (EXIT_SUCCESS);
}


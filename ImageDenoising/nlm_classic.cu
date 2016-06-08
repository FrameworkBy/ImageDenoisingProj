#include <cuda_runtime.h>
#include <stdio.h>
#include <iostream>
#include "helper_cuda.h"
#include <omp.h>

#define BLOCKDIM_X 32
#define BLOCKDIM_Y 32

#define WARP_SIZE 128

__device__ float Max(float x, float y)
{
    return (x > y) ? x : y;
}

__device__ float Min(float x, float y)
{
    return (x < y) ? x : y;
}

__device__ int Max(int x, int y)
{
    return (x > y) ? x : y;
}

__device__ int Min(int x, int y)
{
    return (x < y) ? x : y;
}

int iDivUp(int a, int b)
{
    return ((a % b) != 0) ? (a / b + 1) : (a / b);
}

__global__ void nlm_classic_global(float* d_src,
                                   float* d_dst,
                                   int patch, int window,
                                   int width, int height, float fSigma2, float fH2, float icwl,
                                   int elemsPerThread = 8) {

    const int ix = blockDim.x * blockIdx.x + threadIdx.x;
    const int iy = blockDim.y * blockIdx.y + threadIdx.y;

    const int loop_start =  (ix/WARP_SIZE * WARP_SIZE)*(elemsPerThread-1)+ix;
    for (int i=loop_start, j=0; j<elemsPerThread && i<width; i+=WARP_SIZE, ++j) {

        int w = width - (patch*2+1)+1;
        int h = height - (patch*2+1)+1;
        //        if (ix < width-patch*2 && iy < height-patch*2)

        int i1 = i+patch;
        int j1 = iy+patch;

        float wmax = 0;
        float average = 0;
        float sweight = 0;

        int rmin = Max(i1-window,patch);
        int rmax = Min(i1+window,w+patch);
        int smin = Max(j1-window,patch);
        int smax = Min(j1+window,h+patch);

        for (int r = rmin; r < rmax; r++) {
            for (int s = smin; s < smax; s++) {
                if (r == i1 && s == j1) {
                    continue;
                }

                float diff = 0;
                for (int ii = -patch; ii <= patch; ii++) {
                    for (int jj = -patch; jj <= patch; jj++) {
                        float a = d_src[width*(j1+jj)+(i1+ii)];
                        float b = d_src[width*(s+jj)+(r+ii)];
                        float c = a-b;
                        diff += c*c;
                    }
                }
                diff = Max(float(diff - 2.0 * (double) icwl *  fSigma2), 0.0f);
                diff = diff / fH2;

                float W = expf(-diff);

                if (W > wmax) {
                    wmax = W;
                }

                sweight += W;
                average += W * d_src[width*s + r];
            }
        }
        average += wmax * d_src[width*j1+i1];
        sweight += wmax;

        if (sweight > 0) {
            d_dst[width*j1+i1] = average / sweight;
        }
        else {
            d_dst[width*j1+i1] = d_src[width*j1+i1];
        }

        //        const int offset = iy*width+i;
        //        d_dst[offset] = d_src[offset];

        //    if (ix < width && iy < height) {
        //        d_dst[width*iy+ix] = d_src[width*iy+ix];
        //    }
    }

    //        if (ix < width && iy < height) {
    //            d_dst[width*iy+ix] = d_src[width*iy+ix];
    //        }

    //    __syncthreads();
}

__global__ void nlm_classic_global2(float* d_src,
                                    float* d_dst,
                                    int patch, int window,
                                    int width, int height, float fSigma2, float fH2, float icwl,
                                    int elemsPerThread = 1) {

    const int ix = blockDim.x * blockIdx.x + threadIdx.x;
    const int iy = blockDim.y * blockIdx.y + threadIdx.y;

    int w = width - (patch*2+1)+1;
    int h = height - (patch*2+1)+1;
    if (ix < width-patch*2 && iy < height-patch*2)
    {
        int i1 = ix+patch;
        int j1 = iy+patch;

        float wmax = 0;
        float average = 0;
        float sweight = 0;

        int rmin = Max(i1-window,patch);
        int rmax = Min(i1+window,w+patch);
        int smin = Max(j1-window,patch);
        int smax = Min(j1+window,h+patch);

        for (int r = rmin; r < rmax; r++) {
            for (int s = smin; s < smax; s++) {
                if (r == i1 && s == j1) {
                    continue;
                }

                float diff = 0;
                for (int ii = -patch; ii <= patch; ii++) {
                    for (int jj = -patch; jj <= patch; jj++) {
                        float a = d_src[width*(j1+jj)+(i1+ii)];
                        float b = d_src[width*(s+jj)+(r+ii)];
                        float c = a-b;
                        diff += c*c;
                    }
                }
                diff = Max(float(diff - 2.0 * (double) icwl *  fSigma2), 0.0f);
                diff = diff / fH2;

                float W = expf(-diff);

                if (W > wmax) {
                    wmax = W;
                }

                sweight += W;
                average += W * d_src[width*s + r];
            }
        }
        average += wmax * d_src[width*j1+i1];
        sweight += wmax;

        if (sweight > 0) {
            d_dst[width*j1+i1] = average / sweight;
        }
        else {
            d_dst[width*j1+i1] = d_src[width*j1+i1];
        }
    }
}


void nlm_filter_classic_CUDA(float* h_src, float* h_dst, int width, int height, float fSigma, float fParam, int patch, int window) {
    cudaError_t err = cudaSuccess;

    float* d_src = NULL, *d_dst = NULL;
    unsigned int nBytes = sizeof(float) * (width*height);

    err = cudaMalloc((void **)& d_src, nBytes*2);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector SRC (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMalloc((void **)& d_src, nBytes);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector SRC (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMalloc((void **)& d_dst, nBytes);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector DST (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMemcpy(d_src, h_src, nBytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector SRC from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMemcpy(d_dst, h_dst, nBytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector DST from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    dim3 threads(BLOCKDIM_X, BLOCKDIM_Y);
    dim3 grid(iDivUp(width, BLOCKDIM_X), iDivUp(height, BLOCKDIM_Y));

    int patchSize = patch*2+1;
    float fSigma2 = fSigma * fSigma;
    float fH = fParam * fSigma;
    float fH2 = fH * fH;
    float icwl = patchSize * patchSize;
    fH2 *= icwl;

    dim3 blockSize;
    dim3 gridSize;
    int threadNum;
    threadNum = 1024;
    int elemsPerThread = 64;
    blockSize = dim3(threadNum, 1, 1);
    gridSize = dim3(height / (threadNum*elemsPerThread) + 1, width, 1);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

//    printf("X: %d, Y: %d, Z: %d\n", threads.x,threads.y,threads.z);
//    printf("X: %d, Y: %d, Z: %d\n", grid.x,grid.y,grid.z);

//        nlm_classic_global2<<<grid, threads>>>(d_src, d_dst, patch, window, width, height, fSigma2, fH2, icwl);
    nlm_classic_global<<<gridSize, blockSize>>>(d_src, d_dst, patch, window, width, height, fSigma2, fH2, icwl,elemsPerThread);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaDeviceSynchronize();

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "CUDA time simple (ms): " << milliseconds << std::endl;

    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch nlm_classic_device kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // load the answer back into the host
    checkCudaErrors(cudaMemcpy(h_dst, d_dst, nBytes, cudaMemcpyDeviceToHost));

    cudaFree(d_src);
    cudaFree(d_dst);

    cudaDeviceReset();
}

__global__ void kernelCopy(float* d_src, float* d_dst, int width, int height, int num_of_threads = 1) {
    //    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    //    d_dst[idx] = d_src[idx];
    const int ix = blockDim.x * blockIdx.x + threadIdx.x;
    const int iy = blockDim.y * blockIdx.y + threadIdx.y;

    int width1 = width/ num_of_threads;

    int n = (width*height)/num_of_threads;

    if (ix < width && iy < height) {
        d_dst[width1*iy+ix] = d_src[width*iy+ix];
    }
}

void nlm_filter_classic_CUDA2(float* h_src, float* h_dst, int width, int height, float fSigma, float fParam, int patch, int window) {
    int num_gpus = 0;   // number of CUDA GPUs

    /////////////////////////////////////////////////////////////////
    // determine the number of CUDA capable GPUs
    //
    cudaGetDeviceCount(&num_gpus);

    if (num_gpus < 1)
    {
        printf("no CUDA capable devices were detected\n");
        return;
    }

    /////////////////////////////////////////////////////////////////
    // display CPU and GPU configuration
    //
    printf("number of host CPUs:\t%d\n", omp_get_num_procs());
    printf("number of CUDA devices:\t%d\n", num_gpus);

    for (int i = 0; i < num_gpus; i++)
    {
        cudaDeviceProp dprop;
        cudaGetDeviceProperties(&dprop, i);
        printf("   %d: %s\n", i, dprop.name);
    }

    printf("---------------------------\n");

    unsigned int n = width*height;
    unsigned int nBytes = sizeof(float) * (n);

    int patchSize = patch*2+1;
    float fSigma2 = fSigma * fSigma;
    float fH = fParam * fSigma;
    float fH2 = fH * fH;
    float icwl = patchSize * patchSize;
    fH2 *= icwl;

    printf("Elements: %d\n",n);

    omp_set_num_threads(1);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

#pragma omp parallel
    {
        unsigned int cpu_thread_id = omp_get_thread_num();
        unsigned int num_cpu_threads = omp_get_num_threads();

        // set and check the CUDA device for this CPU thread
        int gpu_id = -1;
        checkCudaErrors(cudaSetDevice(0));   // "% num_gpus" allows more CPU threads than GPU devices
        checkCudaErrors(cudaGetDevice(&gpu_id));
//        printf("CPU thread %d (of %d) uses CUDA device %d\n", cpu_thread_id, num_cpu_threads, gpu_id);
//        printf("Elements per block: %d\n",(n) / num_cpu_threads);

        float *d_src_gpu = 0;   // pointer to memory on the device associated with this CPU thread
        float *d_dst_gpu = 0;
        float *sub_src = h_src;   // pointer to this CPU thread's portion of data
        float *sub_dst = h_dst + cpu_thread_id * (n) / num_cpu_threads;
        unsigned int nbytes_per_kernel = nBytes / num_cpu_threads;
        dim3 gpu_threads(1024);  // 128 threads per block
        dim3 gpu_blocks(n / (gpu_threads.x * num_cpu_threads));
        //printf("X: %d, Y: %d, Z: %d\n", gpu_threads.x,gpu_threads.y,gpu_threads.z);
        //printf("X: %d, Y: %d, Z: %d\n", gpu_blocks.x,gpu_blocks.y,gpu_blocks.z);

        checkCudaErrors(cudaMalloc((void **)&d_src_gpu, nBytes));
        checkCudaErrors(cudaMemset(d_src_gpu, 0, nBytes));
        checkCudaErrors(cudaMemcpy(d_src_gpu, sub_src, nBytes, cudaMemcpyHostToDevice));

        checkCudaErrors(cudaMalloc((void **)&d_dst_gpu, nbytes_per_kernel));
        checkCudaErrors(cudaMemset(d_dst_gpu, 0, nbytes_per_kernel));
        //checkCudaErrors(cudaMemcpy(d_dst_cpu, sub_dst, nbytes_per_kernel, cudaMemcpyHostToDevice));

        dim3 threads(BLOCKDIM_X/num_cpu_threads, BLOCKDIM_Y/num_cpu_threads);
        dim3 grid(iDivUp(width, BLOCKDIM_X/num_cpu_threads), iDivUp(height, BLOCKDIM_Y/num_cpu_threads));

//        printf("X: %d, Y: %d, Z: %d\n", threads.x,threads.y,threads.z);
//        printf("X: %d, Y: %d, Z: %d\n", grid.x,grid.y,grid.z);

//        kernelCopy<<<grid,threads>>>(d_src_gpu, d_dst_gpu, width, height);
                nlm_classic_global2<<<grid, threads>>>(d_src_gpu, d_dst_gpu, patch, window, width, height, fSigma2, fH2, icwl,num_cpu_threads);

        checkCudaErrors(cudaMemcpy(sub_dst, d_dst_gpu, nbytes_per_kernel, cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaFree(d_src_gpu));
        checkCudaErrors(cudaFree(d_dst_gpu));

    }
    printf("---------------------------\n");

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaDeviceSynchronize();

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "CUDA time simple (ms): " << milliseconds << std::endl;

    if (cudaSuccess != cudaGetLastError())
        printf("%s\n", cudaGetErrorString(cudaGetLastError()));
}

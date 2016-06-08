#include <cuda_runtime.h>
#include <cuda.h>
#include <stdio.h>
#include <iostream>
#include "helper_cuda.h"
#include <omp.h>

#define BLOCKDIM_X 32
#define BLOCKDIM_Y 32

__device__ float Max3(float x, float y)
{
    return (x > y) ? x : y;
}

__device__ float Min3(float x, float y)
{
    return (x < y) ? x : y;
}

int iDivUp3(int a, int b)
{
    return ((a % b) != 0) ? (a / b + 1) : (a / b);
}


// Allocate CUDA array in device memory
cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0,cudaChannelFormatKindFloat);
cudaArray* cuArray;

texture<float, cudaTextureType2D, cudaReadModeElementType> texImage;



__global__ void nlm_classic_global3(float* d_dst,
                                    int patch, int window,
                                    int width, int height, float fSigma2, float fH2, float icwl) {

    const int ix = blockDim.x * blockIdx.x + threadIdx.x;
    const int iy = blockDim.y * blockIdx.y + threadIdx.y;

    const float x = (float)ix + 0.5f;
    const float y = (float)iy + 0.5f;

    int w = width - (patch*2+1)+1;
    int h = height - (patch*2+1)+1;
    if (x < width-patch*2 && y < height-patch*2)
    {
        float i1 = x+(float)patch;
        float j1 = y+(float)patch;

        float wmax = 0;
        float average = 0;
        float sweight = 0;

        float rmin = Max3(i1-window,patch);
        float rmax = Min3(i1+window,w+patch);
        float smin = Max3(j1-window,patch);
        float smax = Min3(j1+window,h+patch);

        for (float r = rmin; r < rmax; r++) {
            for (float s = smin; s < smax; s++) {
                if (r == i1 && s == j1) {
                    continue;
                }

                float W = 0;
                float diff = 0;
                for (float ii = -patch; ii <= patch; ii++) {
                    for (float jj = -patch; jj <= patch; jj++) {
                        float c = tex2D(texImage, i1+ii, j1+jj) - tex2D(texImage, r+ii, s+jj);
                        diff += c*c;
                    }
                }
                diff = Max3(float(diff - 2.0 * (float) icwl *  fSigma2), 0.0f);
                diff = diff / fH2;

                W = expf(-diff);

                if (W > wmax) {
                    wmax = W;
                }

                sweight += W;
                average += W * tex2D(texImage, r, s);
            }
        }
        average += wmax * tex2D(texImage,i1,j1);
        sweight += wmax;

        int offset = width*(iy+patch)+ix+patch;
        if (sweight > 0) {
            d_dst[offset] = average / sweight;
        }
        else {
            d_dst[offset] = tex2D(texImage, i1, j1);
        }
    }
}

void nlm_filter_classic_CUDA3(float* h_src, float* h_dst, int width, int height, float fSigma, float fParam, int patch, int window) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    cudaError_t err = cudaSuccess;

    float *d_dst = NULL;
    float* d_dist = NULL;
    unsigned int nBytes = sizeof(float) * (width*height);


    cudaMallocArray(&cuArray, &channelDesc, width, height);
    // Copy to device memory some data located at address h_data in host memory
    cudaMemcpyToArray(cuArray, 0, 0, h_src, width * height * sizeof(float) , cudaMemcpyHostToDevice);
    // Set texture parameters
    texImage.addressMode[0] = cudaAddressModeWrap;
    texImage.addressMode[1] = cudaAddressModeWrap;
    texImage.filterMode     = cudaFilterModeLinear;
    texImage.normalized     = false;

    cudaBindTextureToArray(texImage, cuArray, channelDesc);

    checkCudaErrors(cudaMalloc((void **)&d_dist, nBytes*2));

    checkCudaErrors(cudaMalloc((void **)& d_dst, nBytes));
    checkCudaErrors(cudaMemset(d_dst, 0, nBytes));
    //    checkCudaErrors(cudaMemcpy(d_dst, h_dst, nBytes, cudaMemcpyHostToDevice));

    dim3 threads(BLOCKDIM_X, BLOCKDIM_Y);
    dim3 grid(iDivUp3(width, BLOCKDIM_X), iDivUp3(height, BLOCKDIM_Y));

    int patchSize = patch*2+1;
    float fSigma2 = fSigma * fSigma;
    float fH = fParam * fSigma;
    float fH2 = fH * fH;
    float icwl = patchSize * patchSize;
    fH2 *= icwl;



    //    printf("X: %d, Y: %d, Z: %d\n", threads.x,threads.y,threads.z);
    //    printf("X: %d, Y: %d, Z: %d\n", grid.x,grid.y,grid.z);

    nlm_classic_global3<<<grid, threads>>>(d_dst, patch, window, width, height, fSigma2, fH2, icwl);



    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch nlm_classic_device kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // load the answer back into the host
    checkCudaErrors(cudaMemcpy(h_dst, d_dst, nBytes, cudaMemcpyDeviceToHost));

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaDeviceSynchronize();

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
        std::cout << "CUDA time simple (ms): " << milliseconds << std::endl;

    cudaUnbindTexture(texImage);
    cudaFree(cuArray);
    cudaFree(d_dst);
}

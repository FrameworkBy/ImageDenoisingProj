#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <stdio.h>

#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/sort.h>

using namespace thrust;

#define QUEUE_SIZE 128

#define BLOCKDIM_X 32
#define BLOCKDIM_Y 32

int iDivUp2(int a, int b)
{
    return ((a % b) != 0) ? (a / b + 1) : (a / b);
}

__device__ float Max2(float x, float y)
{
    return (x > y) ? x : y;
}

__device__ int XX[300*300][128];
__device__ int YY[300*300][128];
__device__ float DD[300*300][128];

__device__ void mSort(int w, int x, int y, int N) {
    int xy = w*y+x;
    for (int i = N - 1; i >= 0; i--)
    {
        for (int j = 0; j < i; j++)
        {
            if (DD[xy][j] < DD[xy][j + 1])
            {
                int tmpX = XX[xy][j];
                int tmpY = YY[xy][j];
                float tmpD = DD[xy][j];

                XX[xy][j] = XX[xy][j + 1];
                YY[xy][j] = YY[xy][j + 1];
                DD[xy][j] = DD[xy][j + 1];


                XX[xy][j + 1] = tmpX;
                YY[xy][j + 1] = tmpY;
                DD[xy][j + 1] = tmpD;
            }
        }
    }
}

__device__ void mSort(int X[], int Y[], float DIST[], int N) {
    for (int i = N - 1; i >= 0; i--)
    {
        for (int j = 0; j < i; j++)
        {
            if (DIST[j] < DIST[j + 1])
            {
                int tmpX = X[j];
                int tmpY = Y[j];
                float tmpD = DIST[j];

                X[j] = X[j + 1];
                Y[j] = Y[j + 1];
                DIST[j] = DIST[j + 1];


                X[j + 1] = tmpX;
                Y[j + 1] = tmpY;
                DIST[j + 1] = tmpD;
            }
        }
    }
}

__device__ int2 randPoint(int cx, int cy, int sigma, int i) {
    curandState randState;
    unsigned int seed = (unsigned int) clock64();
    curand_init(seed, 0, 0, &randState);
    float2 fpoint;
    fpoint = curand_normal2(&randState);
    double k = 1;
    if (i != 0)
        k  = pow(0.5, i);
    int2 ipoint;
    ipoint.x = cx + int(sigma * k * fpoint.x);
    ipoint.y = cy + int(sigma * k * fpoint.y);
    return ipoint;
}

__device__ int2 get_random_pixel(int cx, int cy, int maxx, int maxy, int minx, int miny, int sigma, int i = 0) {
    int2 r = randPoint(cx, cy, sigma, 0);
    while (r.x < minx || r.x > maxx || r.y < miny || r.y > maxy)
        r = randPoint(cx, cy, sigma, 0);
    return r;
}

__device__ float get_distance(const float* img, int width, int x1, int y1, int x2, int y2, int r) {
    float dist = 0;
    for (int ii = -r; ii <= r; ii++) {
        for (int jj = -r; jj <= r; jj++) {
            int diff = img[width*(y1-jj)+(x1-ii)] - img[width*(y2-jj)+(x2-ii)];
            dist += diff*diff;
        }
    }
    return dist;
}

__global__ void nlm_filter_random_global(const float *d_src, float* d_dst,
                                         int width, int height,
                                         float fSigma, float fParam, int patchRadius,
                                         int searchRadius, int queueSize, int steps) {
    int ix = blockDim.x * blockIdx.x + threadIdx.x;
    int iy = blockDim.y * blockIdx.y + threadIdx.y;
    if (ix < width && iy < height)
    {

        int iSigmaS = width / searchRadius;

        steps = 1;

        float fSigma2 = fSigma * fSigma;
        float fH = fParam * fSigma;
        float fH2 = fH * fH;
        int patchSize = patchRadius+patchRadius+1;
        float icwl = patchSize * patchSize;
        fH2 *= icwl;
        int i1 = ix+patchRadius;
        int j1 = iy+patchRadius;
        int xy = j1*width+i1;
        for (int kk = 0; kk < steps; kk++) {
            int X[QUEUE_SIZE];
            int Y[QUEUE_SIZE];
            float DIST[QUEUE_SIZE];

            // Initialization

            for (int i = 0; i < QUEUE_SIZE; i++) {
                int2 rr = get_random_pixel(i1, j1,
                                           width+patchRadius-2, height+patchRadius-2, patchRadius+1, patchRadius+1,
                                           iSigmaS);
                float dist = get_distance(d_src,width,i1,j1,rr.x,rr.y,patchRadius);
                X[i] = rr.x;
                Y[i] = rr.y;
                DIST[i] = dist;

//                XX[xy][i] = rr.x;
//                YY[xy][i] = rr.y;
//                DD[xy][i] = dist;
            }
//            mSort(X, Y, DIST, QUEUE_SIZE);

            // Random search
            int maxJ = Max2(QUEUE_SIZE, static_cast<int>(log2(fSigma)));

            for (int N = 0; N < 1; N++) {
                for (int jj = 0; jj < maxJ*15; jj++) {
                /*    int2 rr = get_random_pixel(i1, j1,
                                               width+patchRadius-1, height+patchRadius-1, patchRadius, patchRadius,
                                               iSigmaS, jj);
//                    int2 rr;rr.x=2;rr.y=3;
                    if (rr.x == i1 && rr.y == j1) {
                        continue;
                    }
                    /*float dist = get_distance(d_src,width,i1,j1,rr.x,rr.y,patchRadius);

                    /*if(DIST[QUEUE_SIZE-1] > dist) {
                        X[QUEUE_SIZE-1] = rr.x;
                        Y[QUEUE_SIZE-1] = rr.y;
                        DIST[QUEUE_SIZE-1] = dist;
//                        mSort();
                    }*/
                }
            }

            float wmax = 0;
            float average = 0;
            float sweight = 0;

            for (size_t i = 0; i < QUEUE_SIZE; i++) {
                float fDif = DIST[i];

                fDif = Max2(fDif - 2.0 * (float) icwl *  fSigma2, 0.0);
                fDif = fDif / fH2;
                float W = expf(-fDif);

                if (W > wmax) {
                    wmax = W;
                }

                sweight += W;
                average += W * d_src[X[i]+width*Y[i]];
            }
            average += wmax * d_src[width*j1 + i1];
            sweight += wmax;

            if (sweight > 0) {
                d_dst[width*j1+i1] += (average / sweight) / steps;
            }
            else {
                d_dst[width*j1+i1] += d_src[width*j1+i1] / steps;
            }
        }
//        d_dst[width*j1 + i1] = d_src[width*j1 + i1];
    }
}

void nlm_filter_random_CUDA(const float* h_src, float* h_dst,
                            int width, int height,
                            float fSigma, float fParam,
                            int patchRadius, int searchRadius,
                            int queueSize, int steps) {
    cudaError_t err = cudaSuccess;

    float* d_src = NULL, *d_dst = NULL;
    unsigned int nBytes = sizeof(float) * (width*height);

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
    dim3 grid(iDivUp2(width, BLOCKDIM_X), iDivUp2(height, BLOCKDIM_Y));

    for (int i = 0; i < steps; ++i) {

        nlm_filter_random_global<<<grid, threads>>>(d_src, d_dst, width, height,
                                                    fSigma, fParam, patchRadius, searchRadius, queueSize, steps);
    }

    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch nlm_random_device kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // load the answer back into the host
    err = cudaMemcpy(h_dst, d_dst, nBytes, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector DST from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    cudaFree(d_src);
    cudaFree(d_dst);
}

#include "nlm_filter_classic.h"
#include "utils.h"
#include <cmath>
#include <chrono>

#ifndef QT_DEBUG
#include <cuda_runtime.h>
#include "helper_cuda.h"
#endif

using namespace std;
using namespace std::chrono;


void nlm_filter_classic_private
(const float* h_input, float* h_output, int incWidth, int incHeight, float fSigma, float fParam,
 int halfPatchSize, int halfWindowSize);
#ifndef QT_DEBUG
// Forward declare the function in the .cu file
void nlm_filter_classic_CUDA
(float* h_src, float* h_dst, int width, int height, float fSigma, float fParam, int patch, int window);
#endif


void nlm_filter_classic(QImage* imageNoise,
                        QImage* imageFiltered,
                        float fSigma, bool isCuda) {
    int halfPatchSize = 1;
    int halfWindowSize = 10;
    float fParam = 0.4f;
    if (fSigma > 0.0f && fSigma <= 15.0f) {
        halfPatchSize = 1;
        halfWindowSize = 10;
        fParam = 0.4f;
    } else if ( fSigma > 15.0f && fSigma <= 30.0f) {
        halfPatchSize = 2;
        halfWindowSize = 10;
        fParam = 0.4f;
    } else if ( fSigma > 30.0f && fSigma <= 45.0f) {
        halfPatchSize = 3;
        halfWindowSize = 17;
        fParam = 0.35f;
    } else if ( fSigma > 45.0f && fSigma <= 75.0f) {
        halfPatchSize = 4;
        halfWindowSize = 17;
        fParam = 0.35f;
    } else if (fSigma <= 100.0f) {
        halfPatchSize = 5;
        halfWindowSize = 17;
        fParam = 0.30f;
    }

//    halfPatchSize = 3;
//    halfWindowSize = 17;
//    fParam = 0.30f;


    /* SIZES */
    QSize imageSize = imageNoise->size();
    int patchSize = halfPatchSize * 2 + 1;
    int width = imageSize.width();
    int height = imageSize.height();
    int incWidth = patchSize - 1 + width;
    int incHeight = patchSize - 1 + height;

    /* CREATE ARRAYS */
    float** colorInput = new float*[width];
    float** colorOutput = new float*[width];
    for (int i = 0; i < width; i++) {
        colorInput[i] = new float[height];
        colorOutput[i] = new float[height];
    }
    for (int i = 0; i < width; i++) {
        for (int j = 0; j < height; j++) {
            colorInput[i][j] = qGray(imageNoise->pixel(i,j));
            colorOutput[i][j] = 0;
        }
    }

    /* INCREASE IMAGE */
    float** increasedImage = new float*[incWidth];
    for (int i = 0; i < incWidth; i++) {
        increasedImage[i] = new float[incHeight];
    }

    nlm_increse_image2(colorInput, increasedImage, imageSize, halfPatchSize);

    /* Creating arrays for processing on device */
    size_t nBytes = sizeof(float)*(incHeight*incWidth);
    float *h_input, *h_output;
    checkCudaErrors(cudaMallocHost((void**)&h_input, nBytes));
    checkCudaErrors(cudaMallocHost((void**)&h_output, nBytes));

    for (int i = 0; i < incWidth; i++) {
        for (int j = 0; j < incHeight; j++) {
            h_output[incWidth*j+i] = 0;
            h_input[incWidth*j+i] = increasedImage[i][j];
        }
    }

    high_resolution_clock::time_point t1 = high_resolution_clock::now();

    if (!isCuda) {
        nlm_filter_classic_private
                (h_input, h_output, incWidth, incHeight, fSigma, fParam, halfPatchSize, halfWindowSize);
    }
    else {
#ifndef QT_DEBUG
        nlm_filter_classic_CUDA
                (h_input, h_output, incWidth, incHeight, fSigma, fParam, halfPatchSize, halfWindowSize);
#endif
    }
    high_resolution_clock::time_point t2 = high_resolution_clock::now();
    auto duration = duration_cast<std::chrono::milliseconds>( t2 - t1 ).count();
    std::cout << duration << std::endl;


    /* CREATE FILTERED IMAGE */
    for (int i = 0; i < incWidth; i++) {
        for (int j = 0; j < incHeight; j++) {
            increasedImage[i][j] = h_output[incWidth*j+i];
        }
    }
    for (int i = 0; i < width; i++) {
        for (int j = 0; j < height; j++) {
            int gray = increasedImage[i+halfPatchSize][j+halfPatchSize];
            imageFiltered->setPixel(i, j, qRgb(gray, gray, gray));
        }
    }

    /* CLEAR MEMORY */
    for (int i = 0; i < width; i++) {
        delete []colorInput[i];
        delete []colorOutput[i];
    }
    for (int i = 0; i < incWidth; ++i) {
        delete []increasedImage[i];
    }
    delete []colorOutput;
    delete []colorInput;
    delete []increasedImage;
    cudaFreeHost(h_input);
    cudaFreeHost(h_output);
    cudaDeviceReset();
}

void nlm_filter_classic_private(const float* h_input, float* h_output, int incWidth, int incHeight, float fSigma, float fParam,
                                int halfPatchSize, int halfWindowSize) {
    int patchSize = halfPatchSize*2+1;
    float fSigma2 = fSigma * fSigma;
    float fH = fParam * fSigma;
    float fH2 = fH * fH;
    float icwl = patchSize * patchSize;
    fH2 *= icwl;

    int w1 = incWidth - (halfPatchSize*2+1) + 1;
    int h1 = incHeight - (halfPatchSize*2+1) + 1;
    int w = incWidth;
#ifndef QT_DEBUG
#pragma omp parallel shared(h_input, h_output)
#endif
    {
#ifndef QT_DEBUG
#pragma omp for schedule(dynamic) nowait
#endif
        for (int i = 0; i < incWidth-halfPatchSize*2; i++) {
            for (int j = 0; j < incHeight-halfPatchSize*2; j++) {

                int i1 = i + halfPatchSize;
                int j1 = j + halfPatchSize;

                float wmax = 0;
                float average = 0;
                float sweight = 0;

                int rmin = std::max(i1-halfWindowSize,halfPatchSize);
                int rmax = std::min(i1+halfWindowSize,w1+halfPatchSize);
                int smin = std::max(j1-halfWindowSize,halfPatchSize);
                int smax = std::min(j1+halfWindowSize,h1+halfPatchSize);

                for (int r = rmin; r < rmax; r++) {
                    for (int s = smin; s < smax; s++) {
                        if (r == i1 && s == j1) {
                            continue;
                        }

                        float fDif = 0;
                        float dif = 0;
                        for (int ii = -halfPatchSize; ii <= halfPatchSize; ii++) {
                            for (int jj = -halfPatchSize; jj <= halfPatchSize; jj++) {
                                int xy1 = w*(j1-jj)+(i1-ii);
                                int xy2 = w*(s-jj)+(r-ii);
                                dif = h_input[xy1]-h_input[xy2];
                                fDif += dif*dif;
                            }
                        }

                        fDif = std::max(fDif - 2.0 * (float) icwl *  fSigma2, 0.0);
                        fDif = fDif / fH2;
                        float W = exp(-fDif);

                        if (W > wmax) {
                            wmax = W;
                        }

                        sweight += W;
                        average += W * h_input[w*s+r];
                    }
                }

                average += wmax * h_input[w*j1+i1];
                sweight += wmax;

                if (sweight > 0) {
                    h_output[w*j1+i1] = average / sweight;
                }
                else {
                    h_output[w*j1+i1] = h_input[w*j1+i1];
                }
            }
        }
    }
}

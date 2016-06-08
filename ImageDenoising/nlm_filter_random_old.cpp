#include "nlm_filter_random.h"

#include <omp.h>

#include "../addnoise/awgn.h"
#include "utils.h"
#include <queue>
#include <vector>
#include <iostream>
#include <cmath>
#include <random>
#include <iomanip>
#include <string>
#include <map>
#include <random>
#include <cmath>
#include <vector>
#include <queue>

#define GENERATOR std::default_random_engine
#define DISTRIB std::normal_distribution<float>

#define QUEUE_SIZE 128
#define PATCH_RADIUS 2
#define STEPS 4
#define RAND_SEARCH_STEPS 25 //25
#define SEARCH_RADIUS 10 //10
#define FPARAM 0.4f

void image2array(QImage* input, float** output);
void array2image(float** input, QImage* output, int iWidth, int iHeight);
void nlm_filter_random_private(float** fImI, float** fImO, int iWidth, int iHeight, float fSigma, int iK);

struct PatchDist {
    int iX;
    int iY;
    float fDist;
    bool operator<(PatchDist const &other) { return fDist < other.fDist; }
};

bool compare(PatchDist d1, PatchDist d2) {
    return d1.fDist < d2.fDist;
}

class PriorityQueue {
public:
    PriorityQueue() {
    }

    size_t size() {
        return pqDist.size();
    }

    void add(int x, int y, float dist) {
        PatchDist pd;
        pd.iX = x;
        pd.iY = y;
        pd.fDist = dist;
        if (pqDist.size() == 0 || pqDist.size() <= ss) {
            pqDist.push_back(pd);
            std::sort(pqDist.begin(), pqDist.end(), compare);
        } else {
            PatchDist pqLast = pqDist[pqDist.size()-1];
            if (pqDist.size() < ss || pqLast.fDist > dist) {
                pqDist.pop_back();
                pqDist.push_back(pd);
                std::sort(pqDist.begin(), pqDist.end(), compare);
            }
        }
    }

    PatchDist get(int pos) {
        return pqDist[pos];
    }

    void clear() {
        pqDist.clear();
    }

private:
    size_t ss = QUEUE_SIZE;
//    std::priority_queue <PatchDist> pqDist;
    std::vector<PatchDist> pqDist;
};

float distance(float** img, int x1, int y1, int x2, int y2, int r) {
    float dist = 0;
    for (int ii = -r; ii <= r; ii++) {
        for (int jj = -r; jj <= r; jj++) {
            float diff = img[x1-ii][y1-jj] - img[x2-ii][y2-jj];
            dist += diff*diff;
        }
    }
    return dist;
}

float distanceVertical(float** img, int x1, int y1, int x2, int y2, float dist, int r, int shift) {
    float distNeg = 0;
    float distPos = 0;
    float diff = 0;
    for (int ii = -r; ii <= r; ii++) {
        diff = img[x1-ii][y1-shift*r]-img[x2-ii][y2-shift*r];
        distNeg += diff*diff;
        diff = img[x1-ii][y1+shift*r]-img[x2-ii][y2+shift*r];
        distPos += diff;
    }
    return dist - distNeg + distPos;
}

float distanceHorizontal(float** img, int x1, int y1, int x2, int y2, float dist, int r, int shift) {
    float distNeg = 0;
    float distPos = 0;
    float diff = 0;
    for (int ii = -r; ii <= r; ii++) {
        diff = img[x1-shift*r][y1-ii]-img[x2-shift*r][y2-ii];
        distNeg += diff*diff;
        diff = img[x1+shift*r][y1-ii]-img[x2+shift*r][y2-ii];
        distPos += diff;
    }
    return dist - distNeg + distPos;
}

void fArrClean(float** array, int iW, int iH, float value) {
    for (int i = 0; i < iW; ++i)
        for (int j = 0; j < iH; ++j)
            array[i][j] = value;
}


GENERATOR generator;
DISTRIB distribution(0,1.0);
std::random_device rd;
std::mt19937 gen(rd());
std::normal_distribution<> d(0,1.0);


void rand_px(int* x, int* y, int cx, int cy, int sigma, int i) {
    //    float rx = d(gen);
    //    float ry = d(gen);
    float rx = distribution(generator);
    float ry = distribution(generator);

    /*    while(std::abs(rx) > 1)
        rx = distribution(generator);
    //        rx = d(gen);
    while (std::abs(ry) > 1) {
        ry = distribution(generator);
        //        ry = d(gen);
    }*/

    float k = 1;
    if (i != 0)
        k  = pow(0.5, i);
    *x = cx + sigma * k * rx;
    *y = cy + sigma * k * ry;
}

void get_random_pixel(int* x, int* y, int cx, int cy, int maxx, int maxy, int minx, int miny, int sigma, int i = 0) {
    rand_px(x,y,cx,cy,sigma, i);
    while (*x < minx || *x > maxx || *y < miny || *y > maxy)
        rand_px(x,y,cx,cy,sigma,i);
}

void step_Initialization_Array(PriorityQueue** pq, int x, int y, float** fImI, int iPatch, int iK, int cx, int cy, int maxx, int maxy, int minx, int miny, int sigma) {
    if (pq[x][y].size() > 0)
        pq[x][y].clear();

    while (pq[x][y].size() < (size_t)iK) {
        int randX;
        int randY;
        get_random_pixel(&randX, &randY, cx, cy, maxx, maxy, minx, miny, sigma);
        float dist = distance(fImI,cx,cy,randX,randY,iPatch);
        //        if (dist < 0.000001) continue;
        pq[x][y].add(randX, randY, dist);

    }
}

void nlm_filter_random(QImage* input, QImage* output, float fSigma, bool cuda) {
    int inc = PATCH_RADIUS;

    int iK = QUEUE_SIZE;

    int iWidth = input->width();
    int iHeight = input->height();
    int incWidth = iWidth + inc*2;
    int incHeight = iHeight + inc*2;

    float** output_array = new float*[iWidth];
    float** input_array = new float*[iWidth];
    for (int i = 0; i < iWidth; ++i) {
        output_array[i] = new float[iHeight];
        input_array[i] = new float[iHeight];
    }

    for (int i = 0; i < iWidth; i++) {
        for (int j = 0; j < iHeight; j++) {
            output_array[i][j] = 0;
        }
    }

    image2array(input, input_array);

    float** increasedImage = new float*[incWidth];
    for (int i = 0; i < incWidth; i++) {
        increasedImage[i] = new float[incHeight];
    }
    nlm_increse_image2(input_array, increasedImage, QSize(iWidth,iHeight), inc);


    fArrClean(output_array,iWidth,iHeight,0.0f);
T_START
    nlm_filter_random_private(increasedImage, output_array, iWidth, iHeight, fSigma, iK);
T_END

    array2image(output_array, output, iWidth, iHeight);

    for (int i = 0; i < iWidth; i++) {
        delete []input_array[i];
        delete []output_array[i];
    }
    for (int i = 0; i < incWidth; ++i) {
        delete []increasedImage[i];
    }
    delete []input_array;
    delete []output_array;
    delete []increasedImage;
}

void nlm_filter_random_private(float** fImI, float** fImO, int iWidth, int iHeight, float fSigma, int iK) {
    int iSigmaS = iWidth / SEARCH_RADIUS;
    iK = QUEUE_SIZE;
    int iPatch = PATCH_RADIUS;

    int steps = STEPS;

    float fParam = FPARAM;
    float fSigma2 = fSigma * fSigma;
    float fH = fParam * fSigma;
    float fH2 = fH * fH;
    int patchSize = iPatch+iPatch+1;
    float icwl = patchSize * patchSize;
    fH2 *= icwl;

    PriorityQueue** pqArray = new PriorityQueue*[iWidth];
    for (int x = 0; x < iWidth; ++x) {
        pqArray[x] = new PriorityQueue[iHeight];
    }
    for (int kkk = 0; kkk < steps; kkk++) {
        //        std::cout << "Step: " << kkk+1 << "." << std::endl;
        //        std::cout << "Initialazation." << std::endl;

#ifndef QT_DEBUG
#pragma omp parallel shared(fImI, fImO)
#endif
        {
#ifndef QT_DEBUG
#pragma omp for schedule(dynamic) nowait
#endif
            for (int x = 0; x < iWidth; ++x) {
                for (int y = 0; y < iHeight; ++y) {
                    int xx = x + iPatch;
                    int yy = y + iPatch;
                    step_Initialization_Array(pqArray, x, y, fImI, iPatch, iK, xx, yy, iWidth+iPatch-2, iHeight+iPatch-2, iPatch+1, iPatch+1, iSigmaS);
                }
            }
        }
        //        std::cout << "Denoising." << std::endl;

#ifndef QT_DEBUG
#pragma omp parallel shared(fImI, fImO)
#endif
        {
#ifndef QT_DEBUG
#pragma omp for schedule(dynamic) nowait
#endif
            for (int x = 0; x < iWidth; ++x) {
                for (int y = 0; y < iHeight; ++y) {
                    int xx = x + iPatch;
                    int yy = y + iPatch;

                    PriorityQueue pq;

                    // Initialization
                    PatchDist tmpPatchDist;
                    for (size_t i = 0; i < pqArray[x][y].size(); i++) {
                        tmpPatchDist = pqArray[x][y].get(i);
                        pq.add(tmpPatchDist.iX, tmpPatchDist.iY, tmpPatchDist.fDist);
                    }
/*
                    // Propagation
                    // TOP
                    if ((y - 1) >= 0) {
                        for (int i = 0; i < pqArray[x][y-1].size(); i++) {
                            tmpPatchDist =  pqArray[x][y-1].get(i);
                            float dist = distance(fImI,xx,yy,tmpPatchDist.iX,tmpPatchDist.iY,iPatch);
                            //                    float dist = distanceHorizontal(fImI,xx,yy,tmpPatchDist.iX,tmpPatchDist.iY,tmpPatchDist.fDist,iPatch,1);
                            pq.add(tmpPatchDist.iX, tmpPatchDist.iY, dist);
                        }
                    }

                    // BOTTOM
                    if ((y + 1) < iHeight) {
                        for (int i = 0; i < pqArray[x][y+1].size(); i++) {
                            tmpPatchDist =  pqArray[x][y+1].get(i);
                            float dist = distance(fImI,xx,yy,tmpPatchDist.iX,tmpPatchDist.iY,iPatch);
                            //                    float dist = distanceHorizontal(fImI,xx,yy,tmpPatchDist.iX,tmpPatchDist.iY,tmpPatchDist.fDist,iPatch,-1);
                            pq.add(tmpPatchDist.iX, tmpPatchDist.iY, dist);
                        }
                    }

                    // LEFT
                    if ((x - 1) >= 0) {
                        for (int i = 0; i < pqArray[x-1][y].size(); i++) {
                            tmpPatchDist =  pqArray[x-1][y].get(i);
                            float dist = distance(fImI,xx,yy,tmpPatchDist.iX,tmpPatchDist.iY,iPatch);
                            //                    float dist = distanceVertical(fImI,xx,yy,tmpPatchDist.iX,tmpPatchDist.iY,tmpPatchDist.fDist,iPatch,-1);
                            pq.add(tmpPatchDist.iX, tmpPatchDist.iY, dist);
                        }
                    }

                    // RIGHT
                    if ((x + 1) < iWidth) {
                        for (int i = 0; i < pqArray[x+1][y].size(); i++) {
                            tmpPatchDist =  pqArray[x+1][y].get(i);
                            float dist = distance(fImI,xx,yy,tmpPatchDist.iX,tmpPatchDist.iY,iPatch);
                            //                    float dist = distanceVertical(fImI,xx,yy,tmpPatchDist.iX,tmpPatchDist.iY,tmpPatchDist.fDist,iPatch,-1);
                            pq.add(tmpPatchDist.iX, tmpPatchDist.iY, dist);
                        }
                    }
*/
                    // Random search
                    std::vector<PatchDist> pdP;
                    for (size_t i = 0; i < pq.size(); i++) {
                        pdP.push_back(pq.get(i));
                    }

                    int maxJ = std::min(iK, static_cast<int>(log2(fSigma)));

                    for (int N = 0; N < RAND_SEARCH_STEPS; N++) {
                        for (int jj = 0; jj < maxJ; jj++) {
                            int randX;
                            int randY;
                            get_random_pixel(&randX, &randY, xx, yy, iWidth+iPatch-1, iHeight+iPatch-1, iPatch, iPatch, iSigmaS, jj);
                            if (randX == xx && randY == yy) {
                                continue;
                            }
                            float dist = distance(fImI, xx, yy, randX, randY, iPatch);
                            pq.add(randX, randY, dist);
                        }
                    }


                    float wmax = 0;
                    float average = 0;
                    float sweight = 0;

                    for (size_t i = 0; i < pq.size(); i++) {
                        PatchDist pd = pq.get(i);

                        float fDif = pd.fDist;

                        fDif = std::max(fDif - 2.0 * (float) icwl *  fSigma2, 0.0);
                        fDif = fDif / fH2;
                        float W = exp(-fDif);

                        //                float W = exp(-(fDif/fSigma2));

                        if (W > wmax) {
                            wmax = W;
                        }

                        sweight += W;
                        average += W * fImI[pd.iX][pd.iY];
                    }
                    average += wmax * fImI[xx][yy];
                    sweight += wmax;

                    if (sweight > 0) {
                        fImO[x][y] += (average / sweight) / steps;
                    }
                    else {
                        fImO[x][y] += (fImI[xx][yy]) / steps;
                    }
                }
            }
        }
    }
}

void image2array(QImage* input, float** output) {
    int iWidth = input->width();
    int iHeight = input->height();

    for (int x = 0; x < iWidth; ++x) {
        for (int y = 0; y < iHeight; ++y) {
            output[x][y] = qGray(input->pixel(x,y));
        }
    }
}

void array2image(float** input, QImage* output, int iWidth, int iHeight) {
    if (output == NULL) {
        output = new QImage(iWidth, iHeight, QImage::Format_RGB32);
    }
    for (int x = 0; x < iWidth; ++x) {
        for (int y = 0; y < iHeight; ++y) {
            int gray = input[x][y];
            output->setPixel(x,y, qRgb(gray,gray,gray));
        }
    }
}

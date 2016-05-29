#include "utils.h"

void nlm_increse_image2(float ** src, float ** dst, QSize srcImageSize, int inc) {
    int w = srcImageSize.width();
    int h = srcImageSize.height();

    // COPY ORIGINAL IMAGE
    for (int i = inc; i < w+inc; i++) {
        for  (int j = inc; j < h+inc; j++) {
            dst[i][j] = src[i-inc][j-inc];
        }
    }

    float **left, **right, **top, **bottom;
    left = new float*[inc];
    right = new float*[inc];
    for (int i = 0; i < inc; i++) {
        left[i] = new float[h];
        right[i] = new float[h];
    }

    top = new float*[w+2*inc];
    bottom = new float*[w+2*inc];
    for (int i = 0; i < w+2*inc; i++) {
        top[i] = new float[inc];
        bottom[i] = new float[inc];
    }
    ////////////////////////////////
    for (int i = 0; i < inc; i++) {
        for (int j = 0; j < h; j++) {
            left[i][j] = src[inc-i-1][j];
        }
    }
    for (int i = 0; i < inc; i++) {
        for (int j = 0; j < h; j++) {
            right[inc-i-1][j] = src[w-inc+i][j];
        }
    }

    //////////////
    for (int i = 0; i < inc; i++) {
        for (int j = 0; j < h; j++) {
            dst[i][j+inc] = left[i][j];
        }
    }

    for (int i = 0; i < inc; i++) {
        for (int j = 0; j < h; j++) {
            dst[w+inc+i][j+inc] = right[i][j];
        }
    }

    ////////////////////////////////

    for (int i = 0; i < w+2*inc; i++) {
        for (int j = 0; j < inc; j++) {
            top[i][inc-j-1] = dst[i][j+inc];
        }
    }

    for (int i = 0; i < w+2*inc; i++) {
        for (int j = 0; j < inc; j++) {
            bottom[i][inc-j-1] = dst[i][j+h];
        }
    }

    //////////////

    for (int i = 0; i < w+2*inc; i++) {
        for (int j = 0; j < inc; j++) {
            dst[i][j] = top[i][j];
        }
    }

    for (int i = 0; i < w+2*inc; i++) {
        for (int j = 0; j < inc; j++) {
            dst[i][inc+h+j] = bottom[i][j];
        }
    }

    ////////////////////////////////

    for (int i = 0; i < inc; i++) {
        delete[] left[i];
        delete[] right[i];
    }

    for (int i = 0; i < w+2*inc; i++) {
        delete[] top[i];
        delete[] bottom[i];
    }
    delete[] left;
    delete[] right;
    delete[] top;
    delete[] bottom;
}

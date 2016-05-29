#include "noise.h"

QImage* addAWGN_GRAY(QImage* image, float std) {
    QSize size = image->size();
    QImage* imageNoise = new QImage(size, QImage::Format_RGB32);
    for (int i = 0; i < size.width(); i++) {
        for (int j = 0; j < size.height(); j++) {
            QRgb p = image->pixel(i, j);
            int gray = qGray(p);

            float gray_noise = gray+AWGN_generator(std);

            if (gray_noise > 255) {
                gray_noise = 255;
            } else if (gray_noise < 0) {
                gray_noise = 0;
            }

            QRgb n = qRgb(gray_noise, gray_noise, gray_noise);

            imageNoise->setPixel(i, j, n);
        }
    }
    return imageNoise;
}

#include "diffimages.h"


void diff_images(QImage *image1, QImage *image2, float fSigma, QString fileName)
{
    // TODO: Check images size

    QImage* image3 = new QImage(image1->size(), QImage::Format_RGB32);
    fSigma *= 0.4f;
    for (int i = 0; i < image1->size().width(); i++) {
        for (int j = 0; j < image1->size().height(); j++) {
            float p1 = qGray(image1->pixel(i,j));
            float p2 = qGray(image2->pixel(i,j));
            float p3 = p1-p2;
            float diff = (p3 + fSigma) * 255.0f / (2.0f * fSigma);
            int pp = diff;
            if (pp < 0) diff = 0;
            if (pp > 255) diff = 255;
            image3->setPixel(i,j,qRgb(pp,pp,pp));
        }
    }

    QImageWriter* imageDiff = new QImageWriter();
    imageDiff->setFileName(fileName);
    imageDiff->write(*image3);

    delete image3;
    delete imageDiff;
}

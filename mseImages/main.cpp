#include <QCoreApplication>
#include <QCommandLineParser>
#include <QImage>
#include <QImageReader>
#include <QImageWriter>

#include <cmath>
#include <stdio.h>

void calcMsePsnr(double* mse, double* psnr, QImage *image1, QImage *image2, QSize size) {
    double P = 0;
    for (int i = 0; i < size.width(); i++) {
        for (int j = 0; j < size.height(); j++) {
            int pixel1 = qGray(image1->pixel(i,j));
            int pixel2 = qGray(image2->pixel(i,j));
            double SQ = pixel1-pixel2;
            P += SQ*SQ;
        }
    }
    P /= size.width()*size.height();
    *mse = sqrt(P);
    *psnr = 10 * log10(65025./((*mse) * (*mse)));

}

int main(int argc, char *argv[])
{
    QCoreApplication a(argc, argv);
    QCoreApplication::setApplicationName("Mse&Psnr");
    QCoreApplication::setApplicationVersion("1.0");

    QCommandLineParser parser;
    parser.setApplicationDescription("Calculate mse and psnr.");
    parser.addHelpOption();
    parser.addVersionOption();

    parser.addPositionalArgument("image1", QCoreApplication::translate("main", "Image 1"));
    parser.addPositionalArgument("image2", QCoreApplication::translate("main", "Image 2"));

    parser.process(a);

    const QStringList args = parser.positionalArguments();

    if (args.size() < 2) {
        parser.showHelp(1);
    }

    QImageReader* imageInputReader0 = new QImageReader(args.at(0));
    QImage* imageInput0 = new QImage();
    imageInputReader0->read(imageInput0);

    QImageReader* imageInputReader1 = new QImageReader(args.at(1));
    QImage* imageInput1 = new QImage();
    imageInputReader1->read(imageInput1);

    double mse = 0;
    double psnr = 0;

    calcMsePsnr(&mse, &psnr, imageInput0, imageInput1, imageInput1->size());

    printf("MSE: %f\nPSNR: %f\n", mse, psnr);

    a.exit(0);
    return 0;
}

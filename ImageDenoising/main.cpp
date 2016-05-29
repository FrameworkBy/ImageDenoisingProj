#include <QCoreApplication>
#include <QCommandLineParser>
#include <QImageReader>
#include <QImageWriter>

#include <iostream>

#include <nlm_filter_classic.h>

using namespace std;

int main(int argc, char *argv[])
{
    QCoreApplication a(argc, argv);
    QCoreApplication::setApplicationName("ImageDenoising");
    QCoreApplication::setApplicationVersion("1.0");

    QCommandLineParser parser;
    parser.setApplicationDescription("Denoise image with NLM algoritm.");
    parser.addHelpOption();
    parser.addVersionOption();

    QCommandLineOption inputImageOption(QStringList() << "i" << "input",
                                        QCoreApplication::translate("main", "Denoise <input>."),
                                        QCoreApplication::translate("main", "input"));

    QCommandLineOption outputImageOption(QStringList() << "o" << "output",
                                         QCoreApplication::translate("main", "Denoise <output>."),
                                         QCoreApplication::translate("main", "output"));

    QCommandLineOption sigmaOption(QStringList() << "s" << "sigma",
                                   QCoreApplication::translate("main", "Option <sigma>."),
                                   QCoreApplication::translate("main", "sigma"));

    QCommandLineOption cudaOption(QStringList() << "c" << "cuda",
                                  QCoreApplication::translate("main", "Use CUDA."));


    parser.addOption(inputImageOption);
    parser.addOption(outputImageOption);
    parser.addOption(sigmaOption);
    parser.addOption(cudaOption);
    parser.process(a);

    if (
            !parser.isSet(inputImageOption) ||
            !parser.isSet(outputImageOption) ||
            !parser.isSet(sigmaOption)
            ) {
        parser.showHelp(1);
    }

    QString inputImageName = parser.value(inputImageOption);
    QString outputImageName = parser.value(outputImageOption);

    float sigma = parser.value(sigmaOption).toFloat();

    bool cuda = parser.isSet(cudaOption);

    QImageReader* imageInputReader = new QImageReader(inputImageName);
    QImage* imageInput = new QImage();
    imageInputReader->read(imageInput);

    QImage* imageFiltered = new QImage(imageInput->size(), QImage::Format_RGB32);

    nlm_filter_classic(imageInput, imageFiltered, sigma, cuda);

    QImageWriter* imageWriterFiltered = new QImageWriter();
    imageWriterFiltered->setFileName(outputImageName);
    imageWriterFiltered->write(*imageFiltered);

    a.exit(0);
    return 0;
}

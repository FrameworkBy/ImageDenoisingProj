#include <QCoreApplication>
#include <QCommandLineParser>
#include <QImageReader>
#include <QImageWriter>

#include <noise.h>

int main(int argc, char *argv[])
{
    QCoreApplication a(argc, argv);
    QCoreApplication::setApplicationName("Add Noise");
    QCoreApplication::setApplicationVersion("1.0");

    QCommandLineParser parser;
    parser.setApplicationDescription("Add noise to image.");
    parser.addHelpOption();
    parser.addVersionOption();

    QCommandLineOption inputImageOption(QStringList() << "i" << "input",
                                        QCoreApplication::translate("main", "Denoise <input>."),
                                        QCoreApplication::translate("main", "input"));

    QCommandLineOption outputImageOption(QStringList() << "o" << "output",
                                         QCoreApplication::translate("main", "Denoise <output>."),
                                         QCoreApplication::translate("main", "output"));
    QCommandLineOption noiseOption(QStringList() << "n" << "noise",
                                   QCoreApplication::translate("main", "Level of <noise>."),
                                   QCoreApplication::translate("main", "noise"));

    parser.addOption(inputImageOption);
    parser.addOption(outputImageOption);
    parser.addOption(noiseOption);
    parser.process(a);

    if (
            !parser.isSet(inputImageOption) ||
            !parser.isSet(outputImageOption) ||
            !parser.isSet(noiseOption)
            ) {
        parser.showHelp(1);
    }

    QString inputImageName = parser.value(inputImageOption);
    QString outputImageName = parser.value(outputImageOption);
    QString noise = parser.value(noiseOption);

    QImageReader* imageInputReader = new QImageReader(inputImageName);
    QImage* imageInput = new QImage();
    imageInputReader->read(imageInput);

    QImage* imageNoise = addAWGN_GRAY(imageInput, noise.toFloat());
    QImageWriter* imageWriterNoise = new QImageWriter();
    imageWriterNoise->setFileName(outputImageName);
    imageWriterNoise->write(*imageNoise);

    a.exit(0);
    return 0;
}

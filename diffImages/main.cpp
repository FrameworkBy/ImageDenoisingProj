#include <QCoreApplication>
#include <QCommandLineParser>

#include <diffimages.h>

int main(int argc, char *argv[])
{
    QCoreApplication a(argc, argv);
    QCoreApplication::setApplicationName("Diff images");
    QCoreApplication::setApplicationVersion("1.0");

    QCommandLineParser parser;
    parser.setApplicationDescription("Denoise image with NLM algoritm.");
    parser.addHelpOption();
    parser.addVersionOption();

    parser.addPositionalArgument("image1", QCoreApplication::translate("main", "Image 1"));
    parser.addPositionalArgument("image2", QCoreApplication::translate("main", "Image 2"));
    parser.addPositionalArgument("output", QCoreApplication::translate("main", "Output image"));
    parser.addPositionalArgument("sigma", QCoreApplication::translate("main", "sigma"));

    parser.process(a);

    const QStringList args = parser.positionalArguments();

    if (args.size() < 4) {
        parser.showHelp(1);
    }

    QImageReader* imageInputReader0 = new QImageReader(args.at(0));
    QImage* imageInput0 = new QImage();
    imageInputReader0->read(imageInput0);

    QImageReader* imageInputReader1 = new QImageReader(args.at(1));
    QImage* imageInput1 = new QImage();
    imageInputReader1->read(imageInput1);

    diff_images(imageInput0, imageInput1, args.at(3).toFloat(), args.at(2));



    a.exit(0);
    return 0;
}

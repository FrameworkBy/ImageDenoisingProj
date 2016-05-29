QT += core
QT += gui

CONFIG += c++11

TARGET = ImageDenoising
CONFIG += console
CONFIG -= app_bundle

TEMPLATE = app

SOURCES += main.cpp \
    imagedenoising.cpp \
    utils.cpp \
    nlm_filter_classic.cpp \
    nlm_filter_random.cpp \
    ../addnoise/awgn.cpp

HEADERS += \
    imagedenoising.h \
    utils.h \
    nlm_filter_classic.h \
    nlm_filter_random.h \
    ../addnoise/awgn.h

DESTDIR = ../bin

QMAKE_CXXFLAGS += /openmp

CONFIG(release, debug|release) {
include(cuda.pri)
}

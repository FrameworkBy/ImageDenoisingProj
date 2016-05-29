QT += core
QT += gui

CONFIG += c++11

TARGET = addnoise
CONFIG += console
CONFIG -= app_bundle

TEMPLATE = app

SOURCES += main.cpp \
    noise.cpp \
    awgn.cpp

HEADERS += \
    noise.h \
    awgn.h

DESTDIR = ../bin

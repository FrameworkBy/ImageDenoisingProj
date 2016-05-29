QT += core
QT += gui

CONFIG += c++11

TARGET = diffImages
CONFIG += console
CONFIG -= app_bundle

TEMPLATE = app

SOURCES += main.cpp \
    diffimages.cpp

DESTDIR = ../bin

HEADERS += \
    diffimages.h

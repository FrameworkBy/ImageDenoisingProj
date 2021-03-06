#ifndef UTILS
#define UTILS

#include <QImage>
#include <QSize>

#include <QDebug>
#include <QDateTime>

#include <iostream>

#define T_START QDateTime mStartTime = QDateTime::currentDateTime();
#define T_END QDateTime mFinishTime = QDateTime::currentDateTime(); std::cout << QDateTime::fromMSecsSinceEpoch(mFinishTime.toMSecsSinceEpoch() - mStartTime.toMSecsSinceEpoch()).time().msec();// << std::endl;

void nlm_increse_image2(float ** src, float ** dst, QSize srcImageSize, int inc);

#endif // UTILS


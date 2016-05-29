#ifndef UTILS
#define UTILS

#include <QImage>
#include <QSize>

#include <QDebug>
#include <QDateTime>

#define T_START QDateTime mStartTime = QDateTime::currentDateTime();
#define T_END QDateTime mFinishTime = QDateTime::currentDateTime();qDebug() << "NLM" << QDateTime::fromMSecsSinceEpoch(mFinishTime.toMSecsSinceEpoch() - mStartTime.toMSecsSinceEpoch()).time();

void nlm_increse_image2(float ** src, float ** dst, QSize srcImageSize, int inc);

#endif // UTILS


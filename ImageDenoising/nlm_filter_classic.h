#ifndef NLM_FILTER_CLASSIC_H
#define NLM_FILTER_CLASSIC_H

#include <QImage>
#include <QSize>

void nlm_filter_classic(QImage* imageNoise,
                        QImage *imageFiltered,
                        float fSigma, bool isCuda = false);

#endif // NLM_FILTER_CLASSIC_H

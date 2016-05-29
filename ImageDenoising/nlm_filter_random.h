#ifndef NLM_FILTER_RANDOM_H
#define NLM_FILTER_RANDOM_H

#include <QImage>

void nlm_filter_random(QImage* input, QImage* output, float fSigma, bool cuda = false);

#endif // NLM_FILTER_RANDOM_H
